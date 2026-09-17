"""The one interface every retention policy implements.

The spec's entire contribution is replacing one line of TG -- "the oldest entry is removed"
(§3.1). Everything else in the architecture is unchanged. So every baseline and RSR itself
implement the *same* protocol and are interchangeable by config, and no comparison in this project
can depend on which code path ran.

§3.7's reduction to exact TG must be reachable **by configuration alone**:

    psi-hat = -a_i,  b = 0,  nu = 0,  beta = 0,  A_max = M,  T_warm = inf,  shadow buffer off

Under those settings the eviction argmin becomes ``argmin_i(-a_i)`` = the oldest slot, and the loss
curve must be bit-identical to stock TG. ``tests/test_reduction.py`` asserts it. If a policy needs a
separate code path to reduce, the reduction is not a reduction and E0b stops testing what it claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "AttentionTrace",
    "EvictionDecision",
    "MemoryState",
    "RetentionPolicy",
]


@dataclass
class MemoryState:
    """The live working memory at one sentence step.

    ``gestalts`` is ``(n_live, d)``, ordered oldest-first, matching [P2]'s memory layout.
    ``ages`` is the number of sentence steps since each slot was written; ``ages[0]`` is the
    largest. Age is carried here because the *eviction rule* may use it (FIFO, LRU, A_max) --
    but it is deliberately **not** available to the learned value head. See §3.2.2: supplying age
    to psi-hat "invites collapse onto recency and makes the vacuity failure mode invisible rather
    than merely possible."
    """

    gestalts: Tensor
    ages: Tensor
    capacity: int
    step: int = 0
    slot_ids: Tensor | None = None
    """Stable identifiers, so a policy can track a slot across steps without using its index."""

    def __post_init__(self) -> None:
        if self.gestalts.ndim != 2:
            raise ValueError(f"gestalts must be (n_live, d), got {tuple(self.gestalts.shape)}")
        if self.ages.shape[0] != self.gestalts.shape[0]:
            raise ValueError(
                f"ages has {self.ages.shape[0]} entries for {self.gestalts.shape[0]} slots"
            )
        if self.slot_ids is None:
            self.slot_ids = torch.arange(self.n_live, device=self.gestalts.device)

    @property
    def n_live(self) -> int:
        return int(self.gestalts.shape[0])

    @property
    def d(self) -> int:
        return int(self.gestalts.shape[1])

    @property
    def is_full(self) -> bool:
        return self.n_live >= self.capacity

    @property
    def fill_fraction(self) -> float:
        """§3.2.1 requires ``mean(|memory_t|/M)`` per experiment (~0.76 at M=40, S=80)."""
        return self.n_live / self.capacity


@dataclass
class AttentionTrace:
    """What one sentence step's cross-attention did, per layer and head.

    ``weights``      (n_layers, n_heads, n_live) -- the softmax probabilities alpha.
    ``contributions`` (n_layers, n_heads, n_live) -- ||alpha * W_O v||_2, the norm-weighted
        contribution of [P5]. **W_O is not optional** (D-6): it is where head-specific rescaling
        lives, and dropping it reintroduces the confound norm-weighting was adopted to remove.
    ``memory_gates`` (n_layers,) -- TG's learnable scalar ``g_mem`` per cross-attention layer,
        which scales the cross-attention increment *before* it is added to the residual stream.

    ``memory_gates`` is carried because of correction B-1: ``contributions`` is measured
    **upstream** of the gate, and [P2] App. C measures that the gates grow over training and are
    larger in deeper layers. So the layer sum mis-weights layers by a moving, depth-stratified
    factor. E0d reports ``r_i`` both ways. Attention weight alone is not evidence of use ([P14],
    attention sinks) and is kept only for diagnostics.
    """

    weights: Tensor
    contributions: Tensor
    memory_gates: Tensor | None = None

    def __post_init__(self) -> None:
        if self.contributions.ndim != 3:
            raise ValueError(
                f"contributions must be (n_layers, n_heads, n_live), "
                f"got {tuple(self.contributions.shape)}"
            )
        if self.memory_gates is not None:
            n_layers = self.contributions.shape[0]
            if self.memory_gates.shape[0] != n_layers:
                raise ValueError(
                    f"memory_gates has {self.memory_gates.shape[0]} entries "
                    f"for {n_layers} layers"
                )

    @property
    def n_live(self) -> int:
        return int(self.contributions.shape[-1])

    def per_layer_profile(self, *, gated: bool = False) -> Tensor:
        """``(n_layers, n_live)`` -- report this once before collapsing to a scalar (§3.2.1).

        Summing across layers conflates contributions to different residual-stream positions, and
        [P2] App. C-D show memory gates and cross-attention gradient share are strongly
        depth-stratified, so the layer sum is not obviously the right aggregate.
        """
        profile = self.contributions.sum(dim=1)
        if gated:
            if self.memory_gates is None:
                raise ValueError(
                    "gated=True needs memory_gates. B-1: r_i measured upstream of TG's g_mem "
                    "mis-weights layers by a factor that moves during training."
                )
            profile = profile * self.memory_gates.unsqueeze(-1)
        return profile


@dataclass
class EvictionDecision:
    """One eviction, with enough detail for §3.4's required decision attribution.

    §3.4: "Log, per eviction, which term determined the argmin ... **If ``b`` flips a large share,
    the balance controller is the policy** -- which is the v0.2 objection that z-scoring was
    introduced to prevent, reopened by D-1's fix." Same test for ``nu``. Required from the first
    policy run, not added later.
    """

    slot_index: int
    step: int
    score: float | None = None
    value_only_choice: int | None = None
    """What argmin z(psi-hat) alone would have picked."""
    bias_flipped: bool = False
    redundancy_flipped: bool = False
    terms: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class RetentionPolicy(Protocol):
    """Every baseline and RSR implement this. Nothing else is a policy."""

    name: str

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        """Return the index into ``slots.gestalts`` of the slot to evict."""
        ...

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        """Called after every step, whether or not an eviction happened.

        This is where accumulating policies (H2O, LRU) and learning policies (RSR) update state.
        A policy that needs nothing implements it as a no-op -- it is not optional, because a
        policy that silently never observes is indistinguishable from one that does.
        """
        ...

    def reset(self) -> None:
        """Called at every stream boundary. [P2] resets memory (stop-gradient) per stream, so any
        per-stream accumulator must reset with it -- including ``b`` (§3.5)."""
        ...
