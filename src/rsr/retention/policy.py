"""The retention-policy interface (spec section 3.4, kickoff scaffolding item 1).

Every baseline and RSR implement `RetentionPolicy`, so FIFO, LRU, H2O,
Expire-Span, the leading-edge strategy, random, oracle and RSR are interchangeable
by config.

**Section 3.7's reduction to exact TG must be reachable by configuration alone,
not by a separate code path.** Set `psi_hat == -a_i`, `b == 0`, `nu = 0`,
`beta = 0`, `A_max = M`, `T_warm = inf`, shadow buffer off, and eviction becomes
`argmin(-a_i)` -- the oldest slot. Bit-exact TG.

That is why every term added to the eviction score ships a documented off-switch.
Without one, section 3.7 silently stops being a reduction and E0b stops testing
what it claims to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = ["AttentionTrace", "MemoryState", "RetentionPolicy"]


@dataclass
class MemoryState:
    """The working memory of gestalt vectors at one step.

    From [P2]: a recurrent transformer processing one sentence per step, where
    tokens reach prior sentences only through cross-attention to this memory.
    Each gestalt is `s_t = W_sent . H^(7)[EOS]`, appended **without detaching its
    computation graph** -- so `gestalts` carries live autograd history and must not
    be `.detach()`ed casually.

    Note that eviction does **not** free a slot's graph (section 3.6, [P2] App. A).
    Memory capacity bounds the forward view only; backward depth is bounded by
    stream length `S`. v0.1's claim that value-based retention causes unbounded
    activation memory was false, and this dataclass is where that misreading would
    otherwise reappear.
    """

    gestalts: Tensor
    """`[M, d]`. Row `i` is slot `i`'s gestalt. Rows where `live` is False are
    undefined, not zero -- read them and you are reading a dead slot."""

    written_at: Tensor
    """`[M]` int64. The step at which each slot was written. Age is
    `step - written_at`. **Age is deliberately not exposed to `psi_hat`**
    (section 3.2.2): supplying it invites collapse onto recency and makes the
    vacuity failure mode invisible rather than merely possible. Policies that are
    *meant* to use age -- FIFO, LRU, the age-only head -- read it from here."""

    live: Tensor
    """`[M]` bool. Slots currently occupied. Early in a stream this is partly
    False, which is what section 3.2.1's fill-level rescale exists to correct."""

    step: int
    """Index of the current sentence step within the stream."""

    accum: dict[str, Tensor] = field(default_factory=dict)
    """Scratch space for accumulators whose lifetime is a single step.

    🔴 **Not for state that must survive a step.** Gauntlet 0.4: a training loop
    builds a fresh `MemoryState` per step, so anything left here is erased, and the
    policy that depended on it degrades **silently into a different policy**. LRU
    kept `last_used` here and became FIFO.

    Cross-step per-slot state belongs on the policy object, keyed by slot index and
    invalidated through `on_write`. Anything that must survive a *stream* boundary
    belongs on the policy object too, cleared in `reset`."""

    @property
    def capacity(self) -> int:
        return int(self.gestalts.shape[0])

    @property
    def n_live(self) -> int:
        return int(self.live.sum().item())

    @property
    def is_full(self) -> bool:
        return self.n_live >= self.capacity

    def ages(self) -> Tensor:
        """`[M]` int64 age per slot. Dead slots are reported as -1."""
        return torch.where(
            self.live,
            self.step - self.written_at,
            torch.full_like(self.written_at, -1),
        )

    def ranks(self) -> Tensor:
        """`[M]` int64. Each live slot's position in the memory ordering, oldest
        first, 1-based. Dead slots are -1.

        [P2] section 2.2 adds `P^(sent)_{1:Mt}` to the memory **keys** over a
        memory "ordered from oldest to most recent", so this is the index the
        positional encoding actually uses -- **rank, not age**. Under FIFO the two
        orderings coincide and the distinction is invisible; under RSR, evicting a
        middle slot shifts every slot behind it. See ADR-0006.
        """
        order = torch.where(
            self.live, self.written_at, torch.full_like(self.written_at, 2**62)
        )
        rank = torch.empty_like(self.written_at)
        rank.fill_(-1)
        live_idx = order.argsort()[: self.n_live]
        rank[live_idx] = torch.arange(
            1, self.n_live + 1, dtype=self.written_at.dtype, device=rank.device
        )
        return rank

    def fill_fraction(self) -> float:
        """`|memory_t| / M`. Section 5.2 requires this be reported per experiment
        (~0.76 at `M = 40, S = 80`). Under the section 3.2.1 target rescale it is a
        descriptive statistic, not a discount on usable data -- which is why the
        rescale was chosen over v0.2's mask or a loss weight."""
        return self.n_live / self.capacity


@dataclass
class AttentionTrace:
    """One step of cross-attention, kept per layer and per head.

    **Not collapsed to a scalar at capture time, deliberately.** Section 3.2.1
    defines the retrieval-demand signal as

        r_i(t) = sum_{l,h} || alpha_{l,h,i} . W_O^{(l,h)} v_{l,h,i} ||_2

    and makes two demands this shape exists to satisfy:

    1. **`W_O` is not optional.** [P5] measures the norm of the *transformed*
       vector including the output projection, and `W_O` is precisely where
       head-specific rescaling lives. Dropping it reintroduces the confound that
       norm-weighting was adopted to remove.
    2. **Report the per-layer profile once before collapsing to a scalar.** Summing
       across layers conflates contributions to different residual-stream
       positions, and [P2] App. C-D show memory gates and cross-attention gradient
       share are strongly depth-stratified, so the layer sum is not obviously the
       right aggregate.

    Storing `alpha` and `W_O v` separately rather than their product keeps the
    attention-sink diagnostic [P14] available: a slot can receive large `alpha`
    while contributing almost nothing to the residual stream, and that gap is the
    whole reason `r_i` is not raw attention probability.
    """

    alpha: Tensor
    """`[L, H, M]` attention probability over slots, per layer and head.

    `L` is the number of **cross-attention** layers, which is **six**, not twelve
    (D-E): TG alternates self/cross blocks `S,C,S,C,...` over 12 layers, so cross
    attention lives at `l in {2,4,6,8,10,12}`. A twelve-entry profile would be six
    real rows and six zeros, and the zeros would be read as a depth finding."""

    wo_v: Tensor
    """`[L, H, M, d_model]` the value vector after the head's output projection."""

    live: Tensor
    """`[M]` bool, matching the `MemoryState` at capture time."""

    step: int

    eval_mode: bool
    """Whether the forward pass that produced this trace ran with dropout off.

    **Required, not defaulted** (D-F). `attn_dropout = 0.2` in the reference
    `tg_config.py`, so during training `alpha` is stochastically zeroed -- and the
    zeroing is *policy relevant*: a slot can score zero demand because a mask fell
    on it, and an on-policy `r_i` would be noisy in exactly the direction that
    matters. The retention target is collected in **eval** mode; the LM loss is
    computed in **train** mode. Making this a required field is the point: an
    ambient default is how the two get confused."""

    gate: Tensor | None = None
    """`[L]` TG's learnable per-layer memory gate `g_mem`, or None if not captured.

    **`r_i` includes it** (D-E). [P2] puts a learnable scalar on each
    cross-attention layer scaling the increment *before* the residual add, and
    App. C measures it growing over training and larger in deeper layers. D-6's
    argument for keeping `W_O` -- "precisely where head-specific rescaling lives"
    -- applies verbatim to `g_mem`, which is where *layer*-specific rescaling
    lives, and because the gates grow over training the weighting is
    **non-stationary**: `r_i` at epoch 1 and at epoch 12 are not the same
    measurement. Computed both ways and reported both against LOO in E0d."""


@runtime_checkable
class RetentionPolicy(Protocol):
    """What every eviction arm implements.

    The contract is deliberately narrow: a policy chooses which slot dies, and
    observes what happened. It does not own the memory, write gestalts, or touch
    the transformer -- section 3.3 forbids any retention gradient reaching the
    transformer or `W_sent`, and a policy that cannot see them cannot violate that
    by accident.
    """

    name: str

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        """Return the index of the slot to evict. Called only when memory is full.

        `context` is `c_t`, the current sentence gestalt (section 3.2.2). It
        arrives **stop-gradiented on the transformer side**: `psi_hat` is a
        function of representations that are themselves training, and section 3.3
        permits gradient to `phi` only.

        Must return an index where `slots.live` is True.
        """
        ...

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        """Record what the model actually retrieved at this step.

        This is where H2O accumulates attention, where RSR computes realized
        `r_i(t)`, and where the anti-collapse loop updates `u_bar`. Called every
        step, including steps where nothing is evicted, and including steps before
        `T_warm` where the policy is inert but `psi_hat` still trains passively on
        realized `r_i` (section 3.4).
        """
        ...

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """A gestalt was just written into `slot`. **Called on every write.**

        Gauntlet 0.4's root cause. Without this hook the protocol never tells a
        policy that a slot changed occupant, so per-slot accumulated state silently
        describes the *previous* tenant. Three consequences, all of which bite:

        1. **LRU was silently FIFO.** Its `last_used` lived in `MemoryState.accum`,
           and a training loop that builds a fresh `MemoryState` per step erased it
           every step. Under the natural call order -- evict, write, forward,
           observe -- LRU's eviction sequence was byte-identical to FIFO's. Section
           7.1 makes "RSR must beat LRU" the behavioural vacuity test and section
           10.1 makes LRU one of E7's two legitimate controls, so the comparator was
           crippled in RSR's favour.
        2. **H2O would have inherited it.** Its accumulated attention must reset to
           zero for a new occupant, or a slot inherits the heat of whatever used to
           live there and the heavy-hitter set is an artifact of slot reuse.
        3. Expire-Span's learned span, and the anti-collapse loop's `u_bar`, have
           the same shape.

        A policy with no per-slot state implements this as a no-op, and says so.
        """
        ...

    def reset(self) -> None:
        """Clear per-stream state at a stream boundary.

        [P2] fact 2: memory is reset (stop-gradient) at each stream boundary, and
        the anti-collapse bias `b` resets with it -- which is exactly why defect
        D-1's `gamma_b = 0.001` was inert, since `b` never got more than `S` steps
        to accumulate. Anything that must survive a boundary belongs on the policy
        object, not in `MemoryState.accum`.
        """
        ...
