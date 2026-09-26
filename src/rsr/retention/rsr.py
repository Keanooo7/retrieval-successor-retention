"""The RSR policy (spec sections 3.3, 3.4).

Eviction rule:

    i* = argmin_i [ z(psi_hat_phi(s_i, c_t)) + b_i - nu * max_{j != i} cos(s_i, s_j) ]

**Marginal value, not independent value** (defect D-3). `psi_hat` scores each slot
in isolation, but what makes a memory worth keeping is the loss it saves *that no
other retained slot can save*. Two slots carrying the same proposition split the
attention mass, both look half as valuable, and both become evictable -- losing
the information entirely. Set value under leave-one-out is **submodular**, so
greedy argmin over independent scores is not an approximation to marginal value;
where slots are redundant it is anti-correlated with it (falsifier 5).

**Eviction warmup is required** (defect C5):

    k <  T_warm : eviction is FIFO; psi_hat trains passively on realized r_i
    k >= T_warm : eviction switches to argmin[ z(psi_hat) + b - nu * ... ]

`k` is the **optimizer step** -- training progress -- not the sentence index `t`
inside a stream (correction 31). The warmup protects an untrained head early in
*training*; compared against `t` it became either a FIFO prefix of every stream or,
once `T_warm >= S`, FIFO forever. The training loop hands the policy `k` through
`set_train_step()`.

`psi_hat` is a function of representations that are themselves training, and [P2]
states early sentence representations are "largely uninformative". An untrained
head evicts near-randomly during exactly the period when those evictions shape the
gestalts it later depends on. Warmup also gives a clean on-policy/off-policy
boundary: everything before `T_warm` is unbiased FIFO-collected data.

**Learning rule: Monte-Carlo is the default; TD(0) is ablation A8** (correction 2,
defect D-8):

    G_i(t) = sum_{k=0}^{S-t} gamma^k * r_i(t+k)
    L_MC   = sum_i ( psi_hat_phi(s_i, c_t) - sg[ G_i(t) ] )^2

Streams are finite (`S <= 80`), `gamma <= 0.97`, no gradient reaches the
transformer, and only the *policy* must act online -- the *target* need not. MC
removes the bootstrap, the moving target, the EMA target copy and the divergence
risk outright. The cost is variance, plus censoring, which the shadow buffer
handles.

**Reversion check against v0.1.** v0.1's degeneracy was *algebraic*: the same
function of the same argument on both sides of a bootstrap, collapsing the fixed
point to `r_i(t)/(1-gamma)`. MC has no bootstrap, so that collapse cannot recur by
construction. But v0.4's defence -- "there is no closed-form regression that
produces this" -- **is false under MC and is withdrawn**: MC *is* a regression.
What distinguishes it from v0.1 is the target, not the estimator class. Anyone
re-deriving this must check the target, not the presence of a bootstrap.

---

## Gauntlet 0.1 and 0.2 -- `T_warm = inf` made the whole apparatus decorative

Dispatch is `k < T_warm -> FIFO` (`k` the optimizer step since correction 31; it
read `t`, the sentence index, until then). `inf` makes that true forever.

* `reduction_to_tg()` set it, so **under the section 3.7 reduction no eviction ever
  reached the score** and E0b certified FIFO against FIFO. The gate that exists to
  prove RSR reduces to TG was passing without ever executing the thing being
  reduced. It is now `0.0`, so the reduction runs **through the score path** and
  `psi_hat == -a_i` has to actually produce the oldest slot.
* The **default** `RSRConfig` set it too, so a default-constructed RSR policy was
  stock TG forever while reporting `name = "rsr"` -- arm and control silently one
  arm.

The default is fixed by **removing it**. `nu`, `beta`, `gamma`, `t_warm` and
`a_max` have no defaults at all now: they are MEASURED or CONDITIONAL constants
(gauntlet 0.3), and a dataclass default is exactly the "frozen unmeasured constant"
that section 4.5 forbids and that defect D-1 was. Build configs with
`RSRConfig.from_registry()`, which raises `UnmeasuredConstant` on an empty ledger.

## Section 3.7 reduction, reachable by configuration alone

`psi_override="neg_age"`, `b_enabled=False`, `nu=0.0`, `beta=0.0`, `gamma=0.0`,
`t_warm=0.0`, `shadow_enabled=False`, `a_max=M`, `sent_pos_index="rank"`,
`context_source="current_sentence_gestalt"`. Eviction becomes `argmin(-a_i)` = the
oldest slot. Bit-exact TG.

**`REDUCTION_SWITCHES` below is the single enumeration of that set**, and a test
asserts it covers every field of `RSRConfig`. Add a term without an off-switch and
the test reddens -- which is the mechanical form of "every term added to the
eviction score ships a documented off-switch, or section 3.7 stops being a
reduction and E0b stops testing what it claims to test."
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from rsr import constants as C
from rsr.baselines.fifo import FIFOPolicy
from rsr.retention.instrumentation import EvictionRecord, RankShiftLog, rank_shift
from rsr.retention.policy import AttentionTrace, MemoryState
from rsr.retention.value_head import BilinearValueHead

__all__ = [
    "CAPACITY",
    "REDUCTION_SWITCHES",
    "ContextSource",
    "RSRConfig",
    "RSRPolicy",
    "SentPosIndex",
]


class ContextSource(StrEnum):
    """What `c_t` is (D-C).

    **Decided: the current sentence gestalt.** Three reasons, in the order they
    carry weight:

    1. It is the simplest reading of section 3.2.2, whose own text is *"`c_t` is
       the current sentence gestalt (or a running context vector)"* -- the
       parenthesis is an aside, not a second specification.
    2. It makes **both arguments to the bilinear form unit-norm** (correction 15 /
       D-B: the reference L2-normalizes every gestalt), so the muP analysis is
       single-valued instead of forked. A running context vector has no norm
       guarantee at all, and section 4.3's multiplier would need a different
       derivation for each branch.
    3. It keeps `psi_hat` positionally blind. A running context accumulates over
       the stream and is therefore weakly time-indexed, which would hand the
       estimator an indirect age channel -- see ADR-0006, where that is the
       independent argument for the same decision.

    The running-context version is a clean **ablation**, not a branch in the
    critical path. It is named here, and it raises.
    """

    CURRENT_SENTENCE_GESTALT = "current_sentence_gestalt"
    RUNNING_CONTEXT = "running_context"


class SentPosIndex(StrEnum):
    """How `P^(sent)` indexes the memory (ADR-0006 / D-D).

    `RANK` is [P2] section 2.2's own indexing -- position in a memory "ordered from
    oldest to most recent" -- and is what the released code does, so it is what the
    fidelity harness compares against (D-A). It is the default and it is what the
    section 3.7 reduction pins.

    `ABSOLUTE_AGE` indexes by `t - written_at` instead, so a slot's key depends only
    on the slot. It is a **research variant, not a fix**: it changes TG, it breaks
    fidelity against the pinned reference, and anything run under it must be
    reported as a modified base model. Default off; **must** be `RANK` in the
    reduction, or E0b tests a model the fidelity harness never saw.
    """

    RANK = "rank"
    ABSOLUTE_AGE = "absolute_age"


CAPACITY = object()
"""Sentinel in `REDUCTION_SWITCHES` for "whatever `M` is in this configuration"."""

REDUCTION_SWITCHES: dict[str, Any] = {
    "psi_override": "neg_age",
    "b_enabled": False,
    "nu": 0.0,
    "beta": 0.0,
    "gamma": 0.0,
    "t_warm": 0.0,
    "shadow_enabled": False,
    "a_max": CAPACITY,
    "context_source": ContextSource.CURRENT_SENTENCE_GESTALT,
    "sent_pos_index": SentPosIndex.RANK,
}
"""Section 3.7's switch set, enumerated once.

`tests/test_reduction.py` asserts this dict's keys are **exactly** the fields of
`RSRConfig`. A new term with no entry here reddens that test, which is the only
mechanical guarantee that section 3.7 stays a reduction.

`t_warm = 0.0`, not `inf`: gauntlet 0.1. With `inf` the dispatch routed every
eviction to FIFO and the reduction never exercised the score path at all.
"""


@dataclass(frozen=True)
class RSRConfig:
    """Every term in the eviction score has an off-switch in `REDUCTION_SWITCHES`.

    **No defaults for registry-owned values** (gauntlet 0.3). `nu`, `beta` and
    `gamma` are MEASURED from E1; `t_warm` and `a_max` are CONDITIONAL. A dataclass
    default for any of them is a frozen unmeasured constant one import away from the
    module built to prevent exactly that -- and `nu = 0.0` is additionally the
    section 3.7 *disabled* value, so the default arm shipped the off-switch.
    """

    nu: float
    beta: float
    gamma: float
    t_warm: float
    a_max: int
    psi_override: str | None = None
    """None = use the learned head. "neg_age" = section 3.7's `psi_hat == -a_i`."""

    b_enabled: bool = True
    shadow_enabled: bool = True
    context_source: ContextSource = ContextSource.CURRENT_SENTENCE_GESTALT
    sent_pos_index: SentPosIndex = SentPosIndex.RANK

    @classmethod
    def reduction_to_tg(cls, capacity: int) -> RSRConfig:
        """Section 3.7's settings. Consumed by `tests/test_reduction.py`.

        Built **from** `REDUCTION_SWITCHES` rather than repeating it, so the
        enumeration and the config cannot drift apart.
        """
        kw = {
            k: (capacity if v is CAPACITY else v) for k, v in REDUCTION_SWITCHES.items()
        }
        return cls(**kw)

    @classmethod
    def from_registry(
        cls,
        scope: str,
        *,
        steps_per_epoch: float,
        registry: C.Registry | None = None,
        **overrides: Any,
    ) -> RSRConfig:
        """Read every measured/conditional value from the constants registry.

        **This is the only supported way to build a training config** (gauntlet 0.3
        and 0.5). On an empty ledger it raises `UnmeasuredConstant`, naming E1 --
        which is the point: the registry refusing the read is what stops an
        unmeasured constant being frozen by a default.

        `overrides` exists for ablation arms that deliberately depart from the
        frozen value (A2's `A_max = M`, the `gamma = 0` control arm). Every override
        is a deliberate, visible departure -- not a default.
        """
        reg = registry or C.REGISTRY
        s = reg.get("S", scope)
        # S0-01: the reads are LAZY, one thunk per field, and a field the caller
        # overrode is never read. Built eagerly, `overrides` could not save you: on
        # an empty ledger `from_registry(..., nu=0.0, gamma=0.0)` raised
        # `UnmeasuredConstant` on `nu` -- the value the caller had just supplied --
        # so the `gamma = 0` control arm and A2's `A_max = M`, the two arms this
        # docstring names as the reason `overrides` exists, could not be built at
        # all until E1 had logged constants neither arm uses.
        #
        # 🔴 This is not a default, and the distinction is the whole D-1 guard. A
        # field the caller did NOT override is still read, and still raises naming
        # its experiment. `tests/test_train_loop.py` asserts both halves so the fix
        # cannot decay into the thing it is careful not to be.
        sources: dict[str, Any] = {
            "nu": lambda: reg.get("nu", scope),
            "beta": lambda: reg.get("beta", scope),
            "gamma": lambda: reg.get("gamma", scope),
            "t_warm": lambda: reg.get("T_warm", steps_per_epoch=steps_per_epoch),
            "a_max": lambda: reg.get("A_max", scope, S=s),
        }
        kw: dict[str, Any] = {
            k: read() for k, read in sources.items() if k not in overrides
        }
        kw.update(overrides)
        cfg = cls(**kw)
        C.check_gamma_horizon(cfg.gamma, s=s, a_max=cfg.a_max)
        return cfg

    def is_reduction(self, capacity: int) -> bool:
        """True iff every switch sits at its section 3.7 off value."""
        return self == RSRConfig.reduction_to_tg(capacity)


class RSRPolicy:
    """Learned eviction (sections 3.3-3.5), with section 3.4 decision attribution.

    Every eviction produces an `EvictionRecord`. That is not instrumentation added
    for comfort: section 3.4 requires per-eviction attribution **from the first
    policy run**, gauntlet 0.2 is a run whose records would all have read
    `fifo_warmup`, and ADR-0006 needs the rank-shift field on the same record.
    """

    name = "rsr"

    def __init__(
        self,
        config: RSRConfig,
        d_model: int,
        *,
        value_head: BilinearValueHead | None = None,
        bias: object | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if config.context_source is not ContextSource.CURRENT_SENTENCE_GESTALT:
            raise NotImplementedError(
                f"context_source={config.context_source.value!r} is an ablation "
                f"(D-C), not the critical path. c_t is the current sentence "
                f"gestalt: it is section 3.2.2's simplest reading, it makes both "
                f"arguments to the bilinear form unit-norm so the muP analysis is "
                f"single-valued, and it keeps psi_hat positionally blind "
                f"(ADR-0006). Implement the ablation deliberately, in its own arm."
            )
        if config.sent_pos_index is not SentPosIndex.RANK:
            # 🔴 A switch that is set and does nothing is worse than no switch.
            # `sent_pos_index` selects how TG indexes `P^(sent)` on the memory
            # KEYS -- it is consumed by the model, not by the policy, and the
            # PyTorch TG that would consume it does not exist yet (gauntlet 2.4).
            # Until it does, accepting the value here would let someone run an
            # "absolute-age" arm that was rank-indexed throughout.
            raise NotImplementedError(
                f"sent_pos_index={config.sent_pos_index.value!r} is a MODEL-side "
                f"switch (ADR-0006 item 3): it changes how TG indexes P^(sent) on "
                f"the memory keys, and the policy never sees the positional "
                f"encoding at all. It lands with the PyTorch TG transcription "
                f"(gauntlet 2.4) and must stay 'rank' until then -- an arm that "
                f"reported absolute-age indexing while running rank-indexed would "
                f"be worse than no arm. In the reference the ablated variant is "
                f"stm_cross_pos_mode='none' / stm_positional_weight=0.0."
            )
        if config.b_enabled and bias is None:
            raise ValueError(
                "b_enabled=True needs the anti-collapse bias (section 3.5), whose "
                "gamma_b and tau are measured by E0e. Pass one, or set "
                "b_enabled=False and say so -- b == 0 is also the section 3.7 "
                "reduction condition, so a silently absent b is defect D-1's shape."
            )
        self.config = config
        self.d_model = d_model
        self.bias = bias
        self.head: BilinearValueHead | None = None
        if config.psi_override is None:
            self.head = value_head or BilinearValueHead(d_model, generator=generator)
        elif config.psi_override != "neg_age":
            raise ValueError(
                f"psi_override={config.psi_override!r} is not a known override. "
                f"Section 3.7 defines exactly one: 'neg_age'."
            )
        self._fifo = FIFOPolicy()
        self.records: list[EvictionRecord] = []
        # Correction 31: the warmup counter is the optimizer step, owned by the
        # training loop. None until told -- see `select_eviction`.
        self._train_step: int | None = None
        self.rank_shifts = RankShiftLog()

    # -- the eviction rule --------------------------------------------------- #

    def to(self, device: torch.device | str) -> RSRPolicy:
        """Move `phi` to `device`. Returns self, so it chains like `nn.Module.to`.

        🔴 **A policy is device-bound even though it is not an `nn.Module`.**
        `RSRPolicy` held its value head on the CPU while the memory lived on the
        accelerator, and every tensor it created was a CPU tensor. Both paths --
        the learned head AND `neg_age`, which has no head at all -- raised
        `Expected all tensors to be on the same device` the first time the policy
        ran on MPS. E0b did not catch it because §3.7's reduction runs on CPU by
        design (ADR-0001 D3), so the entire RSR arm would have failed at the first
        accelerated run.
        """
        if self.head is not None:
            self.head.to(device)
        return self

    def _device_of(self, slots: MemoryState) -> torch.device:
        """The memory's device is authoritative; the policy follows it."""
        device = slots.gestalts.device
        if self.head is not None and next(self.head.parameters()).device != device:
            self.head.to(device)
        return device

    def _psi(self, slots: MemoryState, context: Tensor) -> Tensor:
        """`[M]` raw `psi_hat` per slot, before z-scoring. Dead slots are +inf."""
        device = self._device_of(slots)
        dead = torch.full((slots.capacity,), float("inf"), device=device)
        if self.config.psi_override == "neg_age":
            # Section 3.7: psi_hat == -a_i. argmin(-a) = the oldest slot = FIFO.
            psi = -slots.ages().to(torch.float32)
        else:
            assert self.head is not None
            # Section 3.3: only phi receives gradient. The policy is the caller, so
            # the policy applies the stop-gradient -- the head deliberately does
            # not, so that a caller who meant to backpropagate cannot do it
            # silently.
            psi = self.head(slots.gestalts.detach(), context.detach())
        return torch.where(slots.live, psi, dead)

    def _score(self, slots: MemoryState, context: Tensor) -> tuple[Tensor, str]:
        psi = self._psi(slots, context)
        live = slots.live
        finite = psi[live]
        # Section 3.5 item 1: z-scored across live slots, which is what makes
        # b_max = 1.0 mean one standard deviation. Monotone, so it cannot change
        # the argmin -- it exists to put b on a common scale.
        sd = finite.std(unbiased=False)
        if sd > 0:
            psi = torch.where(live, (psi - finite.mean()) / sd, psi)
        terms = "neg_age" if self.config.psi_override else "psi"

        score = psi
        if self.config.b_enabled:
            assert self.bias is not None
            score = score + self.bias.b(slots)  # type: ignore[attr-defined]
            terms += "+b"
        if self.config.nu != 0.0:
            score = score - self.config.nu * self._max_cosine(slots)
            terms += "-nu"
        return torch.where(live, score, torch.full_like(score, float("inf"))), terms

    def _max_cosine(self, slots: MemoryState) -> Tensor:
        """`[M]` `max_{j != i} cos(s_i, s_j)` over live slots (defect D-3)."""
        g = torch.nn.functional.normalize(slots.gestalts.detach(), dim=-1)
        sim = g @ g.T
        sim.fill_diagonal_(-1.0)
        sim = torch.where(slots.live.unsqueeze(0), sim, torch.full_like(sim, -1.0))
        best = sim.max(dim=1).values
        return torch.where(slots.live, best, torch.zeros_like(best))

    def set_train_step(self, k: int) -> None:
        """Tell the policy which optimizer step is running (§3.4; correction 31).

        Survives `reset()`: it is training progress, not per-stream state. The
        training loop calls this before every stream batch.
        """
        self._train_step = int(k)

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        # `step` is the sentence index inside this stream; it is recorded, never
        # compared against T_warm (correction 31).
        if self.config.t_warm > 0 and self._train_step is None:
            raise RuntimeError(
                f"T_warm = {self.config.t_warm} optimizer steps, but no training step "
                "was set: call policy.set_train_step(k) first. Defaulting to 0 would "
                "make every eval-time use of a trained policy silently FIFO "
                "(correction 31; gauntlet 0.1)."
            )
        warm = self.config.t_warm > 0 and self._train_step < self.config.t_warm
        if warm:
            victim = self._fifo.select_eviction(slots, context, step)
            attribution, margin = "fifo_warmup", None
        else:
            # Section 3.5: `b` never enters psi_hat "or any differentiable path --
            # eviction argmin only". The whole selection is therefore under
            # no_grad. psi_hat's gradient comes from the retention loss in
            # `observe`, never from the decision.
            with torch.no_grad():
                score, attribution = self._score(slots, context)
                victim = int(score.argmin().item())
                ordered = score[slots.live].sort().values
                margin = float(ordered[1] - ordered[0]) if ordered.numel() > 1 else None

        victim_rank, shifted = rank_shift(slots, victim)
        record = EvictionRecord(
            step=step,
            victim=victim,
            victim_rank=victim_rank,
            rank_shift=shifted,
            n_live=slots.n_live,
            policy=self.name,
            warm=warm,
            attribution=attribution,
            score_margin=margin,
            train_step=self._train_step,
        )
        self.records.append(record)
        self.rank_shifts.add(record)
        return victim

    # -- bookkeeping --------------------------------------------------------- #

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        """Realized `r_i` lands here. Sprint 2 wires it to the MC return.

        The trace must be an eval-mode capture (D-F); `rsr.retention.reward`
        refuses a train-mode one rather than quietly averaging over dropout masks.
        """
        raise NotImplementedError(
            "Sprint 2: the MC return and the shadow buffer (sections 3.3-3.4). "
            "r_i itself is implemented in rsr.retention.reward."
        )

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """No per-slot accumulator yet; the MC return keyed by slot arrives with
        `observe` in Sprint 2, and it resets here (gauntlet 0.4)."""
        return None

    def reset(self) -> None:
        """Stream boundary. The eviction log persists -- it is the run's record, not
        per-stream state -- but anything keyed by slot does not."""
        self._fifo.reset()

    # -- reporting ----------------------------------------------------------- #

    def attribution_counts(self) -> dict[str, int]:
        """How many evictions each term selected. **A run that is all
        `fifo_warmup` is stock TG reporting itself as RSR** (gauntlet 0.2)."""
        out: dict[str, int] = {}
        for r in self.records:
            out[r.attribution] = out.get(r.attribution, 0) + 1
        return out
