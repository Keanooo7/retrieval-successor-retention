"""Per-eviction decision attribution, and D-D's rank-shift logging.

Section 3.4 requires per-eviction decision attribution **from the first policy
run**, not added later once something looks wrong. ADR-0006 adds one field to that
record and makes it mandatory for every RSR run.

## What rank shift is, and why it is logged rather than argued

[P2] section 2.2 adds `P^(sent)_{1:Mt}` to the memory **keys** over a memory
"ordered from oldest to most recent". The index is the slot's **rank in that
ordering**, not its age.

**Ranks shift under FIFO too, and that is not the confound.** The memory is a
sliding window: every step the window advances and every surviving slot's rank drops
by one. Under FIFO the victim is always rank 1, so the shift is uniform and
`rank(i)` stays a deterministic function of `age(i)`. `P^(sent)` then means exactly
what the paper's prose says it means -- recency.

**Under RSR the victim can be a middle slot, and then the map from age to rank
breaks.** Every slot older than the victim survives an eviction FIFO would have
spent on the oldest, and every slot behind them carries a positional index one lower
than its age implies. **A slot's positional encoding becomes a function of which
other slots the policy killed.** The RSR and FIFO arms then differ in more than the
eviction rule, which is the exact property E0b exists to guarantee.

So the statistic is **displacement relative to the FIFO counterfactual**, not raw
movement. `victim_rank - 1` is that displacement, and it is 0 under FIFO by
construction. Counting slots that merely moved would report 7-of-8 under stock TG
and measure the sliding window.

🔴 **E0b structurally cannot catch it.** Under section 3.7's reduction the policy
*is* FIFO, so no rank ever shifts and the test passes correctly while saying
nothing. The confound is invisible in the reduction and present in every arm the
project cares about. Hence this module.

**Report the distribution, not the mean.** A policy that evicts the oldest slot 95%
of the time and the newest 5% of the time has a small mean rank shift and a real
confound.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean

from rsr.retention.policy import MemoryState

__all__ = ["EvictionRecord", "RankShiftLog", "rank_shift"]


def rank_shift(slots: MemoryState, victim: int) -> tuple[int, int]:
    """`(victim_rank, displacement)` for evicting `victim`.

    `victim_rank` is 1-based in the oldest-to-newest memory ordering.

    `displacement` is `victim_rank - 1`: the number of live slots **older than the
    victim**, each of which survives an eviction FIFO would have spent on the
    oldest. It is the count of slots whose `P^(sent)` index no longer follows from
    their age, and therefore the size of the divergence from the ordering the
    paper's prose describes.

    **Under FIFO this is `(1, 0)` always** -- which is the property that makes it a
    measure of the confound rather than a measure of the sliding window. A count of
    slots that merely *moved* is 7-of-8 under stock TG and says nothing.
    """
    ranks = slots.ranks()
    victim_rank = int(ranks[victim].item())
    if victim_rank < 0:
        raise ValueError(
            f"slot {victim} is not live; a policy must return a live slot "
            f"(RetentionPolicy.select_eviction)."
        )
    return victim_rank, victim_rank - 1


@dataclass(frozen=True)
class EvictionRecord:
    """One eviction, attributable (section 3.4) and rank-instrumented (ADR-0006)."""

    step: int
    victim: int
    victim_rank: int
    """1-based position in the oldest-to-newest memory ordering."""

    rank_shift: int
    """Displacement from the FIFO counterfactual: how many live slots older than
    the victim survived this eviction, and so how many slots' `P^(sent)` index
    stopped following from their age. **0 under FIFO, always.**"""

    n_live: int
    policy: str
    warm: bool
    """True if `t < T_warm`, i.e. this eviction was FIFO by dispatch and **not**
    attributable to the score. Section 3.4's warmup boundary is also section 7.2's
    on-policy/off-policy boundary, so it has to be on the record."""

    attribution: str
    """Which term selected this slot: `"fifo_warmup"`, `"psi"`, `"psi+b"`,
    `"psi+b-nu"`, or `"neg_age"` under the section 3.7 reduction. A policy run whose
    records are all `"fifo_warmup"` is stock TG reporting itself as RSR -- gauntlet
    0.2, and the reason this field is not optional."""

    score_margin: float | None = None
    """`score[runner_up] - score[victim]`. A margin at or near zero means the
    decision was a tie the argmin broke arbitrarily, not a decision the score made."""


@dataclass
class RankShiftLog:
    """Accumulates ADR-0006's distribution across a run.

    Reported every RSR run. Two derived numbers go in the heartbeat (gauntlet 3.8):
    the fraction of evictions that shift anything at all, and the mean displacement.
    """

    counts: Counter[int] = field(default_factory=Counter)
    victim_ranks: Counter[int] = field(default_factory=Counter)

    def add(self, record: EvictionRecord) -> None:
        self.counts[record.rank_shift] += 1
        self.victim_ranks[record.victim_rank] += 1

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    @property
    def fraction_shifting(self) -> float:
        """How often the confound fires at all. Exactly 0.0 under FIFO."""
        if not self.n:
            return 0.0
        return 1.0 - (self.counts.get(0, 0) / self.n)

    @property
    def mean_shift(self) -> float:
        if not self.n:
            return 0.0
        return sum(k * v for k, v in self.counts.items()) / self.n

    def distribution(self) -> dict[int, int]:
        """The histogram itself. **This is the reportable object** -- the two
        scalars above are a summary of it and must not replace it."""
        return dict(sorted(self.counts.items()))

    def summary(self) -> dict[str, object]:
        return {
            "n_evictions": self.n,
            "fraction_shifting": self.fraction_shifting,
            "mean_shift": self.mean_shift,
            "mean_victim_rank": (
                mean(self.victim_ranks.elements()) if self.victim_ranks else 0.0
            ),
            "shift_distribution": self.distribution(),
            "victim_rank_distribution": dict(sorted(self.victim_ranks.items())),
        }
