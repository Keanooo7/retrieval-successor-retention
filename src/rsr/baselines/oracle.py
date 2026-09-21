"""Oracle demand -- the upper bound (spec section 5.4).

Synthetic: exact, because ground-truth discounted demand is known per slot per
step. PG-19: offline via the shadow-buffer machinery (section 3.4) -- **build
once, use twice.**

**Run the oracle first on each corpus as a feasibility probe (E-feas).** If oracle
~= FIFO, that corpus cannot exhibit the effect. One day, saves six weeks.

**Sprint 2 -- typed stub.** Kickoff scope boundary: Sprint 1 implements sections
3.2-3.5 only as far as E0a needs, and stubs the rest against the interfaces.

---

**Implemented for the synthetic corpus, 2026-09-20 (E-feas).** The paragraphs above are
kept as written: the first is the question this class answers, and the last records
what it was for a month. PG-19 is still Sprint 2 -- it needs the shadow buffer to know
future demand, and `OraclePolicy` refuses to be built without a demand matrix rather
than guess one.

The rule is `argmin` over live slots of the **true discounted demand from the next
step on**, `discounted_demand(doc, gamma)[step + 1][sentence]`. On the synthetic corpus
each fact is queried exactly once, so that ordering is the order of next use and the
rule is Belady's MIN for every `gamma` in (0, 1): the optimal eviction, not merely a
good one. That is what makes `oracle - FIFO` an upper bound on what any retention
policy can buy on this corpus (`src/rsr/metrics/headroom.py`).
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState


class OraclePolicy:
    """Evict the live slot whose sentence has the least true discounted demand left.

    `demand[t][i]` is `rsr.data.synthetic.discounted_demand(doc, gamma)` for ONE
    document: the discounted future demand, at step `t`, for the sentence written at
    step `i`. Slots are identified through `MemoryState.written_at`, which is the
    sentence index because the loop writes sentence `t` at step `t`.
    """

    name = "oracle"

    def __init__(self, demand: list[list[float]] | Tensor) -> None:
        if demand is None or len(demand) == 0:
            raise ValueError(
                "OraclePolicy needs the document's true discounted demand. Synthetic: "
                "rsr.data.synthetic.discounted_demand(doc, gamma). PG-19 has no "
                "ground truth until the shadow buffer exists (Sprint 2)."
            )
        self.demand = [[float(x) for x in row] for row in demand]

    def _future(self, sentence: int, step: int) -> float:
        # Demand from the NEXT step on: this step's query has already read memory
        # (the forward runs before the write), so it is not the victim's to lose.
        nxt = step + 1
        return self.demand[nxt][sentence] if nxt < len(self.demand) else 0.0

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        best, best_v = -1, float("inf")
        for k in range(len(slots.live)):
            if not bool(slots.live[k]):
                continue
            v = self._future(int(slots.written_at[k]), step)
            if v < best_v:  # strict: ties go to the lowest slot, i.e. the oldest
                best, best_v = k, v
        return best

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        """No-op: the oracle reads demand from ground truth, not from attention."""
        return None

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """No-op: `written_at`, owned by the memory, is the oracle's only state."""
        return None

    def reset(self) -> None:
        return None
