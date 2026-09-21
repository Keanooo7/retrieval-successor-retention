"""Retention headroom -- how much any eviction rule could buy on a corpus (E-feas).

`src/rsr/baselines/oracle.py` asks it: *if oracle ~= FIFO, that corpus cannot exhibit
the effect.* This module answers it **without a model**. On the synthetic corpus the
demand is known by construction (`Document.pairs`), so whether a policy kept the
sentence a query needs is a fact about the policy and the corpus, not about what a
trained TG does with what it kept. That matters: §10.3 and
`runs/shuffle-control/ledger.json` show this repo's trained memory is inert, so a
headroom read through the model's loss would say "oracle ~= FIFO" for a reason that
has nothing to do with the corpus.

**The memory mechanics are the real ones.** `init_memory`, `write_at` and
`memory_state` from the training path, with a 1-d placeholder gestalt, and the loop
order of `rsr.model.tg.policy_loop.run_policy_loop`:

1. the step's forward reads memory as it stands -- so a query at step `j` **hits** if
   the sentence that asserted its fact is resident *before* step `j`'s write;
2. if the row's memory is full, `policy.select_eviction(state, ctx, j)` names a victim;
3. sentence `j` is written (every synthetic sentence ends in EOS, so every one is).

What this does **not** say: that a model would use a retained sentence, or that
`psi_hat` can learn the oracle's ordering. It bounds what retention can buy; E1
measures what a learned rule does buy.
"""

from __future__ import annotations

from itertools import pairwise
from types import SimpleNamespace

import torch

from rsr.data.synthetic import Document
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import memory_state, write_at


def simulate(doc: Document, policy, memory_slots: int) -> dict:
    """Run one document through a `memory_slots`-slot memory under `policy`.

    Returns one record per query: `{"gap", "hit"}`, plus the residency trace length.
    """
    cfg = SimpleNamespace(M=memory_slots, D=1)
    mem = init_memory(1, cfg)
    ctx = torch.zeros(1)
    placeholder = torch.zeros(1, 1)
    write = torch.ones(1, dtype=torch.bool)
    assert_of = {q: a for a, q in doc.pairs}
    queries = []
    policy.reset()
    for t in range(len(doc.sentences)):
        if t in assert_of:  # the forward reads the pre-write memory
            resident = mem.step[0][mem.valid[0]].tolist()
            queries.append({"gap": t - assert_of[t], "hit": assert_of[t] in resident})
        if bool(mem.valid[0].all()):
            victim = policy.select_eviction(memory_state(mem, t, 0), ctx, t)
        else:
            victim = 0  # unused: the row is not full
        mem = write_at(mem, placeholder, write, torch.tensor([victim]), t)
    return {"queries": queries}


def hit_rate(records: list[dict]) -> float | None:
    """Fraction of queries whose asserting sentence was resident. `None` for none."""
    qs = [q for r in records for q in r["queries"]]
    return sum(q["hit"] for q in qs) / len(qs) if qs else None


def hit_rate_by_gap(records: list[dict], edges: list[int]) -> list[dict]:
    """Hit rate per gap bucket `[edges[k], edges[k+1])`, with counts."""
    qs = [q for r in records for q in r["queries"]]
    out = []
    for lo, hi in pairwise(edges):
        b = [q for q in qs if lo <= q["gap"] < hi]
        out.append(
            {
                "gap_lo": lo,
                "gap_hi": hi,
                "n": len(b),
                "hit_rate": sum(q["hit"] for q in b) / len(b) if b else None,
            }
        )
    return out
