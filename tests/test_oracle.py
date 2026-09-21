"""The oracle and the headroom simulator, pinned where the answer is known.

E-feas reads `oracle - FIFO` as the most any retention rule could buy on a corpus. Two
ways that number could lie, each with a test:

* **the oracle is not optimal** -- a hand-built stream where keeping one sentence
  across a gap longer than memory is possible, and FIFO provably cannot;
* **the simulator's timing is off by a step** -- FIFO keeps exactly the last `M`
  sentences, so on the real corpus it must hit a query **iff** its gap is `<= M`.
  An off-by-one in "reads before the write" moves that boundary and fails here.
"""

from __future__ import annotations

import pytest
import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.baselines.oracle import OraclePolicy
from rsr.data.synthetic import (
    Document,
    Sentence,
    SyntheticConfig,
    discounted_demand,
    generate,
)
from rsr.metrics.headroom import hit_rate, simulate

GAMMA = 0.97


def _doc(n: int, pairs: list[tuple[int, int]]) -> Document:
    kinds = {a: "assert" for a, _ in pairs} | {q: "query" for _, q in pairs}
    sentences = tuple(
        Sentence(index=i, kind=kinds.get(i, "filler"), text=f"s{i}") for i in range(n)
    )
    return Document(doc_id=0, sentences=sentences, pairs=tuple(pairs))


def _oracle(doc: Document) -> OraclePolicy:
    return OraclePolicy(discounted_demand(doc, GAMMA))


def test_the_oracle_keeps_a_sentence_across_a_gap_fifo_cannot():
    # M = 2, one fact asserted at 0 and queried at 4, fillers between. FIFO holds
    # {2, 3} at step 4 and misses; the oracle evicts the fillers instead.
    doc = _doc(5, [(0, 4)])
    assert simulate(doc, FIFOPolicy(), 2)["queries"] == [{"gap": 4, "hit": False}]
    assert simulate(doc, _oracle(doc), 2)["queries"] == [{"gap": 4, "hit": True}]


def test_the_oracle_ties_fifo_when_every_gap_fits():
    doc = _doc(6, [(0, 1), (2, 4), (3, 5)])
    for policy in (FIFOPolicy(), _oracle(doc)):
        assert all(q["hit"] for q in simulate(doc, policy, 2)["queries"])


def test_fifo_hits_exactly_the_gaps_that_fit_on_the_real_corpus():
    m = 16
    for doc in generate(SyntheticConfig(n_documents=8, seed=0)):
        for q in simulate(doc, FIFOPolicy(), m)["queries"]:
            assert q["hit"] == (q["gap"] <= m), q


def test_the_oracle_never_loses_to_fifo_on_the_real_corpus():
    for doc in generate(SyntheticConfig(n_documents=8, seed=0)):
        fifo = hit_rate([simulate(doc, FIFOPolicy(), 16)])
        oracle = hit_rate([simulate(doc, _oracle(doc), 16)])
        assert oracle >= fifo, (doc.doc_id, oracle, fifo)


def test_the_oracle_refuses_to_run_without_ground_truth():
    with pytest.raises(ValueError, match="shadow buffer"):
        OraclePolicy([])


def test_the_oracle_evicts_a_sentence_nobody_will_ask_for():
    doc = _doc(5, [(0, 4)])
    oracle = _oracle(doc)
    from rsr.retention.policy import MemoryState

    # Slot 0 holds sentence 0 (queried at 4); slot 1 holds filler 1. At step 2 the
    # oracle must drop the filler.
    state = MemoryState(
        gestalts=torch.zeros(2, 1),
        written_at=torch.tensor([0, 1]),
        live=torch.tensor([True, True]),
        step=2,
    )
    assert oracle.select_eviction(state, torch.zeros(1), 2) == 1
