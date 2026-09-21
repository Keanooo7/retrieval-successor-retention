"""The demand-controlled synthetic corpus (spec section 5.3, gauntlet 2.5).

🔴 **This is the artifact E1 and E2 are built on**, and the gauntlet recorded it as
existing in neither repo. On this machine `src/rsr/data/` did exist -- as three
stubs that raised `NotImplementedError`. Stubs are not the thing.

The acceptance criteria are 2.5's, verbatim:

* a fact asserted at *i* and queried at *i+k*, **max gap 40**
* ground-truth demand known **per slot per step**
* **byte-identical across two separate processes** at one seed
* a different seed gives different bytes
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from rsr.data.synthetic import (
    SyntheticConfig,
    discounted_demand,
    generate,
    to_bytes,
    true_demand,
)

CFG = SyntheticConfig(n_documents=16, seed=0)


@pytest.fixture(scope="module")
def docs():
    return generate(CFG)


# --- the structure: asserted at i, queried at i+k ---------------------------- #


def test_every_pair_is_an_assert_followed_by_its_query(docs):
    for doc in docs:
        for assert_at, query_at in doc.pairs:
            a, q = doc.sentences[assert_at], doc.sentences[query_at]
            assert a.kind == "assert" and q.kind == "query"
            assert a.fact_id == q.fact_id
            assert query_at > assert_at


def test_a_query_can_be_answered_from_the_sentence_it_points_at(docs):
    """The query's answer must be recoverable from the asserted sentence -- if it
    were not, the corpus would not be demand-controlled, it would be noise."""
    for doc in docs:
        for assert_at, query_at in doc.pairs:
            assert doc.sentences[query_at].answer in doc.sentences[assert_at].text


def test_no_sentence_serves_two_roles(docs):
    for doc in docs:
        asserted = [a for a, _ in doc.pairs]
        queried = [q for _, q in doc.pairs]
        assert len(set(asserted)) == len(asserted)
        assert len(set(queried)) == len(queried)
        assert not set(asserted) & set(queried)


def test_the_max_gap_is_forty(docs):
    gaps = [g for doc in docs for g in doc.gaps]
    assert max(gaps) <= 40
    assert min(gaps) >= 1


def test_the_gap_distribution_is_a_mixture_not_a_single_mode(docs):
    """Section 5.3: short-range geometric **and** a heavy tail. A pure geometric at
    p = 0.30 essentially never produces a gap past 20, and then `gamma` has nothing
    to earn its keep on -- E1 would be measuring the optimizer on a corpus with no
    lookahead to exploit."""
    gaps = sorted(g for doc in docs for g in doc.gaps)
    assert gaps[len(gaps) // 2] <= 8, "the short-range mode is missing"
    assert sum(g > 20 for g in gaps) / len(gaps) > 0.05, "the heavy tail is missing"


def test_a_generator_whose_max_gap_does_not_fit_in_S_is_refused():
    """Section 5.2, as a config invariant: gaps longer than `S` do not exist in the
    data, so a generator that cannot fit its own longest gap silently produces a
    shorter-tailed corpus. That was the largest hole in v0.1."""
    with pytest.raises(ValueError, match="max_gap"):
        SyntheticConfig(sentences_per_document=30, max_gap=40)


def test_S_is_48_on_synthetic_and_exceeds_the_max_gap():
    assert CFG.sentences_per_document == 48 > CFG.max_gap == 40


# --- ground truth ------------------------------------------------------------ #


def test_true_demand_is_one_exactly_where_a_query_retrieves(docs):
    doc = docs[0]
    r = true_demand(doc)
    n = len(doc.sentences)
    expected = {(q, a) for a, q in doc.pairs}
    for t in range(n):
        for i in range(n):
            assert r[t][i] == (1.0 if (t, i) in expected else 0.0)


def test_discounted_demand_is_gamma_to_the_gap(docs):
    """The whole point of the corpus: `psi_hat`'s regression target is known in
    closed form, so it is measurable against truth rather than through perplexity."""
    doc = docs[0]
    for gamma in (0.0, 0.5, 0.9, 0.97):
        g = discounted_demand(doc, gamma)
        for assert_at, query_at in doc.pairs:
            gap = query_at - assert_at
            assert g[assert_at][assert_at] == pytest.approx(gamma**gap)


def test_demand_is_zero_after_the_query_has_been_answered(docs):
    """A fact retrieved once is never retrieved again in this corpus, so its
    discounted demand is 0 from the step after its query. That is what makes
    eviction regret against the oracle exactly computable."""
    doc = docs[0]
    g = discounted_demand(doc, 0.9)
    for assert_at, query_at in doc.pairs:
        assert g[query_at][assert_at] == pytest.approx(1.0)
        if query_at + 1 < len(doc.sentences):
            assert g[query_at + 1][assert_at] == 0.0


def test_the_gamma_097_horizon_attenuates_the_longest_gaps(docs):
    """The caveat the writeup has to state: horizon 33 < max gap 40, so the longest
    gaps enter at ~0.30 weight. **Attenuated, not invisible.**"""
    assert pytest.approx(0.296, abs=0.005) == 0.97**40
    assert pytest.approx(1 / 2.718, abs=0.02) == 0.97**33


def test_gamma_zero_leaves_only_the_instantaneous_reward(docs):
    """Section 2's separating control arm: context conditioning without lookahead."""
    doc = docs[0]
    assert discounted_demand(doc, 0.0) == true_demand(doc)


def test_demand_rejects_an_illegal_gamma(docs):
    with pytest.raises(ValueError, match="gamma"):
        discounted_demand(docs[0], 1.0)


# --- determinism: 2.5's hard criterion --------------------------------------- #

_CHILD = textwrap.dedent(
    """
    import hashlib, sys
    from rsr.data.synthetic import SyntheticConfig, generate, to_bytes
    b = to_bytes(generate(SyntheticConfig(n_documents=16, seed=int(sys.argv[1]))))
    sys.stdout.write(hashlib.sha256(b).hexdigest())
    """
)


def _child_digest(seed: int, hashseed: str) -> str:
    import os

    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(seed)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return proc.stdout.strip()


def test_byte_identical_across_two_separate_processes():
    """2.5's criterion, and it is run in real subprocesses rather than simulated.

    The two children get **different `PYTHONHASHSEED`s**, because the failure mode
    worth catching is an ordering that depends on string hashing -- which is
    reproducible within one process and not across two.
    """
    a = _child_digest(0, "0")
    b = _child_digest(0, "12345")
    assert a == b, "the corpus is not reproducible across processes"


def test_a_different_seed_gives_different_bytes():
    assert _child_digest(0, "0") != _child_digest(1, "0")


def test_the_in_process_corpus_matches_the_subprocess_digest():
    """Ties the fixture above to the subprocess check, so a future change that
    breaks only one of them is still caught."""
    import hashlib

    assert hashlib.sha256(to_bytes(generate(CFG))).hexdigest() == _child_digest(0, "0")


def test_generation_does_not_touch_the_global_rng():
    """A generator that consumes global draws shifts everything downstream of it --
    the same hazard §3.7 documents for the value head, in a different module."""
    import random

    random.seed(1234)
    before = [random.random() for _ in range(3)]
    random.seed(1234)
    generate(SyntheticConfig(n_documents=4, seed=9))
    after = [random.random() for _ in range(3)]
    assert before == after


def test_a_document_does_not_depend_on_the_documents_before_it():
    """Per-document seeding. Otherwise generating a subset gives a different corpus
    from taking a subset, and the two are indistinguishable in a RESULTS.md."""
    full = generate(SyntheticConfig(n_documents=8, seed=3))
    just_one = generate(SyntheticConfig(n_documents=8, seed=3))[5]
    assert full[5] == just_one
    sliced = generate(SyntheticConfig(n_documents=6, seed=3))
    assert sliced == full[:6]


# --- S0-03: the answer is in the token stream -------------------------------- #
#
# Before S0-03 every query read "What does Hal-6 measures?" and its answer lived
# only in `Sentence.answer`, out of band: no next-token target required retrieving
# the asserted fact, so "the memory is inert" was a finding about the corpus.

M_SYNTHETIC = 16
"""`rsr.constants.get("M", "synthetic")`, asserted equal below rather than trusted."""

PRE_S003_DIGEST = "521eb69c7132d846895647e1683e305d9fcb8bcdb8ed96495245045c860d8181"
"""sha256 of `to_bytes(generate(SyntheticConfig(n_documents=16, seed=0)))` measured
on the unmodified generator at a802521, before S0-03 touched it."""


def test_every_query_carries_its_answer_as_its_final_token(docs):
    """The S0-03 invariant, and the gate its fixture mutation reddens: re-emitting
    the answer out of band (`answer_in_stream=False` as the default) must fail
    here, by name."""
    from rsr.data.synthetic import answer_symbol

    n_queries = 0
    for doc in docs:
        for assert_at, query_at in doc.pairs:
            q = doc.sentences[query_at]
            assert q.text.split()[-1] == answer_symbol(q.answer), q.text
            # ...and the asserted sentence is where it can be retrieved from.
            assert q.answer in doc.sentences[assert_at].text
            n_queries += 1
    assert n_queries > 0


def test_an_answer_is_one_symbol_and_chance_on_it_is_ln_16():
    """Single-symbol, not word-level: one chance figure, `ln 16`, for every answer.
    A symbol must not collide with any word a model could copy from elsewhere."""
    import math

    from rsr.data.synthetic import ANSWER_SYMBOLS

    assert len(ANSWER_SYMBOLS) == len(set(ANSWER_SYMBOLS)) == 16
    assert math.log(len(ANSWER_SYMBOLS)) == pytest.approx(2.7726, abs=1e-4)
    words = {
        w
        for d in generate(SyntheticConfig(seed=0, answer_in_stream=False))
        for s in d.sentences
        for w in s.text.split()
    }
    assert not words & set(ANSWER_SYMBOLS)
    assert all(" " not in s for s in ANSWER_SYMBOLS)


def test_the_off_switch_is_the_pre_s003_corpus_byte_for_byte():
    """`answer_in_stream=False` must reproduce the old corpus exactly, so the runs
    measured on it (`runs/efeas-synthetic`, `runs/shuffle-control`) stay
    reproducible at this sha."""
    import hashlib

    off = generate(SyntheticConfig(n_documents=16, seed=0, answer_in_stream=False))
    assert hashlib.sha256(to_bytes(off)).hexdigest() == PRE_S003_DIGEST


def test_the_answer_is_the_only_difference_between_the_two_corpora():
    """No RNG draw is added, so facts, gaps and fillers are identical and the only
    change is one appended token per query. Otherwise "the corpus alone" in the
    decisive run would also change the gap distribution."""
    from rsr.data.synthetic import answer_symbol

    on = generate(SyntheticConfig(seed=1))
    off = generate(SyntheticConfig(seed=1, answer_in_stream=False))
    for a, b in zip(on, off, strict=True):
        assert a.pairs == b.pairs
        for sa, sb in zip(a.sentences, b.sentences, strict=True):
            if sa.kind == "query":
                assert sa.text == f"{sb.text} {answer_symbol(sb.answer)}"
            else:
                assert sa == sb


def test_a_reportable_fraction_of_pairs_has_gap_beyond_M():
    """S0-03 item 4: eviction can only bite on pairs FIFO cannot retrieve, i.e.
    `gap > M`. Measured at a802521 on the default config: 0.195, 0.174, 0.174 for
    seeds 0, 1, 2. The floor is 0.10 -- below it the `gap > M` answer-loss bucket
    rests on too few targets per seed to compare with the `gap <= M` bucket."""
    from rsr import constants as C
    from rsr.data.synthetic import fraction_of_pairs_beyond

    assert C.get("M", "synthetic") == M_SYNTHETIC
    for seed in (0, 1, 2):
        frac = fraction_of_pairs_beyond(generate(SyntheticConfig(seed=seed)), 16)
        assert 0.10 <= frac < 1.0, (seed, frac)


# --- S0-03: the supervision mask (`rsr.train.loop.answer_targets`) ----------- #


def _encoded(seed=0, n_documents=8, **kw):
    from rsr.train.loop import answer_targets, build_vocab, encode

    d = generate(SyntheticConfig(n_documents=n_documents, seed=seed, **kw))
    v = build_vocab(d)
    ids, mask = encode(d, v, max_tokens=64, steps=48)
    tmask, gap = answer_targets(d, v, max_tokens=64, steps=48)
    return d, v, ids, mask, tmask, gap


def test_the_target_mask_marks_exactly_the_answer_token_of_every_query():
    """One target per query, zero elsewhere, and the id under it is the answer's.
    Scored as `target_mask[..., 1:]` against the shifted targets, the same frame
    `encode()`'s padding mask is scored in."""
    from rsr.data.synthetic import answer_symbol

    d, v, ids, mask, tmask, _ = _encoded()
    assert not (tmask & ~mask).any(), "an answer target on a PAD position"
    assert not tmask[..., 0].any(), "position 0 is never a next-token target"
    for di, doc in enumerate(d):
        for si, s in enumerate(doc.sentences):
            row = tmask[di, si]
            if s.kind == "query":
                assert int(row.sum()) == 1
                pos = int(row.nonzero())
                assert int(ids[di, si, pos]) == v[answer_symbol(s.answer)]
                assert int(ids[di, si, pos + 1]) == 2, "the EOS follows the answer"
            else:
                assert not row.any()


def test_the_gap_tensor_is_the_pairs_gap_at_each_query():
    d, _, _, _, tmask, gap = _encoded()
    for di, doc in enumerate(d):
        want = {q: q - a for a, q in doc.pairs}
        for si in range(len(doc.sentences)):
            assert int(gap[di, si]) == want.get(si, 0)
    assert bool(((gap > 0) == tmask.any(dim=-1)).all())


def test_the_answer_targets_are_a_minority_that_a_pooled_loss_would_hide():
    """Why the mask exists: pooled into the padding mask the answers are a few
    percent of the real targets (8.5% at seed 0 over 64 documents)."""
    _, _, _, mask, tmask, _ = _encoded(n_documents=64)
    frac = int(tmask[..., 1:].sum()) / int(mask[..., 1:].sum())
    assert 0.0 < frac < 0.15


def test_answer_targets_refuse_a_corpus_whose_answers_are_out_of_band():
    """An empty mask would make every answer-token loss a mean over nothing; the
    pre-S0-03 corpus must be refused, not silently scored."""
    with pytest.raises(ValueError, match="out of band"):
        _encoded(answer_in_stream=False)
