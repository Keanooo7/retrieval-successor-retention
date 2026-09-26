"""E0e -- the pieces of `experiments/e0e/run.py`, each on inputs with a known answer.

PREREG: `experiments/e0e/PREREG.md` (committed alone, before the code). Its mutation
bar names the defects these tests exist to catch: `r_i` read ungated; a half-life
that is not `E_lifetime/4`; lifetime counted over evicted sentences only; the `tau`
quantile moved off 0.75; a non-live seed admitted to the values.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.model.tg import TGConfig, TGModel
from rsr.retention.policy import AttentionTrace
from rsr.train.loop import build_vocab, encode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "e0e" / "run.py"


@pytest.fixture(scope="module")
def e0e():
    spec = importlib.util.spec_from_file_location("e0e_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e0e_run"] = mod  # a @dataclass resolves its module by name
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# r_i: §3.2.1's target, gated (correction 17) and fill-rescaled
# --------------------------------------------------------------------------- #


def test_r_i_is_the_gated_rescaled_target(e0e):
    """Two cross layers with gates 1 and 3, two live slots of M = 4. Slot 0 is
    read only in layer 0 and slot 1 only in layer 1, with equal |alpha W_O v|,
    so the gate alone decides their shares: 1:3, rescaled by n_live/M = 1/2."""
    L, H, M, D = 2, 1, 4, 2
    alpha = torch.zeros(L, H, M)
    alpha[0, 0, 0] = 1.0
    alpha[1, 0, 1] = 1.0
    wo_v = torch.zeros(L, H, M, D)
    wo_v[..., 0] = 1.0
    live = torch.tensor([True, True, False, False])
    trace = AttentionTrace(
        alpha=alpha, wo_v=wo_v, live=live, step=0, eval_mode=True,
        gate=torch.tensor([1.0, 3.0]),
    )  # fmt: skip
    r = e0e.r_of(trace, n_live=2, M=M)
    assert torch.allclose(r, torch.tensor([0.125, 0.375, 0.0, 0.0], dtype=r.dtype))


# --------------------------------------------------------------------------- #
# lifetime, half-life, EMA
# --------------------------------------------------------------------------- #


def _sent(e0e, written, n, evicted):
    s = e0e.Sentence(written=written)
    s.rs = [0.1] * n
    s.steps = list(range(written + 1, written + 1 + n))
    s.full = [True] * n
    s.evicted = evicted
    return s


def test_lifetime_counts_every_written_sentence(e0e):
    """PREREG *Definitions* 1: survivors truncated by the stream end count, and a
    sentence written at the last step counts with lifetime 0."""
    S = 6
    sents = [_sent(e0e, 0, 4, True), _sent(e0e, 1, 4, True), _sent(e0e, 4, 1, False)]
    sents.append(_sent(e0e, 5, 0, False))
    st = e0e.lifetime_stats(sents, S)
    assert st["E_lifetime"] == 9 / 4
    assert st["n_written"] == 4
    assert st["evicted_only_mean"] == 4.0
    assert st["write_to_boundary_mean"] == (4 + 4 + 2 + 1) / 4


def test_the_half_life_is_a_quarter_of_the_lifetime(e0e):
    """§3.5 item 2: `E[lifetime]/4`, not v0.4's frozen 16."""
    assert e0e.half_life(632 / 48) == pytest.approx(3.2916667, abs=1e-6)
    h = e0e.half_life(20.0)
    assert h == 5.0
    assert (1 - e0e.ema_alpha(h)) ** h == pytest.approx(0.5)


def test_the_ema_starts_where_it_is_told(e0e):
    a = 0.5
    assert e0e.ema_series([1.0, 0.0], a, init=0.25) == [0.625, 0.3125]
    assert e0e.ema_series([1.0, 0.0], a, init=None) == [1.0, 0.5]


# --------------------------------------------------------------------------- #
# tau and gamma_b
# --------------------------------------------------------------------------- #


def test_tau_is_the_75th_percentile_of_the_relative_deviation(e0e):
    """PREREG *Definitions* 4: b fires on the 25% of observations outside the band
    -- the 0.25 in §3.5 item 4's `gamma_b` derivation."""
    M = 4
    dev = [i / 100 for i in range(101)]  # |M u - 1| = 0.00 .. 1.00
    ubars = [(1 + d) / M for d in dev]
    assert e0e.tau_from(ubars, M) == pytest.approx(0.75)
    assert e0e.fraction_inside(ubars, M, e0e.tau_from(ubars, M)) == pytest.approx(
        76 / 101
    )
    assert e0e.tau_from(ubars, M, 0.5) == pytest.approx(0.5)


def test_gamma_b_is_the_registrys_formula(e0e):
    import rsr.constants as C

    assert e0e.gamma_b(40.0, 1.0) == pytest.approx(0.1)
    assert e0e.gamma_b(632 / 48, 1.0) == pytest.approx(C._derive_gamma_b(1.0, 632 / 48))


def test_a_seed_that_is_not_live_is_excluded(e0e):
    """PREREG: only `live` seeds contribute. An inert or inconclusive memory's
    shares are not the retrieval demand of a memory that retrieves."""
    bands = {0: "live", 1: "inert", 2: "inconclusive", 3: "invalid", 4: "live"}
    assert e0e.live_seeds(bands) == [0, 4]


# --------------------------------------------------------------------------- #
# consistency checks, and the recorder end to end on a tiny model
# --------------------------------------------------------------------------- #


def test_the_consistency_checks_catch_what_they_name(e0e):
    ok = _sent(e0e, 0, 3, True)
    gap = _sent(e0e, 1, 3, True)
    gap.steps = [2, 4, 5]
    early = _sent(e0e, 3, 1, True)
    early.steps = [3]
    probs = e0e.consistency_problems(
        [ok, gap, early], [(0, 5, 4, 4, 1.0), (0, 6, 4, 4, 0.9), (0, 1, 2, 4, 0.5)]
    )
    assert len(probs) == 4, probs  # one sum, one gap, early twice (first + range)


def test_fifo_lifetimes_on_a_tiny_live_model_are_the_analytic_ones(e0e):
    """S = 10, M = 4, every sentence writes: 6 sentences live 4 steps, the last four
    live 3, 2, 1, 0 -> E_lifetime = 30/10 exactly, 6 evicted, and every
    full-memory step's shares sum to 1."""
    S, M, L = 10, 4, 16
    docs = generate(SyntheticConfig(sentences_per_document=48, seed=0))[:1]
    ids, mask = encode(docs, build_vocab(docs), max_tokens=L, steps=S)
    torch.manual_seed(0)
    cfg = TGConfig(
        D=32, V=4 + len(build_vocab(docs)), F=86, max_sentence_tokens=L,
        max_sentences_in_short_term=M, pad_id=0, bos_id=1, eos_id=2, eod_id=3,
    )  # fmt: skip
    model = TGModel(cfg).eval()
    sents, sums = e0e.fifo_stream(model, ids[0], mask[0], M)
    st = e0e.lifetime_stats(sents, S)
    assert st["E_lifetime"] == 3.0
    assert st["n_evicted"] == 6
    assert [s.lifetime for s in sents] == [4, 4, 4, 4, 4, 4, 3, 2, 1, 0]
    assert e0e.consistency_problems(sents, [(0, *x[1:]) for x in sums]) == []
    assert any(x[2] == M for x in sums)


def test_the_run_never_calls_record(e0e):
    """D2 is the owner's: no `record(` call anywhere in the script's code. The
    would-be calls exist only as strings."""
    tree = ast.parse(RUN.read_text())
    calls = [
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute | ast.Name)
        and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) == "record"
    ]
    assert calls == [], calls
