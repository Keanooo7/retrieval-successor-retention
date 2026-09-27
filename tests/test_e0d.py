"""E0d -- the pieces of `experiments/e0d/run.py`, each on inputs with a known answer.

PREREG: `experiments/e0d/PREREG.md` (committed alone at 5357ad2, before this code).
Spec §3.2.1 (`r_i` vs leave-one-out delta-loss), §6 (E0d is a kill gate),
corrections 17/18/20.

🔴 **No test here generates or reads a document of `D_E0d = [262144, 263168)`.**
The fixtures are tiny synthetic worlds from documents `[0, 64)` of seed 0, which every
committed run already uses. The runner's own guard (`e0d_documents(..., cleared=)`)
refuses the E0d range until the preconditions have passed, and a test below proves
the guard. Nothing here runs `run.py` end to end on arm B.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.metrics import loo
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import cross_capture, trace_for_row, write_at
from rsr.retention.reward import retrieval_demand
from rsr.train.loop import build_vocab, encode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "e0d" / "run.py"
PREREG = ROOT / "experiments" / "e0d" / "PREREG.md"

S, L, M = 16, 10, 4


@pytest.fixture(scope="module")
def e0d():
    spec = importlib.util.spec_from_file_location("e0d_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e0d_run"] = mod  # a @dataclass resolves its module by name
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# fixture world: documents [0, 64) of seed 0, a tiny untrained TG
# --------------------------------------------------------------------------- #


def _cfg(n, seed=0):
    return SyntheticConfig(
        n_documents=n, sentences_per_document=S, max_gap=12, heavy_tail_min=6, seed=seed
    )


def _docs(n=8, seed=0):
    return generate(_cfg(n, seed))


def _vocab(seed=0):
    return build_vocab(_docs(64, seed))


def _model(V, model_seed=0):
    torch.manual_seed(model_seed)
    cfg = TGConfig(
        D=16,
        H=2,
        N=4,
        V=V,
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg)


@pytest.fixture(scope="module")
def world():
    docs = _docs(8)
    vmap = _vocab()
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    model = _model(4 + len(vmap))
    return model, docs, ids, mask


@pytest.fixture(scope="module")
def measured(e0d, world):
    model, docs, ids, mask = world
    return e0d.measure_batch(model, docs, ids, mask, seed=0)


# --------------------------------------------------------------------------- #
# PREREG constants: the runner's numbers are the PREREG's (text parse only)
# --------------------------------------------------------------------------- #


def test_constants_are_the_preregs(e0d):
    """Every registered number, read out of the committed PREREG text, equals the
    runner's constant. A threshold moved after data is not a threshold."""
    text = PREREG.read_text()
    assert "D_E0d = [262144, 263168)" in text
    assert e0d.D_E0D == (262144, 263168)
    assert re.search(r'RHO_STAR: "0\.5 -- PROPOSED', text)
    assert e0d.RHO_STAR_PROPOSED == 0.5
    assert 'RESAMPLE_MIN_COVERAGE: "0.80' in text
    assert e0d.RESAMPLE_MIN_COVERAGE == 0.80
    assert "more than 1% of cells" in text
    assert e0d.UNDERFULL_MAX == 0.01
    assert "within 1e-6" in text
    assert e0d.LIVE_TOL == 1e-6
    assert e0d.SEEDS == (0, 1, 2)
    assert e0d.BATCH == 16
    assert e0d.KEY_STRIDE == 1_000_003
    for s, h in {
        0: "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60",
        1: "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8",
        2: "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8",
    }.items():
        assert f"seed{s}: {h}" in text
        assert e0d.CKPT_SHA256[s] == h


# --------------------------------------------------------------------------- #
# C8 authority: three rulings, and a ratified rho*
# --------------------------------------------------------------------------- #


def _rulings(tmp_path, names, rho_text="RHO_STAR: 0.5\n"):
    d = tmp_path / "rulings"
    d.mkdir()
    for n in names:
        body = rho_text if "rho-star" in n else "ruling\n"
        (d / n).write_text(body)
    return d


ALL3 = (
    "R-2026-10-01-retrieval-shown.md",
    "R-2026-10-01-sprint0-gate.md",
    "R-2026-10-01-rho-star.md",
)


@pytest.mark.parametrize("missing", range(3))
def test_authority_refuses_without_each_ruling(e0d, tmp_path, missing):
    names = [n for i, n in enumerate(ALL3) if i != missing]
    d = _rulings(tmp_path, names)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_authority(d, require_committed=False)
    assert ei.value.control == "C8"


def test_authority_passes_and_reads_the_ratified_rho(e0d, tmp_path):
    d = _rulings(tmp_path, ALL3, rho_text="Ratified.\nRHO_STAR = 0.45\n")
    out = e0d.check_authority(d, require_committed=False)
    assert out["rho_star"] == 0.45
    assert set(out["rulings"]) == {"retrieval_shown", "sprint0_gate", "rho_star"}


def test_a_rho_ruling_that_states_no_value_is_refused(e0d, tmp_path):
    d = _rulings(tmp_path, ALL3, rho_text="I ratify it.\n")
    with pytest.raises(e0d.ControlFailed):
        e0d.check_authority(d, require_committed=False)
    (d / ALL3[2]).write_text("RHO_STAR: 0.5\nRHO_STAR: 0.7\n")  # two values
    with pytest.raises(e0d.ControlFailed):
        e0d.check_authority(d, require_committed=False)


def test_the_real_rulings_dir_does_not_authorise_e0d_today(e0d):
    """At this commit none of the three rulings exists (PREREG C8)."""
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_authority(ROOT / "docs" / "owner" / "rulings")
    assert ei.value.control == "C8"


# --------------------------------------------------------------------------- #
# C2 disjointness on the generator key
# --------------------------------------------------------------------------- #


def test_disjointness_is_on_the_generator_key_across_seeds(e0d):
    """Doc i of seed s is doc i + 1_000_003 of seed s - 1 (§2.1). A used range of
    seed 0 above one key stride aliases E0d documents of seed 1 whose indices it
    does not contain -- an index check misses it, the key check must not."""
    K = e0d.KEY_STRIDE
    used = [{"name": "x", "range": (K, K + 64), "seeds": (0,)}]
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_disjoint((0, 10), seeds=(1,), used=used)
    assert ei.value.control == "C2"
    # the same indices under seed 0 (keys 0..9) do not collide
    e0d.check_disjoint((0, 10), seeds=(0,), used=used)


def test_disjointness_catches_a_plain_index_overlap(e0d):
    used = [{"name": "x", "range": (100, 200), "seeds": (0, 1, 2)}]
    with pytest.raises(e0d.ControlFailed):
        e0d.check_disjoint((199, 300), seeds=(2,), used=used)
    e0d.check_disjoint((200, 300), seeds=(0, 1, 2), used=used)


def test_the_committed_manifests_leave_d_e0d_disjoint(e0d):
    """Arithmetic on the committed manifests (no document generated)."""
    used = e0d.used_ranges(ROOT / "runs")
    names = {u["name"] for u in used}
    for want in (
        "fresh-stream.vocab_documents",
        "fresh-stream.stream_documents",
        "fresh-escape.stream_documents",
        "corpus-size-curve.arms_n_documents",
        "retention-readability.sets.E",
        "carry-forward.sets.EXT",
        "e0e.heldout_documents",
        "scaffold-dose.stream_documents.k600",
    ):
        assert want in names, want
    assert max(u["range"][1] for u in used) == 148160
    out = e0d.check_disjoint(e0d.D_E0D, seeds=e0d.SEEDS, used=used)
    assert out["n_keys"] == 3 * 1024 and out["collisions"] == 0


def test_a_missing_manifest_is_a_refusal_not_a_skip(e0d, tmp_path):
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.used_ranges(tmp_path)
    assert ei.value.control == "C2"


# --------------------------------------------------------------------------- #
# the E0d-range guard, and C3 vocabulary closure
# --------------------------------------------------------------------------- #


def test_the_e0d_range_is_not_generated_until_cleared(e0d, monkeypatch):
    calls = []
    monkeypatch.setattr(e0d, "_generate_document", lambda i, cfg: calls.append(i))
    with pytest.raises(RuntimeError, match="not cleared"):
        e0d.e0d_documents(0, 48, e0d.D_E0D, cleared=False)
    with pytest.raises(RuntimeError, match="not cleared"):
        e0d.e0d_documents(0, 48, (262000, 262200), cleared=False)  # overlaps
    assert calls == []
    # a fixture range is fine
    docs = e0d.e0d_documents(0, 48, (0, 3), cleared=False)
    assert calls == [0, 1, 2] and len(docs) == 3


def test_vocabulary_closure_is_a_runtime_control(e0d):
    vocab = build_vocab(_docs(1))  # one document's words only
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_vocab(_docs(8), vocab)
    assert ei.value.control == "C3"
    e0d.check_vocab(_docs(1), vocab)


# --------------------------------------------------------------------------- #
# the measurement on a tiny live model
# --------------------------------------------------------------------------- #


def test_r_i_is_read_from_the_same_forward_gated_and_rescaled(e0d, world, measured):
    """Recompute step 5's r_i by hand: FIFO to step 5, capture the forward that
    reads sentence 5 over the pre-write memory, `retrieval_demand` gated and not."""
    model, _, ids, mask = world
    model.eval()
    B = ids.shape[0]
    cfg = model.cfg
    with torch.no_grad():
        mem = init_memory(B, cfg)
        bos = torch.zeros(B, cfg.D)
        bv = torch.zeros(B, dtype=torch.bool)
        for t in range(6):
            out = model(ids[:, t], mask[:, t], mem.kv, mem.valid, bos, bv, capture=True)
            if t == 5:
                cap = cross_capture(model, out, mem.kv, mask[:, t], mem.valid)
                for b in range(B):
                    tr = trace_for_row(cap, mem.valid, b, t)
                    n = int(mem.valid[b].sum())
                    for gated, key in ((True, "r_gated"), (False, "r_ungated")):
                        want = retrieval_demand(tr, n_live=n, capacity=M, gated=gated)
                        got = measured[key][b, t]
                        assert torch.allclose(got, want.double(), atol=1e-12), key
            mem = write_at(mem, out.srep, out.has_eos, torch.zeros(B).long(), t)
            bos, bv = out.srep, out.has_eos & (t + 1 < S)
    # gated and ungated really differ on a model whose gates differ
    assert not torch.equal(measured["r_gated"], measured["r_ungated"]) or (
        model.blocks[1].memory_gate.item() == model.blocks[3].memory_gate.item()
    )


def test_r_i_sums_to_fill_share_and_is_zero_on_dead_slots(e0d, measured):
    r = measured["r_gated"]
    n_live = measured["n_live"].double()
    tot = r.sum(-1)
    ok = tot > 0
    assert torch.allclose(tot[ok], (n_live / M)[ok], atol=1e-9)
    dead = measured["slot_sentence"] < 0
    assert (r[dead] == 0).all()


def test_delta_is_loo_delta_loss_verbatim(e0d, world, measured):
    model, docs, ids, mask = world
    for mode in ("resample", "zero"):
        ref = loo.loo_delta_loss(model, docs, ids, mask, mode=mode, seed=0)
        got = measured[f"delta_{mode}"]
        assert torch.equal(got.isnan(), ref["delta"].isnan())
        assert torch.equal(got.nan_to_num(0.0), ref["delta"].nan_to_num(0.0))


def test_alignment_control_passes_on_the_real_pass(e0d, measured):
    assert measured["c4"]["live_loss_max_abs_diff"] <= e0d.LIVE_TOL
    assert measured["c4"]["slot_map_equal"]
    assert measured["c4"]["eval_mode"]


def test_alignment_control_catches_a_misaligned_slot_map(e0d, world, monkeypatch):
    model, docs, ids, mask = world
    real = loo.loo_delta_loss

    def shifted(*a, **k):
        out = real(*a, **k)
        out["slot_sentence"] = out["slot_sentence"].roll(1, dims=1)
        return out

    monkeypatch.setattr(e0d, "loo_delta_loss", shifted)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.measure_batch(model, docs, ids, mask, seed=0)
    assert ei.value.control == "C4"


def test_alignment_control_catches_a_live_loss_mismatch(e0d, world, monkeypatch):
    model, docs, ids, mask = world
    real = loo.loo_delta_loss

    def off(*a, **k):
        out = real(*a, **k)
        out["live_loss"] = out["live_loss"] + 1e-4
        return out

    monkeypatch.setattr(e0d, "loo_delta_loss", off)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.measure_batch(model, docs, ids, mask, seed=0)
    assert ei.value.control == "C4"


def test_fifo_slot_map_control(e0d, measured):
    """C9: the memory each forward read is the FIFO one given the write flags."""
    e0d.check_fifo(measured["slot_sentence"], measured["wrote"], M)
    bad = measured["slot_sentence"].clone()
    bad[0, M + 2] = bad[0, M + 2].flip(0)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_fifo(bad, measured["wrote"], M)
    assert ei.value.control == "C9"


def test_determinism_control(e0d, measured):
    e0d.check_determinism(measured, measured)
    other = dict(measured)
    other["delta_zero"] = measured["delta_zero"].clone()
    i = torch.nonzero(~other["delta_zero"].isnan())[0]
    other["delta_zero"][tuple(i)] += 1e-12
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_determinism(measured, other)
    assert ei.value.control == "C7"


# --------------------------------------------------------------------------- #
# cells, C5 fill, C6 resample coverage
# --------------------------------------------------------------------------- #


def test_cells_are_full_memory_occupied_slots_with_targets(e0d, world, measured):
    docs = world[1]
    c = e0d.cells_from([measured], [docs], M)
    assert set(np.unique(c["rank"])) <= set(range(M))
    assert (c["n_live"][c["full"]] == M).all()
    assert (c["t"][c["full"]] >= M).all()
    # a cell exists for every non-NaN zero-knockout delta, and no other
    assert len(c["t"]) == int((~measured["delta_zero"].isnan()).sum())
    assert set(np.unique(c["kind"])) <= set(e0d.CONTENT_KINDS)


def test_fill_control_counts_and_refuses_above_one_percent(e0d):
    n_live = np.array([4] * 99 + [3])
    t = np.full(100, 10)
    out = e0d.check_fill(n_live, t, M)
    assert out["underfull"] == 1 and out["fraction"] == pytest.approx(0.01)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_fill(np.array([4] * 98 + [3, 3]), t, M)
    assert ei.value.control == "C5"


def test_resample_coverage_below_080_is_exit_3(e0d):
    full = np.ones(10, dtype=bool)
    d_res = np.array([0.1] * 8 + [np.nan] * 2)
    assert e0d.check_coverage(d_res, full)["coverage"] == pytest.approx(0.8)
    d_res[7] = np.nan
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_coverage(d_res, full)
    assert ei.value.control == "C6"


def test_coverage_is_a_runtime_control_on_a_small_batch(e0d, world):
    """Two documents per batch leave most slots without a same-kind donor: the
    runner refuses rather than read a primary on the survivors."""
    model, docs, ids, mask = world
    m = e0d.measure_batch(model, docs[:2], ids[:2], mask[:2], seed=0)
    c = e0d.cells_from([m], [docs[:2]], M)
    cov = np.mean(~np.isnan(c["d_resample"][c["full"]]))
    assert cov < e0d.RESAMPLE_MIN_COVERAGE  # the fixture is built to fall short
    with pytest.raises(e0d.ControlFailed):
        e0d.check_coverage(c["d_resample"], c["full"])


# --------------------------------------------------------------------------- #
# main(): C8 -> C1 -> C2, then exit 3 before any document (pending amendment)
# --------------------------------------------------------------------------- #


class _FakeLedger:
    last = None
    STATUSES = ()

    def __init__(self, run_id, **kw):
        self.run_id, self.rows, self.meta = run_id, {}, {}
        self._status, self.commands = None, []
        _FakeLedger.last = self

    def manifest(self, cfg):
        self.cfg = cfg

    def run_meta(self, **kw):
        self.meta.update(kw)

    def note(self, k, v, *, how):
        self.rows[k] = v

    def stat(self, k, xs, *, how, allow_single=False):
        self.rows[k] = list(xs)

    def status(self, s):
        self._status = s

    def command(self, argv, *, exit_code, **kw):
        self.commands.append(exit_code)

    def verdict(self, **kw):
        self.v = kw

    def write(self):
        return Path("/dev/null")


@pytest.fixture
def fake_ledger(monkeypatch):
    import ledger as real

    monkeypatch.setattr(_FakeLedger, "STATUSES", real.STATUSES)
    fake = type("ledger", (), {"Ledger": _FakeLedger, "STATUSES": real.STATUSES})
    monkeypatch.setitem(sys.modules, "ledger", fake)
    return _FakeLedger


def test_main_refuses_before_any_e0d_document_without_the_rulings(
    e0d, monkeypatch, fake_ledger
):
    gen = []
    monkeypatch.setattr(e0d, "_generate_document", lambda i, cfg: gen.append(i))
    monkeypatch.setattr(e0d, "sha256", lambda p: pytest.fail("C1 before C8"))
    code = e0d.main([])
    assert int(code) == 3
    assert gen == []
    assert fake_ledger.last._status == "did_not_run"
    assert fake_ledger.last.commands == [3]


def test_main_exits_3_before_any_document_even_when_preconditions_pass(
    e0d, monkeypatch, fake_ledger
):
    """At this commit the analysis awaits the PREREG amendment, so a cleared run
    still refuses (3) and generates nothing -- never a silent 0."""
    gen = []
    monkeypatch.setattr(e0d, "_generate_document", lambda i, cfg: gen.append(i))
    monkeypatch.setattr(
        e0d, "check_authority", lambda *a, **k: {"rho_star": 0.5, "rulings": {}}
    )
    monkeypatch.setattr(e0d, "check_substrate", lambda root: {})
    monkeypatch.setattr(e0d, "used_ranges", lambda runs: [])
    monkeypatch.setattr(e0d, "measure_seed", lambda *a, **k: pytest.fail("measured"))
    code = e0d.main([])
    assert int(code) == 3
    assert gen == []
    assert fake_ledger.last._status == "did_not_run"
    assert "not_implemented" in fake_ledger.last.rows


def test_main_ledgers_the_failing_control(e0d, monkeypatch, fake_ledger):
    monkeypatch.setattr(
        e0d, "check_authority", lambda *a, **k: {"rho_star": 0.5, "rulings": {}}
    )

    def c1(root):
        raise e0d.ControlFailed("C1", "sha mismatch")

    monkeypatch.setattr(e0d, "check_substrate", c1)
    code = e0d.main([])
    assert int(code) == 3
    assert fake_ledger.last.rows["control_failed"]["control"] == "C1"


def test_substrate_recheck_refuses_a_failing_manifest(e0d, tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    man = tmp_path / "MANIFEST.sha256"
    man.write_text("0" * 64 + "  x.bin\n")
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.substrate_recheck(man, cwd=tmp_path)
    assert ei.value.control == "C1"
    import hashlib

    man.write_text(hashlib.sha256(b"abc").hexdigest() + "  x.bin\n")
    assert e0d.substrate_recheck(man, cwd=tmp_path)["rc"] == 0


def test_the_runner_imports_no_retention_policy(e0d):
    """C9 / T3: no RSRPolicy, no b, nu = 0, no shadow -- FIFO only. The runner
    imports nothing that could evict other than FIFO."""
    import ast

    tree = ast.parse(RUN.read_text())
    mods, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
    assert not any(
        m.startswith(
            ("rsr.retention.rsr", "rsr.retention.shadow", "rsr.retention.balance")
        )
        for m in mods
    ), mods
    assert not names & {"RSRPolicy", "ShadowBuffer", "BalanceController"}, names
