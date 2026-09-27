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
import json
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
    model = TGModel(cfg)
    # Distinct memory gates, so gated and ungated r_i differ (correction 17): with
    # every g_mem at its init 1.0 the two forms coincide and a swap is invisible.
    with torch.no_grad():
        for g, block in zip((0.5, 2.0), (b for b in model.blocks if b.block_type == "C"),
                            strict=True):  # fmt: skip
            block.memory_gate.fill_(g)
    return model


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
    # the fixture's gates differ, so the two forms must differ somewhere
    assert not torch.equal(measured["r_gated"], measured["r_ungated"])


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
    # 0.75 sits between any relaxed floor and 0.80: still a refusal
    three_of_four = np.tile([0.1, 0.1, 0.1, np.nan], 5)
    with pytest.raises(e0d.ControlFailed):
        e0d.check_coverage(three_of_four, np.ones(20, dtype=bool))


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

    @property
    def path(self):
        return _FakeLedger.root / self.run_id / "ledger.json"

    def write(self):
        """Writes the rows to disk, as the real ledger does: A1.4's exit guard
        reads `e0d.class` back out of the written file."""
        self.writes = getattr(self, "writes", 0) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"key": k, "value": v} for k, v in self.rows.items()]
        self.path.write_text(
            json.dumps({"rows": rows, "status": self._status}, default=str)
        )
        return self.path


@pytest.fixture
def fake_ledger(monkeypatch, tmp_path):
    import ledger as real

    monkeypatch.setattr(_FakeLedger, "root", tmp_path / "runs", raising=False)
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


# =========================================================================== #
# PREREG Amendment 1 (84e21c5, erratum 268b947): strata, statistics, C10,
# flatness, labels, classification, exits, and the A1.5-A1.7 guards.
# Every fixture below is hand-built or from documents [0, 64) of seed 0.
# =========================================================================== #

A1 = PREREG.read_text().split("## Amendment 1", 1)[1]


def _ref_rank(x):
    """Average ranks by brute force (1-based), independent of the runner's ranker."""
    x = np.asarray(x, dtype=float)
    return np.array([(x < v).sum() + ((x == v).sum() + 1) / 2 for v in x])


def _ref_spearman(x, y):
    return float(np.corrcoef(_ref_rank(x), _ref_rank(y))[0, 1])


def _cells(n_docs=40, *, M=4, t0=4, n_steps=12, score="good", a_delta=0.5,
           noise=1e-3, seed=0):  # fmt: skip
    """Hand-built cells with the columns `cells_from` returns, every one full.
    Every third step is a query; its assert is resident (gap <= M) at FIFO rank
    M - gap, and there Delta is `a_delta`. Elsewhere Delta is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_docs):
        for t in range(t0, t0 + n_steps):
            q = t % 3 == 0
            gap = 1 + (d + t) % (M + 2) if q else -1
            a_rank = M - gap if q and gap <= M else -1
            for i in range(M):
                a = i == a_rank
                dr = (a_delta if a else 0.0) + rng.normal(0, noise)
                dz = (2 * a_delta if a else 0.0) + rng.normal(0, noise)
                rows.append((d, t, i, t - M + i, q, a, gap, dr, dz))
    cols = list(zip(*rows, strict=True))
    n = len(rows)
    c = {
        "doc": np.array(cols[0]),
        "t": np.array(cols[1]),
        "rank": np.array(cols[2]),
        "sentence": np.array(cols[3]),
        "n_live": np.full(n, M),
        "q_step": np.array(cols[4], dtype=bool),
        "a_cell": np.array(cols[5], dtype=bool),
        "gap": np.array(cols[6]),
        "d_resample": np.array(cols[7]),
        "d_zero": np.array(cols[8]),
    }
    c["pc"] = c["a_cell"].astype(float)
    d = c["d_resample"]
    if score == "good":
        r = d - d.min() + 1e-3
    elif score == "bad":
        r = rng.uniform(0.01, 1.0, n)
    elif score == "inverted":
        r = d.max() - d + 1e-3
    else:
        raise ValueError(score)
    c["r_gated"], c["r_ungated"] = r, r.copy()
    c["layer_gated"] = np.stack([r, r], 1)
    c["layer_ungated"] = np.stack([r, r], 1)
    c["pre_share_gated"] = np.ones(n)
    c["pre_share_ungated"] = np.ones(n)
    c["kind"] = np.where(c["a_cell"], "pending_assert", "filler").astype(object)
    c["full"] = np.ones(n, dtype=bool)
    return c


def _W(e0d, c, n_boot=60, seed=7):
    return e0d.bootstrap_doc_weights(int(c["doc"].max()) + 1, n_boot, seed)


def test_amendment_constants_are_the_preregs(e0d):
    """A1's numbers, read back out of the committed text."""
    assert "median over A-cells of `Δ_resample` is < 1e-2 nats" in A1
    assert e0d.LOO_FLAT_A == 1e-2
    assert "q90 over cells of |delta_resample| < 1e-3 nats" in PREREG.read_text()
    assert e0d.LOO_FLAT_Q90 == 1e-3
    assert "`< 0.05`" in PREREG.read_text() and e0d.R_FLAT == 0.05
    assert "2000 resamples, generator seed 20260927, 95% percentile" in (
        PREREG.read_text()
    )
    assert (e0d.BOOT_N, e0d.BOOT_SEED, e0d.CI_LEVEL) == (2000, 20260927, 0.95)
    assert "t_cap ≤ 16123" in A1 and e0d.B5_T_CAP_MAX == 16123
    assert "ρ\\* ∈ {0.3, 0.7}" in A1  # noqa: RUF001
    assert e0d.SENSITIVITY_RHO == (0.3, 0.7)


# --------------------------------------------------------------------------- #
# A1.1 strata: fixed by document structure, never by Delta or r_i
# --------------------------------------------------------------------------- #


def test_strata_are_fixed_by_document_structure(e0d, world, measured):
    from rsr.data.synthetic import true_demand

    docs = world[1]
    c = e0d.cells_from([measured], [docs], M)
    for j in range(len(c["t"])):
        doc, t, s = docs[c["doc"][j]], int(c["t"][j]), int(c["sentence"][j])
        qa = {q: a for a, q in doc.pairs}
        q = doc.sentences[t].kind == "query"
        assert bool(c["q_step"][j]) == q
        assert bool(c["a_cell"][j]) == (q and s == qa[t])
        assert int(c["gap"][j]) == (t - qa[t] if q else -1)
        assert c["pc"][j] == true_demand(doc)[t][s]
    st = e0d.strata(c)
    assert np.array_equal(st["Q"], c["full"] & c["q_step"])
    assert np.array_equal(st["N"], c["full"] & ~c["q_step"])
    assert np.array_equal(st["A"], st["Q"] & c["a_cell"])
    assert st["Q"].any() and st["A"].any() and st["N"].any()
    # never by Delta: scrambling Delta and r_i leaves every stratum unchanged
    c2 = dict(c)
    rng = np.random.default_rng(0)
    c2["d_resample"] = rng.permutation(c["d_resample"])
    c2["r_gated"] = rng.permutation(c["r_gated"])
    for k, v in e0d.strata(c2).items():
        assert np.array_equal(v, st[k]), k


def test_rho_q_is_on_query_step_cells_only(e0d):
    """rho_Q (primary) reads Q-step cells; N-step cells cannot move it."""
    c = _cells(score="good")
    Q, N = c["q_step"], ~c["q_step"]
    r = c["r_gated"].copy()
    r[N] = -r[N]  # anti-agreement everywhere off the query steps
    keep = np.ones(len(r), dtype=bool)
    s = e0d.score_stats(c, r, c["d_resample"], keep=keep, W=_W(e0d, c), M=4)
    assert s["rho_Q"] == pytest.approx(_ref_spearman(r[Q], c["d_resample"][Q]))
    assert s["rho_Q"] == pytest.approx(1.0)
    assert s["rho_N"] == pytest.approx(-1.0)
    assert s["rho_pool"] == pytest.approx(_ref_spearman(r, c["d_resample"]))
    lo, hi = s["rho_Q_ci"]
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_rho_q_rank_is_the_count_weighted_within_rank_mean(e0d):
    c = _cells(score="bad")
    r, d = c["r_gated"], c["d_resample"]
    keep = np.ones(len(r), dtype=bool)
    s = e0d.score_stats(c, r, d, keep=keep, W=_W(e0d, c), M=4)
    num = den = 0.0
    for i in range(4):
        sel = c["q_step"] & (c["rank"] == i)
        rho = _ref_spearman(r[sel], d[sel])
        assert s["per_rank_Q"][i] == pytest.approx(rho)
        num, den = num + sel.sum() * rho, den + sel.sum()
    assert s["rho_Q_rank"] == pytest.approx(num / den)


def test_bootstrap_is_per_document_seeded_and_shared(e0d):
    W1 = e0d.bootstrap_doc_weights(10, 5, 20260927)
    W2 = e0d.bootstrap_doc_weights(10, 5, 20260927)
    assert np.array_equal(W1, W2) and W1.shape == (5, 10)
    assert (W1.sum(1) == 10).all()
    assert not np.array_equal(W1, e0d.bootstrap_doc_weights(10, 5, 1))
    c = _cells(score="bad")
    keep = np.ones(len(c["t"]), dtype=bool)
    W = _W(e0d, c)
    a = e0d.score_stats(c, c["r_gated"], c["d_resample"], keep=keep, W=W, M=4)
    b = e0d.score_stats(c, c["r_gated"], c["d_resample"], keep=keep, W=W, M=4)
    assert a["rho_Q_ci"] == b["rho_Q_ci"]
    assert a["rho_Q_ci"][0] <= a["rho_Q"] <= a["rho_Q_ci"][1]


def test_a_nan_replicate_makes_the_interval_nan(e0d):
    lo, hi = e0d.percentile_ci(np.array([0.1, np.nan, 0.3]))
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# C10 positive control (true_demand as the score) and RANK_CEILING
# --------------------------------------------------------------------------- #


def test_c10_ceiling_scores_true_demand_through_the_identical_code(e0d):
    c = _cells(score="bad")
    an = e0d.analyse_seed(c, M=4, n_docs=40, rho_star=0.5, n_boot=40)
    Q = c["q_step"]
    for k in ("resample", "zero"):
        pc = an["populations"]["all"]["stats"]["pc"][k]
        d = c[f"d_{k}"]
        assert pc["rho_Q"] == pytest.approx(_ref_spearman(c["pc"][Q], d[Q]))
    # the same bootstrap: the PC and r_i intervals come from the same W
    got = an["populations"]["all"]["stats"]
    assert got["pc"]["resample"]["n_boot"] == got["gated"]["resample"]["n_boot"] == 40


def test_c10_ceiling_label_comes_first(e0d):
    base = dict(loo_flat=True, r_flat=True, rho_Q_ci=(0.9, 0.95),
                rho_Q_rank_ci=(0.9, 0.95))  # fmt: skip
    assert e0d.seed_label({**base, "pc_rho_Q_ci": (0.1, 0.4)}, rho_star=0.5) == "CEILING"
    assert e0d.seed_label({**base, "pc_rho_Q_ci": (0.6, 0.9)}, rho_star=0.5) == (
        "DEGENERATE"
    )
    # a PC with no defined rho cannot reach rho*: CEILING, never DISAGREE
    nan = (float("nan"), float("nan"))
    lab = e0d.seed_label({**base, "loo_flat": False, "pc_rho_Q_ci": nan,
                          "rho_Q_ci": (0.0, 0.1)}, rho_star=0.5)  # fmt: skip
    assert lab == "CEILING"


def test_c10_ceiling_counts_as_unresolved_in_section_8(e0d):
    assert e0d.classify(["CEILING", "DISAGREE", "DISAGREE"]) == ("MIXED_UNRESOLVED", 2)
    assert e0d.classify(["CEILING"] * 3) == ("MIXED_UNRESOLVED", 2)
    assert e0d.classify(["CEILING", "DEGENERATE", "LOO_UNINFORMATIVE"]) == (
        "DEGENERATE_UNINFORMATIVE",
        2,
    )


def _an(label, *, all_label=None, ungated=None, zero=None, loo_flat=False,
        r_flat=False, pc_rank_ceiling=False):  # fmt: skip
    def pop(lab):
        return {
            "labels": {
                "gated_resample": lab,
                "ungated_resample": ungated or lab,
                "gated_zero": zero or lab,
            },
            "sensitivity": {"0.3": lab, "0.7": lab},
            "flags": {
                "loo_flat": loo_flat,
                "r_flat": r_flat,
                "pc_rank_ceiling": pc_rank_ceiling,
            },
        }

    return {"populations": {"bos_excluded": pop(label), "all": pop(all_label or label)}}


def test_rank_ceiling_flags_agree_via_rank_and_changes_nothing(e0d):
    ans = {
        0: _an("AGREE_VIA_RANK", pc_rank_ceiling=True),
        1: _an("AGREE_VIA_RANK", pc_rank_ceiling=False),
        2: _an("DISAGREE", pc_rank_ceiling=True),
    }
    out = e0d.classify_run(ans)
    assert out["rank_ceiling_seeds"] == [0]
    # the flag moves neither the class nor its exit
    assert out["class"] == "RECENCY_ONLY"
    assert out["exit"] == e0d.CLASS_EXIT["RECENCY_ONLY"]


def test_a_degenerate_run_hidden_by_ceiling_is_reported(e0d):
    ans = {
        0: _an("CEILING", loo_flat=True, r_flat=True),
        1: _an("CEILING", loo_flat=True),
        2: _an("AGREE"),
    }
    out = e0d.classify_run(ans)
    assert out["class"] == "MIXED_UNRESOLVED"
    assert out["degenerate_under_ceiling"] is True
    assert out["seed_flatness"]["0"] == {"loo_flat": True, "r_flat": True}
    assert (
        e0d.classify_run({s: _an("AGREE") for s in range(3)})["degenerate_under_ceiling"]
        is False
    )


# --------------------------------------------------------------------------- #
# §5 flatness (A1.1): A-cell median of Delta_resample < 1e-2, R_FLAT unchanged
# --------------------------------------------------------------------------- #


def test_loo_flatness_is_the_a_cell_median_at_1e_2(e0d):
    c = _cells(a_delta=0.02, noise=1e-4)
    A = c["q_step"] & c["a_cell"]
    out = e0d.loo_flat_a(c["d_resample"], A)
    assert out["median_a"] == pytest.approx(np.median(c["d_resample"][A]))
    assert out["flat"] is False
    # the withdrawn q90 rule would have called this flat: the scale matters
    assert e0d.loo_flat_q90(c["d_resample"])["flat"] is True
    c9 = _cells(a_delta=0.009, noise=1e-4)
    assert e0d.loo_flat_a(c9["d_resample"], c9["q_step"] & c9["a_cell"])["flat"] is True
    with pytest.raises(e0d.MeasurementUndefined):
        e0d.loo_flat_a(c["d_resample"], np.zeros(len(A), dtype=bool))


def test_analysis_reads_flatness_on_a_cells(e0d):
    c = _cells(a_delta=0.02, noise=1e-4, score="bad")
    an = e0d.analyse_seed(c, M=4, n_docs=40, rho_star=0.5, n_boot=20)
    pa = an["populations"]["all"]
    assert pa["flags"]["loo_flat"] is False
    assert pa["flatness"]["q90_abs_delta_resample"] < 1e-3
    assert pa["flatness"]["loo_flat_resample"]["n_a"] == int(
        (c["q_step"] & c["a_cell"]).sum()
    )


def test_r_flat_is_the_median_within_step_cv(e0d):
    flat = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.02, 0.98, 1.0]])
    assert e0d.r_flat(flat)["flat"] is True
    steep = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    assert e0d.r_flat(steep)["flat"] is False
    # an excluded slot (NaN) is left out of its step, not the whole step
    holed = np.array([[1.0, 1.0, 1.0, np.nan], [1.0, 1.0, 1.0, np.nan]])
    assert e0d.r_flat(holed)["median_cv"] == 0.0


# --------------------------------------------------------------------------- #
# §7 labels and §8 classification (amended)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rho_q, rho_rank, lf, rf, want",
    [
        ((0.6, 0.8), (0.6, 0.8), True, True, "DEGENERATE"),
        ((0.6, 0.8), (0.6, 0.8), True, False, "LOO_UNINFORMATIVE"),
        ((-0.3, -0.1), (0.0, 0.1), False, False, "INVERTED"),
        ((0.1, 0.4), (0.0, 0.1), False, False, "DISAGREE"),
        ((0.6, 0.8), (0.55, 0.7), False, True, "AGREE"),
        ((0.6, 0.8), (0.1, 0.4), False, False, "AGREE_VIA_RANK"),
        ((0.4, 0.6), (0.4, 0.6), False, False, "UNRESOLVED"),
        ((0.6, 0.8), (0.4, 0.6), False, False, "UNRESOLVED"),
    ],
)
def test_seed_labels_first_match_wins(e0d, rho_q, rho_rank, lf, rf, want):
    inp = dict(pc_rho_Q_ci=(0.8, 0.9), loo_flat=lf, r_flat=rf, rho_Q_ci=rho_q,
               rho_Q_rank_ci=rho_rank)  # fmt: skip
    assert e0d.seed_label(inp, rho_star=0.5) == want


@pytest.mark.parametrize(
    "labels, want",
    [
        (["AGREE"] * 3, ("AGREE", 0)),
        (["INVERTED"] * 3, ("CONFOUND_INVERTED", 1)),
        (["DISAGREE", "INVERTED", "DISAGREE"], ("CONFOUND", 1)),
        (["DEGENERATE", "LOO_UNINFORMATIVE", "AGREE"], ("DEGENERATE_UNINFORMATIVE", 2)),
        (["DEGENERATE", "AGREE", "AGREE"], ("MIXED_UNRESOLVED", 2)),
        (["AGREE", "AGREE", "UNRESOLVED"], ("MIXED_UNRESOLVED", 2)),
    ],
)
def test_classification_rows(e0d, labels, want):
    assert e0d.classify(labels) == want


def test_recency_only_exits_2_never_1(e0d):
    """A1.2: AGREE_VIA_RANK is agreement that fails within age strata -- not the
    §3.2.1 kill. Exit 1 is reachable only from CONFOUND and CONFOUND_INVERTED."""
    assert e0d.classify(["AGREE_VIA_RANK", "DISAGREE", "INVERTED"]) == (
        "RECENCY_ONLY",
        2,
    )
    assert e0d.classify(["AGREE_VIA_RANK"] * 3) == ("RECENCY_ONLY", 2)
    assert e0d.CLASS_EXIT["RECENCY_ONLY"] == 2
    assert {k for k, v in e0d.CLASS_EXIT.items() if v == 1} == {
        "CONFOUND",
        "CONFOUND_INVERTED",
    }


def test_gate_sensitive_and_zero_are_reported_never_the_exit(e0d):
    ans = {s: _an("DISAGREE", ungated="AGREE", zero="AGREE") for s in range(3)}
    out = e0d.classify_run(ans)
    assert (out["class"], out["exit"]) == ("CONFOUND", 1)
    assert out["class_ungated"] == "AGREE" and out["gate_sensitive"] is True
    assert out["class_zero"] == "AGREE"
    assert set(out["sensitivity"]) == {"0.3", "0.7"}


# --------------------------------------------------------------------------- #
# A1.3 the bos-copy exclusion recomputation
# --------------------------------------------------------------------------- #


def test_bos_exclusion_drops_rank_m_minus_1_and_gap_1_query_steps(e0d):
    c = _cells()
    keep = e0d.bos_keep(c, M=4)
    want = (c["rank"] != 3) & ~(c["q_step"] & (c["gap"] == 1))
    assert np.array_equal(keep, want)
    assert (c["q_step"] & (c["gap"] == 1)).any() and (c["rank"] == 3).any()


def test_bos_exclusion_recomputes_every_decision_statistic(e0d):
    c = _cells(score="bad")
    an = e0d.analyse_seed(c, M=4, n_docs=40, rho_star=0.5, n_boot=20)
    keep = e0d.bos_keep(c, M=4)
    pe = an["populations"]["bos_excluded"]
    assert pe["n_cells"] == int(keep.sum())
    Q = c["q_step"] & keep
    d = c["d_resample"]
    assert pe["stats"]["gated"]["resample"]["rho_Q"] == pytest.approx(
        _ref_spearman(c["r_gated"][Q], d[Q])
    )
    assert pe["stats"]["pc"]["resample"]["rho_Q"] == pytest.approx(
        _ref_spearman(c["pc"][Q], d[Q])
    )
    A = Q & c["a_cell"]
    assert pe["flatness"]["loo_flat_resample"]["median_a"] == pytest.approx(
        np.median(d[A])
    )
    assert an["populations"]["all"]["n_cells"] == len(d)


def test_exit_follows_the_bos_excluded_class(e0d):
    out = e0d.classify_run({s: _an("DISAGREE", all_label="AGREE") for s in range(3)})
    assert (out["class"], out["exit"]) == ("CONFOUND", 1)
    assert out["class_all_cells"] == "AGREE" and out["bos_sensitive"] is True
    out = e0d.classify_run({s: _an("AGREE", all_label="DISAGREE") for s in range(3)})
    assert (out["class"], out["exit"]) == ("AGREE", 0)
    same = e0d.classify_run({s: _an("AGREE") for s in range(3)})
    assert same["bos_sensitive"] is False


# --------------------------------------------------------------------------- #
# A1.5 B5 re-check, A1.6 the separate r_i capture forward, A1.7 the T0 record
# --------------------------------------------------------------------------- #


def _stream_manifest(root, name, **kw):
    d = root / name
    d.mkdir(parents=True)
    body = {"stream": {"offset": 4160, "stride": 16}, **kw}
    (d / "manifest.json").write_text(json.dumps(body))


def test_c2_rechecks_a_b5_stream_against_its_recorded_last_step(e0d, tmp_path):
    _stream_manifest(tmp_path, "b5", resume_step=3000, end_step=16123)
    got = e0d.stream_ranges(tmp_path)
    assert {"name": "b5.stream(resume_step..end_step)", "range": (52160, 262128),
            "seeds": (0, 1, 2)} in got  # fmt: skip
    e0d.check_disjoint(e0d.D_E0D, seeds=e0d.SEEDS, used=got)
    (tmp_path / "b5" / "manifest.json").unlink()
    (tmp_path / "b5").rmdir()
    _stream_manifest(tmp_path, "b5", resume_step=3000, end_step=16124)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.stream_ranges(tmp_path)
    assert ei.value.control == "C2"


def test_a_stream_manifest_without_a_range_is_a_refusal(e0d, tmp_path):
    _stream_manifest(tmp_path, "b5", resume_step=3000)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.stream_ranges(tmp_path)
    assert ei.value.control == "C2"


def test_c2_reads_every_committed_stream_manifest(e0d):
    names = {u["name"] for u in e0d.used_ranges(ROOT / "runs")}
    assert "fresh-escape.stream(resume_step..end_step)" in names


def test_r_i_comes_from_its_own_capture_forward(e0d, world, monkeypatch):
    """A1.6: `loo_delta_loss` does not capture; r_i is read from live_pass's
    own FIFO forward with capture=True, once per step."""
    model, docs, ids, mask = world
    calls = []
    real = e0d.cross_capture

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(e0d, "cross_capture", counting)
    e0d.measure_batch(model, docs, ids, mask, seed=0)
    assert len(calls) == ids.shape[1]


def test_c1_reads_the_t0_manifest_from_the_t0_record(e0d, tmp_path):
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.t0_manifest(tmp_path)
    assert ei.value.control == "C1"
    rec = tmp_path / "t0-substrate"
    rec.mkdir()
    man = tmp_path / "MANIFEST.sha256"
    (rec / "manifest.json").write_text(
        json.dumps({"substrate_manifest": str(man), "repo_root": "/y"})
    )
    with pytest.raises(e0d.ControlFailed) as ei:  # names a manifest that is absent
        e0d.t0_manifest(tmp_path)
    assert ei.value.control == "C1"
    man.write_text("")
    got = e0d.t0_manifest(tmp_path)
    assert got["manifest"] == man and got["cwd"] == Path("/y")
    assert "rsr-substrate/2026-09-27" not in RUN.read_text()


# --------------------------------------------------------------------------- #
# main(): A1.4 crash/kill separation, the exit mapping, claims.json
# --------------------------------------------------------------------------- #


@pytest.fixture
def cleared(e0d, monkeypatch, fake_ledger, tmp_path):
    """main() with C8/C1/C2 passing and measure_seed replaced by hand-built cells.
    Nothing here generates a document of any range."""
    gen = []
    monkeypatch.setattr(e0d, "_generate_document", lambda i, cfg: gen.append(i))
    monkeypatch.setattr(
        e0d, "check_authority", lambda *a, **k: {"rho_star": 0.5, "rulings": {}}
    )
    monkeypatch.setattr(e0d, "check_substrate", lambda root: {})
    monkeypatch.setattr(
        e0d, "t0_manifest", lambda runs: {"manifest": tmp_path / "M", "cwd": tmp_path}
    )
    monkeypatch.setattr(e0d, "used_ranges", lambda runs: [])
    monkeypatch.setattr(e0d, "substrate_recheck", lambda m, cwd: {"rc": 0})
    monkeypatch.setattr(e0d, "BOOT_N", 30)
    state = {"score": "bad", "raise": None}

    def fake_measure(seed, **kw):
        if state["raise"] is not None:
            raise state["raise"]
        return {"seed": seed, "M": 4, "n_docs": 40, "S": 16,
                "cells": _cells(score=state["score"], seed=seed)}  # fmt: skip

    monkeypatch.setattr(e0d, "measure_seed", fake_measure)
    state["gen"] = gen
    return state


def _rows(fake_ledger):
    return json.loads(fake_ledger.last.path.read_text())["rows"]


def _row(fake_ledger, key):
    got = [r["value"] for r in _rows(fake_ledger) if r["key"] == key]
    return got[-1] if got else None


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("boom"),
        ZeroDivisionError(),
        KeyboardInterrupt(),
        SystemExit(1),
        SystemExit(0),
    ],
    ids=[
        "ValueError",
        "ZeroDivisionError",
        "KeyboardInterrupt",
        "SystemExit1",
        "SystemExit0",
    ],
)
def test_a_crash_exits_3_never_1(e0d, cleared, fake_ledger, exc):
    """A1.4: any exception, SystemExit and KeyboardInterrupt included, exits 3
    with the traceback in the ledger; no class is written."""
    cleared["raise"] = exc
    assert int(e0d.main([])) == 3
    tb = _row(fake_ledger, "measurement_raised")
    assert tb is not None and "Traceback" in tb["traceback"]
    assert _row(fake_ledger, "e0d.class") is None
    assert fake_ledger.last._status in ("crashed", "did_not_run")
    assert fake_ledger.last.commands[-1] == 3


def test_exit_1_is_emitted_only_after_the_class_is_written(
    e0d, cleared, fake_ledger, monkeypatch
):
    cleared["score"] = "bad"
    assert int(e0d.main([])) == 1
    assert _row(fake_ledger, "e0d.class") == "CONFOUND"
    assert fake_ledger.last._status == "failed"
    # the same run with the class never reaching the written ledger: exit 3
    real_note = _FakeLedger.note

    def drop_class(self, k, v, *, how):
        if k != "e0d.class":
            real_note(self, k, v, how=how)

    monkeypatch.setattr(_FakeLedger, "note", drop_class)
    assert int(e0d.main([])) == 3


def test_the_exit_guard_reads_the_class_back(e0d, tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"rows": [{"key": "e0d.class", "value": "CONFOUND"}]}))
    assert e0d.confirm_exit(p, 1) == 1
    with pytest.raises(RuntimeError):
        e0d.confirm_exit(p, 0)
    p.write_text(json.dumps({"rows": [{"key": "e0d.class", "value": "AGREE"}]}))
    assert e0d.confirm_exit(p, 0) == 0
    with pytest.raises(RuntimeError):
        e0d.confirm_exit(p, 1)
    p.write_text(json.dumps({"rows": []}))
    with pytest.raises(RuntimeError):
        e0d.confirm_exit(p, 2)


def test_main_runs_the_amended_analysis_when_cleared(e0d, cleared, fake_ledger):
    cleared["score"] = "good"
    assert int(e0d.main([])) == 0
    assert _row(fake_ledger, "e0d.class") == "AGREE"
    assert _row(fake_ledger, "e0d.labels") == ["AGREE"] * 3
    for k in ("e0d.class_all_cells", "e0d.bos_sensitive", "e0d.gate_sensitive",
              "e0d.class_ungated", "e0d.class_zero", "e0d.rank_ceiling_seeds",
              "e0d.degenerate_under_ceiling", "c1_substrate_end"):  # fmt: skip
        assert _row(fake_ledger, k) is not None, k
    assert cleared["gen"] == []


def test_claims_json_is_written_only_by_a_real_run(e0d, cleared, fake_ledger):
    cleared["raise"] = ValueError("boom")
    assert int(e0d.main([])) == 3
    claims = fake_ledger.last.path.parent / "claims.json"
    assert not claims.exists()
    cleared["raise"] = None
    assert int(e0d.main([])) == 1
    got = json.loads(claims.read_text())
    assert isinstance(got, list) and len(got) >= 4
    for c in got:
        assert set(c) == {"claim", "command", "expected"}
        assert all(isinstance(c[k], str) and c[k] for k in c)
    assert any(c["expected"] == "CONFOUND" for c in got)
    assert any(c["expected"] == "1" for c in got)
    # the claims that read ledger rows run as written, from the repo root
    import subprocess

    for c in got:
        if "['rows']" in c["command"]:
            proc = subprocess.run(
                c["command"], shell=True, cwd=ROOT, capture_output=True, text=True
            )
            assert proc.returncode == 0, proc.stderr
            assert proc.stdout.strip() == c["expected"], c["claim"]


def test_a_refused_run_writes_no_claims(e0d, monkeypatch, fake_ledger):
    assert int(e0d.main([])) == 3  # C8, today
    assert not (fake_ledger.last.path.parent / "claims.json").exists()


def test_a_raise_before_the_ledger_exists_still_exits_3(e0d, monkeypatch, fake_ledger):
    """A1.4: the ledger's construction and the manifest write sit outside the
    measurement's guard; a raise there (or a bad argument) is still 3, never 1."""

    def boom(self, cfg):
        raise OSError("manifest write refused")

    assert int(e0d.main(["--no-such-flag"])) == 3
    monkeypatch.setattr(_FakeLedger, "manifest", boom)
    assert int(e0d.main([])) == 3
