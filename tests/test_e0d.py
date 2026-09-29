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
import math
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
# C8 authority (PREREG A2.13): retrieval-shown, sprint0-gate, and BOTH
# e0d-statistic rulings (the first and its amendment); AUROC_STAR read from them.
# --------------------------------------------------------------------------- #

RULINGS = ROOT / "docs" / "owner" / "rulings"
FIRST_RULING = RULINGS / "R-2026-09-27-e0d-statistic.md"
#: the amended ruling's text. Byte-identical to docs/owner/rulings/ at 4252408
#: (night/2026-09-27); vendored so T14 runs on the real text on any branch.
AMENDED_TEXT = (
    ROOT / "experiments" / "e0d" / "amendment-2" / "owner-drafts"
    / "R-2026-09-27-e0d-statistic-amended.md"
)  # fmt: skip
FIRST = "R-2026-09-27-e0d-statistic.md"
AMENDED = "R-2026-09-27-e0d-statistic-amended.md"
SHOWN = "R-2026-09-27-retrieval-shown.md"
GATE = "R-2026-09-27-sprint0-gate-green.md"


def _rulings(tmp_path, bodies: dict[str, str]):
    d = tmp_path / "rulings"
    d.mkdir(parents=True)
    for n, body in bodies.items():
        (d / n).write_text(body)
    return d


def _real_bodies(**over):
    """Copies of the two real ruling texts, plus stand-ins for the other two."""
    b = {
        SHOWN: "ruling\n",
        GATE: "ruling\n",
        FIRST: FIRST_RULING.read_text(),
        AMENDED: AMENDED_TEXT.read_text(),
    }
    b.update(over)
    return {k: v for k, v in b.items() if v is not None}


ALL4 = (SHOWN, GATE, FIRST, AMENDED)


@pytest.mark.parametrize("missing", ALL4)
def test_authority_refuses_without_each_ruling(e0d, tmp_path, missing):
    """T14: each of the four rulings is required; the first ruling alone and the
    amended ruling alone each exit 3 (A2.13 "Either alone exits 3")."""
    d = _rulings(tmp_path, _real_bodies(**{missing: None}))
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_authority(d, require_committed=False)
    assert ei.value.control == "C8"


def test_authority_passes_on_the_real_ruling_texts(e0d, tmp_path):
    """T14: the key sits in a table cell (first ruling) and in a list item
    (amended); the unanchored A2.13 form reads 0.85 from both."""
    assert "| **`AUROC_STAR: 0.85`**" in FIRST_RULING.read_text()
    assert "2. **`AUROC_STAR: 0.85` now applies" in AMENDED_TEXT.read_text()
    d = _rulings(tmp_path, _real_bodies())
    out = e0d.check_authority(d, require_committed=False)
    assert out["auroc_star"] == 0.85
    assert set(out["rulings"]) >= {"retrieval_shown", "sprint0_gate", "e0d_statistic",
                                   "e0d_statistic_amended"}  # fmt: skip
    # the line-anchored _RHO_LINE-style form matches neither real text (D.2 item 5)
    anchored = re.compile(r"^\s*[`*]*AUROC_STAR[`*]*\s*[:=]\s*([0-9.]+)", re.M)
    assert not anchored.findall(FIRST_RULING.read_text())
    assert not anchored.findall(AMENDED_TEXT.read_text())


def test_authority_harm_ratio_star_is_not_required(e0d, tmp_path):
    """T14: HARM_RATIO_STAR is not required and not read (A2.13)."""
    first = FIRST_RULING.read_text()
    assert "HARM_RATIO_STAR" in first  # its presence is not an error
    e0d.check_authority(_rulings(tmp_path, _real_bodies()), require_committed=False)
    no_harm = "".join(ln for ln in first.splitlines(True) if "HARM_RATIO_STAR" not in ln)
    d = _rulings(tmp_path / "b", _real_bodies(**{FIRST: no_harm}))
    assert e0d.check_authority(d, require_committed=False)["auroc_star"] == 0.85


def test_a_rho_ruling_that_states_no_value_is_refused(e0d, tmp_path):
    """T14: no AUROC_STAR anywhere, two distinct values, or a value outside (0.5, 1)
    exits 3; nothing falls back to a typed value."""
    strip = "".join(
        ln for ln in FIRST_RULING.read_text().splitlines(True) if "AUROC_STAR" not in ln
    )
    strip_a = "".join(
        ln for ln in AMENDED_TEXT.read_text().splitlines(True) if "AUROC_STAR" not in ln
    )
    cases = {
        "none": {FIRST: strip, AMENDED: strip_a},
        "two": {AMENDED: AMENDED_TEXT.read_text() + "\nAUROC_STAR: 0.9\n"},
        "half": {FIRST: strip + "AUROC_STAR: 0.5\n", AMENDED: strip_a},
        "one": {FIRST: strip + "AUROC_STAR = 1\n", AMENDED: strip_a},
        "low": {FIRST: strip + "AUROC_STAR: 0.3\n", AMENDED: strip_a},
    }
    for name, over in cases.items():
        d = _rulings(tmp_path / name, _real_bodies(**over))
        with pytest.raises(e0d.ControlFailed) as ei:
            e0d.check_authority(d, require_committed=False)
        assert ei.value.control == "C8", name
    # one value stated in only one file is enough, if both files exist
    d = _rulings(tmp_path / "single", _real_bodies(**{AMENDED: strip_a}))
    assert e0d.check_authority(d, require_committed=False)["auroc_star"] == 0.85


def test_a_rho_star_ruling_neither_satisfies_nor_is_read(e0d, tmp_path):
    """T14: `R-*-rho-star*` is neither required nor read. A rho-star file standing
    in for the first ruling does not satisfy C8, even if it states AUROC_STAR."""
    rho = "R-2026-10-01-rho-star.md"
    d = _rulings(tmp_path, _real_bodies(**{FIRST: None, rho: "AUROC_STAR: 0.85\n"}))
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_authority(d, require_committed=False)
    assert ei.value.control == "C8"
    # beside the two real rulings it changes nothing and is not read
    d = _rulings(tmp_path / "b", _real_bodies(**{rho: "AUROC_STAR: 0.6\n"}))
    assert e0d.check_authority(d, require_committed=False)["auroc_star"] == 0.85


def _git_repo(path, files):
    import subprocess

    g = "/opt/homebrew/bin/git"
    d = path / "rulings"
    d.mkdir(parents=True)
    subprocess.run([g, "init", "-q"], cwd=d, check=True)
    for n, body in files.items():
        (d / n).write_text(body)
    env_c = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([g, "add", "."], cwd=d, check=True)
    subprocess.run([g, *env_c, "commit", "-qm", "rulings"], cwd=d, check=True)
    return d


def test_c8_committed_rulings_in_a_temp_repo(e0d, tmp_path):
    """Step 4's fixture: with the four rulings committed in a git repo, C8 passes
    and reads 0.85; the first ruling alone, the amended ruling alone, and an
    uncommitted edit each exit 3."""
    d = _git_repo(tmp_path / "both", _real_bodies())
    assert e0d.check_authority(d)["auroc_star"] == 0.85
    (d / AMENDED).write_text(AMENDED_TEXT.read_text() + "\nedited\n")
    with pytest.raises(e0d.ControlFailed):
        e0d.check_authority(d)
    for drop in (AMENDED, FIRST):
        d = _git_repo(tmp_path / drop, _real_bodies(**{drop: None}))
        with pytest.raises(e0d.ControlFailed) as ei:
            e0d.check_authority(d)
        assert ei.value.control == "C8"


def test_the_real_rulings_dir_authorises_e0d_iff_both_statistic_rulings(e0d):
    """A2.13 on this tree: C8 passes iff docs/owner/rulings holds, committed, both
    the first e0d-statistic ruling and an `-amended` one. At run/e0d before the
    amended ruling (4252408) is merged, that is exit 3."""
    import fnmatch

    names = [p.name for p in RULINGS.iterdir()]
    have_amended = any(fnmatch.fnmatch(n, "R-*-e0d-statistic-amended*") for n in names)
    if have_amended:
        assert e0d.check_authority(RULINGS)["auroc_star"] == 0.85
    else:
        with pytest.raises(e0d.ControlFailed) as ei:
            e0d.check_authority(RULINGS)
        assert ei.value.control == "C8"
        assert "e0d-statistic-amended" in str(ei.value)


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
        e0d, "check_authority", lambda *a, **k: {"auroc_star": 0.85, "rulings": {}}
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
    an = e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=0.85, n_boot=40)
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


COMBOS = ("gated_resample", "ungated_resample", "gated_zero", "ungated_zero")


def _an(label, *, all_label=None, ungated=None, zero=None, loo_flat=False,
        r_flat=False, pc_rank_ceiling=False, label_a1=None, row5b=False,
        raw_label=None, tau_labels=None):  # fmt: skip
    """A per-seed analysis as `classify_run` reads it (PREREG A2.8): A2 labels per
    grid cell, `label_A1`, the 5b boolean, the raw-substituted label and the tau
    sensitivity labels, per population."""

    def pop(lab):
        labs = {
            "gated_resample": lab,
            "ungated_resample": ungated or lab,
            "gated_zero": zero or lab,
            "ungated_zero": zero or lab,
        }
        a1 = label_a1 or lab
        return {
            "labels": labs,
            "labels_A1": {"gated_resample": a1, "ungated_resample": a1, "gated_zero": a1},
            "sensitivity_A1": {"0.3": a1, "0.7": a1},
            "row5b": dict.fromkeys(COMBOS, row5b),
            "labels_raw_substituted": {c: raw_label or labs[c] for c in COMBOS},
            "tau_sensitivity": {
                q: dict.fromkeys(COMBOS, (tau_labels or {}).get(q, lab))
                for q in ("0.99", "0.999")
            },
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
    # A2.4: one critical A-cell, so the gated statistic has a value (else exit 3);
    # the A-cell median stays 0.02 and q90 stays < 1e-3
    j = int(np.flatnonzero(c["q_step"] & c["a_cell"] & (c["rank"] == 1))[0])
    c["d_resample"][j] = c["d_zero"][j] = 1.0
    an = e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=0.85, n_boot=20)
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
    an = e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=0.85, n_boot=20)
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
        e0d, "check_authority", lambda *a, **k: {"auroc_star": 0.85, "rulings": {}}
    )
    monkeypatch.setattr(e0d, "check_substrate", lambda root: {})
    monkeypatch.setattr(
        e0d, "t0_manifest", lambda runs: {"manifest": tmp_path / "M", "cwd": tmp_path}
    )
    monkeypatch.setattr(e0d, "used_ranges", lambda runs: [])
    monkeypatch.setattr(e0d, "substrate_recheck", lambda m, cwd: {"rc": 0})
    monkeypatch.setattr(e0d, "BOOT_N", 30)
    state = {"score": "bad", "raise": None, "post": None}

    def fake_measure(seed, **kw):
        if state["raise"] is not None:
            raise state["raise"]
        c = _cells(score=state["score"], seed=seed)
        if state.get("post") is not None:
            state["post"](c)
        return {"seed": seed, "M": 4, "n_docs": 40, "S": 16, "cells": c}

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


# =========================================================================== #
# PREREG Amendment 2 (8c63ecd): AUROC_strat,pct (the single gate), raw
# AUROC_strat (read only by row 5b), per-seed tau, the A2.8 table and §8 with
# row 4b, C8/C11/C12, the bootstrap's undefined replicates, and Part B's
# calibration fixture (`experiments/e0d/calibration.py`). Test ids are Part B's
# (`experiments/e0d/amendment-2/E0D-AMENDMENT-2-FINAL-v3.md`).
#
# 🔴 No fixture here reads or generates a document of D_E0d. The calibration
# fixture is numbers only. The provenance tests read the vendored set-E cells
# ([64, 128), already inspected; PREREG A2.2).
# =========================================================================== #

A2 = PREREG.read_text().split("## Amendment 2", 1)[1]
A_STAR = 0.85  # the ruled value, used only as a test input; the runner reads it (C8)
CALIB = ROOT / "experiments" / "e0d" / "calibration.py"
VEND = ROOT / "experiments" / "e0d" / "amendment-2" / "reviews"


@pytest.fixture(scope="module")
def cal():
    spec = importlib.util.spec_from_file_location("e0d_calibration", CALIB)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e0d_calibration"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fam(e0d, c, r, *, tau=None, keep=None, n_boot=0, seed=11, delta=None, M=16, **kw):
    """`auroc_family` on the A1.3 population (default), with a small bootstrap."""
    keep = e0d.bos_keep(c, M) if keep is None else keep
    n_docs = int(np.max(c["doc"])) + 1
    W = e0d.bootstrap_doc_weights(n_docs, n_boot, seed)
    d = c["d_resample"] if delta is None else delta
    t = c.get("tau", e0d.TAU_RESAMPLE[0]) if tau is None else tau
    return e0d.auroc_family(c, r, d, tau=t, keep=keep, W=W, M=M, **kw)


def _ref_strat_auroc(pct, y, age):
    """Brute-force A2.4 steps 3-4 over explicit positive-negative pairs."""
    num = den = 0.0
    for a in np.unique(age):
        m = age == a
        p, n = pct[m & y], pct[m & ~y]
        if len(p) == 0 or len(n) == 0:
            continue
        au = ((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])).mean()
        num, den = num + len(p) * au, den + len(p)
    return num / den


def _grid_cells(D, R, *, q, gap, docs, ts=None, M=16, pc=None):
    """Full-memory cells from `[n_steps, M]` Delta and r grids by write-order rank
    (rank i holds sentence t - (M - i): one slot per age, rank = M - age)."""
    D, R = np.asarray(D, float), np.asarray(R, float)
    n = D.shape[0]
    ts = np.arange(M, M + n) if ts is None else np.asarray(ts)
    rows = {k: [] for k in ("doc", "t", "rank", "sentence", "q_step", "a_cell", "gap",
                            "d_resample", "r")}  # fmt: skip
    for j in range(n):
        for i in range(M):
            age = M - i
            rows["doc"].append(docs[j])
            rows["t"].append(ts[j])
            rows["rank"].append(i)
            rows["sentence"].append(ts[j] - age)
            rows["q_step"].append(bool(q[j]))
            rows["a_cell"].append(bool(q[j]) and age == gap[j])
            rows["gap"].append(gap[j] if q[j] else -1)
            rows["d_resample"].append(D[j, i])
            rows["r"].append(R[j, i])
    c = {k: np.asarray(v) for k, v in rows.items()}
    c["n_live"] = np.full(len(c["doc"]), M)
    c["full"] = np.ones(len(c["doc"]), dtype=bool)
    c["d_zero"] = c["d_resample"].copy()
    c["pc"] = c["a_cell"].astype(float) if pc is None else pc
    return c


# --------------------------------------------------------------------------- #
# constants, tau (A2.2, C12), and their provenance in the vendored set-E cells
# --------------------------------------------------------------------------- #


def test_amendment_2_constants_are_the_preregs(e0d):
    """Every A2 number the runner holds, read back out of the committed text."""
    for s in (0, 1, 2):
        assert repr(e0d.TAU_RESAMPLE[s]) in A2
        assert repr(e0d.TAU_ZERO[s]) in A2
        for q in ("0.99", "0.999"):
            for k in ("resample", "zero"):
                assert repr(e0d.TAU_SENSITIVITY[q][k][s]) in A2
    assert "q = 0.995 quantile" in A2 and e0d.TAU_Q == 0.995
    assert "`AUROC_strat,pct` CI upper < 0.5" in A2 and e0d.AUROC_INVERTED == 0.5
    assert "exactly one distinct `AUROC_STAR` value, in (0.5, 1)" in A2
    assert tuple(range(1, 17)) == e0d.AGE_BINS
    assert "0, 1, 2, 5b, 3, 4, 5, 6, 7" in A2
    assert e0d.LABEL_ORDER == ("0", "1", "2", "5b", "3", "4", "5", "6", "7")


def test_c12_tau_tables_equal_the_preregs_and_hold_exactly_seeds_0_1_2(e0d, monkeypatch):
    """C12 (A2.13): the runner's TAU tables hold exactly {0, 1, 2} and equal A2.2's
    registered values, read from the PREREG text at run time; else exit 3."""
    out = e0d.check_tau_tables()
    assert out["seeds"] == [0, 1, 2]
    bad = dict(e0d.TAU_RESAMPLE)
    bad[2] = e0d.TAU_RESAMPLE[0]  # seed 0's tau for seed 2
    monkeypatch.setattr(e0d, "TAU_RESAMPLE", bad)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_tau_tables()
    assert ei.value.control == "C12"
    monkeypatch.setattr(e0d, "TAU_RESAMPLE", {0: bad[0], 1: bad[1]})
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_tau_tables()
    assert ei.value.control == "C12"


def test_c12_refuses_a_zero_tau_that_differs(e0d, monkeypatch):
    z = dict(e0d.TAU_ZERO)
    z[1] = z[1] + 1e-12
    monkeypatch.setattr(e0d, "TAU_ZERO", z)
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_tau_tables()
    assert ei.value.control == "C12"


def _set_e(seed):
    """Vendored set-E cells [64, 128) (PREREG A2.2), as `cells_from`-shaped arrays."""
    p = (VEND / "e0d-a2" / "step1_cells.npz" if seed == 0
         else VEND / "e0d-a2-final" / f"tau_cells_seed{seed}.npz")  # fmt: skip
    z = dict(np.load(p))
    return p, {
        "doc": z["doc"], "t": z["t"], "rank": z["rank"], "sentence": z["sent"],
        "q_step": z["q"].astype(bool), "a_cell": z["a"].astype(bool), "gap": z["gap"],
        "full": z["tfull"].astype(bool), "n_live": np.where(z["tfull"], 16, 0),
        "d_resample": z["dres"], "d_zero": z["dzero"],
        "pc": (z["q"] & z["a"]).astype(float),
    }  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tau_is_the_q0995_of_off_a_q_cells_in_the_vendored_set_e_cells(e0d, seed):
    """A2.2 provenance: each cells file has the registered sha256, and tau (both
    knockouts, and the sensitivities) re-derives exactly from it by numpy's default
    quantile. The runner never computes this on D_E0d; it reads the constant."""
    import hashlib

    p, c = _set_e(seed)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha in A2
    Q = c["full"] & c["q_step"]
    A = Q & c["a_cell"]
    for k, table in (("resample", e0d.TAU_RESAMPLE), ("zero", e0d.TAU_ZERO)):
        d = c[f"d_{k}"]
        ad = np.abs(d[Q & ~A & ~np.isnan(d)])
        assert (ad == 0).sum() == 0
        assert float(np.quantile(ad, 0.995)) == table[seed]
        for q in ("0.99", "0.999"):
            assert float(np.quantile(ad, float(q))) == e0d.TAU_SENSITIVITY[q][k][seed]


def test_the_runner_reproduces_the_a24_table_on_set_e(e0d, cal):
    """A2.4's table (m6_out_012.json) through the runner's own code: on real set-E
    labels, the binary perfect proxy is 1.0, age-only and recency are exactly 0.5,
    true_demand is 0.9672 / 0.9731 / 0.9750, and the continuous perfect proxy and
    the pure step-concentration score (m6's own draws) match to 4 decimals."""
    m6 = json.loads((VEND / "e0d-a2-final" / "m6_out_012.json").read_text())
    for seed in (0, 1, 2):
        _, c = _set_e(seed)
        tau = e0d.TAU_RESAMPLE[seed]
        keep = e0d.bos_keep(c, 16)
        pop = e0d.q_population(c, keep, M=16)
        # m6's grid order: steps by doc*49 + t, ranks 0..15; eps is its first draw
        cases = m6[f"seed{seed}"]["cases"]
        ns = pop["n_steps"]
        rng = np.random.default_rng(1000 + seed)
        eps = rng.standard_normal((ns, 16))
        cell_eps = eps[pop["step"], np.asarray(c["rank"])[pop["idx"]]]
        d = np.asarray(c["d_resample"])[pop["idx"]]
        y = ~np.isnan(d) & (d > tau)
        kt = np.bincount(pop["step"], weights=y, minlength=ns)
        has = (kt >= 1)[pop["step"]]
        full_r = np.zeros(len(c["doc"]))

        def run(r_pop, c=c, pop=pop, tau=tau, keep=keep, full_r=full_r):
            full_r = full_r.copy()
            full_r[pop["idx"]] = r_pop
            return e0d.auroc_family(c, full_r, c["d_resample"], tau=tau, keep=keep,
                                    W=np.zeros((0, 64)), M=16)  # fmt: skip

        def sm(x, pop=pop, ns=ns):
            return cal.softmax_steps(x, pop["step"])

        got = {
            "perfect, binary y (tied)": run(y.astype(float)),
            "true_demand (A-cell, C10 control)": run(np.asarray(c["pc"])[pop["idx"]]),
            "recency -age": run(-pop["age"].astype(float)),
            "perfect, softmax(10y+eps)": run(sm(10 * y + cell_eps)),
            "pure step-concentration (flat on k>=1, peaked else)": run(
                sm(np.where(has, 0.0, 50.0) * cell_eps)
            ),
        }
        for name, fam in got.items():
            assert round(fam["pct"], 4) == cases[name]["AUROC_strat_pct"], (seed, name)
            assert round(fam["raw"], 4) == cases[name]["AUROC_strat_raw"], (seed, name)
        assert got["perfect, binary y (tied)"]["pct"] == 1.0
        assert got["recency -age"]["pct"] == 0.5 and got["recency -age"]["raw"] == 0.5
        pos = m6[f"seed{seed}"]["positives"]
        assert got["perfect, binary y (tied)"]["positives"] == pos
        assert got["perfect, binary y (tied)"]["n_q_steps"] == m6[f"seed{seed}"][
            "n_Q_steps_excl"]  # fmt: skip


def test_labels_use_each_seeds_own_tau_and_strict_greater(e0d, monkeypatch):
    """T11 / A2.2: y = 1[Delta > tau_s], strict; a NaN has no label; the labelling
    path computes no quantile (np.quantile raises inside it)."""

    def boom(*a, **k):
        raise AssertionError("a quantile was computed in the labelling path")

    for s in (0, 1, 2):
        tau = e0d.tau_for(s, "resample")
        assert tau == e0d.TAU_RESAMPLE[s]
        assert e0d.tau_for(s, "zero") == e0d.TAU_ZERO[s]
        d = np.array([tau, np.nextafter(tau, 1.0), tau - 0.1, np.nan, 5.0])
        with monkeypatch.context() as m:
            m.setattr(np, "quantile", boom)
            has, y = e0d.critical_labels(d, tau)
        assert has.tolist() == [True, True, True, False, True]
        assert y.tolist() == [False, True, False, False, True]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analyse_seed_labels_with_that_seeds_frozen_tau(e0d, seed):
    """T11: the analysis uses tau_s of the seed it analyses (never seed 0's for
    all, never a quantile of the run's cells), and records it."""
    # A-cell Delta ~ N(0.46, 0.05) straddles tau_0 = 0.416 < tau_1 = 0.432 < tau_2 = 0.488
    c = _cells(score="good", a_delta=0.46, noise=0.05)
    an = e0d.analyse_seed(c, M=4, n_docs=40, seed=seed, auroc_star=A_STAR, n_boot=4)
    cell = an["populations"]["all"]["a2"]["gated_resample"]
    assert cell["r"]["tau"] == e0d.TAU_RESAMPLE[seed]
    assert an["tau"]["resample"] == e0d.TAU_RESAMPLE[seed]
    assert an["tau"]["zero"] == e0d.TAU_ZERO[seed]
    Q = c["q_step"]
    want = {
        s: int((Q & (c["d_resample"] > e0d.TAU_RESAMPLE[s])).sum()) for s in (0, 1, 2)
    }
    assert cell["r"]["positives"] == want[seed]
    assert len(set(want.values())) == 3  # the three taus label differently here


# --------------------------------------------------------------------------- #
# A2.3 / C11 age strata, A1.3 population (T15, T16)
# --------------------------------------------------------------------------- #


def test_c11_one_slot_per_age_and_rank_is_m_minus_age(e0d):
    c = _cells()
    out = e0d.check_ages(c, M=4)
    assert out["ok"] and out["n_full_steps"] > 0
    bad = dict(c)
    bad["sentence"] = c["sentence"].copy()
    j = int(np.flatnonzero(c["full"])[1])
    bad["sentence"][j] = bad["sentence"][j - 1]  # T15: a duplicated age at one step
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_ages(bad, M=4)
    assert ei.value.control == "C11"
    bad2 = dict(c)
    bad2["rank"] = c["rank"][::-1].copy()  # rank != M - age
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.check_ages(bad2, M=4)
    assert ei.value.control == "C11"


def test_analyse_seed_runs_c11(e0d):
    c = _cells()
    c["sentence"] = c["sentence"].copy()
    c["sentence"][1] = c["sentence"][0]
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=A_STAR, n_boot=2)
    assert ei.value.control == "C11"


def test_a13_population_has_n_t_15_no_bin_1_no_gap_1_steps(e0d, cal):
    """T16: under A1.3 every Q-step has n_t = 15, bin {1} is empty and logged as
    excluded, and no gap-1 Q-step is present; all-cells has n_t = 16."""
    c = cal.synthetic_ledger(3, 64)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    assert set(pop["n_t"].tolist()) == {15}
    assert 1 not in set(pop["age"].tolist())
    assert (np.asarray(c["gap"])[pop["idx"]] != 1).all()
    assert ((np.asarray(c["gap"]) == 1) & c["q_step"] & c["full"]).any()
    fam = _fam(e0d, c, cal.perfect_binary(c))
    assert {b["bin"] for b in fam["pct_excluded_bins"]} >= {1}
    assert fam["n_t_values"] == [15]
    allc = e0d.q_population(c, np.ones(len(c["doc"]), dtype=bool), M=16)
    assert set(allc["n_t"].tolist()) == {16}


def test_pct_spans_zero_to_one_at_n_t_15(e0d, cal):
    """A2.4 step 1 under A1.3: pct = (midrank - 1) / (n_t - 1) with n_t = 15 -- the
    top slot of every step reads exactly 1.0 and the bottom 0.0, never 14/15."""
    c = cal.synthetic_ledger(19, 16)
    r = cal.content(c, np.random.default_rng(19), 1.0)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    pct = e0d.step_percentiles(r[pop["idx"]], pop)
    top = np.full(pop["n_steps"], -1.0)
    np.maximum.at(top, pop["step"], pct)
    bot = np.full(pop["n_steps"], 2.0)
    np.minimum.at(bot, pop["step"], pct)
    assert (top == 1.0).all() and (bot == 0.0).all()
    assert set(np.round(pct * 14, 9)) == set(range(15))


def test_the_a13_population_asserts_n_t_is_m_minus_1(e0d, cal):
    """C11 under A1.3: n_t = M - 1 at every Q-step; a Q-step missing a slot is exit 3."""
    c = cal.synthetic_ledger(4, 16)
    j = int(np.flatnonzero(c["full"] & c["q_step"] & (c["rank"] == 3))[0])
    drop = np.ones(len(c["doc"]), dtype=bool)
    drop[j] = False
    c2 = {k: (v[drop] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    with pytest.raises(e0d.ControlFailed) as ei:
        e0d.q_population(c2, e0d.bos_keep(c2, 16), M=16)
    assert ei.value.control == "C11"


# --------------------------------------------------------------------------- #
# A2.4: the within-step percentile, labels, bins, NaN donors (T5, T8, T9, T10)
# --------------------------------------------------------------------------- #


def test_pct_is_the_midrank_over_all_eligible_slots(e0d):
    """A2.4 step 1: pct = (midrank - 1) / (n_t - 1) over the step's eligible slots;
    tied slots share the midrank (T8)."""
    R = np.array([[5, 1, 3, 3, 9, 0, 2, 7, 7, 7, 4, 6, 8, 10, 11, 12.0]])
    c = _grid_cells(np.zeros((1, 16)), R, q=[True], gap=[5], docs=[0])
    keep = np.ones(16, dtype=bool)
    pop = e0d.q_population(c, keep, M=16)
    pct = e0d.step_percentiles(c["r"][pop["idx"]], pop)
    want = (_ref_rank(R[0]) - 1) / 15
    assert np.array_equal(pct, want[np.asarray(c["rank"])[pop["idx"]]])
    assert pct[np.asarray(c["rank"])[pop["idx"]] == 2][0] == pct[
        np.asarray(c["rank"])[pop["idx"]] == 3][0]  # fmt: skip


def test_ties_count_half_in_the_auroc(e0d):
    """T8: two slots tied in r, one critical: they share a pct, and the pair counts
    1/2. Hand value on one step of 16 (all-cells population, one bin per age)."""
    R = np.arange(16, dtype=float)[None, :].repeat(2, 0)
    R[0, 3] = R[0, 4] = 3.5  # ranks 3 and 4 tied at step 0
    D = np.zeros((2, 16))
    D[0, 3] = 1.0  # critical, tied with the negative at rank 4
    D[1, 4] = 1.0  # step 1: the positive at rank 4, same bin (age 12) as step 0's
    c = _grid_cells(D, R, q=[True, True], gap=[13, 12], docs=[0, 1])
    keep = np.ones(len(c["doc"]), dtype=bool)
    fam = _fam(e0d, c, c["r"], tau=0.5, keep=keep)
    pop = e0d.q_population(c, keep, M=16)
    pct = e0d.step_percentiles(c["r"][pop["idx"]], pop)
    y = np.asarray(c["d_resample"])[pop["idx"]] > 0.5
    assert fam["pct"] == pytest.approx(_ref_strat_auroc(pct, y, pop["age"]), abs=0)
    # bin of age 13 holds the step-0 positive (pct 3.5/15) and step-1's negative (3/15)
    b13 = next(b for b in fam["pct_per_bin"] if b["bin"] == 13)
    assert b13["auroc"] == 1.0 and b13["positives"] == 1 and b13["negatives"] == 1
    # bin of age 12: step-0 negative tied (pct 3.5/15) vs step-1 positive (4/15)
    b12 = next(b for b in fam["pct_per_bin"] if b["bin"] == 12)
    assert b12["auroc"] == 1.0
    # a pure tie inside one bin counts exactly 1/2
    R2 = np.zeros((2, 16))
    D2 = np.zeros((2, 16))
    D2[0, 5] = 1.0
    c2 = _grid_cells(D2, R2, q=[True, True], gap=[11, 11], docs=[0, 1])
    fam2 = _fam(e0d, c2, c2["r"], tau=0.5, keep=np.ones(32, dtype=bool))
    assert next(b for b in fam2["pct_per_bin"] if b["bin"] == 11)["auroc"] == 0.5


def test_a_nan_donor_keeps_its_step_and_every_other_pct(e0d, cal):
    """T9: a NaN Delta on one eligible slot leaves n_t at 15 and every other slot's
    pct unchanged; only that cell leaves the AUROC; the step stays."""
    c = cal.synthetic_ledger(5, 32, nan_mode="independent")
    keep = e0d.bos_keep(c, 16)
    r = cal.content(c, np.random.default_rng(1), 1.5)
    filled = dict(c)
    filled["d_resample"] = np.nan_to_num(c["d_resample"], nan=0.0)
    pop = e0d.q_population(c, keep, M=16)
    pop_f = e0d.q_population(filled, keep, M=16)
    assert np.isnan(np.asarray(c["d_resample"])[pop["idx"]]).any()
    assert np.array_equal(pop["idx"], pop_f["idx"]) and set(pop["n_t"]) == {15}
    assert np.array_equal(e0d.step_percentiles(r[pop["idx"]], pop),
                          e0d.step_percentiles(r[pop_f["idx"]], pop_f))  # fmt: skip
    a, b = _fam(e0d, c, r), _fam(e0d, filled, r)
    assert a["n_q_steps"] == b["n_q_steps"]
    assert a["labelled"] == b["labelled"] - int(
        np.isnan(np.asarray(c["d_resample"])[pop["idx"]]).sum()
    )


def test_a_non_finite_r_on_an_eligible_slot_is_undefined(e0d, cal):
    """T10: a non-finite r on an eligible slot is MeasurementUndefined (exit 3)."""
    c = cal.synthetic_ledger(6, 16)
    r = cal.content(c, np.random.default_rng(0), 1.0)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    for bad in (np.nan, np.inf):
        r2 = r.copy()
        r2[pop["idx"][7]] = bad
        with pytest.raises(e0d.MeasurementUndefined):
            _fam(e0d, c, r2)
    # a non-finite r OFF the population (rank 15, excluded by A1.3) is not read
    r3 = r.copy()
    r3[np.flatnonzero(c["full"] & c["q_step"] & (c["rank"] == 15))[0]] = np.nan
    _fam(e0d, c, r3)


def test_a_non_finite_r_through_main_exits_3(e0d, cleared, fake_ledger):
    cleared["post"] = lambda c: c.__setitem__("r_gated", np.where(
        c["q_step"] & (c["rank"] == 0), np.nan, c["r_gated"]))  # fmt: skip
    assert int(e0d.main([])) == 3
    assert _row(fake_ledger, "e0d.class") is None


def test_bins_without_both_classes_are_excluded_and_logged(e0d):
    """T5: a bin with positives but no negatives, and a bin with neither, are
    excluded and logged with counts; w excludes them; the value is the hand value
    on the rest."""
    rng = np.random.default_rng(3)
    n = 6
    R = rng.random((n, 16))
    D = np.zeros((n, 16))
    D[:, 0] = 1.0  # age 16: positive in every step -> no negative in bin 16
    D[0, 5] = 1.0
    D[1, 7] = 1.0
    c = _grid_cells(D, R, q=[True] * n, gap=[40] * n, docs=list(range(n)))
    keep = np.ones(len(c["doc"]), dtype=bool)
    # a bin with neither class: drop every age-3 cell's label (NaN Delta)
    c["d_resample"] = np.where(c["rank"] == 13, np.nan, c["d_resample"])
    fam = _fam(e0d, c, c["r"], tau=0.5, keep=keep)
    ex = {b["bin"]: b for b in fam["pct_excluded_bins"]}
    assert ex[16]["positives"] == n and ex[16]["negatives"] == 0
    assert ex[3]["positives"] == 0 and ex[3]["negatives"] == 0
    pop = e0d.q_population(c, keep, M=16)
    pct = e0d.step_percentiles(c["r"][pop["idx"]], pop)
    d = np.asarray(c["d_resample"])[pop["idx"]]
    lab = ~np.isnan(d)
    y = d[lab] > 0.5
    ok = ~np.isin(pop["age"][lab], [16, 3])
    want = _ref_strat_auroc(pct[lab][ok], y[ok], pop["age"][lab][ok])
    assert fam["pct"] == pytest.approx(want, abs=1e-15)
    used = {b["bin"]: b for b in fam["pct_per_bin"]}
    assert 16 not in used and 3 not in used
    assert sum(b["positives"] for b in used.values()) == 2


def test_w_a_is_the_positives_in_the_bin(e0d):
    """A2.4 step 4: w_a is the positive count, not the cell count."""
    R = np.tile(np.arange(16.0), (4, 1))
    D = np.zeros((4, 16))
    D[0, 2] = D[1, 2] = D[2, 2] = 1.0  # bin 14: three positives, pct 2/15 (low)
    D[3, 10] = 1.0  # bin 6: one positive, pct 10/15 (high)
    R[3, 10] = 20.0
    c = _grid_cells(D, R, q=[True] * 4, gap=[40] * 4, docs=[0, 1, 2, 3])
    fam = _fam(e0d, c, c["r"], tau=0.5, keep=np.ones(64, dtype=bool))
    b = {x["bin"]: x for x in fam["pct_per_bin"]}
    want = (3 * b[14]["auroc"] + 1 * b[6]["auroc"]) / 4
    assert fam["pct"] == pytest.approx(want, abs=1e-15)
    assert b[14]["auroc"] == 0.5 and b[6]["auroc"] == 1.0


def test_unstratified_pct_is_a_single_bin(e0d, cal):
    c = cal.synthetic_ledger(7, 32)
    r = cal.content(c, np.random.default_rng(2), 1.5)
    fam = _fam(e0d, c, r)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    pct = e0d.step_percentiles(r[pop["idx"]], pop)
    d = np.asarray(c["d_resample"])[pop["idx"]]
    lab = ~np.isnan(d)
    want = _ref_strat_auroc(pct[lab], d[lab] > c["tau"], np.zeros(lab.sum()))
    assert fam["unstrat_pct"] == pytest.approx(want, abs=1e-12)
    assert fam["raw"] == pytest.approx(
        _ref_strat_auroc(r[pop["idx"]][lab], d[lab] > c["tau"], pop["age"][lab]),
        abs=1e-12,
    )


# --------------------------------------------------------------------------- #
# Part B.1 calibration (T1a, T1b, T1c, T2, T3, T4a, T4b, T7, T18)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("noise", ["continuous", "tied"])
@pytest.mark.parametrize("nan_mode", ["matched", "independent"])
def test_t1a_perfect_binary_proxy_is_exactly_1(e0d, cal, noise, nan_mode):
    c = cal.synthetic_ledger(0, 64, noise=noise, nan_mode=nan_mode)
    fam = _fam(e0d, c, cal.perfect_binary(c))
    assert fam["pct"] == 1.0 and fam["raw"] == 1.0


@pytest.mark.parametrize("nan_mode", ["matched", "independent"])
def test_t1b_continuous_perfect_one_per_step_is_exactly_1(e0d, cal, nan_mode):
    c = cal.synthetic_ledger(1, 64, k_profile="one_per_step", nan_mode=nan_mode)
    fam = _fam(e0d, c, cal.perfect_continuous(c, np.random.default_rng(1)))
    assert fam["pct"] == 1.0


def test_t1c_continuous_perfect_matched_is_the_hand_ceiling_below_1(e0d, cal):
    """T1c: on the matched profile the k = 0 steps' pct = 1 negatives tie or beat
    positives, so the continuous perfect proxy sits below 1 at the brute-force
    value (A2.4: about 0.987 on set E)."""
    c = cal.synthetic_ledger(2, 64)
    r = cal.perfect_continuous(c, np.random.default_rng(2))
    fam = _fam(e0d, c, r)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    pct = e0d.step_percentiles(r[pop["idx"]], pop)
    d = np.asarray(c["d_resample"])[pop["idx"]]
    lab = ~np.isnan(d)
    want = _ref_strat_auroc(pct[lab], d[lab] > c["tau"], pop["age"][lab])
    assert fam["pct"] == pytest.approx(want, abs=1e-12)
    assert 0.97 < fam["pct"] < 1.0
    print(f"T1c continuous perfect proxy AUROC_strat,pct = {fam['pct']:.4f}")


def _sd(e0d, c, r, n_boot=200):
    fam = _fam(e0d, c, r, n_boot=n_boot, replicates=True)
    reps = fam["_replicates"]["pct"]
    return fam, float(np.nanstd(reps))


def test_t2_within_step_shuffle_is_null(e0d, cal):
    """T2: |pct - 0.5| < 4 sd and C_ws within 4 sd of 0, sd at the fixture's own n
    (D.2 item 6), 1024 documents."""
    c = cal.synthetic_ledger(8, 1024)
    r = cal.shuffle(c, np.random.default_rng(8))
    fam = _fam(e0d, c, r, n_boot=100, replicates=True)
    sd = float(np.std(fam["_replicates"]["pct"]))
    sd_c = float(np.std(fam["_replicates"]["c_ws"]))
    assert abs(fam["pct"] - 0.5) < 4 * sd, (fam["pct"], sd)
    assert abs(fam["c_ws"]) < 4 * sd_c, (fam["c_ws"], sd_c)
    print(f"T2 shuffle pct = {fam['pct']:.4f} (sd {sd:.4f}); C_ws = {fam['c_ws']:.4f}")


@pytest.mark.parametrize("which", ["age_only", "recency"])
@pytest.mark.parametrize("population", ["a13", "all"])
def test_t3_age_only_scores_are_exactly_half(e0d, cal, which, population):
    """T3: an age-only score gives AUROC_strat,pct == 0.5 and raw == 0.5 exactly,
    C_ws == 0 within 1e-12 and R_H == 1.0 exactly, in both populations."""
    c = cal.synthetic_ledger(9, 64)
    r = getattr(cal, which)(c)
    keep = None if population == "a13" else np.ones(len(c["doc"]), dtype=bool)
    fam = _fam(e0d, c, r, keep=keep)
    assert fam["pct"] == 0.5 and fam["raw"] == 0.5
    assert abs(fam["c_ws"]) < 1e-12
    assert fam["R_H"] == 1.0
    print(
        f"T3 {which} [{population}]: pct {fam['pct']}, raw {fam['raw']}, R_H {fam['R_H']}"
    )


def test_t4a_pct_family_is_invariant_to_per_step_monotone_transforms(e0d, cal):
    """T4a: rescaled c_t r and step temperature leave pct, C_ws, H and the argmins
    unchanged, bit for bit."""
    c = cal.synthetic_ledger(10, 64)
    eps = np.random.default_rng(10).standard_normal(len(c["doc"]))
    base = cal.softmax_steps(2.0 * cal._crit(c) + eps, cal.step_ids(c))
    temp = cal.step_temperature(c, None, 2.0, eps=eps)
    temp_r = cal.step_temperature(c, None, 2.0, reverse=True, eps=eps)
    resc = cal.rescaled(c, np.random.default_rng(11), base)
    pop = e0d.q_population(c, e0d.bos_keep(c, 16), M=16)
    f0 = _fam(e0d, c, base)
    for r in (temp, temp_r, resc):
        f = _fam(e0d, c, r)
        for k in ("pct", "c_ws", "H", "unstrat_pct"):
            assert f[k] == f0[k], k
        assert np.array_equal(e0d.tied_minima(r[pop["idx"]], pop),
                              e0d.tied_minima(base[pop["idx"]], pop))  # fmt: skip


def test_t4b_raw_auroc_moves_under_rescaling(e0d, cal):
    """T4b: raw AUROC_strat is not invariant to per-step rescaling (the pinned
    counter-example), so STEP_SENSITIVE is computable."""
    c = cal.synthetic_ledger(10, 64)
    base = cal.content(c, np.random.default_rng(12), 2.0)
    resc = cal.rescaled(c, np.random.default_rng(13), base)
    a, b = _fam(e0d, c, base), _fam(e0d, c, resc)
    assert a["pct"] == b["pct"] and a["raw"] != b["raw"]


def test_t7_spearman_of_a_binary_proxy_is_capped(e0d, cal):
    """T7 / A2.12: against continuous Delta a perfect binary score's Spearman is at
    most sqrt(3 p (1 - p)) (+ 0.01)."""
    c = cal.synthetic_ledger(14, 256)
    Q = c["full"] & c["q_step"] & ~np.isnan(c["d_resample"])
    y = c["crit"][Q].astype(float)
    p = y.mean()
    rho = e0d.spearman(y, c["d_resample"][Q])
    assert rho <= math.sqrt(3 * p * (1 - p)) + 0.01
    assert rho > 0


def test_t18_pure_step_concentration_is_half_on_pct_not_on_raw(e0d, cal):
    """T18: pure step-level signal (no within-step content) gives pct within 4 sd
    of 0.5, while raw AUROC_strat sits in (0.55, 0.70)."""
    c = cal.synthetic_ledger(15, 256)
    r = cal.step_concentration(c, np.random.default_rng(15))
    fam, sd = _sd(e0d, c, r, n_boot=100)
    assert abs(fam["pct"] - 0.5) < 4 * sd, (fam["pct"], sd)
    assert 0.55 < fam["raw"] < 0.70, fam["raw"]
    print(
        f"T18 step-concentration: pct {fam['pct']:.4f} (sd {sd:.4f}), "
        f"raw {fam['raw']:.4f}"
    )


def test_graded_content_and_the_age_term(e0d, cal):
    """A2.11's shape on the fixture: pct rises with content strength a, and an
    age term u(age) lowers pct while raw does not fall (D.2 item 3)."""
    c = cal.synthetic_ledger(16, 256)
    vals = [_fam(e0d, c, cal.content(c, np.random.default_rng(16), a))["pct"]
            for a in (0.5, 1.0, 1.5, 2.0, 3.0)]  # fmt: skip
    assert vals == sorted(vals) and vals[0] > 0.55 and vals[-1] > 0.95
    plain = _fam(e0d, c, cal.content(c, np.random.default_rng(17), 2.0))
    aged = _fam(e0d, c, cal.age_steered(c, np.random.default_rng(17), 2.0))
    assert aged["pct"] < plain["pct"]


def test_calibration_fixture_expected_values(e0d, cal):
    """The §6A / B.1 calibration values, each printed for the report: the pct
    artefact (pure step-concentration) is about 0.5, age-only is 0.5 exactly, the
    binary perfect proxy is 1.0 exactly, and the continuous perfect proxy is about
    0.987; under both NaN modes."""
    for nan_mode in ("matched", "independent"):
        c = cal.synthetic_ledger(21, 1024, nan_mode=nan_mode)
        rng = np.random.default_rng(21)
        got = {
            "pct artefact (step-concentration)": _fam(
                e0d, c, cal.step_concentration(c, rng))["pct"],
            "age-only": _fam(e0d, c, cal.age_only(c))["pct"],
            "perfect binary": _fam(e0d, c, cal.perfect_binary(c))["pct"],
            "perfect continuous": _fam(e0d, c, cal.perfect_continuous(c, rng))["pct"],
        }  # fmt: skip
        print(f"calibration [{nan_mode}]: "
              + ", ".join(f"{k} = {v:.4f}" for k, v in got.items()))  # fmt: skip
        assert abs(got["pct artefact (step-concentration)"] - 0.5) < 0.01
        assert got["age-only"] == 0.5
        assert got["perfect binary"] == 1.0
        assert 0.98 < got["perfect continuous"] < 0.995


# --------------------------------------------------------------------------- #
# A2.6 H (reported), C_ws, and the bootstrap (T6, T17, m-7)
# --------------------------------------------------------------------------- #


def test_t6_h_random_is_the_mean_of_k_over_n_t(e0d):
    """T6: H_random == mean(k_t / n_t) exactly over Q_crit, n_t = 16 all-cells and
    15 under A1.3; a step with a NaN eligible slot leaves Q_crit and is counted."""
    rng = np.random.default_rng(4)
    ks = [1, 2, 3, 1, 2]
    D = np.zeros((6, 16))
    for j, k in enumerate(ks):
        D[j, 1 : 1 + k] = 1.0  # ranks 1..k (never rank 15)
    D[5, 2] = 1.0
    D[5, 6] = np.nan  # the sixth step has a NaN eligible slot
    R = rng.random((6, 16))
    c = _grid_cells(D, R, q=[True] * 6, gap=[40] * 6, docs=list(range(6)))
    for keep, n_t in ((np.ones(96, dtype=bool), 16), (e0d.bos_keep(c, 16), 15)):
        fam = _fam(e0d, c, c["r"], tau=0.5, keep=keep)
        assert fam["n_q_crit"] == 5 and fam["n_q_crit_dropped_nan"] == 1
        assert fam["H_random"] == float(np.mean(np.array(ks) / n_t))


def test_h_is_the_critical_share_of_the_tied_minima(e0d):
    """A2.6: h_t is the fraction of the step's tied minima of r that are critical
    (1/2 for a two-way tie with one critical); FIFO and the age oracle as defined."""
    R = np.ones((3, 16))
    D = np.zeros((3, 16))
    R[0, 4] = 0.0
    D[0, 4] = 1.0  # unique argmin, critical: h = 1
    R[1, 4] = R[1, 7] = 0.0
    D[1, 7] = 1.0  # two-way tie, one critical: h = 1/2
    R[2, 9] = 0.0
    D[2, 0] = 1.0  # argmin not critical: h = 0; the critical cell is the oldest
    c = _grid_cells(D, R, q=[True] * 3, gap=[40] * 3, docs=[0, 1, 2])
    fam = _fam(e0d, c, c["r"], tau=0.5, keep=np.ones(48, dtype=bool))
    assert fam["H"] == pytest.approx(0.5)  # (1 + 1/2 + 0) / 3
    assert fam["H_FIFO"] == pytest.approx(1 / 3)
    assert fam["H_age_oracle"] == 0.0


def test_t17_one_bootstrap_matrix_per_seed(e0d, monkeypatch):
    """T17: one W for every statistic of a seed."""
    calls = []
    real = e0d.bootstrap_doc_weights

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(e0d, "bootstrap_doc_weights", counting)
    e0d.analyse_seed(_cells(), M=4, n_docs=40, seed=0, auroc_star=A_STAR, n_boot=3)
    assert len(calls) == 1


def test_t17_pct_is_computed_once_not_per_resample(e0d, cal, monkeypatch):
    """T17: pct is computed once per step on the original data; a resample never
    recomputes it (a document's steps are resampled whole)."""
    c = cal.synthetic_ledger(17, 32)
    r = cal.content(c, np.random.default_rng(0), 1.5)
    counts = {}
    real = e0d.step_percentiles
    for n_boot in (2, 9):
        calls = []

        def counting(*a, _calls=calls, **k):
            _calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(e0d, "step_percentiles", counting)
        _fam(e0d, c, r, n_boot=n_boot)
        counts[n_boot] = len(calls)
    assert counts[2] == counts[9] >= 1


def test_t17_m_a_and_pi_are_re_estimated_per_resample(e0d, cal):
    """T17: C_ws's m_a and H_age-random's pi are re-estimated in each resample (the
    replicate equals the hand value from the weighted sample, not from fixed m_a)."""
    c = cal.synthetic_ledger(18, 24)
    r = cal.content(c, np.random.default_rng(18), 1.0)
    keep = e0d.bos_keep(c, 16)
    w_doc = np.random.default_rng(5).integers(0, 4, 24).astype(float)
    fam = e0d.auroc_family(c, r, c["d_resample"], tau=c["tau"], keep=keep,
                           W=w_doc[None, :], M=16, replicates=True)  # fmt: skip
    pop = e0d.q_population(c, keep, M=16)
    pct = e0d.step_percentiles(r[pop["idx"]], pop)
    d = np.asarray(c["d_resample"])[pop["idx"]]
    lab = ~np.isnan(d)
    y = lab & (d > c["tau"])
    wc = w_doc[pop["doc"]]
    m_a = {}
    for a in np.unique(pop["age"]):
        sel = lab & (pop["age"] == a)
        m_a[a] = np.sum((wc * pct)[sel]) / np.sum(wc[sel])
    num = den = 0.0
    for s in range(pop["n_steps"]):
        m = (pop["step"] == s) & y
        if not m.any():
            continue
        v = np.mean([pct[j] - m_a[pop["age"][j]] for j in np.flatnonzero(m)])
        ws = w_doc[pop["step_doc"][s]]
        num, den = num + ws * v, den + ws
    assert fam["_replicates"]["c_ws"][0] == pytest.approx(num / den, abs=1e-12)
    fixed = fam["c_ws"]
    assert fam["_replicates"]["c_ws"][0] != pytest.approx(fixed, abs=1e-6)
    # pi from the weighted sample of r's own tied minima on Q_crit
    assert np.isfinite(fam["_replicates"]["H_age_random"][0])


def test_undefined_replicates_are_counted_and_the_ci_is_on_the_rest(e0d):
    """m-7 / A2.4: a resample in which the statistic is undefined is counted and
    reported; the label reads the CI over the defined resamples."""
    lo, hi, n_und = e0d.percentile_ci_defined(np.array([0.1, np.nan, 0.3, 0.2]))
    assert n_und == 1
    assert (lo, hi) == e0d.percentile_ci(np.array([0.1, 0.3, 0.2]))
    lo, hi, n_und = e0d.percentile_ci_defined(np.array([np.nan, np.nan]))
    assert n_und == 2 and np.isnan(lo) and np.isnan(hi)
    # in the family: a resample that draws only documents with no positive
    R = np.random.default_rng(0).random((4, 16))
    D = np.zeros((4, 16))
    D[0, 3] = 1.0
    c = _grid_cells(D, R, q=[True] * 4, gap=[40] * 4, docs=[0, 1, 2, 3])
    # docs 0+1 (a positive and same-bin negatives), doc 1 alone (no positive), all
    W = np.array([[2.0, 2, 0, 0], [0, 4.0, 0, 0], [1.0, 1, 1, 1]])
    fam = e0d.auroc_family(c, c["r"], c["d_resample"], tau=0.5,
                           keep=np.ones(64, dtype=bool), W=W, M=16)  # fmt: skip
    assert fam["pct_n_undefined"] == 1
    assert np.isfinite(fam["pct_ci"][0]) and np.isfinite(fam["pct_ci"][1])


def test_an_undefined_primary_statistic_is_exit_3_unless_loo_flat(e0d):
    """A2.4 "Undefined": no bin with both classes -> MeasurementUndefined (exit 3),
    unless the LOO-flat rule labels the seed. PREREG-OPEN: the control shares the
    labels, so it is undefined too; rows 1-2 then label directly."""
    c = _cells(a_delta=0.3, noise=1e-5)  # every Delta below every tau: no positive
    with pytest.raises(e0d.MeasurementUndefined):
        e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=A_STAR, n_boot=3)
    flat = _cells(a_delta=1e-3, noise=1e-5)  # LOO flat (A-cell median < 1e-2)
    an = e0d.analyse_seed(flat, M=4, n_docs=40, seed=0, auroc_star=A_STAR, n_boot=3)
    lab = an["populations"]["bos_excluded"]["labels"]["gated_resample"]
    assert lab in ("DEGENERATE", "LOO_UNINFORMATIVE")


# --------------------------------------------------------------------------- #
# A2.8 labels (T12), row 5b's boolean, STEP_SENSITIVE, T19
# --------------------------------------------------------------------------- #


def _inp(pct, raw=(0.3, 0.4), unstrat=(0.3, 0.4), pc=(0.95, 0.97), lf=False, rf=False,
         **kw):  # fmt: skip
    return {"pc_pct_ci": pc, "loo_flat": lf, "r_flat": rf, "pct_ci": pct,
            "raw_ci": raw, "unstrat_ci": unstrat, **kw}  # fmt: skip


T12 = [
    # (name, inp, label)
    ("row0 ceiling beats everything", _inp((0.9, 0.95), pc=(0.8, 0.849), lf=True),
     "CEILING"),
    ("row0 boundary pc_hi == A* is not ceiling", _inp((0.9, 0.95), pc=(0.8, 0.85)),
     "AGREE"),
    ("row1 degenerate", _inp((0.9, 0.95), lf=True, rf=True), "DEGENERATE"),
    ("row2 loo uninformative", _inp((0.9, 0.95), lf=True), "LOO_UNINFORMATIVE"),
    ("row5b", _inp((0.6, 0.8), raw=(0.86, 0.9), unstrat=(0.5, 0.6)),
     "STEP_OR_AGE_AMBIGUOUS"),
    ("row5b boundary raw_lo == A*", _inp((0.6, 0.8), raw=(0.85, 0.9)),
     "STEP_OR_AGE_AMBIGUOUS"),
    ("row5b tuple with raw lo < A* is DISAGREE", _inp((0.6, 0.8), raw=(0.849, 0.9)),
     "DISAGREE"),
    ("5b beats 3 (INVERTED)", _inp((0.3, 0.45), raw=(0.9, 0.95)),
     "STEP_OR_AGE_AMBIGUOUS"),
    ("5b beats 5 (AGREE_VIA_RANK)",
     _inp((0.6, 0.8), raw=(0.9, 0.95), unstrat=(0.9, 0.95)),
     "STEP_OR_AGE_AMBIGUOUS"),
    ("row3 inverted", _inp((0.3, 0.45)), "INVERTED"),
    ("row3 boundary pct_hi == 0.5 is not inverted", _inp((0.3, 0.5)), "DISAGREE"),
    ("row4 agree", _inp((0.86, 0.9), raw=(0.6, 0.8)), "AGREE"),
    ("row4 boundary pct_lo == A*", _inp((0.85, 0.9)), "AGREE"),
    ("row4 is not read from raw", _inp((0.8, 0.9), raw=(0.86, 0.95)), "UNRESOLVED"),
    ("row5 agree via rank", _inp((0.6, 0.8), unstrat=(0.86, 0.9)), "AGREE_VIA_RANK"),
    ("row5 boundary unstrat_lo == A*", _inp((0.6, 0.8), unstrat=(0.85, 0.9)),
     "AGREE_VIA_RANK"),
    ("row6 disagree; raw CI containing A* does not block",
     _inp((0.6, 0.8), raw=(0.8, 0.9)),
     "DISAGREE"),
    ("row7 unresolved", _inp((0.8, 0.9)), "UNRESOLVED"),
    ("row7 boundary pct_hi == A*", _inp((0.8, 0.85)), "UNRESOLVED"),
]  # fmt: skip


@pytest.mark.parametrize("name, inp, want", T12, ids=[x[0] for x in T12])
def test_t12_a28_labels_first_match_in_order(e0d, name, inp, want):
    assert e0d.seed_label_a2(inp, auroc_star=A_STAR) == want


def test_row5b_boolean_is_reported_whatever_the_label(e0d):
    """A2.8: the 5b condition (pct CI upper < A*, raw CI lower >= A*) is reported as
    a boolean on every seed, even when rows 0-2 absorb it."""
    absorbed = _inp((0.6, 0.8), raw=(0.9, 0.95), pc=(0.5, 0.6))
    assert e0d.seed_label_a2(absorbed, auroc_star=A_STAR) == "CEILING"
    assert e0d.row5b(absorbed, auroc_star=A_STAR) is True
    assert e0d.row5b(_inp((0.86, 0.9), raw=(0.9, 0.95)), auroc_star=A_STAR) is False
    an = e0d.analyse_seed(_cells(score="bad"), M=4, n_docs=40, seed=0,
                          auroc_star=A_STAR, n_boot=4)  # fmt: skip
    for p in an["populations"].values():
        assert set(p["row5b"]) == set(COMBOS)
        assert all(isinstance(v, bool) for v in p["row5b"].values())


def test_step_sensitive_substitutes_raw_and_skips_5b(e0d):
    """A2.5: the raw-substituted label replaces AUROC_strat,pct by raw AUROC_strat
    and skips row 5b; STEP_SENSITIVE flags a difference and moves no exit."""
    inp = _inp((0.6, 0.8), raw=(0.86, 0.9))
    assert e0d.seed_label_a2(inp, auroc_star=A_STAR) == "STEP_OR_AGE_AMBIGUOUS"
    assert e0d.seed_label_a2(inp, auroc_star=A_STAR, stat="raw") == "AGREE"
    ans = {s: _an("DISAGREE", raw_label="AGREE") for s in range(3)}
    out = e0d.classify_run(ans)
    assert out["step_sensitive"] == {"0": True, "1": True, "2": True}
    assert (out["class"], out["exit"]) == ("CONFOUND", 1)


def test_t19_h_is_read_by_no_gate(e0d):
    """T19: H = 0 and H = 1 with the same AUROC_strat,pct CI give the identical
    label, class and exit (A2.1, A2.6)."""
    for pct in ((0.86, 0.9), (0.6, 0.8), (0.3, 0.4), (0.8, 0.9)):
        a = _inp(pct, H=0.0, H_ci=(0.0, 0.0), R_H=0.0, R_H_ci=(0.0, 0.0))
        b = _inp(pct, H=1.0, H_ci=(1.0, 1.0), R_H=9.0, R_H_ci=(9.0, 9.0))
        assert e0d.seed_label_a2(a, auroc_star=A_STAR) == e0d.seed_label_a2(
            b, auroc_star=A_STAR)  # fmt: skip


# --------------------------------------------------------------------------- #
# §8 classification (T13), HALT on 5b, pooled never gates
# --------------------------------------------------------------------------- #

SA = "STEP_OR_AGE_AMBIGUOUS"
T13 = [
    (["AGREE"] * 3, ("AGREE", 0)),
    (["INVERTED"] * 3, ("CONFOUND_INVERTED", 1)),
    (["DISAGREE", "INVERTED", "DISAGREE"], ("CONFOUND", 1)),
    (["DISAGREE", "DISAGREE", SA], (SA, 2)),
    (["AGREE_VIA_RANK", "INVERTED", SA], (SA, 2)),
    ([SA] * 3, (SA, 2)),
    (["AGREE_VIA_RANK", "DISAGREE", "INVERTED"], ("RECENCY_ONLY", 2)),
    (["AGREE", "AGREE", "DISAGREE"], ("MIXED_UNRESOLVED", 2)),
    (["AGREE", "AGREE", SA], ("MIXED_UNRESOLVED", 2)),
    (["DEGENERATE", "LOO_UNINFORMATIVE", SA], ("DEGENERATE_UNINFORMATIVE", 2)),
    (["CEILING", "DISAGREE", "DISAGREE"], ("MIXED_UNRESOLVED", 2)),
    (["CEILING", "INVERTED", "INVERTED"], ("MIXED_UNRESOLVED", 2)),
    (["CEILING", "AGREE", "AGREE"], ("MIXED_UNRESOLVED", 2)),
    (["UNRESOLVED", "DISAGREE", "DISAGREE"], ("MIXED_UNRESOLVED", 2)),
]


@pytest.mark.parametrize("labels, want", T13, ids=["-".join(x[0]) for x in T13])
def test_t13_classification_rows(e0d, labels, want):
    assert e0d.classify(labels) == want


def test_t13_every_section_8_row_is_reachable_and_exit_1_only_from_rows_3_4(e0d):
    got = {e0d.classify(ls)[0] for ls, _ in T13}
    got.add(e0d.classify(["DEGENERATE", "DEGENERATE", "AGREE"])[0])
    assert got == set(e0d.CLASS_EXIT)
    assert {k for k, v in e0d.CLASS_EXIT.items() if v == 1} == {
        "CONFOUND",
        "CONFOUND_INVERTED",
    }
    assert e0d.CLASS_EXIT[SA] == 2


@pytest.mark.parametrize(
    "labels",
    [
        ["DISAGREE", "DISAGREE", SA],
        ["AGREE", "AGREE", SA],
        ["DEGENERATE", "LOO_UNINFORMATIVE", SA],
    ],
)
def test_halt_on_5b_names_the_seeds_and_exits_2(e0d, labels):
    """A2.8 "HALT on row 5b": any STEP_OR_AGE_AMBIGUOUS seed -> exit 2, the report
    names the seed(s) and halts to Brendan; never 1."""
    ans = {s: _an(lab, row5b=lab == SA) for s, lab in enumerate(labels)}
    out = e0d.classify_run(ans)
    assert out["exit"] == 2 and out["halt"] is True
    assert out["halt_seeds"] == [2]
    assert out["row5b"] == {"0": False, "1": False, "2": True}
    ok = e0d.classify_run({s: _an("DISAGREE") for s in range(3)})
    assert ok["halt"] is False and ok["halt_seeds"] == []


def test_pooled_results_never_gate(e0d):
    """The ruling: each seed separately; pooled results are reported and never
    gate (A2.7, A2.10)."""
    ans = {0: _an("AGREE"), 1: _an("AGREE"), 2: _an("DISAGREE")}
    out = e0d.classify_run(ans, pooled={"label": "AGREE", "pct": 0.9})
    assert (out["class"], out["exit"]) == ("MIXED_UNRESOLVED", 2)
    assert out["pooled"]["label"] == "AGREE"
    out = e0d.classify_run({s: _an("DISAGREE") for s in range(3)},
                           pooled={"label": "AGREE"})  # fmt: skip
    assert (out["class"], out["exit"]) == ("CONFOUND", 1)


def test_tau_sensitivity_and_label_a1_are_reported_never_the_exit(e0d):
    ans = {s: _an("DISAGREE", label_a1="AGREE", tau_labels={"0.99": "AGREE"})
           for s in range(3)}  # fmt: skip
    out = e0d.classify_run(ans)
    assert (out["class"], out["exit"]) == ("CONFOUND", 1)
    assert out["class_A1"] == "AGREE"
    assert out["tau_sensitivity"]["0.99"]["class"] == "AGREE"
    assert out["tau_sensitivity"]["0.999"]["class"] == "CONFOUND"


def test_analyse_seed_reports_the_full_grid(e0d):
    """A2.7 item 1: the 2x2 grid, every family statistic and the A2.8 label in each
    cell; the zero column uses TAU_ZERO; label_A1 is reported."""
    an = e0d.analyse_seed(_cells(score="good"), M=4, n_docs=40, seed=1,
                          auroc_star=A_STAR, n_boot=5)  # fmt: skip
    for p in an["populations"].values():
        assert set(p["a2"]) == set(COMBOS) and set(p["labels"]) == set(COMBOS)
        assert p["a2"]["gated_zero"]["r"]["tau"] == e0d.TAU_ZERO[1]
        for k in ("pct", "pct_ci", "raw", "raw_ci", "unstrat_pct", "c_ws", "H",
                  "H_random", "R_H_uniform"):  # fmt: skip
            assert k in p["a2"]["gated_resample"]["r"], k
        assert "pct_ci" in p["a2"]["gated_resample"]["pc"]
        assert set(p["tau_sensitivity"]) == {"0.99", "0.999"}
        assert "gated_resample" in p["labels_A1"]
    assert an["populations"]["bos_excluded"]["labels"]["gated_resample"] == "AGREE"


def test_main_writes_the_a2_class_and_the_5b_booleans(e0d, cleared, fake_ledger):
    cleared["score"] = "good"
    assert int(e0d.main([])) == 0
    assert _row(fake_ledger, "e0d.class") == "AGREE"
    assert _row(fake_ledger, "e0d.row5b") == {"0": False, "1": False, "2": False}
    assert _row(fake_ledger, "e0d.halt") is False
    assert _row(fake_ledger, "c12_tau_tables") is not None
    assert _row(fake_ledger, "e0d.pooled") is not None
    vals = fake_ledger.last.rows["primary.AUROC_strat_pct"]
    # _cells' Q-steps without a resident assert hold a pct = 1 negative (the T1c
    # ceiling), so the good score sits below 1.0 and above A*
    assert len(vals) == 3 and all(A_STAR < v < 1.0 for v in vals), vals
    for s in (0, 1, 2):
        an = fake_ledger.last.rows[f"seed{s}.analysis"]
        prim = an["populations"]["bos_excluded"]["a2"]["gated_resample"]["r"]
        assert prim["pct"] == vals[s]


def test_main_refuses_on_c12_before_any_document(e0d, cleared, fake_ledger, monkeypatch):
    monkeypatch.setattr(e0d, "TAU_RESAMPLE", {0: 0.4, 1: 0.4, 2: 0.4})
    assert int(e0d.main([])) == 3
    assert fake_ledger.last.rows["control_failed"]["control"] == "C12"
    assert cleared["gen"] == []


def test_sink_probe_reports_share_by_kind(e0d):
    """A2.7 item 8: the r_i share by sentence kind at Q-steps (reported only)."""
    an = e0d.analyse_seed(_cells(score="good"), M=4, n_docs=40, seed=0,
                          auroc_star=A_STAR, n_boot=2)  # fmt: skip
    sp = an["populations"]["all"]["sink_probe"]["gated"]
    assert set(sp["mean_r_by_kind_Q"]) >= {"pending_assert", "filler"}


def test_c10_prime_control_is_true_demand_through_the_same_family(e0d):
    """A2.8 row 0 (C10'): the control is `true_demand` (1 on the A-cell), run through
    the identical family, labels and bootstrap as r_i; it never re-scores r_i."""
    c = _cells(score="bad")
    an = e0d.analyse_seed(c, M=4, n_docs=40, seed=0, auroc_star=A_STAR, n_boot=6)
    p = an["populations"]["bos_excluded"]
    W = e0d.bootstrap_doc_weights(40, 6, e0d.BOOT_SEED)
    for k in ("resample", "zero"):
        want = e0d.auroc_family(c, c["pc"], c[f"d_{k}"], tau=e0d.tau_for(0, k),
                                keep=e0d.bos_keep(c, 4), W=W, M=4)  # fmt: skip
        got = p["a2"][f"gated_{k}"]["pc"]
        assert got["pct"] == want["pct"] and got["pct_ci"] == want["pct_ci"]
    assert p["a2"]["gated_resample"]["pc"]["pct"] != p["a2"]["gated_resample"]["r"]["pct"]


def test_calibration_nan_modes(cal):
    """B.1: under nan_mode="matched" a NaN d_resample is about 2x more frequent on
    A-cells (1.7 %) than elsewhere (0.8 %), so it depends on y; "independent" keeps
    0.8 % regardless of y."""
    m = cal.synthetic_ledger(20, 2048, nan_mode="matched")
    i = cal.synthetic_ledger(20, 2048, nan_mode="independent")
    for c, lo, hi in ((m, 1.5, 3.0), (i, 0.6, 1.5)):
        a = np.isnan(c["d_resample"][c["a_cell"]]).mean()
        o = np.isnan(c["d_resample"][~c["a_cell"]]).mean()
        assert lo < a / o < hi, (a, o)
