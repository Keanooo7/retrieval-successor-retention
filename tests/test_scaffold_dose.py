"""Scaffold dose's own gates (`experiments/scaffold-dose/PREREG.md`, 4249567).

No test trains or measures a real checkpoint: the parent loop is driven by fake
children over checkpoint files touched on disk and a fake clock, the rule is fed
synthetic S0-03 tables, and the child's ``train()`` call is recorded, not run.
Mutation-battery gates (`scripts/mutation_battery.py`, ``# --- scaffold-dose ---``):

* ``test_classification_table`` -- the table's AT_MEMORISATION and EARLY rows swapped;
* ``test_k_star_is_the_smallest`` -- k* the largest k with U(k), not the smallest;
* ``test_U_needs_every_seed`` -- U(k) on ANY seed instead of every seed;
* ``test_control_2_fails_at_2e_6`` -- the measurement-path tolerance widened;
* ``test_arm_stream_ids_are_disjoint`` -- control 3 drops the held-out overlap;
* ``test_start_checkpoint_sha_mismatch_refuses`` -- the sha256 comparison dropped;
* ``test_monotonicity_is_reported`` -- a violation not named;
* ``test_measurement_is_the_corpus_size_function`` -- the measurement retyped.
"""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import rsr.train.loop as loop  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "scaffold_dose_run", ROOT / "experiments" / "scaffold-dose" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]
KS = (100, 200, 300, 400, 600)
YES, NO = 0.0625, 0.015625  # binary-exact, either side of DELTA
C_YES = 0.875


def _prereg_text() -> str:
    return (ROOT / "experiments" / "scaffold-dose" / "PREREG.md").read_text()


def _prereg() -> dict:
    return yaml.safe_load(_prereg_text().split("---")[1])


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed; the instrument, imported
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    p = _prereg()
    t = p["thresholds"]
    assert run.SEEDS == p["seeds"] == SEEDS
    assert list(run.ARMS_K) == p["arms_k"] == list(KS)
    assert p["arm_order"].startswith("600, 400, 300, 200, 100 ")
    assert run.ARM_ORDER == (600, 400, 300, 200, 100)
    assert "3 seeds as parallel children x 5 threads" in p["arm_order"]
    assert run.THREADS_PER_SEED == 5 and run.DEVICE == "cpu"
    assert "train steps k -> k + 500" in p["arm"] and run.STEPS == 500
    assert "measure ckpt k + 500" in p["arm"]
    assert "[4160 + 16 t, 4160 + 16 t + 16)" in p["arm"]
    assert (run.STREAM.offset, run.STREAM.stride, run.BATCH) == (4160, 16, 16)
    assert run.STREAM is run.FS.STREAM  # "the fresh-stream stream"
    assert run.N_TRAIN == 64 and run.CKPT_EVERY == 100
    assert t["DELTA"].startswith("0.03 ") and run.DELTA == 0.03 == run.ST.DELTA
    assert t["stream_loss_threshold"].startswith("ln 16 - 0.10 = 2.6726 ")
    assert pytest.approx(2.6726, abs=1e-4) == run.STREAM_LOSS_THRESHOLD
    assert t["reproduction_tolerance"] == "exact (resume) / 1e-6 (measurement)"
    assert run.REPRO_TOL == 1e-6
    assert t["deadline"].startswith("2026-09-26 06:00 America/Los_Angeles")
    la = dt.datetime(2026, 9, 26, 6, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert run.deadline_epoch() == la.timestamp()
    assert "held-out live Brier16 ≤ 0.8875" in _prereg_text()
    assert pytest.approx(0.8875) == run.C_THRESHOLD
    assert run.CONTROL_K == 600 and run.CONTROL_PREFIX == "n64.ckpt600."
    assert "`n64.ckpt600.*` within 1e-6" in " ".join(_prereg_text().split())
    assert run.HELDOUT == (4096, 4160) and run.PROBE == (0, 64)
    assert " ".join(p["decision_rule"].split()) == run.DECISION_RULE
    assert " ".join(p["question"].split()) == run.QUESTION
    assert run.PREREG_COMMIT == "4249567"


def test_expected_is_the_preregs_verbatim():
    sec = _prereg_text().split("## Author's expectation")[1].split("\n\n", 1)[1]
    sec = sec.split("\n\n")[0]
    assert " ".join(sec.split()) == run.EXPECTED


def test_reading_is_the_preregs_table():
    rows = [
        line
        for line in _prereg_text().splitlines()
        if line.startswith("| ") and "**" in line
    ]
    got = {}
    for line in rows:
        cell = line.split("|")[2].strip()
        name, reading = cell.split(" — ", 1)
        got[name.strip("*")] = reading
    assert got == run.READING


def test_measurement_is_the_corpus_size_function():
    """Identity, not a copy: the object the parent calls is the corpus-size curve's
    ``measure_checkpoint``, the same object scaffold-timing and fresh-stream call."""
    assert run.measure_checkpoint is run.CSC.measure_checkpoint
    assert run.measure_checkpoint is run.ST.measure_checkpoint
    assert run.measure_checkpoint is run.FS.measure_checkpoint
    assert run.measure_checkpoint.__code__.co_filename.endswith(
        "experiments/corpus-size-curve/run.py"
    )
    src = (ROOT / "experiments" / "scaffold-dose" / "run.py").read_text()
    assert "def measure(" not in src and "def measure_checkpoint(" not in src
    assert "def doc_sets(" not in src and "def run_arm(" not in src
    assert "measure_fn=measure_checkpoint" in src


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kstar", "want"),
    [
        (None, "UNLOCK_AFTER_600"),
        (600, "AT_RETRIEVAL_ONSET"),
        (400, "AT_MEMORISATION"),
        (300, "AT_MEMORISATION"),
        (200, "EARLY"),
        (100, "EARLY"),
    ],
)
def test_classification_table(kstar, want):
    assert run.classify(kstar) == want
    assert want in run.READING


def test_classification_refuses_a_k_outside_the_sweep():
    with pytest.raises(ValueError, match="not in the sweep"):
        run.classify(500)


def test_k_star_is_the_smallest():
    u = {100: False, 200: True, 300: False, 400: True, 600: True}
    assert run.k_star(u) == 200
    assert run.k_star({100: True, 200: True, 300: True, 400: True, 600: True}) == 100
    assert run.k_star({k: False for k in KS}) is None
    assert run.k_star({600: True, 400: None}) == 600
    v = run.verdict(res_for({k: YES if k in (300, 600) else NO for k in KS}))
    assert v["k_star"] == 300 and v["classification"] == "AT_MEMORISATION"


def test_monotonicity_is_reported():
    """U(200) and not U(300): named, and k* is still the smallest k."""
    u = {100: False, 200: True, 300: False, 400: True, 600: False}
    m = run.monotonicity(u)
    assert not m["monotone"]
    assert m["violations"] == [[200, 300], [200, 600], [400, 600]]
    assert run.monotonicity({100: False, 200: False, 300: True, 400: True, 600: True})[
        "monotone"
    ]
    assert run.monotonicity({k: False for k in KS}) == {
        "monotone": True,
        "violations": [],
    }
    v = run.verdict(res_for({k: YES if u[k] else NO for k in KS}))
    assert v["classification"] == "EARLY" and v["k_star"] == 200
    assert v["monotonicity"]["violations"] == [[200, 300], [200, 600], [400, 600]]


def table(r: float = YES, brier: float = C_YES, *, base: float = 0.125) -> dict:
    """An S0-03 measure() result: at gap_2_to_M held-out live = base + r, held-out
    slots_zeroed = base, held-out live Brier16 = ``brier``."""
    ds = {"n_documents": 64, "fraction_of_pairs_gap_gt_M": 0.1}
    for cond in run.S003.CONDITIONS:
        ds[cond] = {"real_token_nll": 1.5}
        for b in run.S003.BUCKETS:
            ds[cond][b] = {
                "n": 10,
                "answer_nll": 2.5,
                "answer_acc": base,
                "answer_nll_over_16": 2.25,
                "answer_brier_over_16": 0.9375,
            }
    ho = copy.deepcopy(ds)
    ho["live"]["gap_2_to_M"]["answer_acc"] = base + r
    ho["live"]["gap_2_to_M"]["answer_brier_over_16"] = brier
    tr = copy.deepcopy(ds)
    tr["live"]["gap_2_to_M"]["answer_acc"] = 0.75
    tr["slots_zeroed"]["gap_2_to_M"]["answer_acc"] = 0.5
    tr["gate_zeroed"]["gap_2_to_M"]["answer_acc"] = 0.5
    tr["gate_zeroed_bos_off"]["gap_2_to_M"]["answer_acc"] = 0.375
    return {"heldout": ho, "train": tr, "memory_gates": [0.25]}


def arm_per(k: int, r: float = YES, cells=None) -> dict:
    """``{seed: {k + 500: measurement}}``; ``cells[seed]`` overrides one seed's R."""
    cells = cells or {}
    e = k + 500
    return {s: {e: {"stored_step": e, "s003": table(cells.get(s, r))}} for s in SEEDS}


def ok_control() -> dict:
    c = run.ST.reproduction_control({"k": [1.0, 2.0, 3.0]}, {"k": [1.0, 2.0, 3.0]})
    return {**c, "error": None}


def res_for(r_by_k: dict[int, float], **over) -> dict:
    res = {
        "control_2": ok_control(),
        "preflight": {"ok": True, "error": None},
        "arms": {
            k: {"per": arm_per(k, r), "error": None, "stopped": None}
            for k, r in r_by_k.items()
        },
        "resume_check": {k: {s: {"ok": True} for s in SEEDS} for k in r_by_k},
        "not_run": {},
        "error": None,
    }
    res.update(over)
    return res


@pytest.mark.parametrize(
    ("unlocked", "want"),
    [
        ((), "UNLOCK_AFTER_600"),
        ((600,), "AT_RETRIEVAL_ONSET"),
        ((400, 600), "AT_MEMORISATION"),
        ((300, 400, 600), "AT_MEMORISATION"),
        ((200, 300, 400, 600), "EARLY"),
        (KS, "EARLY"),
    ],
)
def test_verdict_every_row_from_tables(unlocked, want):
    v = run.verdict(res_for({k: YES if k in unlocked else NO for k in KS}))
    assert (v["classification"], v["exit"]) == (want, Exit.OK)
    assert v["U"] == {k: k in unlocked for k in KS}
    assert v["monotonicity"]["monotone"]


def test_U_needs_every_seed():
    """One seed below DELTA at ckpt k + 500 and U(k) fails; DELTA itself holds."""
    res = res_for({k: YES for k in KS})
    res["arms"][100]["per"] = arm_per(100, YES, cells={1: NO})
    v = run.verdict(res)
    assert v["U"][100] is False and v["U"][200] is True
    assert v["k_star"] == 200 and v["classification"] == "EARLY"
    res["arms"][100]["per"] = arm_per(100, run.DELTA)
    assert run.verdict(res)["U"][100] is True


def test_U_reads_the_end_checkpoint_only():
    res = res_for({k: YES for k in KS})
    per = arm_per(300, NO)
    for s in SEEDS:
        per[s][300 + 100] = {"stored_step": 400, "s003": table(YES)}
    res["arms"][300]["per"] = per
    assert run.verdict(res)["U"][300] is False


@pytest.mark.parametrize(
    "case",
    [
        "resume_failed",
        "resume_missing",
        "end_missing",
        "measurement_raised",
        "arm_not_run",
        "control_2_failed",
        "control_2_absent",
        "preflight_failed",
        "run_error",
    ],
)
def test_unreadable_or_failed_controls_are_inconclusive(case):
    res = res_for({k: YES for k in KS})
    if case == "resume_failed":
        res["resume_check"][400][1] = {"ok": False}
    elif case == "resume_missing":
        res["resume_check"][200][2] = None
    elif case == "end_missing":
        del res["arms"][100]["per"][0][600]
    elif case == "measurement_raised":
        res["arms"][300]["error"] = "RuntimeError: bad table"
    elif case == "arm_not_run":
        del res["arms"][100]
        res["not_run"][100] = "deadline"
    elif case == "control_2_failed":
        res["control_2"] = {
            **run.ST.reproduction_control({"k": [1.0] * 3}, {"k": [2.0] * 3}),
            "error": None,
        }
    elif case == "control_2_absent":
        res["control_2"] = None
    elif case == "preflight_failed":
        res["preflight"] = {"ok": False, "error": "word 'x' not in the vocabulary"}
    elif case == "run_error":
        res["control_2"] = {"ok": False, "error": "OSError: gone", "per_seed": {}}
    v = run.verdict(res)
    assert (v["classification"], v["exit"]) == ("inconclusive", Exit.DID_NOT_RUN)
    assert v["k_star"] is None


def test_unreadable_arm_is_named_and_its_U_is_none():
    res = res_for({k: YES for k in KS})
    res["resume_check"][400][1] = {"ok": False}
    v = run.verdict(res)
    assert v["U"][400] is None and v["cells"][400]["R"] is True
    assert "arm k400 unreadable" in v["detail"] and "seed1" in v["detail"]


# --------------------------------------------------------------------------- #
# control 2: the measurement path
# --------------------------------------------------------------------------- #


def _control_per() -> dict:
    return {s: {600: {"stored_step": 600, "s003": table(0.01 * s, 1.25)}} for s in SEEDS}


def _measure_600(calls=None):
    def m(seed_dir, seed, label, n):
        if calls is not None:
            calls.append((seed_dir, seed, label, n))
        return _control_per()[seed][label]

    return m


def test_control_2_reference_keys_are_the_scaffold_ledgers():
    """Scaffold-timing's own arm_rows, fed ckpt-600 tables, produces exactly the
    statistic keys its committed ledger holds under n64.ckpt600."""
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    ref = run.reference_rows(doc)
    got = run.remeasured_rows(_control_per())
    assert sorted(got) == sorted(ref) and len(ref) > 200
    assert all(k.startswith("n64.ckpt600.") for k in ref)
    assert "n64.ckpt600.R_quantity" in ref and "n64.ckpt600.M_quantity" in ref


def test_control_2_measures_ckpt_600_of_every_seed_from_the_source(tmp_path):
    calls = []
    ref = run.remeasured_rows(_control_per())
    c = run.control_2(ref, tmp_path, _measure_600(calls), log=lambda s: None)
    assert c["ok"] and c["error"] is None
    assert calls == [(tmp_path / "n64" / f"seed{s}", s, 600, 64) for s in SEEDS]


def test_control_2_exact_passes_and_inside_tolerance_passes(tmp_path):
    ref = run.remeasured_rows(_control_per())
    near = copy.deepcopy(ref)
    near[sorted(near)[5]][1] += 5e-7
    assert run.control_2(near, tmp_path, _measure_600(), log=lambda s: None)["ok"]


def test_control_2_fails_at_2e_6(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[7]
    ref[k][2] += 2e-6
    c = run.control_2(ref, tmp_path, _measure_600(), log=lambda s: None)
    assert not c["ok"] and c["tolerance"] == 1e-6
    assert c["per_seed"][2]["ok"] is False and c["per_seed"][0]["ok"]
    assert c["per_seed"][2]["failures"][0]["key"] == k


def test_control_2_raise_or_missing_key_fails(tmp_path):
    def boom(*a):
        raise RuntimeError("unreadable checkpoint")

    ref = run.remeasured_rows(_control_per())
    c = run.control_2(ref, tmp_path, boom, log=lambda s: None)
    assert not c["ok"] and "unreadable checkpoint" in c["error"]
    ref["n64.ckpt600.not_a_key"] = [0.0, 0.0, 0.0]
    c = run.control_2(ref, tmp_path, _measure_600(), log=lambda s: None)
    assert not c["ok"] and c["missing_keys"] == ["n64.ckpt600.not_a_key"]


# --------------------------------------------------------------------------- #
# control 3: the stream ids each arm uses
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k", KS)
def test_arm_stream_ids_are_disjoint(k):
    d = run.arm_disjointness(k)
    assert d["ok"] and d["repeats"] == 0 and d["n_ids"] == 16 * 500
    assert (d["first_id"], d["last_id"]) == (4160 + 16 * k, 4160 + 16 * (k + 500) - 1)
    assert not d["ids_in_probe"] and not d["ids_in_heldout"]
    assert not d["ids_in_vocab_documents"]
    ids = run.arm_stream_ids(k)
    assert ids == list(range(4160 + 16 * k, 4160 + 16 * (k + 500)))
    assert min(ids) >= 5760 and max(ids) < 4160 + 16 * run.closure_iters()


def test_arm_disjointness_catches_overlap(monkeypatch):
    from rsr.train.loop import DocumentStream

    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=2400, stride=16))
    d = run.arm_disjointness(100)
    assert not d["ok"] and d["ids_in_heldout"][0] == 4096 and not d["ids_in_probe"]
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=0, stride=16))
    d = run.arm_disjointness(1)
    assert not d["ok"] and d["ids_in_probe"][0] == 16
    assert d["ids_in_vocab_documents"][0] == 16
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=4160 + 16, stride=16))
    assert run.arm_disjointness(0)["ok"]


def test_preflight_stops_on_an_overlapping_arm_and_on_a_closure_failure(monkeypatch):
    closures = []
    monkeypatch.setattr(
        run.FS, "vocabulary_closure", lambda s, n: closures.append((s, n)) or {"ok": 1}
    )
    pf = run.preflight()
    assert pf["ok"] and closures == [(s, 1100) for s in SEEDS]
    assert sorted(pf["arms"]) == sorted(f"k{k}" for k in KS)

    def bad(s, n):
        raise ValueError("word 'the_harbour' is not in the vocabulary")

    monkeypatch.setattr(run.FS, "vocabulary_closure", bad)
    pf = run.preflight()
    assert not pf["ok"] and "the_harbour" in pf["error"]
    real = run.arm_disjointness
    monkeypatch.setattr(run, "arm_disjointness", lambda k: {**real(k), "ok": k != 300})
    pf = run.preflight()
    assert not pf["ok"] and "k300" in pf["error"]


def test_the_closure_is_fresh_streams_over_a_short_stream():
    c = run.FS.vocabulary_closure(0, 2)
    assert c["ok"] and c["n_stream_documents"] == 32


# --------------------------------------------------------------------------- #
# the start checkpoints
# --------------------------------------------------------------------------- #


def _fake_sources(tmp_path) -> tuple[Path, dict]:
    src = tmp_path / "src"
    shas = {}
    for k in KS:
        for s in SEEDS:
            f = run.start_ckpt(k, s, src)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(f"k{k}s{s}".encode())
            shas[f"n64/seed{s}/{f.name}"] = run.sha256_file(f)
    return src, {"checkpoint_sha256": shas}


def _step_from_name(f: Path) -> int:
    return int(f.stem.split("-")[1])


def test_start_checkpoint_sha_mismatch_refuses(tmp_path):
    src, man = _fake_sources(tmp_path)
    got = run.verify_start_checkpoints(src, man, step_fn=_step_from_name)
    assert sorted(got) == sorted(f"k{k}" for k in KS)
    assert got["k300"]["seed1"] == man["checkpoint_sha256"]["n64/seed1/ckpt-000300.pt"]
    bad = run.start_ckpt(400, 2, src)
    bad.write_bytes(b"not the checkpoint scaffold-timing measured")
    with pytest.raises(run.StartCheckpointRefused, match=r"ckpt-000400\.pt sha256"):
        run.verify_start_checkpoints(src, man, step_fn=_step_from_name)


def test_start_checkpoint_absent_or_wrong_step_refuses(tmp_path):
    src, man = _fake_sources(tmp_path)
    with pytest.raises(run.StartCheckpointRefused, match="does not store step"):
        run.verify_start_checkpoints(src, man, step_fn=lambda f: 1000)
    run.start_ckpt(100, 0, src).unlink()
    with pytest.raises(run.StartCheckpointRefused, match="is absent"):
        run.verify_start_checkpoints(src, man, step_fn=_step_from_name)


def test_start_checkpoints_are_scaffold_timings_files():
    """The committed scaffold-timing manifest records every start checkpoint."""
    man = json.loads((ROOT / run.REF_MANIFEST).read_text())
    for k in KS:
        for s in SEEDS:
            assert f"n64/seed{s}/ckpt-{k:06d}.pt" in man["checkpoint_sha256"]


def test_start_quantities_read_the_committed_ledger():
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    st = run.start_quantities(doc)
    assert sorted(st) == list(KS)
    rows = {r["key"]: r for r in doc["rows"]}
    for k in KS:
        assert st[k]["R_quantity"] == rows[f"n64.ckpt{k}.R_quantity"]["samples"]
        assert st[k]["M_quantity"] == rows[f"n64.ckpt{k}.M_quantity"]["samples"]


# --------------------------------------------------------------------------- #
# the child: fresh-stream's arm B from ckpt k
# --------------------------------------------------------------------------- #


def test_single_is_fresh_streams_arm_B_from_ckpt_k(tmp_path, monkeypatch):
    """The corpus-size N = 64 ``train()`` call + the stream + the resume from
    ``ckpt-{k:06d}.pt``, ending at k + 500; fresh-stream's RESUME_STEP restored."""
    seen = []
    monkeypatch.setattr(loop, "train", lambda **kw: seen.append(kw) or {"ok": 1})
    checks = []

    class NoCheck:
        def __init__(self, ckpt, out):
            checks.append((ckpt, out))
            self.result = {"ok": True}

        def installed(self):
            import contextlib

            return contextlib.nullcontext()

    monkeypatch.setattr(run.FS, "ResumeCheck", NoCheck)
    run.CSC.single(1, 800, 64, tmp_path)
    r = run.single(300, 1, tmp_path, source_root=tmp_path / "src")
    ref, got = seen
    want = tmp_path / "src" / "n64" / "seed1" / "ckpt-000300.pt"
    assert "n_documents" not in ref and ref["iters"] == 800
    assert got == {**ref, "stream": run.STREAM, "resume": str(want)}
    assert checks == [(want, tmp_path / "resume_check.json")]
    assert run.FS.RESUME_STEP == 1000
    assert r["arm"] == "k300" and r["k"] == 300 and r["resume_check"] == {"ok": True}
    with pytest.raises(ValueError):
        run.single(500, 0, tmp_path)


def test_resume_check_is_fresh_streams_exact_check():
    assert run.FS.ResumeCheck.check.__code__.co_filename.endswith(
        "experiments/fresh-stream/run.py"
    )
    src = (ROOT / "experiments" / "scaffold-dose" / "run.py").read_text()
    assert "class ResumeCheck" not in src and "def compare_to_checkpoint" not in src


def test_fresh_stream_with_restores_and_refuses_unknown_names():
    before = run.FS.CHECKPOINTS
    with run.fresh_stream_with(CHECKPOINTS={"k100": (600,)}):
        assert run.FS.CHECKPOINTS == {"k100": (600,)}
    assert run.FS.CHECKPOINTS is before
    with pytest.raises(AttributeError), run.fresh_stream_with(NOT_A_NAME=1):
        pass


def test_child_argv(tmp_path):
    argv = run.child_argv(400, 2, tmp_path / "o", tmp_path / "s")
    assert argv[1].endswith("experiments/scaffold-dose/run.py")
    assert argv[2:] == [
        "--single",
        "--k",
        "400",
        "--seed",
        "2",
        "--out-dir",
        str(tmp_path / "o"),
        "--source-root",
        str(tmp_path / "s"),
    ]


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


class FakeChild:
    """Writes ckpt ``end`` at once (unless ``upto`` is below it) and the arm's
    resume_check.json, then exits ``rc`` -- or stays alive if ``alive``."""

    def __init__(self, d: Path, end: int, upto=None, rc=0, alive=False, resume=True):
        d.mkdir(parents=True, exist_ok=True)
        if upto is None or end <= upto:
            run.RC.ckpt_path(d, end).write_bytes(b"x")
        if resume is not None:
            (d / "resume_check.json").write_text(json.dumps({"ok": resume}))
        self.argv = ["py", "run.py", "--single"]
        self.rc, self.alive = rc, alive

    def poll(self):
        return None if self.alive else self.rc

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return self.rc


def fake_measure(r_by_k=None):
    calls = []
    r_by_k = r_by_k or {}

    def m(seed_dir, seed, label, n):
        k = int(seed_dir.parent.name[1:])
        calls.append((k, seed, label, n))
        return {"stored_step": label, "s003": table(r_by_k.get(k, YES), C_YES)}

    return m, calls


def _kw(measure_fn, *, control=None, upto=None, resume=True, clock=None, **extra):
    upto = upto or {}
    spawned, order = [], []

    def spawn(k, s, d):
        spawned.append((k, s))
        order.append(("spawn", k, s))
        return FakeChild(
            d, k + 500, upto=upto.get((k, s)), alive=(k, s) in upto, resume=resume
        )

    def control_fn():
        order.append(("control_2",))
        return control if control is not None else ok_control()

    def preflight_fn():
        order.append(("preflight",))
        return {"ok": True, "error": None, "arms": {}, "closure": {}}

    kw = {
        "control_fn": control_fn,
        "preflight_fn": preflight_fn,
        "spawn": spawn,
        "measure_fn": measure_fn,
        "deadline": 1e9,
        "clock": clock or (lambda: 0.0),
        "sleep": lambda s: None,
        "log": lambda s: None,
        **extra,
    }
    return kw, spawned, order


def test_run_all_order_and_classification(tmp_path):
    m, calls = fake_measure({600: YES, 400: YES, 300: NO, 200: NO, 100: NO})
    kw, spawned, order = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert order[:2] == [("control_2",), ("preflight",)]
    assert [k for k, _ in spawned] == [
        k for k in (600, 400, 300, 200, 100) for _ in SEEDS
    ]
    assert [c[0] for c in calls] == [k for k in (600, 400, 300, 200, 100) for _ in SEEDS]
    assert {(c[2] - c[0], c[3]) for c in calls} == {(500, 64)}  # ckpt k+500, n=64
    assert res["resume_check"][300][1] == {"ok": True}
    v = run.verdict(res)
    assert (v["classification"], v["exit"], v["k_star"]) == (
        "AT_MEMORISATION",
        Exit.OK,
        400,
    )


def test_control_2_failure_stops_before_any_arm(tmp_path):
    m, calls = fake_measure()
    bad = {
        **run.ST.reproduction_control({"k": [1.0] * 3}, {"k": [2.0] * 3}),
        "error": None,
    }
    kw, spawned, order = _kw(m, control=bad)
    res = run.run_all(tmp_path, **kw)
    assert spawned == [] and calls == [] and order == [("control_2",)]
    assert res["not_run"] == {k: res["stopped"] for k in run.ARM_ORDER}
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN
    assert "control_2" in v["detail"]


def test_preflight_failure_trains_nothing(tmp_path):
    m, calls = fake_measure()
    kw, spawned, _ = _kw(m)
    kw["preflight_fn"] = lambda: {"ok": False, "error": "word 'x' not in the vocabulary"}
    res = run.run_all(tmp_path, **kw)
    assert spawned == [] and calls == []
    v = run.verdict(res)
    assert v["exit"] == Exit.DID_NOT_RUN and "vocabulary" in v["detail"]


def test_deadline_stops_the_arm_and_later_arms(tmp_path):
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    m, _ = fake_measure()
    kw, _, _ = _kw(m, upto={(400, 1): 800}, clock=clock)
    kw["deadline"] = 30.0
    res = run.run_all(tmp_path, **kw)
    assert res["arms"][400]["stopped"] == "deadline"
    assert res["not_run"] == {300: "deadline", 200: "deadline", 100: "deadline"}
    v = run.verdict(res)
    assert v["classification"] == "inconclusive"
    assert "arm k400 unreadable" in v["detail"] and "ckpt900" in v["detail"]
    assert "seed1" in v["detail"] and v["U"][600] is True


def test_failed_resume_check_is_inconclusive(tmp_path):
    m, _ = fake_measure()
    kw, _, _ = _kw(m, resume=False)
    v = run.verdict(run.run_all(tmp_path, **kw))
    assert v["classification"] == "inconclusive" and "control_1" in v["detail"]


def test_measurement_raised_stops_later_arms(tmp_path):
    def m(seed_dir, seed, label, n):
        if seed_dir.parent.name == "k400":
            raise RuntimeError("bad table")
        return fake_measure()[0](seed_dir, seed, label, n)

    kw, _, _ = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert "bad table" in res["error"] and sorted(res["arms"]) == [400, 600]
    assert res["not_run"][100] == "an earlier arm's measurement raised"
    assert run.verdict(res)["exit"] == Exit.DID_NOT_RUN


# --------------------------------------------------------------------------- #
# the ledger, RESULTS.md, the manifest, the wrapper
# --------------------------------------------------------------------------- #


def _start() -> dict:
    return {
        k: {
            "R_quantity": [0.0625, 0.125, 0.0625],
            "M_quantity": [0.5, 0.75, 0.5],
            "train.live.gap_2_to_M.answer_acc": [1.0, 1.0, 1.0],
            "heldout.live.gap_2_to_M.answer_acc": [0.25, 0.25, 0.25],
        }
        for k in KS
    }


def _ledger_doc(tmp_path, res: dict) -> dict:
    import ledger as ledger_mod

    led = ledger_mod.Ledger(run.RUN_ID, question=run.QUESTION, runs_root=tmp_path)
    led.run_meta(device="cpu", seeds_actually_run=SEEDS)
    led.status("ok")
    run.write_rows(led, res, run.verdict(res), _start())
    return led.doc


def _audit(tmp_path, doc, page) -> list:
    import render_scoreboard as rs

    (tmp_path / run.RUN_ID).mkdir(exist_ok=True)
    (tmp_path / run.RUN_ID / "ledger.json").write_text(json.dumps(doc))
    return rs.audit_prose(tmp_path, page)


def test_render_results_passes_the_audit(tmp_path):
    m, _ = fake_measure({600: YES, 400: NO, 300: YES, 200: NO, 100: NO})
    kw, _, _ = _kw(m)
    res = run.run_all(tmp_path, **kw)
    res["root"] = str(tmp_path)
    hb = tmp_path / "k600" / "seed0" / "heartbeat.jsonl"
    hb.write_text(
        "\n".join(
            json.dumps({"kind": "beat", "step": s, "loss_answer_tokens": 2.5})
            for s in range(700, 800)
        )
        + "\n"
    )
    doc = _ledger_doc(tmp_path, res)
    rows = {r["key"]: r for r in doc["rows"]}
    assert rows["classification"]["value"] == "AT_MEMORISATION"
    assert rows["k_star"]["value"] == 300 and rows["monotone"]["value"] is False
    assert rows["monotonicity_violations"]["value"] == [["k300", "k400"]]
    assert rows["k600.stream_answer_loss_first_window_below"]["value"]["seed0"] == 700
    assert rows["k600.ckpt1100.R_quantity"]["samples"] == [YES] * 3
    assert rows["k600.ckpt1100.R_holds"]["value"] is True
    assert rows["k400.ckpt900.R_holds"]["value"] is False
    assert rows["k100.start.R_quantity"]["value"] == [0.0625, 0.125, 0.0625]
    assert rows["k200.control_1.ok"]["value"] == [True] * 3
    assert rows["U"]["value"] == {
        "k100": False,
        "k200": False,
        "k300": True,
        "k400": False,
        "k600": True,
    }
    assert run.FS.CHECKPOINTS == {
        "A": (300, 1000, 1500, 2000, 2500, 3000),
        "B": (
            1500,
            2000,
            2500,
            3000,
        ),
    }
    page = run.render_results(doc, "none")
    assert "`classification` **AT_MEMORISATION**" in page
    assert _audit(tmp_path, doc, page) == []


def test_render_results_after_a_failed_control_passes_the_audit(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[3]
    ref[k] = [x + 2e-6 for x in ref[k]]
    c2 = run.control_2(ref, tmp_path, _measure_600(), log=lambda s: None)
    m, _ = fake_measure()
    kw, _, _ = _kw(m, control=c2)
    res = run.run_all(tmp_path, **kw)
    doc = _ledger_doc(tmp_path, res)
    page = run.render_results(doc, None)
    assert "**inconclusive**" in page and k in page
    assert _audit(tmp_path, doc, page) == []


def test_manifest_records_the_start_checkpoints():
    sha = {f"k{k}": {f"seed{s}": f"{k}{s}" for s in SEEDS} for k in KS}
    m = run.manifest(Path("/src"), 12, "abc", sha)
    assert m["start_checkpoint_sha256"] == sha
    assert m["stream"] == {"offset": 4160, "stride": 16, "vocab_documents": 64}
    assert m["stream_documents"]["k600"] == [4160 + 16 * 600, 4160 + 16 * 1100]
    assert m["stream_documents"]["k100"] == [5760, 4160 + 16 * 600]
    assert m["measured_checkpoint"] == {f"k{k}": k + 500 for k in KS}
    assert m["prereg_commit"] == "4249567" and m["deadline"] == run.DEADLINE
    assert m["expected"] == run.EXPECTED and m["arm_order"] == [600, 400, 300, 200, 100]
    assert json.dumps(m, default=str)


def test_dry_run_trains_and_measures_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ran")

    monkeypatch.setattr(run.CSC, "measure_checkpoint", boom)
    monkeypatch.setattr(loop, "train", boom)
    text = run.dry_run(tmp_path, now=0.0)
    assert "NOTHING TRAINED OR MEASURED" in text and text.count("exists=False") == 15
    assert text.count("repeats 0, disjoint True") == 5
    assert "start checkpoints REFUSED" in text


def test_run_outputs_are_gitignored_and_the_wrapper_writes_run_rc():
    for rel in (
        "runs/scaffold-dose/k600/seed0/ckpt-001100.pt",
        "runs/scaffold-dose/k100/seed2/heartbeat.jsonl",
        "runs/scaffold-dose/run.rc",
        "runs/scaffold-dose.parent.log",
    ):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        assert r.returncode == 0, rel
    for rel in (
        "runs/scaffold-dose/ledger.json",
        "runs/scaffold-dose/manifest.json",
        "runs/scaffold-dose/k600/seed0/result.json",
    ):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        assert r.returncode == 1, rel
    sh = ROOT / "experiments" / "scaffold-dose" / "run.sh"
    assert sh.stat().st_mode & 0o111
    text = sh.read_text()
    assert "rc=$?" in text and "runs/scaffold-dose/run.rc" in text
