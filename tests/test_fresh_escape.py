"""Fresh escape's own gates (`experiments/fresh-escape/PREREG.md`, d5b9a23).

No test trains or measures a real checkpoint: the parent loop is driven by fake
children over checkpoint files touched on disk and a fake clock, the rule is fed
synthetic S0-03 tables, and the child's ``train()`` call is recorded, not run.
Mutation-battery gates (`scripts/mutation_battery.py`, ``# --- fresh-escape ---``):

* ``test_classification_table`` -- STIRRING and NO_ESCAPE swapped;
* ``test_P_is_the_largest_checkpoint_on_every_seed`` -- P the smallest, not largest;
* ``test_P_below_6000_is_inconclusive`` -- the 6000 minimum dropped;
* ``test_control_1_fails_at_2e_6`` -- the measurement-path tolerance widened;
* ``test_stream_disjointness_catches_overlap`` -- control 3 drops the held-out;
* ``test_start_checkpoint_sha_mismatch_refuses`` -- the sha256 comparison dropped;
* ``test_stirring_window_must_end_at_or_before_P`` -- windows after P counted;
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
        "fresh_escape_run", ROOT / "experiments" / "fresh-escape" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]
CKPTS = (4000, 5000, 6000, 7000, 8000, 9000)
YES, NO = 0.0625, 0.015625  # binary-exact, either side of DELTA
C_YES = 0.875
FLAT, LOW = 2.78125, 2.5  # stream answer loss: at the plateau / below 2.6726


def _prereg_text() -> str:
    return (ROOT / "experiments" / "fresh-escape" / "PREREG.md").read_text()


def _prereg() -> dict:
    return yaml.safe_load(_prereg_text().split("---")[1])


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed; the instrument, imported
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    p = _prereg()
    t = p["thresholds"]
    assert run.SEEDS == p["seeds"] == SEEDS
    assert list(run.CHECKPOINTS) == p["checkpoints_measured"] == list(CKPTS)
    assert run.MIN_PRIMARY_CKPT == t["MIN_PRIMARY_CKPT"] == 6000
    assert t["DELTA"].startswith("0.03 ") and run.DELTA == 0.03
    assert t["C"].startswith("0.8875 ") and round(run.C_THRESHOLD, 4) == 0.8875
    assert t["stream_loss_threshold"].startswith("ln 16 - 0.10 = 2.6726 ")
    assert round(run.STREAM_LOSS_THRESHOLD, 4) == 2.6726
    assert run.STREAM_LOSS_WINDOW == 100 and run.REPRO_TOL == 1e-6
    assert "within 1e-6" in _prereg_text()
    assert t["deadline"].startswith("2026-09-26 08:30 America/Los_Angeles, absolute")
    want = dt.datetime(2026, 9, 26, 8, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert run.deadline_epoch() == want.timestamp()
    assert {int(k[4:]): v for k, v in p["start_checkpoint_sha256"].items()} == (
        run.START_SHA256
    )
    assert "3 seeds as parallel children x 5 threads" in p["arm"]
    assert run.THREADS_PER_SEED == 5 and run.DEVICE == "cpu"
    assert "train steps 3000 -> 9000" in p["arm"]
    assert (run.RESUME_STEP, run.END_STEP) == (3000, 9000)
    assert "(model, optimizer, policy, RNG)" in p["arm"]
    assert "n = 64" in p["instrument"] and run.N_TRAIN == 64
    assert "held-out [4096, 4160), probe [0, 64)" in p["instrument"]
    assert (run.HELDOUT, run.PROBE) == ((4096, 4160), (0, 64))
    src = ".worktrees/fresh-stream/runs/fresh-stream/A/seed<s>/ckpt-003000.pt"
    assert src in p["arm"]
    assert str(run.start_ckpt(2)).endswith(src.replace("<s>", "2"))
    assert run.PREREG_COMMIT == "d5b9a23"


def test_decision_rule_and_expected_are_the_preregs_verbatim():
    assert " ".join(_prereg()["decision_rule"].split()) == run.DECISION_RULE
    sec = _prereg_text().split("## Author's expectation")[1].split("\n\n", 1)[1]
    assert " ".join(sec.split("\n\n")[0].split()) == run.EXPECTED


def test_reading_is_the_preregs_table():
    got = {}
    for line in _prereg_text().splitlines():
        if line.startswith("| ") and "**" in line:
            name, reading = line.split("|")[2].strip().split(" — ", 1)
            got[name.strip("*").removesuffix(" by P")] = reading
    assert got == run.READING


def test_measurement_is_the_corpus_size_function():
    """Identity, not a copy: the object the parent calls is the corpus-size curve's
    ``measure_checkpoint``, the same object fresh-stream calls."""
    assert run.measure_checkpoint is run.CSC.measure_checkpoint
    assert run.measure_checkpoint is run.FS.measure_checkpoint
    assert run.measure_checkpoint.__code__.co_filename.endswith(
        "experiments/corpus-size-curve/run.py"
    )
    src = (ROOT / "experiments" / "fresh-escape" / "run.py").read_text()
    assert "def measure(" not in src and "def measure_checkpoint(" not in src
    assert "def doc_sets(" not in src and "def run_arm(" not in src
    assert "measure_fn=deadline_stamped(measure_checkpoint, deadline_epoch())" in src
    assert "reference, a.source_root, measure_checkpoint, log=timestamped_log" in src


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("p", "r", "stirs", "want"),
    [
        (None, None, False, "inconclusive"),
        (4000, True, True, "inconclusive"),
        (5000, True, False, "inconclusive"),
        (6000, True, False, "ESCAPES"),
        (9000, True, True, "ESCAPES"),
        (6000, False, True, "STIRRING"),
        (9000, False, True, "STIRRING"),
        (7000, False, False, "NO_ESCAPE"),
        (9000, False, False, "NO_ESCAPE"),
    ],
)
def test_classification_table(p, r, stirs, want):
    assert run.classify(p, r, stirs) == want


def table(r: float = NO, brier: float = C_YES, *, base: float = 0.125) -> dict:
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


def per_for(upto: int = 9000, r_at=None, cells=None) -> dict:
    """``{seed: {c: measurement}}`` for every listed c <= upto; ``r_at[c]`` sets R
    at c (default NO); ``cells[(seed, c)]`` overrides one seed."""
    r_at, cells = r_at or {}, cells or {}
    return {
        s: {
            c: {"stored_step": c, "s003": table(cells.get((s, c), r_at.get(c, NO)))}
            for c in CKPTS
            if c <= upto
        }
        for s in SEEDS
    }


def ok_control() -> dict:
    c = run.ST.reproduction_control({"k": [1.0, 2.0, 3.0]}, {"k": [1.0, 2.0, 3.0]})
    return {**c, "error": None}


def res_for(per: dict, windows=None, **over) -> dict:
    res = {
        "control_1": ok_control(),
        "preflight": {"ok": True, "error": None},
        "arm": {"per": per, "error": None, "stopped": None},
        "resume_check": {s: {"ok": True} for s in SEEDS},
        "stream_windows": windows or {s: {3000: FLAT} for s in SEEDS},
        "not_run": None,
        "error": None,
    }
    res.update(over)
    return res


def test_P_is_the_largest_checkpoint_on_every_seed():
    per = per_for(9000)
    assert run.primary_checkpoint(per) == 9000
    del per[1][9000]
    assert run.primary_checkpoint(per) == 8000
    del per[2][8000]  # a hole: 7000 is still on every seed, 8000 is not
    assert run.primary_checkpoint(per) == 7000
    assert run.primary_checkpoint(per_for(3000)) is None
    assert run.primary_checkpoint({}) is None
    v = run.verdict(res_for(per))
    assert v["P"] == 7000 and v["reported_as"] == "NO_ESCAPE by 7000"


def test_P_below_6000_is_inconclusive():
    for upto in (3000, 4000, 5000):
        v = run.verdict(res_for(per_for(upto, r_at={4000: YES, 5000: YES})))
        assert (v["classification"], v["exit"]) == ("inconclusive", Exit.DID_NOT_RUN)
        assert "P = " in v["detail"]
    per = per_for(6000, r_at={6000: YES})
    del per[0][6000]  # 6000 not on every seed: P = 5000
    assert run.verdict(res_for(per))["classification"] == "inconclusive"
    v = run.verdict(res_for(per_for(6000, r_at={6000: YES})))
    assert (v["classification"], v["exit"], v["P"]) == ("ESCAPES", Exit.OK, 6000)


@pytest.mark.parametrize(
    ("r_at", "windows", "want", "reported"),
    [
        ({9000: YES}, None, "ESCAPES", "ESCAPES"),
        ({c: YES for c in CKPTS}, {0: {3000: LOW}}, "ESCAPES", "ESCAPES"),
        ({}, {1: {8800: LOW}}, "STIRRING", "STIRRING"),
        ({8000: YES}, {2: {3000: LOW}}, "STIRRING", "STIRRING"),
        ({}, None, "NO_ESCAPE", "NO_ESCAPE by 9000"),
    ],
    ids=["escapes", "escapes_also_stirring", "stirring", "stirring_R_before_P", "none"],
)
def test_verdict_every_row_from_tables(r_at, windows, want, reported):
    wins = {s: {3000: FLAT, **((windows or {}).get(s) or {})} for s in SEEDS}
    v = run.verdict(res_for(per_for(9000, r_at=r_at), wins))
    assert (v["classification"], v["reported_as"], v["exit"]) == (want, reported, Exit.OK)
    assert v["P"] == 9000 and want in v["detail"]


def test_R_needs_every_seed_and_reads_P_only():
    per = per_for(9000, r_at={9000: YES}, cells={(1, 9000): NO})
    v = run.verdict(res_for(per))
    assert v["R_at_P"] is False and v["classification"] == "NO_ESCAPE"
    assert run.verdict(res_for(per_for(9000, r_at={9000: run.DELTA})))["R_at_P"]
    v = run.verdict(res_for(per_for(9000, r_at={8000: YES})))  # R at 8000, not at P
    assert v["readout"][8000]["R"] and v["classification"] == "NO_ESCAPE"


def test_stirring_window_must_end_at_or_before_P():
    """ckpt-P holds steps [0, P): the window [P - 100, P) counts, [P, P + 100) not."""
    per = per_for(7000)
    wins = {s: {6800: FLAT, 6900: FLAT, 7000: LOW} for s in SEEDS}
    v = run.verdict(res_for(per, wins))
    assert v["classification"] == "NO_ESCAPE" and v["reported_as"] == "NO_ESCAPE by 7000"
    wins[2][6900] = LOW
    v = run.verdict(res_for(per, wins))
    assert v["classification"] == "STIRRING"
    assert v["stirring"]["first_window_below"] == {0: None, 1: None, 2: 6900}
    assert run.windows_at_or_before({6900: 1.0, 7000: 2.0}, 7000) == {6900: 1.0}
    # the threshold itself is not below it
    wins = {s: {3000: run.STREAM_LOSS_THRESHOLD} for s in SEEDS}
    assert run.verdict(res_for(per, wins))["classification"] == "NO_ESCAPE"


@pytest.mark.parametrize(
    "case",
    [
        "control_1_failed",
        "control_1_absent",
        "control_1_raised",
        "preflight_failed",
        "preflight_absent",
        "resume_failed",
        "resume_missing",
        "measurement_raised",
    ],
)
def test_failed_controls_are_inconclusive(case):
    res = res_for(per_for(9000, r_at={9000: YES}))
    if case == "control_1_failed":
        res["control_1"] = {
            **run.ST.reproduction_control({"k": [1.0] * 3}, {"k": [2.0] * 3}),
            "error": None,
        }
    elif case == "control_1_absent":
        res["control_1"] = None
    elif case == "control_1_raised":
        res["control_1"] = {"ok": False, "error": "OSError: gone", "per_seed": {}}
    elif case == "preflight_failed":
        res["preflight"] = {"ok": False, "error": "word 'x' not in the vocabulary"}
    elif case == "preflight_absent":
        res["preflight"] = None
    elif case == "resume_failed":
        res["resume_check"][1] = {"ok": False}
    elif case == "resume_missing":
        res["resume_check"][2] = None
    elif case == "measurement_raised":
        res["error"] = "RuntimeError: bad table"
    v = run.verdict(res)
    assert (v["classification"], v["exit"]) == ("inconclusive", Exit.DID_NOT_RUN)
    assert v["P"] == 9000  # every reason is named, not only the P rule


# --------------------------------------------------------------------------- #
# the deadline, literally: P counts measurements COMPLETED by 08:30 only
# --------------------------------------------------------------------------- #


def _at(offset_s: float) -> str:
    """An ISO time ``offset_s`` seconds from the PREREG deadline."""
    t = run.deadline_epoch() + offset_s
    return dt.datetime.fromtimestamp(t, ZoneInfo("America/Los_Angeles")).isoformat()


def test_post_deadline_measurement_is_excluded_from_P():
    per = per_for(9000, r_at={9000: YES})
    for s in SEEDS:
        for c in CKPTS:
            per[s][c]["measured_at"] = _at(-3600 + c / 1000)
    per[1][9000]["measured_at"] = _at(+1)  # completed one second after 08:30
    v = run.verdict(res_for(per))
    assert v["P"] == 8000 and v["classification"] == "NO_ESCAPE"
    assert v["reported_as"] == "NO_ESCAPE by 8000"
    per[1][9000]["measured_at"] = _at(0)  # completed exactly at 08:30: counts
    v = run.verdict(res_for(per))
    assert (v["P"], v["classification"]) == (9000, "ESCAPES")
    per[1][9000]["post_deadline"] = True  # the wrapper's flag also excludes
    assert run.verdict(res_for(per))["P"] == 8000
    del per[1][9000]["s003"]  # not measured before the deadline: no table
    per[1][9000]["post_deadline"] = False
    assert run.verdict(res_for(per))["P"] == 8000


def test_P_below_6000_under_the_deadline_filter():
    per = per_for(6000, r_at={6000: YES})
    per[0][6000]["measured_at"] = _at(+60)
    v = run.verdict(res_for(per))
    assert (v["P"], v["classification"], v["exit"]) == (
        5000,
        "inconclusive",
        Exit.DID_NOT_RUN,
    )
    per[0][6000]["measured_at"] = _at(-60)
    v = run.verdict(res_for(per))
    assert (v["P"], v["classification"]) == (6000, "ESCAPES")


def test_deadline_stamped_stamps_and_refuses_to_start_after_the_deadline():
    t = {"now": 0.0}
    calls = []

    def clock():
        return t["now"]

    def measure(seed_dir, seed, label, n):
        calls.append(label)
        t["now"] += 5.0  # the measurement takes 5 s
        return {"stored_step": label, "s003": table()}

    logs = []
    m = run.deadline_stamped(measure, 10.0, clock=clock, log=logs.append)
    a = m(Path("d"), 0, 4000, 64)  # 0 -> 5: before
    assert a["post_deadline"] is False and a["measured_at"] and a["s003"]
    assert dt.datetime.fromisoformat(a["measured_at"]).tzinfo is not None
    t["now"] = 8.0
    b = m(Path("d"), 0, 5000, 64)  # 8 -> 13: completed after
    assert b["post_deadline"] is True and "s003" in b
    t["now"] = 10.5
    c = m(Path("d"), 0, 6000, 64)  # starts after: refused, never measured
    assert calls == [4000, 5000] and "s003" not in c
    assert c["not_measured"] == "not measured before deadline" and c["post_deadline"]
    assert "not measured before deadline" in logs[0]
    per = {0: {4000: a, 5000: b, 6000: c}, 1: {}, 2: {}}
    assert run.counted_before_deadline(per)[0] == {4000: a}
    assert run.post_deadline(per)[0] == {5000: b, 6000: c}


def test_run_all_under_the_stamped_deadline(tmp_path):
    """Each measurement takes two clock ticks; with the deadline at tick 30 the 14th
    completes after it and later ones are never started: P = 7000, and the late
    measurements are secondary ledger rows."""
    clock = _ticking()
    m, _ = fake_measure({c: YES for c in CKPTS})
    kw, _, _ = _kw(m, clock=clock)
    kw["measure_fn"] = run.deadline_stamped(m, 30.0, clock=clock, log=lambda s: None)
    kw["deadline"] = 30.0
    res = run.run_all(tmp_path / "r", **kw)
    v = run.verdict(res)
    assert (v["P"], v["classification"]) == (7000, "ESCAPES")
    assert res["arm"]["per"][0][8000]["post_deadline"] is False
    assert res["arm"]["per"][1][8000]["post_deadline"] is True
    assert res["arm"]["per"][2][9000]["not_measured"] == "not measured before deadline"
    doc = _ledger_doc(tmp_path, res)
    rows = {r["key"]: r for r in doc["rows"]}
    late = rows["post_deadline_measurements"]["value"]
    assert late["seed1"]["8000"]["post_deadline"] is True
    assert late["seed1"]["8000"]["R_quantity"] == YES
    assert late["seed2"]["9000"]["not_measured"] == "not measured before deadline"
    assert rows["A.ckpt8000.seeds_measured"]["value"] == [0]  # seed 1 is late
    assert "A.ckpt9000.seeds_measured" not in rows  # never a primary row
    assert rows["P"]["value"] == 7000
    assert rows["measured_at"]["value"]["seed0"]["4000"] is not None


def test_the_parent_log_is_timestamped(capsys):
    run.timestamped_log("hello")
    line = capsys.readouterr().out.strip()
    stamp, rest = line.split(" ", 1)
    assert rest == "hello" and dt.datetime.fromisoformat(stamp).tzinfo is not None
    src = (ROOT / "experiments" / "fresh-escape" / "run.py").read_text()
    assert "log=timestamped_log," in src


# --------------------------------------------------------------------------- #
# control 1: the measurement path
# --------------------------------------------------------------------------- #


def _control_per() -> dict:
    return {
        s: {3000: {"stored_step": 3000, "s003": table(0.01 * s, 1.25)}} for s in SEEDS
    }


def _measure_3000(calls=None):
    def m(seed_dir, seed, label, n):
        if calls is not None:
            calls.append((seed_dir, seed, label, n))
        return _control_per()[seed][label]

    return m


def test_control_1_reference_keys_are_the_fresh_stream_ledgers():
    """fresh-stream's own arm_rows, fed ckpt-3000 tables, produces exactly the
    statistic keys its committed ledger holds under A.ckpt3000."""
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    ref = run.reference_rows(doc)
    got = run.remeasured_rows(_control_per())
    assert sorted(got) == sorted(ref) and len(ref) > 200
    assert all(k.startswith("A.ckpt3000.") for k in ref)
    assert "A.ckpt3000.R_quantity" in ref and "A.ckpt3000.C_quantity" in ref
    assert run.FS.CHECKPOINTS["A"] == (300, 1000, 1500, 2000, 2500, 3000)


def test_control_1_measures_ckpt_3000_of_every_seed_from_the_source(tmp_path):
    calls = []
    ref = run.remeasured_rows(_control_per())
    c = run.control_1(ref, tmp_path, _measure_3000(calls), log=lambda s: None)
    assert c["ok"] and c["error"] is None
    assert calls == [(tmp_path / "A" / f"seed{s}", s, 3000, 64) for s in SEEDS]


def test_control_1_within_1e_6_passes(tmp_path):
    ref = run.remeasured_rows(_control_per())
    near = copy.deepcopy(ref)
    near[sorted(near)[5]][1] += 5e-7
    near[sorted(near)[9]][0] -= 9e-7
    assert run.control_1(near, tmp_path, _measure_3000(), log=lambda s: None)["ok"]


def test_control_1_fails_at_2e_6(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[7]
    ref[k][2] += 2e-6
    c = run.control_1(ref, tmp_path, _measure_3000(), log=lambda s: None)
    assert not c["ok"] and c["tolerance"] == 1e-6
    assert c["per_seed"][2]["ok"] is False and c["per_seed"][0]["ok"]
    assert c["per_seed"][2]["failures"][0]["key"] == k


def test_control_1_raise_or_missing_key_fails(tmp_path):
    def boom(*a):
        raise RuntimeError("unreadable checkpoint")

    ref = run.remeasured_rows(_control_per())
    c = run.control_1(ref, tmp_path, boom, log=lambda s: None)
    assert not c["ok"] and "unreadable checkpoint" in c["error"]
    ref["A.ckpt3000.not_a_key"] = [0.0, 0.0, 0.0]
    c = run.control_1(ref, tmp_path, _measure_3000(), log=lambda s: None)
    assert not c["ok"] and c["missing_keys"] == ["A.ckpt3000.not_a_key"]


# --------------------------------------------------------------------------- #
# control 3: the stream ids, the closure, the start checkpoints
# --------------------------------------------------------------------------- #


def test_stream_ids_are_the_preregs_and_disjoint():
    ids = run.stream_ids()
    assert ids == list(range(4160 + 16 * 3000, 4160 + 16 * 9000))
    assert (ids[0], ids[-1] + 1) == (52160, 148160)
    for t in (3000, 5555, 8999):
        assert list(run.STREAM.window(t, run.BATCH)) == list(
            range(4160 + 16 * t, 4160 + 16 * t + 16)
        )
    d = run.stream_disjointness()
    assert d["ok"] and d["repeats"] == 0 and d["n_ids"] == 96000
    assert (d["first_id"], d["last_id"]) == (52160, 148159)
    assert not (d["ids_in_probe"] or d["ids_in_heldout"] or d["ids_in_vocab_documents"])


def test_stream_disjointness_catches_overlap(monkeypatch):
    from rsr.train.loop import DocumentStream

    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=0, stride=16))
    monkeypatch.setattr(run, "RESUME_STEP", 250)  # ids from 4000: the held-out
    d = run.stream_disjointness()
    assert not d["ok"] and d["ids_in_heldout"][0] == 4096 and not d["ids_in_probe"]
    monkeypatch.setattr(run, "RESUME_STEP", 0)  # ids from 0: the probe
    d = run.stream_disjointness()
    assert not d["ok"] and d["ids_in_probe"][0] == 0
    assert d["ids_in_vocab_documents"][0] == 0
    monkeypatch.setattr(run, "RESUME_STEP", 3000)
    with pytest.raises(ValueError, match="overlap"):  # a repeating stream is refused
        run.STREAM.__class__(offset=4160, stride=8).window(3000, run.BATCH)
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=4160, stride=16))
    assert run.stream_disjointness()["ok"]


def test_preflight_stops_on_overlap_and_on_a_closure_failure(monkeypatch):
    seen = []
    monkeypatch.setattr(run, "vocabulary_closure", lambda s: seen.append(s) or {"ok": 1})
    assert run.preflight()["ok"] and seen == SEEDS

    def bad(s):
        raise ValueError("word 'the_harbour' is not in the vocabulary")

    monkeypatch.setattr(run, "vocabulary_closure", bad)
    pf = run.preflight()
    assert not pf["ok"] and "the_harbour" in pf["error"]
    monkeypatch.setattr(run, "stream_disjointness", lambda: {"ok": False})
    pf = run.preflight()
    assert not pf["ok"] and "not disjoint" in pf["error"]


def test_the_closure_is_fresh_streams_over_exactly_this_runs_ids(monkeypatch):
    got = {}

    def fake(seed, iters):
        ws = run.FS.stream_windows(iters)
        got[seed] = (iters, ws[0], ws[-1], len(ws))
        return {"ok": True}

    monkeypatch.setattr(run.FS, "vocabulary_closure", fake)
    real_windows = run.FS.stream_windows
    run.vocabulary_closure(1)
    assert got[1] == (
        9000,
        run.STREAM.window(3000, 16),
        run.STREAM.window(8999, 16),
        6000,
    )
    assert run.FS.stream_windows is real_windows
    monkeypatch.undo()
    monkeypatch.setattr(run, "END_STEP", 3002)  # a real closure over 32 documents
    c = run.vocabulary_closure(0)
    assert c["ok"] and c["n_stream_documents"] == 32


def _fake_sources(tmp_path) -> tuple[Path, dict]:
    src = tmp_path / "src"
    shas = {}
    for s in SEEDS:
        f = run.start_ckpt(s, src)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(f"seed{s}".encode())
        shas[s] = run.sha256_file(f)
    return src, shas


def test_start_checkpoint_sha_mismatch_refuses(tmp_path, monkeypatch):
    src, shas = _fake_sources(tmp_path)
    monkeypatch.setattr(run, "START_SHA256", shas)
    got = run.verify_start_checkpoints(src, step_fn=lambda f: 3000)
    assert got == {f"seed{s}": shas[s] for s in SEEDS}
    run.start_ckpt(1, src).write_bytes(b"not the checkpoint fresh-stream wrote")
    with pytest.raises(run.StartCheckpointRefused, match=r"seed1/ckpt-003000\.pt sha256"):
        run.verify_start_checkpoints(src, step_fn=lambda f: 3000)
    # the real front matter refuses these fake files outright
    monkeypatch.undo()
    with pytest.raises(run.StartCheckpointRefused, match="not the PREREG's"):
        run.verify_start_checkpoints(src, step_fn=lambda f: 3000)


def test_start_checkpoint_absent_or_wrong_step_refuses(tmp_path, monkeypatch):
    src, shas = _fake_sources(tmp_path)
    monkeypatch.setattr(run, "START_SHA256", shas)
    with pytest.raises(run.StartCheckpointRefused, match="does not store step 3000"):
        run.verify_start_checkpoints(src, step_fn=lambda f: 2900)
    run.start_ckpt(0, src).unlink()
    with pytest.raises(run.StartCheckpointRefused, match="is absent"):
        run.verify_start_checkpoints(src, step_fn=lambda f: 3000)


# --------------------------------------------------------------------------- #
# the child: fresh-stream's resume child from arm A's ckpt 3000
# --------------------------------------------------------------------------- #


def test_single_is_fresh_streams_resume_child_from_arm_A(tmp_path, monkeypatch):
    """The corpus-size N = 64 ``train()`` call + the stream + the resume from arm A's
    ``ckpt-003000.pt``, ending at 9000; fresh-stream's names restored after."""
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
    real_start = run.FS.start_ckpt
    run.FS.single("A", 1, tmp_path, iters=9000)  # fresh-stream arm A's call
    r = run.single(1, tmp_path, source_root=tmp_path / "src")
    arm_a, got = seen
    want = tmp_path / "src" / "A" / "seed1" / "ckpt-003000.pt"
    assert arm_a["iters"] == 9000 and arm_a["stream"] == run.STREAM
    assert got == {**arm_a, "resume": str(want)}  # arm A's call, plus the resume
    assert checks == [(want, tmp_path / "resume_check.json")]
    assert run.FS.start_ckpt is real_start and run.FS.RESUME_STEP == 1000
    assert r["arm"] == "A" and r["resume_check"] == {"ok": True}


def test_resume_check_is_fresh_streams_exact_check():
    assert run.FS.ResumeCheck.check.__code__.co_filename.endswith(
        "experiments/fresh-stream/run.py"
    )
    src = (ROOT / "experiments" / "fresh-escape" / "run.py").read_text()
    assert "class ResumeCheck" not in src and "def compare_to_checkpoint" not in src


def test_child_argv(tmp_path):
    argv = run.child_argv(2, tmp_path / "o", tmp_path / "s")
    assert argv[1].endswith("experiments/fresh-escape/run.py")
    assert argv[2:] == [
        "--single",
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
    """Writes every listed ckpt up to ``upto``, a heartbeat of ``loss`` over steps
    [3000, upto) and resume_check.json, then exits -- or stays alive if ``alive``."""

    def __init__(self, d: Path, upto=9000, loss=FLAT, alive=False, resume=True):
        d.mkdir(parents=True, exist_ok=True)
        for c in CKPTS:
            if c <= upto:
                run.RC.ckpt_path(d, c).write_bytes(b"x")
        (d / "heartbeat.jsonl").write_text(
            "".join(
                json.dumps({"kind": "beat", "step": t, "loss_answer_tokens": loss}) + "\n"
                for t in range(3000, upto)
            )
        )
        if resume is not None:
            (d / "resume_check.json").write_text(json.dumps({"ok": resume}))
        self.argv = ["py", "run.py", "--single"]
        self.alive = alive

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0


def fake_measure(r_at=None):
    calls = []

    def m(seed_dir, seed, label, n):
        calls.append((seed_dir.name, seed, label, n))
        return {"stored_step": label, "s003": table((r_at or {}).get(label, NO))}

    return m, calls


def _kw(measure_fn, *, control=None, child=None, clock=None):
    spawned, order = [], []

    def spawn(s, d):
        spawned.append(s)
        order.append(("spawn", s))
        return FakeChild(d, **((child or {}).get(s) or {}))

    def control_fn():
        order.append(("control_1",))
        return control if control is not None else ok_control()

    def preflight_fn():
        order.append(("preflight",))
        return {"ok": True, "error": None, "closure": {}}

    kw = {
        "control_fn": control_fn,
        "preflight_fn": preflight_fn,
        "spawn": spawn,
        "measure_fn": measure_fn,
        "deadline": 1e9,
        "clock": clock or (lambda: 0.0),
        "sleep": lambda s: None,
        "log": lambda s: None,
    }
    return kw, spawned, order


def test_run_all_order_and_classification(tmp_path):
    m, calls = fake_measure({9000: YES})
    kw, spawned, order = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert order[:2] == [("control_1",), ("preflight",)] and spawned == SEEDS
    assert sorted({c[2] for c in calls}) == list(CKPTS)
    assert {(c[0], c[3]) for c in calls} == {(f"seed{s}", 64) for s in SEEDS}
    assert res["resume_check"][2] == {"ok": True}
    assert res["stream_windows"][0][8900] == FLAT
    v = run.verdict(res)
    assert (v["classification"], v["exit"], v["P"]) == ("ESCAPES", Exit.OK, 9000)


def test_run_all_stirring_from_the_heartbeats(tmp_path):
    m, _ = fake_measure()
    kw, _, _ = _kw(m, child={1: {"loss": LOW}})
    v = run.verdict(run.run_all(tmp_path, **kw))
    assert v["classification"] == "STIRRING"
    assert v["stirring"]["first_window_below"] == {0: None, 1: 3000, 2: None}


def test_control_1_failure_trains_nothing(tmp_path):
    m, calls = fake_measure()
    bad = {
        **run.ST.reproduction_control({"k": [1.0] * 3}, {"k": [2.0] * 3}),
        "error": None,
    }
    kw, spawned, order = _kw(m, control=bad)
    res = run.run_all(tmp_path, **kw)
    assert spawned == [] and calls == [] and order == [("control_1",)]
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN
    assert "control_1" in v["detail"]


def test_preflight_failure_trains_nothing(tmp_path):
    m, calls = fake_measure()
    kw, spawned, _ = _kw(m)
    kw["preflight_fn"] = lambda: {"ok": False, "error": "word 'x' not in the vocabulary"}
    res = run.run_all(tmp_path, **kw)
    assert spawned == [] and calls == []
    v = run.verdict(res)
    assert v["exit"] == Exit.DID_NOT_RUN and "vocabulary" in v["detail"]


def _ticking():
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    return clock


def test_deadline_stops_the_children_and_P_is_what_exists(tmp_path):
    m, _ = fake_measure()
    kw, _, _ = _kw(
        m,
        child={1: {"upto": 7000, "alive": True}, 2: {"upto": 8000, "alive": True}},
        clock=_ticking(),
    )
    kw["deadline"] = 30.0
    res = run.run_all(tmp_path, **kw)
    assert res["stopped"] == "deadline"
    v = run.verdict(res)
    assert (v["classification"], v["reported_as"], v["P"]) == (
        "NO_ESCAPE",
        "NO_ESCAPE by 7000",
        7000,
    )


def test_deadline_before_6000_is_inconclusive(tmp_path):
    m, _ = fake_measure({4000: YES, 5000: YES})
    kw, _, _ = _kw(m, child={0: {"upto": 5000, "alive": True}}, clock=_ticking())
    kw["deadline"] = 30.0
    v = run.verdict(run.run_all(tmp_path, **kw))
    assert (v["classification"], v["exit"], v["P"]) == (
        "inconclusive",
        Exit.DID_NOT_RUN,
        5000,
    )


def test_failed_resume_check_is_inconclusive(tmp_path):
    m, _ = fake_measure({9000: YES})
    kw, _, _ = _kw(m, child={0: {"resume": False}})
    v = run.verdict(run.run_all(tmp_path, **kw))
    assert v["classification"] == "inconclusive" and "control_2" in v["detail"]


def test_measurement_raised_is_inconclusive(tmp_path):
    def m(seed_dir, seed, label, n):
        if label == 8000:
            raise RuntimeError("bad table")
        return fake_measure()[0](seed_dir, seed, label, n)

    kw, _, _ = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert "bad table" in res["error"]
    v = run.verdict(res)
    assert v["exit"] == Exit.DID_NOT_RUN and "a measurement raised" in v["detail"]


# --------------------------------------------------------------------------- #
# the ledger, RESULTS.md, the manifest, the wrapper
# --------------------------------------------------------------------------- #


def _ledger_doc(tmp_path, res: dict) -> dict:
    import ledger as ledger_mod

    led = ledger_mod.Ledger(run.RUN_ID, question=run.QUESTION, runs_root=tmp_path)
    led.run_meta(device="cpu", seeds_actually_run=SEEDS)
    led.status("ok")
    run.write_rows(led, res, run.verdict(res))
    return led.doc


def _audit(tmp_path, doc, page) -> list:
    import render_scoreboard as rs

    (tmp_path / run.RUN_ID).mkdir(exist_ok=True)
    (tmp_path / run.RUN_ID / "ledger.json").write_text(json.dumps(doc))
    return rs.audit_prose(tmp_path, page)


def test_render_results_passes_the_audit(tmp_path):
    m, _ = fake_measure({8000: YES})
    kw, _, _ = _kw(m, child={2: {"loss": LOW}})
    res = run.run_all(tmp_path / "r", **kw)
    doc = _ledger_doc(tmp_path, res)
    rows = {r["key"]: r for r in doc["rows"]}
    assert rows["classification"]["value"] == "STIRRING"
    assert rows["P"]["value"] == 9000 and rows["R_at_P"]["value"] is False
    assert rows["A.ckpt8000.R_holds"]["value"] is True
    assert rows["A.ckpt9000.R_quantity"]["samples"] == [NO] * 3
    assert rows["stream_answer_loss_first_window_below_at_or_before_P"]["value"] == {
        "seed0": None,
        "seed1": None,
        "seed2": 3000,
    }
    assert rows["control_2.ok"]["value"] == [True] * 3
    assert run.FS.CHECKPOINTS["A"] == (300, 1000, 1500, 2000, 2500, 3000)
    page = run.render_results(doc, "none")
    assert "`classification_reported_as` **STIRRING**" in page
    assert _audit(tmp_path, doc, page) == []


def test_render_results_after_a_failed_control_passes_the_audit(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[3]
    ref[k] = [x + 2e-6 for x in ref[k]]
    c1 = run.control_1(ref, tmp_path, _measure_3000(), log=lambda s: None)
    m, _ = fake_measure()
    kw, _, _ = _kw(m, control=c1)
    res = run.run_all(tmp_path / "r", **kw)
    doc = _ledger_doc(tmp_path, res)
    page = run.render_results(doc, None)
    assert "**inconclusive**" in page and k in page
    assert _audit(tmp_path, doc, page) == []


def test_manifest_records_the_start_checkpoints():
    sha = {f"seed{s}": f"{s}" for s in SEEDS}
    m = run.manifest(Path("/src"), 12, "abc", sha)
    assert m["start_checkpoint_sha256"] == sha
    assert m["stream"] == {"offset": 4160, "stride": 16, "vocab_documents": 64}
    assert m["stream_documents"] == [52160, 148160]
    assert m["checkpoints_measured"] == list(CKPTS) and m["min_primary_ckpt"] == 6000
    assert m["prereg_commit"] == "d5b9a23" and m["deadline"] == run.DEADLINE
    assert m["expected"] == run.EXPECTED
    assert json.dumps(m, default=str)


def test_dry_run_trains_and_measures_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ran")

    monkeypatch.setattr(run.CSC, "measure_checkpoint", boom)
    monkeypatch.setattr(loop, "train", boom)
    text = run.dry_run(tmp_path, now=0.0)
    assert "NOTHING TRAINED OR MEASURED" in text and text.count("exists=False") == 3
    assert "repeats 0, disjoint True" in text and "52160..148159" in text
    assert "start checkpoints REFUSED" in text
    assert list(tmp_path.iterdir()) == []


def test_run_outputs_are_gitignored_and_the_wrapper_writes_run_rc():
    for rel in (
        "runs/fresh-escape/A/seed0/ckpt-009000.pt",
        "runs/fresh-escape/A/seed2/heartbeat.jsonl",
        "runs/fresh-escape/run.rc",
        "runs/fresh-escape.parent.log",
    ):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        assert r.returncode == 0, rel
    for rel in ("runs/fresh-escape/ledger.json", "runs/fresh-escape/manifest.json"):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        assert r.returncode == 1, rel
    sh = ROOT / "experiments" / "fresh-escape" / "run.sh"
    assert sh.stat().st_mode & 0o111
    text = sh.read_text()
    assert "rc=$?" in text and "runs/fresh-escape/run.rc" in text
