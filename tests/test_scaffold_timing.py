"""Scaffold timing's own gates (`experiments/scaffold-timing/PREREG.md`, 167d650).

Nothing here reads a real checkpoint. The rule is fed synthetic S0-03 tables; the
run is driven by fake executors; the reproduction control is checked against the
REAL committed corpus-size ledger's key set. Mutation-battery gates
(`scripts/mutation_battery.py`, ``# --- scaffold-timing ---``):

* ``test_single_noisy_crossing_that_lapses_is_not_an_onset`` -- sustained dropped;
* ``test_reproduction_control_fails_at_2e_6`` -- the tolerance widened;
* ``test_classify_table`` -- the onset comparison swapped;
* ``test_R_and_M_need_every_seed`` -- "every seed" weakened to "any seed";
* ``test_measurement_is_the_corpus_size_function`` -- the measurement copied.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "scaffold_timing_run", ROOT / "experiments" / "scaffold-timing" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]
CK = run.CHECKPOINTS
YES, NO = 0.0625, 0.015625  # binary-exact, well either side of DELTA


def _prereg() -> dict:
    text = (ROOT / "experiments" / "scaffold-timing" / "PREREG.md").read_text()
    return yaml.safe_load(text.split("---")[1])


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed; the instrument, imported
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    p = _prereg()
    assert run.SEEDS == p["seeds"] == SEEDS
    assert [run.PRIMARY_ARM] == p["arms"]["primary"] == [64]
    assert list(run.SECONDARY_ARMS) == p["arms"]["secondary"] == [512, 4096]
    assert list(run.CHECKPOINTS) == p["checkpoints"]
    assert p["thresholds"]["DELTA"].startswith(f"{run.DELTA} ") and run.DELTA == 0.03
    t = p["thresholds"]["reproduction_tolerance"]
    assert t.startswith("1e-6 ") and run.REPRO_TOL == 1e-6
    assert "ckpt 300 and ckpt 1000" in t and run.CONTROL_CHECKPOINTS == (300, 1000)
    assert run.CSC.HELDOUT == (4096, 4160) and run.CSC.PROBE == (0, 64)
    assert "[4096, 4160)" in p["instrument"] and "[0, 64)" in p["instrument"]
    assert run.DEVICE == "cpu" and run.BUCKET == "gap_2_to_M"
    assert " ".join(p["decision_rule"].split()) == run.DECISION_RULE


def test_expected_is_the_preregs_verbatim():
    text = (ROOT / "experiments" / "scaffold-timing" / "PREREG.md").read_text()
    sec = text.split("## Author's expectation")[1].split("\n\n", 1)[1].split("\n\n")[0]
    assert " ".join(sec.split()) == run.EXPECTED


def test_measurement_is_the_corpus_size_function():
    """Identity, not a copy: the object the default executor calls is the corpus-size
    curve's ``measure_checkpoint``."""
    assert run.measure_checkpoint is run.CSC.measure_checkpoint
    code = run.measure_checkpoint.__code__.co_filename
    assert code.endswith("experiments/corpus-size-curve/run.py")
    default = inspect.signature(run.run_sequential).parameters["measure_fn"].default
    assert default is run.CSC.measure_checkpoint
    child = inspect.signature(run.child_main).parameters["measure_fn"].default
    assert child is run.CSC.measure_checkpoint
    src = (ROOT / "experiments" / "scaffold-timing" / "run.py").read_text()
    assert "def measure(" not in src and "def measure_checkpoint(" not in src
    assert "def doc_sets(" not in src and "def answer_readout(" not in src


# --------------------------------------------------------------------------- #
# synthetic S0-03 tables
# --------------------------------------------------------------------------- #


def table(r: float, m: float, *, base: float = 0.125, mem: float = 0.25) -> dict:
    """An S0-03 measure() result. At gap_2_to_M: heldout live = base + r, heldout
    slots_zeroed = base, train live = base + r + m, train slots_zeroed = train live
    - mem, train gate_zeroed = 0.5, gate_zeroed_bos_off = 0.375."""
    ds = {"n_documents": 64, "fraction_of_pairs_gap_gt_M": 0.1}
    for cond in run.S003.CONDITIONS:
        ds[cond] = {"real_token_nll": 1.5}
        for b in run.S003.BUCKETS:
            ds[cond][b] = {
                "n": 10,
                "answer_nll": run.CSC.CHANCE,
                "answer_acc": base,
                "answer_nll_over_16": run.CSC.CHANCE - 0.2,
                "answer_brier_over_16": 0.9,
            }
    ho = copy.deepcopy(ds)
    ho["live"]["gap_2_to_M"]["answer_acc"] = base + r
    tr = copy.deepcopy(ds)
    tr["live"]["gap_2_to_M"]["answer_acc"] = base + r + m
    tr["slots_zeroed"]["gap_2_to_M"]["answer_acc"] = base + r + m - mem
    tr["gate_zeroed"]["gap_2_to_M"]["answer_acc"] = 0.5
    tr["gate_zeroed_bos_off"]["gap_2_to_M"]["answer_acc"] = 0.375
    return {"heldout": ho, "train": tr, "memory_gates": [0.1]}


def per_arm(r_on: int | None, m_on: int | None, seedwise=None) -> dict:
    """R holds from ``r_on`` on, M from ``m_on`` on (None: never), every seed;
    ``seedwise[(s, c)] = (r, m)`` overrides one cell."""
    seedwise = seedwise or {}
    out = {}
    for s in SEEDS:
        out[s] = {}
        for c in CK:
            r = YES if r_on is not None and c >= r_on else NO
            m = YES if m_on is not None and c >= m_on else NO
            r, m = seedwise.get((s, c), (r, m))
            out[s][c] = {"stored_step": c, "s003": table(r, m)}
    return out


def full_per(per64: dict) -> dict:
    return {64: per64, 512: per_arm(None, None), 4096: per_arm(None, None)}


def ok_control() -> dict:
    return run.reproduction_control({"k": [1.0, 2.0, 3.0]}, {"k": [1.0, 2.0, 3.0]})


# --------------------------------------------------------------------------- #
# the onset and the classification
# --------------------------------------------------------------------------- #


def flags(true_at) -> dict[int, bool]:
    return {c: c in set(true_at) for c in CK}


def test_sustained_onset_all_true_is_the_first_checkpoint():
    assert run.sustained_onset(flags(CK)) == 100


def test_single_noisy_crossing_that_lapses_is_not_an_onset():
    # holds at 300 alone, then lapses; holds again from 700 on: the onset is 700
    assert run.sustained_onset(flags([300, 700, 800, 900, 1000])) == 700
    # holds 100..800, lapses at 900, holds at 1000: only the last checkpoint counts
    assert run.sustained_onset(flags([*range(100, 900, 100), 1000])) == 1000


def test_onset_absent_when_the_last_checkpoint_fails():
    assert run.sustained_onset(flags(range(100, 1000, 100))) is None
    assert run.sustained_onset(flags([])) is None


def test_onset_needs_every_checkpoint():
    f = flags(CK)
    del f[500]
    with pytest.raises(ValueError, match="500"):
        run.sustained_onset(f)


@pytest.mark.parametrize(
    ("onset_r", "onset_m", "want"),
    [
        (300, 500, "BEFORE"),
        (100, 1000, "BEFORE"),
        (500, 500, "WITH"),
        (700, 300, "AFTER"),
        (1000, 100, "AFTER"),
        (None, 300, "inconclusive"),
        (300, None, "inconclusive"),
        (None, None, "inconclusive"),
    ],
)
def test_classify_table(onset_r, onset_m, want):
    assert run.classify(onset_r, onset_m) == want


def test_holds_at_delta_exactly_and_just_below():
    assert run.holds([run.DELTA] * 3)
    assert not run.holds([run.DELTA - 1e-9] * 3)
    assert not run.holds([YES, YES])  # a missing seed is not "every seed"


def test_quantities_are_the_preregs():
    t = table(0.0625, 0.25, base=0.125, mem=0.1875)
    assert run.r_quantity(t) == 0.0625  # heldout live - heldout slots_zeroed
    assert run.m_quantity(t) == 0.25  # train live - heldout live
    assert run.train_memory_share(t) == 0.1875  # train live - train slots_zeroed
    assert run.side_channel_share(t) == 0.125  # train gate_zeroed - bos_off


def test_R_and_M_need_every_seed():
    per = per_arm(100, 100, seedwise={(1, 500): (NO, YES), (2, 600): (YES, NO)})
    p = run.primary(per)
    assert p["R_holds"][500] is False and p["R_holds"][400] is True
    assert p["M_holds"][600] is False and p["M_holds"][500] is True
    assert p["onset_R"] == 600 and p["onset_M"] == 700
    assert p["classification"] == "BEFORE"


def test_equal_onsets_give_WITH():
    p = run.primary(per_arm(400, 400))
    assert (p["onset_R"], p["onset_M"], p["classification"]) == (400, 400, "WITH")


def test_after_and_before_from_tables():
    assert run.primary(per_arm(600, 200))["classification"] == "AFTER"
    assert run.primary(per_arm(200, 600))["classification"] == "BEFORE"


def test_noisy_crossing_in_tables_does_not_move_the_onset():
    lapse = {(s, 300): (YES, YES) for s in SEEDS}
    p = run.primary(per_arm(800, 100, seedwise=lapse))
    assert p["R_holds"][300] is True and p["onset_R"] == 800
    assert p["classification"] == "AFTER"


def test_missing_onset_gives_inconclusive_exit_0():
    res = {"per": full_per(per_arm(None, 200)), "control": ok_control(), "error": None}
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.OK
    assert v["onset_R"] is None and v["onset_M"] == 200
    res["per"] = full_per(per_arm(300, None))
    assert run.verdict(res)["classification"] == "inconclusive"


def test_verdict_reads_the_n64_arm_only():
    res = {
        "per": {64: per_arm(500, 200), 512: per_arm(100, 900), 4096: per_arm(100, 900)},
        "control": ok_control(),
        "error": None,
    }
    v = run.verdict(res)
    assert (v["onset_R"], v["onset_M"], v["classification"]) == (500, 200, "AFTER")


# --------------------------------------------------------------------------- #
# the reproduction control
# --------------------------------------------------------------------------- #

REF = {"a.x": [0.5, 0.25, 0.125], "a.y": [1.0, 2.0, 3.0]}


def test_reproduction_control_exact_passes():
    c = run.reproduction_control(REF, copy.deepcopy(REF))
    assert c["ok"] and all(p["n_keys_compared"] == 2 for p in c["per_seed"].values())
    assert c["tolerance"] == 1e-6


def test_reproduction_control_fails_at_2e_6():
    got = copy.deepcopy(REF)
    got["a.y"][1] += 2e-6
    c = run.reproduction_control(REF, got)
    assert not c["ok"]
    assert c["per_seed"][1]["ok"] is False
    assert c["per_seed"][0]["ok"] and c["per_seed"][2]["ok"]
    (f,) = c["per_seed"][1]["failures"]
    assert f["key"] == "a.y" and f["reference"] == 2.0 and f["measured"] == got["a.y"][1]


def test_reproduction_control_passes_inside_the_tolerance():
    got = copy.deepcopy(REF)
    got["a.x"][0] += 5e-7
    assert run.reproduction_control(REF, got)["ok"]


def test_reproduction_control_missing_key_or_nan_fails():
    got = {"a.x": list(REF["a.x"])}
    c = run.reproduction_control(REF, got)
    assert not c["ok"] and c["missing_keys"] == ["a.y"]
    got = copy.deepcopy(REF)
    got["a.x"][2] = math.nan
    assert not run.reproduction_control(REF, got)["ok"]
    assert not run.reproduction_control({}, {})["ok"]  # nothing compared is no pass


def _control_per64() -> dict:
    return {
        s: {
            c: {"stored_step": c, "s003": table(0.01 * (s + 1), 0.1 * c / 1000)}
            for c in run.CONTROL_CHECKPOINTS
        }
        for s in SEEDS
    }


def test_remeasured_rows_name_every_reference_key():
    """The corpus-size curve's own write_rows, fed the re-measured tables, produces
    exactly the statistic keys the committed ledger holds under n64.ckpt300/1000."""
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    ref = run.reference_rows(doc)
    got = run.remeasured_rows(_control_per64())
    assert sorted(got) == sorted(ref)
    assert all(len(v) == 3 for v in got.values())


def test_arm_rows_agree_with_the_corpus_size_naming():
    """Our every-checkpoint rows use the corpus-size ledger's names and values."""
    per64 = _control_per64()
    col = run._Rows()
    run.arm_rows(col, 64, per64)
    ours = {
        r["key"]: r["samples"]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(run.CONTROL_PREFIXES)
    }
    theirs = run.remeasured_rows(per64)
    assert {k: ours[k] for k in theirs} == theirs


# --------------------------------------------------------------------------- #
# the run: control first, stop on failure
# --------------------------------------------------------------------------- #


class FakeExec:
    def __init__(self, per: dict, raise_at=None):
        self.per, self.raise_at, self.calls = per, raise_at, []

    def __call__(self, jobs, source_root):
        self.calls.append(list(jobs))
        out = {}
        for j in jobs:
            if j == self.raise_at:
                return out, f"boom at {j}"
            n, s, c = j
            out[j] = self.per[n][s][c]
        return out, None


def _reference_for(per64) -> dict:
    return run.remeasured_rows(per64)


def test_run_measures_the_control_first_and_stops_on_failure():
    per = full_per(per_arm(500, 200))
    ref = _reference_for(per[64])
    k = sorted(ref)[0]
    ref[k] = [x + 2e-6 for x in ref[k]]
    ex = FakeExec(per)
    res = run.run(ex, Path("/nowhere"), ref, log=lambda s: None)
    assert ex.calls == [run.control_jobs()]  # nothing further measured
    assert res["stopped"] == "reproduction control failed"
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


def test_run_passes_the_control_then_measures_everything():
    per = full_per(per_arm(500, 200))
    ex = FakeExec(per)
    res = run.run(ex, Path("/nowhere"), _reference_for(per[64]), log=lambda s: None)
    assert ex.calls[0] == run.control_jobs() and ex.calls[1] == run.main_jobs()
    assert res["control"]["ok"] and run.complete(res["per"]) == []
    assert len(res["jobs_run"]) == 90
    v = run.verdict(res)
    assert (v["classification"], v["exit"]) == ("AFTER", Exit.OK)


@pytest.mark.parametrize("phase", ["control", "main"])
def test_measurement_raised_is_exit_3(phase):
    per = full_per(per_arm(500, 200))
    at = (64, 1, 1000) if phase == "control" else (512, 2, 700)
    ex = FakeExec(per, raise_at=at)
    res = run.run(ex, Path("/nowhere"), _reference_for(per[64]), log=lambda s: None)
    assert len(ex.calls) == (1 if phase == "control" else 2)
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN
    assert "boom" in v["detail"]


def test_incomplete_or_absent_control_is_exit_3():
    per = full_per(per_arm(500, 200))
    del per[4096][2][900]
    v = run.verdict({"per": per, "control": ok_control(), "error": None})
    assert v["exit"] == Exit.DID_NOT_RUN and "n4096.seed2.ckpt900" in v["detail"]
    v = run.verdict({"per": full_per(per_arm(500, 200)), "control": None, "error": None})
    assert v["exit"] == Exit.DID_NOT_RUN


def test_job_order_is_the_preregs():
    assert run.control_jobs() == [(64, s, c) for s in SEEDS for c in (300, 1000)]
    mj = run.main_jobs()
    assert len(run.control_jobs()) + len(mj) == 3 * 3 * 10
    assert not set(mj) & set(run.control_jobs())
    assert [n for n, _, _ in mj] == sorted(n for n, _, _ in mj)  # N=64 first


def test_run_sequential_stops_at_the_first_raise(tmp_path):
    seen = []

    def fake(seed_dir, s, c, n):
        seen.append((n, s, c))
        if c == 200:
            raise RuntimeError("bad")
        return {"n": n}

    jobs = [(64, 0, 100), (64, 0, 200), (64, 0, 300)]
    out, err = run.run_sequential(jobs, tmp_path, fake, log=lambda s: None)
    assert seen == jobs[:2] and list(out) == [(64, 0, 100)] and "bad" in err
    # the checkpoint path handed over is the corpus-size layout
    got = []
    run.run_sequential(
        [(512, 2, 700)], tmp_path, lambda d, s, c, n: got.append(d) or {}, log=print
    )
    assert got == [tmp_path / "n512" / "seed2"]
    assert (
        run.ckpt_file(tmp_path, (512, 2, 700)) == tmp_path / "n512/seed2/ckpt-000700.pt"
    )


def test_parallel_executor_round_trips_through_children(tmp_path):
    """One child per seed; each child's JSON results come back exactly."""

    def fake_measure(seed_dir, s, c, n):
        return {"stored_step": c, "x": 0.1 + s / 7 + c / 13 + n}

    class P:
        def __init__(self, argv, env):
            a = argv[argv.index("--seed") + 1]
            jobs = run._parse_jobs(int(a), argv[argv.index("--jobs") + 1])
            out = Path(argv[argv.index("--out") + 1])
            assert env["OMP_NUM_THREADS"] == str(run.THREADS_PER_CHILD)
            self.rc = int(run.child_main(int(a), jobs, tmp_path, out, 1, fake_measure))

        def wait(self):
            return self.rc

    ex = run.ParallelExecutor(tmp_path / "children", spawn=P)
    jobs = run.control_jobs()
    out, err = ex(jobs, tmp_path)
    assert err is None and sorted(out) == sorted(jobs)
    for n, s, c in jobs:
        assert out[(n, s, c)] == fake_measure(None, s, c, n)
    assert [c["seed"] for c in ex.commands] == SEEDS


def test_manifest_hashes_every_checkpoint_read(tmp_path):
    for j in run.control_jobs() + run.main_jobs():
        f = run.ckpt_file(tmp_path, j)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(repr(j).encode())
    m = run.manifest(tmp_path, "sequential", 12, "abc", {"provenance": {"git_sha": "e"}})
    shas = m["checkpoint_sha256"]
    assert len(shas) == 90
    assert (
        shas["n512/seed2/ckpt-000700.pt"] == hashlib.sha256(b"(512, 2, 700)").hexdigest()
    )
    assert m["reference_ledger_sha256"] == "abc" and m["expected"] == run.EXPECTED


def test_dry_run_measures_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("measured")

    monkeypatch.setattr(run.CSC, "measure_checkpoint", boom)
    monkeypatch.setattr(run.S003, "measure", boom)
    text = run.dry_run(tmp_path, parallel=False)
    assert "NOTHING MEASURED" in text and "total 90 measurements" in text
    assert text.count("exists=False") == 90
    assert "536 statistic keys" in text


# --------------------------------------------------------------------------- #
# the ledger rows and RESULTS.md
# --------------------------------------------------------------------------- #


def _ledger_doc(tmp_path, res: dict) -> dict:
    import ledger as ledger_mod

    led = ledger_mod.Ledger(run.RUN_ID, question=run.QUESTION, runs_root=tmp_path)
    v = run.verdict(res)
    led.run_meta(device="cpu", seeds_actually_run=SEEDS)
    led.status("ok")
    run.write_rows(led, res, v)
    return led.doc


def test_render_results_passes_the_audit(tmp_path):
    import render_scoreboard as rs

    per = full_per(per_arm(500, 200))
    per[512][2][900]["s003"] = table(YES, NO)
    res = run.run(FakeExec(per), Path("/x"), _reference_for(per[64]), log=lambda s: None)
    doc = _ledger_doc(tmp_path, res)
    rows = {r["key"]: r for r in doc["rows"]}
    assert rows["classification"]["value"] == "AFTER"
    assert rows["onset_R"]["value"] == 500 and rows["onset_M"]["value"] == 200
    assert rows["next_run_arms"]["value"] == "arms A and B"
    assert rows["secondary_R_reaches_delta"]["value"] == ["n512.ckpt900.seed2"]
    assert rows["n512.ckpt900.R_reaches_delta_on_seeds"]["value"] == [2]
    assert rows["reproduction_control.n_keys_compared"]["value"] == [536] * 3
    (tmp_path / run.RUN_ID).mkdir(exist_ok=True)
    (tmp_path / run.RUN_ID / "ledger.json").write_text(json.dumps(doc))
    page = run.render_results(doc, "none")
    assert "`classification` **AFTER**" in page and "not measured" not in page
    assert rs.audit_prose(tmp_path, page) == []


def test_render_results_after_a_failed_control_passes_the_audit(tmp_path):
    import render_scoreboard as rs

    per = full_per(per_arm(500, 200))
    ref = _reference_for(per[64])
    k = sorted(ref)[3]
    ref[k] = [x + 2e-6 for x in ref[k]]
    res = run.run(FakeExec(per), Path("/x"), ref, log=lambda s: None)
    doc = _ledger_doc(tmp_path, res)
    (tmp_path / run.RUN_ID).mkdir(exist_ok=True)
    (tmp_path / run.RUN_ID / "ledger.json").write_text(json.dumps(doc))
    page = run.render_results(doc, None)
    assert "**inconclusive**" in page and k in page
    assert rs.audit_prose(tmp_path, page) == []
