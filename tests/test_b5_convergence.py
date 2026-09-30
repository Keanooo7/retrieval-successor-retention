"""B5 substrate convergence: its gates (`experiments/b5-convergence/PREREG.md`, dfda530).

No test trains or measures a real checkpoint. The rule is fed synthetic window means,
the protection guard and the resume selection work on tmp directories, and the copy
reads files written here.

Mutation-battery gates (`scripts/mutation_battery.py`, ``# --- b5-convergence ---``):

* ``test_output_dir_inside_protected_dir_refused`` -- the protection guard dropped;
* ``test_start_copy_refuses_sha_mismatch`` -- the start sha256 comparison dropped;
* ``test_common_step_needs_every_seed`` -- E* read on any seed, not every seed;
* ``test_rule_reads_the_confidence_interval_not_the_point`` -- the CI replaced by the
  point estimate;
* ``test_resume_picks_the_latest_own_checkpoint`` -- the earliest checkpoint resumed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "b5_convergence_run", ROOT / "experiments" / "b5-convergence" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# the PREREG, transcribed
# --------------------------------------------------------------------------- #


def test_constants_are_the_prereg():
    assert run.PREREG_COMMIT == "dfda530"
    assert run.SEEDS == SEEDS
    assert run.ARM == "B"
    assert run.RESUME_STEP == 3000
    assert run.CAP == 9000
    assert run.WINDOW == 250
    assert run.N_WINDOWS == 16
    assert run.REL_BOUND == 1.0
    assert run.THREADS_PER_SEED == 4
    assert run.CKPT_EVERY == run.WINDOW
    assert tuple(range(7000, 9001, 250)) == run.EVAL_STEPS
    assert run.TRAJECTORY == (4000, 5000, 6000, 7000, 8000, 9000)
    assert run.DELTA == 0.03


def test_t_quantile_is_student_t_14_df():
    """P(T <= t) = 0.975 for 14 df: Simpson's rule on the Student-t density."""
    nu, t, n = 14, run.T_975_14, 20000
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2))
    c /= math.sqrt(nu * math.pi)

    def pdf(x):
        return c * (1 + x * x / nu) ** (-(nu + 1) / 2)

    h = t / n
    inner = sum((4 if i % 2 else 2) * pdf(i * h) for i in range(1, n))
    area = (pdf(0) + pdf(t) + inner) * h / 3
    assert abs(0.5 + area - 0.975) < 1e-10


def test_measurement_is_the_fresh_stream_function():
    assert run.measure_checkpoint is run.FS.measure_checkpoint


# --------------------------------------------------------------------------- #
# the plateau rule
# --------------------------------------------------------------------------- #


def _noise(i: int) -> float:
    # deterministic, zero-mean-ish, bounded: a stand-in for window noise
    return 0.004 * math.sin(1.7 * i + 0.3)


def test_ols_exact_line_has_zero_se_and_right_slope():
    # y = 1.0 - 0.02 * x, x in thousands of steps, windows 0.25 apart
    ys = [1.0 - 0.02 * (0.25 * i + 0.125) for i in range(16)]
    r = run.ols_rule(ys)
    assert r["slope_per_1000"] == pytest.approx(-0.02, abs=1e-12)
    assert r["se"] == pytest.approx(0.0, abs=1e-12)
    assert r["rel_slope_pct"] == pytest.approx(-0.02 / r["mean"] * 100, rel=1e-12)


def test_ols_matches_a_hand_computation():
    ys = [0.8 + _noise(i) - 0.001 * i for i in range(16)]
    x = [0.25 * i for i in range(16)]
    mx, my = sum(x) / 16, sum(ys) / 16
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (y - my) for a, y in zip(x, ys, strict=True)) / sxx
    rss = sum((y - my - b * (a - mx)) ** 2 for a, y in zip(x, ys, strict=True))
    se = math.sqrt(rss / 14 / sxx)
    r = run.ols_rule(ys)
    assert sxx == pytest.approx(21.25)
    assert r["slope_per_1000"] == pytest.approx(b, rel=1e-12)
    assert r["se"] == pytest.approx(se, rel=1e-12)
    assert r["rel_ci_pct"][0] == pytest.approx(
        100 * (b - 2.1447866879 * se) / my, rel=1e-9
    )
    assert r["rel_ci_pct"][1] == pytest.approx(
        100 * (b + 2.1447866879 * se) / my, rel=1e-9
    )


def test_rule_needs_sixteen_windows():
    with pytest.raises(ValueError):
        run.ols_rule([1.0] * 15)


def test_rule_reads_the_confidence_interval_not_the_point():
    # A point slope well inside +/-1 %, but noise wide enough that the CI crosses -1 %.
    ys = [0.8 - 0.0008 * i + (0.02 if i % 2 else -0.02) for i in range(16)]
    r = run.ols_rule(ys)
    assert abs(r["rel_slope_pct"]) < 1.0
    assert r["rel_ci_pct"][0] < -1.0 or r["rel_ci_pct"][1] > 1.0
    assert r["holds"] is False
    # and a genuinely flat, quiet series holds
    flat = run.ols_rule([0.8 + _noise(i) * 0.1 for i in range(16)])
    assert flat["holds"] is True


def test_steep_decline_does_not_hold():
    ys = [0.9 - 0.05 * (0.25 * i) + _noise(i) * 0.1 for i in range(16)]
    assert run.ols_rule(ys)["holds"] is False


def _windows(fn) -> dict[int, float]:
    """``{w0: fn(w0)}`` for every 250-window from 3000 to 9000."""
    return {w0: fn(w0) for w0 in range(3000, 9000, 250)}


def test_rule_at_reads_exactly_the_sixteen_windows_before_E():
    seen = {}

    def fn(w0):
        return 0.8 + 1e-3 * _noise(w0 // 250)

    wins = _windows(fn)
    wins[2750] = 99.0  # before B5's own steps: must never be read
    r = run.rule_at(wins, 7000)
    assert r is not None and r["windows"] == list(range(3000, 7000, 250))
    seen["7000"] = r
    assert run.rule_at(wins, 6750) is None  # fewer than 16 own windows
    del wins[5000]
    assert run.rule_at(wins, 7000) is None  # a missing window: no evaluation


def test_windows_are_own_steps_only_and_complete(tmp_path):
    hb = tmp_path / "heartbeat.jsonl"
    lines = [json.dumps({"kind": "header", "start_step": 2750})]
    for t in range(2750, 3600):
        lines.append(json.dumps({"kind": "beat", "step": t, "loss_answer_tokens": 1.0}))
    hb.write_text("\n".join(lines) + "\n")
    w = run.stream_windows(hb)
    assert sorted(w) == [3000, 3250]  # 2750 is before B5; 3500 incomplete


def test_common_step_needs_every_seed():
    good = _windows(lambda w0: 0.8 + 1e-4 * _noise(w0 // 250))
    steep = _windows(lambda w0: 0.9 - 0.00005 * w0)
    by_seed = {0: good, 1: good, 2: steep}
    trace = run.rule_trace(by_seed)
    assert trace[7000][0]["holds"] and not trace[7000][2]["holds"]
    assert run.common_step(trace) is None
    by_seed[2] = good
    assert run.common_step(run.rule_trace(by_seed)) == 7000


def test_common_step_is_the_first_common_E():
    def late(w0):
        # steep until 5000, flat after: the rule holds once [E-4000, E) is all flat
        return (
            0.8 + (0.00002 * (5000 - w0) if w0 < 5000 else 0.0) + 1e-4 * _noise(w0 // 250)
        )

    early = _windows(lambda w0: 0.8 + 1e-4 * _noise(w0 // 250))
    trace = run.rule_trace({0: early, 1: _windows(late), 2: early})
    e = run.common_step(trace)
    assert e is not None and e > 7000
    assert all(trace[e][s]["holds"] for s in SEEDS)
    assert not all(trace[e - 250][s]["holds"] for s in SEEDS)


# --------------------------------------------------------------------------- #
# the classification
# --------------------------------------------------------------------------- #


def test_classification_table():
    c = run.classify
    assert c(e_star=7500, r_at_e=True, reached_cap=False) == "PLATEAU"
    assert c(e_star=7500, r_at_e=False, reached_cap=True) == "PLATEAU_WITHOUT_RETRIEVAL"
    assert c(e_star=None, r_at_e=None, reached_cap=True) == "NO_PLATEAU"
    # NO_PLATEAU only at the cap; E* found but not yet measured is in progress
    assert c(e_star=None, r_at_e=None, reached_cap=False) == "IN_PROGRESS"
    assert c(e_star=8000, r_at_e=None, reached_cap=False) == "IN_PROGRESS"


# --------------------------------------------------------------------------- #
# protection, the start copy, resume
# --------------------------------------------------------------------------- #


def test_protected_dirs_come_from_the_t0_manifest(tmp_path):
    man = tmp_path / "MANIFEST.sha256"
    man.write_text(
        "aa  .worktrees/fresh-stream/runs/fresh-stream/B/seed0/ckpt-003000.pt\n"
        "bb  .worktrees/lookahead-room/runs/lookahead-room/raw/x.D.pt\n"
    )
    repo = tmp_path / "repo"
    prot = run.protected_dirs(run.t0_entries(man), repo)
    assert repo / ".worktrees/fresh-stream/runs/fresh-stream/B/seed0" in prot
    assert repo / ".worktrees/lookahead-room/runs/lookahead-room/raw" in prot
    assert repo / ".worktrees/fresh-stream" in prot


def test_output_dir_inside_protected_dir_refused(tmp_path):
    prot = {tmp_path / "fs" / "B" / "seed0", tmp_path / "wt"}
    with pytest.raises(run.ProtectedPath):
        run.assert_writable(tmp_path / "fs" / "B" / "seed0", prot)
    with pytest.raises(run.ProtectedPath):
        run.assert_writable(tmp_path / "wt" / "runs" / "b5", prot)
    run.assert_writable(tmp_path / "runs" / "b5-convergence" / "B" / "seed0", prot)


def _fake_ckpt(path: Path, step: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "step": step, "model": {}, "rng": None}, path)
    return path


def test_start_copy_refuses_sha_mismatch(tmp_path):
    src = _fake_ckpt(tmp_path / "src" / "ckpt-003000.pt", 3000)
    dest = tmp_path / "dest"
    with pytest.raises(run.StartCheckpointRefused):
        run.copy_start(src, dest, expected_sha="0" * 64)
    assert not (dest / "ckpt-003000.pt").exists()
    good = run.sha256_file(src)
    out = run.copy_start(src, dest, expected_sha=good)
    assert out == dest / "ckpt-003000.pt" and run.sha256_file(out) == good


def test_start_copy_refuses_wrong_stored_step(tmp_path):
    src = _fake_ckpt(tmp_path / "src" / "ckpt-003000.pt", 2999)
    with pytest.raises(run.StartCheckpointRefused):
        run.copy_start(src, tmp_path / "dest", expected_sha=run.sha256_file(src))


def test_resume_picks_the_latest_own_checkpoint(tmp_path):
    d = tmp_path / "seed0"
    for s in (3000, 3250, 3500):
        _fake_ckpt(d / f"ckpt-{s:06d}.pt", s)
    (d / ".ckpt-003750.pt.tmp.123").write_bytes(b"partial")  # an interrupted write
    assert run.latest_ckpt(d) == d / "ckpt-003500.pt"
    assert run.latest_ckpt(tmp_path / "empty") is None


def test_child_env_sets_every_thread_variable():
    from orchestrator.lanes import THREAD_ENV_VARS

    env = run.child_env(4)
    assert all(env[v] == "4" for v in THREAD_ENV_VARS)
    argv = run.child_argv(1, Path("/x/seed1"), threads=4, iters=9000, ckpt_every=250)
    assert argv[argv.index("--threads") + 1] == "4"
    assert argv[argv.index("--seed") + 1] == "1"
    assert argv[argv.index("--iters") + 1] == "9000"
    assert argv[argv.index("--ckpt-every") + 1] == "250"


def test_compare_payloads_is_exact():
    a = {
        "step": 3000,
        "model": {"w": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {0: {"m": torch.tensor([0.5])}}},
        "policy": {},
        "rng": {
            "cpu": torch.tensor([1, 2], dtype=torch.uint8),
            "python": (3, (1, 2), None),
        },
    }
    b = {**a, "model": {"w": torch.tensor([1.0, 2.0])}}
    assert run.compare_payloads(a, b)["equal"] is True
    c = {**a, "model": {"w": torch.tensor([1.0, 2.0 + 2**-20])}}
    r = run.compare_payloads(a, c)
    assert r["equal"] is False and any("model" in m for m in r["mismatches"])
    d = {**a, "rng": {**a["rng"], "cpu": torch.tensor([1, 3], dtype=torch.uint8)}}
    assert run.compare_payloads(a, d)["equal"] is False


def test_report_key_ignores_the_step_counter():
    """The report and ledger update per completed window / measurement, not per poll."""
    base = {
        "windows": {s: {3000: 0.8} for s in SEEDS},
        "per": {s: {} for s in SEEDS},
        "trace": {},
        "e_star": None,
        "classification": "IN_PROGRESS",
        "reached": {s: 3300 for s in SEEDS},
    }
    moved = {**base, "reached": {s: 3310 for s in SEEDS}}
    assert run.report_key(base) == run.report_key(moved)
    window = {**base, "windows": {**base["windows"], 1: {3000: 0.8, 3250: 0.79}}}
    assert run.report_key(base) != run.report_key(window)
    measured = {**base, "per": {**base["per"], 0: {4000: {}}}}
    assert run.report_key(base) != run.report_key(measured)
