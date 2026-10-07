"""`experiments/rsr-run2/run.py` -- RSR run 2's harness (PREREG rsr-run2 @ a406d39).

The decision logic and the controls are tested on hand-built inputs; the pipeline end to
end is the launcher smoke (scratch output only, never runs/rsr-run2). The load-bearing
ones:

* §2's per-seed label at every boundary, and the class on hand-built label tables;
* C14 fails on a tampered referendum vector, a tampered pinned checkpoint or a wrong
  config; C15 fails on a tampered FIFO vector, a wrong registered sha or a (t, gap)
  mismatch with ES;
* §5's age-1 share on a synthetic victim log, and the newcomer label at its thresholds;
* the command the harness checks is the launcher's, token for token, and run 1's
  module (as run 1's own tests load it) is not touched by run 2's rebinding.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "rsr-run2" / "run.py"


@pytest.fixture(scope="module")
def r2():
    spec = importlib.util.spec_from_file_location("rsr_run2_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rsr_run2_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_pass(d: Path, name: str, recs: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return p


def _recs(n=3, seed=0, flip=None):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ans = [
            [t, rng.randint(1, 30), rng.random() < 0.7, rng.random()] for t in (5, 9, 20)
        ]
        out.append(
            {
                "doc": 995000 + i,
                "answers": ans,
                "real_sum": 1.0,
                "real_n": 10,
                "victims": {"16": 0},
                "evictions": [[16, 16, 0]],
                "hits": [[3, True]],
                "n_live_before": [0, 1],
            }
        )
    if flip is not None:
        out[flip]["answers"][0][3] += 1e-9
    return out


# --------------------------------------------------------------------------- #
# §2: the label and the class
# --------------------------------------------------------------------------- #

D = 0.038


@pytest.mark.parametrize(
    ("point", "lo", "hi", "want"),
    [
        (D, 1e-12, 0.05, "WIN"),  # point == +delta, lo just above 0
        (D - 1e-12, 0.001, 0.05, "UNRESOLVED"),  # point just below delta
        (0.05, 0.0, 0.06, "UNRESOLVED"),  # lo == 0 is not > 0
        (-D, -0.05, -1e-12, "LOSS"),  # point == -delta, hi just below 0
        (-0.05, -0.06, 0.0, "UNRESOLVED"),  # hi == 0 is not < 0
        (0.0, -D + 1e-12, D - 1e-12, "EQUIV"),
        (0.0, -D, 0.01, "UNRESOLVED"),  # lo == -delta: not strictly within
        (0.0, -0.01, D, "UNRESOLVED"),  # hi == delta: not strictly within
        (0.02, 0.001, 0.03, "EQUIV"),  # CI above 0 but within (-d, d): EQUIV, not WIN
        (float("nan"), 0.1, 0.2, "UNRESOLVED"),
        (0.1, None, 0.2, "UNRESOLVED"),
        (0.1, 0.05, 0.2, "WIN"),
    ],
)
def test_seed_label_boundaries(r2, point, lo, hi, want):
    assert r2.seed_label(point, lo, hi, D) == want


def test_labels_are_mutually_exclusive_as_the_prereg_says(r2):
    """§2: WIN needs point >= delta > hi under EQUIV; LOSS symmetric. Brute force over a
    grid: no (point, lo, hi) with lo <= point <= hi satisfies two definitions."""
    vals = [-0.1, -D, -0.02, -1e-9, 0.0, 1e-9, 0.02, D, 0.1]
    for lo, pt, hi in itertools.product(vals, repeat=3):
        if not lo <= pt <= hi:
            continue
        hits = [
            lo > 0 and pt >= D,
            hi < 0 and pt <= -D,
            lo > -D and hi < D,
        ]
        assert sum(hits) <= 1, (lo, pt, hi)


@pytest.mark.parametrize(
    ("labels", "want"),
    [
        ({0: "WIN", 1: "WIN", 2: "WIN"}, "BEATS-FIFO"),
        ({0: "WIN", 1: "WIN", 2: "EQUIV"}, "MIXED"),
        ({0: "WIN", 1: "WIN", 2: "UNRESOLVED"}, "MIXED"),
        ({0: "WIN", 1: "WIN", 2: "LOSS"}, "FAILS"),
        ({0: "LOSS", 1: "EQUIV", 2: "EQUIV"}, "FAILS"),
        ({0: "EQUIV", 1: "EQUIV", 2: "EQUIV"}, "MIXED"),
        ({0: "WIN", 1: "WIN"}, "MIXED"),  # a missing seed is never a WIN
        ({}, "MIXED"),
    ],
)
def test_class_rule_on_hand_built_tables(r2, labels, want):
    assert r2.run_class(labels) == want
    assert r2.CLASS_ROUTE[want]


def test_delta_is_machine_read_and_checked_against_the_prereg_table(r2, tmp_path):
    led = tmp_path / "ledger.json"
    rows = [
        {"key": r2.DELTA_KEY.format(s=s), "value": {"0.9": v, "0.0": v}}
        for s, v in r2.prereg_delta_table(r2.REFS.prereg).items()
    ]
    led.write_text(json.dumps({"rows": rows}))
    refs = r2.Refs(b2_ledger=str(led))
    for s in (0, 1, 2):
        d = r2.read_delta(s, refs)
        assert d["ok"] and d["delta"] == d["registered"] > 0
    rows[1]["value"]["0.9"] += 1e-12
    led.write_text(json.dumps({"rows": rows}))
    assert not r2.read_delta(1, refs)["ok"]
    assert r2.prereg_delta_table(r2.REFS.prereg) == {
        0: 0.03802185560447721,
        1: 0.03675691427360389,
        2: 0.03185756551141167,
    }


# --------------------------------------------------------------------------- #
# §5: the age-1 share and the newcomer label
# --------------------------------------------------------------------------- #


def test_age1_share_on_a_synthetic_victim_log(r2):
    recs = [
        {"evictions": [[16, 1, 15], [17, 16, 0], [18, 1, 15], [19, 3, 13]]},
        {"evictions": [[16, 2, 14]]},
        {"evictions": []},
    ]
    a = r2.age1_share(recs)
    assert (a["n_age1"], a["n_evictions"]) == (2, 5) and a["share"] == 2 / 5
    assert r2.age1_share([{"evictions": []}])["share"] is None


@pytest.mark.parametrize(
    ("shares", "want"),
    [
        ({0: 0.05, 1: 0.0, 2: 0.01}, "NEWCOMER-RESOLVED"),  # <= 0.05 inclusive
        ({0: 0.0500001, 1: 0.0, 2: 0.01}, "NEWCOMER-REDUCED"),
        ({0: 0.20, 1: 0.0, 2: 0.0}, "NEWCOMER-PERSISTS"),  # >= 0.20 inclusive
        ({0: 0.1999, 1: 0.1, 2: 0.06}, "NEWCOMER-REDUCED"),
        ({0: 0.01, 1: None, 2: 0.01}, "NEWCOMER-REDUCED"),  # missing: not RESOLVED
        ({0: 0.38, 1: None, 2: None}, "NEWCOMER-PERSISTS"),
    ],
)
def test_newcomer_label_thresholds(r2, shares, want):
    assert r2.newcomer_label(shares) == want


# --------------------------------------------------------------------------- #
# §5: the probe statistics
# --------------------------------------------------------------------------- #


def _decisions(n_docs=4, per_doc=5, K=6, sign=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for d in range(n_docs):
        for j in range(per_doc):
            cos = torch.rand(K, generator=g, dtype=torch.float64)
            psi = sign * cos * 3.0 + 1.0
            age = torch.arange(K, 0, -1)
            out.append((d, 16 + j, psi, cos, age, int(cos.argmax())))
    return out


def test_probe_stats_by_hand(r2):
    W = torch.ones(7, 4, dtype=torch.float64)  # every replicate = the sample itself
    st = r2.probe_stats(_decisions(sign=1.0), 4, W, m=6)
    assert st["rho_lower_median"]["point"] == pytest.approx(1.0)
    assert st["rho_lower_median"]["lo"] == pytest.approx(1.0)
    assert st["victim_cos_is_max_share"]["point"] == 1.0
    for b in st["by_age_bucket"].values():
        assert b["pearson_pct"]["point"] == pytest.approx(1.0)
    st = r2.probe_stats(_decisions(sign=-1.0), 4, W, m=6)
    assert st["rho_lower_median"]["point"] == pytest.approx(-1.0)
    assert st["by_age_bucket"]["5..M"]["n_cells"] == 4 * 5 * 2  # ages 5, 6 of 6
    assert st["cells_older_than_M"] == 0
    st = r2.probe_stats(_decisions(K=6), 4, W, m=4)  # ages 5, 6 > M = 4: counted
    assert st["cells_older_than_M"] == 4 * 5 * 2


def test_weighted_lower_median_equals_brute_force(r2):
    g = torch.Generator().manual_seed(3)
    v = torch.randn(40, generator=g, dtype=torch.float64)
    doc = torch.randint(0, 6, (40,), generator=g)
    W = torch.randint(0, 3, (25, 6), generator=g).to(torch.float64)
    got = r2.weighted_lower_medians(v, doc, W, chunk=7)
    for r in range(W.shape[0]):
        ms = [float(x) for j, x in enumerate(v) for _ in range(int(W[r, doc[j]]))]
        want = sorted(ms)[(len(ms) - 1) // 2] if ms else float("nan")
        assert (math.isnan(want) and math.isnan(float(got[r]))) or float(got[r]) == want


def test_avg_ranks_match_rankdata_with_ties(r2):
    from rsr.metrics.vacuity import rankdata

    x = torch.tensor([[3.0, 1.0, 3.0, 2.0], [0.5, 0.5, 0.5, 0.1]], dtype=torch.float64)
    got = r2.avg_ranks_rows(x)
    for i in range(2):
        assert torch.equal(got[i], rankdata(x[i]))


# --------------------------------------------------------------------------- #
# §7: C14 and C15 on tampered inputs
# --------------------------------------------------------------------------- #


@pytest.fixture
def es_world(r2, tmp_path):
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    ck = pinned / "L48-k1e-2-seed0-ckpt-003000.pt"
    ck.write_bytes(b"weights")
    (pinned / "MANIFEST.sha256").write_text(f"{_sha(ck)}  {ck.name}\n")
    es_out = tmp_path / "es"
    es_recs = _recs(seed=1)
    for r, f in zip(es_recs, _recs(), strict=True):  # same (doc, t, gap) as run 2's psiC
        r["answers"] = [
            [a[0], b[1], a[2], a[3]]
            for a, b in zip(r["answers"], f["answers"], strict=True)
        ]
    p = _write_pass(es_out / "eval" / "seed0", "es", es_recs)
    (es_out / "eval" / "seed0" / "eval.json").write_text(
        json.dumps({"seed": 0, "config": "L48-k1e-2", "records_sha256": _sha(p)})
    )
    run1 = tmp_path / "run1"
    fp = _write_pass(run1 / "heldout" / "seed0", "fifo", _recs(seed=2))
    refs = r2.Refs(
        run1_out=str(run1),
        es_out=str(es_out),
        es_pinned=str(pinned),
        es_sha=((0, _sha(p)),),
        run1_fifo_sha=((0, _sha(fp)),),
    )
    return {"refs": refs, "ck": ck, "es": p, "es_recs": es_recs}


def test_c14_passes_on_the_pinned_source_and_fails_on_tampering(r2, es_world, tmp_path):
    refs = es_world["refs"]
    assert r2.c14_es_source(0, refs)["ok"] is True
    es = es_world["es"]
    good = es.read_bytes()
    es.write_bytes(good + b"\n")  # a tampered vector: sha != registered and != eval.json
    c = r2.c14_es_source(0, refs)
    assert c["ok"] is False and c["es_sha256"] != c["registered"]
    es.write_bytes(good)
    assert r2.c14_es_source(0, refs)["ok"] is True
    es_world["ck"].write_bytes(b"weightz")  # a tampered pinned checkpoint
    c = r2.c14_es_source(0, refs)
    assert c["ok"] is False and c["manifest_rc"] != 0
    es_world["ck"].write_bytes(b"weights")
    ev = es.parent / "eval.json"
    body = json.loads(ev.read_text())
    ev.write_text(json.dumps({**body, "config": "L32-k1e-2"}))  # another config
    assert r2.c14_es_source(0, refs)["ok"] is False
    ev.write_text(json.dumps(body))
    wrong = r2.Refs(**{**refs.__dict__, "es_sha": ((0, "0" * 64),)})
    assert r2.c14_es_source(0, wrong)["ok"] is False


def test_c15_passes_when_paired_and_fails_on_a_tampered_vector(r2, es_world):
    refs = es_world["refs"]
    fifo2, psic2 = _recs(seed=2), _recs()
    es = es_world["es_recs"]
    assert r2.c15_pairing(0, fifo2, psic2, es, refs)["ok"] is True
    c = r2.c15_pairing(0, _recs(seed=2, flip=1), psic2, es, refs)  # one nll off by 1e-9
    assert c["ok"] is False and c["fifo_records_equal"] is False
    bad = json.loads(json.dumps(psic2))
    bad[0]["answers"][1][1] += 1  # one gap differs from ES
    c = r2.c15_pairing(0, fifo2, bad, es, refs)
    assert c["ok"] is False and c["psiC_es_tgap_equal"] is False
    wrong = r2.Refs(**{**refs.__dict__, "run1_fifo_sha": ((0, "0" * 64),)})
    c = r2.c15_pairing(0, fifo2, psic2, es, wrong)
    assert c["ok"] is False and c["fifo_sha_ok"] is False


def test_the_real_pinned_shas_are_the_prereg_s(r2):
    text = (ROOT / r2.PREREG).read_text()
    for v in (*r2.ES_SHA.values(), *r2.RUN1_FIFO_SHA.values()):
        assert v in text


# --------------------------------------------------------------------------- #
# §6 / §7: aggregation and the run-level controls
# --------------------------------------------------------------------------- #


def _beat(step, *, ok=95, nd=5, target="loo_resample", loo_s=1.0, g=0.5):
    return {
        "kind": "beat",
        "step": step,
        "loss_mc": 0.1,
        "phi_grad_norm": 0.2,
        "loo_s": loo_s,
        "eval_forward_s": 0.1,
        "step_s": 2.0,
        "mc": {
            "target": target,
            "g_mean": g,
            "loo": {
                "counts": {"full.ok.assert": ok, "full.no_donor.filler": nd},
                "donor_age_gap_mean": 2.0,
                "donor_age_gap_zero_share": 0.5,
            },
            "target_diag": {
                "g_by_age": {"1": [1.0, 2], "16": [0.5, 1]},
                "g_by_evict_age": {"1": [0.25, 1], "kept": [1.25, 2]},
                "rho_g_age": {"n": 3, "median": -0.1},
            },
        },
    }


def test_target_diagnostics_split_at_t_warm_and_c9_floor(r2):
    beats = [_beat(k) for k in range(4)]
    d = r2.target_diagnostics(beats, 2.0)
    assert d["warm"]["n_steps"] == 2 and d["post_warm"]["n_steps"] == 2
    assert d["warm"]["g_by_age"][1] == {"mean": 0.5, "n": 4}
    assert list(d["warm"]["g_by_evict_age"]) == ["1", "kept"]
    assert d["warm"]["full_coverage"] == 0.95
    assert r2.c9_coverage({"psiC": d})["ok"]
    edge = r2.target_diagnostics([_beat(0, ok=80, nd=20)], 1.0)
    assert r2.c9_coverage({"psiC": edge})["per_arm"]["psiC"]["coverage"] == 0.80
    assert r2.c9_coverage({"psiC": edge})["ok"]  # >= 0.80 inclusive (E0d C6)
    low = r2.target_diagnostics([_beat(0, ok=79, nd=21)], 1.0)
    assert not r2.c9_coverage({"psiC": low})["ok"]


def test_c11_and_c13(r2):
    lab = "rsr[b=off,beta=1,gamma=0.9,nu=0,shadow=off,target=loo_resample]+frozen[eval]"
    res = {
        "psiC": {"config": {"arm_label": lab}},
        "age": {"config": {"arm_label": lab.replace("nu=0,", "nu=0,psi=age_only,")}},
        "fifo": {"config": {"arm_label": "fifo+frozen[eval]"}},
    }
    hbs = {a: {"beats": [_beat(0), _beat(1)]} for a in ("psiC", "age")}
    hbs["fifo"] = {"beats": []}
    launch = {"target": "loo_resample", "smoke": ""}
    assert r2.c11_target(res, hbs, launch)["ok"]
    assert not r2.c11_target(res, hbs, {**launch, "smoke": "1"})["ok"]
    hbs2 = {**hbs, "age": {"beats": [_beat(0), _beat(1, loo_s=float("nan"))]}}
    assert not r2.c11_target(res, hbs2, launch)["ok"]
    hbs3 = {**hbs, "psiC": {"beats": [_beat(0, target="r_i")]}}
    assert not r2.c11_target(res, hbs3, launch)["ok"]
    assert r2.c13_finite(hbs)["ok"]
    assert not r2.c13_finite({**hbs, "psiC": {"beats": [_beat(0, g=float("inf"))]}})["ok"]


def test_c2_prime_reads_the_warm_bitwise_check(r2):
    c2 = {"ok": True, "warm_steps_compared": 16, "c_loss_lm_warm_mismatch_steps": []}
    assert r2.c2_prime(c2)["ok"]
    assert not r2.c2_prime({**c2, "c_loss_lm_warm_mismatch_steps": [3]})["ok"]
    assert not r2.c2_prime(None)["ok"]


def test_heartbeat_comparison_ignores_only_wall_clock_and_git_sha(r2, tmp_path):
    def hb(path, sha, el, loss):
        rows = [
            {
                "kind": "header",
                "elapsed_s": 0.0,
                "provenance": {"git_sha": sha, "seed": 0},
            },
            {
                "kind": "beat",
                "step": 0,
                "elapsed_s": el,
                "sent_per_s": 1 / el,
                "loss": loss,
            },
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    a = hb(tmp_path / "a.jsonl", "aaa", 0.9, 1.25)
    b = hb(tmp_path / "b.jsonl", "bbb", 1.7, 1.25)
    c = hb(tmp_path / "c.jsonl", "aaa", 0.9, 1.2500001)
    assert r2.comparable_heartbeat(a) == r2.comparable_heartbeat(b)
    assert r2.comparable_heartbeat(a) != r2.comparable_heartbeat(c)


# --------------------------------------------------------------------------- #
# the command, the launcher, and run 1's module untouched
# --------------------------------------------------------------------------- #


def _launcher_command(r2, arm: str, seed: int, out: str) -> list[str]:
    text = (ROOT / r2.LAUNCHER).read_text()
    i = text.index(".venv/bin/python -m rsr.train.loop")
    j = text.index('"${P[@]}" --out-dir $OUT/$a/seed$s') + len(
        '"${P[@]}" --out-dir $OUT/$a/seed$s'
    )
    cmd = text[i:j].replace("\\\n", " ")
    rsr = (
        "--policy rsr --policy-override b_enabled=false --policy-override "
        "shadow_enabled=false --policy-override nu=0 --policy-override beta=1 "
        "--policy-override gamma=0.9 --policy-override target=loo_resample"
    )
    p = {
        "psiC": rsr,
        "fifo": "--policy fifo",
        "age": rsr + " --policy-override psi_override=age_only",
    }[arm]
    s = r2.R1.SUBSTRATE[seed]
    for k, v in (
        ('"${C1[@]}"', " ".join(r2.R1.C1_FLAGS["eval"])),
        ('"${C3[@]}"', ""),
        ('"${C2[@]}"', " ".join(r2.R1.C2_FLAGS)),
        ('"${P[@]}"', p),
        ("${SHA[$s]}", s["sha256"]),
        ("$STEPS", "3000"),
        ("$EVERY", "40"),
        ("$OUT/$a/seed$s", out),
        ("$B", str(Path(s["checkpoint"]).parent)),
        ("$R", r2.R1._R),
        ("$s", str(seed)),
    ):
        cmd = cmd.replace(k, v)
    return shlex.split(cmd)


def test_cli_argv_is_the_launchers_command_token_for_token(r2):
    ow = r2.R1.OwnerValues(0.9, 0.01, 3000, "eval", (0, 1, 2))
    for seed in (0, 1, 2):
        cfg = r2.R1.prereg_config(seed, 3000)
        for arm in r2.ARMS:
            out = f"runs/rsr-run2/{arm}/seed{seed}"
            got = r2.cli_argv(cfg, arm, ow, Path(out))
            assert got == _launcher_command(r2, arm, seed, out), (arm, seed)
            r1 = r2._R1_CLI_ARGV(cfg, arm, ow, Path(out))
            if arm == "fifo":
                assert got == r1  # C12: fifo is run 1's command byte for byte
            else:  # PREREG §0: exactly one change, the target override
                i = got.index("target=loo_resample")
                assert got[i - 1] == "--policy-override" and got[i - 2] == "gamma=0.9"
                assert got[: i - 1] + got[i + 1 :] == r1


def test_the_rebinding_is_private_to_run_2(r2):
    assert r2.R1.RUN_ID == "rsr-run2" and r2.R1.cli_argv is r2.cli_argv
    spec = importlib.util.spec_from_file_location(
        "rsr_run1_fresh", ROOT / "experiments" / "rsr-run1" / "run.py"
    )
    fresh = importlib.util.module_from_spec(spec)
    sys.modules["rsr_run1_fresh"] = fresh
    spec.loader.exec_module(fresh)
    assert fresh.RUN_ID == "rsr-run1" and fresh.cli_argv is not r2.cli_argv
    assert r2.R1.job_record.__globals__["RUN_ID"] == "rsr-run2"


@pytest.mark.parametrize(
    ("env", "msg"),
    [
        ({}, "slot GAMMA unfilled"),
        (
            {"GAMMA": "0.9", "WD": "0.01", "STEPS": "31", "O5": "eval",
             "C1_FLAGS": "--frozen-model-mode eval", "SMOKE": "1", "OUT": "/tmp/x",
             "JOB_SUFFIX": "-s"},
            "SMOKE=1 needs 1 <= STEPS <= 30",
        ),
        (
            {"GAMMA": "0.9", "WD": "0.01", "STEPS": "30", "O5": "eval",
             "C1_FLAGS": "--frozen-model-mode eval", "SMOKE": "1", "JOB_SUFFIX": "-s"},
            "SMOKE=1 needs OUT",
        ),
        (
            {"GAMMA": "0.9", "WD": "0.01", "STEPS": "30", "O5": "eval",
             "C1_FLAGS": "--frozen-model-mode eval", "SMOKE": "1", "OUT": "/tmp/x"},
            "SMOKE=1 needs JOB_SUFFIX",
        ),
        (
            {"GAMMA": "0.9", "WD": "0.01", "STEPS": "30", "O5": "eval",
             "C1_FLAGS": "--frozen-model-mode eval", "OUT": "/tmp/x"},
            "OUT=/tmp/x is set",
        ),
    ],
)  # fmt: skip
def test_launcher_refusals_exit_3_before_anything_runs(r2, tmp_path, env, msg):
    p = subprocess.run(
        ["/bin/zsh", str(ROOT / r2.LAUNCHER)],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **env},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 3, p.stderr
    assert msg in p.stderr
    assert not (tmp_path / "runs").exists()
