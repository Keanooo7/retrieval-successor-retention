"""E-feas -- the oracle against FIFO on the synthetic corpus, model-free.

Pre-registration: `experiments/efeas/PREREG.md`, committed ahead of this file.
`decide()` below is its decision rule, transcribed.

Per corpus seed: simulate every document under FIFO, the oracle and a seeded random
policy (`rsr.metrics.headroom.simulate`, the training path's real memory mechanics),
check the two controls, and report headroom `H = hit(oracle) - hit(FIFO)` at M = 16,
with M = 8 and M = 32 as secondary rows. No model is trained, so this runs on CPU in
seconds; it needs no device (ADR-0007 governs training).

Usage:

    uv run python experiments/efeas/run.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.baselines.random_policy import RandomPolicy  # noqa: E402
from rsr.data.synthetic import (  # noqa: E402
    SyntheticConfig,
    discounted_demand,
    generate,
    to_bytes,
)
from rsr.metrics.headroom import hit_rate, hit_rate_by_gap, simulate  # noqa: E402

EXPERIMENT = "experiments/efeas/run.py"
SEEDS = [0, 1, 2]
M_PRIMARY = 16
M_SECONDARY = [8, 32]
GAMMA = 0.97
GAP_EDGES = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]

#: PREREG.md "Decision rule".
H_MIN = 0.05


def arm_records(docs, m: int, seed: int) -> dict[str, list[dict]]:
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list[dict]] = {"fifo": [], "oracle": [], "random": []}
    for doc in docs:
        out["fifo"].append(simulate(doc, FIFOPolicy(), m))
        oracle = OraclePolicy(discounted_demand(doc, GAMMA))
        out["oracle"].append(simulate(doc, oracle, m))
        out["random"].append(simulate(doc, RandomPolicy(gen), m))
    return out


def controls(docs, recs: dict[str, list[dict]], m: int) -> list[str]:
    why = []
    bad = [
        (d.doc_id, q)
        for d, r in zip(docs, recs["fifo"], strict=True)
        for q in r["queries"]
        if q["hit"] != (q["gap"] <= m)
    ]
    if bad:
        why.append(f"FIFO hit != (gap <= M) on {len(bad)} queries, e.g. {bad[:3]}")
    lost = [
        d.doc_id
        for d, f, o in zip(docs, recs["fifo"], recs["oracle"], strict=True)
        if f["queries"] and hit_rate([o]) < hit_rate([f])
    ]
    if lost:
        why.append(f"oracle below FIFO on documents {lost}")
    return why


#: 🔒 Transcribed from PREREG.md, which is committed ahead of this file.
def decide(per_seed: dict[int, dict]) -> tuple[str, str]:
    broken = {
        s: v["controls_failed"] for s, v in per_seed.items() if v["controls_failed"]
    }
    if broken:
        return "inconclusive", f"a control failed: {broken}"
    h = {s: v["headroom"] for s, v in per_seed.items()}
    if all(x < H_MIN for x in h.values()):
        return (
            "falsified",
            f"oracle ~= FIFO on every seed (H < {H_MIN}: {h}); the synthetic corpus "
            f"cannot exhibit the effect",
        )
    if all(x >= H_MIN for x in h.values()):
        return (
            "survived",
            f"headroom >= {H_MIN} on every seed ({h}), and both controls held",
        )
    return "inconclusive", f"seeds disagree about the threshold: {h}"


def measure_seed(seed: int) -> dict:
    docs = generate(SyntheticConfig(seed=seed))
    recs = arm_records(docs, M_PRIMARY, seed)
    rates = {arm: hit_rate(r) for arm, r in recs.items()}
    out = {
        "n_documents": len(docs),
        "n_queries": sum(len(r["queries"]) for r in recs["fifo"]),
        "hit_rate": rates,
        "headroom": rates["oracle"] - rates["fifo"],
        "controls_failed": controls(docs, recs, M_PRIMARY),
        "by_gap": {arm: hit_rate_by_gap(r, GAP_EDGES) for arm, r in recs.items()},
        "secondary": {},
    }
    for m in M_SECONDARY:
        r2 = arm_records(docs, m, seed)
        rr = {arm: hit_rate(r) for arm, r in r2.items()}
        out["secondary"][f"M={m}"] = {
            "hit_rate": rr,
            "headroom": rr["oracle"] - rr["fifo"],
        }
    return out


def write_ledger(per_seed: dict[int, dict], led) -> Path:
    seeds = sorted(per_seed)
    how = (
        f"{EXPERIMENT}::measure_seed -> rsr.metrics.headroom.simulate over all 64 "
        f"documents of SyntheticConfig(seed=s), M={M_PRIMARY}; a query hits if its "
        f"asserting sentence is resident before the query step's write"
    )
    led.stat(
        "headroom_oracle_minus_fifo",
        [per_seed[s]["headroom"] for s in seeds],
        how=f"{how}; hit(oracle) - hit(FIFO), pooled per seed; PREREG.md statistic",
    )
    for arm in ("fifo", "oracle", "random"):
        led.stat(
            f"hit_rate.{arm}",
            [per_seed[s]["hit_rate"][arm] for s in seeds],
            how=f"{how}; arm {arm}",
        )
    for m in M_SECONDARY:
        led.stat(
            f"secondary.M{m}.headroom",
            [per_seed[s]["secondary"][f"M={m}"]["headroom"] for s in seeds],
            how=f"{how.replace(f'M={M_PRIMARY}', f'M={m}')}; secondary row, not judged",
        )
    for s in seeds:
        v = per_seed[s]
        led.note(f"seed{s}.n_queries", v["n_queries"], how=how)
        led.note(
            f"seed{s}.controls_failed", v["controls_failed"], how="PREREG.md 'Controls'"
        )
        led.note(f"seed{s}.by_gap", v["by_gap"], how=f"{how}; gap buckets {GAP_EDGES}")
    outcome, detail = decide(per_seed)
    led.verdict(
        falsifier=(
            "PREREG.md: every seed headroom < 0.05 => oracle ~= FIFO, the synthetic "
            "corpus cannot exhibit the effect; a failed control => inconclusive"
        ),
        outcome=outcome,
        detail=detail,
    )
    return led.write()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="efeas-synthetic")
    ap.add_argument(
        "--prereg",
        default="experiments/efeas/PREREG.md",
        help="the pre-registration governing this run, recorded in the manifest",
    )
    ap.add_argument(
        "--expect",
        default=None,
        help="the expectation, written before the run and frozen in the manifest",
    )
    a = ap.parse_args(argv)
    argv_str = "uv run python " + " ".join(
        [EXPERIMENT, *(sys.argv[1:] if argv is None else argv)]
    )

    from ledger import Ledger

    led = Ledger(
        a.run_id,
        question=(
            "On the synthetic corpus, does an optimal eviction rule with perfect "
            "knowledge of future demand keep >= 5 points more of what queries need "
            "than FIFO does?"
        ),
    )
    led.manifest(
        {
            "corpus": "generate(SyntheticConfig(seed=s)), committed defaults",
            "seeds": SEEDS,
            "memory_slots": M_PRIMARY,
            "secondary_memory_slots": M_SECONDARY,
            "gamma": GAMMA,
            "arms": ["fifo", "oracle", "random"],
            "prereg": a.prereg,
            "threshold": {"h_min": H_MIN},
            "device": "cpu (no model)",
            # Added for efeas-synthetic-s003 (PREREG-s003.md): the generator's
            # defaults verbatim, the corpus bytes it produced, and torch, so a
            # changed default is visible in the manifest rather than inferred.
            "synthetic_config_defaults": asdict(SyntheticConfig()),
            "corpus_sha256": {
                str(s): hashlib.sha256(
                    to_bytes(generate(SyntheticConfig(seed=s)))
                ).hexdigest()
                for s in SEEDS
            },
            "torch_version": torch.__version__,
            "expected": a.expect,
        }
    )
    per_seed = {s: measure_seed(s) for s in SEEDS}
    led.run_meta(device="cpu", seeds_actually_run=SEEDS)
    led.command(
        argv_str,
        exit_code=0,
        note=(
            "SELF-REPORTED: this is the process writing the ledger, so 0 means "
            "'reached write()', not an observed exit status. The caller's $? is "
            "the authority, and RESULTS.md quotes it"
        ),
    )
    led.status("ok")
    root = ROOT / "runs" / a.run_id
    (root / "raw.json").write_text(json.dumps(per_seed, indent=2) + "\n")
    p = write_ledger(per_seed, led)
    print(
        json.dumps(
            {
                s: {k: v[k] for k in ("hit_rate", "headroom", "controls_failed")}
                for s, v in per_seed.items()
            },
            indent=2,
        )
    )
    print(json.dumps(led.doc["verdict"], indent=2))
    print(f"ledger: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
