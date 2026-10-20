"""Shared row builders for the model-free ceiling runs (b0-ceilings, gamma-ceiling,
cf-ceiling). Their ledgers hold ``statistic`` rows: ``samples`` = one value per seed
(seed order 0, 1, 2). Raw check: ``recompute_ceilings.py <what> --seed s`` re-runs the
run's own scoring on the archived raw records (one command per seed, cached by the
builder) and the builder picks the ledger key from its JSON output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

SEEDS = [0, 1, 2]
BOOT = (
    "paired per-document bootstrap percentile, n_boot 2000, 1088 U docs, seed 20260927+s"
)
RC = f"{C.PY} docs/claims-register/recompute_ceilings.py"


def stats(run: str) -> dict[str, dict]:
    return {
        r["key"]: r
        for r in C.ledger(run).get("rows") or []
        if isinstance(r, dict) and isinstance(r.get("key"), str)
    }


def samples(run: str, key: str) -> list:
    return stats(run)[key]["samples"]


def raw_cmd(what: str, s: int) -> str:
    return f"{C.SLOT}{RC} {what} --seed {s}"


def cap_row(
    run: str,
    what: str,
    rid: str,
    key: str,
    claim: str,
    *,
    tier: str,
    prereg: str,
    seen: str,
    seen_why: str,
    with_ci: bool = True,
    note: str | None = None,
) -> C.Row:
    """``key`` is the stat prefix: with_ci -> ``key.{point,lo,hi}``, else ``key``."""
    fields = ("point", "lo", "hi") if with_ci else (None,)
    got = {f: samples(run, f"{key}.{f}" if f else key) for f in fields}
    checks = [
        C.Check(
            label=f"s{s} {f or 'value'}",
            cmd=raw_cmd(what, s),
            expected=C.fmt_exact(got[f][s]),
            kind="raw",
            pick=f"{key}.{f}" if f else key,
        )
        for s in SEEDS
        for f in fields
    ]
    return C.Row(
        id=rid,
        experiment=run,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=list(got["point"] if with_ci else got[None]),
        ci=list(zip(got["lo"], got["hi"], strict=True)) if with_ci else None,
        ci_method=BOOT if with_ci else "none computed",
        source=f"runs/{run}/ledger.json :: {key}"
        + (".{point,lo,hi}" if with_ci else "")
        + ".samples[s]",
        prereg=prereg,
        seen=seen,
        seen_why=seen_why,
        checks=checks,
        note=note,
    )


def label_row(
    run: str,
    which: str,
    rid: str,
    claim: str,
    *,
    prereg: str,
    seen: str,
    seen_why: str,
    class_key: str,
) -> C.Row:
    """Per-gamma labels + class, re-derived by re-applying the run's own reading rule
    to the committed caps (kind ``ledger``) and read from the run's ledger copy in the
    archive (kind ``archive-ledger``)."""
    lab = C.lget(run, "B.ckpt3000.label")
    cls = C.lget(run, class_key)
    arch = f"{C.ARCHIVE}/{run}/runs/{run}/ledger.json"
    return C.Row(
        id=rid,
        experiment=run,
        tier="headline",
        claim=claim + f"; class {cls}",
        seeds=SEEDS,
        values=[", ".join(f"{g} {v}" for g, v in lab.items())] * len(SEEDS),
        source=f"runs/{run}/ledger.json :: B.ckpt3000.label; {class_key}",
        prereg=prereg,
        seen=seen,
        seen_why=seen_why,
        note="one label per gamma over all 3 seeds (the rule reads all seeds' caps)",
        checks=[
            C.Check(
                label="labels+class (own rule on committed caps)",
                cmd=f"{RC} labels {which}",
                expected=json.dumps({"labels": lab, "class": cls}, sort_keys=True),
                kind="ledger",
            ),
            C.archive_ledger_check(arch, run, "B.ckpt3000.label", None, "archive labels"),
            C.archive_ledger_check(arch, run, class_key, None, "archive class"),
        ],
    )
