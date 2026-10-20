"""Shared row builders for runs whose ledger rows are recomputed from the committed
``runs/<id>/raw.json`` by ``recompute_from_raw.py`` (decisive-shuffle, fresh-stream,
fresh-escape).

These ledgers hold ``statistic`` rows (per-seed ``samples``, no ``value``), so they are
read here directly rather than through ``common.lrows`` (which returns ``value``)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

SEEDS = [0, 1, 2]
_CACHE: dict[str, dict] = {}


def L(run: str) -> dict:
    """key -> ``samples`` (statistic rows) or ``value`` (observation rows); last wins."""
    if run not in _CACHE:
        out = {}
        for r in C.ledger(run).get("rows") or []:
            if isinstance(r, dict) and isinstance(r.get("key"), str):
                out[r["key"]] = (
                    r["samples"] if r.get("kind") == "statistic" else r.get("value")
                )
        _CACHE[run] = out
    return _CACHE[run]


def cmd(run: str) -> str:
    return f"{C.SLOT}{C.PY} docs/claims-register/recompute_from_raw.py --run {run}"


def check(run: str, key: str, label: str | None = None) -> C.Check:
    """Raw check: the run's own row writer, re-run on raw.json, must give this key's
    ledger value (a statistic's whole per-seed samples list) exactly."""
    if key == "__verdict__":  # the Ledger.verdict(...) call's three arguments
        v = {k: C.ledger(run)["verdict"][k] for k in ("falsifier", "outcome", "detail")}
    else:
        v = L(run)[key]
    return C.Check(
        label=label or key, cmd=cmd(run), expected=C.fmt_exact(v), kind="raw", pick=key
    )


def none_as(x, word: str):
    return word if x is None else x


def module_constant(rel: str, name: str) -> float:
    """A frozen constant read from the run's committed source (never retyped here)."""
    src = (C.ROOT / rel).read_text()
    m = re.search(rf"^{name}\s*=\s*([0-9.eE+-]+)", src, re.M)
    if not m:
        raise KeyError(f"{name} not found in {rel}")
    return float(m.group(1))


def samples_row(
    run: str,
    rid: str,
    key: str,
    claim: str,
    *,
    tier: str = "secondary",
    prereg: str,
    seen: str,
    seen_why: str,
    note: str | None = None,
) -> C.Row:
    xs = L(run)[key]
    if not isinstance(xs, list) or len(xs) != len(SEEDS):
        raise ValueError(f"{run}:{key} is not a per-seed samples list: {xs!r}")
    return C.Row(
        id=rid,
        experiment=run,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=xs,
        ci=None,
        ci_method="none computed (per-seed point values; this run computed no CI)",
        source=f"runs/{run}/ledger.json :: {key} (statistic samples, seed order 0,1,2)",
        prereg=prereg,
        seen=seen,
        seen_why=seen_why,
        checks=[check(run, key, "samples s0-s2")],
        note=note,
    )
