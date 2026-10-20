"""Shared row builders for run 1 / run 2's held-out numbers (``seed<s>.heldout.*``).

Raw check: ``recompute_heldout.py`` re-runs run 1's own ``analyse_heldout`` on the
archived per-document records (one command per seed, cached by the builder) and the
builder picks the leaf; it must string-equal the committed ledger's value."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

SEEDS = [0, 1, 2]
BOOT = "paired per-document bootstrap percentile, n_boot 2000, 2048 EVAL docs, seed 20260927+s"


def heldout_cmd(run: str, s: int) -> str:
    return (
        f"{C.SLOT}{C.PY} docs/claims-register/recompute_heldout.py --run {run} --seed {s}"
    )


def leaf(run: str, s: int, path: str):
    """The committed ledger's value at ``seed<s>.heldout.<path>``: the row key is the
    longest dotted prefix present in the ledger; the remainder is a sub-path."""
    rows = C.lrows(run)
    full = f"seed{s}.heldout.{path}"
    parts = full.split(".")
    for i in range(len(parts), 0, -1):
        k = ".".join(parts[:i])
        if k in rows:
            rest = parts[i:]
            return C.dig(rows[k], rest) if rest else rows[k], k, ".".join(rest)
    raise KeyError(full)


def stat_row(
    run: str,
    rid: str,
    path: str,
    claim: str,
    *,
    tier: str = "secondary",
    prereg: str,
    seen: str,
    seen_why: str,
    with_ci: bool = True,
    note: str | None = None,
) -> C.Row:
    """One held-out statistic (a dict with point/lo/hi) or scalar at
    ``seed<s>.heldout.<path>``, per seed, with raw checks of point (and lo/hi)."""
    vals, cis, checks = [], [], []
    keys = set()
    for s in SEEDS:
        fields = ("point", "lo", "hi") if with_ci else (None,)
        got = {}
        for f in fields:
            p = f"{path}.{f}" if f else path
            v, k, _rest = leaf(run, s, p)
            keys.add(k)
            got[f] = v
            checks.append(
                C.Check(
                    label=f"s{s} {f or 'value'}",
                    cmd=heldout_cmd(run, s),
                    expected=C.fmt_exact(v),
                    kind="raw",
                    pick=f"seed{s}.heldout.{p}",
                )
            )
        vals.append(got["point"] if with_ci else got[None])
        cis.append((got["lo"], got["hi"]) if with_ci else None)
    src = f"runs/{run}/ledger.json :: seed{{s}}.heldout.{path}" + (
        ".{point,lo,hi}" if with_ci else ""
    )
    return C.Row(
        id=rid,
        experiment=run,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=vals,
        ci=cis if with_ci else None,
        ci_method=BOOT if with_ci else "none computed",
        source=src,
        prereg=prereg,
        seen=seen,
        seen_why=seen_why,
        checks=checks,
        note=note,
    )
