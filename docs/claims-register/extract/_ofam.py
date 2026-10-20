"""Shared row builders for offline-psi, diagnostic A (offline-psi-converge) and LS
(psi-ls-capacity). Raw checks run ``recompute_offline_psi_family.py`` (each run's own
reading code over its saved phi / gestalts / records; one command per seed, cached by
the builder) and pick the leaf.

diagnostic A and LS have no committed ledger: their per-seed record is
``~/rsr-runs-archive/<exp>/runs/<exp>/seed<s>/result.json`` (and ``summary.json``),
read here; the sha256 cited in each row's source is read from
``~/rsr-runs-archive/MANIFEST.sha256``, not typed."""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

SEEDS = [0, 1, 2]
FAM = f"{C.PY} docs/claims-register/recompute_offline_psi_family.py"
UBOOT = "paired per-document bootstrap percentile over the U-val documents, n_boot 2000"


def fam_cmd(exp: str, s: int) -> str:
    return f"{C.SLOT}{FAM} --exp {exp} --seed {s}"


def labels_cmd(exp: str) -> str:
    return f"{C.SLOT}{FAM} --exp {exp} --labels"


def arch_root(exp: str) -> Path:
    return C.ARCHIVE / exp / "runs" / exp


@cache
def record(exp: str, rel: str):
    return json.loads((arch_root(exp) / rel).read_text())


@cache
def manifest() -> dict[str, str]:
    out = {}
    for line in (C.ARCHIVE / "MANIFEST.sha256").read_text().splitlines():
        h, _, p = line.partition("  ")
        out[p.strip()] = h
    return out


def sha(exp: str, rel: str) -> str:
    return manifest().get(f"./{exp}/runs/{exp}/{rel}", "NOT IN MANIFEST")[:12]


def src(exp: str, rel_tmpl: str, key: str) -> str:
    shas = ", ".join(f"s{s} {sha(exp, rel_tmpl.format(s=s))}" for s in SEEDS)
    return (
        f"~/rsr-runs-archive/{exp}/runs/{exp}/{rel_tmpl.replace('{s}', '{s}')} :: {key} "
        f"(manifest sha256[:12] {shas})"
    )


def rec_row(
    exp: str,
    rid: str,
    path: str,
    claim: str,
    *,
    tier: str = "secondary",
    prereg: str,
    seen: str,
    seen_why: str,
    with_ci: bool = True,
    ci_fields: tuple[str, str] = ("lo", "hi"),
    point: str = "point",
    extra_checks: list[C.Check] | None = None,
    note: str | None = None,
) -> C.Row:
    """A statistic at ``path`` (dotted) in diagnostic A's / LS's ``seed<s>/result.json``,
    per seed; raw checks pick the same path from the family recompute."""
    vals, cis, checks = [], [], []
    for s in SEEDS:
        rec = record(exp, f"seed{s}/result.json")
        fields = (point, *ci_fields) if with_ci else (None,)
        got = {}
        for f in fields:
            p = f"{path}.{f}" if f else path
            v = C.dig(rec, p)
            got[f] = v
            checks.append(
                C.Check(
                    label=f"s{s} {f or 'value'}",
                    cmd=fam_cmd(exp, s),
                    expected=C.fmt_exact(v),
                    kind="raw",
                    pick=p,
                )
            )
        vals.append(got[point] if with_ci else got[None])
        cis.append((got[ci_fields[0]], got[ci_fields[1]]) if with_ci else None)
    return C.Row(
        id=rid,
        experiment=exp,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=vals,
        ci=cis if with_ci else None,
        ci_method=UBOOT if with_ci else "none computed",
        source=src(
            exp,
            "seed{s}/result.json",
            path + (f".{{{point},{','.join(ci_fields)}}}" if with_ci else ""),
        ),
        prereg=prereg,
        seen=seen,
        seen_why=seen_why,
        checks=checks + (extra_checks or []),
        note=note,
    )
