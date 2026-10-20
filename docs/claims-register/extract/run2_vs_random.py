"""run2-vs-random (experiments/run2-vs-random; diag/run2-vs-random 6ed6583, 2026-10-08).

DESCRIPTIVE and EXPLORATORY: its docstring says nothing in it is pre-registered and
nothing gates. Values are read from the committed ``experiments/run2-vs-random/
results.json`` (the script's own output; the experiment has no ledger). Raw check:
``recompute_run2_vs_random.py`` re-runs the experiment's own ``analyse_seed`` on the
archived run 1 / run 2 per-document records (one command per seed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C

EXPERIMENT = "run2-vs-random"
RES = "experiments/run2-vs-random/results.json"
SEEDS = [0, 1, 2]
BOOT = (
    "paired per-document bootstrap percentile, n_boot 2000, 2048 EVAL docs, "
    "seed 20260927+s; random = mean of 5 draws, paired per replicate"
)
E = (
    f"{C.EXPLORATORY} (post-hoc diagnostic after run 2's FAILS; "
    "docstring: not pre-registered)"
)
SEEN_PTS = (
    C.SEEN,
    "point estimates derived by arithmetic from run 1/run 2 tables "
    "before the script was written (run.py docstring)",
)
SEEN_R1 = (C.SEEN, "run 1's psiC records, read in run 1 (labelled _SEEN in results.json)")
UNSEEN_RES = (
    C.UNSEEN,
    "residency not derivable from earlier tables; first computed here, but exploratory",
)


def cmd(s: int) -> str:
    return f"{C.SLOT}{C.PY} docs/claims-register/recompute_run2_vs_random.py --seed {s}"


def stat(
    rid: str,
    part: str,
    bucket: str,
    stat_key: str,
    claim: str,
    *,
    tier: str,
    seen: tuple,
    note: str | None = None,
) -> C.Row:
    d = C.jload(C.ROOT / RES)
    vals, cis, checks = [], [], []
    for s in SEEDS:
        base = f"seeds.{s}.{part}.{bucket}.{stat_key}"
        got = {f: C.dig(d, f"{base}.{f}") for f in ("point", "lo", "hi")}
        vals.append(got["point"])
        cis.append((got["lo"], got["hi"]))
        for f in ("point", "lo", "hi"):
            checks.append(
                C.Check(
                    label=f"s{s} {f}",
                    cmd=cmd(s),
                    expected=C.fmt_exact(got[f]),
                    kind="raw",
                    pick=f"{base}.{f}",
                )
            )
    return C.Row(
        id=rid,
        experiment=EXPERIMENT,
        tier=tier,
        claim=claim,
        seeds=SEEDS,
        values=vals,
        ci=cis,
        ci_method=BOOT,
        source=f"{RES} :: seeds.{{s}}.{part}.{bucket}.{stat_key}.{{point,lo,hi}} "
        "(committed script output; no ledger)",
        prereg=E,
        seen=seen[0],
        seen_why=seen[1],
        checks=checks,
        note=note,
    )


def rows() -> list[C.Row]:
    out = [
        stat(
            "r2vr.acc.psiC-random",
            "accuracy",
            "all",
            "run2_psiC_minus_random",
            "run 2 psiC - random all-query held-out accuracy",
            tier="headline",
            seen=SEEN_PTS,
        )
    ]
    for b in ("gap_eq_1", "gap_2_to_M", "gap_gt_M"):
        out.append(
            stat(
                f"r2vr.acc.psiC-random.{b}",
                "accuracy",
                b,
                "run2_psiC_minus_random",
                f"run 2 psiC - random accuracy, {b} bucket",
                tier="secondary",
                seen=SEEN_PTS,
            )
        )
    out.append(
        stat(
            "r2vr.acc.fifo-random",
            "accuracy",
            "all",
            "fifo_minus_random",
            "FIFO - random all-query accuracy",
            tier="secondary",
            seen=SEEN_PTS,
        )
    )
    out.append(
        stat(
            "r2vr.acc.run1_psiC-random",
            "accuracy",
            "all",
            "run1_psiC_minus_random_SEEN",
            "run 1 psiC - random all-query accuracy",
            tier="secondary",
            seen=SEEN_R1,
        )
    )
    out.append(
        stat(
            "r2vr.residency.psiC-random",
            "residency",
            "all",
            "run2_psiC_minus_random",
            "run 2 psiC - random residency (share of queries whose queried assert "
            "is resident), all queries",
            tier="secondary",
            seen=UNSEEN_RES,
        )
    )
    return out
