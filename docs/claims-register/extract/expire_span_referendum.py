"""Expire-Span referendum (experiments/expire-span-referendum; PREREG 3d5ed87 + A1
8ed8cbb (delta) + A2 13fc843 (NaN fix, full re-run), all ancestors of the launch head
99646cb; EVAL read 2026-10-06 23:49 local after selection; verified at 3a241fc).

"psiC" in this run is RSR run 1's psiC (PREREG §7.1: run 1's held-out records), not
run 2's. Every number is recomputed from raw records (ES's archived EVAL records, run 1's
archived held-out records, the archived VAL summaries) through the run's own
``paired_seed`` / ``classify`` / ``selection`` (``recompute_es.py``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

EXPERIMENT = "expire-span-referendum"
LEDGER = "runs/expire-span-referendum/ledger.json"
SEEDS = [0, 1, 2]
CMD = f"{C.SLOT}{C.PY} docs/claims-register/recompute_es.py"
BOOT = (
    "paired per-document bootstrap percentile, run 1's weights boot_weights(2048, s, 2000)"
)
P_HEAD = f"{C.PREREG} (§7.1 headline / §7.2 class; PREREG 3d5ed87, A2 13fc843)"
P_SEC = f"{C.PREREG} (§7.1 secondary, reported, never classified)"
U = (
    C.UNSEEN,
    "ES's EVAL records first read once, after selection and the class rule (A2.6); the "
    "run-1 records it is paired with were read in run 1, this contrast was not",
)
S_R1 = (C.SEEN, "run 1's own held-out records, read in run 1")


def _led() -> dict:
    return C.jload(C.ROOT / LEDGER)


def _check(key: str, expected, label: str) -> C.Check:
    return C.Check(label=label, cmd=CMD, expected=C.fmt_exact(expected), kind="raw", pick=key)


def contrast(rid, con, claim, *, tier="secondary", prereg=P_SEC, seen=U, note=None):
    d = _led()
    vals, cis, checks = [], [], []
    for s in SEEDS:
        c = d["seeds"][str(s)]["con"][con]
        vals.append(c["point"])
        cis.append((c["lo"], c["hi"]))
        for f in ("point", "lo", "hi"):
            checks.append(_check(f"seeds.{s}.con.{con}.{f}", c[f], f"s{s} {f}"))
    readings = "/".join(d["seeds"][str(s)]["con"][con]["reading"] for s in SEEDS)
    return C.Row(
        id=rid, experiment=EXPERIMENT, tier=tier, claim=f"{claim} (readings: {readings})",
        seeds=SEEDS, values=vals, ci=cis, ci_method=BOOT,
        source=f"{LEDGER} :: seeds.<s>.con.{con}.{{point,lo,hi}}",
        prereg=prereg, seen=seen[0], seen_why=seen[1], checks=checks, note=note,
    )


def level(rid, arm, claim, seen):
    d = _led()
    vals, cis, checks = [], [], []
    for s in SEEDS:
        c = d["seeds"][str(s)]["acc"]["all"][arm]
        vals.append(c["point"])
        cis.append((c["lo"], c["hi"]))
        for f in ("point", "lo", "hi"):
            checks.append(_check(f"seeds.{s}.acc.all.{arm}.{f}", c[f], f"s{s} {f}"))
    return C.Row(
        id=rid, experiment=EXPERIMENT, tier="secondary", claim=claim, seeds=SEEDS,
        values=vals, ci=cis, ci_method=BOOT,
        source=f"{LEDGER} :: seeds.<s>.acc.all.{arm}.{{point,lo,hi}}",
        prereg=P_SEC, seen=seen[0], seen_why=seen[1], checks=checks,
    )


def rows() -> list[C.Row]:
    d = _led()
    out: list[C.Row] = []
    out.append(contrast(
        "es.acc.es-psiC_run1", "acc.all.es_minus_psiC",
        "Expire-Span (selected config) - run 1's psiC, all-query held-out accuracy "
        "(the referendum's headline)", tier="headline", prereg=P_HEAD,
    ))
    r = contrast(
        "es.acc.es-fifo", "acc.all.es_minus_fifo",
        "Expire-Span - FIFO, all-query held-out accuracy", tier="headline",
    )
    # wording derived from the numbers, not typed: sign mix and CI exclusion of 0
    signs = {(lo > 0) - (hi < 0) for lo, hi in r.ci}
    excl = all(lo > 0 or hi < 0 for lo, hi in r.ci)
    if excl and signs == {1, -1}:
        r.claim += "; every CI excludes 0 and the sign is mixed across seeds"
    out.append(r)
    out.append(C.Row(
        id="es.class", experiment=EXPERIMENT, tier="headline",
        claim=f"§7.2 class (SUPERSEDES iff ES - psiC CI lower > 0 on all 3 seeds); "
              f"falsifier_6 = {d['falsifier_6']}",
        seeds=SEEDS, values=[d["class"]] * 3,
        source=f"{LEDGER} :: class; falsifier_6",
        prereg=P_HEAD, seen=U[0], seen_why=U[1],
        checks=[_check("class", d["class"], "class"),
                _check("falsifier_6", d["falsifier_6"], "falsifier_6")],
        note="one class for the run; repeated per seed",
    ))
    out.append(C.Row(
        id="es.delta", experiment=EXPERIMENT, tier="secondary",
        claim="delta per seed (TIES margin; B2's FIT_VAL delta, rounded to 6 dp in the "
              "PREREG table)",
        seeds=SEEDS, values=[d["delta"][str(s)] for s in SEEDS],
        source=f"{LEDGER} :: delta.<s> (parsed from PREREG A1.1 by delta_per_seed)",
        prereg=f"{C.PREREG} (Amendment 1, 8ed8cbb)", seen=C.SEEN,
        seen_why="B2's measured delta, known before the amendment",
        checks=[_check(f"delta.{s}", d["delta"][str(s)], f"s{s}") for s in SEEDS],
    ))
    sel = d["selection"]
    row = next(r for r in sel["table"] if r["config"] == sel["selected"])
    out.append(C.Row(
        id="es.selection", experiment=EXPERIMENT, tier="secondary",
        claim=f"selected config {sel['selected']} (max mean VAL all-query accuracy over "
              f"seeds; tied at max: {sel['tied_at_max']}); value = its VAL accuracy per seed",
        seeds=SEEDS, values=row["per_seed"],
        source=f"{LEDGER} :: selection.selected; selection.table[config={sel['selected']}]",
        prereg=f"{C.PREREG} (§6.3 rule)", seen=C.UNSEEN,
        seen_why="VAL [994000,994512) read once by the §6.3 rule; no VAL read before (A2.6)",
        checks=[_check("selection.selected", sel["selected"], "selected"),
                _check("selection.tied_at_max", sel["tied_at_max"], "tied"),
                _check("selection.table", sel["table"], "table (all 9 configs)")],
    ))
    out.append(level("es.acc.es", "es", "Expire-Span all-query held-out accuracy", U))
    out.append(level("es.acc.fifo", "fifo", "FIFO all-query held-out accuracy (run 1's pass)",
                     S_R1))
    out.append(level("es.acc.psiC_run1", "psiC",
                     "run 1's psiC all-query held-out accuracy", S_R1))
    for b in ("gap_eq_1", "gap_2_to_M", "gap_gt_M"):
        out.append(contrast(f"es.acc.es-fifo.{b}", f"acc.{b}.es_minus_fifo",
                            f"Expire-Span - FIFO accuracy, {b} bucket"))
    for b in ("gap_2_to_M", "gap_gt_M"):
        out.append(contrast(f"es.acc.es-psiC_run1.{b}", f"acc.{b}.es_minus_psiC",
                            f"Expire-Span - run 1's psiC accuracy, {b} bucket"))
    out.append(contrast("es.nll.real.es-fifo", "nll.real.es_minus_fifo",
                        "Expire-Span - FIFO real-token NLL (positive = ES worse)"))
    out.append(contrast("es.nll.real.es-psiC_run1", "nll.real.es_minus_psiC",
                        "Expire-Span - run 1's psiC real-token NLL"))
    return out
