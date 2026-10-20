"""E0d (experiments/e0d; PREREG 5357ad2 + A1 84e21c5 + A2 8c63ecd + A3 f5a8752, all
committed 2026-09-27..29 before the 2026-09-29T20:11Z run; verified at verify/e0d).

No raw per-cell record of this run survives (the archive holds only ledger, manifest,
claims and verification; no seed cache was kept), so no number is recomputed from raw
records: re-measuring needs the full ~2.1 h LOO + r_i measurement over D_E0d
[262144, 263168) on 3 seeds (ledger started 20:11:44Z, finished 22:20:09Z) -- over the
10-minute bound. Each number is checked against the run's own out-root ledger in the
archive (``archive-ledger``), and the class is recomputed from the ledger's labels by
the run's own ``classify``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

EXPERIMENT = "e0d"
RUN = "e0d"
SEEDS = [0, 1, 2]
ARCH_LEDGER = "/Users/keanooo7/rsr-runs-archive/e0d/runs/e0d/ledger.json"
FLAT = f"{C.PY} docs/claims-register/flat_json.py"
COST = (
    "not recomputed from raw (cost): no per-cell raw record survives; re-measuring is the "
    "full ~2.1 h E0d run"
)
BOOT = "per-document cluster bootstrap percentile, n_boot 2000, 1024 docs, seed 20260927"
P = f"{C.PREREG} (E0d PREREG + A1/A2/A3, last pre-data commit f5a8752)"
P_SEC = f"{C.PREREG} (A2.7 secondary, reported only, never gates)"
U = (
    C.UNSEEN,
    "D_E0d [262144,263168) first read by this run; but the AUROC statistic itself was "
    "adopted in A2 after the rho statistics were inspected on set E [64,1088)",
)
U_RHO = (
    C.UNSEEN,
    "D_E0d first read by this run; the same statistics had been read on set E [64,1088) "
    "(A2 design), where they were already low",
)
POPS = {"A1.3": "bos_excluded", "all cells": "all"}


def _rows():
    return C.lrows(RUN)


def _check(rowkey: str, sub: str, expected, label: str) -> C.Check:
    return C.Check(
        label=label,
        cmd=f"{FLAT} {ARCH_LEDGER}",
        expected=C.fmt_exact(expected),
        kind="archive-ledger",
        pick=f"{rowkey}|value.{sub}" if sub else f"{rowkey}|value",
    )


def stat(
    rid: str, claim: str, sub_point: str, sub_ci: str | None, *, tier="secondary",
    prereg=P_SEC, seen=U_RHO, note=None,
) -> C.Row:
    L = _rows()
    vals, cis, checks = [], [], []
    for s in SEEDS:
        a = L[f"seed{s}.analysis"]
        v = C.dig(a, sub_point)
        vals.append(v)
        checks.append(_check(f"seed{s}.analysis", sub_point, v, f"s{s} point"))
        if sub_ci:
            ci = C.dig(a, sub_ci)
            cis.append((ci[0], ci[1]))
            checks.append(_check(f"seed{s}.analysis", sub_ci, ci, f"s{s} ci"))
    return C.Row(
        id=rid, experiment=EXPERIMENT, tier=tier, claim=claim, seeds=SEEDS, values=vals,
        ci=cis if sub_ci else None,
        ci_method=BOOT if sub_ci else "none computed",
        source=f"runs/{RUN}/ledger.json :: seed{{s}}.analysis -> {sub_point}"
        + (f" (+ {sub_ci.rsplit('.', 1)[-1]})" if sub_ci else ""),
        prereg=prereg, seen=seen[0], seen_why=seen[1], checks=checks,
        rederive_note=None, note=note,
    )


def rows() -> list[C.Row]:
    L = _rows()
    out: list[C.Row] = []
    star = L["seed0.analysis"]["auroc_star"]
    g = "populations.{p}.a2.gated_resample.{f}"
    r = stat(
        "e0d.auroc_pct", "AUROC_strat,pct of gated r_i vs resample-LOO positives, "
        f"A1.3 population (the gate: CI lower >= A* = {star}). Scope (RESULTS.md:23-33): "
        "~92% of positives (set-E figure) are the queried assert at its own query step; "
        "'E0d does NOT show that r_i predicts later demand' (RESULTS.md:33)",
        g.format(p="bos_excluded", f="r.pct"), g.format(p="bos_excluded", f="r.pct_ci"),
        tier="headline", prereg=P, seen=U,
    )
    # the across-seed 'primary' statistic row must hold the same three samples
    r.checks.append(C.Check(
        label="primary.AUROC_strat_pct samples", cmd=f"{FLAT} {ARCH_LEDGER}",
        expected=C.fmt_exact(list(r.values)), kind="archive-ledger",
        pick="primary.AUROC_strat_pct|samples"))
    out.append(r)
    labels = L["e0d.labels"]
    out.append(C.Row(
        id="e0d.label", experiment=EXPERIMENT, tier="headline",
        claim=f"per-seed A2.8 labels; class {L['e0d.class']} (scope: agreement at the moment "
              "of retrieval only, RESULTS.md:23-33)",
        seeds=SEEDS, values=labels,
        source=f"runs/{RUN}/ledger.json :: e0d.labels; e0d.class",
        prereg=P, seen=U[0], seen_why=U[1],
        checks=[
            _check("e0d.labels", "", labels, "labels"),
            _check("e0d.class", "", L["e0d.class"], "class"),
            C.Check(
                label="class recomputed from the ledger's labels by e0d/run.py classify",
                cmd=(f"{C.PY} -c \"import importlib.util as u, json; "
                     "s=u.spec_from_file_location('e0d_run', 'experiments/e0d/run.py'); "
                     "m=u.module_from_spec(s); s.loader.exec_module(m); "
                     "r=json.load(open('runs/e0d/ledger.json'))['rows']; "
                     "print(m.classify([x['value'] for x in r "
                     "if x['key']=='e0d.labels'][-1])[0])\""),
                expected=C.fmt_exact(L["e0d.class"]), kind="ledger",
            ),
        ],
    ))
    out.append(stat(
        "e0d.auroc_pct.all_cells", "AUROC_strat,pct, gated r_i x resample, all cells "
        "(no A1.3 exclusion)", g.format(p="all", f="r.pct"), g.format(p="all", f="r.pct_ci"),
        prereg=P, seen=U,
    ))
    out.append(stat(
        "e0d.auroc_pct.true_demand", "AUROC_strat,pct of the true_demand control (C10', "
        "row 0), A1.3", g.format(p="bos_excluded", f="pc.pct"),
        g.format(p="bos_excluded", f="pc.pct_ci"), prereg=P, seen=U,
    ))
    st = "populations.{p}.stats.{sc}.resample.{f}"
    for lab, p in POPS.items():
        tag = "A13" if p == "bos_excluded" else "all"
        out.append(stat(
            f"e0d.bottom1_Q.{tag}", f"bottom-1 agreement on Q-steps, gated r_i x resample, "
            f"{lab} population (chance per the ledger: see e0d.bottom1_chance)",
            st.format(p=p, sc="gated", f="bottom1_Q"),
            st.format(p=p, sc="gated", f="bottom1_Q_ci"),
        ))
        out.append(stat(
            f"e0d.bottom1_full.{tag}", f"bottom-1 agreement on all full-memory steps, gated "
            f"r_i x resample, {lab} population (the RESULTS.md A1 table's column)",
            st.format(p=p, sc="gated", f="bottom1_full"),
            st.format(p=p, sc="gated", f="bottom1_full_ci"),
        ))
        out.append(stat(
            f"e0d.rho_Q.{tag}", f"Spearman rho_Q (Q-steps), gated r_i x resample, {lab}",
            st.format(p=p, sc="gated", f="rho_Q"), st.format(p=p, sc="gated", f="rho_Q_ci"),
        ))
    out.append(stat(
        "e0d.rho_Q_rank.A13", "rho_Q,rank, gated r_i x resample, A1.3",
        st.format(p="bos_excluded", sc="gated", f="rho_Q_rank"),
        st.format(p="bos_excluded", sc="gated", f="rho_Q_rank_ci"),
    ))
    out.append(stat(
        "e0d.rho_pool.A13", "rho_pool, gated r_i x resample, A1.3",
        st.format(p="bos_excluded", sc="gated", f="rho_pool"),
        st.format(p="bos_excluded", sc="gated", f="rho_pool_ci"),
    ))
    out.append(stat(
        "e0d.true_demand.rho_Q_rank.A13", "the true_demand control's own rho_Q,rank, A1.3 "
        "(the ceiling of the rho metric)",
        st.format(p="bos_excluded", sc="pc", f="rho_Q_rank"),
        st.format(p="bos_excluded", sc="pc", f="rho_Q_rank_ci"),
    ))
    out.append(stat(
        "e0d.true_demand.bottom1_full.A13", "the true_demand control's bottom-1, all "
        "full-memory steps, A1.3",
        st.format(p="bos_excluded", sc="pc", f="bottom1_full"),
        st.format(p="bos_excluded", sc="pc", f="bottom1_full_ci"),
    ))
    out.append(stat(
        "e0d.true_demand.bottom1_Q.A13", "the true_demand control's bottom-1 on Q-steps, A1.3",
        st.format(p="bos_excluded", sc="pc", f="bottom1_Q"),
        st.format(p="bos_excluded", sc="pc", f="bottom1_Q_ci"),
    ))
    out.append(stat(
        "e0d.bottom1_chance", "the bottom-1 chance level the run used (A1.3 population)",
        st.format(p="bos_excluded", sc="gated", f="bottom1_chance"), None,
        note="the run uses 1/M for both populations although A1.3 leaves rank M-1 out of "
             "its steps; the 1/16 vs 1/15 question is e0d follow-up m-5",
    ))
    out.append(C.Row(
        id="e0d.class_A1", experiment=EXPERIMENT, tier="secondary",
        claim=f"A1 rho labels at rho*=0.5 (reported only); class_A1 {L['e0d.class_A1']}",
        seeds=SEEDS, values=L["e0d.labels_A1"],
        source=f"runs/{RUN}/ledger.json :: e0d.labels_A1; e0d.class_A1",
        prereg=P_SEC, seen=U_RHO[0], seen_why=U_RHO[1],
        checks=[_check("e0d.labels_A1", "", L["e0d.labels_A1"], "labels_A1"),
                _check("e0d.class_A1", "", L["e0d.class_A1"], "class_A1")],
    ))
    tau = [L[f"seed{s}.analysis"]["tau"]["resample"] for s in SEEDS]
    out.append(C.Row(
        id="e0d.tau_resample", experiment=EXPERIMENT, tier="secondary",
        claim="frozen per-seed tau (resample) defining LOO positives (A2.2)",
        seeds=SEEDS, values=tau,
        source=f"runs/{RUN}/ledger.json :: seed{{s}}.analysis -> tau.resample",
        prereg=P, seen=C.SEEN,
        seen_why="computed on the already-inspected set E [64,128) and frozen in A2.2",
        checks=[_check(f"seed{s}.analysis", "tau.resample", tau[s], f"s{s}") for s in SEEDS],
    ))
    for r in out:
        if all(c.kind != "raw" for c in r.checks):
            r.note = (r.note + "; " if r.note else "") + COST
    return out
