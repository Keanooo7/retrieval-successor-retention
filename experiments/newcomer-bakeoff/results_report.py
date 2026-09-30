# ruff: noqa: E501, RUF001
"""Write `experiments/newcomer-bakeoff/RESULTS.md` from the run's ledger.

Every number below is read from `runs/newcomer-bakeoff/ledger.json` (and its
`manifest.json`); none is typed. `--check` regenerates in memory and exits 1 if the
committed RESULTS.md differs (the verifier's claim).

    .venv/bin/python experiments/newcomer-bakeoff/results_report.py [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys_path_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(sys_path_root / "src"))

from rsr.exit_codes import Exit, run_main  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "newcomer-bakeoff"
OUT = ROOT / "experiments" / "newcomer-bakeoff" / "RESULTS.md"
SEEDS = (0, 1, 2)
BUCKETS = ("all", "gap_2_to_M", "gap_gt_M", "gap_eq_M")
NAME = {
    "psiC": "(a) ψ̂-C",
    "psiU": "(b) ψ̂-U",
    "graceU1": "(c) ψ̂-U + grace g=1",
    "graceU2": "(c) ψ̂-U + grace g=2",
    "graceU4": "(c) ψ̂-U + grace g=4",
    "psiR1": "(d) ψ̂-R1 (λ_shadow 0.5)",
    "psiR1L1": "(d′) ψ̂-R1λ1 (non-gating)",
    "psiU+": "(e) ψ̂-U⁺",
    "psiC+": "(e) ψ̂-C⁺",
    "fifo": "FIFO",
    "lru": "LRU",
    "ageU": "age-only (U)",
    "ageC": "age-only (C)",
    "ageR1": "age-only (R1)",
    "ageR1L1": "age-only (R1λ1)",
    "random": "random (5-seed mean)",
    "kind": "kind-oracle (random tie)",
    "oracle": "oracle",
}
GATED = (
    "psiC",
    "psiU",
    "graceU1",
    "graceU2",
    "graceU4",
    "psiR1",
    "psiR1L1",
    "psiU+",
    "psiC+",
)
SHOWN = (
    *GATED,
    "fifo",
    "lru",
    "ageU",
    "ageC",
    "ageR1",
    "ageR1L1",
    "random",
    "kind",
    "oracle",
)
EVICTING = tuple(a for a in SHOWN if a != "random") + tuple(
    f"random{k}" for k in range(5)
)


def f4(x):
    return "—" if x is None else f"{x:+.4f}"


def p4(x):
    return "—" if x is None else f"{x:.4f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def ci(d):
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def acc_ci(d):
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def load():
    led = json.loads((RUN / "ledger.json").read_text())
    man = json.loads((RUN / "manifest.json").read_text())
    rows = {r["key"]: r["value"] for r in led["rows"]}
    return led, man, rows


def render() -> str:
    led, man, R = load()
    L: list[str] = []
    w = L.append
    prov = led["provenance"]
    dq = {
        k: R[f"decision.{k}"]
        for k in ("DQ1", "DQ2", "DQ3", "replication", "replication_companion")
    }
    w("# Newcomer bake-off — results (offline, ckpt3000, γ = 0.9)")
    w("")
    w(
        "**Generated** by `experiments/newcomer-bakeoff/results_report.py` from "
        "`runs/newcomer-bakeoff/ledger.json`; no number below is typed. Written by a model "
        "(Claude Opus 5.5, an `rsr-researcher` session). **Not written by Brendan.**"
    )
    w("")
    w(
        f"- PREREG: `experiments/newcomer-bakeoff/PREREG.md` (commit `{man['prereg_commit']}`, "
        "erratum E1 `6897548`)."
    )
    w(
        f"- Code that ran: git `{prov.get('git_sha')}` (dirty: {prov.get('git_dirty')}); "
        f"`run.py` sha256 `{man['code_sha256'][:16]}…`; config hash `{led['config_hash']}`."
    )
    w(
        f"- Status `{led['status']}`; seeds actually run {led['seeds_actually_run']}; device "
        f"`{led['device']}`; EVAL_NB {man['eval_nb']} (N_E = {man['N_E']} per seed)."
    )
    for c in led["commands"]:
        if c["note"] == "the parent":
            w(f"- Parent command: `{c['argv']}` → exit {c['exit_code']}.")
    t0s, t0e = R["T0.start"], R["T0.end"]
    w(
        f"- T0 at start: in-process ok={t0s['in_process']['ok']} "
        f"(n={t0s['in_process'].get('n')}); `shasum -c` rc {t0s['shasum']['rc']}, "
        f"{t0s['shasum']['n_ok']} OK. At end: in-process ok={t0e['in_process']['ok']}; "
        f"`shasum -c` rc {t0e['shasum']['rc']}, {t0e['shasum']['n_ok']} OK."
    )
    w("")
    w("## 0. What this is and is not")
    w("")
    w(
        "- **Offline probe evidence.** Closed-form ridge heads on FIFO-world captures, run as "
        "fixed eviction rules. **It says nothing about on-policy behaviour after `T_warm`.**"
    )
    w("- **No build recommendation is made by this run.**")
    w(
        "- **Whether the grace rule (arm c) is acceptable under the age prohibition is "
        "OWNER-ONLY, whatever the numbers below say.**"
    )
    w("")
    w("## 1. Headline: the decision table (PREREG §7)")
    w("")
    d1, d2, d3 = dq["DQ1"], dq["DQ2"], dq["DQ3"]
    w(
        f"- **DQ1 — does grace buy anything over (a) ψ̂-C? → {d1['class']}.** "
        "Per-seed labels of grace_g − ψ̂-C (seeds 0/1/2): "
        + "; ".join(f"g={g}: {' / '.join(v)}" for g, v in d1["labels"].items())
        + "."
    )
    w(
        f"- **DQ2 — does (d) ψ̂-R1 land near (a) or (b)? → {d2['class']}.** Per seed: "
        f"{' / '.join(d2['per_seed'])}; (d)'s §9.5 outcome vs its ref: "
        f"{' / '.join(d2['psiR1_outcome'])}."
    )
    pis = []
    for p in d2["pi"]:
        pis.append(
            "undefined"
            if not p["defined"]
            else f"{p['point']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]"
        )
    w(
        f"  - π = (acc_R1 − acc_U)/(acc_C − acc_U) (0 = at U, 1 = at C): {' / '.join(pis)}."
    )
    w(
        f"- **DQ3 — does the harm travel with the counterfactual target? → {d3['class']}.** "
        f"Harm seeds (ψ̂-U LOSS on EVAL_NB): {d3['harm_seeds']}."
    )
    for s, v in d3.get("per_seed", {}).items():
        w(
            f"  - seed {s}: {v['label']} (R1λ1 − C label {v['contrast_label']}"
            f"; acc C {v['acc_C']:.4f} ≥ R1 {v['acc_R1']:.4f} ≥ R1λ1 {v['acc_R1L1']:.4f}: "
            f"{v['monotone']})"
        )
    rep, comp = dq["replication"], dq["replication_companion"]
    w(
        f"- **B2 replication on fresh documents (non-binding):** §9.6 row {rep['row']}, "
        f"**{rep['class']}** — ψ̂-U {' / '.join(rep['psiU'])}; ψ̂-C {' / '.join(rep['psiC'])}. "
        f"Companion (U⁺, C⁺): row {comp['row']}, {comp['class']}."
    )
    w("")
    w("## 2. Every gated arm against its reference (B2 §9.5 + A1.13, δ from B2 phase A)")
    w("")
    w(
        "| seed | arm | ref | δ | Δ all [95% CI] | gap_2_to_M Δ lo | arm − random [lo] | outcome |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        OUT, dl = R[f"seed{s}.outcome"], R[f"seed{s}.delta"]
        for a in GATED:
            o = OUT[a]
            w(
                f"| {s} | {NAME[a]} | {NAME.get(o['ref'], o['ref'])} | {dl:.4f} | {ci(o)} | "
                f"{o['noninf_lo']:+.4f} | {o['rand_point']:+.4f} [{o['rand_lo']:+.4f}] | "
                f"**{o['label']}**{' ' + ','.join(o['flags']) if o['flags'] else ''} |"
            )
    w("")
    w("## 3. Decision contrasts (paired, all-query), labels against δ")
    w("")
    names = [
        "graceU1-psiC",
        "graceU2-psiC",
        "graceU4-psiC",
        "psiR1-psiC",
        "psiR1-psiU",
        "psiC-psiR1",
        "psiR1-psiR1L1",
        "psiR1L1-psiC",
        "psiU-psiR1L1",
        "psiU-psiU+",
        "psiC-psiU",
        "oracle-fifo",
        "kind-fifo",
        "lru-fifo",
    ]
    w("| contrast | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
    w("|---|" + "---|" * len(SEEDS))
    for n in names:
        cells = []
        for s in SEEDS:
            c = R[f"seed{s}.contrasts"][n]
            cells.append(f"{ci(c)} {c['label']}")
        w(f"| {n} | " + " | ".join(cells) + " |")
    w("")
    w("## 4. Accuracy per arm (model-read, pooled over queries, paired percentile CI)")
    for s in SEEDS:
        A = R[f"seed{s}.acc"]
        w("")
        w(f"**Seed {s}.**")
        w("")
        w("| arm | " + " | ".join(BUCKETS) + " |")
        w("|---|" + "---|" * len(BUCKETS))
        for a in SHOWN:
            w(f"| {NAME[a]} | " + " | ".join(acc_ci(A[b][a]) for b in BUCKETS) + " |")
    w("")
    w("## 5. Residency (in-loop; equal to the model-free replay by control C7)")
    for s in SEEDS:
        Rs = R[f"seed{s}.residency"]
        w("")
        w(
            f"**Seed {s}.** "
            + "; ".join(
                f"{NAME[a]}: " + " / ".join(p4(Rs[b][a]) for b in BUCKETS)
                for a in SHOWN
                if a != "random"
            )
            + f"  (buckets {' / '.join(BUCKETS)})"
        )
    w("")
    w("## 6. Newcomer and pending-assert evictions (§6.2) and grace flips (§6.4)")
    w("")
    w(
        "| seed | arm | evictions | age-1 share | age-1 pending (share) | pending victims, "
        "query within gap M | past M | mean victim age | grace flip share |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        for a in (a for a in EVICTING if not a.startswith("random")):
            r = R[f"seed{s}.readouts.{a}"]
            w(
                f"| {s} | {NAME[a]} | {r['n_evictions']} | {pct(r['age1_share'])} | "
                f"{r['age1_pending']} ({pct(r['age1_pending_share'])}) | {r['pending_le_M']} | "
                f"{r['pending_gt_M']} | {p4(r['mean_victim_age'])} | "
                f"{pct(r.get('grace_flip_share'))} |"
            )
    w("")
    w(
        'RESEARCH-CONTEXT §4.4, quoted, not converted into a threshold: *"if grace flips a '
        'large share, grace is the policy."*'
    )
    w("")
    w("## 7. §7.1 vacuity on the EFFECTIVE rule (§6.3)")
    w("")
    w(
        "Partial ρ(victim indicator, age | content), within step; `raw` is without the content "
        "regression. For ψ̂ and age arms the score-level ρ(score, age | content) is beside it. "
        "**No threshold is read**; FIFO and LRU are the recency references."
    )
    w("")
    w(
        "| seed | arm | decision partial ρ | decision raw ρ | score partial ρ | score raw ρ | "
        "vs its age-only head | vs LRU |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        C = R[f"seed{s}.contrasts"]
        for a in (a for a in EVICTING if not a.startswith("random")):
            v = R[f"seed{s}.readouts.{a}"]["vacuity"]
            dd, sc = v["decision"], v["score"]
            age_c = C.get(f"{a}-age")
            lru_c = C.get(f"{a}-lru")
            w(
                f"| {s} | {NAME[a]} | {f4(dd['partial'])} | {f4(dd['raw'])} | "
                f"{f4(sc['partial']) if sc else '—'} | {f4(sc['raw']) if sc else '—'} | "
                f"{(ci(age_c) + ' ' + age_c['label']) if age_c else '—'} | "
                f"{(ci(lru_c) + ' ' + lru_c['label']) if lru_c else '—'} |"
            )
    w("")
    w("## 8. ADR-0006 displacement histograms (victim_rank − 1; FIFO ≡ 0)")
    w("")
    for s in SEEDS:
        for a in (
            "psiC",
            "psiU",
            "graceU1",
            "graceU2",
            "graceU4",
            "psiR1",
            "psiR1L1",
            "psiU+",
            "psiC+",
            "lru",
            "fifo",
            "oracle",
            "kind",
        ):
            h = R[f"seed{s}.readouts.{a}"]["displacement_hist"]
            tot = sum(h.values())
            w(
                f"- seed {s} {NAME[a]}: "
                + ", ".join(
                    f"{k}: {v} ({100 * v / tot:.1f}%)"
                    for k, v in sorted(h.items(), key=lambda kv: int(kv[0]))
                )
            )
    w("")
    w("## 9. Victim age histograms (ages 1 … 47)")
    w("")
    for s in SEEDS:
        for a in (
            "psiC",
            "psiU",
            "graceU1",
            "graceU2",
            "graceU4",
            "psiR1",
            "psiR1L1",
            "lru",
        ):
            h = R[f"seed{s}.readouts.{a}"]["age_hist"]
            w(
                f"- seed {s} {NAME[a]}: "
                + " ".join(str(x) for x in h[:20])
                + f" … (ages ≥ 21: {sum(h[20:])})"
            )
    w("")
    w("## 10. The (d) fits and the fit-side controls (FIT_TRAIN / FIT_VAL only)")
    w("")
    for s in SEEDS:
        f = R[f"fit.seed{s}"]
        c = f["controls"]
        w(
            f"**Seed {s}.** λ: "
            + ", ".join(f"{k} {v:g}" for k, v in f["lam"].items())
            + ". Selected residuals: "
            + ", ".join(f"{k} {v:.2e}" for k, v in f["resid"].items())
            + f". ref (FIT_VAL): {f['refs']}. FIT_VAL accuracy: "
            + ", ".join(f"{k} {v:.4f}" for k, v in f["fit_val_acc"].items())
            + f". K-truncated cells: {f['k_truncated_cells']}. Rows: {f['n']}."
        )
        c5 = c["C5"]
        w(
            f"- C5 (the (d) pipeline is B2's): λ B2 {c5['lam_b2']:g} / new {c5['lam_new']:g}; "
            f"val MSE rel diff {c5['val_mse_rel']:.2e}; argmin agreement {c5['agree']}/{c5['n']} "
            f"= {c5['rate']:.5f}; max |Δw| {c5['w_max_abs_diff']:.3e}; bit-identical "
            f"{c5['w_bit_identical']}; ok {c5['ok']}."
        )
        w(
            "- C6 (tree reproduces B2's FIT_VAL): "
            + "; ".join(
                f"{k} {v['new']!r} vs {v['b2']!r} equal={v['equal']}"
                for k, v in c["C6"].items()
            )
        )
        w(
            f"- C4 identity worst {c['C4']['identity_worst']}, Σr worst {c['C4']['sum_worst']:.2e}; "
            f"C8 determinism {c['C8']}; C9 grace off-switch {c['C9_grace_off']}; B2 §11.3 "
            f"control ok {c['C3_b2']['ok']}."
        )
        w(
            f"- seconds {f['seconds']}; threads {f['threads']}; peak RSS {f['peak_rss_gb']:.2f} GB."
        )
        w("")
    w("## 11. Unfavourable and unexpected numbers (listed, not interpreted away)")
    w("")
    w(
        "Every table above is complete; nothing is omitted for being unfavourable. See the "
        "report `~/Documents/RSR-2026-09-29-day/reports/NEWCOMER-BAKEOFF.md` for the "
        "researcher's reading and the BELIEVED-NOT-VERIFIED list."
    )
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    text = render()
    if "--check" in argv:
        return Exit.OK if OUT.exists() and OUT.read_text() == text else Exit.FAIL
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
