"""Prose that is checked against the register (PLAN-2026-10-19 phase 0.2 / 0.3).

``NUMERIC``: a figure quoted verbatim from the prose (PLAN.md is the PM's copy at
``~/Documents/RSR-2026-10-19-review/PLAN.md``; it is not edited here) and the register
row it should equal. ``render`` rounds the row's per-seed value to the quote's own
precision and marks each seed agree / DISAGREE; the ledger side is never typed.

``FINDINGS``: statements in the prose that the records contradict or qualify, each
citing the register rows that show it; ``render`` prints those rows' values and CIs.
"""

# ruff: noqa: RUF001  -- the quotes keep PLAN's own minus sign (U+2212) verbatim

from __future__ import annotations

PLAN = "PLAN.md"  # ~/Documents/RSR-2026-10-19-review/PLAN.md (PM's copy)

#: (where, what, quoted per-seed figures as printed in the prose, row id)
NUMERIC = [
    (
        f"{PLAN}:30",
        "psiC - FIFO accuracy",
        ["−0.072", "−0.045", "−0.083"],
        "run2.acc.psiC-fifo",
    ),
    (
        f"{PLAN}:31",
        "psiC - random accuracy",
        ["−0.047", "−0.022", "−0.038"],
        "r2vr.acc.psiC-random",
    ),
    (
        f"{PLAN}:36",
        "psiC real-token NLL",
        ["1.417", "1.395", "1.431"],
        "run2.nll.real.psiC",
    ),
    (
        f"{PLAN}:37",
        "FIFO real-token NLL",
        ["1.351", "1.356", "1.345"],
        "run2.nll.real.fifo",
    ),
    (f"{PLAN}:42", "oracle accuracy", ["0.922", "0.912", "0.928"], "run2.acc.oracle"),
    (f"{PLAN}:42", "FIFO accuracy", ["0.775", "0.767", "0.803"], "run2.acc.fifo"),
    (f"{PLAN}:44", "LS cap on U-val", ["0.39", "0.53", "0.35"], "ls.cf_all.cap_val"),
    (f"{PLAN}:44", "LS cap in-sample", ["0.41", "0.55", "0.39"], "ls.cf_all.cap_train"),
    (f"{PLAN}:46", "B0 ko cap", ["+1.00", "−0.95", "+1.00"], "b0.cap.ko.g09"),
    (
        f"{PLAN}:47",
        "ES - FIFO accuracy",
        ["+0.006", "−0.046", "−0.004"],
        "es.acc.es-fifo",
    ),
    (f"{PLAN}:52", "bottom-1", ["0.043", "0.028", "0.024"], "e0d.bottom1_Q.A13"),
    (f"{PLAN}:61", "B0 kb cap", ["0.90", "−0.29", "1.07"], "b0.cap.kb.g09"),
]

#: (where, the prose's statement, what the records show, row ids)
FINDINGS = [
    (
        f"{PLAN}:52",
        "Bottom-1 (0.043 / 0.028 / 0.024) and ρ (0.17–0.19) were ungated and have no CI.",
        "Both have per-seed CIs in E0d's ledger. 'Ungated' holds only as 'never gates' "
        "(A2.1). PLAN's bottom-1 figures are the Q-step bottom-1 of the A1.3 population "
        "(gated r_i), quoted from `experiments/b2-psi-probe/RESULTS.md:757`; E0d's own "
        "RESULTS table reports bottom-1 over all full-memory steps, a different "
        "statistic "
        "(rows below). The ledger's chance level is 1/16 for both populations although "
        "A1.3 leaves rank M−1 out (e0d follow-up m-5, unresolved).",
        [
            "e0d.bottom1_Q.A13",
            "e0d.bottom1_full.A13",
            "e0d.rho_Q.A13",
            "e0d.bottom1_chance",
        ],
    ),
    (
        f"{PLAN}:50",
        "E0d's AUROC of 0.94 positives are about 92% the queried assert at its own query "
        "step.",
        "That composition was measured on the already-inspected set E, not on E0d's data "
        "D_E0d; `experiments/e0d/RESULTS.md:27-30` says the ledger does not re-measure "
        "it. "
        "Attribute it to set E.",
        ["e0d.auroc_pct"],
    ),
    (
        f"{PLAN}:62",
        "the ridge path shows the cap rising as L_MC worsens, so cap and L_MC are "
        "misaligned.",
        "The rise is cf_live's only (the secondary arm); cf_all's ridge path is flat "
        "within its CIs. LS states the reading as an unregistered hypothesis on "
        "descriptive cf_live and U-val readings, never routed (`experiments/"
        "psi-ls-capacity/RESULTS.md` F1). It is a hypothesis, not grounds for rejecting "
        "a "
        "route. The ridge rows are not re-derived (cost).",
        [
            "ls.cf_live.ridge.cap@max_rho",
            "ls.cf_all.ridge.cap@max_rho",
            "ls.cf_live.cap_val",
            "ls.cf_all.cap_val",
        ],
    ),
    (
        f"{PLAN}:46",
        "A kind-mean ranking scores cap +1.00 / −0.95 / +1.00 … That is a 3-number rule "
        "beating the least-squares optimum on 2 of 3 seeds.",
        "The two caps are over different document sets: B0's ko cap is over all of U "
        "(1088 documents per seed, in-sample, SEEN), the LS cap over U-val only. The "
        "comparison is not paired; at most it is suggestive.",
        ["b0.cap.ko.g09", "ls.cf_all.cap_val"],
    ),
    (
        f"{PLAN}:37",
        "FIFO real-token NLL seed 1 = 1.356.",
        "The ledger value rounds to 1.355 (see the numeric table).",
        ["run2.nll.real.fifo"],
    ),
    (
        f"{PLAN}:47",
        "Expire-Span − FIFO is +0.006 / −0.046 / −0.004.",
        "Seed 2 rounds to −0.003. The reading (every CI excludes 0, sign mixed: small "
        "and "
        "inconsistent in sign) is unaffected.",
        ["es.acc.es-fifo"],
    ),
]
