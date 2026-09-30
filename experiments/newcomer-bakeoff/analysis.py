"""Newcomer bake-off: the pure half -- aggregation of EVAL units, the §6 readouts
and the §7 decision table. No model, no IO beyond what the caller hands in.

Governing text: `experiments/newcomer-bakeoff/PREREG.md` (commit 801635c). Every
rule here cites the PREREG section it transcribes. B2's own rules (the §9.5 outcome
with A1.13, the paired bootstrap, the attribution accumulator) are imported from
`experiments/b2-psi-probe/run.py`, never copied.

Kept separate from `run.py` on purpose: an EVAL unit is keyed on `run.py`'s sha256
(its measurement code), so a fix to the aggregation below cannot silently mix units
measured by different code, and does not force a rerun either.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from array import array
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: B2's runner: ProbeArgminPolicy, the capture, the ridge, the §9.5 rule, the
#: paired bootstrap, the attribution accumulator.
B2 = _load("_nb_b2_psi_probe", "experiments/b2-psi-probe/run.py")

SEEDS = [0, 1, 2]
GAMMA = 0.9
GRACE_GS = (1, 2, 4)  # PREREG §3 (c)
RANDOMS = tuple(f"random{k}" for k in range(B2.N_RANDOM))
BUCKETS = B2.BUCKETS  # all, gap_2_to_M, gap_gt_M, gap_eq_M (S0-03's)

#: PREREG §6.3: the content label of a live slot (the generator's kind; an
#: assert's status at that step). Descriptive, never an input to any psi-hat.
CONTENT_LABELS = (
    "assert:pending",
    "assert:querying",
    "assert:answered",
    "query",
    "filler",
)
CONTENT_INDEX = {c: k for k, c in enumerate(CONTENT_LABELS)}

#: PREREG §3 table: the 22 arm-runs per (document, seed), in run order (the
#: PREREG text says "24"; that is a miscount of its own table -- erratum E1).
ARMS = (
    "psiC",
    "psiU",
    *(f"graceU{g}" for g in GRACE_GS),
    "psiR1",
    "psiR1L1",
    "psiU+",
    "psiC+",
    "fifo",
    "lru",
    "ageU",
    "ageC",
    "ageR1",
    "ageR1L1",
    *RANDOMS,
    "kind",
    "oracle",
)
PSI_ARMS = (
    "psiC",
    "psiU",
    "psiR1",
    "psiR1L1",
    "psiU+",
    "psiC+",
    *(f"graceU{g}" for g in GRACE_GS),
)

#: PREREG §6.1: which reference each gated arm is read against ("U", "C", "R1",
#: "R1L1" name the ref *family*; the family's choice FIFO / age-only comes from
#: FIT_VAL).
REF_FAMILY = {
    "psiU": "U",
    **{f"graceU{g}": "U" for g in GRACE_GS},
    "psiU+": "U",
    "psiC": "C",
    "psiC+": "C",
    "psiR1": "R1",
    "psiR1L1": "R1L1",
}
AGE_HEAD_OF = {"U": "ageU", "C": "ageC", "R1": "ageR1", "R1L1": "ageR1L1"}


def content_label(kind: str, status: str | None) -> str:
    """PREREG §6.3: `kind`, or `assert:<status>` for an assert."""
    if kind == "assert":
        if status not in ("pending", "querying", "answered"):
            raise ValueError(f"assert with status {status!r}")
        return f"assert:{status}"
    if kind not in ("query", "filler"):
        raise ValueError(f"kind {kind!r}")
    return kind


# --------------------------------------------------------------------------- #
# §6.1: the paired bootstrap's replicates (B2 §9.2, reproduced so that a ratio
# such as π can be read off the SAME replicates B2.paired_bootstrap uses)
# --------------------------------------------------------------------------- #


def boot_reps(per_doc: dict[str, Tensor], seed: int, n_boot: int = B2.N_BOOT) -> dict:
    """Per-replicate pooled accuracy of every arm, with the pseudo-arm `random`.

    Identical draws to `B2.paired_bootstrap` (same generator, seed, shapes and
    order of calls); a test asserts that contrasts computed from these replicates
    equal B2's."""
    arms = list(per_doc)
    D = per_doc[arms[0]].shape[0]
    X = {a: per_doc[a].to(torch.float64).reshape(D, 2) for a in arms}
    g = torch.Generator().manual_seed(B2.BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (n_boot, D), generator=g)
    W = torch.zeros(n_boot, D, dtype=torch.float64).scatter_add_(
        1, idx, torch.ones(n_boot, D, dtype=torch.float64)
    )
    rep = {a: (W @ x[:, 1]) / (W @ x[:, 0]) for a, x in X.items()}
    pt = {a: x[:, 1].sum() / x[:, 0].sum() for a, x in X.items()}
    rnd = sorted(a for a in arms if B2._RANDOM_ARM.fullmatch(a))
    if rnd:
        rep["random"] = B2._rand_combine([rep[a] for a in rnd])
        pt["random"] = B2._rand_combine([pt[a] for a in rnd])
    return {"rep": rep, "point": {a: float(v) for a, v in pt.items()}}


def arm_ci(reps: dict, arm: str) -> dict:
    r = reps["rep"][arm]
    return {
        "point": reps["point"][arm],
        "lo": float(torch.quantile(r, 0.025)),
        "hi": float(torch.quantile(r, 0.975)),
    }


def ratio_ci(reps: dict, num: tuple[str, str], den: tuple[str, str]) -> dict:
    """π = (acc_a - acc_b) / (acc_c - acc_d), point and percentile CI on the same
    replicates. Undefined (flagged) when the denominator's CI contains 0."""
    R, P = reps["rep"], reps["point"]
    dnum = R[num[0]] - R[num[1]]
    dden = R[den[0]] - R[den[1]]
    lo_d, hi_d = float(torch.quantile(dden, 0.025)), float(torch.quantile(dden, 0.975))
    pden = P[den[0]] - P[den[1]]
    defined = not (lo_d <= 0.0 <= hi_d)
    if not defined or pden == 0:
        return {
            "point": None,
            "lo": None,
            "hi": None,
            "defined": False,
            "den_ci": [lo_d, hi_d],
        }
    r = dnum / dden
    return {
        "point": (P[num[0]] - P[num[1]]) / pden,
        "lo": float(torch.quantile(r, 0.025)),
        "hi": float(torch.quantile(r, 0.975)),
        "defined": True,
        "den_ci": [lo_d, hi_d],
    }


# --------------------------------------------------------------------------- #
# §6.1 / §6.3 / §7: labels
# --------------------------------------------------------------------------- #


def contrast_label(point: float, lo: float, hi: float, delta: float | None) -> str:
    """PREREG §7 (and §6.3's age-only / LRU contrasts): BETTER / SAME / WORSE /
    UNRESOLVED against δ, with A1.13 (a CI excluding its own estimate is
    UNRESOLVED)."""
    if delta is None or not delta > 0 or B2.outcome_flags(point, lo, hi):
        return "UNRESOLVED"
    if lo > 0 and point >= delta:
        return "BETTER"
    if hi < -delta:
        return "WORSE"
    if -delta < lo and hi < delta:
        return "SAME"
    return "UNRESOLVED"


def dq1(labels: dict[int, list[str]]) -> str:
    """PREREG §7 DQ1: `labels[g]` = the per-seed labels of grace_g - ψ̂-C."""
    if any(v.count("BETTER") >= 2 and v.count("WORSE") == 0 for v in labels.values()):
        return "YES"
    if all(v.count("BETTER") == 0 for v in labels.values()):
        return "NO"
    return "MIXED"


def dq2_seed(r1_c: str, r1_u: str, r1_minus_u_lo: float, c_minus_r1_lo: float) -> str:
    """PREREG §7 DQ2, one seed: where (d) sits between (a) and (b)."""
    sc, su = r1_c == "SAME", r1_u == "SAME"
    if sc and su:
        return "INDISTINGUISHABLE"
    if sc:
        return "NEAR-C"
    if su:
        return "NEAR-U"
    if r1_minus_u_lo > 0 and c_minus_r1_lo > 0:
        return "BETWEEN"
    return "UNRESOLVED"


def dq2(r1_outcomes: list[str], seed_labels: list[str]) -> str:
    """PREREG §7 DQ2 class, first match wins."""
    if "LOSS" in r1_outcomes or seed_labels.count("NEAR-U") >= 2:
        return "HARMFUL (near b)"
    if all(x in ("NEAR-C", "INDISTINGUISHABLE") for x in seed_labels):
        return "SAFE (near a)"
    return "INTERMEDIATE"


def dq3(u_outcomes: list[str], per_seed: list[dict]) -> dict:
    """PREREG §7 DQ3. `per_seed[s]` = {label: R1λ1 - C label, acc_C, acc_R1,
    acc_R1L1}. Harm seeds are those on which ψ̂-U is LOSS."""
    H = [s for s, o in enumerate(u_outcomes) if o == "LOSS"]
    out: dict[str, Any] = {"harm_seeds": H, "per_seed": {}}
    if not H:
        out["class"] = "NOT ASSESSABLE"
        return out
    n_tr = n_st = 0
    for s in H:
        p = per_seed[s]
        mono = p["acc_C"] >= p["acc_R1"] >= p["acc_R1L1"]
        if p["label"] == "WORSE" and mono:
            lab = "TRAVELS"
            n_tr += 1
        elif p["label"] == "SAME":
            lab = "STAYS"
            n_st += 1
        else:
            lab = "UNRESOLVED"
        out["per_seed"][s] = {"label": lab, "monotone": mono, **p}
    if n_tr > len(H) / 2:
        out["class"] = "YES"
    elif n_st > len(H) / 2:
        out["class"] = "NO"
    else:
        out["class"] = "MIXED"
    return out


# --------------------------------------------------------------------------- #
# §6.3: partial rho on the effective rule
# --------------------------------------------------------------------------- #


def partial_rho(
    y: Tensor, a: Tensor, c: Tensor, groups: Tensor, n_labels: int = len(CONTENT_LABELS)
) -> dict:
    """PREREG §6.3: rank `y` and `a` (average ranks); demean both ranks and the
    one-hot of `c` within each group (document, step); regress each demeaned rank
    on the demeaned one-hot (OLS, no intercept); rho = Pearson of the residuals.
    `raw` is the same without the content regression (within-step Spearman)."""
    if len(y) < 3:
        return {"partial": None, "raw": None, "n": len(y)}
    ry = B2.within_step_demean(B2._ranks(y.to(torch.float64)), groups)
    ra = B2.within_step_demean(B2._ranks(a.to(torch.float64)), groups)
    Z = torch.nn.functional.one_hot(c.long(), n_labels).to(torch.float64)
    dZ = B2.within_step_demean(Z, groups)
    P = torch.linalg.pinv(dZ.T @ dZ)
    ey = ry - dZ @ (P @ (dZ.T @ ry))
    ea = ra - dZ @ (P @ (dZ.T @ ra))
    return {"partial": B2._pearson(ey, ea), "raw": B2._pearson(ry, ra), "n": len(y)}


class VacuityAcc:
    """Rows (document, step, live slot) of one arm's rollout, compactly."""

    def __init__(self) -> None:
        self.y = array("b")
        self.age = array("h")
        self.c = array("b")
        self.group = array("q")
        self.score = array("d")
        self.n_groups = 0

    def add(self, log: list[dict]) -> None:
        for rec in log:
            n = len(rec["live_ages"])
            g = self.n_groups
            self.n_groups += 1
            self.y.extend(1 if j == rec["victim_pos"] else 0 for j in range(n))
            self.age.extend(rec["live_ages"])
            self.c.extend(rec["live_content"])
            self.group.extend([g] * n)
            if rec.get("psi") is not None:
                self.score.extend(rec["psi"])

    def result(self) -> dict:
        if not self.y:
            return {"decision": {"partial": None, "raw": None, "n": 0}, "score": None}
        t = {
            "y": torch.tensor(self.y.tolist(), dtype=torch.float64),
            "a": torch.tensor(self.age.tolist(), dtype=torch.float64),
            "c": torch.tensor(self.c.tolist(), dtype=torch.long),
            "g": torch.tensor(self.group.tolist(), dtype=torch.long),
        }
        out = {"decision": partial_rho(t["y"], t["a"], t["c"], t["g"]), "score": None}
        if len(self.score) == len(self.y):
            s = torch.tensor(self.score.tolist(), dtype=torch.float64)
            out["score"] = partial_rho(s, t["a"], t["c"], t["g"])
        return out


# --------------------------------------------------------------------------- #
# §6.2 / §6.4 / §6.5: per-arm accumulation over units
# --------------------------------------------------------------------------- #


class ArmAcc:
    def __init__(self, arm: str, S: int, m: int, want_vacuity: bool = True) -> None:
        self.arm = arm
        self.S, self.m = S, m
        self.counts = {b: [] for b in BUCKETS}
        self.resid = {b: [] for b in BUCKETS}
        self.attr = B2.AttributionAcc(S)
        self.n_ev = 0
        self.age1 = 0
        self.age1_pending = 0
        self.pend = 0
        self.pend_le_M = 0
        self.pend_gt_M = 0
        self.flips = 0
        self.flip_known = 0
        self.vac = VacuityAcc() if want_vacuity else None

    def add(self, unit: dict, q_of: dict[int, int]) -> None:
        a = self.arm
        for b in BUCKETS:
            self.counts[b].append(tuple(unit["counts"][a][b]))
            sel = BUCKETS[b]
            hits = unit["hits"][a]
            if hits:
                g = torch.tensor([h[0] for h in hits])
                h = torch.tensor([bool(h[1]) for h in hits])
                s = sel(g)
                self.resid[b].append((int(s.sum()), int((h & s).sum())))
            else:
                self.resid[b].append((0, 0))
        log = unit["logs"][a]
        self.attr.add(log)
        for rec in log:
            self.n_ev += 1
            t, w = rec["step"], rec["written_at"]
            pending = rec["kind"] == "assert" and rec["status"] == "pending"
            if rec["age"] == 1:
                self.age1 += 1
                if pending:
                    self.age1_pending += 1
            if w in q_of and q_of[w] > t:  # B2 results_report.replay's definition
                self.pend += 1
                if q_of[w] - w <= self.m:
                    self.pend_le_M += 1
                else:
                    self.pend_gt_M += 1
            if "flipped" in rec:
                self.flip_known += 1
                self.flips += int(bool(rec["flipped"]))
        if self.vac is not None:
            self.vac.add(log)

    def tensors(self) -> dict:
        return {
            "counts": {
                b: torch.tensor(v, dtype=torch.long).reshape(-1, 2)
                for b, v in self.counts.items()
            },
            "residency": {
                b: torch.tensor(v, dtype=torch.long).reshape(-1, 2)
                for b, v in self.resid.items()
            },
        }

    def result(self) -> dict:
        n = max(self.n_ev, 1)
        att = self.attr.result()
        out = {
            "n_evictions": self.n_ev,
            "age1": self.age1,
            "age1_share": self.age1 / n,
            "age1_pending": self.age1_pending,
            "age1_pending_share": self.age1_pending / n,
            "pending_victims": self.pend,
            "pending_le_M": self.pend_le_M,
            "pending_gt_M": self.pend_gt_M,
            "age_hist": att["age_hist"],
            "rank_hist": att["rank_hist"],
            "kind_status": att["kind_status"],
            "margin": att["margin"],
            "displacement_hist": {int(k): v for k, v in att["rank_shift"].items()},
            "mean_victim_age": (
                sum((k + 1) * c for k, c in enumerate(att["age_hist"])) / n
                if self.n_ev
                else None
            ),
        }
        if self.flip_known:
            out["grace_flips"] = self.flips
            out["grace_flip_share"] = self.flips / self.flip_known
        if self.vac is not None:
            out["vacuity"] = self.vac.result()
        return out


def aggregate_seed(units, docs, arms, *, S: int, m: int) -> dict:
    """Fold one seed's units (in document order) into counts, residency and the
    §6.2-6.5 readouts. `docs` supplies each document's (assert, query) pairs."""
    acc = {a: ArmAcc(a, S, m) for a in arms}
    ids = []
    for u, d in zip(units, docs, strict=True):
        if u["doc_id"] != d.doc_id:
            raise ValueError(f"unit {u['doc_id']} != document {d.doc_id}")
        ids.append(d.doc_id)
        q_of = {a: q for a, q in d.pairs}
        for a in arms:
            acc[a].add(u, q_of)
    ten = {a: acc[a].tensors() for a in arms}
    return {
        "doc_ids": ids,
        "counts": {a: ten[a]["counts"] for a in arms},
        "residency": {a: ten[a]["residency"] for a in arms},
        "readouts": {a: acc[a].result() for a in arms},
    }


# --------------------------------------------------------------------------- #
# §6.1 / §6.3 / §7 per seed
# --------------------------------------------------------------------------- #


def ref_arm(family: str, refs: dict[str, str]) -> str:
    """`refs[family]` is 'fifo' or 'age' (B2 §9.3's `select_ref` output)."""
    return "fifo" if refs[family] == "fifo" else AGE_HEAD_OF[family]


def seed_contrasts(refs: dict[str, str]) -> dict[str, tuple[str, str]]:
    con: dict[str, tuple[str, str]] = {}
    for arm, fam in REF_FAMILY.items():
        con[f"{arm}-ref"] = (arm, ref_arm(fam, refs))
        con[f"{arm}-random"] = (arm, "random")
        con[f"{arm}-age"] = (arm, AGE_HEAD_OF[fam])
        con[f"{arm}-lru"] = (arm, "lru")
    for g in GRACE_GS:
        con[f"graceU{g}-psiC"] = (f"graceU{g}", "psiC")
    con["psiR1-psiC"] = ("psiR1", "psiC")
    con["psiR1-psiU"] = ("psiR1", "psiU")
    con["psiC-psiR1"] = ("psiC", "psiR1")
    con["psiR1-psiR1L1"] = ("psiR1", "psiR1L1")
    con["psiR1L1-psiC"] = ("psiR1L1", "psiC")
    con["psiU-psiR1L1"] = ("psiU", "psiR1L1")
    con["psiU-psiU+"] = ("psiU", "psiU+")
    con["psiC-psiU"] = ("psiC", "psiU")
    con["oracle-fifo"] = ("oracle", "fifo")
    con["kind-fifo"] = ("kind", "fifo")
    con["lru-fifo"] = ("lru", "fifo")
    return con


def seed_outcomes(counts: dict, refs: dict[str, str], delta: float, seed: int) -> dict:
    """§6.1's outcome per gated arm, every contrast's CI and label, per-arm
    accuracy CIs, and π (§7 DQ2)."""
    con = seed_contrasts(refs)
    boot = {}
    acc = {}
    for b in BUCKETS:
        per = {a: c[b] for a, c in counts.items()}
        boot[b] = B2.paired_bootstrap(per, con, seed)
        reps = boot_reps(per, seed)
        acc[b] = {a: arm_ci(reps, a) for a in (*counts, "random")}
        if b == "all":
            pi = ratio_ci(reps, ("psiR1", "psiU"), ("psiC", "psiU"))
    out: dict[str, Any] = {"delta": delta, "refs": refs, "acc": acc, "pi": pi}
    lab = {}
    for name, v in boot["all"].items():
        lab[name] = dict(v, label=contrast_label(v["point"], v["lo"], v["hi"], delta))
    out["contrasts"] = lab
    out["contrasts_by_bucket"] = {b: boot[b] for b in BUCKETS if b != "all"}
    oc = {}
    for arm, fam in REF_FAMILY.items():
        a = boot["all"][f"{arm}-ref"]
        n = boot["gap_2_to_M"][f"{arm}-ref"]
        r = boot["all"][f"{arm}-random"]
        oc[arm] = {
            "ref": ref_arm(fam, refs),
            "point": a["point"],
            "lo": a["lo"],
            "hi": a["hi"],
            "noninf_lo": n["lo"],
            "rand_point": r["point"],
            "rand_lo": r["lo"],
            "label": B2.outcome(
                a["point"], a["lo"], a["hi"], delta, noninf_lo=n["lo"], rand_lo=r["lo"]
            ),
            "flags": B2.outcome_flags(a["point"], a["lo"], a["hi"]),
        }
    out["outcome"] = oc
    return out


def decisions(per_seed: dict[int, dict]) -> dict:
    """PREREG §7's three questions, plus §6.1's replication classes (non-binding)."""
    S = sorted(per_seed)
    L = {s: per_seed[s]["contrasts"] for s in S}
    OUT = {s: per_seed[s]["outcome"] for s in S}
    A = {s: per_seed[s]["acc"]["all"] for s in S}
    g_labels = {g: [L[s][f"graceU{g}-psiC"]["label"] for s in S] for g in GRACE_GS}
    d2_seed = [
        dq2_seed(
            L[s]["psiR1-psiC"]["label"],
            L[s]["psiR1-psiU"]["label"],
            L[s]["psiR1-psiU"]["lo"],
            L[s]["psiC-psiR1"]["lo"],
        )
        for s in S
    ]
    r1_out = [OUT[s]["psiR1"]["label"] for s in S]
    u_out = [OUT[s]["psiU"]["label"] for s in S]
    d3 = dq3(
        u_out,
        [
            {
                "label": L[s]["psiR1L1-psiC"]["label"],
                "acc_C": A[s]["psiC"]["point"],
                "acc_R1": A[s]["psiR1"]["point"],
                "acc_R1L1": A[s]["psiR1L1"]["point"],
            }
            for s in S
        ],
    )
    rep = B2.classify(u_out, [OUT[s]["psiC"]["label"] for s in S])
    comp = B2.classify(
        [OUT[s]["psiU+"]["label"] for s in S], [OUT[s]["psiC+"]["label"] for s in S]
    )
    return {
        "DQ1": {"class": dq1(g_labels), "labels": g_labels},
        "DQ2": {
            "class": dq2(r1_out, d2_seed),
            "per_seed": d2_seed,
            "psiR1_outcome": r1_out,
            "pi": [per_seed[s]["pi"] for s in S],
        },
        "DQ3": d3,
        "replication": {
            "row": rep[0],
            "class": rep[1],
            "psiU": u_out,
            "psiC": [OUT[s]["psiC"]["label"] for s in S],
            "label": "replication, non-binding",
        },
        "replication_companion": {
            "row": comp[0],
            "class": comp[1],
            "label": "replication companion (U+, C+), non-binding",
        },
    }


def finite(x) -> bool:
    return x is not None and isinstance(x, float) and math.isfinite(x)
