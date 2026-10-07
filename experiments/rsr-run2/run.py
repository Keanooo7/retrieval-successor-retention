"""RSR run 2: run 1's harness with the C3 LOO target.

PREREG `experiments/rsr-run2/PREREG.md`, committed alone at a406d39; owner rulings
R-2026-10-07-c3-loo-target and R-2026-10-07-falsifier-6-deferred-to-e1.

**Run 1's harness is reused by import, never copied** (`experiments/rsr-run1/run.py`,
loaded as a private module instance `R1`). Exactly two of its names are rebound:

* ``R1.RUN_ID = "rsr-run2"`` -- the job ids the controls read
  (`rsr-run2-<arm>-s<seed>`);
* ``R1.cli_argv`` -- run 1's command plus PREREG §0's one change,
  ``--policy-override target=loo_resample`` on both RSR arms (fifo untouched).

Everything run 1's `measure_seed` does -- C0-C10 (C9/C9b scoped), every held-out pass,
§7.1's contrasts, the bootstrap -- runs unchanged (PREREG §0: run 1's PREREG binds
wherever run 2's is silent). On top of it, this file adds PREREG §2-§7:

* §2 the per-seed label (WIN / LOSS / EQUIV / UNRESOLVED against delta_seed, read from
  B2's ledger and checked against §2.1's table) and the class (BEATS-FIFO / FAILS /
  MIXED);
* §4 hit rate and the FactFiller ceiling share; psiC(run 2) - psiC(run 1); psiC -
  pinned Expire-Span L48-k1e-2 (scoped controls C14, C15);
* §5 the age-1 eviction share and the self-alignment probe (run 2, and run 1
  retroactively);
* §6 the target diagnostics, aggregated from the heartbeats;
* §7 C2', C9, C11, C12, C13 (run-level, exit 3) and C14, C15 (scoped to the ES
  contrast).

Subcommands: ``preflight``, ``measure-seed``, ``finalize``, ``recompute`` (claims.json).
Exit codes (`rsr.exit_codes`): 0 every run-level control passed in every seed; 3 did
not run, or a run-level control failed (numbers written, the class marked not read);
5 an arm INERT.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402

EXPERIMENT = "experiments/rsr-run2/run.py"
LAUNCHER = "experiments/rsr-run2/launch_run2.zsh"
RUN_ID = "rsr-run2"
PREREG = "experiments/rsr-run2/PREREG.md"
PREREG_COMMIT = "a406d39"
TARGET = "loo_resample"
TARGET_OVERRIDE = ("--policy-override", f"target={TARGET}")


def _load(name: str, rel: str):
    """Import an experiment script by path, as its own module instance. Never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: Run 1's harness, a PRIVATE instance (never the module run 1's own tests load).
R1 = _load("_rsr_run2_r1", "experiments/rsr-run1/run.py")
_R1_CLI_ARGV = R1.cli_argv


def cli_argv(cfg, arm: str, owner, out_dir: Path) -> list[str]:
    """Run 1's command (`R1.cli_argv`) plus PREREG §0's one change: both RSR arms get
    ``--policy-override target=loo_resample`` right after the gamma override -- the
    launcher's token order (``tests/test_rsr_run2_harness.py`` compares them)."""
    argv = list(_R1_CLI_ARGV(cfg, arm, owner, out_dir))
    if arm in R1.RSR_ARMS:
        i = argv.index(f"gamma={owner.gamma!r}") + 1
        argv[i:i] = TARGET_OVERRIDE
    return argv


R1.RUN_ID = RUN_ID
R1.cli_argv = cli_argv

ARMS, RSR_ARMS, SEEDS, M = R1.ARMS, R1.RSR_ARMS, R1.SEEDS, R1.M_SLOTS
HEADLINE = R1.HEADLINE  # "acc.all.psiC_minus_fifo"
LABELS = ("WIN", "LOSS", "EQUIV", "UNRESOLVED")
CLASSES = ("BEATS-FIFO", "FAILS", "MIXED")
CLASS_ROUTE = {
    "BEATS-FIFO": "ROADMAP step 5 (E1)",
    "FAILS": "ROADMAP step 6 (write-up)",
    "MIXED": "owner",
}
NEWCOMER_RESOLVED, NEWCOMER_PERSISTS = 0.05, 0.20  # PREREG §5


def a_max_from_registry() -> int:
    """PREREG Amendment 1 (d8dfd14): ``A_max`` read from the registry, as in run 1 §6
    (CONDITIONAL, synthetic scope, ``A_max = S``; correction 9)."""
    from rsr import constants as C

    return int(C.REGISTRY.get("A_max", "synthetic", S=C.REGISTRY.get("S", "synthetic")))


def age_buckets(m: int, a_max: int) -> dict[str, tuple[int, int]]:
    """PREREG §5 probe item 2 as amended (Amendment 1, d8dfd14):
    {1, 2..4, 5..M, M+1..A_max}."""
    return {"1": (1, 1), "2..4": (2, 4), "5..M": (5, m), "M+1..A_max": (m + 1, a_max)}


ES_CONFIG = "L48-k1e-2"
#: PREREG §4 item 4: each referendum vector's sha256 (C14 checks them; preflight checks
#: that every one appears verbatim in the committed PREREG).
ES_SHA = {
    0: "f6a3d97f25c520c84091562002e719c489d5824ba53b484f2ad892ff12d194e1",
    1: "d647767927b7c3655a8d473fdef990c534315ed461271c1a109d466dcd3f8db2",
    2: "6ded09f44abdefd90d68c9dfcd2c8d904f7f1e8dc07c6f934078b4425ab5a74a",
}
#: PREREG §7 C15: run 1's fifo held-out vectors the referendum paired against.
RUN1_FIFO_SHA = {
    0: "b05387509be57d12587b4577ca49fd3443f07ebaec735db52715d2335d8caa1e",
    1: "819bd8af12a70c194fc17ddc58d9d3967a0e2c8d6bff4bee5128d9e578420413",
    2: "3a7ad2cecb55e1096bf981aa42d72dd2eac1758a6a0d75a039c3537fa60a2cb5",
}
DELTA_KEY = "ckpt3000.seed{s}.decisions.delta"  # PREREG §2.1, value["0.9"]
CF_CAP_KEY = "B.ckpt3000.cf.g09.cap.point"  # PREREG §4 item 1 (samples by seed)
C9_FLOOR = 0.80  # PREREG §7 C9 = E0d C6
#: Fields a heartbeat or diagnostics line may differ in between two identical runs (C12):
#: wall-clock and memory readings. ``provenance.git_sha`` is dropped from the header.
VOLATILE_KEYS = frozenset(
    {"elapsed_s", "sent_per_s", "mem_gb", "step_s", "eval_forward_s", "loo_s", "seconds"}
)


@dataclass(frozen=True)
class Refs:
    """Where run 2 reads what it did not produce. Tests point these at fixtures."""

    run1_out: str = f"{R1._R}/.worktrees/rsr-run1-run/runs/rsr-run1"
    es_out: str = f"{R1._R}/.worktrees/es-referendum-a2-run/runs/expire-span-referendum"
    es_pinned: str = "/Users/keanooo7/rsr-substrate/2026-10-07-es-pinned"
    b2_ledger: str = f"{R1._R}/.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/ledger.json"
    cf_ledger: str = f"{R1._R}/.worktrees/cf-ceiling/runs/cf-ceiling/ledger.json"
    prereg: str = str(ROOT / PREREG)
    es_sha: tuple = tuple(sorted(ES_SHA.items()))
    run1_fifo_sha: tuple = tuple(sorted(RUN1_FIFO_SHA.items()))


REFS = Refs()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- #
# §2: delta, the per-seed label, the class
# --------------------------------------------------------------------------- #


def prereg_delta_table(prereg_path: str) -> dict[int, float]:
    """PREREG §2.1's table, read from the committed file: ``| s | delta | ... |`` rows."""
    text = Path(prereg_path).read_text()
    sec = text[text.index("### 2.1") : text.index("## 3.")]
    rows = re.findall(r"^\| ([0-2]) \| ([0-9.]+) \|", sec, flags=re.M)
    return {int(s): float(v) for s, v in rows}


def ledger_rows(path: str) -> dict[str, Any]:
    return {r["key"]: r for r in json.loads(Path(path).read_text())["rows"]}


def read_delta(seed: int, refs: Refs = REFS) -> dict:
    """delta_seed read from B2's ledger and compared with PREREG §2.1 (control
    Cdelta)."""
    out: dict[str, Any] = {"ledger": refs.b2_ledger, "key": DELTA_KEY.format(s=seed)}
    try:
        val = ledger_rows(refs.b2_ledger)[out["key"]]["value"]
        out["delta"] = float(val["0.9"])
        out["registered"] = prereg_delta_table(refs.prereg).get(seed)
        out["ok"] = out["registered"] is not None and out["delta"] == out["registered"]
    except Exception as e:  # did not run: recorded, never a pass
        out.update(delta=None, ok=False, error=f"{type(e).__name__}: {e}")
    return out


def seed_label(point, lo, hi, delta) -> str:
    """PREREG §2, verbatim: WIN if CI lower > 0 and point >= +delta; LOSS if CI upper
    < 0 and point <= -delta; EQUIV if -delta < lo and hi < delta; UNRESOLVED otherwise
    (a missing number or delta is UNRESOLVED). The four are mutually exclusive, so the
    order is immaterial."""
    if not all(_finite(x) for x in (point, lo, hi, delta)):
        return "UNRESOLVED"
    if lo > 0 and point >= delta:
        return "WIN"
    if hi < 0 and point <= -delta:
        return "LOSS"
    if -delta < lo and hi < delta:
        return "EQUIV"
    return "UNRESOLVED"


def run_class(labels: dict[int, str], seeds=SEEDS) -> str:
    """PREREG §2: BEATS-FIFO iff WIN on all 3 seeds; FAILS iff LOSS on any seed; MIXED
    otherwise. A seed with no label counts as not-WIN (it can only make MIXED)."""
    got = [labels.get(s) for s in seeds]
    if any(x == "LOSS" for x in got):
        return "FAILS"
    if all(x == "WIN" for x in got):
        return "BEATS-FIFO"
    return "MIXED"


def label_contrast(c: dict | None, delta) -> dict:
    if c is None:
        return {"label": "UNRESOLVED", "why": "no contrast"}
    return {
        **c,
        "delta": delta,
        "label": seed_label(c.get("point"), c.get("lo"), c.get("hi"), delta),
    }


# --------------------------------------------------------------------------- #
# §5: the age-1 eviction share and the newcomer label
# --------------------------------------------------------------------------- #


def age1_share(records: list[dict]) -> dict:
    """Of the held-out evictions in ``records`` (run 1's per-document format: each
    ``evictions`` entry is ``(t, age, displacement)``), how many evicted an age-1 slot."""
    ages = Counter(int(e[1]) for r in records for e in r["evictions"])
    n = sum(ages.values())
    return {
        "n_evictions": n,
        "n_age1": ages.get(1, 0),
        "share": ages.get(1, 0) / n if n else None,
    }


def newcomer_label(shares: dict[int, float | None], seeds=SEEDS) -> str:
    """PREREG §5: RESOLVED if <= 0.05 on all seeds; PERSISTS if >= 0.20 on any seed; else
    REDUCED. A missing seed cannot be RESOLVED; PERSISTS is read first (any seed)."""
    vals = [shares.get(s) for s in seeds]
    if any(v is not None and v >= NEWCOMER_PERSISTS for v in vals):
        return "NEWCOMER-PERSISTS"
    if all(v is not None and v <= NEWCOMER_RESOLVED for v in vals):
        return "NEWCOMER-RESOLVED"
    return "NEWCOMER-REDUCED"


# --------------------------------------------------------------------------- #
# §5: the self-alignment probe
# --------------------------------------------------------------------------- #


class ProbePolicy:
    """Wraps a trained RSR policy for a held-out pass and records, at every eviction
    decision, raw psi_hat(s_i, c_t) (`RSRPolicy._psi`, before z-scoring -- monotone, so
    the ranking is the decision's), cos(s_i, c_t), the ages and the victim, over the live
    slots. The decision is the inner policy's, unchanged; nothing is trained."""

    learns = False

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = f"probe:{getattr(inner, 'name', 'rsr')}"
        self.decisions: list[tuple] = []

    @property
    def needs_eval_context(self) -> bool:
        return bool(getattr(self.inner, "needs_eval_context", False))

    def to(self, device):
        self.inner.to(device)
        return self

    def reset(self) -> None:
        self.inner.reset()

    def select_eviction(self, slots, context, step: int) -> int:
        with torch.no_grad():
            psi = self.inner._psi(slots, context)
            g = torch.nn.functional.normalize(slots.gestalts.detach(), dim=-1)
            c = torch.nn.functional.normalize(context.detach(), dim=-1)
            cos = g @ c
        k = int(self.inner.select_eviction(slots, context, step))
        live = slots.live
        self.decisions.append(
            (
                int(slots.row),
                int(step),
                psi[live].detach().to("cpu", torch.float64),
                cos[live].to("cpu", torch.float64),
                slots.ages()[live].to("cpu"),
                int(live[:k].sum()),  # the victim's position among the live slots
            )
        )
        return k

    def observe(self, slots, attn, step: int) -> None:
        raise RuntimeError("observe is off in the held-out probe (PREREG §5)")

    def on_write(self, slots, slot: int, step: int) -> None:
        self.inner.on_write(slots, slot, step)


def probe_pass(cfg, payload: dict, docs: list, vmap) -> dict:
    """The psiC policy of ``payload`` over ``docs``, run 1's held-out batching and
    decision path (`R1.run_batch`, `eval_forward` context), recording every decision.
    -> ``{"decisions": [(doc_index, step, psi, cos, age, victim_pos)],
    "victims": [...]}``."""
    model = R1.model_from_payload(payload).to(cfg.device)
    decisions: list[tuple] = []
    victims: list[dict] = []
    for i in range(0, len(docs), cfg.eval_batch):
        chunk = docs[i : i + cfg.eval_batch]
        pol = ProbePolicy(R1.eval_policy("psiC", payload, cfg))
        recs = R1.run_batch(model, pol, chunk, vmap, cfg)
        victims += [r["victims"] for r in recs]
        decisions += [(i + d[0], *d[1:]) for d in pol.decisions]
    return {"decisions": decisions, "victims": victims}


def avg_ranks_rows(x: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` -> 1-based average ranks within each row (scipy's
    ``method="average"``)."""
    lt = (x.unsqueeze(2) > x.unsqueeze(1)).sum(-1).to(torch.float64)
    eq = (x.unsqueeze(2) == x.unsqueeze(1)).sum(-1).to(torch.float64)
    return lt + (eq + 1.0) / 2.0


def pearson_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Pearson; NaN where either row has zero variance."""
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    den = (a.pow(2).sum(1) * b.pow(2).sum(1)).sqrt()
    return torch.where(den > 0, (a * b).sum(1) / den, torch.full_like(den, float("nan")))


def lower_median(v: torch.Tensor) -> float | None:
    s = v.sort().values
    return float(s[(s.numel() - 1) // 2]) if s.numel() else None


def weighted_lower_medians(
    v: torch.Tensor, doc: torch.Tensor, W: torch.Tensor, chunk: int = 50
) -> torch.Tensor:
    """Per bootstrap replicate (a row of ``W``, document multiplicities), the lower median
    of the multiset in which value ``v[j]`` appears ``W[r, doc[j]]`` times."""
    order = torch.argsort(v, stable=True)
    vs, ds = v[order], doc[order]
    out = []
    for r0 in range(0, W.shape[0], chunk):
        w = W[r0 : r0 + chunk][:, ds]  # [c, N]
        cum = w.cumsum(1)
        tot = cum[:, -1]
        want = torch.floor((tot - 1) / 2) + 1  # 1-based position of the lower median
        idx = torch.searchsorted(cum.contiguous(), want.unsqueeze(1)).squeeze(1)
        idx = idx.clamp(max=vs.numel() - 1)
        out.append(torch.where(tot > 0, vs[idx], torch.full_like(tot, float("nan"))))
    return torch.cat(out)


def _q(rep: torch.Tensor) -> dict:
    return R1._ci(rep)


def probe_stats(
    decisions: list[tuple],
    n_docs: int,
    W: torch.Tensor,
    m: int = M,
    a_max: int | None = None,
) -> dict:
    """PREREG §5's three probe statistics from ``probe_pass`` decisions.

    1. within-step Spearman rho(psi_hat, cos) over the live slots of each decision;
       the LOWER median over decisions, with the per-document bootstrap CI;
    2. by age bucket {1, 2..4, 5..M, M+1..A_max} (Amendment 1; ``A_max`` from the
       registry unless given): within-step percentile ranks ``(rank - 1)/(K - 1)`` of
       psi_hat and of cos; per bucket, the Pearson correlation of the two over the
       pooled (decision, slot) cells, with the same bootstrap, beside each bucket's mean
       percentiles. ``cells_older_than_M`` is still reported and is CHECKED equal to the
       M+1..A_max bucket's count (``older_than_M_equals_bucket``);
    3. the share of victims whose cos(s_i, c_t) is the step's maximum."""
    if not decisions:
        return {"n_decisions": 0}
    ks = {d[2].numel() for d in decisions}
    if len(ks) != 1:
        raise R1.ControlFailure(
            f"probe decisions over {sorted(ks)} live slots, not one K"
        )
    K = ks.pop()
    D = torch.tensor([d[0] for d in decisions], dtype=torch.long)
    P = torch.stack([d[2] for d in decisions])
    C = torch.stack([d[3] for d in decisions])
    A = torch.stack([d[4] for d in decisions]).to(torch.long)
    V = torch.tensor([d[5] for d in decisions], dtype=torch.long)
    rp, rc = avg_ranks_rows(P), avg_ranks_rows(C)
    rho = pearson_rows(rp, rc)
    ok = torch.isfinite(rho)
    out: dict[str, Any] = {
        "n_decisions": len(decisions),
        "K": K,
        "rho_undefined": int((~ok).sum()),
    }
    rep = weighted_lower_medians(rho[ok], D[ok], W)
    out["rho_lower_median"] = {**_q(rep), "point": lower_median(rho[ok])}
    xp, yc = (rp - 1) / (K - 1), (rc - 1) / (K - 1)
    buckets: dict[str, Any] = {}
    a_max = a_max_from_registry() if a_max is None else int(a_max)
    for name, (lo, hi) in age_buckets(m, a_max).items():
        sel = (lo <= A) & (hi >= A)
        doc_of = D.unsqueeze(1).expand_as(A)[sel]
        x, y = xp[sel], yc[sel]
        S = torch.zeros(n_docs, 6, dtype=torch.float64)
        for j, col in enumerate((torch.ones_like(x), x, y, x * x, y * y, x * y)):
            S[:, j].index_add_(0, doc_of, col)

        def stats(T):
            n, sx, sy, sxx, syy, sxy = (T[..., j] for j in range(6))
            cov = n * sxy - sx * sy
            den = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt()
            r = torch.where(den > 0, cov / den, torch.full_like(den, float("nan")))
            return r, sx / n, sy / n

        R = W @ S
        rr, mx, my = stats(R)
        pr, pmx, pmy = stats(S.sum(0))
        buckets[name] = {
            "n_cells": int(sel.sum()),
            "pearson_pct": {**_q(rr), "point": float(pr)},
            "mean_psi_pct": {**_q(mx), "point": float(pmx)},
            "mean_cos_pct": {**_q(my), "point": float(pmy)},
        }
    out["by_age_bucket"] = buckets
    out["cells_older_than_M"] = int((m < A).sum())
    out["a_max"] = a_max
    out["older_than_M_equals_bucket"] = (
        out["cells_older_than_M"] == buckets["M+1..A_max"]["n_cells"]
    )
    is_max = C.gather(1, V.unsqueeze(1)).squeeze(1) == C.max(1).values
    per = torch.zeros(n_docs, 2, dtype=torch.float64)
    per[:, 0].index_add_(0, D, torch.ones_like(is_max, dtype=torch.float64))
    per[:, 1].index_add_(0, D, is_max.to(torch.float64))
    out["victim_cos_is_max_share"] = R1.level(per, W)
    return out


# --------------------------------------------------------------------------- #
# §4: hit rate, FactFiller, the ceiling share
# --------------------------------------------------------------------------- #


def _lr():
    return _load("_rsr_run2_lookahead", "experiments/lookahead-room/run.py")


def factfiller_hits(docs: list, seed: int, m: int = M) -> list[list]:
    """B0-CF's FactFiller reference, model-free, seeded
    `random.Random(f"ff:{seed}:{doc_id}")`."""
    out = []
    for d in docs:
        pol = _lr().FactFiller(d, random.Random(f"ff:{seed}:{d.doc_id}"))
        out.append([(q["gap"], q["hit"]) for q in simulate(d, pol, m)["queries"]])
    return out


def hit_counts(hits: list[list], sel) -> torch.Tensor:
    """``[D, 2]`` = (queries in the bucket, resident) per document."""
    rows = []
    for h in hits:
        if not h:
            rows.append([0.0, 0.0])
            continue
        g = torch.tensor([int(x[0]) for x in h], dtype=torch.long)
        ok = torch.tensor([bool(x[1]) for x in h], dtype=torch.bool)
        m = sel(g)
        rows.append([float(m.sum()), float(ok[m].sum())])
    return torch.tensor(rows, dtype=torch.float64).reshape(-1, 2)


def ceiling(per: dict[str, torch.Tensor], arm: str, W: torch.Tensor) -> dict:
    """PREREG §4 item 1: ``(hit_arm - hit_FIFO)/(hit_factfiller - hit_FIFO)``, point and
    paired CI; a replicate with a denominator <= 0 is dropped and counted (B0's rule)."""
    pa, pf, pff = (R1._point(per[k]) for k in (arm, "fifo", "factfiller"))
    point = None
    if None not in (pa, pf, pff) and pff - pf > 0:
        point = (pa - pf) / (pff - pf)
    ra, rf, rff = (R1.pooled_replicates(per[k], W) for k in (arm, "fifo", "factfiller"))
    den = rff - rf
    rep = torch.where(den > 0, (ra - rf) / den, torch.full_like(den, float("nan")))
    return {**_q(rep), "point": point}


# --------------------------------------------------------------------------- #
# §6: target diagnostics, from the heartbeats
# --------------------------------------------------------------------------- #


def _evict_key(kv):
    k = kv[0]
    return (1, 0) if k == "kept" else (0, int(k))


def target_diagnostics(beats: list[dict], t_warm: float) -> dict:
    """PREREG §6: sums over steps, split at T_warm (warm: k < T_warm)."""
    out: dict[str, Any] = {}
    for phase, keep in (
        ("warm", lambda k: k < t_warm),
        ("post_warm", lambda k: k >= t_warm),
    ):
        g_age: dict[int, list[float]] = {}
        g_ev: dict[str, list[float]] = {}
        rho_meds, rho_n = [], 0
        counts: Counter = Counter()
        gap_w = gap_sum = zero_sum = 0.0
        loo_s, ev_s, st_s = [], [], []
        n_steps = 0
        for b in beats:
            if not keep(int(b["step"])):
                continue
            mc = b.get("mc") or {}
            td, lo = mc.get("target_diag") or {}, mc.get("loo") or {}
            n_steps += 1
            for k, (s, n) in (td.get("g_by_age") or {}).items():
                acc = g_age.setdefault(int(k), [0.0, 0])
                acc[0] += s
                acc[1] += n
            for k, (s, n) in (td.get("g_by_evict_age") or {}).items():
                acc = g_ev.setdefault(str(k), [0.0, 0])
                acc[0] += s
                acc[1] += n
            r = td.get("rho_g_age") or {}
            if r.get("median") is not None:
                rho_meds.append(r["median"])
            rho_n += int(r.get("n") or 0)
            counts.update(lo.get("counts") or {})
            n_ok = sum(v for k, v in (lo.get("counts") or {}).items() if ".ok." in k)
            if lo.get("donor_age_gap_mean") is not None and n_ok:
                gap_w += n_ok
                gap_sum += lo["donor_age_gap_mean"] * n_ok
                zero_sum += lo["donor_age_gap_zero_share"] * n_ok
            loo_s.append(b.get("loo_s"))
            ev_s.append(b.get("eval_forward_s"))
            st_s.append(b.get("step_s"))
        full = {k: v for k, v in counts.items() if k.startswith("full.")}
        n_full = sum(full.values())
        ok_full = sum(v for k, v in full.items() if k.startswith("full.ok."))
        by_kind: dict[str, dict] = {}
        for k, v in counts.items():
            fill, st, kind = k.split(".")
            by_kind.setdefault(f"{fill}.{kind}", Counter())[st] += v
        out[phase] = {
            "n_steps": n_steps,
            "g_by_age": {
                a: {"mean": s / n if n else None, "n": n}
                for a, (s, n) in sorted(g_age.items())
            },
            "g_by_evict_age": {
                a: {"mean": s / n if n else None, "n": n}
                for a, (s, n) in sorted(g_ev.items(), key=_evict_key)
            },
            "rho_g_age_step_medians": R1._summary(rho_meds),
            "rho_g_age_n_groups": rho_n,
            "loo_counts": dict(sorted(counts.items())),
            "coverage_by_fill_kind": {
                k: {
                    "ok": c.get("ok", 0),
                    "total": sum(c.values()),
                    "share": c.get("ok", 0) / sum(c.values())
                    if sum(c.values())
                    else None,
                }
                for k, c in sorted(by_kind.items())
            },
            "full_cells": n_full,
            "full_covered": ok_full,
            "full_coverage": ok_full / n_full if n_full else None,
            "donor_age_gap_mean": gap_sum / gap_w if gap_w else None,
            "donor_age_gap_zero_share": zero_sum / gap_w if gap_w else None,
            "loo_s": R1._summary(loo_s),
            "eval_forward_s": R1._summary(ev_s),
            "step_s": R1._summary(st_s),
        }
    return out


# --------------------------------------------------------------------------- #
# §7: run-2 controls
# --------------------------------------------------------------------------- #


def c2_prime(c2: dict | None) -> dict:
    """C2': run 1's C2 (pairing), with its k < T_warm bitwise ``loss_lm`` reading made
    explicit: no warm-step mismatch, and every warm step compared."""
    if c2 is None:
        return {"ok": False, "error": "C2 was not evaluated"}
    warm = c2.get("warm_steps_compared")
    mism = c2.get("c_loss_lm_warm_mismatch_steps")
    return {
        "C2_ok": c2.get("ok"),
        "warm_steps_compared": warm,
        "warm_loss_lm_mismatch_steps": mism,
        "ok": bool(c2.get("ok")) and mism == [] and bool(warm),
    }


def c9_coverage(diag: dict[str, dict]) -> dict:
    """C9 (E0d C6): whole run, full-memory cells covered >= 0.80, per RSR arm."""
    per = {}
    for a, d in diag.items():
        n = sum(d[p]["full_cells"] for p in ("warm", "post_warm"))
        k = sum(d[p]["full_covered"] for p in ("warm", "post_warm"))
        cov = k / n if n else None
        per[a] = {
            "full_cells": n,
            "full_covered": k,
            "coverage": cov,
            "ok": cov is not None and cov >= C9_FLOOR,
        }
    return {
        "floor": C9_FLOOR,
        "per_arm": per,
        "ok": bool(per) and all(v["ok"] for v in per.values()),
    }


def c11_target(results: dict, hbs: dict, launch: dict | None) -> dict:
    """C11: both RSR arms' labels carry ``target=loo_resample``; every beat has
    ``mc.target == "loo_resample"`` and a finite ``loo_s``; fifo's label carries none;
    the launch record says the target and is not a smoke."""
    per, probs = {}, []
    for a in ARMS:
        label = ((results.get(a) or {}).get("config") or {}).get("arm_label") or ""
        beats = (hbs.get(a) or {}).get("beats") or []
        if a in RSR_ARMS:
            bad = [
                b["step"]
                for b in beats
                if (b.get("mc") or {}).get("target") != TARGET
                or not _finite(b.get("loo_s"))
            ]
            ok = f"target={TARGET}" in label and bool(beats) and not bad
            per[a] = {
                "label": label,
                "n_beats": len(beats),
                "bad_steps": bad[:20],
                "ok": ok,
            }
        else:
            per[a] = {"label": label, "ok": "target=" not in label}
    if (launch or {}).get("target") != TARGET:
        probs.append(f"launch target {(launch or {}).get('target')!r}")
    if (launch or {}).get("smoke", "") != "":
        probs.append(
            f"launch smoke {(launch or {}).get('smoke')!r}: a smoke is never the run"
        )
    return {
        "per_arm": per,
        "launch_problems": probs,
        "ok": not probs and all(v["ok"] for v in per.values()),
    }


def c13_finite(hbs: dict) -> dict:
    """C13: per RSR beat, ``loss_mc``, ``phi_grad_norm``, ``mc.g_mean`` and every
    ``target_diag`` sum finite. (Δ itself is finite wherever its status is OK by
    construction: `step_delta` sets OK only where the knockout delta is not NaN.)"""
    per = {}
    for a in RSR_ARMS:
        bad = []
        for b in (hbs.get(a) or {}).get("beats") or []:
            mc = b.get("mc") or {}
            td = mc.get("target_diag") or {}
            sums = [s for s, _n in (td.get("g_by_age") or {}).values()]
            sums += [s for s, _n in (td.get("g_by_evict_age") or {}).values()]
            vals = [b.get("loss_mc"), b.get("phi_grad_norm"), mc.get("g_mean"), *sums]
            if not all(_finite(v) for v in vals):
                bad.append(b["step"])
        per[a] = {"bad_steps": bad[:20], "n_bad": len(bad), "ok": not bad}
    return {"per_arm": per, "ok": all(v["ok"] for v in per.values())}


def _strip(x: Any) -> Any:
    if isinstance(x, dict):
        return {
            k: _strip(v)
            for k, v in x.items()
            if k not in VOLATILE_KEYS and not k.endswith("_utc")
        }
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def comparable_heartbeat(p: Path) -> list[dict]:
    rows = []
    for r in _jsonl(p):
        r = _strip(r)
        if r.get("kind") == "header":
            prov = dict(r.get("provenance") or {})
            prov.pop("git_sha", None)
            r["provenance"] = prov
        rows.append(r)
    return rows


def c12_fifo(
    seed: int, out_root: Path, fifo_result: dict | None, run2_fifo_recs, refs: Refs = REFS
) -> dict:
    """C12: run 2's fifo arm IS run 1's fifo arm -- run_id and config_hash, heartbeats and
    diagnostics apart from wall-clock / memory fields and the header's git_sha, and the
    held-out per-document records."""
    r1 = Path(refs.run1_out)
    out: dict[str, Any] = {}
    try:
        m1 = json.loads((r1 / f"measure-seed{seed}.json").read_text())["arms"]["fifo"]
        res = fifo_result or {}
        out["run_id"] = [m1.get("run_id"), res.get("run_id")]
        out["config_hash"] = [m1.get("config_hash"), res.get("config_hash")]
        ids_ok = (
            out["run_id"][0] == out["run_id"][1]
            and out["config_hash"][0] == out["config_hash"][1]
        )
        a1, a2 = r1 / "fifo" / f"seed{seed}", Path(out_root) / "fifo" / f"seed{seed}"
        hb_ok = comparable_heartbeat(a1 / "heartbeat.jsonl") == comparable_heartbeat(
            a2 / "heartbeat.jsonl"
        )
        dg_ok = [_strip(x) for x in _jsonl(a1 / "diagnostics.jsonl")] == [
            _strip(x) for x in _jsonl(a2 / "diagnostics.jsonl")
        ]
        ref = R1.read_pass(r1 / "heldout" / f"seed{seed}", "fifo")
        ho_ok = (
            run2_fifo_recs is not None and ref[: len(run2_fifo_recs)] == run2_fifo_recs
        )
        out.update(
            ids_equal=ids_ok,
            heartbeat_equal=hb_ok,
            diagnostics_equal=dg_ok,
            heldout_records_equal=ho_ok,
            heldout_n_compared=None if run2_fifo_recs is None else len(run2_fifo_recs),
            ok=ids_ok and hb_ok and dg_ok and ho_ok,
        )
    except Exception as e:  # did not run: recorded, never a pass
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def c14_es_source(seed: int, refs: Refs = REFS) -> dict:
    """C14 (scoped to the ES contrast): the pinned manifest verifies
    (``shasum -a 256 -c``, rc read directly); the seed's ``es.jsonl.gz`` has PREREG §4
    item 4's sha256 and its ``eval.json`` ``records_sha256``; ``eval.json`` names
    L48-k1e-2."""
    out: dict[str, Any] = {}
    try:
        p = subprocess.run(
            ["shasum", "-a", "256", "-c", "MANIFEST.sha256"],
            cwd=refs.es_pinned,
            capture_output=True,
            text=True,
            check=False,
        )
        out["manifest_rc"] = p.returncode
        out["manifest_out"] = p.stdout.strip().splitlines()[-6:]
        d = Path(refs.es_out) / "eval" / f"seed{seed}"
        ev = json.loads((d / "eval.json").read_text())
        sha = sha256_file(d / "es.jsonl.gz")
        want = dict(refs.es_sha).get(seed)
        out.update(es_sha256=sha, registered=want, eval_json=ev)
        out["ok"] = (
            p.returncode == 0
            and sha == want
            and sha == ev.get("records_sha256")
            and ev.get("config") == ES_CONFIG
        )
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def _tg(recs: list[dict]) -> list:
    return [(r["doc"], [(a[0], a[1]) for a in r["answers"]]) for r in recs]


def c15_pairing(
    seed: int,
    fifo_recs: list[dict],
    psic_recs: list[dict],
    es_recs: list[dict],
    refs: Refs = REFS,
) -> dict:
    """C15 (scoped): run 2's fifo records equal run 1's fifo vector record for record (the
    vector the referendum paired against; its sha256 is PREREG §7's), and the
    ``(doc, [(t, gap)])`` sequence of run 2's psiC pass equals ES's."""
    out: dict[str, Any] = {}
    try:
        d = Path(refs.run1_out) / "heldout" / f"seed{seed}"
        sha = sha256_file(d / "fifo.jsonl.gz")
        ref = R1.read_pass(d, "fifo")[: len(fifo_recs)]
        out.update(
            run1_fifo_sha256=sha,
            registered=dict(refs.run1_fifo_sha).get(seed),
            n_compared=len(fifo_recs),
        )
        out["fifo_sha_ok"] = sha == out["registered"]
        out["fifo_records_equal"] = ref == fifo_recs
        out["psiC_es_tgap_equal"] = _tg(psic_recs) == _tg(es_recs[: len(psic_recs)])
        out["ok"] = (
            out["fifo_sha_ok"] and out["fifo_records_equal"] and out["psiC_es_tgap_equal"]
        )
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


# --------------------------------------------------------------------------- #
# measure-seed
# --------------------------------------------------------------------------- #


RUN_LEVEL_CONTROLS = ("Cdelta", "C2prime", "C9", "C11", "C12", "C13")


def _paired(records: dict[str, list[dict]], seed: int, n_boot: int) -> tuple:
    order = [r["doc"] for r in records["fifo"]]
    for k, rs in records.items():
        if [r["doc"] for r in rs] != order:
            raise R1.ControlFailure(f"{k} is not in the fifo pass's document order")
    return order, R1.boot_weights(len(order), seed, n_boot)


def run2_heldout(doc: dict, cfg, out_root: Path, refs: Refs, delta, limit) -> dict:
    """PREREG §2 label, §4 items 1-4, §5 (age-1 share, probe incl. run 1 retro)."""
    hd = Path(out_root) / "heldout" / f"seed{cfg.seed}"
    recs = {a: R1.read_pass(hd, a) for a in ARMS}
    order, W = _paired(recs, cfg.seed, cfg.n_boot)
    buckets = R1.buckets_for(cfg.memory_slots)
    h: dict[str, Any] = {"n_documents": len(order)}
    con = (doc.get("heldout") or {}).get("contrasts") or {}
    h["headline"] = label_contrast(con.get(HEADLINE), delta)
    h["gap_buckets"] = {
        b: con.get(f"acc.{b}.psiC_minus_fifo") for b in ("gap_2_to_M", "gap_gt_M")
    }
    # §4 item 1: hit rate and the FactFiller ceiling share
    docs = R1.docs_by_id(cfg, order)
    hits = {a: [r["hits"] for r in recs[a]] for a in ARMS}
    hits["factfiller"] = factfiller_hits(docs, cfg.seed, cfg.memory_slots)
    hb: dict[str, Any] = {}
    for b in ("all", "gap_2_to_M", "gap_gt_M"):
        per = {k: hit_counts(v, buckets[b]) for k, v in hits.items()}
        hb[b] = {
            "hit": {k: R1.level(p, W) for k, p in per.items()},
            "psiC_minus_fifo": R1.contrast(per, "psiC", "fifo", W),
            "ceiling_share": {a: ceiling(per, a, W) for a in RSR_ARMS},
        }
    h["hit_rate"] = hb
    try:
        cf = ledger_rows(refs.cf_ledger)[CF_CAP_KEY]
        h["b0cf_cap_g09"] = {
            "value": cf["samples"][cfg.seed],
            "key": CF_CAP_KEY,
            "ledger": refs.cf_ledger,
            "note": "beside, never contrasted",
        }
    except Exception as e:
        h["b0cf_cap_g09"] = {"value": None, "error": f"{type(e).__name__}: {e}"}
    # §4 item 3: psiC(run 2) - psiC(run 1)
    try:
        r1p = R1.read_pass(Path(refs.run1_out) / "heldout" / f"seed{cfg.seed}", "psiC")
        r1p = [r for r in r1p if r["doc"] in set(order)]
        same = [r["doc"] for r in r1p] == order and [len(r["answers"]) for r in r1p] == [
            len(r["answers"]) for r in recs["psiC"]
        ]
        if same:
            per = {
                "run2": R1.answer_counts(recs["psiC"], buckets["all"]),
                "run1": R1.answer_counts(r1p, buckets["all"]),
            }
            h["psiC_run2_minus_run1"] = R1.contrast(per, "run2", "run1", W)
        else:
            h["psiC_run2_minus_run1"] = {
                "omitted": "document ids or per-document query counts differ"
            }
    except Exception as e:
        h["psiC_run2_minus_run1"] = {"omitted": f"{type(e).__name__}: {e}"}
    # §4 item 4: psiC - pinned Expire-Span, scoped by C14 and C15
    sc: dict[str, Any] = {"C14": c14_es_source(cfg.seed, refs)}
    try:
        es = R1.read_pass(Path(refs.es_out) / "eval" / f"seed{cfg.seed}", "es")
        es = [r for r in es if r["doc"] in set(order)]
    except Exception as e:
        es = None
        sc["C15"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if es is not None:
        sc["C15"] = c15_pairing(cfg.seed, recs["fifo"], recs["psiC"], es, refs)
    if sc["C14"]["ok"] and sc["C15"]["ok"]:
        out = {}
        for b in ("all", "gap_2_to_M", "gap_gt_M"):
            per = {
                "psiC": R1.answer_counts(recs["psiC"], buckets[b]),
                "es": R1.answer_counts(es, buckets[b]),
            }
            out[b] = label_contrast(R1.contrast(per, "psiC", "es", W), delta)
        h["psiC_minus_es"] = {
            **out,
            "gates_run2": False,
            "note": "SECONDARY: E1's number, reported early (PREREG §4 item 4)",
        }
    else:
        failed = [k for k in ("C14", "C15") if not sc[k]["ok"]]
        h["psiC_minus_es"] = {
            "omitted": f"scoped control(s) {failed} failed (PREREG §4 item 4)"
        }
    doc["scoped_controls"].update(sc)
    # §5: the age-1 share (run 2's arms, and run 1's psiC labelled seen)
    h["age1_share"] = {a: age1_share(recs[a]) for a in ARMS}
    r1p_all = None
    try:
        r1p_all = R1.read_pass(
            Path(refs.run1_out) / "heldout" / f"seed{cfg.seed}", "psiC"
        )
        r1p_all = [r for r in r1p_all if r["doc"] in set(order)]
        h["age1_share_run1_psiC_SEEN"] = age1_share(r1p_all)
    except Exception as e:
        h["age1_share_run1_psiC_SEEN"] = {
            "share": None,
            "error": f"{type(e).__name__}: {e}",
        }
    # §5: the self-alignment probe, run 2 and run 1 (retroactive, labelled seen)
    vmap = R1.vocab_map(cfg)
    probes: dict[str, Any] = {}
    payload2 = R1.read_checkpoint(
        R1.checkpoint_path(R1.arm_dir(out_root, "psiC", cfg.seed), cfg.iters)
    )
    pr = probe_pass(cfg, payload2, docs, vmap)
    probes["run2_psiC"] = probe_stats(pr["decisions"], len(order), W, cfg.memory_slots)
    probes["run2_psiC"]["victims_equal_psiC_pass"] = pr["victims"] == [
        r["victims"] for r in recs["psiC"]
    ]
    try:
        p1 = Path(refs.run1_out) / "psiC" / f"seed{cfg.seed}" / "ckpt-003000.pt"
        payload1 = R1.read_checkpoint(p1)
        pr1 = probe_pass(cfg, payload1, docs, vmap)
        st = probe_stats(pr1["decisions"], len(order), W, cfg.memory_slots)
        r1p_v = None if r1p_all is None else [r["victims"] for r in r1p_all]
        st["victims_equal_run1_psiC_records"] = r1p_v == pr1["victims"]
        st["label"] = "run 1 psiC ckpt-003000, retroactive: ALREADY SEEN (PREREG §5)"
        st["checkpoint"] = str(p1)
        probes["run1_psiC_SEEN"] = st
    except Exception as e:
        probes["run1_psiC_SEEN"] = {"error": f"{type(e).__name__}: {e}"}
    h["probe"] = probes
    h["limit"] = limit
    return h


def measure_seed(
    cfg,
    owner,
    out_root: Path,
    *,
    threads: int,
    limit: int | None = None,
    dry_run: bool = False,
    refs: Refs = REFS,
    environ=None,
) -> dict:
    """Run 1's `measure_seed` (every run-1 control, pass and number), then PREREG
    §2-§7's run-2 additions on top; ``measure-seed<s>.json`` is rewritten with both."""
    out_root = Path(out_root)
    doc = R1.measure_seed(
        cfg,
        owner,
        out_root,
        threads=threads,
        limit=limit,
        dry_run=dry_run,
        environ=environ,
    )
    doc["run2"] = {
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "target": TARGET,
        "limit": limit,
        "smoke_or_dry": dry_run,
    }
    dl = read_delta(cfg.seed, refs)
    doc["controls"]["Cdelta"] = dl
    arms = doc.get("arms") or {}
    hbs = {a: R1.read_heartbeat(R1.arm_dir(out_root, a, cfg.seed)) for a in ARMS}
    results = {
        a: R1.read_cli_result(Path(arms[a]["log"])) if a in arms else None for a in ARMS
    }
    ran = all(results.get(a) is not None for a in ARMS)
    if ran:
        t_warm = float(
            ((results["psiC"]["config"].get("epoch")) or {}).get("t_warm_steps")
        )
        diag = {a: target_diagnostics(hbs[a]["beats"], t_warm) for a in RSR_ARMS}
        doc["run2"]["target_diagnostics"] = diag
        doc["controls"]["C2prime"] = c2_prime(doc["controls"].get("C2"))
        doc["controls"]["C9"] = c9_coverage(diag)
        doc["controls"]["C11"] = c11_target(results, hbs, doc.get("launch"))
        doc["controls"]["C13"] = c13_finite(hbs)
    fifo_recs = None
    hd = out_root / "heldout" / f"seed{cfg.seed}"
    if doc.get("heldout") is not None and (hd / "fifo.jsonl.gz").exists():
        fifo_recs = R1.read_pass(hd, "fifo")
        doc["run2"]["heldout"] = run2_heldout(
            doc, cfg, out_root, refs, dl.get("delta"), limit
        )
    if ran:
        doc["controls"]["C12"] = c12_fifo(
            cfg.seed, out_root, results.get("fifo"), fifo_recs, refs
        )
    # Status: run 1's, worsened by any failed run-2 run-level control (3 < 5 kept).
    failed2 = [
        k
        for k in RUN_LEVEL_CONTROLS
        if (doc["controls"].get(k) or {}).get("ok") is not True
    ]
    doc["run2"]["run_level_controls_failed"] = failed2
    if failed2:
        doc["notes"].append(f"CONTROL_FAILED (run 2): {failed2}")
        if R1.EXIT_RANK[int(doc["exit"])] < R1.EXIT_RANK[int(Exit.DID_NOT_RUN)]:
            doc.update(exit=int(Exit.DID_NOT_RUN), status="did_not_run")
    doc["controls_failed"] = [
        k for k, v in doc["controls"].items() if v.get("ok") is not True
    ]
    (out_root / f"measure-seed{cfg.seed}.json").write_text(
        json.dumps(doc, indent=2, default=str) + "\n"
    )
    return doc


# --------------------------------------------------------------------------- #
# finalize: class, newcomer label, RESULTS.md, ledger, claims.json
# --------------------------------------------------------------------------- #


def summarize(docs: list[dict]) -> dict:
    labels, shares, es_labels = {}, {}, {}
    for d in docs:
        h = ((d.get("run2") or {}).get("heldout")) or {}
        if h.get("headline"):
            labels[d["seed"]] = h["headline"]["label"]
        shares[d["seed"]] = ((h.get("age1_share") or {}).get("psiC") or {}).get("share")
        es = (h.get("psiC_minus_es") or {}).get("all")
        if es:
            es_labels[d["seed"]] = es["label"]
    readable = all(int(d["exit"]) == int(Exit.OK) for d in docs) and len(docs) == len(
        SEEDS
    )
    cls = run_class(labels)
    return {
        "labels": labels,
        "class": cls,
        "route": CLASS_ROUTE[cls],
        "class_readable": readable,
        "newcomer": newcomer_label(shares),
        "age1_share_psiC": shares,
        "es_labels_secondary": es_labels,
    }


def finalize(
    docs: list[dict],
    owner,
    out_root: Path,
    *,
    results_path: Path,
    write_ledger: bool = True,
) -> Exit:
    worst = max(docs, key=lambda d: (R1.EXIT_RANK[int(d["exit"])], -int(d["seed"])))
    code = Exit(worst["exit"])
    run = {
        "run_id": RUN_ID,
        "seeds": [d["seed"] for d in docs],
        "exit": int(code),
        "status": worst["status"],
        "owner_values": dataclasses.asdict(owner),
        "summary": summarize(docs),
        "headline_run1_form": R1.headline(docs),
        "tree": R1.tree_state(),
    }
    out_root = Path(out_root)
    (out_root / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    Path(results_path).write_text(render_results(run, docs, out_root))
    if write_ledger:
        write_ledger_run2(run, docs, out_root)
        write_claims(run, docs, out_root)
    return code


def write_ledger_run2(run: dict, docs: list[dict], out_root: Path) -> Path:
    L = R1.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=Path(out_root).parent)
    lg = Path(out_root) / "launch-git.json"
    if lg.exists():
        led.doc["config_hash"] = hashlib.sha256(lg.read_bytes()).hexdigest()
        led.doc["manifest_written_utc"] = json.loads(lg.read_text()).get("launched_utc")
    for d in docs:
        for a, r in (d.get("arms") or {}).items():
            led.command(
                r["argv"], exit_code=r.get("rc"), note=f"arm {a} seed {d['seed']}"
            )
    led.command(
        [".venv/bin/python", EXPERIMENT, "finalize"],
        exit_code=run["exit"],
        note="this measurement",
    )
    by_job = {
        f"{a}-s{d['seed']}": n
        for d in docs
        for a, n in sorted((d.get("steps_done") or {}).items())
    }
    ran = sorted({s for d in docs for s in d.get("seeds_actually_run") or []})
    led.run_meta(
        device=R1.DEVICE,
        seeds_actually_run=ran,
        steps_requested=run["owner_values"]["iters"],
        steps_done=min(by_job.values()) if by_job else None,
    )
    led.note("summary", run["summary"], how=f"{EXPERIMENT} run.json")
    led.note("prereg", {"path": PREREG, "commit": PREREG_COMMIT}, how=EXPERIMENT)
    for d in docs:
        keys = ("controls", "scoped_controls", "heldout", "training", "arms", "launch")
        for key, val in R1._flatten(f"seed{d['seed']}", {k: d.get(k) for k in keys}):
            led.note(key, val, how=f"{EXPERIMENT} measure-seed{d['seed']}.json")
        for key, val in R1._flatten(f"seed{d['seed']}.run2", d.get("run2") or {}):
            led.note(key, val, how=f"{EXPERIMENT} measure-seed{d['seed']}.json run2")
    led.status(run["status"])
    return led.write()


QUESTION = (
    "On fresh-stream arm B's ckpt-003000 with the transformer frozen, when psi_hat is "
    "trained online on the on-policy resample-knockout LOO return at gamma 0.9, does "
    "the trained eviction rule beat FIFO on held-out all-query accuracy, and does it "
    "still evict newcomers?"
)


def _fmt(x, nd: int = 4) -> str:
    return "None" if not _finite(x) else f"{x:+.{nd}f}"


def _ci(c: dict | None) -> str:
    if not c or "point" not in c:
        return str((c or {}).get("omitted", "n/a"))
    return f"{_fmt(c.get('point'))} [{_fmt(c.get('lo'))}, {_fmt(c.get('hi'))}]"


def _seed_lines(d: dict, p) -> None:
    h = ((d.get("run2") or {}).get("heldout")) or {}
    ctl, sc = d["controls"], d["scoped_controls"]
    p(f"## Seed {d['seed']} (exit `{d['exit']}`, status `{d['status']}`)\n")
    p(f"- controls failed: `{d.get('controls_failed')}`")
    p(f"- notes: `{d.get('notes')}`")
    for k in ("Cdelta", "C2prime", "C9", "C11", "C12", "C13"):
        p(f"- **{k}** ok `{(ctl.get(k) or {}).get('ok')}`")
    for k in ("C9_C9b", "C14", "C15"):
        p(f"- scoped **{k}** ok `{(sc.get(k) or {}).get('ok')}`")
    if not h:
        p("- no held-out reading\n")
        return
    gb = h["gap_buckets"]
    p(f"- documents read: {h.get('n_documents')} (limit `{h.get('limit')}`)")
    p(f"- psiC - FIFO, gap 2..M: {_ci(gb.get('gap_2_to_M'))}")
    p(f"- psiC - FIFO, gap > M: {_ci(gb.get('gap_gt_M'))}")
    hr = h["hit_rate"]["all"]
    p("- hit rate: " + ", ".join(f"{k} {_ci(v)}" for k, v in hr["hit"].items()))
    cs = hr["ceiling_share"]
    p(f"- ceiling share: psiC {_ci(cs['psiC'])}, age {_ci(cs['age'])}")
    cap = h["b0cf_cap_g09"].get("value")
    p(f"- B0-CF gamma=0.9 cap, beside and never contrasted: {cap}")
    p(f"- psiC(run 2) - psiC(run 1): {_ci(h.get('psiC_run2_minus_run1'))}")
    es = h.get("psiC_minus_es") or {}
    if "all" in es:
        p("- psiC - pinned ES L48-k1e-2 (SECONDARY; does not gate run 2):")
        for b in ("all", "gap_2_to_M", "gap_gt_M"):
            p(f"  - {b}: {_ci(es[b])} **{es[b]['label']}**")
    else:
        p(f"- psiC - pinned ES: {es.get('omitted')}")
    a1 = ", ".join(f"{a} {v.get('share')}" for a, v in h["age1_share"].items())
    p(f"- age-1 share: {a1}")
    p(f"- run 1 psiC age-1 share (SEEN): {h['age1_share_run1_psiC_SEEN'].get('share')}")
    for name, st in h["probe"].items():
        if "rho_lower_median" not in st:
            p(f"- probe {name}: {st}")
            continue
        p(f"- probe {name}: {st['n_decisions']} decisions")
        p(f"  - rho(psi_hat, cos) lower median: {_ci(st['rho_lower_median'])}")
        p(f"  - victim cos is the step max: {_ci(st['victim_cos_is_max_share'])}")
        p(
            f"  - cells older than M: {st['cells_older_than_M']} (equals the M+1..A_max "
            f"bucket's count: {st['older_than_M_equals_bucket']}; A_max {st['a_max']})"
        )
        for b, v in st["by_age_bucket"].items():
            p(
                f"  - age {b}: r(pct) {_ci(v['pearson_pct'])}; mean psi_hat pct "
                f"{_ci(v['mean_psi_pct'])}; mean cos pct {_ci(v['mean_cos_pct'])}; "
                f"n {v['n_cells']}"
            )
    td = (d.get("run2") or {}).get("target_diagnostics") or {}
    for a, v in td.items():
        for ph in ("warm", "post_warm"):
            x = v[ph]
            rho = x["rho_g_age_step_medians"].get("median")
            p(
                f"- target diag {a} {ph}: steps {x['n_steps']}, full coverage "
                f"{x['full_coverage']}, rho(G, age) median of step medians {rho}, "
                f"loo_s median {x['loo_s'].get('median')}"
            )
    p("")


def render_results(run: dict, docs: list[dict], out_root: Path) -> str:
    s = run["summary"]
    w: list[str] = []
    p = w.append
    p("# RESULTS: RSR run 2, the C3 LOO target\n")
    p(f"Generated by `{EXPERIMENT} finalize` from `{out_root}/measure-seed<s>.json`.")
    p("No number is typed. Written by a model; not by Brendan.")
    p(f"PREREG `{PREREG}` @ `{PREREG_COMMIT}`.\n")
    p(f"- exit `{run['exit']}`, status `{run['status']}`; tree `{run['tree']}`")
    p(f"- owner values (run 1's, by citation): `{run['owner_values']}`\n")
    p("## 1. Headline (PREREG §2): psiC - FIFO, all-query held-out accuracy\n")
    p("| seed | delta_seed | point [95% CI] | label |")
    p("|---|---|---|---|")
    for d in docs:
        h = ((d.get("run2") or {}).get("heldout") or {}).get("headline") or {}
        p(f"| {d['seed']} | {h.get('delta')} | {_ci(h)} | **{h.get('label', 'n/a')}** |")
    p(f"\n**Class: {s['class']}** -> {s['route']}.")
    if not s["class_readable"]:
        p("**NOT READ: a seed's run-level control failed, or a seed is missing.**")
    p(f"\nNewcomer label (§5, psiC age-1 share): **{s['newcomer']}**")
    p(f"(shares by seed: {s['age1_share_psiC']})\n")
    for d in docs:
        _seed_lines(d, p)
    p("## What this run does not show (PREREG §11)\n")
    p("No scaling claim. No cognitive claim. §15 untouched. Falsifier 6 is deferred")
    p("to E1; the ES contrast gates nothing here. The return is censored at the")
    p("policy's own evictions (shadow off).")
    return "\n".join(w) + "\n"


def write_claims(run: dict, docs: list[dict], out_root: Path) -> Path:
    """``runs/rsr-run2/claims.json``: ``{"claim", "command", "expected"}`` for
    `orchestrator.verify draw/check`. Every command RECOMPUTES its number from the
    per-document records (``run.py recompute``), never from this run's ledger."""
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        f"--slots 1 -- .venv/bin/python {EXPERIMENT} recompute --out-root {out_root}"
    )
    claims = []
    for d in docs:
        s = d["seed"]
        h = ((d.get("run2") or {}).get("heldout")) or {}
        if not h:
            continue
        hl = h["headline"]
        claims.append(
            {
                "claim": f"seed {s} headline psiC - FIFO and label",
                "command": f"{base} headline --seed {s}",
                "expected": recompute_line("headline", hl),
            }
        )
        claims.append(
            {
                "claim": f"seed {s} psiC age-1 eviction share",
                "command": f"{base} age1 --seed {s}",
                "expected": recompute_line("age1", h["age1_share"]["psiC"]),
            }
        )
        claims.append(
            {
                "claim": f"seed {s} C14 ES source sha256 and config",
                "command": f"{base} c14 --seed {s}",
                "expected": recompute_line("c14", d["scoped_controls"].get("C14") or {}),
            }
        )
        es = (h.get("psiC_minus_es") or {}).get("all")
        if es:
            claims.append(
                {
                    "claim": f"seed {s} psiC - pinned ES (secondary) and label",
                    "command": f"{base} es --seed {s}",
                    "expected": recompute_line("headline", es),
                }
            )
    claims.append(
        {
            "claim": "the class and the newcomer label",
            "command": f"{base} class",
            "expected": f"{run['summary']['class']} {run['summary']['newcomer']}",
        }
    )
    p = Path(out_root) / "claims.json"
    p.write_text(json.dumps(claims, indent=1) + "\n")
    return p


def recompute_line(kind: str, x: dict) -> str:
    if kind == "headline":
        return (
            f"{x.get('point'):.6f} {x.get('lo'):.6f} {x.get('hi'):.6f} {x.get('label')}"
        )
    if kind == "age1":
        return f"{x['n_age1']} {x['n_evictions']}"
    if kind == "c14":
        return f"{x.get('ok')} {x.get('es_sha256')}"
    raise ValueError(kind)


def recompute(what: str, out_root: Path, seed: int | None, refs: Refs = REFS) -> str:
    """Recompute one claimed number from the per-document records (claims.json)."""
    if what == "class":
        docs = [
            json.loads((out_root / f"measure-seed{s}.json").read_text()) for s in SEEDS
        ]
        labels, shares = {}, {}
        for d in docs:
            line = recompute("headline", out_root, d["seed"], refs).split()
            labels[d["seed"]] = line[3]
            n1, n = recompute("age1", out_root, d["seed"], refs).split()
            shares[d["seed"]] = int(n1) / int(n) if int(n) else None
        return f"{run_class(labels)} {newcomer_label(shares)}"
    hd = out_root / "heldout" / f"seed{seed}"
    if what == "age1":
        return recompute_line("age1", age1_share(R1.read_pass(hd, "psiC")))
    if what == "c14":
        return recompute_line("c14", c14_es_source(seed, refs))
    recs = {a: R1.read_pass(hd, a) for a in ("fifo", "psiC")}
    other = "fifo"
    if what == "es":
        es = R1.read_pass(Path(refs.es_out) / "eval" / f"seed{seed}", "es")
        order = {r["doc"] for r in recs["fifo"]}
        recs["es"] = [r for r in es if r["doc"] in order]
        other = "es"
    order, W = _paired(recs, seed, _n_boot(out_root, seed))
    b = R1.buckets_for(M)["all"]
    per = {k: R1.answer_counts(v, b) for k, v in recs.items()}
    c = R1.contrast(per, "psiC", other, W)
    return recompute_line("headline", label_contrast(c, read_delta(seed, refs)["delta"]))


def _n_boot(out_root: Path, seed: int) -> int:
    d = json.loads((out_root / f"measure-seed{seed}.json").read_text())
    return int(d["config"]["n_boot"])


# --------------------------------------------------------------------------- #
# preflight and CLI
# --------------------------------------------------------------------------- #


def preflight(refs: Refs = REFS) -> dict:
    """Before any training: run 1's C0 per seed and substrate pins (R1.preflight's checks
    that are not run 1's launcher text), the delta table, the ES and run-1 references, and
    that every sha256 this file holds appears verbatim in the committed PREREG."""
    out: dict[str, Any] = {"seeds": {}}
    text = Path(refs.prereg).read_text()
    out["shas_in_prereg"] = all(
        v in text for v in (*ES_SHA.values(), *RUN1_FIFO_SHA.values())
    )
    out["launcher_exists"] = (ROOT / LAUNCHER).exists()
    for s in SEEDS:
        cfg = R1.prereg_config(s, 3)
        row: dict[str, Any] = {}
        try:
            row["C0"] = R1.check_c0(cfg)
        except R1.ControlFailure as e:
            row["C0"] = {"ok": False, "error": str(e)}
        ck = Path(cfg.substrate.checkpoint)
        row["substrate_sha256_ok"] = (
            ck.exists() and sha256_file(ck) == cfg.substrate.sha256
        )
        row["delta"] = read_delta(s, refs)
        row["C14"] = c14_es_source(s, refs)
        hd = Path(refs.run1_out) / "heldout" / f"seed{s}"
        row["run1_fifo_sha_ok"] = (hd / "fifo.jsonl.gz").exists() and sha256_file(
            hd / "fifo.jsonl.gz"
        ) == RUN1_FIFO_SHA[s]
        row["run1_psiC_ckpt"] = (
            Path(refs.run1_out) / "psiC" / f"seed{s}" / "ckpt-003000.pt"
        ).exists()
        out["seeds"][s] = row
    out["ok"] = (
        out["shas_in_prereg"]
        and out["launcher_exists"]
        and all(
            r["C0"]["ok"]
            and r["substrate_sha256_ok"]
            and r["delta"]["ok"]
            and r["C14"]["ok"]
            and r["run1_fifo_sha_ok"]
            and r["run1_psiC_ckpt"]
            for r in out["seeds"].values()
        )
    )
    return out


def _owner_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gamma", type=float, required=True, help="O1 (run 1's)")
    p.add_argument("--phi-weight-decay", type=float, required=True, help="O4 (run 1's)")
    p.add_argument("--iters", type=int, required=True, help="R-1 (run 1's)")
    p.add_argument("--o5", required=True, help="O5 (run 1's)")


def _owner(a):
    ov = R1.OwnerValues(a.gamma, a.phi_weight_decay, a.iters, a.o5, SEEDS)
    R1.validate_owner_values(ov)
    return ov


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = ArgumentParser(prog="rsr-run2", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    ps = sub.add_parser("measure-seed")
    _owner_args(ps)
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--threads", type=int, required=True)
    ps.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    ps.add_argument(
        "--smoke",
        action="store_true",
        help="a launcher SMOKE=1 output: OUT / JOB_SUFFIX read (C10 fails by design)",
    )
    ps.add_argument(
        "--limit", type=int, default=None, help="smoke only: the first N EVAL documents"
    )
    pf = sub.add_parser("finalize")
    _owner_args(pf)
    pf.add_argument("--out-root", default=str(ROOT / "runs" / RUN_ID))
    pf.add_argument("--results", default=None)
    pr = sub.add_parser("recompute")
    pr.add_argument("--out-root", required=True)
    pr.add_argument("what", choices=("headline", "age1", "c14", "es", "class"))
    pr.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)
    try:
        if a.cmd == "preflight":
            r = preflight()
            print(json.dumps(r, indent=2, default=str))
            return Exit.OK if r["ok"] else Exit.DID_NOT_RUN
        if a.cmd == "measure-seed":
            ow = _owner(a)
            if a.limit is not None and not a.smoke:
                raise R1.Refused(
                    "--limit is a smoke-only override (PREREG §3: all of EVAL)"
                )
            out = Path(a.out_root).resolve()
            if a.smoke and out == (ROOT / "runs" / RUN_ID).resolve():
                raise R1.Refused("a smoke never writes runs/rsr-run2")
            doc = measure_seed(
                R1.prereg_config(a.seed, ow.iters),
                ow,
                out,
                threads=a.threads,
                limit=a.limit,
                dry_run=a.smoke,
            )
            return status(doc["exit"])
        if a.cmd == "finalize":
            ow = _owner(a)
            out = Path(a.out_root)
            docs = []
            for s in SEEDS:
                f = out / f"measure-seed{s}.json"
                if not f.exists():
                    raise R1.Refused(f"{f} is missing: run measure-seed --seed {s} first")
                docs.append(json.loads(f.read_text()))
            res = Path(a.results) if a.results else out / "RESULTS.md"
            return status(finalize(docs, ow, out, results_path=res))
        if a.cmd == "recompute":
            print(recompute(a.what, Path(a.out_root), a.seed))
            return Exit.OK
    except R1.Refused as e:
        print(f"DID NOT RUN: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    except R1.ControlFailure as e:
        print(f"CONTROL FAILED: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    return Exit.DID_NOT_RUN


if __name__ == "__main__":
    run_main(main)
