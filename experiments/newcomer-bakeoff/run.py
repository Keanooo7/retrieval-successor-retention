"""The newcomer bake-off (ADR-0009 sign-off pack §4.3): the measurement half.

Governing text: `experiments/newcomer-bakeoff/PREREG.md`, committed alone at
`801635c` before this file existed. Every rule cites the section it transcribes.

Offline, ckpt3000, gamma = 0.9, three seeds, `ProbeArgminPolicy` (B2's, imported),
never `RSRPolicy`. Nothing is trained by gradient: the transformer and `W_sent` are
read only (CLAUDE.md). Age never enters a psi-hat: the grace rule (arm (c)) is a
mask in the argmin, outside the score, and whether it is acceptable at all is
OWNER-ONLY (PREREG §0). No `rsr.constants.record()` call; `K` and `lambda_shadow`
are READ from the registry.

Entry points (exit codes: `rsr.exit_codes`)::

    run.py check                       # §2 ranges, aliasing; no model (0 / 3)
    run.py run [--parallel 12]         # fit children, eval children, aggregate
    run.py aggregate                   # re-aggregate existing units into the ledger
    run.py fit-child --seed S          # (internal)
    run.py eval-child --seed S --shard J --n-shards N   # (internal)
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr import constants as RC  # noqa: E402
from rsr.baselines.lru import LRUPolicy  # noqa: E402
from rsr.data.synthetic import discounted_demand  # noqa: E402
from rsr.exit_codes import (  # noqa: E402
    ArgumentParser,
    Exit,
    did_not_run,
    run_main,
    status,
)


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AN = _load("_nb_analysis", "experiments/newcomer-bakeoff/analysis.py")
B2 = AN.B2
LR = B2.LR

EXPERIMENT = "experiments/newcomer-bakeoff/run.py"
PREREG = "experiments/newcomer-bakeoff/PREREG.md"
PREREG_COMMIT = "801635c"
RUN_ID = "newcomer-bakeoff"

# --------------------------------------------------------------------------- #
# 🔒 PREREG, transcribed.
# --------------------------------------------------------------------------- #

SEEDS = AN.SEEDS
LABEL = 3000  # §1: ckpt3000 only
GAMMA = AN.GAMMA
GRACE_GS = AN.GRACE_GS
ARMS = AN.ARMS
EVAL_NB = (990000, 992048)  # §2: fresh
N_E = 2048  # §2: fixed before any data
N_F = 264  # B2 TBD-1; asserted equal to B2.n_f(p)
K_SHADOW = int(RC.get("K", "synthetic"))  # §5: read, never typed (40)
LAMBDA_SHADOW = float(RC.get("lambda_shadow"))  # §5: FROZEN 0.5
R1_LAMBDAS = {"R1": LAMBDA_SHADOW, "R1L1": 1.0}  # §3: (d) and the (d') companion
C5_MSE_REL = 1e-6  # §8 C5 (ii)
C5_ARGMIN_AGREE = 0.999  # §8 C5 (iii)
N_DETERMINISM_DOCS = B2.N_DETERMINISM_DOCS  # §8 C8 (8)
NEW_ARMS_C8 = ("psiR1", "psiR1L1", *(f"graceU{g}" for g in GRACE_GS), "lru")

#: §1: B2's phase-A files, pinned. Read, never recomputed.
B2_FIT_SHA256 = {
    0: "0227d80a2f77d52ff84b4311250636ce55e1509ed5a89fb833b3f406336a4846",
    1: "cdb2c3640a584dcac3dbb22a20025df21f0aefa0f238cce1d83d219e4c98c0db",
    2: "5ed0142cb0601046e790c470607f65273f802f9edf0899944f0867ac76861cfe",
}

QUESTION = (
    "Newcomer bake-off (pack §4.3): offline at ckpt3000, gamma 0.9, which newcomer "
    "design avoids psi-U's harm -- psi-C, psi-U, psi-U + hard grace (g=1,2,4), the R1 "
    "shadow target (K=40, lambda_shadow=0.5), U+/C+ -- against FIFO, LRU, age-only, "
    "random, kind-oracle, oracle?"
)
FALSIFIER = (
    "none of the spec's falsifiers directly: MEASURE-THEN-DECIDE evidence for "
    "ADR-0009 L3/L4 and the newcomer options (pack §4); §7.1 vacuity (falsifier 1) is "
    "read on the effective rule as a report, not a gate."
)
EXPECTED = (
    "PREREG §9: replication U LOSS s0,s1 / C EQUIV x3 ~70%; DQ1 NO 60 / MIXED 30 / "
    "YES 10; DQ2 INTERMEDIATE 40 / HARMFUL 35 / SAFE 25; DQ3 YES 55 / NO 25 / other 20."
)


class ControlFailure(B2.ControlFailure):
    """A §8 control failed. Exit 3."""


# --------------------------------------------------------------------------- #
# §2: ranges
# --------------------------------------------------------------------------- #

#: Every range this run must avoid: B2's claimed ranges (EVAL [940000, 980000)
#: included), E0d's reservation and D_E0d, and B2's used-range table.
AVOID = {
    **{f"B2_{k}": v for k, v in B2.RANGES.items()},
    "B2_E0D_RESERVED": B2.E0D_RESERVED,
    **B2.USED_RANGES,
}


def range_problems(eval_nb=EVAL_NB, avoid=None) -> list[str]:
    avoid = dict(AVOID if avoid is None else avoid)
    avoid.update(B2.e0d_prereg_ranges())
    probs = []
    lo, hi = eval_nb
    if not lo < hi:
        probs.append(f"EVAL_NB {list(eval_nb)} is empty")
    if hi - lo != N_E:
        probs.append(f"EVAL_NB holds {hi - lo} ids, N_E = {N_E}")
    for name, rng in avoid.items():
        if B2._overlap(eval_nb, rng):
            probs.append(f"EVAL_NB {list(eval_nb)} overlaps {name} {list(rng)}")
    if hi > B2.GEN_SEED_K:
        probs.append(f"EVAL_NB reaches {hi} > {B2.GEN_SEED_K}: seeds could alias")
    for s in SEEDS:  # §2: no cross-seed aliasing (by construction; still asserted)
        mine = (s * B2.GEN_SEED_K + lo, s * B2.GEN_SEED_K + hi)
        for name, (a, b) in avoid.items():
            for s2 in SEEDS:
                theirs = (s2 * B2.GEN_SEED_K + a, s2 * B2.GEN_SEED_K + b)
                if B2._overlap(mine, theirs):
                    probs.append(f"EVAL_NB@seed{s} aliases {name}@seed{s2}")
    return probs


def require_eval_nb(doc_ids) -> None:
    ids = [int(i) for i in doc_ids]
    bad = [i for i in ids if not EVAL_NB[0] <= i < EVAL_NB[1]]
    if not ids or bad:
        raise B2.RangeError(
            f"{len(bad)} id(s) outside EVAL_NB {list(EVAL_NB)}: {bad[:5]}"
        )


def eval_docs(seed: int, lo: int, hi: int) -> list:
    require_eval_nb(range(lo, hi))
    return [B2.doc_by_id(seed, i) for i in range(lo, hi)]


def shard_bounds(j: int, n_shards: int, n: int = N_E) -> tuple[int, int]:
    """Contiguous shards of EVAL_NB, in document order."""
    a = EVAL_NB[0] + (n * j) // n_shards
    b = EVAL_NB[0] + (n * (j + 1)) // n_shards
    return a, b


# --------------------------------------------------------------------------- #
# §5: the R1 shadow target, computed offline from a B2 capture
# --------------------------------------------------------------------------- #


def r1_X(D0: Tensor, resident: Tensor, lam: float, K: int) -> tuple[Tensor, int]:
    """§5: `X_R1[t, i]` = `D0` while FIFO holds `i`; `lam · D0` in the depth-`K`
    shadow window after FIFO drops it (`t - t_e(i) < K`, `t_e(i)` the first step
    FIFO no longer holds `i`); 0 beyond; NaN where `D0` is NaN (`i ≥ t`).
    Returns `(X, n_truncated)`: the cells past the window (expected 0 at S = 48)."""
    S = D0.shape[0]
    t = torch.arange(S).unsqueeze(1).expand(S, S)
    i = torch.arange(S).unsqueeze(0).expand(S, S)
    written = i < t
    gone = written & ~resident
    # t_e(i): the first step with i written and not resident (S if none)
    big = torch.full((S, S), S, dtype=torch.long)
    t_e = torch.where(gone, t, big).min(dim=0).values  # [S] over i
    after = written & (t >= t_e.unsqueeze(0))
    if bool((after & resident).any()):
        raise ControlFailure("a sentence is resident again after FIFO dropped it")
    in_win = after & ((t - t_e.unsqueeze(0)) < K)
    trunc = after & ~in_win
    base = torch.nan_to_num(D0, nan=0.0)
    X = torch.where(
        resident, base, torch.where(in_win, lam * base, torch.zeros_like(base))
    )
    X = X.masked_fill(torch.isnan(D0), float("nan"))
    return X, int(trunc.sum())


_B2_TARGET = B2.target_matrix


def target_matrix(cap, arm: str, gamma: float | None) -> Tensor:
    """B2's `target_matrix`, extended with the R1 targets (§5). Every B2 arm is
    delegated unchanged."""
    if arm in R1_LAMBDAS:
        X, _n = r1_X(cap.D[0], cap.resident, R1_LAMBDAS[arm], K_SHADOW)
        return B2.returns(X, gamma)
    return _B2_TARGET(cap, arm, gamma)


def install_r1_targets() -> None:
    """Point B2's fit machinery (`fit_heads` -> `_stack_targets`) at the extended
    `target_matrix`; the R1 targets use B2's C rows (§5)."""
    B2.target_matrix = target_matrix
    for a in R1_LAMBDAS:
        B2.ROWSET_OF[a] = "C"


# --------------------------------------------------------------------------- #
# §4: the grace wrapper, outside psi-hat
# --------------------------------------------------------------------------- #


class GraceArgmin:
    """§4: argmin of the inner psi-hat over ELIGIBLE live slots; a slot of age
    `≤ g` is ineligible unless every live slot is. `g = 0` is the off-switch
    (the inner policy's victim, exactly). Ties to the lowest slot index, as B2.

    The mask is applied to a detached fp64 score; nothing here reads a tensor
    that requires grad, and the score vector is the inner policy's, unchanged."""

    name = "grace_argmin"

    def __init__(self, inner, g: int, *, arm: str) -> None:
        if not isinstance(inner, B2.ProbeArgminPolicy) or inner.feature != "bilinear":
            raise TypeError("grace wraps a bilinear ProbeArgminPolicy")
        if int(g) < 0:
            raise ValueError(f"g={g}")
        self.inner = inner
        self.g = int(g)
        self.arm = arm
        self.doc = inner.doc
        self.doc_id = inner.doc_id
        self.model_seed = inner.model_seed
        self.log: list[dict] = []

    @staticmethod
    def _argmin(vals: list[float], allowed: list[bool]) -> int:
        j = None
        for q, v in enumerate(vals):
            if allowed[q] and (j is None or v < vals[j]):  # strict: ties to lowest
                j = q
        return j

    def eligible(self, ages: list[int]) -> list[bool]:
        ok = [a > self.g for a in ages]
        return ok if any(ok) else [True] * len(ages)  # "unless all are"

    def select_eviction(self, slots, context: Tensor, step: int) -> int:
        live = torch.nonzero(slots.live).flatten().tolist()
        with torch.no_grad():
            psi = self.inner.scores(slots, context, step)
        if psi.requires_grad:
            raise ControlFailure("grace: the inner score carries graph")
        if not bool(torch.isfinite(psi).all()):
            raise FloatingPointError(f"doc {self.doc_id} step {step}: non-finite psi")
        vals = psi.tolist()
        ages = [step - int(slots.written_at[k]) for k in live]
        j0 = self._argmin(vals, [True] * len(vals))
        j = self._argmin(vals, self.eligible(ages))
        k = live[j]
        rec = B2.eviction_record(
            slots, k, step, doc=self.doc, seed=self.model_seed, arm=self.arm, psi=vals
        )
        rec["unmasked_slot"] = live[j0]
        rec["flipped"] = j != j0
        rec["grace_g"] = self.g
        self.log.append(rec)
        return k

    def observe(self, slots, attn, step: int) -> None:
        return None

    def on_write(self, slots, slot: int, step: int) -> None:
        return None

    def reset(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# §3 / §6.3: logging with content labels
# --------------------------------------------------------------------------- #


def content_of(doc, w: int, step: int) -> int:
    kind = doc.sentences[w].kind
    st = B2._status_of(doc, w, step) if kind == "assert" else None
    return AN.CONTENT_INDEX[AN.content_label(kind, st)]


class LoggedNB(B2.Logged):
    """B2's `Logged`, plus each live slot's content label and the victim's
    position among the live slots (§6.3). Nothing else changes."""

    def __init__(self, inner, *, doc, model_seed, arm: str, check_sum: bool = False):
        if getattr(inner, "doc_id", doc.doc_id) != doc.doc_id:
            raise ControlFailure(
                f"inner policy built for {inner.doc_id}, not {doc.doc_id}"
            )
        super().__init__(
            inner, doc=doc, model_seed=model_seed, arm=arm, check_sum=check_sum
        )

    def select_eviction(self, slots, context, step: int) -> int:
        live = torch.nonzero(slots.live).flatten().tolist()
        content = [content_of(self.doc, int(slots.written_at[j]), step) for j in live]
        k = super().select_eviction(slots, context, step)
        rec = self.log[-1]
        rec["live_content"] = content
        rec["victim_pos"] = live.index(k)
        return k


def slim(rec: dict) -> dict:
    keep = (
        "step",
        "written_at",
        "age",
        "rank",
        "rank_shift",
        "kind",
        "status",
        "margin",
        "psi",
        "live_ages",
        "live_content",
        "victim_pos",
        "victim_slot",
        "unmasked_slot",
        "flipped",
    )
    return {k: rec[k] for k in keep if k in rec}


# --------------------------------------------------------------------------- #
# §3: the arms
# --------------------------------------------------------------------------- #


def lru_policy(doc):
    pol = LRUPolicy()
    pol.doc_id = doc.doc_id
    return pol


def arm_makers(b2fits: dict, nbfits: dict, seed: int, S: int, names=None) -> dict:
    """§3 table: 22 arms, each a `doc -> policy` factory (a fresh policy per document)."""
    h, n = b2fits["heads"], nbfits["heads"]

    def probe(w, feature, arm):
        return lambda d: B2.ProbeArgminPolicy(
            w, feature, doc=d, model_seed=seed, arm=arm, S=S
        )

    def grace(g):
        arm = f"graceU{g}"
        return lambda d: GraceArgmin(
            B2.ProbeArgminPolicy(
                h[f"U@{GAMMA}"]["w"], "bilinear", doc=d, model_seed=seed, arm=arm, S=S
            ),
            g,
            arm=arm,
        )

    mk = {
        "psiC": probe(h[f"C@{GAMMA}"]["w"], "bilinear", "psiC"),
        "psiU": probe(h[f"U@{GAMMA}"]["w"], "bilinear", "psiU"),
        **{f"graceU{g}": grace(g) for g in GRACE_GS},
        "psiR1": probe(n[f"R1@{GAMMA}"]["w"], "bilinear", "psiR1"),
        "psiR1L1": probe(n[f"R1L1@{GAMMA}"]["w"], "bilinear", "psiR1L1"),
        "psiU+": probe(h[f"U+@{GAMMA}"]["w"], "bilinear", "psiU+"),
        "psiC+": probe(h[f"C+@{GAMMA}"]["w"], "bilinear", "psiC+"),
        "fifo": B2.fifo_policy,
        "lru": lru_policy,
        "ageU": probe(h[f"age_U@{GAMMA}"]["w"], "age", "ageU"),
        "ageC": probe(h[f"age_C@{GAMMA}"]["w"], "age", "ageC"),
        "ageR1": probe(n[f"age_R1@{GAMMA}"]["w"], "age", "ageR1"),
        "ageR1L1": probe(n[f"age_R1L1@{GAMMA}"]["w"], "age", "ageR1L1"),
        "kind": lambda d: B2.KindOracle(
            b2fits["class_means"][GAMMA], doc=d, model_seed=seed, tie="random", S=S
        ),
        "oracle": lambda d: B2.CheckedOracle(d, GAMMA, discounted_demand(d, GAMMA)),
    }
    for k in range(B2.N_RANDOM):
        mk[f"random{k}"] = lambda d, k=k: B2.random_policy(k, seed, d.doc_id)
    if set(mk) != set(ARMS):
        raise AssertionError(f"arm set {sorted(mk)} != PREREG {sorted(ARMS)}")
    order = ARMS if names is None else names
    return {a: mk[a] for a in order}


def run_one(model, doc, a: str, mk, *, seed, vmap, S, m, L) -> dict:
    pol = mk(doc)
    inst = LoggedNB(
        pol,
        doc=doc,
        model_seed=seed,
        arm=a,
        check_sum=a.startswith(("psi", "age", "grace")),
    )
    return B2.run_arm(model, doc, inst, vmap=vmap, S=S, m=m, L=L)


def run_unit(model, doc, makers: dict, *, seed, vmap, S, m, L) -> dict:
    """Every arm on one document: counts per bucket, in-loop residency, the
    slimmed per-eviction logs, §8 C7."""
    u: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "counts": {},
        "hits": {},
        "logs": {},
        "residency_ok": True,
        "sum_worst": 0.0,
    }
    for a, mk in makers.items():
        r = run_one(model, doc, a, mk, seed=seed, vmap=vmap, S=S, m=m, L=L)
        u["counts"][a] = {b: list(B2.counts(r["answers"], b)) for b in AN.BUCKETS}
        u["hits"][a] = [(int(g), bool(h)) for g, h in r["hits"]]
        u["logs"][a] = [slim(x) for x in r["log"]]
        u["residency_ok"] = bool(u["residency_ok"] and r["residency_ok"])
        u["sum_worst"] = max(u["sum_worst"], r["sum_worst"])
    return u


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

UNIT_FORMAT = 1


def code_sha256() -> str:
    """sha256 of this file (the measurement code): units written by other code
    are never resumed."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def unit_path(out_dir: Path, seed: int, doc_id: int) -> Path:
    return Path(out_dir) / f"seed{seed}" / "units" / f"{doc_id}.pt.gz"


def save_unit(u: dict, path: Path) -> None:
    """Atomic (§8 crash safety): temp file, fsync, `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(u, buf)
    tmp = path.parent / f"{path.name}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(gzip.compress(buf.getvalue(), compresslevel=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_unit(path: Path) -> dict:
    try:
        return torch.load(
            io.BytesIO(gzip.decompress(Path(path).read_bytes())), weights_only=True
        )
    except Exception as e:  # a unit that exists must be readable: exit 3
        raise ControlFailure(f"unreadable unit {path}: {e!r}") from e


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def b2_fits_dir() -> Path:
    """§1: `<main checkout>/.worktrees/b2-psi-probe/runs/b2-psi-probe-fit/phaseA`."""
    env = os.environ.get("RSR_NB_B2_FITS")
    if env:
        return Path(env)
    common = subprocess.run(
        [_git(), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return (
        Path(common).parent
        / ".worktrees"
        / "b2-psi-probe"
        / "runs"
        / "b2-psi-probe-fit"
        / "phaseA"
    )


def _git() -> str:
    return "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"


def load_b2_fits(seed: int, fits_dir: Path | None = None) -> dict:
    """§8 C3: the pinned phase-A file, or exit 3."""
    p = (fits_dir or b2_fits_dir()) / f"ckpt{LABEL}-seed{seed}.pt"
    got = LR.sha256(p)
    if got != B2_FIT_SHA256[seed]:
        raise ControlFailure(f"C3: B2 fit {p} sha256 {got} != {B2_FIT_SHA256[seed]}")
    return torch.load(p, weights_only=False)


# --------------------------------------------------------------------------- #
# the fit child (§5, §8 C4-C6, C8-C10)
# --------------------------------------------------------------------------- #


def c5_argmin_agreement(val_caps, w_new: Tensor, w_old: Tensor, *, m: int) -> dict:
    """§8 C5 (iii): argmin of the refit vs B2's stored C@0.9 over FIFO-resident
    slots at every FIT_VAL full-memory step (ties to the lowest index)."""
    agree = n = 0
    worst = 0.0
    for cap in val_caps:
        t, i = B2.row_index(cap, "C", m)
        keep = t >= m
        t, i = t[keep], i[keep]
        X = B2.design(cap, t, i, "bilinear")
        pn, po = X @ w_new.to(torch.float64), X @ w_old.to(torch.float64)
        worst = max(worst, float((pn - po).abs().max()))
        for tt in torch.unique(t).tolist():
            sel = t == tt
            a, b = pn[sel].tolist(), po[sel].tolist()
            ja = min(range(len(a)), key=lambda q: (a[q], q))
            jb = min(range(len(b)), key=lambda q: (b[q], q))
            agree += int(ja == jb)
            n += 1
    return {
        "agree": agree,
        "n": n,
        "rate": agree / n if n else None,
        "max_abs_psi_diff": worst,
    }


def fit_core(model, train_docs, val_docs, b2fits: dict, *, seed, vmap, S, m, L) -> dict:
    """The (d)/(d') fits and every fit-side control, IO-free (except the model).
    Raises ControlFailure (3) or B2.RidgeFailure (1)."""
    install_r1_targets()
    B2.require_range([d.doc_id for d in train_docs], "FIT_TRAIN")
    B2.require_range([d.doc_id for d in val_docs], "FIT_VAL")
    clos = B2.vocabulary_closure(list(train_docs) + list(val_docs), vmap)
    if not clos["ok"]:
        raise ControlFailure(f"C2 closure: {clos}")
    kw = {"vmap": vmap, "S": S, "m": m, "L": L}
    t0 = time.time()
    tcaps = [B2.capture_doc(model, d, **kw) for d in train_docs]
    vcaps = [B2.capture_doc(model, d, **kw) for d in val_docs]
    cap_s = time.time() - t0
    ident = {r: max(c.identity_worst[r] for c in tcaps + vcaps) for r in B2.RANKS}
    sumw = max(c.sum_worst for c in tcaps + vcaps)
    if max(ident.values()) > B2.IDENTITY_TOL or sumw > B2.SUM_TOL:
        raise ControlFailure(f"C4: identity {ident} sum {sumw}")
    trunc = {
        a: sum(r1_X(c.D[0], c.resident, lam, K_SHADOW)[1] for c in tcaps + vcaps)
        for a, lam in R1_LAMBDAS.items()
    }
    t0 = time.time()
    specs = [("C", GAMMA), ("R1", GAMMA), ("R1L1", GAMMA)]
    bil = B2.fit_heads(tcaps, vcaps, "C", "bilinear", specs, m=m)
    age = B2.fit_heads(tcaps, vcaps, "C", "age", [("R1", GAMMA), ("R1L1", GAMMA)], m=m)
    fit_s = time.time() - t0
    heads = {
        f"R1@{GAMMA}": bil[("R1", GAMMA)],
        f"R1L1@{GAMMA}": bil[("R1L1", GAMMA)],
        f"age_R1@{GAMMA}": age[("R1", GAMMA)],
        f"age_R1L1@{GAMMA}": age[("R1L1", GAMMA)],
    }
    for k, f in heads.items():
        if not bool(torch.isfinite(f["w"]).all()):
            raise FloatingPointError(f"non-finite w for {k}")
    # §8 C5: the (d) pipeline is B2's
    old, new = b2fits["heads"][f"C@{GAMMA}"], bil[("C", GAMMA)]
    sel = old["selected"]
    mo, mn = old["val_mse_demeaned"][sel], new["val_mse_demeaned"][sel]
    c5 = {
        "lam_b2": old["lam"],
        "lam_new": new["lam"],
        "same_lambda": old["lam"] == new["lam"],
        "val_mse_b2": mo,
        "val_mse_new": mn,
        "val_mse_rel": abs(mn - mo) / abs(mo),
        "w_max_abs_diff": float((new["w"] - old["w"]).abs().max()),
        "w_bit_identical": bool(torch.equal(new["w"], old["w"])),
        **c5_argmin_agreement(vcaps, new["w"], old["w"], m=m),
    }
    c5["ok"] = bool(
        c5["same_lambda"]
        and c5["val_mse_rel"] <= C5_MSE_REL
        and c5["rate"] is not None
        and c5["rate"] >= C5_ARGMIN_AGREE
    )
    if not c5["ok"]:
        raise ControlFailure(f"C5: {c5}")
    # §8 C9: no age in the (d)/(d') heads
    g = torch.randn(m, int(model.cfg.D), generator=torch.Generator().manual_seed(1))
    ctx = torch.randn(int(model.cfg.D), generator=torch.Generator().manual_seed(2))
    for k in (f"R1@{GAMMA}", f"R1L1@{GAMMA}"):
        pol = B2.ProbeArgminPolicy(
            heads[k]["w"], "bilinear", doc=val_docs[0], model_seed=seed, arm=k, S=S
        )
        if not B2.no_age_check(pol, g, ctx):
            raise ControlFailure(f"C9: age reaches {k}")
    nbfits = {"heads": heads}
    # FIT_VAL arms: ref for (d)/(d') (§5), C6 reproduction, C8 determinism, C9 grace
    mk = arm_makers(b2fits, nbfits, seed, S)
    t_arms = time.time()
    val_names = ("fifo", "psiC", "psiU", "ageR1", "ageR1L1")
    val_counts = {a: [] for a in val_names}
    ctrl = {"residency_ok": True, "sum_worst": 0.0}
    for d in val_docs:
        for a in val_names:
            r = run_one(model, d, a, mk[a], seed=seed, **kw)
            val_counts[a].append(B2.counts(r["answers"], "all"))
            ctrl["residency_ok"] &= r["residency_ok"]
            ctrl["sum_worst"] = max(ctrl["sum_worst"], r["sum_worst"])
    if not ctrl["residency_ok"] or ctrl["sum_worst"] > B2.SUM_TOL:
        raise ControlFailure(f"C7 on FIT_VAL: {ctrl}")
    acc = {
        a: B2.pooled_acc(torch.tensor(v).reshape(-1, 2)) for a, v in val_counts.items()
    }
    b2acc = b2fits["decisions"]["acc"][GAMMA]
    c6 = {
        a: {"new": acc[a], "b2": b2acc[a], "equal": acc[a] == b2acc[a]}
        for a in ("fifo", "psiC", "psiU")
    }
    if not all(v["equal"] for v in c6.values()):
        raise ControlFailure(f"C6: {c6}")
    ids = [d.doc_id for d in val_docs]
    refs = {
        "R1": B2.select_ref({"fifo": acc["fifo"], "age": acc["ageR1"]}, ids),
        "R1L1": B2.select_ref({"fifo": acc["fifo"], "age": acc["ageR1L1"]}, ids),
    }
    # §8 C8: determinism; §8 C9: grace off-switch reproduces psi-U
    c8 = True
    for d in val_docs[:N_DETERMINISM_DOCS]:
        for a in NEW_ARMS_C8:
            r1 = run_one(model, d, a, mk[a], seed=seed, **kw)
            r2 = run_one(model, d, a, mk[a], seed=seed, **kw)
            c8 &= (r1["victims"], r1["answers"]) == (r2["victims"], r2["answers"])
    if not c8:
        raise ControlFailure("C8: determinism")
    off = {"grace_g0_equals_psiU": True, "scores_equal": True}
    wU = b2fits["heads"][f"U@{GAMMA}"]["w"]
    for d in val_docs[:N_DETERMINISM_DOCS]:
        g0 = lambda doc: GraceArgmin(  # noqa: E731
            B2.ProbeArgminPolicy(wU, "bilinear", doc=doc, model_seed=seed, arm="g0", S=S),
            0,
            arm="graceU0",
        )
        ra = run_one(model, d, "graceU0", g0, seed=seed, **kw)
        rb = run_one(model, d, "psiU", mk["psiU"], seed=seed, **kw)
        off["grace_g0_equals_psiU"] &= ra["victims"] == rb["victims"]
        off["scores_equal"] &= [x["psi"] for x in ra["log"]] == [
            x["psi"] for x in rb["log"]
        ]
    if not all(off.values()):
        raise ControlFailure(f"C9 grace off-switch: {off}")
    return {
        "heads": heads,
        "refs": refs,
        "fit_val_acc": acc,
        "controls": {
            "C4": {"identity_worst": ident, "sum_worst": sumw},
            "C5": c5,
            "C6": c6,
            "C7_fit_val": ctrl,
            "C8": c8,
            "C9_grace_off": off,
            "closure": clos,
        },
        "k_truncated_cells": trunc,
        "n": {
            "N_F": len(train_docs),
            "N_V": len(val_docs),
            "n_C_rows": bil[("R1", GAMMA)]["n_train_rows"],
        },
        "seconds": {"capture": cap_s, "fit": fit_s, "val_arms": time.time() - t_arms},
        "threads": torch.get_num_threads(),
    }


def fit_key(seed: int) -> dict:
    return {
        "format": UNIT_FORMAT,
        "seed": seed,
        "prereg": PREREG_COMMIT,
        "b2_fit_sha256": B2_FIT_SHA256[seed],
        "n_f": N_F,
    }


def fit_child(seed: int, source: Path, out: Path) -> int:  # pragma: no cover
    torch.set_num_threads(int(os.environ.get("RSR_NB_THREADS", "4")))
    wall = time.time()
    path = out / f"fit-seed{seed}.pt"
    if path.exists() and torch.load(path, weights_only=False).get("key") == fit_key(seed):
        print(f"fit seed {seed}: resumed from {path}", file=sys.stderr, flush=True)
        return 0
    try:
        model, vmap = B2.load_checked(source, LABEL, seed)
        b2fits = load_b2_fits(seed)
        c3 = B2.control_c3(model, seed, LABEL, vmap, source)
        if not c3["ok"]:
            raise ControlFailure(f"C3 (B2 §11.3): {json.dumps(c3)[:400]}")
        p = B2.p_of(int(model.cfg.D))
        if B2.n_f(p) != N_F:
            raise ControlFailure(f"N_F {B2.n_f(p)} != {N_F}")
        res = fit_core(
            model,
            B2.docs_in(seed, "FIT_TRAIN", N_F),
            B2.docs_in(seed, "FIT_VAL"),
            b2fits,
            seed=seed,
            vmap=vmap,
            S=B2.S_STEPS,
            m=B2.M_STEPS,
            L=B2.L_TOKENS,
        )
    except (ControlFailure, B2.ControlFailure, B2.RangeError) as e:
        print(f"CONTROL FAILED (exit 3): {e}", file=sys.stderr, flush=True)
        return 3
    except B2.RidgeFailure as e:
        print(f"RIDGE FAILURE (exit 1): {e}", file=sys.stderr, flush=True)
        return 1
    res["controls"]["C3_b2"] = c3
    res["key"] = fit_key(seed)
    res["seconds"]["child_wall"] = time.time() - wall
    res["peak_rss_gb"] = B2._peak_rss_gb()
    atomic_torch_save(res, path)
    return 0


# --------------------------------------------------------------------------- #
# the eval child (§3; units atomic, resumable)
# --------------------------------------------------------------------------- #


def unit_key(seed: int, b2sha: str, nbsha: str) -> dict:
    return {
        "format": UNIT_FORMAT,
        "seed": seed,
        "code_sha256": code_sha256(),
        "b2_fit_sha256": b2sha,
        "nb_fit_sha256": nbsha,
        "arms": list(ARMS),
        "prereg": PREREG_COMMIT,
    }


def eval_shard(
    model, docs, makers, *, seed, out: Path, key: dict, vmap, S, m, L, progress=None
) -> dict:
    n_run = n_res = 0
    for j, d in enumerate(docs):
        path = unit_path(out, seed, d.doc_id)
        want = dict(key, doc_id=d.doc_id)
        if path.exists():
            u = load_unit(path)
            if u.get("key") != want:
                raise ControlFailure(f"{path}: written under {u.get('key')} != {want}")
            n_res += 1
            sec = None
        else:
            t0 = time.perf_counter()
            u = run_unit(model, d, makers, seed=seed, vmap=vmap, S=S, m=m, L=L)
            u["key"] = want
            if not u["residency_ok"] or u["sum_worst"] > B2.SUM_TOL:
                raise ControlFailure(
                    f"C7 document {d.doc_id}: residency_ok={u['residency_ok']} "
                    f"sum_worst={u['sum_worst']}"
                )
            save_unit(u, path)
            sec = time.perf_counter() - t0
            n_run += 1
        if progress:
            progress(j + 1, len(docs), sec)
        del u
    return {"n_run": n_run, "n_resumed": n_res}


def eval_child(
    seed: int, shard: int, n_shards: int, source: Path, out: Path
) -> int:  # pragma: no cover
    torch.set_num_threads(int(os.environ.get("RSR_NB_THREADS", "1")))
    tag = f"seed{seed}-shard{shard}"
    try:
        model, vmap = B2.load_checked(source, LABEL, seed)
        b2fits = load_b2_fits(seed)
        npath = out / f"fit-seed{seed}.pt"
        nbfits = torch.load(npath, weights_only=False)
        if nbfits.get("key") != fit_key(seed):
            raise ControlFailure(f"{npath} key {nbfits.get('key')}")
        lo, hi = shard_bounds(shard, n_shards)
        docs = eval_docs(seed, lo, hi)
        clos = B2.vocabulary_closure(docs, vmap)
        if not clos["ok"]:
            raise ControlFailure(f"C2 closure {clos}")
        mk = arm_makers(b2fits, nbfits, seed, B2.S_STEPS)
        key = unit_key(seed, B2_FIT_SHA256[seed], LR.sha256(npath))

        def progress(j, n, sec):
            what = "resumed" if sec is None else f"{sec:.2f}s"
            print(
                f"{time.strftime('%H:%M:%S')} {tag} {j}/{n} {what} "
                f"peak_rss_gb={B2._peak_rss_gb():.2f}",
                file=sys.stderr,
                flush=True,
            )

        r = eval_shard(
            model,
            docs,
            mk,
            seed=seed,
            out=out,
            key=key,
            vmap=vmap,
            S=B2.S_STEPS,
            m=B2.M_STEPS,
            L=B2.L_TOKENS,
            progress=progress,
        )
    except (ControlFailure, B2.ControlFailure, B2.RangeError) as e:
        print(f"CONTROL FAILED (exit 3): {e}", file=sys.stderr, flush=True)
        return 3
    print(f"{tag} done {r}", file=sys.stderr, flush=True)
    return 0


# --------------------------------------------------------------------------- #
# the parent
# --------------------------------------------------------------------------- #


def _children(
    argvs: dict, root: Path, parallel: int, led, env_extra: dict
) -> dict:  # pragma: no cover
    rcs, running = {}, {}
    (root / "logs").mkdir(parents=True, exist_ok=True)
    jobs = list(argvs)
    env = {**os.environ, **env_extra}
    while jobs or running:
        while jobs and len(running) < parallel:
            j = jobs.pop(0)
            log = open(root / "logs" / f"{j}.log", "a")  # noqa: SIM115
            running[j] = subprocess.Popen(
                argvs[j], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env
            )
        for j, proc in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                rcs[j] = rc
                led.command(argvs[j], exit_code=rc)
                del running[j]
        time.sleep(2)
    return rcs


def _argv(cmd: str, source: Path, out: Path, *extra) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        cmd,
        "--source",
        str(source),
        "--out",
        str(out),
        *extra,
    ]


def aggregate(out: Path, led) -> dict:
    """Every §6 readout and the §7 table from the units and the fit files."""
    per_seed, fits_rows = {}, {}
    for s in SEEDS:
        b2fits = load_b2_fits(s)
        nbfits = torch.load(out / f"fit-seed{s}.pt", weights_only=False)
        docs = eval_docs(s, *EVAL_NB)
        units = (load_unit(unit_path(out, s, d.doc_id)) for d in docs)
        ag = AN.aggregate_seed(units, docs, ARMS, S=B2.S_STEPS, m=B2.M_STEPS)
        refs = {
            "U": b2fits["decisions"]["ref"][f"U@{GAMMA}"],
            "C": b2fits["decisions"]["ref"][f"C@{GAMMA}"],
            **nbfits["refs"],
        }
        delta = b2fits["decisions"]["delta"][GAMMA]
        oc = AN.seed_outcomes(ag["counts"], refs, delta, s)
        res = {
            b: {a: B2.pooled_acc(ag["residency"][a][b]) for a in ARMS} for b in AN.BUCKETS
        }
        per_seed[s] = {
            **oc,
            "readouts": ag["readouts"],
            "residency": res,
            "n_docs": len(ag["doc_ids"]),
        }
        fits_rows[s] = {
            "lam": {k: v["lam"] for k, v in nbfits["heads"].items()},
            "resid": {
                k: v["path"][v["selected"]]["resid"] for k, v in nbfits["heads"].items()
            },
            "refs": nbfits["refs"],
            "fit_val_acc": nbfits["fit_val_acc"],
            "controls": nbfits["controls"],
            "k_truncated_cells": nbfits["k_truncated_cells"],
            "n": nbfits["n"],
            "seconds": nbfits["seconds"],
            "threads": nbfits["threads"],
            "peak_rss_gb": nbfits.get("peak_rss_gb"),
        }
    dec = AN.decisions(per_seed)
    return {"per_seed": per_seed, "fits": fits_rows, "decisions": dec}


def write_rows(led, agg: dict) -> None:
    how = "run.py::aggregate over runs/newcomer-bakeoff/seed{s}/units + fit-seed{s}.pt"
    for s, f in agg["fits"].items():
        led.note(f"fit.seed{s}", f, how=f"runs/{RUN_ID}/fit-seed{s}.pt")
    for s, p in agg["per_seed"].items():
        led.note(f"seed{s}.n_docs", p["n_docs"], how=how)
        led.note(f"seed{s}.delta", p["delta"], how="B2 phase A decisions.delta[0.9]")
        led.note(f"seed{s}.refs", p["refs"], how="B2 phase A + fit-seed{s}.pt refs")
        led.note(f"seed{s}.acc", p["acc"], how=how)
        led.note(f"seed{s}.residency", p["residency"], how=how)
        led.note(f"seed{s}.outcome", p["outcome"], how=how + " (B2 §9.5 + A1.13)")
        led.note(f"seed{s}.contrasts", p["contrasts"], how=how)
        led.note(f"seed{s}.contrasts_by_bucket", p["contrasts_by_bucket"], how=how)
        led.note(f"seed{s}.pi", p["pi"], how=how)
        for a, r in p["readouts"].items():
            led.note(f"seed{s}.readouts.{a}", r, how=how)
    for k, v in agg["decisions"].items():
        led.note(f"decision.{k}", v, how="analysis.py::decisions (PREREG §7)")


def t0_shasum() -> dict:
    """§8 C1: the literal `shasum -a 256 -c MANIFEST.sha256` from the main checkout."""
    common = subprocess.run(
        [_git(), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    main = Path(common).parent
    man = Path.home() / "rsr-substrate" / "2026-09-27" / "MANIFEST.sha256"
    p = subprocess.run(
        ["shasum", "-a", "256", "-c", str(man)], cwd=main, capture_output=True, text=True
    )
    ok_lines = sum(1 for ln in p.stdout.splitlines() if ln.endswith(": OK"))
    return {
        "rc": p.returncode,
        "n_ok": ok_lines,
        "ok": p.returncode == 0 and ok_lines == 435,
    }


def run_all(
    source: Path, runs_root: Path, parallel: int, only_aggregate=False
) -> Exit:  # pragma: no cover
    from ledger import Ledger

    probs = range_problems()
    if probs:
        return did_not_run("; ".join(probs))
    try:
        for s in SEEDS:
            load_b2_fits(s)
    except (ControlFailure, OSError) as e:
        return did_not_run(f"C3: {e}")
    root = runs_root / RUN_ID
    prev = root / "manifest.json"
    if prev.exists():
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        (root / f"manifest.before-{stamp}.json").write_bytes(prev.read_bytes())
    led = Ledger(RUN_ID, question=QUESTION, runs_root=runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(
        {
            "run_id": RUN_ID,
            "prereg": PREREG,
            "prereg_commit": PREREG_COMMIT,
            "eval_nb": EVAL_NB,
            "N_E": N_E,
            "N_F": N_F,
            "seeds": SEEDS,
            "label": LABEL,
            "gamma": GAMMA,
            "grace_g": GRACE_GS,
            "K": K_SHADOW,
            "lambda_shadow": LAMBDA_SHADOW,
            "r1_lambdas": R1_LAMBDAS,
            "arms": ARMS,
            "b2_fit_sha256": B2_FIT_SHA256,
            "b2_fits_dir": str(b2_fits_dir()),
            "source": str(source),
            "parallel": parallel,
            "code_sha256": code_sha256(),
            "analysis_sha256": hashlib.sha256(
                (ROOT / "experiments/newcomer-bakeoff/analysis.py").read_bytes()
            ).hexdigest(),
            "only_aggregate": only_aggregate,
            "expected": EXPECTED,
            "falsifier": FALSIFIER,
            "threads": {"fit": 4, "eval": 1},
        }
    )
    t0 = B2.t0_checked()
    led.note("T0.start", {"in_process": t0, "shasum": t0_shasum()}, how="run.py t0")
    if not t0["ok"]:
        led.status("failed")
        led.write()
        return did_not_run(f"T0 {t0}")
    rcs: dict = {}
    if not only_aggregate:
        fit = {
            f"fit-seed{s}": _argv("fit-child", source, root, "--seed", str(s))
            for s in SEEDS
        }
        rcs.update(
            _children(
                fit,
                root,
                max(1, min(len(SEEDS), parallel // 4)),
                led,
                {"RSR_NB_THREADS": "4"},
            )
        )
        if all(v == 0 for v in rcs.values()):
            n_sh = max(1, parallel // len(SEEDS))
            ev = {
                f"eval-seed{s}-shard{j}": _argv(
                    "eval-child",
                    source,
                    root,
                    "--seed",
                    str(s),
                    "--shard",
                    str(j),
                    "--n-shards",
                    str(n_sh),
                )
                for s in SEEDS
                for j in range(n_sh)
            }
            rcs.update(_children(ev, root, parallel, led, {"RSR_NB_THREADS": "1"}))
    t0e = B2.t0_checked()
    led.note("T0.end", {"in_process": t0e, "shasum": t0_shasum()}, how="run.py t0")
    rc = B2.parent_rc(rcs, t0_ok=t0e["ok"])
    if rc != Exit.OK:
        led.run_meta(seeds_actually_run=[])
        led.status("failed")
        led.verdict(falsifier=FALSIFIER, outcome="inconclusive", detail=f"children {rcs}")
        led.write()
        return rc
    try:
        agg = aggregate(root, led)
    except (ControlFailure, B2.ControlFailure) as e:
        led.note("aggregate.error", str(e), how="run.py::aggregate")
        led.status("failed")
        led.write()
        return did_not_run(f"aggregate: {e}")
    write_rows(led, agg)
    d = agg["decisions"]
    led.run_meta(seeds_actually_run=SEEDS)
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=(
            f"DQ1 {d['DQ1']['class']}; DQ2 {d['DQ2']['class']}; DQ3 {d['DQ3']['class']}; "
            f"replication {d['replication']['class']} (offline; no build recommendation)"
        ),
    )
    led.command(
        [sys.executable, EXPERIMENT, "aggregate" if only_aggregate else "run"],
        exit_code=0,
        note="the parent",
    )
    led.write()
    return Exit.OK


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "cmd", choices=["check", "run", "aggregate", "fit-child", "eval-child"]
    )
    ap.add_argument("--seed", type=int)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--n-shards", type=int)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--parallel", type=int, default=12)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    a, rest = ap.parse_known_args(argv)
    if rest:
        return did_not_run(f"unknown arguments {rest}")
    if a.cmd == "check":
        probs = range_problems()
        print(json.dumps({"problems": probs}))
        return did_not_run("; ".join(probs)) if probs else Exit.OK
    source = a.source or LR.default_source()
    if a.cmd == "fit-child":
        return status(fit_child(a.seed, source, a.out))
    if a.cmd == "eval-child":
        return status(eval_child(a.seed, a.shard, a.n_shards, source, a.out))
    return status(
        run_all(source, a.runs_root, a.parallel, only_aggregate=a.cmd == "aggregate")
    )


if __name__ == "__main__":
    run_main(main)
