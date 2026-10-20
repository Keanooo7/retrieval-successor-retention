"""Item 1b (post-fails-controls PREREG §4): linear probes of sentence kind and assert
status, and class-mean rank caps, on U.

Pre-registration: `experiments/post-fails-controls/PREREG.md`, committed alone at
9f22d87. `main(argv)` measures and writes `runs/post-fails-controls-1b/{manifest.json,
ledger.json}`; `finalize(argv)` renders `RESULTS.md` and `claims.json` from the
ledger.

Trains no policy. Fits linear probes (float64, full-batch L-BFGS) on precomputed,
frozen FIFO-world gestalts; no forward pass, no gradient into any model; never calls
`rsr.constants.record()`.

* **Split (§4).** U-val is spent and never read. U-train = offline-psi's `u_split`
  train positions; in U-train order position ``j``: P-test = ``j % 5 == 0``, P-fit the
  rest. Probes fit on P-fit, scored on P-test.
* **Probes.** (1) kind from ``s_i`` (multinomial, 3 classes, one cell per sentence);
  (2) status pending/answered from ``s_i``, from ``c_t`` and from ``[s~; c~; s~ (x) c~]``
  (binary; cells ``t in [16, 48)``, ``i in [t-16, t)``, assert, ``q(i) != t``). Features
  standardised per coordinate on P-fit; class-balanced cross-entropy (each class's mean
  loss, averaged over classes); L2 ``lambda = 1e-4`` on weights, not the bias; L-BFGS,
  strong-Wolfe, at most 1000 iterations. ``lambda in {1e-5, 1e-3}`` descriptive only.
* **Caps.** ``ko_D``: argmin ``E[G | kind]``; ``st_D``: argmin ``E[G | class(i, t)]``
  (assert-pending, -querying, -answered, query, filler); ``G = discounted_returns(Delta,
  0.9)``; means pooled over B0's rows, row-weighted; B0's tie rule. Primary: means on
  P-fit, replayed on P-test. Secondary: in-sample on all U, beside B0's SEEN ``ko``.

Exit codes (§0): 0 ran, every control passed; 1 a defect after the controls passed; 3
did not run or a control failed (numbers written, NOT READ).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402

EXPERIMENT = "experiments/post-fails-controls/item_1b.py"
ITEM = "1b"
RUN_ID = "post-fails-controls-1b"
PREREG = "experiments/post-fails-controls/PREREG.md"
PREREG_COMMIT = "9f22d87"
OUT_DEFAULT = ROOT / "runs" / RUN_ID
RUN_PY = "experiments/post-fails-controls/run.py"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: Item 1c's module: the shared loads (diagnostic A, offline-psi, run 1, B0,
#: lookahead-room) and the global-controls helper.
IC = _load("_pfc_item_1c", "experiments/post-fails-controls/item_1c.py")
CV, OP, R1, B0, LR = IC.CV, IC.OP, IC.R1, IC.B0, IC.LR

# --------------------------------------------------------------------------- #
# 🔒 PREREG §4, transcribed.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
M = 16
S = 48
GAMMA = 0.9
P_TEST_MOD = 5
LAMBDA = 1e-4
LAMBDA_DESCRIPTIVE = (1e-5, 1e-3)
MAX_ITER = 1000
N_BOOT = 2000
KINDS = ("assert", "query", "filler")
STATUS = ("pending", "answered")
STATUS_FEATURES = ("s", "c", "scx")  # s only, c only, [s; c; s (x) c]
CLASSES5 = ("assert_pending", "assert_querying", "assert_answered", "query", "filler")
THRESH = {"kind": 0.95, "status": 0.80}
LABELS = ("DECODABLE", "NOT-DECODABLE", "UNRESOLVED")
CF_LEDGER = Path(OP.REFS.cf_ledger)
CF_CAP_KEY = "B.ckpt3000.cf.g09.cap.point"
B0_LEDGER = Path(R1._R) / ".worktrees/b0-ceilings/runs/b0-ceilings/ledger.json"
B0_N_KEY = "B.ckpt3000.g09.E_G.kind.{k}.n"
B0_KO_KEY = "B.ckpt3000.g09.cap.ko.{f}"

QUESTION = (
    "Is sentence kind, and an assert's pending/answered status, linearly decodable from "
    "what psi_hat sees, and where do E[G | kind] and E[G | status] rank (PREREG §4)?"
)
FALSIFIER = (
    "NONE as a spec falsifier: Tier D diagnostic. Registered reading (§4): kind probe "
    "DECODABLE iff CI lo >= 0.95 on all 3 seeds, NOT-DECODABLE iff CI hi < 0.95 on any "
    "seed, else UNRESOLVED; the [s; c; s(x)c] status probe likewise at 0.80. Caps "
    "reported with CIs, no label. Gates nothing."
)
EXPECTED = (
    "PREREG §10: kind DECODABLE; status from [s; c; s(x)c] UNRESOLVED or "
    "NOT-DECODABLE; st_D cap above ko_D cap on seed 1."
)


# --------------------------------------------------------------------------- #
# split, cells, labels (generator only)
# --------------------------------------------------------------------------- #


def p_split(train_positions: list[int]) -> tuple[list[int], list[int]]:
    """§4: in U-train order, position ``j``: P-test iff ``j % 5 == 0``."""
    p_test = [p for j, p in enumerate(train_positions) if j % P_TEST_MOD == 0]
    p_fit = [p for j, p in enumerate(train_positions) if j % P_TEST_MOD != 0]
    return p_fit, p_test


def q_of(doc) -> dict[int, int]:
    return {a: q for a, q in doc.pairs}


def kind_cells(docs: list, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """One cell per sentence: ``(pos, i)`` [n, 2] and its kind label index."""
    idx, y = [], []
    for p in positions:
        for i, s_ in enumerate(docs[p].sentences):
            idx.append((p, i))
            y.append(KINDS.index(s_.kind))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def status_cells(
    docs: list, positions: list[int], m: int = M, s: int = S
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(pos, t, i)`` for ``t in [m, s)``, ``i in [t-m, t)`` an assert with ``q(i) !=
    t``; label 0 pending (``q > t``), 1 answered (``q < t``). Also the number of
    excluded ``q(i) == t`` (querying) cells."""
    idx, y, n_q = [], [], 0
    for p in positions:
        doc = docs[p]
        qq = q_of(doc)
        kinds = [x.kind for x in doc.sentences]
        for t in range(m, s):
            for i in range(t - m, t):
                if kinds[i] != "assert":
                    continue
                q = qq[i]
                if q == t:
                    n_q += 1
                    continue
                idx.append((p, t, i))
                y.append(0 if q > t else 1)
    return torch.tensor(idx, dtype=torch.long), torch.tensor(y, dtype=torch.long), n_q


def class5(doc, i: int, t: int) -> str:
    k = doc.sentences[i].kind
    if k != "assert":
        return k
    q = q_of(doc)[i]
    return (
        "assert_pending"
        if q > t
        else ("assert_querying" if q == t else "assert_answered")
    )


# --------------------------------------------------------------------------- #
# the probes
# --------------------------------------------------------------------------- #


def standardiser(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(0)
    sd = X.std(0, unbiased=False)
    sd = torch.where(sd > 0, sd, torch.ones_like(sd))
    return mu, sd


def balanced_ce(logits: torch.Tensor, y: torch.Tensor, n_cls: int) -> torch.Tensor:
    """Class-balanced cross-entropy: each class's mean loss, averaged over classes
    (multinomial ``logits [n, C]``)."""
    lp = torch.log_softmax(logits, dim=1)
    nll = -lp.gather(1, y.unsqueeze(1)).squeeze(1)
    tot = logits.new_zeros(())
    for c in range(n_cls):
        sel = y == c
        if bool(sel.any()):
            tot = tot + nll[sel].mean()
    return tot / n_cls


class Probe:
    """A linear (or, ``bilinear``, the ``[s; c; s (x) c]``) probe. Multinomial over
    ``n_cls`` classes when ``n_cls > 2``; binary logistic (one logit, class 1 vs 0)
    when ``n_cls == 2``. Parameters float64. ``feats`` = list of ``[n, d]`` blocks."""

    def __init__(self, dims: list[int], n_cls: int, bilinear: bool) -> None:
        self.n_cls = n_cls
        self.out = n_cls if n_cls > 2 else 1
        self.bilinear = bilinear
        f64 = dict(dtype=torch.float64)
        self.W = torch.zeros(self.out, sum(dims), **f64, requires_grad=True)
        self.b = torch.zeros(self.out, **f64, requires_grad=True)
        self.A = (
            torch.zeros(self.out, dims[0], dims[1], **f64, requires_grad=True)
            if bilinear
            else None
        )

    def params(self) -> list[torch.Tensor]:
        return [p for p in (self.W, self.b, self.A) if p is not None]

    def logits(self, feats: list[torch.Tensor]) -> torch.Tensor:
        X = torch.cat(feats, dim=1)
        z = X @ self.W.T + self.b
        if self.bilinear:
            s_, c_ = feats[0], feats[1]
            # s~^T A c~ per output, without materialising the outer product
            z = z + torch.einsum("nd,ode,ne->no", s_, self.A, c_)
        return z

    def loss(self, feats, y, lam: float) -> torch.Tensor:
        z = self.logits(feats)
        z2 = torch.cat([torch.zeros_like(z), z], dim=1) if self.out == 1 else z
        pen = self.W.pow(2).sum()
        if self.A is not None:
            pen = pen + self.A.pow(2).sum()
        return balanced_ce(z2, y, self.n_cls) + lam * pen

    def predict(self, feats) -> torch.Tensor:
        with torch.no_grad():
            z = self.logits(feats)
            return (z[:, 0] > 0).long() if self.out == 1 else z.argmax(1)


def fit_probe(probe: Probe, feats, y, lam: float, max_iter: int = MAX_ITER) -> dict:
    """Full-batch L-BFGS (strong Wolfe), at most ``max_iter`` iterations. Converged iff
    it stopped by its tolerance (fewer iterations and evaluations than the caps) with
    a finite loss and gradient (B-C6)."""
    opt = torch.optim.LBFGS(
        probe.params(), lr=1, max_iter=max_iter, line_search_fn="strong_wolfe"
    )
    t0 = time.perf_counter()

    def closure():
        opt.zero_grad()
        L = probe.loss(feats, y, lam)
        L.backward()
        return L

    opt.step(closure)
    st = opt.state[opt._params[0]]
    with torch.enable_grad():
        opt.zero_grad()
        L = probe.loss(feats, y, lam)
        L.backward()
    g = torch.cat([p.grad.reshape(-1) for p in probe.params()])
    n_iter, n_eval = int(st.get("n_iter", 0)), int(st.get("func_evals", 0))
    max_eval = opt.param_groups[0]["max_eval"]
    finite = bool(torch.isfinite(L)) and bool(torch.isfinite(g).all())
    for p in probe.params():
        finite = finite and bool(torch.isfinite(p).all())
    return {
        "loss": float(L.detach()),
        "grad_max_abs": float(g.abs().max()),
        "n_iter": n_iter,
        "func_evals": n_eval,
        "max_iter": max_iter,
        "max_eval": max_eval,
        "converged": finite and n_iter < max_iter and n_eval < max_eval,
        "finite": finite,
        "seconds": time.perf_counter() - t0,
    }


def per_doc_recall_counts(
    pred: torch.Tensor, y: torch.Tensor, doc_of: torch.Tensor, docs: list[int], n_cls: int
) -> torch.Tensor:
    """``[n_docs, n_cls, 2]`` = (cells of class c, correct of class c) per document."""
    out = torch.zeros(len(docs), n_cls, 2, dtype=torch.float64)
    pos = {p: j for j, p in enumerate(docs)}
    row = torch.tensor([pos[int(p)] for p in doc_of.tolist()], dtype=torch.long)
    for c in range(n_cls):
        sel = y == c
        out[:, c, 0].index_add_(
            0, row[sel], torch.ones(int(sel.sum()), dtype=torch.float64)
        )
        ok = (pred == y) & sel
        out[:, c, 1].index_add_(
            0, row[ok], torch.ones(int(ok.sum()), dtype=torch.float64)
        )
    return out


def balanced_accuracy(counts: torch.Tensor, W: torch.Tensor) -> dict:
    """Mean over classes of per-class recall; CI from the per-document bootstrap
    (recall per replicate = weighted correct / weighted cells); a replicate with an
    empty class is dropped and counted."""
    tot = counts.sum(0)
    if bool((tot[:, 0] == 0).any()):
        return {"point": None, "lo": None, "hi": None, "dropped": None, "recall": None}
    point = float((tot[:, 1] / tot[:, 0]).mean())
    n = W @ counts[:, :, 0]  # [B, C]
    k = W @ counts[:, :, 1]
    ok = (n > 0).all(1)
    rep = (k[ok] / n[ok]).mean(1)
    lo, hi = (float(torch.quantile(rep, 0.025)), float(torch.quantile(rep, 0.975)))
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "sd": float(rep.std()) if rep.numel() > 1 else None,
        "dropped": int((~ok).sum()),
        "recall": (tot[:, 1] / tot[:, 0]).tolist(),
        "n_cells": tot[:, 0].tolist(),
    }


def probe_label(cis: dict[int, dict], thresh: float, converged: dict[int, bool]) -> str:
    """§4 (tightened to the CI): DECODABLE iff lo >= thresh on all 3 seeds;
    NOT-DECODABLE iff hi < thresh on any seed; else UNRESOLVED. A probe that did not
    converge (B-C6) or a missing seed is UNRESOLVED."""
    if any(not converged.get(s, False) for s in SEEDS):
        return "UNRESOLVED"
    vals = [cis.get(s) for s in SEEDS]
    if any(
        v is None or not _finite(v.get("lo")) or not _finite(v.get("hi")) for v in vals
    ):
        return "UNRESOLVED"
    if any(v["hi"] < thresh for v in vals):
        return "NOT-DECODABLE"
    if all(v["lo"] >= thresh for v in vals):
        return "DECODABLE"
    return "UNRESOLVED"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #


def class_means(docs: list, Gs: dict, positions: list[int], m: int = M) -> dict:
    """``E[G | kind]`` and ``E[G | class5]`` pooled over B0's rows (``t in [m, S)``,
    every ``i < t``) of the documents at ``positions``, row-weighted."""
    sk = {k: [0.0, 0] for k in KINDS}
    sc = {c: [0.0, 0] for c in CLASSES5}
    for p in positions:
        doc = docs[p]
        G = Gs[doc.doc_id]
        rows = B0.row_mask(G.shape[0], m).nonzero().tolist()
        Gl = G.tolist()
        for t, i in rows:
            g = float(Gl[t][i])
            k = doc.sentences[i].kind
            sk[k][0] += g
            sk[k][1] += 1
            c = class5(doc, i, t)
            sc[c][0] += g
            sc[c][1] += 1
    return {
        "kind": {k: (v[0] / v[1] if v[1] else None) for k, v in sk.items()},
        "n_kind": {k: v[1] for k, v in sk.items()},
        "class": {c: (v[0] / v[1] if v[1] else None) for c, v in sc.items()},
        "n_class": {c: v[1] for c, v in sc.items()},
    }


def rule_policy(doc, mu: dict, which: str, seed: int):
    """``ko_D`` / ``st_D`` as B0's `ScorePolicy` (argmin, B0 A1.10's random tie)."""
    if which == "ko":

        def score(i, t):
            return mu["kind"][doc.sentences[i].kind]

    elif which == "st":

        def score(i, t):
            return mu["class"][class5(doc, i, t)]

    else:
        raise ValueError(which)
    return B0.ScorePolicy(score, seed=seed, doc_id=doc.doc_id, tie="random")


def caps(docs: list, positions: list[int], mu: dict, seed: int) -> dict:
    """Model-free residency of ko_D and st_D on the documents at ``positions``, cap
    against FIFO and FactFiller, paired CI `R1.boot_weights(n, seed, 2000)`."""
    sub = [docs[p] for p in positions]
    rr = OP.reference_rates(sub, seed)
    W = R1.boot_weights(len(sub), seed, N_BOOT)
    out = {}
    for which in ("ko", "st"):
        rates = [B0.rate(B0.pairs_of(d, rule_policy(d, mu, which, seed))) for d in sub]
        out[which] = B0.boot_rule(W, rates, rr["fifo"], rr["factfiller"])
    out["fifo_hit"] = B0.pooled(rr["fifo"])
    out["factfiller_hit"] = B0.pooled(rr["factfiller"])
    return out


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #


def gather_feats(inp: dict, idx: torch.Tensor, which: str) -> list[torch.Tensor]:
    S64 = inp["S64"]
    C64 = inp["C64"]
    if idx.shape[1] == 2:  # kind cells (pos, i)
        return [S64[idx[:, 0], idx[:, 1]]]
    s_ = S64[idx[:, 0], idx[:, 2]]
    c_ = C64[idx[:, 0], idx[:, 1]]
    return {"s": [s_], "c": [c_], "scx": [s_, c_]}[which]


def std_feats(fit: list[torch.Tensor], test: list[torch.Tensor]):
    out_f, out_t, stats = [], [], []
    for a, b in zip(fit, test, strict=True):
        mu, sd = standardiser(a)
        out_f.append((a - mu) / sd)
        out_t.append((b - mu) / sd)
        stats.append({"n_zero_sd": int((a.std(0, unbiased=False) == 0).sum())})
    return out_f, out_t, stats


def run_probe(
    inp, idx_fit, y_fit, idx_test, y_test, which, n_cls, lam, p_test, seed, *, n_boot
) -> dict:
    ff, ft, st = std_feats(
        gather_feats(inp, idx_fit, which), gather_feats(inp, idx_test, which)
    )
    dims = [x.shape[1] for x in ff]
    probe = Probe(dims, n_cls, bilinear=(which == "scx"))
    fit = fit_probe(probe, ff, y_fit, lam)
    pred = probe.predict(ft)
    counts = per_doc_recall_counts(pred, y_test, idx_test[:, 0], p_test, n_cls)
    W = R1.boot_weights(len(p_test), seed, n_boot)
    ba = balanced_accuracy(counts, W)
    n_features = sum(dims) + (dims[0] * dims[1] if which == "scx" else 0)
    return {
        "fit": fit,
        "bal_acc": ba,
        "n_features": n_features,
        "std": st,
        "n_fit_cells": int(y_fit.numel()),
        "n_test_cells": int(y_test.numel()),
    }


def measure_seed(
    seed: int,
    *,
    limit: int | None = None,
    n_boot: int = N_BOOT,
    lambdas_desc=LAMBDA_DESCRIPTIVE,
    max_docs_fit: int | None = None,
) -> dict:
    t0 = time.perf_counter()
    inp = CV.load_inputs(seed)
    docs = inp["docs"]
    ids = inp["ids"]
    inp["S64"] = inp["S"].to(torch.float64)
    inp["C64"] = inp["C"].to(torch.float64)
    train, val = inp["train"], inp["val"]
    p_fit, p_test = p_split(train)
    if limit is not None:
        p_fit = [p for p in p_fit if p < limit]
        p_test = [p for p in p_test if p < limit]
    # B-C2: split disjointness by doc id, and no U-val document in any probe tensor
    id_of = {p: ids[p] for p in range(len(ids))}
    sets = {
        "P_fit": {id_of[p] for p in p_fit},
        "P_test": {id_of[p] for p in p_test},
        "U_val": {id_of[p] for p in val},
    }
    names = list(sets)
    overl = {
        f"{a}&{b}": len(sets[a] & sets[b])
        for j, a in enumerate(names)
        for b in names[j + 1 :]
        if sets[a] & sets[b]
    }
    c1 = inp["c1"]
    G = {d.doc_id: OP.target_of(inp["blob"]["Delta"][d.doc_id]) for d in docs}
    # B-C3: G reproduces B0-CF's cap (offline-psi C2), on the FULL U
    c3 = {"skipped_smoke": True, "ok": False}
    if limit is None:
        rr_all = OP.reference_rates(docs, seed)
        c2r = OP.c2_reproduce(docs, inp["blob"], seed, rr_all)
        want = IC.ledger_rows(CF_LEDGER)[CF_CAP_KEY]["samples"][SEEDS.index(seed)]
        c3 = {"point": c2r["point"], "ledger": want, "ok": c2r["point"] == want}
    # B-C4: cell counts by kind over U rows equal B0's n
    allpos = list(range(len(docs)))
    mu_all = class_means(docs, G, allpos)
    c4 = {"skipped_smoke": True, "ok": False}
    if limit is None:
        rows = IC.ledger_rows(B0_LEDGER)
        want = {
            k: rows[B0_N_KEY.format(k=k)]["samples"][SEEDS.index(seed)] for k in KINDS
        }
        c4 = {
            "mine": mu_all["n_kind"],
            "ledger": want,
            "ok": all(float(mu_all["n_kind"][k]) == float(want[k]) for k in KINDS),
        }
    # probes
    probes: dict[str, Any] = {}
    kf_idx, kf_y = kind_cells(docs, p_fit)
    kt_idx, kt_y = kind_cells(docs, p_test)
    sf_idx, sf_y, nq_fit = status_cells(docs, p_fit)
    stt_idx, stt_y, nq_test = status_cells(docs, p_test)
    used_ids = {
        ids[p]
        for p in torch.cat([kf_idx[:, 0], kt_idx[:, 0], sf_idx[:, 0], stt_idx[:, 0]])
        .unique()
        .tolist()
    }
    n_tr = OP.N_U - OP.N_U_VAL  # offline-psi's registered U-train size
    want_sizes = (
        n_tr - (n_tr + P_TEST_MOD - 1) // P_TEST_MOD,
        (n_tr + P_TEST_MOD - 1) // P_TEST_MOD,
    )
    c2 = {
        "sizes": {k: len(v) for k, v in sets.items()},
        "expected_P_fit_P_test": list(want_sizes),
        "overlaps": overl,
        "u_val_ids_in_probe_tensors": len(used_ids & sets["U_val"]),
        "n_U_train": len(train),
        "n_U_train_ok": len(train) == n_tr,
        "ok": not overl
        and not (used_ids & sets["U_val"])
        and len(train) == n_tr
        and (limit is not None or (len(p_fit), len(p_test)) == want_sizes),
    }
    lams = (LAMBDA, *lambdas_desc)
    for lam in lams:
        key = f"lam{lam:g}"
        probes.setdefault("kind_s", {})[key] = run_probe(
            inp, kf_idx, kf_y, kt_idx, kt_y, "s", 3, lam, p_test, seed, n_boot=n_boot
        )
        for which in STATUS_FEATURES:
            probes.setdefault(f"status_{which}", {})[key] = run_probe(
                inp,
                sf_idx,
                sf_y,
                stt_idx,
                stt_y,
                which,
                2,
                lam,
                p_test,
                seed,
                n_boot=n_boot,
            )
        print(
            f"1b seed {seed} lambda {lam:g} done {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    # B-C5 (run-time half): the 16,640-feature status probe on label-shuffled P-fit
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(sf_y.numel(), generator=g)
    shuf = run_probe(
        inp,
        sf_idx,
        sf_y[perm],
        stt_idx,
        stt_y,
        "scx",
        2,
        LAMBDA,
        p_test,
        seed,
        n_boot=n_boot,
    )
    ba = shuf["bal_acc"]
    c5 = {
        "shuffled": shuf,
        "ci_contains_half": _finite(ba["lo"]) and ba["lo"] <= 0.5 <= ba["hi"],
    }
    c5["ok"] = c5["ci_contains_half"]
    # B-C6: finite everywhere (convergence is reported per probe, UNRESOLVED if not)
    fin = all(r["fit"]["finite"] for pr in probes.values() for r in pr.values())
    fin = fin and shuf["fit"]["finite"]
    c6 = {
        "finite": fin,
        "converged": {
            k: {lk: r["fit"]["converged"] for lk, r in v.items()}
            for k, v in probes.items()
        },
        "ok": fin,
    }
    # caps: primary out-of-sample (means on P-fit, replay on P-test); secondary in-sample
    mu_fit = class_means(docs, G, p_fit)
    cap_oos = caps(docs, p_test, mu_fit, seed)
    cap_ins = caps(docs, allpos, mu_all, seed) if limit is None else None
    return {
        "seed": seed,
        "n_documents": len(docs),
        "n_P_fit": len(p_fit),
        "n_P_test": len(p_test),
        "n_U_val_never_read": len(val),
        "n_querying_excluded": {"P_fit": nq_fit, "P_test": nq_test},
        "probes": probes,
        "means": {"P_fit": mu_fit, "U": mu_all},
        "caps": {"oos": cap_oos, "insample_U": cap_ins},
        "controls": {
            "B-C1": c1,
            "B-C2": c2,
            "B-C3": c3,
            "B-C4": c4,
            "B-C5": c5,
            "B-C6": c6,
        },
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# ledger, execute, main
# --------------------------------------------------------------------------- #


def b0_ko_seen() -> dict:
    rows = IC.ledger_rows(B0_LEDGER)
    return {f: rows[B0_KO_KEY.format(f=f)]["samples"] for f in ("point", "lo", "hi")}


def write_rows(led, per: dict[int, dict]) -> None:
    how = "item_1b.py::measure_seed; seed order 0,1,2"
    sd = [per[s] for s in SEEDS]

    def stat(key, vals, h=how):
        if all(_finite(v) for v in vals):
            led.stat(key, vals, how=h)
        else:
            led.note(key, vals, how=h + " (not finite on every seed)")

    for pname in sd[0]["probes"]:
        for lk in sd[0]["probes"][pname]:
            rs = [x["probes"][pname][lk] for x in sd]
            for f in ("point", "lo", "hi", "sd", "dropped"):
                stat(f"probe.{pname}.{lk}.bal_acc.{f}", [r["bal_acc"][f] for r in rs])
            for f in ("loss", "grad_max_abs", "n_iter", "func_evals", "seconds"):
                stat(f"probe.{pname}.{lk}.fit.{f}", [r["fit"][f] for r in rs])
            led.note(
                f"probe.{pname}.{lk}.converged",
                [r["fit"]["converged"] for r in rs],
                how=how,
            )
            stat(f"probe.{pname}.{lk}.n_fit_cells", [r["n_fit_cells"] for r in rs])
            stat(f"probe.{pname}.{lk}.n_test_cells", [r["n_test_cells"] for r in rs])
            led.note(
                f"probe.{pname}.{lk}.recall",
                [r["bal_acc"]["recall"] for r in rs],
                how=how,
            )
    for f in ("point", "lo", "hi"):
        stat(
            f"control.B-C5.shuffled.bal_acc.{f}",
            [x["controls"]["B-C5"]["shuffled"]["bal_acc"][f] for x in sd],
        )
    for setname in ("oos", "insample_U"):
        for rule in ("ko", "st"):
            for f in ("point", "lo", "hi"):
                stat(
                    f"cap.{setname}.{rule}_D.{f}",
                    [x["caps"][setname][rule]["cap"][f] for x in sd],
                    how + " -> B0.boot_rule (per document, 2000, paired)",
                )
                stat(
                    f"cap.{setname}.{rule}_D.diff_fifo.{f}",
                    [x["caps"][setname][rule]["diff_fifo"][f] for x in sd],
                )
            stat(
                f"cap.{setname}.{rule}_D.hit",
                [x["caps"][setname][rule]["hit"] for x in sd],
            )
            stat(
                f"cap.{setname}.{rule}_D.dropped",
                [x["caps"][setname][rule]["cap_dropped"] for x in sd],
            )
        stat(f"cap.{setname}.fifo_hit", [x["caps"][setname]["fifo_hit"] for x in sd])
        stat(
            f"cap.{setname}.factfiller_hit",
            [x["caps"][setname]["factfiller_hit"] for x in sd],
        )
    for mset in ("P_fit", "U"):
        for k in KINDS:
            stat(f"means.{mset}.E_G.kind.{k}", [x["means"][mset]["kind"][k] for x in sd])
            stat(
                f"means.{mset}.E_G.kind.{k}.n",
                [x["means"][mset]["n_kind"][k] for x in sd],
            )
        for c in CLASSES5:
            stat(
                f"means.{mset}.E_G.class.{c}", [x["means"][mset]["class"][c] for x in sd]
            )
            stat(
                f"means.{mset}.E_G.class.{c}.n",
                [x["means"][mset]["n_class"][c] for x in sd],
            )
    seen = b0_ko_seen()
    for f in ("point", "lo", "hi"):
        led.stat(f"SEEN.b0.B.ckpt3000.g09.cap.ko.{f}", seen[f], how=f"{B0_LEDGER} (SEEN)")
    for k in ("n_P_fit", "n_P_test", "n_U_val_never_read", "n_documents"):
        stat(k, [x[k] for x in sd])
    for part in ("P_fit", "P_test"):
        stat(f"n_querying_excluded.{part}", [x["n_querying_excluded"][part] for x in sd])
    stat("seconds_per_seed", [x["seconds"] for x in sd], "wall clock")


def labels_of(per: dict[int, dict]) -> dict:
    out = {}
    for pname, th in (("kind_s", THRESH["kind"]), ("status_scx", THRESH["status"])):
        lk = f"lam{LAMBDA:g}"
        cis = {s: per[s]["probes"][pname][lk]["bal_acc"] for s in per}
        conv = {s: per[s]["probes"][pname][lk]["fit"]["converged"] for s in per}
        out[pname] = probe_label(cis, th, conv)
    return out


def manifest_config(args: dict) -> dict:
    return {
        "run_id": RUN_ID,
        "item": ITEM,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": list(SEEDS),
        "gamma": GAMMA,
        "split": "U-train (offline-psi u_split) order j: P-test j % 5 == 0, P-fit rest",
        "lambda": LAMBDA,
        "lambda_descriptive": list(LAMBDA_DESCRIPTIVE),
        "max_iter": MAX_ITER,
        "optimizer": "torch.optim.LBFGS(lr=1, line_search_fn='strong_wolfe', defaults)",
        "loss": "class-balanced CE (mean per class, averaged) + lambda * ||weights||^2",
        "thresholds": THRESH,
        "classes5": list(CLASSES5),
        "bootstrap": {
            "n": N_BOOT,
            "fn": "R1.boot_weights",
            "seed_base": R1.BOOT_SEED_BASE,
        },
        "inputs": {
            "gestalt_sha256": CV.GESTALT_SHA,
            "delta_sha256": OP.DELTA_SHA,
            "cf_ledger": str(CF_LEDGER),
            "b0_ledger": str(B0_LEDGER),
        },
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "args": args,
    }


def execute(led, a) -> Exit:
    g = IC.Globals(led, reads_substrate=True, args=a)
    failed = g.start()
    per: dict[int, dict] = {}
    if not failed or a.limit is not None:
        for s in a.seeds:
            per[s] = measure_seed(s, limit=a.limit, n_boot=a.n_boot)
            print(f"1b seed {s} done {per[s]['seconds']:.0f}s", flush=True)
    g.end(sorted(per))
    for s, r in per.items():
        for name, c in r["controls"].items():
            if not c.get("ok"):
                failed.append(f"{name} seed{s}")
        led.note(f"seed{s}.controls", r["controls"], how="item_1b.py::measure_seed")
    if a.limit is not None:
        failed.append(f"smoke: --limit {a.limit} (not the registered set)")
    led.note("controls_failed", failed, how="item_1b.py::execute")
    labs = labels_of(per) if per else {}
    if sorted(per) == list(SEEDS) and a.limit is None:
        write_rows(led, per)
    led.note("labels", labs, how="item_1b.py::labels_of (PREREG §4, tightened)")
    led.note("labels_read", not failed, how="False: a control failed, NOT READ")
    if failed:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=f"Tier D diagnostic; registered labels {labs}; every control passed",
    )
    return Exit.OK


def _run(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--limit", type=int, default=None, help="smoke only: U positions < N")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--run-id", default=RUN_ID)
    IC.add_global_args(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    from ledger import Ledger

    led = Ledger(a.run_id, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config({k: str(v) for k, v in vars(a).items()}))
    try:
        rc = execute(led, a)
    except Exception as e:
        import traceback

        led.note("exception", traceback.format_exc(), how="item_1b.py::main")
        led.status("crashed")
        sealed = type(e).__name__ in ("SealedRange", "ControlFailure")
        rc = Exit.DID_NOT_RUN if sealed else Exit.FAIL
    led.command(
        [sys.executable, RUN_PY, ITEM] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the item's",
    )
    led.write()
    return status(rc)


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #


def _row(rows: dict, key: str):
    r = rows.get(key)
    return None if r is None else r.get("samples", r.get("value"))


def _fmt(x, nd: int = 4) -> str:
    return "n/a" if not _finite(x) else f"{x:.{nd}f}"


def _ci3(rows, base, i):
    p, lo, hi = (_row(rows, f"{base}.{f}") for f in ("point", "lo", "hi"))
    if p is None:
        return "n/a"
    return f"{_fmt(p[i])} [{_fmt(lo[i])}, {_fmt(hi[i])}]"


def render(out_root: Path = OUT_DEFAULT) -> Path:
    led = json.loads((Path(out_root) / "ledger.json").read_text())
    rows = {r["key"]: r for r in led["rows"]}
    labs = rows["labels"]["value"]
    read = rows["labels_read"]["value"]
    failed = rows["controls_failed"]["value"]
    cmd = led["commands"][-1] if led["commands"] else {}
    L = [
        "# post-fails-controls 1b: linear probes and class-mean rank caps on U",
        "",
        f"Generated by `{EXPERIMENT}::finalize` from `{Path(out_root) / 'ledger.json'}`. "
        "No number below is typed.",
        "",
        f"- PREREG: `{PREREG}` @ {PREREG_COMMIT} (§4)",
        f"- git sha: `{led['provenance'].get('git_sha')}` · device: {led['device']} · "
        f"seeds actually run: {led['seeds_actually_run']} · status: {led['status']}",
        f"- command: `{cmd.get('argv')}` · exit code: {cmd.get('exit_code')}",
        f"- config hash: `{led['config_hash']}`",
        f"- controls failed: {failed if failed else 'none'}",
        "",
        "## Registered labels" + ("" if read else " (NOT READ: a control failed)"),
        "",
        f"- kind probe (from s, threshold 0.95 on the CI): **{labs.get('kind_s')}**",
        f"- status probe (from [s; c; s(x)c], threshold 0.80 on the CI): "
        f"**{labs.get('status_scx')}**",
        "- The s-only and c-only status probes are descriptive.",
        "",
        "## Held-out (P-test) balanced accuracy, point [95% CI], per seed",
        "",
        "| probe | lambda | seed 0 | seed 1 | seed 2 | converged |",
        "|---|---|---|---|---|---|",
    ]
    for pname in ("kind_s", "status_s", "status_c", "status_scx"):
        for lam in (LAMBDA, *LAMBDA_DESCRIPTIVE):
            base = f"probe.{pname}.lam{lam:g}"
            conv = _row(rows, f"{base}.converged")
            role = "registered" if lam == LAMBDA else "descriptive"
            L.append(
                f"| {pname} | {lam:g} ({role}) | "
                + " | ".join(_ci3(rows, f"{base}.bal_acc", i) for i in range(3))
                + f" | {conv} |"
            )
    L += [
        "",
        "Shuffled-label control (B-C5): "
        + " · ".join(_ci3(rows, "control.B-C5.shuffled.bal_acc", i) for i in range(3)),
        "",
    ]
    L += [
        "## Rank caps, cap point [95% CI], per seed",
        "",
        "| set | rule | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|---|",
    ]
    for setname, lab in (
        ("oos", "P-fit means -> P-test (primary)"),
        ("insample_U", "all U, in-sample (secondary)"),
    ):
        for rule in ("ko", "st"):
            L.append(
                f"| {lab} | {rule}_D | "
                + " | ".join(_ci3(rows, f"cap.{setname}.{rule}_D", i) for i in range(3))
                + " |"
            )
    L.append(
        "| B0's ko on the r_i return (SEEN) | ko | "
        + " | ".join(_ci3(rows, "SEEN.b0.B.ckpt3000.g09.cap.ko", i) for i in range(3))
        + " |"
    )
    L += [
        "",
        "## Class means of the Delta return (G, gamma 0.9)",
        "",
        "| set | class | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|---|",
    ]
    for mset in ("P_fit", "U"):
        for k in KINDS:
            v = _row(rows, f"means.{mset}.E_G.kind.{k}") or [None] * 3
            L.append(f"| {mset} | kind {k} | " + " | ".join(_fmt(x) for x in v) + " |")
        for c in CLASSES5:
            v = _row(rows, f"means.{mset}.E_G.class.{c}") or [None] * 3
            L.append(f"| {mset} | {c} | " + " | ".join(_fmt(x) for x in v) + " |")
    L += ["", "## Controls", ""]
    for s in SEEDS:
        c = (rows.get(f"seed{s}.controls") or {}).get("value") or {}
        L.append(
            f"- seed {s}: "
            + ", ".join(
                f"{k} {'pass' if (v or {}).get('ok') else 'FAIL'}" for k, v in c.items()
            )
        )
    L += [
        "",
        "## What 1b does NOT show (PREREG §4)",
        "",
        "- Linear decodability is not use.",
        "- The caps are model-free residency of hindsight class-mean rules, not "
        "accuracy through the reader and not a learnable policy's reach.",
        "- Gestalts and Delta are FIFO-world.",
        "- U is already inspected; only P-test is held out from the probe fit.",
        "",
    ]
    p = Path(out_root) / "RESULTS.md"
    p.write_text("\n".join(L) + "\n")
    claims = [
        {
            "claim": f"1b labels are {labs}",
            "command": "see ledger rows 'labels'",
            "expected": labs,
        }
    ]
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        f"--slots 1 --wait 3600 -- .venv/bin/python {RUN_PY} {ITEM} recompute"
    )
    for s in SEEDS:
        for pname in ("kind_s", "status_scx"):
            for f in ("point", "lo", "hi"):
                key = f"probe.{pname}.lam{LAMBDA:g}.bal_acc.{f}"
                v = _row(rows, key)
                if v is not None:
                    claims.append(
                        {
                            "claim": f"seed {s} {key} = {v[s]!r}",
                            "command": f"{base} --seed {s} --what {key}",
                            "expected": v[s],
                        }
                    )
    claims[0]["command"] = f"{base} --what labels"
    (Path(out_root) / "claims.json").write_text(json.dumps(claims, indent=1) + "\n")
    return p


def recompute(what: str, seed: int | None) -> str:
    if what == "labels":
        per = {s: measure_seed(s) for s in SEEDS}
        return repr(labels_of(per))
    r = measure_seed(int(seed))
    _, pname, lk, _, f = what.split(".")
    return repr(r["probes"][pname][lk]["bal_acc"][f])


def finalize(argv: list[str] | None = None) -> int:
    """``run.py finalize 1b [--out DIR]``: RESULTS.md and claims.json, from the ledger
    in DIR (default ``runs/{RUN_ID}``). No number is typed."""
    ap = ArgumentParser(description="finalize 1b")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(list(argv or []))
    print(render(a.out))
    return int(Exit.OK)


def main(argv: list[str] | None = None) -> int:
    """``run.py 1b [measure] [options]`` measures and writes the ledger;
    ``run.py 1b finalize`` / ``run.py finalize 1b`` renders; ``run.py 1b recompute
    --what KEY [--seed S]`` re-derives one value (the verifier's entry)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        return finalize(argv[1:])
    if argv[:1] == ["recompute"]:
        ap = ArgumentParser(description="recompute one 1b value")
        ap.add_argument("--what", required=True)
        ap.add_argument("--seed", type=int, default=None)
        a = ap.parse_args(argv[1:])
        torch.set_num_threads(1)
        IC.common().install_seal()
        print(recompute(a.what, a.seed))
        return 0
    if argv[:1] == ["measure"]:
        argv = argv[1:]
    return int(_run(argv))


if __name__ == "__main__":
    run_main(main)
