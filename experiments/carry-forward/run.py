"""carry-forward -- arm B: answer in its own slot, or surviving that slot's eviction?

Pre-registration: `experiments/carry-forward/PREREG.md` (commit ca34aa7; amendment 1
f4e1c87), committed alone
ahead of this file. Every constant in the "PREREG front matter, transcribed" block is
that file's; changing one is changing the PREREG.

Reads six FROZEN fresh-stream arm B checkpoints (seeds 0-2 x ckpt 2500, 3000), read
only, and trains nothing. Per seed x checkpoint:

* the reproduction control -- S0-03's `answer_readout` (live, slots_zeroed) on H64 =
  fresh-stream's raw.json / ledger values;
* `rsr.metrics.loo.loo_readout` (W4) on H64 (one call) and on EXT = [64, 1088) (four
  calls of 256 consecutive ids -- donors come from the same call), with the bit-exact
  live control against `answer_readout` on every call;
* L (localisation), S (specificity) and reach (carry-forward past eviction), with
  per-document cluster-bootstrap CIs; the classification at ckpt 3000.

Usage:

    uv run python experiments/carry-forward/run.py              # the run (~CPU)
    uv run python experiments/carry-forward/run.py --render-results  # RESULTS.md
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, run_main  # noqa: E402

EXPERIMENT = "experiments/carry-forward/run.py"
RUN_ID = "carry-forward"
PREREG = "experiments/carry-forward/PREREG.md"
PREREG_COMMIT = "ca34aa7"
AMENDMENT_COMMIT = "f4e1c87"
AMENDMENT2_COMMIT = "e3cfcc2"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CSC = _load("_carry_forward_csc", "experiments/corpus-size-curve/run.py")
S003 = CSC.S003

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md front matter, transcribed.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
CHECKPOINTS = (2500, 3000)
FINAL = 3000
M = S003.M  # 16
H64 = CSC.HELDOUT  # (4096, 4160)
EXT = (64, 1088)
EXT_CHUNK = 256
VOCAB_DOCUMENTS = (0, 64)
#: Everything arm B's lineage, or a run sharing its stream, trained on or is measured on.
#: [0, 64): corpus-size n64 (arm B's steps 0-999) = vocab = probe; [4096, 4160): H64;
#: [4160, 52160): fresh-stream's stream (arm A steps 0-2999, arm B steps 1000-2999);
#: [52160, 148160): fresh-escape's continuation of the same stream (steps 3000-8999).
FORBIDDEN = {
    "vocab_probe_and_n64_training": (0, 64),
    "heldout_H64": H64,
    "fresh_stream": (4160, 52160),
    "fresh_escape": (52160, 148160),
}
EXT_ALLOWED = (64, 4096)
L_LOCALISED_LO = 0.5
L_DIFFUSE_HI = 0.5
S_SPECIFIC_HI = 0.25
S_NOT_SPECIFIC_LO = 0.25
S_MIN_COVERAGE = 0.20
REACH_DELTA = 0.03
RESAMPLE_MIN_COVERAGE = 0.80
#: Amendment 2 ruling 3: |L - L_sensitivity| above this, or a label change, is a
#: material disagreement -> L_INCONCLUSIVE.
L_SENSITIVITY_MAX_DIFF = 0.15
REPRO_TOL = 1e-6
BOOT_R = 2000
BOOT_SEED = 20260926
BOOT_Q = (0.025, 0.975)
THREADS = 12  # fresh-stream's parent_torch_threads
DEVICE = "cpu"
BANDS = {
    "17_20": (17, 20),
    "21_24": (21, 24),
    "25_32": (25, 32),
    "33_40": (33, 40),
    "17_40": (17, 40),
}
DECISION_BAND = "17_40"
STATS = ("answer_nll", "answer_acc", "answer_nll_over_16", "answer_brier_over_16")

FALSIFIER = (
    "Arm B's FIFO memory carries answer information past the eviction of the answer's "
    "own slot (CARRY: pooled gap 17-40 excess of live over single-step all-slots-zeroed "
    "accuracy >= 0.03 with its 95% per-document CI lower bound > 0, on 1024 unseen "
    "documents) on at least one seed at ckpt 3000. Falsified if no seed shows CARRY."
)
EXPECTED = (
    "MIXED: CARRY on seed 2 only (pooled 17-40 excess ~0.04-0.08, CI clear of 0); L "
    "LOCALISED (point >= 0.8) and S SPECIFIC on every seed; L_zero >= L. Second most "
    "likely: LOCALISED (PREREG 'Prediction', written before any number of this run)."
)
QUESTION = (
    "In fresh-stream arm B (FIFO, frozen checkpoints), does a query's answer on an "
    "unseen document live in the one memory slot that holds its assert, or is it "
    "carried forward past that slot's eviction?"
)


def default_source() -> Path:
    """fresh-stream's run directory (READ-ONLY): the sibling worktree when this tree
    is itself under `.worktrees/`, else the main checkout's `.worktrees/`."""
    base = ROOT.parent if ROOT.parent.name == ".worktrees" else ROOT / ".worktrees"
    return base / "fresh-stream" / "runs" / "fresh-stream"


def ckpt_path(source: Path, seed: int, ckpt: int) -> Path:
    return source / "B" / f"seed{seed}" / f"ckpt-{ckpt:06d}.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# documents (control 4)
# --------------------------------------------------------------------------- #


def ext_ids() -> list[int]:
    return list(range(*EXT))


def ext_chunks(ids: list[int], chunk: int = EXT_CHUNK) -> list[list[int]]:
    return [ids[k : k + chunk] for k in range(0, len(ids), chunk)]


def disjointness(ids: list[int]) -> dict:
    """Control 4a: every id in EXT_ALLOWED, none in a FORBIDDEN range, none repeated."""
    uniq = set(ids)
    hits = {name: sorted(uniq & set(range(*rng)))[:10] for name, rng in FORBIDDEN.items()}
    outside = sorted(i for i in uniq if not EXT_ALLOWED[0] <= i < EXT_ALLOWED[1])[:10]
    repeats = len(ids) - len(uniq)
    ok = bool(ids) and not repeats and not outside and not any(hits.values())
    return {
        "ok": ok,
        "n_ids": len(ids),
        "first_id": min(ids) if ids else None,
        "last_id": max(ids) if ids else None,
        "repeats": repeats,
        "outside_allowed": outside,
        "hits": hits,
    }


def _cfg(seed: int):
    from rsr.data.synthetic import SyntheticConfig

    return SyntheticConfig(
        sentences_per_document=S003.CONFIG["steps_per_stream"], seed=seed
    )


def gen_docs(seed: int, ids) -> list:
    """Documents by id, prefix-stable (`_generate_document`)."""
    from rsr.data.synthetic import _generate_document

    cfg = _cfg(seed)
    return [_generate_document(i, cfg) for i in ids]


def seed_sets(seed: int) -> dict:
    """-> {"H64": docs, "EXT": docs, "vmap": [0,64) vocab, "V": int, "closure": dict}.

    Control 4b/4c: H64 via `doc_sets(seed, 64)` (fresh-stream's path); the vocabulary
    is `doc_sets`'s and must equal [0, 64)'s built by id; H64 by id must equal
    `doc_sets`'s H64; EXT must be closed under that vocabulary."""
    from rsr.train.loop import build_vocab, check_vocabulary_closure

    sets, vmap, V = CSC.doc_sets(seed, VOCAB_DOCUMENTS[1])
    if build_vocab(gen_docs(seed, range(*VOCAB_DOCUMENTS))) != vmap:
        raise ValueError(f"seed {seed}: doc_sets' vocabulary is not [0, 64)'s")
    h64 = sets["heldout"]
    if gen_docs(seed, range(*H64)) != list(h64):
        raise ValueError(f"seed {seed}: H64 by id differs from doc_sets' H64")
    ext = gen_docs(seed, ext_ids())
    check_vocabulary_closure(ext, vmap)
    check_vocabulary_closure(h64, vmap)
    return {
        "H64": list(h64),
        "EXT": ext,
        "vmap": vmap,
        "V": V,
        "closure": {
            "ok": True,
            "vocab_words": len(vmap),
            "n_ext": len(ext),
            "n_h64": len(h64),
        },
    }


def encode_set(docs, vmap):
    from rsr.train.loop import answer_targets, encode

    S, L = S003.CONFIG["steps_per_stream"], S003.CONFIG["max_tokens"]
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    tmask, gap = answer_targets(docs, vmap, max_tokens=L, steps=S)
    return ids, mask, tmask, gap


def sym_ids_of(vmap) -> torch.Tensor:
    from rsr.data.synthetic import ANSWER_SYMBOLS

    return torch.tensor([vmap[s] for s in ANSWER_SYMBOLS])


def load_model(ckpt: Path, V: int):
    from rsr.model.tg import TGModel
    from rsr.train import checkpoint as ck

    model = TGModel(S003._cfg(V)).to(DEVICE)
    ck.load(ckpt, model=model, restore_rng=False)
    return model


# --------------------------------------------------------------------------- #
# controls 2, 3, 5, 6
# --------------------------------------------------------------------------- #


def compare_summary(ref: dict, got: dict, tol: float = REPRO_TOL) -> dict:
    """Control 2: every bucket's `n` equal, every statistic and `real_token_nll`
    within `tol` of the reference. A key missing on either side is a mismatch."""
    bad: list[str] = []
    max_diff = 0.0
    if set(ref) != set(got):
        bad.append(f"keys differ: {sorted(set(ref) ^ set(got))}")
    for k in sorted(set(ref) & set(got)):
        if k == "real_token_nll":
            d = abs(float(ref[k]) - float(got[k]))
            max_diff = max(max_diff, d)
            if d > tol:
                bad.append(f"{k}: {ref[k]} vs {got[k]}")
            continue
        r, g = ref[k], got[k]
        if r["n"] != g["n"]:
            bad.append(f"{k}.n: {r['n']} vs {g['n']}")
            continue
        for s in STATS:
            if (r[s] is None) != (g[s] is None):
                bad.append(f"{k}.{s}: {r[s]} vs {g[s]}")
            elif r[s] is not None:
                d = abs(float(r[s]) - float(g[s]))
                max_diff = max(max_diff, d)
                if d > tol:
                    bad.append(f"{k}.{s}: {r[s]} vs {g[s]}")
    return {"ok": not bad, "max_abs_diff": max_diff, "mismatches": bad[:20]}


def compare_ledger_samples(ledger: dict, seed: int, got: dict[str, dict]) -> dict:
    """Control 2 at ckpt 3000: `B.ckpt3000.heldout.<cond>.<bucket>.<stat>` samples[seed]
    equal the reproduced value within REPRO_TOL. `got` is {cond: summary}."""
    rows = {r["key"]: r for r in ledger["rows"] if "samples" in r}
    bad, n_checked, max_diff = [], 0, 0.0
    for cond, summ in got.items():
        for bucket, v in summ.items():
            if bucket == "real_token_nll":
                continue
            for s in STATS:
                key = f"B.ckpt{FINAL}.heldout.{cond}.{bucket}.{s}"
                if key not in rows or v[s] is None:
                    continue
                d = abs(float(rows[key]["samples"][seed]) - float(v[s]))
                n_checked += 1
                max_diff = max(max_diff, d)
                if d > REPRO_TOL:
                    bad.append(f"{key}: {rows[key]['samples'][seed]} vs {v[s]}")
    ok = n_checked > 0 and not bad
    return {"ok": ok, "n_checked": n_checked, "max_abs_diff": max_diff, "mismatches": bad}


def bitexact_live(loo_res: dict, ar: dict, prefix: str = "live") -> dict:
    """Control 3: loo_readout's `<prefix>` columns equal answer_readout(cond=prefix)'s,
    torch.equal, target for target (`prefix` "live" or "live_bos_off", A5)."""
    pairs = {
        "nll": (loo_res[f"{prefix}_nll"], ar["nll"].double()),
        "ok": (loo_res[f"{prefix}_ok"], ar["ok"].double()),
        "nll16": (loo_res[f"{prefix}_nll16"], ar["nll16"].double()),
        "brier16": (loo_res[f"{prefix}_brier16"], ar["brier16"].double()),
        "gap": (loo_res["gap"].long(), ar["gap"].long()),
    }
    eq = {
        k: bool(a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
        for k, (a, b) in pairs.items()
    }
    diff = {
        k: (float((a.cpu() - b.cpu()).abs().max()) if a.shape == b.shape else None)
        for k, (a, b) in pairs.items()
    }
    return {"ok": all(eq.values()), "equal": eq, "max_abs_diff": diff}


def _loo():
    from rsr.metrics import loo

    return loo


def residency(rec: dict) -> dict:
    """Control 5 (A5): among targets whose assert was written, own_status is
    resident iff gap <= M (FIFO); a never-written assert is counted, not tested."""
    L = _loo()
    st = rec["own_status"]
    written = st != L.OWN_NEVER_WRITTEN
    want = rec["gap"] <= M
    bad = int(((st == L.OWN_RESIDENT) != want)[written].sum())
    also = int((rec["own_resident"].bool() != (st == L.OWN_RESIDENT)).sum())
    return {
        "ok": bad == 0 and also == 0,
        "n_targets": int(rec["gap"].numel()),
        "n_never_written": int((~written).sum()),
        "n_violations": bad + also,
    }


# --------------------------------------------------------------------------- #
# the cluster bootstrap
# --------------------------------------------------------------------------- #


def cluster_boot(
    doc: torch.Tensor,
    cols: dict[str, torch.Tensor],
    stat: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    R: int = BOOT_R,
    seed: int = BOOT_SEED,
) -> dict:
    """Per-document cluster bootstrap of a statistic of SUMS.

    `cols` are per-target values; `stat` maps {name: sum} (0-d for the point, `[R]`
    for the resamples) to the statistic, so a ratio is a ratio of sums. Documents
    are resampled with replacement (`R` draws of D documents out of D), never
    targets. Returns point, 95% percentile CI over the finite resamples, the
    number of non-finite resamples, n targets and n documents."""
    n = int(doc.numel())
    if n == 0:
        nan = float("nan")
        return {"point": nan, "lo": nan, "hi": nan, "n": 0, "n_docs": 0, "nonfinite": R}
    uniq, inv = torch.unique(doc, return_inverse=True)
    D = int(uniq.numel())
    sums = {
        k: torch.zeros(D, dtype=torch.float64).index_add_(0, inv, v.double())
        for k, v in cols.items()
    }
    point = float(stat({k: s.sum() for k, s in sums.items()}))
    g = torch.Generator().manual_seed(seed)
    draw = torch.randint(D, (R, D), generator=g)
    W = torch.zeros(R, D, dtype=torch.float64).scatter_add_(
        1, draw, torch.ones(R, D, dtype=torch.float64)
    )
    boot = stat({k: W @ s for k, s in sums.items()})
    fin = boot[torch.isfinite(boot)]
    if fin.numel():
        q = torch.quantile(fin, torch.tensor(BOOT_Q, dtype=torch.float64))
        lo, hi = float(q[0]), float(q[1])
    else:
        lo = hi = float("nan")
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "n": n,
        "n_docs": D,
        "nonfinite": int(boot.numel() - fin.numel()),
    }


# --------------------------------------------------------------------------- #
# readouts (PREREG + Amendment 1)
# --------------------------------------------------------------------------- #


def _applied(x: torch.Tensor) -> torch.Tensor:
    return ~torch.isnan(x)


def _flag(rec: dict, k: str) -> torch.Tensor:
    return rec[k].bool()


def window(rec: dict) -> torch.Tensor:
    g = rec["gap"]
    return (g >= 2) & (g <= M)


def l_population(rec: dict) -> torch.Tensor:
    """A1/A3: gap 2..M, own resident, own_resample AND all_slots_resample applied,
    not dup_key_in_doc."""
    return (
        window(rec)
        & (rec["own_status"] == _loo().OWN_RESIDENT)
        & _applied(rec["own_resample_ok"])
        & _applied(rec["all_slots_resample_ok"])
        & ~_flag(rec, "dup_key_in_doc")
    )


def s_population(rec: dict) -> torch.Tensor:
    """A3: the L population, ctrl present and resampled, no object leak anywhere."""
    return (
        l_population(rec)
        & (rec["ctrl_status"] == 0)
        & _applied(rec["ctrl_resample_ok"])
        & ~_flag(rec, "ctrl_same_object")
        & ~_flag(rec, "own_donor_same_object")
        & ~_flag(rec, "ctrl_donor_same_object")
    )


def _col(metric: str) -> str:
    return "ok" if metric == "acc" else "nll16"


def ratio_ci(rec, pop, knock: str, base: str, metric: str, live: str = "live") -> dict:
    """(live - knock) / (live - base) as a ratio of sums (acc); for nll16 the signs
    flip so that 1 = the knockout costs as much as the baseline."""
    c = _col(metric)
    cols = {
        "live": rec[f"{live}_{c}"][pop],
        "k": rec[f"{knock}_{c}"][pop],
        "b": rec[f"{base}_{c}"][pop],
    }
    return cluster_boot(
        rec["doc_id"][pop], cols, lambda s: (s["live"] - s["k"]) / (s["live"] - s["b"])
    )


def drop_ci(rec, pop, knock: str, metric: str, live: str = "live") -> dict:
    """Paired mean drop: mean(live_ok - knock_ok); nll16: mean(knock - live)."""
    c = _col(metric)
    sign = 1.0 if metric == "acc" else -1.0
    cols = {
        "live": rec[f"{live}_{c}"][pop],
        "k": rec[f"{knock}_{c}"][pop],
        "n": torch.ones(int(pop.sum()), dtype=torch.float64),
    }
    return cluster_boot(
        rec["doc_id"][pop], cols, lambda s: sign * (s["live"] - s["k"]) / s["n"]
    )


def l_readout(rec: dict, pop: torch.Tensor | None = None) -> dict:
    """A1: L (own_resample / all_slots_resample), L_zero (own_zero /
    all_slots_zeroed, OOD), their bos-off twins (A2), the denominator guard and
    the in-window memory contribution (live - memory_off)."""
    w = window(rec)
    base_pop = w & ~_flag(rec, "dup_key_in_doc")
    if pop is None:
        pop = l_population(rec)
    resident = base_pop & (rec["own_status"] == _loo().OWN_RESIDENT)
    out: dict[str, Any] = {
        "n_window": int(w.sum()),
        "n_dup_key_excluded": int((w & _flag(rec, "dup_key_in_doc")).sum()),
        "n_resident": int(resident.sum()),
        "n_pop": int(pop.sum()),
        "resample_coverage": int(pop.sum()) / max(int(resident.sum()), 1),
    }
    for m in ("acc", "nll16"):
        out[f"L.{m}"] = ratio_ci(rec, pop, "own_resample", "all_slots_resample", m)
        out[f"L_zero.{m}"] = ratio_ci(rec, pop, "own_zero", "all_slots_zeroed", m)
        out[f"denominator.{m}"] = drop_ci(rec, pop, "all_slots_resample", m)
        out[f"memory_off_drop.{m}"] = drop_ci(rec, pop, "memory_off", m)
    out["L_bos_off.acc"] = ratio_ci(
        rec,
        pop,
        "own_resample_bos_off",
        "all_slots_resample_bos_off",
        "acc",
        live="live_bos_off",
    )
    for cond in (
        "live",
        "own_resample",
        "own_zero",
        "all_slots_resample",
        "all_slots_zeroed",
        "memory_off",
    ):
        out[f"mean.{cond}.acc"] = (
            float(rec[f"{cond}_ok"][pop].mean()) if int(pop.sum()) else None
        )
    out["label"] = l_label(out["L.acc"], out["denominator.acc"])
    return out


def gap1_population(rec: dict) -> torch.Tensor:
    """Amendment 2 ruling 2: gap 1, own resident, both bos-off resamples applied,
    not dup_key_in_doc."""
    return (
        (rec["gap"] == 1)
        & (rec["own_status"] == _loo().OWN_RESIDENT)
        & _applied(rec["own_resample_bos_off_ok"])
        & _applied(rec["all_slots_resample_bos_off_ok"])
        & ~_flag(rec, "dup_key_in_doc")
    )


def l_full(rec: dict) -> dict:
    """l_readout on the L population, plus Amendment 2: the gap-1 bos-off L
    (secondary, unlabelled) and the sensitivity L on `all_donor_has_object ==
    False`, whose material disagreement turns the label into L_INCONCLUSIVE."""
    out = l_readout(rec)
    pop = l_population(rec)
    sens_pop = pop & ~_flag(rec, "all_donor_has_object")
    sens = l_readout(rec, sens_pop)
    out["n_donor_has_object"] = int((pop & _flag(rec, "all_donor_has_object")).sum())
    out["L_sensitivity.acc"] = sens["L.acc"]
    out["denominator_sensitivity.acc"] = sens["denominator.acc"]
    out["label_primary"] = out["label"]
    out["label_sensitivity"] = sens["label"]
    out["sensitivity_disagrees"] = sensitivity_disagrees(
        out["L.acc"], sens["L.acc"], out["label"], sens["label"]
    )
    if out["sensitivity_disagrees"]:
        out["label"] = "L_INCONCLUSIVE"
    g1 = gap1_population(rec)
    out["n_gap1"] = int(g1.sum())
    out["L_gap1_bos_off.acc"] = ratio_ci(
        rec,
        g1,
        "own_resample_bos_off",
        "all_slots_resample_bos_off",
        "acc",
        live="live_bos_off",
    )
    return out


def sensitivity_disagrees(primary: dict, sens: dict, lab_p: str, lab_s: str) -> bool:
    """Amendment 2 ruling 3: a label change, or |point diff| > 0.15 (NaN counts
    as a disagreement: the sensitivity L could not be read)."""
    d = abs(primary["point"] - sens["point"])
    return bool(lab_p != lab_s or d != d or d > L_SENSITIVITY_MAX_DIFF)


def l_label(ci: dict, denominator: dict | None = None) -> str:
    """A1 guard first: a denominator whose CI lower bound is <= 0 -> INDETERMINATE."""
    if denominator is not None:
        d = denominator["lo"]
        if not (d == d and d > 0):
            return "INDETERMINATE"
    lo, hi = ci["lo"], ci["hi"]
    if lo == lo and lo >= L_LOCALISED_LO:
        return "LOCALISED"
    if hi == hi and hi < L_DIFFUSE_HI:
        return "DIFFUSE"
    return "INDETERMINATE"


def s_readout(rec: dict) -> dict:
    """drop_own, drop_ctrl, their difference and ratio, paired on the S rows."""
    lpop, spop = l_population(rec), s_population(rec)
    n_l = int(lpop.sum())
    ctrl_ok = lpop & (rec["ctrl_status"] == 0) & _applied(rec["ctrl_resample_ok"])
    out: dict[str, Any] = {
        "n_L": n_l,
        "n_S": int(spop.sum()),
        "coverage": int(spop.sum()) / max(n_l, 1),
        "ctrl_status_counts_in_L": [
            int(((rec["ctrl_status"] == k) & lpop).sum()) for k in (0, 1, 2)
        ],
        "excluded_in_L_with_ctrl": {
            k: int((ctrl_ok & _flag(rec, k)).sum())
            for k in (
                "ctrl_same_object",
                "own_donor_same_object",
                "ctrl_donor_same_object",
            )
        },
    }
    for m in ("acc", "nll16"):
        c = _col(m)
        sign = 1.0 if m == "acc" else -1.0
        cols = {
            "live": rec[f"live_{c}"][spop],
            "own": rec[f"own_resample_{c}"][spop],
            "ctrl": rec[f"ctrl_resample_{c}"][spop],
            "n": torch.ones(int(spop.sum()), dtype=torch.float64),
        }
        doc = rec["doc_id"][spop]
        out[f"drop_own.{m}"] = cluster_boot(
            doc, cols, lambda s, k=sign: k * (s["live"] - s["own"]) / s["n"]
        )
        out[f"drop_ctrl.{m}"] = cluster_boot(
            doc, cols, lambda s, k=sign: k * (s["live"] - s["ctrl"]) / s["n"]
        )
        out[f"drop_diff.{m}"] = cluster_boot(
            doc, cols, lambda s, k=sign: k * (s["ctrl"] - s["own"]) / s["n"]
        )
        out[f"ratio.{m}"] = cluster_boot(
            doc, cols, lambda s: (s["live"] - s["ctrl"]) / (s["live"] - s["own"])
        )
    out["ratio_bos_off.acc"] = ratio_ci(
        rec,
        spop,
        "ctrl_resample_bos_off",
        "own_resample_bos_off",
        "acc",
        live="live_bos_off",
    )
    out["L_on_S"] = ratio_ci(rec, spop, "own_resample", "all_slots_resample", "acc")
    out["label"] = s_label(out["coverage"], out["ratio.acc"])
    return out


def s_label(coverage: float, ratio: dict) -> str:
    if coverage < S_MIN_COVERAGE:
        return "INSUFFICIENT_COVERAGE"
    lo, hi = ratio["lo"], ratio["hi"]
    if hi == hi and hi <= S_SPECIFIC_HI:
        return "SPECIFIC"
    if lo == lo and lo > S_NOT_SPECIFIC_LO:
        return "NOT_SPECIFIC"
    return "INDETERMINATE"


#: A4 secondaries: name -> (knock column prefix, live column prefix)
REACH_SECONDARY = {
    "excess_zeroed": ("all_slots_zeroed", "live"),
    "excess_memory_off": ("memory_off", "live"),
    "excess_bos_off": ("all_slots_resample_bos_off", "live_bos_off"),
}


def reach_readout(rec: dict, traj_zeroed: dict | None = None) -> dict:
    """A4, per band: excess = mean(live_ok - all_slots_resample_ok), paired, on
    evicted targets with the baseline applied and no duplicated key; the
    secondaries; the baseline coverage and the donor-has-object rate."""
    out: dict[str, Any] = {}
    g = rec["gap"]
    evicted = (rec["own_status"] == _loo().OWN_EVICTED) & ~_flag(rec, "dup_key_in_doc")
    applied = _applied(rec["all_slots_resample_ok"])
    for band, (lo, hi) in BANDS.items():
        inband = (g >= lo) & (g <= hi) & evicted
        pop = inband & applied
        b: dict[str, Any] = {
            "n": int(pop.sum()),
            "n_evicted": int(inband.sum()),
            "coverage": int(pop.sum()) / max(int(inband.sum()), 1),
            "donor_has_object_rate": (
                float(_flag(rec, "all_donor_has_object")[pop].float().mean())
                if int(pop.sum())
                else None
            ),
            "live.acc": float(rec["live_ok"][pop].mean()) if int(pop.sum()) else None,
            "base.acc": (
                float(rec["all_slots_resample_ok"][pop].mean())
                if int(pop.sum())
                else None
            ),
        }
        b["excess.acc"] = drop_ci(rec, pop, "all_slots_resample", "acc")
        b["excess.nll16"] = drop_ci(rec, pop, "all_slots_resample", "nll16")
        for name, (knock, live) in REACH_SECONDARY.items():
            b[f"{name}.acc"] = drop_ci(rec, pop, knock, "acc", live=live)
        b["excess_no_donor_object.acc"] = drop_ci(
            rec, pop & ~_flag(rec, "all_donor_has_object"), "all_slots_resample", "acc"
        )
        if traj_zeroed is not None:
            n = torch.ones(int(pop.sum()), dtype=torch.float64)
            tz = {
                "live": rec["live_ok"][pop],
                "base": traj_zeroed["ok"].double()[pop],
                "n": n,
            }
            b["excess_traj.acc"] = cluster_boot(
                rec["doc_id"][pop], tz, lambda s: (s["live"] - s["base"]) / s["n"]
            )
        out[band] = b
    d = out[DECISION_BAND]
    out["decidable"] = d["coverage"] >= RESAMPLE_MIN_COVERAGE
    out["carry"] = carry(d["excess.acc"])
    return out


def carry(ci: dict) -> bool:
    return bool(ci["point"] >= REACH_DELTA and ci["lo"] == ci["lo"] and ci["lo"] > 0)


def classify(per_seed: dict[int, dict], controls_ok: bool) -> dict:
    """The PREREG table at ckpt 3000. `per_seed[s]` has "carry" (bool), "L_label",
    "S_label"."""
    if not controls_ok or set(per_seed) != set(SEEDS):
        return {
            "outcome": "inconclusive",
            "exit": int(Exit.DID_NOT_RUN),
            "carry_seeds": [],
        }
    carriers = [s for s in SEEDS if per_seed[s]["carry"]]
    if not carriers and any(per_seed[s]["L_label"] == "L_INCONCLUSIVE" for s in SEEDS):
        # Amendment 2 ruling 3: the outcome would rest on an L that is inconclusive
        return {
            "outcome": "inconclusive",
            "exit": int(Exit.DID_NOT_RUN),
            "carry_seeds": [],
        }
    if len(carriers) == len(SEEDS):
        outcome = "CARRIED"
    elif carriers:
        outcome = "MIXED"
    elif all(per_seed[s]["L_label"] == "LOCALISED" for s in SEEDS) and not any(
        per_seed[s]["S_label"] == "NOT_SPECIFIC" for s in SEEDS
    ):
        outcome = "LOCALISED"
    else:
        outcome = "UNCLASSIFIED"
    return {"outcome": outcome, "exit": int(Exit.OK), "carry_seeds": carriers}


# --------------------------------------------------------------------------- #
# one seed x checkpoint
# --------------------------------------------------------------------------- #


def _concat(recs: list[dict]) -> dict:
    keys = [k for k, v in recs[0].items() if isinstance(v, torch.Tensor)]
    return {k: torch.cat([r[k] for r in recs]) for k in keys}


def _bitexact(r, a_live, set_name):
    """Control 3 on `live` only. Amendment 1 A5: `*_bos_off` too "if S0-03 has
    that condition" -- it does not (S003.CONDITIONS), and answer_readout's
    cond="live_bos_off" would switch bos off on every step, changing the memory
    trajectory, whereas loo's live_bos_off is single-step: not the same quantity."""
    return [dict(set=set_name, cond="live", **bitexact_live(r, a_live))]


def measure_one(
    model, sets: dict, seed: int, ref_summary: dict, loo_readout=None, answer_readout=None
) -> dict:
    """Every control and readout for one (seed, checkpoint)."""
    if loo_readout is None:
        loo_readout = _loo().loo_readout
    if answer_readout is None:
        answer_readout = S003.answer_readout
    vmap = sets["vmap"]
    sym = sym_ids_of(vmap)
    out: dict[str, Any] = {"controls": {}}

    def ar(ids, mask, tmask, gap, cond):
        return answer_readout(model, ids, mask, tmask, gap, sym, cond=cond)

    # H64: reproduction + LOO
    ids, mask, tmask, gap = encode_set(sets["H64"], vmap)
    ar_live, ar_zero = (
        ar(ids, mask, tmask, gap, "live"),
        ar(ids, mask, tmask, gap, "slots_zeroed"),
    )
    summ = {"live": S003._summarise(ar_live), "slots_zeroed": S003._summarise(ar_zero)}
    out["h64_summary"] = summ
    out["controls"]["reproduction"] = {
        c: compare_summary(ref_summary[c], summ[c]) for c in summ
    }
    h = loo_readout(model, sets["H64"], ids, mask, tmask, gap, sym, seed=seed)
    out["controls"]["bitexact"] = _bitexact(h, ar_live, "H64")
    h_traj = {"ok": ar_zero["ok"]}
    # EXT: four calls of EXT_CHUNK consecutive ids; donors come from the call
    ext_recs, ext_traj = [], []
    for k in range(0, len(sets["EXT"]), EXT_CHUNK):
        chunk = sets["EXT"][k : k + EXT_CHUNK]
        ids, mask, tmask, gap = encode_set(chunk, vmap)
        r = loo_readout(model, chunk, ids, mask, tmask, gap, sym, seed=seed)
        out["controls"]["bitexact"] += _bitexact(
            r,
            ar(ids, mask, tmask, gap, "live"),
            f"EXT[{chunk[0].doc_id},{chunk[-1].doc_id + 1})",
        )
        ext_recs.append(r)
        ext_traj.append(ar(ids, mask, tmask, gap, "slots_zeroed")["ok"])
    e = _concat(ext_recs)
    e_traj = {"ok": torch.cat(ext_traj)}
    out["controls"]["residency"] = {"H64": residency(h), "EXT": residency(e)}
    Lh = l_full(h)
    out["H64"] = {"L": Lh, "S": s_readout(h), "reach": reach_readout(h, h_traj)}
    out["EXT"] = {"L": l_full(e), "S": s_readout(e), "reach": reach_readout(e, e_traj)}
    out["controls"]["resample_coverage"] = {
        "ok": Lh["resample_coverage"] >= RESAMPLE_MIN_COVERAGE,
        "value": Lh["resample_coverage"],
    }
    out["controls"]["reach_coverage"] = {
        "ok": bool(out["EXT"]["reach"]["decidable"]),
        "value": out["EXT"]["reach"][DECISION_BAND]["coverage"],
    }
    out["records"] = {"H64": h, "EXT": e}
    out["controls_ok"] = controls_ok(out["controls"])
    return out


def controls_ok(c: dict) -> bool:
    return bool(
        all(v["ok"] for v in c["reproduction"].values())
        and all(b["ok"] for b in c["bitexact"])
        and all(v["ok"] for v in c["residency"].values())
        and c["resample_coverage"]["ok"]
        and c["reach_coverage"]["ok"]
        and c.get("ledger_samples", {"ok": True})["ok"]
    )


# --------------------------------------------------------------------------- #
# ledger rows
# --------------------------------------------------------------------------- #


def _ci_value(ci: dict) -> dict:
    return {k: ci[k] for k in ("point", "lo", "hi", "n", "n_docs", "nonfinite")}


def _is_ci(v) -> bool:
    return isinstance(v, dict) and "point" in v and "lo" in v


def write_rows(led, per: dict[int, dict[int, dict]]) -> None:
    """Per seed x ckpt x set x readout: one observation per CI'd quantity
    (`s<seed>.ckpt<c>.<set>.<L|S|reach[.band]>.<name>`), one `.population` row of
    the scalars, one `.labels` row; across seeds, a statistic of the point values."""
    across: dict[str, dict[int, float]] = {}
    for seed, by_ckpt in per.items():
        for ckpt, r in by_ckpt.items():
            how = f"{EXPERIMENT} seed {seed} ckpt {ckpt}; loo_readout (eng/loo)"
            for set_ in ("H64", "EXT"):
                parts = {"L": r[set_]["L"], "S": r[set_]["S"]}
                for band in BANDS:
                    parts[f"reach.{band}"] = r[set_]["reach"][band]
                for name, d in parts.items():
                    scal = {}
                    for k, v in d.items():
                        key = f"s{seed}.ckpt{ckpt}.{set_}.{name}.{k}"
                        if _is_ci(v):
                            led.note(key, _ci_value(v), how=how)
                            across.setdefault(key.split(".", 1)[1], {})[seed] = v["point"]
                        elif not k.startswith("label"):
                            scal[k] = v
                    led.note(
                        f"s{seed}.ckpt{ckpt}.{set_}.{name}.population", scal, how=how
                    )
                led.note(
                    f"s{seed}.ckpt{ckpt}.{set_}.labels",
                    {
                        "L": r[set_]["L"]["label"],
                        "L_primary": r[set_]["L"]["label_primary"],
                        "L_sensitivity": r[set_]["L"]["label_sensitivity"],
                        "S": r[set_]["S"]["label"],
                        "carry": r[set_]["reach"]["carry"],
                        "decidable": r[set_]["reach"]["decidable"],
                    },
                    how=how,
                )
    for key, by_seed in across.items():
        if len(by_seed) >= 2:
            led.stat(
                key + ".point",
                [by_seed[s] for s in sorted(by_seed)],
                how=f"{EXPERIMENT}; per-seed points in seed order {sorted(by_seed)}",
            )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def manifest(source: Path, shas: dict) -> dict:
    return {
        "run_id": RUN_ID,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "amendment2_commit": AMENDMENT2_COMMIT,
        "L_SENSITIVITY_MAX_DIFF": L_SENSITIVITY_MAX_DIFF,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "source": str(source),
        "checkpoint_sha256": shas,
        "seeds": SEEDS,
        "checkpoints": list(CHECKPOINTS),
        "classification_checkpoint": FINAL,
        "sets": {"H64": list(H64), "EXT": list(EXT), "EXT_chunk": EXT_CHUNK},
        "forbidden": {k: list(v) for k, v in FORBIDDEN.items()},
        "thresholds": {
            "L_LOCALISED_LO": L_LOCALISED_LO,
            "L_DIFFUSE_HI": L_DIFFUSE_HI,
            "S_SPECIFIC_HI": S_SPECIFIC_HI,
            "S_NOT_SPECIFIC_LO": S_NOT_SPECIFIC_LO,
            "S_MIN_COVERAGE": S_MIN_COVERAGE,
            "REACH_DELTA": REACH_DELTA,
            "RESAMPLE_MIN_COVERAGE": RESAMPLE_MIN_COVERAGE,
            "REPRO_TOL": REPRO_TOL,
        },
        "bootstrap": {"R": BOOT_R, "seed": BOOT_SEED, "q": list(BOOT_Q)},
        "bands": {k: list(v) for k, v in BANDS.items()},
        "decision_band": DECISION_BAND,
        "M": M,
        "device": DEVICE,
        "torch_threads": THREADS,
        "torch": torch.__version__,
        "s003_config": S003.CONFIG,
        "trains": False,
    }


def _json_safe(o):
    if isinstance(o, torch.Tensor):
        return None
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items() if k != "records"}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


def main(argv: list[str] | None = None) -> int:
    import ledger as ledger_mod

    p = ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--render-results", action="store_true")
    a = p.parse_args(argv)
    runs = ROOT / "runs" / a.run_id
    if a.render_results:
        path = runs / "ledger.json"
        if not path.exists():
            print(f"DID NOT RUN: {path} does not exist", file=sys.stderr)
            return Exit.DID_NOT_RUN
        (ROOT / "experiments" / "carry-forward" / "RESULTS.md").write_text(
            render_results(json.loads(path.read_text()))
        )
        return Exit.OK
    source = a.source or default_source()
    if runs.exists():
        print(f"DID NOT RUN: {runs} exists; move it aside", file=sys.stderr)
        return Exit.DID_NOT_RUN
    torch.set_num_threads(THREADS)
    t0 = time.time()

    # control 1: identity (before the manifest, so the shas are frozen in it)
    shas, missing = {}, []
    for s in SEEDS:
        for c in CHECKPOINTS:
            f = ckpt_path(source, s, c)
            if not f.exists():
                missing.append(str(f))
                continue
            shas[f"seed{s}/ckpt{c}"] = sha256_file(f)
    led = ledger_mod.Ledger(a.run_id, question=QUESTION)
    led.command(
        " ".join(
            ["uv run python", EXPERIMENT, *(argv if argv is not None else sys.argv[1:])]
        ),
        exit_code=None,
        note="this run",
    )
    led.manifest(manifest(source, shas))
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=[],
        steps_requested=len(SEEDS) * len(CHECKPOINTS),
        steps_done=0,
    )
    led.note(
        "steps_semantics",
        "steps_* count (seed, checkpoint) measurements; this run trains nothing",
        how=EXPERIMENT,
    )
    raw: dict[str, Any] = {"per": {}, "controls": {}, "error": None}
    per: dict[int, dict[int, dict]] = {}
    ok_all = True
    done = 0
    try:
        if missing:
            raise FileNotFoundError(f"checkpoints missing: {missing}")
        dis = disjointness(ext_ids())
        raw["controls"]["disjointness"] = dis
        led.note("control.disjointness", dis, how="disjointness(ext_ids())")
        if not dis["ok"]:
            raise ValueError(f"EXT not disjoint: {dis}")
        ref_raw = json.loads((source / "raw.json").read_text())
        ref_led = json.loads((source / "ledger.json").read_text())
        led.note(
            "reference_sha256",
            {
                "raw.json": sha256_file(source / "raw.json"),
                "ledger.json": sha256_file(source / "ledger.json"),
            },
            how="sha256 of fresh-stream's raw.json and ledger.json",
        )
        for s in SEEDS:
            sets = seed_sets(s)
            led.note(f"control.closure.seed{s}", sets["closure"], how="seed_sets()")
            for c in CHECKPOINTS:
                f = ckpt_path(source, s, c)
                stored = CSC.RC.stored_step(f)
                if stored != c:
                    raise ValueError(f"{f} stores step {stored}, measured as {c}")
                model = load_model(f, sets["V"])
                ref = ref_raw["arms"]["B"]["per"][str(s)][str(c)]["s003"]["heldout"]
                ref_summary = {k: ref[k] for k in ("live", "slots_zeroed")}
                print(
                    f"seed {s} ckpt {c}: measuring ({time.time() - t0:.0f}s)", flush=True
                )
                r = measure_one(model, sets, s, ref_summary)
                if c == FINAL:
                    r["controls"]["ledger_samples"] = compare_ledger_samples(
                        ref_led, s, r["h64_summary"]
                    )
                r["controls_ok"] = controls_ok(r["controls"])
                ok_all = ok_all and r["controls_ok"]
                led.note(
                    f"control.s{s}.ckpt{c}",
                    _json_safe(r["controls"]),
                    how="measure_one controls 2, 3, 5, 6",
                )
                led.note(
                    f"s{s}.ckpt{c}.H64.reproduced",
                    _json_safe(r["h64_summary"]),
                    how="S0-03 answer_readout + _summarise on H64 (control 2)",
                )
                torch.save(
                    r.pop("records"),
                    ROOT / "runs" / a.run_id / f"records-s{s}-ckpt{c}.pt",
                )
                per.setdefault(s, {})[c] = r
                raw["per"].setdefault(str(s), {})[str(c)] = _json_safe(r)
                done += 1
                print(
                    f"seed {s} ckpt {c}: controls_ok={r['controls_ok']} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    except Exception as e:  # a measurement that raised -> inconclusive
        raw["error"] = f"{type(e).__name__}: {e}"
        ok_all = False
        print(f"ERROR: {raw['error']}", file=sys.stderr, flush=True)

    seeds_done = sorted(s for s in per if all(c in per[s] for c in CHECKPOINTS))
    led.run_meta(seeds_actually_run=seeds_done, steps_done=done)
    if per:
        write_rows(led, per)
    final = {
        s: {
            "carry": per[s][FINAL]["EXT"]["reach"]["carry"],
            "L_label": per[s][FINAL]["H64"]["L"]["label"],
            "S_label": per[s][FINAL]["H64"]["S"]["label"],
        }
        for s in per
        if FINAL in per[s]
    }
    cls = classify(final, ok_all and raw["error"] is None)
    raw["classification"] = cls
    raw["final_labels"] = final
    led.note("classification", {**cls, "per_seed": final}, how="classify() at ckpt 3000")
    led.note("controls_ok", bool(ok_all and raw["error"] is None), how="every control")
    led.note("elapsed_s", round(time.time() - t0, 1), how="wall clock of this process")
    if raw["error"]:
        led.note("error", raw["error"], how="exception text")
    outcome = (
        "inconclusive"
        if cls["outcome"] == "inconclusive"
        else ("survived" if cls["carry_seeds"] else "falsified")
    )
    led.verdict(
        falsifier=FALSIFIER,
        outcome=outcome,
        detail=f"classification {cls['outcome']}; CARRY seeds {cls['carry_seeds']}",
    )
    led.status("ok" if raw["error"] is None and done == 6 else "partial")
    (runs / "raw.json").write_text(json.dumps(_json_safe(raw), indent=2) + "\n")
    path = led.write()
    print(f"ledger: {path}; classification {cls['outcome']}; exit {cls['exit']}")
    return Exit.OK if cls["exit"] == int(Exit.OK) else Exit.DID_NOT_RUN


# --------------------------------------------------------------------------- #
# RESULTS.md, from the ledger only
# --------------------------------------------------------------------------- #


def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _ci(v: dict, nd=4) -> str:
    return f"{_f(v['point'], nd)} [{_f(v['lo'], nd)}, {_f(v['hi'], nd)}]"


def render_results(doc: dict) -> str:
    rows = {r["key"]: r for r in doc["rows"]}

    def val(k):
        return rows[k]["value"] if k in rows else None

    def ci(k):
        v = val(k)
        return _ci(v) if v else "—"

    cls = val("classification")
    prov = doc["provenance"]
    out = [
        "<!-- GENERATED by experiments/carry-forward/run.py --render-results from "
        "runs/carry-forward/ledger.json. Do not type a number here. -->",
        "",
        "# carry-forward — RESULTS",
        "",
        f"- PREREG: `{PREREG}` (commit {PREREG_COMMIT}; Amendments 1 "
        f"{AMENDMENT_COMMIT} and 2 {AMENDMENT2_COMMIT} appended before any "
        f"readout). Run SHA: `{prov.get('git_sha')}`, dirty {prov.get('dirty')}. "
        f"Hardware: {prov.get('platform')}, device {doc['device']}, torch threads "
        "per the manifest.",
        f"- Command: `{doc['commands'][0]['argv']}`. Status `{doc['status']}`. "
        f"Config hash `{doc['config_hash']}`.",
        f"- Classification (`classification.outcome`): **{cls['outcome']}**. "
        f"`classification.carry_seeds`: {cls['carry_seeds']}. "
        f"Verdict `{doc['verdict']['outcome']}`.",
        f"- `controls_ok`: {val('controls_ok')}. `elapsed_s` {val('elapsed_s')}.",
        "",
        "Cells are point [95% per-document bootstrap CI]. Accuracy is primary. Key "
        "prefix `s<seed>.ckpt<c>.<set>`; every cell is the named ledger key under it.",
        "",
    ]
    cells = [p for p in (f"s{s}.ckpt{c}" for s in SEEDS for c in CHECKPOINTS)]
    for set_ in ("H64", "EXT"):
        out += [
            f"## {set_}: localisation and specificity",
            "",
            "| prefix | L.L.acc | L.denominator.acc | L.L_sensitivity.acc | "
            "L.L_zero.acc (OOD) | L.L_bos_off.acc | L.L_gap1_bos_off.acc | "
            "L.memory_off_drop.acc | S.ratio.acc | "
            "S.drop_own.acc | S.drop_ctrl.acc | S.L_on_S | labels |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in cells:
            p = f"{c}.{set_}"
            lab = val(f"{p}.labels")
            if lab is None:
                continue
            out.append(
                f"| `{p}` | {ci(p + '.L.L.acc')} | {ci(p + '.L.denominator.acc')} | "
                f"{ci(p + '.L.L_sensitivity.acc')} | "
                f"{ci(p + '.L.L_zero.acc')} | {ci(p + '.L.L_bos_off.acc')} | "
                f"{ci(p + '.L.L_gap1_bos_off.acc')} | "
                f"{ci(p + '.L.memory_off_drop.acc')} | {ci(p + '.S.ratio.acc')} | "
                f"{ci(p + '.S.drop_own.acc')} | {ci(p + '.S.drop_ctrl.acc')} | "
                f"{ci(p + '.S.L_on_S')} | L {lab['L']}, S {lab['S']} |"
            )
        out += [
            "",
            "| prefix | L.population.resample_coverage | S.population.coverage |",
            "|---|---|---|",
        ]
        for c in cells:
            p = f"{c}.{set_}"
            lp, sp = val(f"{p}.L.population"), val(f"{p}.S.population")
            if lp is None:
                continue
            out.append(
                f"| `{p}` | {_f(lp['resample_coverage'])} | {_f(sp['coverage'])} |"
            )
        out += [
            "",
            f"## {set_}: reach (evicted; excess over all_slots_resample)",
            "",
            "| prefix | "
            + " | ".join(f"reach.{b}.excess.acc" for b in BANDS)
            + " | reach.17_40.coverage | CARRY |",
            "|---|" + "---|" * (len(BANDS) + 2),
        ]
        for c in cells:
            p = f"{c}.{set_}"
            lab = val(f"{p}.labels")
            if lab is None:
                continue
            cov = val(f"{p}.reach.17_40.population")["coverage"]
            out.append(
                f"| `{p}` | "
                + " | ".join(ci(f"{p}.reach.{b}.excess.acc") for b in BANDS)
                + f" | {_f(cov)} | {lab['carry']} |"
            )
        out += [
            "",
            "Secondary reach baselines, pooled band 17_40 (not classified):",
            "",
            "| prefix | excess_zeroed.acc | excess_memory_off.acc | "
            "excess_bos_off.acc | excess_traj.acc | excess_no_donor_object.acc | "
            "excess.nll16 |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in cells:
            p = f"{c}.{set_}.reach.17_40"
            if val(f"{p}.excess.acc") is None:
                continue
            out.append(
                f"| `{p}` | {ci(p + '.excess_zeroed.acc')} | "
                f"{ci(p + '.excess_memory_off.acc')} | {ci(p + '.excess_bos_off.acc')} "
                f"| {ci(p + '.excess_traj.acc')} | "
                f"{ci(p + '.excess_no_donor_object.acc')} | {ci(p + '.excess.nll16')} |"
            )
        out.append("")
    out += [
        "Band CIs are uncorrected across the 4 x 3 cells; the decision is the pooled "
        "17_40 EXT band only (PREREG). Wrong-looking numbers are printed as measured.",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    run_main(main)
