"""E0D-AMD2-FINAL, OI-4 measured: per-seed tau on set E docs [64, 128), seeds 1 and 2.

Arm B ckpt3000 (fresh-stream), loo.py resample and zero knockouts, donor seed = model seed
(exactly as run.py measure_batch and reviews/e0d-a2/step1_loo_dist.py for seed 0).
LOO Delta ONLY: no r_i, no capture forward, no r_i-vs-LOO agreement is read.
The reserved range D_E0d = [262144, 263168) is asserted untouched (HI <= 262144).

Rule (A2.2): tau = numpy-default (linear, type 7) q-quantile of |Delta| over full-memory
Q-step cells that are not A-cells and have a value. q = 0.995 primary; 0.99, 0.999 sensitivity.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WT = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/e0d")
sys.path.insert(0, str(WT / "src"))
spec = importlib.util.spec_from_file_location("e0d_run", WT / "experiments/e0d/run.py")
m = importlib.util.module_from_spec(spec)
sys.modules["e0d_run"] = m
spec.loader.exec_module(m)

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.metrics.loo import loo_delta_loss  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import build_vocab, encode  # noqa: E402

LO, HI = 64, 128
assert HI <= 262144, "HI must stay below D_E0d"
assert HI <= m.D_E0D[0], "reserved range"
assert (m.D_E0D[0], m.D_E0D[1]) == (262144, 263168)
assert LO >= 64 and HI <= 1088, "set E only"
QS = (0.99, 0.995, 0.999)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def measure(seed):
    seed_dir = m.DEFAULT_CKPT_ROOT / f"seed{seed}"
    path = seed_dir / m.CKPT_NAME
    sha = m.sha256(path)
    assert sha == m.CKPT_SHA256[seed], sha
    config = m.header_config(seed_dir)
    cfg = TGConfig(**config["tg"])
    M = cfg.max_sentences_in_short_term
    S = int(config["steps_per_stream"])
    model = TGModel(cfg)
    ck.load(path, model=model, restore_rng=False)
    model.eval()
    torch.manual_seed(seed)
    vdocs = generate(SyntheticConfig(sentences_per_document=S, seed=seed,
                                     n_documents=int(config.get("stream", {}).get("vocab_documents", 64))))
    vocab = build_vocab(vdocs)
    docs = m.e0d_documents(seed, S, (LO, HI), cleared=False)  # refuses D_E0d
    m.check_vocab(docs, vocab)
    ids, mask = encode(docs, vocab, max_tokens=cfg.max_sentence_tokens, steps=S)
    # doc index is 0..63 relative to LO, as in step1_loo_dist.py
    rows = {k: [] for k in ("doc", "t", "rank", "sent", "dres", "dzero", "q", "a", "gap", "tfull")}
    for lo in range(0, len(docs), m.BATCH):
        sl = slice(lo, lo + m.BATCH)
        res = loo_delta_loss(model, docs[sl], ids[sl], mask[sl], mode="resample", seed=seed)
        zer = loo_delta_loss(model, docs[sl], ids[sl], mask[sl], mode="zero", seed=seed)
        ds = m.doc_structure(docs[sl], S)
        idx = torch.nonzero(~zer["delta"].isnan())
        b, t, i = idx[:, 0].numpy(), idx[:, 1].numpy(), idx[:, 2].numpy()
        s = zer["slot_sentence"][idx[:, 0], idx[:, 1], idx[:, 2]].numpy()
        qs = ds["q_step"][b, t]
        aof = ds["assert_of"][b, t]
        nlive = (zer["slot_sentence"][idx[:, 0], idx[:, 1]] >= 0).sum(-1).numpy()
        rows["doc"].append(b + lo); rows["t"].append(t); rows["rank"].append(i); rows["sent"].append(s)
        rows["dres"].append(res["delta"][idx[:, 0], idx[:, 1], idx[:, 2]].numpy())
        rows["dzero"].append(zer["delta"][idx[:, 0], idx[:, 1], idx[:, 2]].numpy())
        rows["q"].append(qs); rows["a"].append(qs & (aof == s)); rows["gap"].append(np.where(qs, t - aof, -1))
        rows["tfull"].append(nlive == M)
        print(f"seed {seed} batch {lo // m.BATCH + 1}/{-(-len(docs) // m.BATCH)}", flush=True)
    c = {k: np.concatenate(v) for k, v in rows.items()}
    out_p = HERE / f"tau_cells_seed{seed}.npz"
    np.savez(out_p, **c)
    return out_p, sha, M, S


def tau_from(p, M, seed, ckpt_sha):
    c = dict(np.load(p))
    full = c["tfull"].astype(bool); q = c["q"].astype(bool)
    Q = full & q; A = Q & c["a"].astype(bool)
    age = c["t"] - c["sent"]
    o = {"seed": seed, "cells_file": str(p), "cells_sha256": sha256(p), "ckpt_sha256": ckpt_sha,
         "docs": [LO, HI], "n_full_cells": int(full.sum()), "n_Q_cells": int(Q.sum()),
         "n_A_cells": int(A.sum()),
         "rank_eq_M_minus_age_all_full": bool(np.all(c["rank"][full] == M - age[full]))}
    for kn in ("dres", "dzero"):
        d = c[kn]; off = Q & ~A & ~np.isnan(d); ad = np.abs(d[off])
        o[kn] = {"n": int(off.sum()), "n_exact_zero": int((ad == 0).sum()),
                 **{f"q{qq}": float(np.quantile(ad, qq)) for qq in QS}}
        # doc-cluster bootstrap of tau (2000, generator 20260927), reported only
        rng = np.random.default_rng(20260927); dd = c["doc"][off]; docs = np.unique(c["doc"])
        bydoc = [ad[dd == x] for x in docs]
        bt = [np.quantile(np.concatenate([bydoc[j] for j in rng.integers(0, len(docs), len(docs))]), 0.995)
              for _ in range(2000)]
        o[kn]["q0.995_boot95"] = [float(np.quantile(bt, .025)), float(np.quantile(bt, .975))]
        tau = o[kn]["q0.995"]; v = Q & ~np.isnan(d)
        o[kn]["crit_at_tau"] = {"Q_cells_with_value": int(v.sum()), "crit": int((v & (d > tau)).sum()),
                                "crit_A": int((v & A & (d > tau)).sum()),
                                "A_with_value": int((v & A).sum()),
                                "crit_nonA": int((v & ~A & (d > tau)).sum())}
    return o


if __name__ == "__main__":
    out = {"numpy_version": np.__version__, "torch_version": torch.__version__,
           "quantile_method": "numpy default 'linear' (type 7)", "D_E0d": list(m.D_E0D),
           "HI": HI, "HI_le_262144": HI <= 262144}
    s0 = Path("/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2/step1_cells.npz")
    out["seed0"] = tau_from(s0, 16, 0, m.CKPT_SHA256[0])
    for seed in (1, 2):
        p, sha, M, S = measure(seed)
        assert M == 16 and S == 48, (M, S)
        out[f"seed{seed}"] = tau_from(p, M, seed, sha)
    print(json.dumps(out, indent=1))
    (HERE / "m5_out.json").write_text(json.dumps(out, indent=1))
