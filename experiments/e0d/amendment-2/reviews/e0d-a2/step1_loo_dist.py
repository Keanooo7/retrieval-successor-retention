"""E0D-A2 step 1: the real LOO Delta distribution on set E, seed 0, arm B ckpt3000.

Scope (brief E0D-A2): set E only, docs [64, 128) = 64 documents (the first 64 of E =
[64, 1088), already inspected). Seed 0 only. LOO Delta only -- NO r_i is computed
(no capture forward), so no r_i-vs-LOO agreement is read. The reserved range
[262144, 263168) is asserted untouched.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

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

SEED = 0
LO, HI = 64, 128
assert HI - LO <= 64
assert HI <= m.D_E0D[0] or LO >= m.D_E0D[1], "reserved range"
assert LO >= 64 and HI <= 1088, "set E only"

seed_dir = m.DEFAULT_CKPT_ROOT / f"seed{SEED}"
path = seed_dir / m.CKPT_NAME
sha = m.sha256(path)
assert sha == m.CKPT_SHA256[SEED], sha
config = m.header_config(seed_dir)
cfg = TGConfig(**config["tg"])
M = cfg.max_sentences_in_short_term
S = int(config["steps_per_stream"])
model = TGModel(cfg)
ck.load(path, model=model, restore_rng=False)
model.eval()
torch.manual_seed(SEED)
vdocs = generate(SyntheticConfig(sentences_per_document=S, seed=SEED,
                                 n_documents=int(config.get("stream", {}).get("vocab_documents", 64))))
vocab = build_vocab(vdocs)
docs = m.e0d_documents(SEED, S, (LO, HI), cleared=False)  # refuses D_E0d
m.check_vocab(docs, vocab)
ids, mask = encode(docs, vocab, max_tokens=cfg.max_sentence_tokens, steps=S)

rows = {k: [] for k in ("doc", "t", "rank", "sent", "dres", "dzero", "q", "a", "gap", "tfull")}
for lo in range(0, len(docs), m.BATCH):
    sl = slice(lo, lo + m.BATCH)
    res = loo_delta_loss(model, docs[sl], ids[sl], mask[sl], mode="resample", seed=SEED)
    zer = loo_delta_loss(model, docs[sl], ids[sl], mask[sl], mode="zero", seed=SEED)
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
    print(f"batch {lo // m.BATCH + 1}/{-(-len(docs) // m.BATCH)}", flush=True)
c = {k: np.concatenate(v) for k, v in rows.items()}
np.savez(Path(__file__).with_name("step1_cells.npz"), **c)

full = c["tfull"]
Q = full & c["q"]
A = Q & c["a"]
out = {"seed": SEED, "docs": [LO, HI], "ckpt_sha256": sha, "M": M, "S": S,
       "n_full_cells": int(full.sum()), "n_Q_cells": int(Q.sum()), "n_A_cells": int(A.sum()),
       "n_Q_steps": int(Q.sum() // M)}
QS = np.quantile
for kn in ("dres", "dzero"):
    d = c[kn]
    for nm, sel in (("A", A), ("Q_offA", Q & ~A), ("N", full & ~c["q"]), ("full_offA", full & ~A)):
        v = d[sel]
        have = ~np.isnan(v)
        v = v[have]
        av = np.abs(v)
        out[f"{kn}.{nm}"] = {
            "n": int(sel.sum()), "n_with_value": int(have.sum()),
            "frac_nan": float(1 - have.mean()) if sel.sum() else None,
            "frac_exact_zero": float((v == 0).mean()) if len(v) else None,
            "frac_abs_lt_1e-6": float((av < 1e-6).mean()) if len(v) else None,
            "frac_abs_lt_1e-4": float((av < 1e-4).mean()) if len(v) else None,
            "frac_abs_lt_1e-3": float((av < 1e-3).mean()) if len(v) else None,
            "frac_abs_lt_1e-2": float((av < 1e-2).mean()) if len(v) else None,
            "n_distinct": int(len(np.unique(v))),
            "abs_q": {str(q): float(QS(av, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if len(v) else None,
            "abs_max": float(av.max()) if len(v) else None,
            "signed_median": float(np.median(v)) if len(v) else None,
            "frac_positive": float((v > 0).mean()) if len(v) else None,
        }
# how often does the A-cell carry the largest Delta of its Q-step (resample); needs no r_i
key = c["doc"] * (S + 1) + c["t"]
top_ok, top_n, a_rank_in_step = 0, 0, []
for k in np.unique(key[A]):
    sel = Q & (key == k)
    d = c["dres"][sel]; a = c["a"][sel]
    if np.isnan(d[a]).any():
        continue
    dd = np.where(np.isnan(d), -np.inf, d)
    top_n += 1
    top_ok += int(np.argmax(dd) == np.flatnonzero(a)[0])
out["A_is_step_argmax_dres"] = {"hit": top_ok, "of": top_n, "frac": top_ok / top_n if top_n else None}
# Q-steps whose assert is NOT resident: A-less Q-steps
out["n_Q_steps_with_A"] = int(len(np.unique(key[A])))
out["n_Q_steps"] = int(len(np.unique(key[Q])))
print(json.dumps(out, indent=1))
Path(__file__).with_name("step1_out.json").write_text(json.dumps(out, indent=1))
