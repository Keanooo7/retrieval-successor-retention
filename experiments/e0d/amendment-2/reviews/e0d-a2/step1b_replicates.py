"""E0D-A2 step 1b (LOO replicates with other donor seeds; LOO only, no r_i): the real LOO Delta distribution on set E, seed 0, arm B ckpt3000.

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


reps = {}
for dseed in (1, 2):
    out = []
    for lo in range(0, len(docs), m.BATCH):
        sl = slice(lo, lo + m.BATCH)
        res = loo_delta_loss(model, docs[sl], ids[sl], mask[sl], mode="resample", seed=dseed)
        zer_nan = None
        out.append(res["delta"])
        print(f"donor seed {dseed} batch {lo // m.BATCH + 1}", flush=True)
    reps[f"dres_donor{dseed}"] = torch.cat(out).numpy()
np.savez(Path(__file__).with_name("step1b_reps.npz"), **reps)
