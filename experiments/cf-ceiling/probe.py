"""B0-CF feasibility probe: wall time of the Delta measurement, at 1 thread.

Computes **no cap, no hit rate, no residency**. Measures, on 20 documents that are
NOT in the U range (the vocabulary documents ``[0, 20)`` of seed 0, so no U
document's Delta exists before the PREREG), seed 0's fresh-stream arm B
``ckpt-003000.pt`` (sha256-checked):

* the wall time of ``cfdelta.cf_delta`` on one batch of the 20 documents, split into
  loo_delta_loss's resident pass and the FIFO-world evicted probe;
* the forward-pass counts behind it;
* the resample donor coverage at batch 20, by kind;
* the internal checks (live loss and residency equal loo's; identity probe).

Usage (slot-wrapped)::

    .venv/bin/python experiments/cf-ceiling/probe.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import Exit, run_main  # noqa: E402
from rsr.metrics import loo  # noqa: E402
from rsr.train.loop import encode  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LR = _load("_cfp_lookahead", "experiments/lookahead-room/run.py")
CF = _load("_cfp_cfdelta", "experiments/cf-ceiling/cfdelta.py")

N_DOCS = 20
SEED = 0
CKPT = 3000


def main(argv: list[str] | None = None) -> Exit:
    torch.set_num_threads(1)
    src = LR.default_source()
    ck = LR.ckpt_path(src, CKPT, SEED)
    got = LR.sha256(ck)
    if got != LR.CKPT_SHA256[(CKPT, SEED)]:
        print(f"sha256 mismatch {ck}: {got}", flush=True)
        return Exit.DID_NOT_RUN
    _sets, vmap, V = LR.CSC.doc_sets(SEED, LR.VOCAB_DOCUMENTS)
    docs = LR.CSC._generate(SEED, N_DOCS)[:N_DOCS]
    assert all(d.doc_id < LR.E_SET[0] for d in docs), "probe docs must be outside U"
    model = LR.load_model(ck, V)
    ids, mask = encode(docs, vmap, max_tokens=LR.L, steps=LR.S)
    timing: dict = {}
    t0 = time.perf_counter()
    out = CF.cf_delta(model, docs, ids, mask, seed=SEED, timing=timing)
    wall = time.perf_counter() - t0
    ann = loo.stream_annotations(docs, steps=LR.S)
    kinds = {v: k for k, v in loo.KIND.items()}
    occ = out["slot_sentence"] >= 0
    cov: dict = {}
    for b in range(len(docs)):
        for t in range(LR.S):
            for j in range(LR.M):
                s = int(out["slot_sentence"][b, t, j])
                if s < 0:
                    continue
                k = kinds[int(ann["kind"][b, s])]
                c = cov.setdefault(k, [0, 0])
                c[0] += 1
                c[1] += int(bool(out["missing"][b, t, s]))
    n_res_fwd = sum(
        1 for t in range(LR.S) for j in range(LR.M) if bool(occ[:, t, j].any())
    )
    rows_res = int(occ.sum())
    report = {
        "docs": [docs[0].doc_id, docs[-1].doc_id],
        "n_docs": len(docs),
        "threads": torch.get_num_threads(),
        "ckpt": str(ck),
        "ckpt_sha256": got,
        "wall_seconds": wall,
        "timing": timing,
        "forwards": {
            "live_passes": 2 * LR.S,
            "resident_knockout_forwards": n_res_fwd,
            "resident_knockout_rows": rows_res,
            "evicted_probe_forwards": timing["probe_forwards"],
            "evicted_probe_rows": timing["probe_rows"],
        },
        "per_document_seconds": wall / len(docs),
        "per_resident_forward_seconds": timing["loo_seconds"] / (n_res_fwd + LR.S),
        "per_probe_row_seconds": timing["probe_seconds"] / timing["probe_rows"],
        "coverage_by_kind": {
            k: {"cells": v[0], "missing": v[1], "coverage": 1 - v[1] / v[0]}
            for k, v in cov.items()
        },
        "checks": out["checks"],
        "pid": os.getpid(),
    }
    print(json.dumps(report, indent=1), flush=True)
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
