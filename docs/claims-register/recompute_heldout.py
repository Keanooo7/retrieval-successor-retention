"""Recompute run 1's / run 2's held-out statistics for one seed from the per-document
records, through run 1's own analysis path (``analyse_heldout``: the paired
per-document bootstrap, B2 §9.2, seeded ``20260927 + seed``), and print every leaf as
one JSON line ``{"seed<s>.heldout.<dotted leaf>": value}`` -- the same flattening as
the ledger's ``seed<s>.heldout.*`` keys.

Usage: recompute_heldout.py --run rsr-run1|rsr-run2 --seed S

Reads only ``~/rsr-runs-archive/<run>-run/runs/<run>/`` (records + measure-seed config);
writes nothing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

RUN_MODULE = {
    "rsr-run1": "experiments/rsr-run1/run.py",
    "rsr-run2": "experiments/rsr-run1/run.py",  # run 2 reuses run 1's analyse_heldout
}


def _load(rel: str):
    spec = importlib.util.spec_from_file_location("_cr_r1", C.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_cr_r1"] = m
    spec.loader.exec_module(m)
    return m


def flatten(x, prefix: str, out: dict) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUN_MODULE))
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    root = C.ARCHIVE / f"{a.run}-run" / "runs" / a.run
    hd = root / "heldout" / f"seed{a.seed}"
    ms = C.jload(root / f"measure-seed{a.seed}.json")
    cfgd = ms["config"]
    R1 = _load(RUN_MODULE[a.run])
    diag = cfgd.get("diag_step")
    cfg = SimpleNamespace(
        seed=a.seed,
        n_boot=int(cfgd["n_boot"]),
        memory_slots=int(cfgd["memory_slots"]),
        # Run1Config.diag: the A4.2 DIAGNOSTIC step, read from the run's own config.
        diag=None if diag in (None, "None", "") else int(diag),
    )
    names = sorted(p.name[: -len(".jsonl.gz")] for p in hd.glob("*.jsonl.gz"))
    out = R1.analyse_heldout(hd, cfg, [(n, None, None) for n in names])
    flat: dict = {}
    flatten(out, f"seed{a.seed}.heldout", flat)
    print(json.dumps(flat, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
