"""Recompute run2-vs-random's per-seed statistics from the archived per-document
records through that experiment's own ``analyse_seed`` (run 1's paired per-document
bootstrap, 2000 replicates, seed 20260927 + seed), and print the accuracy and
residency leaves as one JSON line ``{"seeds.<s>.<dotted leaf>": value}`` -- the same
paths as the committed ``experiments/run2-vs-random/results.json``.

Usage: recompute_run2_vs_random.py --seed S

Reads only ``~/rsr-runs-archive/rsr-run2-run/runs/rsr-run2`` and
``~/rsr-runs-archive/rsr-run1-run/runs/rsr-run1``; writes nothing (``main`` is never
called, so the experiment's RESULTS/results.json writer never runs).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

RUN2 = C.ARCHIVE / "rsr-run2-run" / "runs" / "rsr-run2"
RUN1 = C.ARCHIVE / "rsr-run1-run" / "runs" / "rsr-run1"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_cr_r2vr", C.ROOT / "experiments/run2-vs-random/run.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_cr_r2vr"] = m
    spec.loader.exec_module(m)
    return m


def flatten(x, prefix: str, out: dict) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}.{k}", out)
    else:
        out[prefix] = x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    m = _load()
    res = m.analyse_seed(RUN2, RUN1, a.seed, a.n_boot)
    if res["pairing_failures"]:
        print(f"pairing failures: {res['pairing_failures']}", file=sys.stderr)
        return 3
    flat: dict = {}
    for part in ("accuracy", "residency"):
        flatten(res[part], f"seeds.{a.seed}.{part}", flat)
    print(json.dumps(flat, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
