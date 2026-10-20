"""Recompute decisive-shuffle / fresh-stream / fresh-escape ledger rows from their
committed ``runs/<id>/raw.json`` through each run's OWN ledger-writing path
(``write_ledger`` / ``verdict`` + ``write_rows``), with a capturing stand-in for
``scripts/ledger.py``'s ``Ledger`` (same ``stat`` arithmetic, nothing written).

Prints one JSON line ``{key: value}``: a statistic row's value is its ``samples``
list, an observation's its ``value``; ``__verdict__`` is the ``Ledger.verdict``
call if the run makes one.

Usage: recompute_from_raw.py --run decisive-shuffle|fresh-stream|fresh-escape

Reads: the committed raw.json (sha256-equal to the archive copy); fresh-stream also
reads the committed ``runs/scaffold-timing/ledger.json`` (its REF_LEDGER) and the
archived heartbeats ``~/rsr-runs-archive/fresh-stream/runs/fresh-stream/<arm>/seed<s>/
heartbeat.jsonl`` for the stream-loss windows. Writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RUNS = {
    "decisive-shuffle": "experiments/decisive-shuffle/run.py",
    "fresh-stream": "experiments/fresh-stream/run.py",
    "fresh-escape": "experiments/fresh-escape/run.py",
}


class CaptureLedger:
    """The subset of ``Ledger`` the three runs' row writers call."""

    def __init__(self) -> None:
        self.doc: dict = {"rows": [], "verdict": None, "commands": []}

    def note(self, key, value, *, how):
        self.doc["rows"].append({"key": key, "kind": "observation", "value": value})

    def stat(self, key, samples, *, how, allow_single=False):
        xs = [float(x) for x in samples]
        mean = sum(xs) / len(xs)
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
            if len(xs) > 1
            else None
        )
        row = {"key": key, "kind": "statistic", "samples": xs, "mean": mean, "sd": sd}
        self.doc["rows"].append(row)
        return row

    def verdict(self, *, falsifier, outcome, detail):
        self.doc["verdict"] = {"falsifier": falsifier, "outcome": outcome, "detail": detail}

    def command(self, *a, **k):
        pass

    def run_meta(self, *a, **k):
        pass

    def status(self, *a, **k):
        pass

    def write(self):
        return None


def intkeys(x):
    """raw.json stringified the int keys (seed, checkpoint); restore them."""
    if isinstance(x, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): intkeys(v)
            for k, v in x.items()
        }
    if isinstance(x, list):
        return [intkeys(v) for v in x]
    return x


def load(run: str):
    name = "_cr_raw_" + run.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, C.ROOT / RUNS[run])
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(RUNS))
    a = ap.parse_args(argv)
    M = load(a.run)
    raw = intkeys(C.jload(C.ROOT / "runs" / a.run / "raw.json"))
    led = CaptureLedger()
    extra: dict = {}
    if a.run == "decisive-shuffle":
        M.write_ledger(raw, led)
    elif a.run == "fresh-stream":
        ref = C.ROOT / M.REF_LEDGER
        extra["__ref_ledger_sha256__"] = hashlib.sha256(ref.read_bytes()).hexdigest()
        start = M.scaffold_start(C.jload(ref))
        raw["root"] = str(C.ARCHIVE / "fresh-stream" / "runs" / "fresh-stream")
        v = M.verdict(raw)
        M.write_rows(led, raw, v, start)
    else:  # fresh-escape
        v = M.verdict(raw)
        M.write_rows(led, raw, v)
    out = {}
    for r in led.doc["rows"]:
        out[r["key"]] = r["samples"] if r["kind"] == "statistic" else r["value"]
    out["__verdict__"] = led.doc["verdict"]
    out.update(extra)
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
