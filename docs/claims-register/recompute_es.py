"""Recompute the Expire-Span referendum's ledger from raw records, through the run's
own code (experiments/expire-span-referendum/run.py): ``paired_seed`` per seed (ES's
archived EVAL records vs run 1's archived psiC/fifo held-out records, run 1's resample
weights ``boot_weights(2048, s, 2000)``), ``classify`` with ``delta_per_seed()`` (parsed
from the PREREG's A1.1 table), and ``selection`` over the archived VAL summaries.

Prints ONE line of flattened JSON with the same keys as ``flat_json.py`` gives for the
committed ``runs/expire-span-referendum/ledger.json`` (e.g. ``class``,
``seeds.0.con.acc.all.es_minus_fifo.point``, ``selection.selected``).

Reads only ``~/rsr-runs-archive/es-referendum-a2-run/runs/expire-span-referendum`` and
``~/rsr-runs-archive/rsr-run1-run/runs/rsr-run1``; writes nothing (``finalize`` and
``select`` would write, so they are not called)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from flat_json import flatten  # noqa: E402

ES_OUT = C.ARCHIVE / "es-referendum-a2-run" / "runs" / "expire-span-referendum"
RUN1_OUT = C.ARCHIVE / "rsr-run1-run" / "runs" / "rsr-run1"


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "_cr_es", C.ROOT / "experiments" / "expire-span-referendum" / "run.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_cr_es"] = m
    spec.loader.exec_module(m)
    delta = m.delta_per_seed()
    per = {s: m.paired_seed(s, ES_OUT, RUN1_OUT) for s in m.SEEDS}
    cls = m.classify({s: per[s]["con"][m.HEADLINE] for s in m.SEEDS}, delta)
    summaries = {
        s: {**C.jload(ES_OUT / "val" / f"seed{s}" / "summary.json"), "seed": s}
        for s in m.SEEDS
    }
    sel = m.selection(summaries, m.grid())
    led = {
        "class": cls,
        "falsifier_6": cls == "SUPERSEDES",
        "delta": delta,
        "selection": sel,
        "seeds": per,
    }
    led = json.loads(json.dumps(led, default=str))  # the ledger's own serialisation
    out: dict = {}
    flatten(led, "", out)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
