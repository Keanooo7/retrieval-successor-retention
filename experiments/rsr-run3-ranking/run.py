"""RSR run 3, the ranking-loss test (phase 2): a thin dispatcher.

PREREG ``experiments/rsr-run3-ranking/PREREG.md`` (committed alone at 32475b5). Two run
ids, two modules beside this file, imported lazily by path::

    run.py 2a <stage> ...    # -> phase2a.main([...])  run id rsr-run3-2a
    run.py 2b <stage> ...    # -> phase2b.main([...])  run id rsr-run3-2b

2a stages: ``launch-checks``, ``fit --seed s (--rc2 | --omega w)``, ``select --seed s``,
``deploy --seed s``, ``finalize``, ``time`` (Amendment 2a-S), ``recompute``.
2b stages: ``plan``, ``launch-checks``, ``measure --seed s``, ``finalize``, ``recompute``
(training is ``launch_2b.zsh``'s; the smoke is ``SMOKE=1`` on it).

Every stage is its own process with its own R-SEAL (``stage_seal.py``, PREREG §1).
Exit codes (``rsr.exit_codes``, PREREG §6): 0 ran and every control passed; 1 a defect
after the controls passed; 3 did not run, or a control failed. Exit 5 is not used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rsr.exit_codes import Exit, did_not_run, run_main, status  # noqa: E402

RUNS = {"2a": "phase2a", "2b": "phase2b"}
USAGE = "usage: run.py {2a,2b} <stage> ..."


def load(run: str):
    name = RUNS[run]
    key = f"run3_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def main(argv: list[str] | None = None) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, file=sys.stderr)
        return Exit.OK if argv else Exit.DID_NOT_RUN
    if argv[0] not in RUNS:
        return did_not_run(f"unknown run {argv[0]!r}. {USAGE}")
    return status(load(argv[0]).main(argv[1:]))


if __name__ == "__main__":
    run_main(main)
