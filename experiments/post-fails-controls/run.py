"""PREREG post-fails-controls, phase 1: a thin dispatcher.

PREREG ``experiments/post-fails-controls/PREREG.md`` (committed alone at 9f22d87),
§0: one harness, subcommands ``1a`` ... ``1e`` and ``finalize``. Each item lives in
its own module beside this file (``item_1a.py`` ... ``item_1e.py``), imported lazily
by path, so an item whose module is absent does not stop the others::

    run.py 1a measure --seed 0         # -> item_1a.main(["measure", "--seed", "0"])
    run.py 1b ...                      # -> item_1b.main([...])
    run.py finalize 1a [--out-root X]  # -> item_1a.finalize([...])

Every item module exposes ``main(argv) -> int`` and ``finalize(argv) -> int``. The
shared controls (G-C1 to G-C5) live in ``common.py``, loaded under the one module
name ``pfc_common`` (the seal is process-wide state).

Exit codes (`rsr.exit_codes`, §0): 0 ran and every control passed; 1 a defect after
the controls passed; 3 did not run (incl. an absent item module, an unknown
subcommand), or a control failed. Exit 5 is not used.
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

ITEMS = {
    "1a": "item_1a",
    "1b": "item_1b",
    "1c": "item_1c",
    "1d": "item_1d",
    "1e": "item_1e",
}
USAGE = "usage: run.py {1a,1b,1c,1d,1e} ... | run.py finalize {1a,1b,1c,1d,1e} [...]"


def load_common():
    """``common.py`` as ``pfc_common`` (the one shared instance)."""
    name = "pfc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def load_item(item: str, here: Path = HERE):
    """The item's module, or None (with the reason on stderr) if it is absent."""
    mod_name = ITEMS.get(item)
    if mod_name is None:
        print(f"DID NOT RUN: unknown item {item!r}. {USAGE}", file=sys.stderr)
        return None
    path = Path(here) / f"{mod_name}.py"
    if not path.exists():
        print(
            f"DID NOT RUN: item {item}'s module {path.name} is absent from {here} "
            f"(not built on this branch).",
            file=sys.stderr,
        )
        return None
    key = f"pfc_{mod_name}"
    if key in sys.modules:
        return sys.modules[key]
    load_common()
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None, *, here: Path = HERE) -> Exit:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, file=sys.stderr)
        return Exit.OK if argv else Exit.DID_NOT_RUN
    cmd, rest = argv[0], argv[1:]
    if cmd == "finalize":
        if not rest:
            return did_not_run(f"finalize needs an item. {USAGE}")
        mod = load_item(rest[0], here)
        if mod is None:
            return Exit.DID_NOT_RUN
        return status(mod.finalize(rest[1:]))
    if cmd in ITEMS:
        mod = load_item(cmd, here)
        if mod is None:
            return Exit.DID_NOT_RUN
        return status(mod.main(rest))
    return did_not_run(f"unknown subcommand {cmd!r}. {USAGE}")


if __name__ == "__main__":
    run_main(main)
