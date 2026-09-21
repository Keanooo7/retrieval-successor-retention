"""The run protocol's exit codes -- one enum, and the helpers every checker uses.

Five codes, as `docs/gates.md` ("Exit codes -- the protocol") specifies them:

====  =================  ==========================================================
code  name               meaning
====  =================  ==========================================================
0     ``OK``             ran, compared, and held: at or above the floor
1     ``FAIL``           a real failure -- the thing the check exists to catch
                         (``DROP`` is an alias: a floor drop is a real failure)
2     ``UNKNOWN``        ran, but there was **nothing to compare against** -- no
                         recorded floor, baseline or value for this key
3     ``DID_NOT_RUN``    environment / not implemented / precondition refused.
                         **Not a pass.**
4     ``UNBANKED_RISE``  measured above the floor, floor not updated
====  =================  ==========================================================

🔴 **`2` and `3` are separate deliberately, and `3` is where this design can be
silently defeated** (`docs/gates.md`). *"The gate did not run"* and *"the gate found
nothing"* are different facts, and so are *"the gate ran and had nothing to compare"*
and *"the gate did not run"*. This module exists because each of those collapses
has already happened here:

* `scripts/canary.py`, cycle 0 of 2026-09-19: a first reading exited **0** --
  `3` collapsing to `0`.
* the same script's fix mapped that first reading to **3** -- `2` collapsing to
  `3`. It ran; it trained, read six beats and wrote the baseline. What it could not
  do is compare. That is `2`, and the script's own docstring said so (S0-05).
* seven `experiments/e0*/run.py` stubs raised an uncaught `NotImplementedError`,
  which Python reports as **1** -- *did not run* collapsing to *real failure*.

Member names match `class Exit` in `src/rsr/gates/floors.py` on
`macbook-local-2026-09-18` (``OK``/``DROP``/``UNKNOWN``/``DID_NOT_RUN``/
``UNBANKED_RISE``), so the port of that branch can import this enum rather than
define a second one. ``4`` is specified here and emitted by nothing on trunk: the
only producer is that unported ratchet (`docs/gates.md`, S0-05 brief).

**Stdlib only, on purpose.** `scripts/extract_golden_tensors.py` runs in a
throwaway JAX venv with no torch; it must be able to import this file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from enum import IntEnum
from typing import NoReturn

__all__ = [
    "ArgumentParser",
    "Exit",
    "did_not_run",
    "not_implemented",
    "refuse",
    "run_main",
    "status",
]


class Exit(IntEnum):
    """`0` ok · `1` real failure · `2` nothing to compare · `3` did not run ·
    `4` unbanked rise. See the module docstring; `docs/gates.md` is the source."""

    OK = 0
    FAIL = 1
    DROP = 1  # alias -- `floors.py`'s name for a real failure of a floor
    UNKNOWN = 2
    DID_NOT_RUN = 3
    UNBANKED_RISE = 4


def status(code: object) -> Exit:
    """The one conversion from a checker's result to a process status.

    🔴 **Refuses a bool.** `True`/`False` are ints in Python, so a checker that
    returns `ok` exits `1` on success and `0` on failure -- and one that returns
    `not failed` has exactly two states where the protocol has five. There is no
    boolean that means *did not run*.

    🔴 **Refuses `None`.** `sys.exit(None)` is exit **0**, so a `main()` that falls
    off its end without deciding is reported as a pass -- `3` collapsing to `0`
    by omission.

    Refuses any int outside the protocol.
    """
    if isinstance(code, bool):
        raise TypeError(
            f"a checker returned the bool {code!r}. A bool has two states and the "
            f"protocol has five; return an `Exit` member (rsr.exit_codes)."
        )
    if code is None:
        raise TypeError(
            "a checker returned None, which sys.exit() reports as 0 -- a pass. "
            "Return an `Exit` member (rsr.exit_codes)."
        )
    if not isinstance(code, int):
        raise TypeError(f"a checker returned {type(code).__name__} {code!r}, not an Exit")
    try:
        return Exit(code)
    except ValueError:
        raise ValueError(
            f"exit status {code} is outside the protocol (0-4, docs/gates.md)"
        ) from None


def did_not_run(reason: str) -> Exit:
    """Say why on stderr, and return `Exit.DID_NOT_RUN`. For a `main()` to return."""
    print(f"DID NOT RUN: {reason}", file=sys.stderr)
    return Exit.DID_NOT_RUN


def not_implemented(experiment: str) -> Exit:
    """What an unimplemented experiment's `main()` returns: `3`, with the reason.

    🔴 Seven `experiments/e0*/run.py` stubs used to `raise NotImplementedError`,
    which escapes uncaught and exits **1** -- *real failure* -- for the one state
    that is the definition of *did not run* (S0-05). They all call this, so one
    mutation here covers the class.
    """
    return did_not_run(f"{experiment} is not implemented yet.")


class ArgumentParser(argparse.ArgumentParser):
    """`argparse.ArgumentParser`, except a usage error exits `3`, not `2`.

    argparse exits **2** on bad arguments. In this protocol `2` means *ran, and
    had nothing to compare*, so a checker using a stock parser gives `2` two
    meanings -- `scripts/render_scoreboard.py` did, next to its own correct
    `2 if not board.rows` (S0-05). Bad arguments mean the check did not run.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        refuse(Exit.DID_NOT_RUN, f"{self.prog}: {message}")


def refuse(code: Exit, reason: str) -> NoReturn:
    """Exit with `code` from anywhere in the call stack, saying why on stderr.

    `raise SystemExit("message")` exits **1** whatever the message says -- which is
    how `scripts/mutation_battery.py` reported *"the battery did not run"* as a real
    failure. This is the replacement: the code is chosen, not defaulted.
    """
    code = status(code)
    label = "DID NOT RUN" if code is Exit.DID_NOT_RUN else code.name
    print(f"{label}: {reason}", file=sys.stderr)
    raise SystemExit(int(code))


def run_main(main: Callable[[], object]) -> NoReturn:
    """`sys.exit(status(main()))`: the entry-point line every checker uses."""
    sys.exit(int(status(main())))
