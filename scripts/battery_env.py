"""I6 (PLAN-v4 §4; red-team m-2): the battery's ``RSR_*`` environment, recorded and gated.

The mutation battery hands its whole environment to pytest. Until this module it
recorded one fact about that environment -- where the thread cap came from -- and
on 2026-09-26/27 its verdicts changed with an ambient ``RSR_BATTERY_THREADS``. A
gate whose answer depends on state nobody wrote down is a gap. So, at start:

1. **Record.** Every variable whose name starts with ``RSR_`` is printed with its
   value (`header`) and written into every row of the ``--json`` record under
   ``rsr_env`` (`stamp`). Per row, not once per file: the record stays a list of
   rows, which is what `scripts/battery_subset.py` and every reader of it expects,
   and rows merged from several batteries each keep the environment they ran in.
2. **Refuse.** A variable that starts with ``RSR_`` and is not *declared* makes the
   battery exit 3 (``DID NOT RUN``) before any suite runs, naming it (`check`).

**What "declared" means.** A name is declared when it is a key of `DECLARED` (or
starts with a key of `DECLARED_PREFIXES`): a name that code in this repository
reads or sets, written down here next to who reads it. ``tests/test_battery_env.py``
holds the list to the tree -- every ``RSR_*`` name under ``src/``, ``scripts/`` and
``experiments/`` (`inventory`) must be declared -- so it cannot go stale in the
direction that matters. An undeclared name is therefore one nothing in this tree
is known to read: a typo of a real control (``RSR_BATTERY_THREAD=4`` caps nothing
while its author believes it does) or a leftover of another tool. Neither is
something a verdict should be issued under.

Declared is **not** "harmless": ``RSR_LEDGER`` moves the constants registry's
ledger and ``RSR_FRESH_STREAM_RUNS`` moves an experiment's inputs, and a suite run
under either may differ. They are allowed because refusing them would also refuse
every legitimate launch, and they are recorded so that a verdict that depends on
one can be seen to.

Two groups must be declared or the battery refuses its own launches:

* **What the battery sets for a shard's suite** -- `REPLACED_IN_SUITE`, and
  ``RSR_TORCH_THREADS`` when the suite is thread-capped. The suite runs the battery
  again, nested (``tests/test_battery_isolation.py`` drives the real ``main`` on a
  stub repository), and the nested battery inherits them.
* **What the launcher sets** -- ``orchestrator.slot run`` forces ``RSR_TORCH_THREADS``
  on its child (``lanes.THREAD_ENV_VARS``); ``orchestrator.dispatch`` exports
  ``RSR_ITEM_ID``, ``RSR_RUN_ID`` and ``RSR_BASE_SHA`` into a researcher session,
  from which a subset battery is run; ``tick`` and launchd export ``RSR_ORCH_ROOT``.

What is recorded is the battery's **own** environment. For a name in
`REPLACED_IN_SUITE` the suite never sees that value -- the header says so.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from rsr.exit_codes import Exit, refuse

PREFIX = "RSR_"

_BATTERY = "scripts/mutation_battery.py"
_ISOLATION = "scripts/battery_isolation.py"
_LOOPCORE = "scripts/orchestrator/loopcore.py"
_CONFIG = f"{_LOOPCORE} load_config: ops/orchestrator.env, overlaid by the environment"

DECLARED: dict[str, str] = {
    # --- the battery's own controls
    "RSR_BATTERY_THREADS": f"{_BATTERY} suite_threads: the suite's thread cap",
    "RSR_GIT": f"{_ISOLATION} _git_bin, orchestrator lanes/lint_brief: the git binary",
    # --- set by the battery for each shard's suite (see REPLACED_IN_SUITE)
    "RSR_BATTERY_PROBE": f"{_ISOLATION} shard_env -> tests/_battery_probe.py",
    "RSR_ORCH_ROOT": f"{_ISOLATION} shard_env -> orchestrator lanes.orch_root et al.",
    "RSR_TEST_COUNT": f"{_BATTERY} _suite_env -> tests/conftest.py: the census file",
    # --- set by the launcher
    "RSR_TORCH_THREADS": "orchestrator lanes.thread_env: forced by slot.run_job and "
    "by the battery's own thread cap",
    "RSR_ITEM_ID": "orchestrator dispatch: exported into a researcher session",
    "RSR_RUN_ID": "orchestrator dispatch -> hooks: the session's run id",
    "RSR_BASE_SHA": "orchestrator dispatch: exported into a researcher session",
    "RSR_PROFILE": "orchestrator hooks: manager | researcher, set in the hook command",
    # --- the orchestrator's own configuration and test seams
    "RSR_CYCLE_CAP": _CONFIG,
    "RSR_NIGHT_CAP": _CONFIG,
    "RSR_WINDOW_OPENS": _CONFIG,
    "RSR_LAST_DISPATCH": _CONFIG,
    "RSR_PARK_BY": _CONFIG,
    "RSR_DIGEST_BY": _CONFIG,
    "RSR_MAX_SESSIONS": _CONFIG,
    "RSR_MAX_ATTEMPTS": _CONFIG,
    "RSR_MANAGER_INTERVAL_MIN": _CONFIG,
    "RSR_ORCH_ENV": f"{_LOOPCORE} env_file: where orchestrator.env is",
    "RSR_NOW": f"{_LOOPCORE} now: pins the clock",
    "RSR_GIT_BIN": f"{_LOOPCORE} git_bin: the git binary",
    "RSR_CLAUDE_BIN": f"{_LOOPCORE}: the claude binary",
    "RSR_GH_BIN": "scripts/orchestrator/notify.py: the gh binary",
    "RSR_UV_BIN": "scripts/orchestrator/submit.py, dispatch.py: the uv binary",
    "RSR_PYTHON": "scripts/orchestrator/tick.zsh: the interpreter",
    "RSR_NO_CAFFEINATE": "scripts/orchestrator/tick.py",
    "RSR_NIGHT_JSON": "scripts/orchestrator/hooks.py: where night.json is",
    "RSR_REPO_ROOT": "scripts/orchestrator/hooks.py: the project root",
    # --- src and experiments: these change what a suite or a run computes
    "RSR_LEDGER": "src/rsr/constants.py: the constants registry's measurement ledger",
    "RSR_FRESH_STREAM_RUNS": "experiments/lookahead-room, retention-readability run.py",
    "RSR_B2_THREADS": "experiments/b2-psi-probe/run.py: threads per child",
    "RSR_LOOKAHEAD_THREADS": "experiments/lookahead-room/run.py: threads per child",
    "RSR_READABILITY_THREADS": "experiments/retention-readability/run.py: threads",
    "RSR_NB_THREADS": "experiments/newcomer-bakeoff/run.py: threads per child",
    "RSR_NB_B2_FITS": "experiments/newcomer-bakeoff/run.py: where B2's fits are",
}
"""Every ``RSR_*`` environment variable this repository reads or sets -> who reads it."""

DECLARED_PREFIXES: dict[str, str] = {
    "RSR_ORCH_CMD_": f"{_LOOPCORE} orch_cmd: RSR_ORCH_CMD_<MODULE> replaces an "
    "orchestrator CLI (how tests stub a module)",
}
"""Families whose full names are built at run time."""

NOT_VARIABLES: dict[str, str] = {
    "RSR_STATUS": "scripts/render_status.py: a JavaScript global, window.RSR_STATUS",
    "RSR_TEST_COUN": f"{_BATTERY}: one half of a split literal in the MUTATIONS table",
}
"""``RSR_*`` tokens in the tree that are not environment variables. They are not
declared: set in the environment, either is refused like any other stray."""

REPLACED_IN_SUITE = ("RSR_BATTERY_PROBE", "RSR_ORCH_ROOT", "RSR_TEST_COUNT")
"""The battery sets these itself for every shard's suite (`battery_isolation.shard_env`,
`mutation_battery._suite_env`): an ambient value is recorded and never reaches pytest."""

SCANNED = ("src", "scripts", "experiments")
_SUFFIXES = (".py", ".zsh", ".sh")
_NAME = re.compile(r"RSR_[A-Z0-9_]+")
_SELF = "scripts/battery_env.py"


def is_declared(name: str) -> bool:
    return name in DECLARED or any(name.startswith(p) for p in DECLARED_PREFIXES)


def _role(name: str) -> str:
    if name in DECLARED:
        return DECLARED[name]
    return next(why for p, why in DECLARED_PREFIXES.items() if name.startswith(p))


def present(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Every ``RSR_*`` variable in ``environ`` (default: this process's), sorted."""
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k in sorted(environ) if k.startswith(PREFIX)}


def undeclared(environ: Mapping[str, str] | None = None) -> list[str]:
    return [name for name in present(environ) if not is_declared(name)]


def check(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The ``RSR_*`` environment, or DID NOT RUN (3) if any of it is undeclared."""
    env = present(environ)
    stray = [name for name in env if not is_declared(name)]
    if stray:
        named = []
        for name in stray:
            near = difflib.get_close_matches(name, DECLARED, n=1, cutoff=0.8)
            hint = f" (did you mean {near[0]}?)" if near else ""
            named.append(f"{name}={env[name]!r}{hint}")
        refuse(
            Exit.DID_NOT_RUN,
            f"{len(stray)} undeclared {PREFIX}* variable(s) in the environment: "
            f"{', '.join(named)}. Nothing in this tree is declared to read them "
            f"({_SELF}), so each is a typo of a real control -- which then did not "
            f"apply -- or a leftover. Unset it, or declare it with the code that "
            f"reads it. No suite ran.",
        )
    return env


def header(env: Mapping[str, str]) -> str:
    """The record's stdout form: one line per variable, name, value and reader."""
    if not env:
        return f"{PREFIX}* environment: none set"
    lines = [f"{PREFIX}* environment: {len(env)} set, all declared ({_SELF})"]
    for name, value in env.items():
        note = " [replaced for each shard's suite]" if name in REPLACED_IN_SUITE else ""
        lines.append(f"  {name}={value!r} -- {_role(name)}{note}")
    return "\n".join(lines)


def stamp(rows: list[dict], environ: Mapping[str, str] | None = None) -> list[dict]:
    """``rows`` with the ``RSR_*`` environment on each, as ``rsr_env``. New dicts."""
    env = present(environ)
    return [{**row, "rsr_env": dict(env)} for row in rows]


def inventory(root: Path) -> dict[str, list[str]]:
    """``{RSR_* token: ["path:line", ...]}`` over ``root``'s src, scripts, experiments.

    Tests are not scanned -- they hold made-up names on purpose -- and neither is
    this file, which would find every name it declares.
    """
    found: dict[str, list[str]] = {}
    for top in SCANNED:
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root).as_posix()
            if path.suffix not in _SUFFIXES or rel == _SELF or not path.is_file():
                continue
            text = path.read_text(errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                for name in _NAME.findall(line):
                    found.setdefault(name, []).append(f"{rel}:{n}")
    return dict(sorted(found.items()))
