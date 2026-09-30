"""I6 (PLAN-v4 §4; red-team m-2): the battery's ``RSR_*`` environment, recorded and gated.

The mutation battery hands its whole environment to pytest. Until this module it
recorded one fact about that environment -- where the thread cap came from -- and
on 2026-09-26/27 its verdicts changed with an ambient ``RSR_BATTERY_THREADS``. A
gate whose answer depends on state nobody wrote down is a gap. So, at start:

1. **Record.** Two environments, both printed and both on every row of the
   ``--json`` record (`stamp`):

   * ``rsr_env`` -- every ``RSR_*`` variable of the battery process itself, as
     `check` passed it (stdout: `header`, at start);
   * ``rsr_env_suite`` -- every ``RSR_*`` variable of the environment the shard's
     pytest actually receives, after `mutation_battery._suite_env` and
     `battery_isolation.shard_env` (stdout: `suite_header`, which marks each name
     the battery set, replaced or removed). Fix round 1 (review MAJOR-3): the
     first version recorded only the battery's own environment, so an ambient
     ``RSR_TORCH_THREADS=14`` was recorded while the suite ran with ``2``.

   Per row, not once per file: the record stays a list of rows, which is what
   `scripts/battery_subset.py` and every reader of it expects. `stamp` never
   replaces an environment a row already carries (review MAJOR-4): rows merged
   from several batteries each keep the one they ran in.
2. **Refuse.** A variable that starts with ``RSR_`` and is not *declared* makes the
   battery exit 3 (``DID NOT RUN``) before any suite runs, naming it (`check`).

**What "declared" means.** A name is declared when it is a key of `DECLARED`, or
``RSR_ORCH_CMD_<MODULE>`` for a module that exists in ``scripts/orchestrator/``
(`ORCH_CMD_NAMES`; ``loopcore.orch_cmd`` reads exactly those): a name that code in
this repository reads or sets, written down here next to who reads it.
``tests/test_battery_env.py`` holds the list to the tree -- every ``RSR_*`` token in
`SCANNED` (`inventory`) must be declared -- so it cannot go stale in the direction
that matters. An undeclared name is therefore one nothing in this tree
is known to read: a typo of a real control (``RSR_BATTERY_THREAD=4`` caps nothing
while its author believes it does) or a leftover of another tool. Neither is
something a verdict should be issued under.

Declared is **not** "harmless": ``RSR_LEDGER`` moves the constants registry's
ledger and ``RSR_FRESH_STREAM_RUNS`` moves an experiment's inputs, and a suite run
under either may differ. They are allowed because refusing them would also refuse
every legitimate launch, and they are recorded so that a verdict that depends on
one can be seen to.

Two groups must be declared or the battery refuses its own launches:

* **What the battery sets for a shard's suite** -- ``RSR_ORCH_ROOT``,
  ``RSR_BATTERY_PROBE``, ``RSR_TEST_COUNT``, and ``RSR_TORCH_THREADS`` when the
  suite is thread-capped (always on the Studio: ``ops/lanes.json`` exists). The
  suite runs the battery again, nested (``tests/test_battery_isolation.py`` drives
  the real ``main`` on a stub repository), and the nested battery inherits them.
* **What the launcher sets** -- ``orchestrator.slot run`` forces ``RSR_TORCH_THREADS``
  on its child (``lanes.THREAD_ENV_VARS``); ``orchestrator.dispatch`` exports
  ``RSR_ITEM_ID``, ``RSR_RUN_ID`` and ``RSR_BASE_SHA`` into a researcher session,
  from which a subset battery is run; ``tick`` and launchd export ``RSR_ORCH_ROOT``.
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
    # --- set by the battery for each shard's suite (recorded as rsr_env_suite)
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

ORCH_CMD_PREFIX = "RSR_ORCH_CMD_"
ORCH_CMD_NAMES: frozenset[str] = frozenset(
    ORCH_CMD_PREFIX + p.stem.upper()
    for p in (Path(__file__).resolve().parent / "orchestrator").glob("*.py")
    if p.stem != "__init__"
)
"""``RSR_ORCH_CMD_<MODULE>``: `loopcore.orch_cmd` reads ``RSR_ORCH_CMD_`` +
``module.upper()`` to replace an orchestrator CLI (how tests stub a module). Only
names for a module that exists are declared (review MINOR-5): a misspelled module
is the typo this gate exists for. A correctly spelled one replaces an orchestrator
CLI inside the suite, so it is recorded, not neutral."""

NOT_VARIABLES: dict[str, str] = {
    "RSR_STATUS": "scripts/render_status.py: a JavaScript global, window.RSR_STATUS",
    "RSR_TEST_COUN": f"{_BATTERY}: one half of a split literal in the MUTATIONS table",
    "RSR_ORCH_CMD_": f"{_LOOPCORE} orch_cmd: the f-string prefix of ORCH_CMD_NAMES",
}
"""``RSR_*`` tokens in the tree that are not environment variables. They are not
declared: set in the environment, either is refused like any other stray."""

SCANNED = ("src", "scripts", "experiments", "tests", "ops", ".claude")
"""Where `inventory` looks. In ``tests/`` only files whose name does not start with
``test_`` (conftest, ``_battery_*`` helpers): test files hold made-up names on
purpose. In ``.claude/`` only tracked-style settings, never ``*.local.json``
(review MINOR-7: ``tests/conftest.py``, ``ops/launchd/*.plist`` and
``.claude/settings.*.json`` read or set names too)."""
_SUFFIXES = (".py", ".zsh", ".sh", ".json", ".plist", ".env")
_NAME = re.compile(r"RSR_[A-Z0-9_]+")
_SELF = "scripts/battery_env.py"


def is_declared(name: str) -> bool:
    return name in DECLARED or name in ORCH_CMD_NAMES


def _role(name: str) -> str:
    if name in DECLARED:
        return DECLARED[name]
    if name in ORCH_CMD_NAMES:
        return f"{_LOOPCORE} orch_cmd: replaces orchestrator.{name[13:].lower()}"
    return "UNDECLARED"


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
            near = difflib.get_close_matches(
                name, [*DECLARED, *ORCH_CMD_NAMES], n=1, cutoff=0.8
            )
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
        lines.append(f"  {name}={value!r} -- {_role(name)}")
    return "\n".join(lines)


def suite_header(own: Mapping[str, str], suite: Mapping[str, str]) -> str:
    """What the shard's pytest receives, each difference from ``own`` marked."""
    lines = [f"{PREFIX}* environment of each shard's suite: {len(suite)} set"]
    for name, value in suite.items():
        if name not in own:
            note = " [set by the battery]"
        elif own[name] != value:
            note = f" [replaced by the battery; its own value is {own[name]!r}]"
        else:
            note = ""
        lines.append(f"  {name}={value!r}{note}")
    for name in own:
        if name not in suite:
            lines.append(
                f"  {name} [removed by the battery; its own value is {own[name]!r}]"
            )
    return "\n".join(lines)


def stamp(row: dict, own: Mapping[str, str], suite: Mapping[str, str]) -> dict:
    """``row`` with ``rsr_env`` (the battery's) and ``rsr_env_suite`` (pytest's).

    A new dict. A key the row already has is kept, never replaced (review MAJOR-4):
    a row is stamped where it is produced, and a driver that merges rows from
    several batteries must not overwrite each one's environment with its own.
    """
    out = dict(row)
    out.setdefault("rsr_env", dict(own))
    out.setdefault("rsr_env_suite", dict(suite))
    return out


def inventory(root: Path) -> dict[str, list[str]]:
    """``{RSR_* token: ["path:line", ...]}`` over `SCANNED` under ``root``.

    Not this file, which would find every name it declares; not ``test_*`` files;
    not ``*.local.json``; nothing under ``.venv`` or ``__pycache__``.
    """
    found: dict[str, list[str]] = {}
    for top in SCANNED:
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root).as_posix()
            if (
                path.suffix not in _SUFFIXES
                or rel == _SELF
                or path.name.startswith("test_")
                or path.name.endswith(".local.json")
                or {".venv", "__pycache__"} & set(path.parts)
                or not path.is_file()
            ):
                continue
            text = path.read_text(errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                for name in _NAME.findall(line):
                    found.setdefault(name, []).append(f"{rel}:{n}")
    return dict(sorted(found.items()))
