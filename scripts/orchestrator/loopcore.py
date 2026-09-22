"""Shared plumbing for the loop driver (agent D): root, git, config, clock, state.

Everything the loop-driver modules (`night`, `reconcile`, `dispatch`, `submit`,
`session`, `merge`, `notify`, `tick`) have in common, and nothing else. Stdlib
only, so the tick can run before `uv sync` has been proven.

Owner rulings this file serves:

* **R-2026-09-22-night-branch** -- unattended work merges only into
  `night/<date>`, cut from a pinned `base_sha`; `main` never moves during a night.
  Nothing in this package writes `refs/heads/main`. `tests/test_orch_tick.py`
  asserts the ref is byte-identical before and after a full night.
* **R-2026-09-22-owner-out-of-loop** -- the owner is contacted only through
  `notify.py` (one pinned GitHub issue), and owner-only decisions are batched into
  the digest.

🔴 **Exit codes are captured, never borrowed.** Every child status in this package
is `CompletedProcess.returncode` of the child itself -- no pipes, no shells. A
signal death (`returncode < 0`) is normalised to `128 + signo` so a SIGTERM reads
`143`, exactly as a shell would report it (CLAUDE.md, "Reading an exit code").
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: The directory holding the `orchestrator` package; put on PYTHONPATH for children.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]

#: Exit code 5 is ruled (R-2026-09-22-inert-exit-5) but is not yet a member of
#: `rsr.exit_codes.Exit`. The routing table maps plain ints so it can carry 5 now.
INERT = 5

#: What the loop does with a child's exit status. Keys are ints on purpose (see
#: INERT). Read by `reconcile` for session/job outcomes and by the digest.
ROUTING: dict[int, str] = {
    0: "ok -- proceed (a researcher result goes to verifying)",
    1: "real failure -- attempt+1, park on the second failure with the same cause",
    2: "nothing to compare -- report as UNKNOWN; never a pass",
    3: "did not run / refused -- a precondition; report it, never a pass",
    4: "unbanked rise -- owner decision, batched into the digest",
    INERT: (
        "INERT -- model/training-config investigation; retention experiments on "
        "that checkpoint refused"
    ),
    143: "SIGTERM (128+15) -- session killed; attempt+1, park on a repeat",
}


def route(rc: int | None) -> str:
    if rc is None:
        return "no exit status recorded -- the process died without reporting one"
    return ROUTING.get(rc, "unmapped exit status -- treated as a real failure")


# --------------------------------------------------------------------------- #
# processes and git
# --------------------------------------------------------------------------- #


def norm_rc(rc: int) -> int:
    """`subprocess` reports death-by-signal as `-signo`; a shell reports `128+signo`."""
    return 128 - rc if rc < 0 else rc


def git_bin() -> str:
    """`/opt/homebrew/bin/git`, never Apple's (CLAUDE.md). `RSR_GIT_BIN` overrides;
    off macOS (CI) the PATH git is the only one there is."""
    env = os.environ.get("RSR_GIT_BIN")
    if env:
        return env
    if Path("/opt/homebrew/bin/git").exists():
        return "/opt/homebrew/bin/git"
    return shutil.which("git") or "git"


def git(cwd: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [git_bin(), *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} (rc={proc.returncode}): {proc.stderr}")
    return proc


class GitError(RuntimeError):
    pass


def git_out(cwd: Path, *args: str) -> str:
    return git(cwd, *args, check=True).stdout.strip()


def root() -> Path:
    """`RSR_ORCH_ROOT`, else the MAIN checkout's toplevel.

    📌 Not `git rev-parse --show-toplevel`: inside `.worktrees/<item>` that is the
    worktree, and `.orchestrator/` would silently fork per worktree. The common git
    dir's parent is the main checkout from anywhere.
    """
    env = os.environ.get("RSR_ORCH_ROOT")
    if env:
        return Path(env).resolve()
    common = git_out(
        Path.cwd(), "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    return Path(common).resolve().parent


def pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def orch_cmd(module: str) -> list[str]:
    """How to invoke another agent's CLI: `RSR_ORCH_CMD_<MODULE>` (shell-split)
    overrides, which is how the tests stub modules this branch does not contain."""
    env = os.environ.get(f"RSR_ORCH_CMD_{module.upper()}")
    if env:
        return shlex.split(env)
    return [sys.executable, "-m", f"orchestrator.{module}"]


def child_env(root_: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SCRIPTS_DIR}{os.pathsep}{pp}" if pp else str(SCRIPTS_DIR)
    env["RSR_ORCH_ROOT"] = str(root_)
    env.update(extra)
    return env


def run_orch(
    root_: Path, module: str, *args: str, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*orch_cmd(module), *args],
        cwd=cwd or root_,
        capture_output=True,
        text=True,
        env=child_env(root_),
        check=False,
    )


# --------------------------------------------------------------------------- #
# paths and state
# --------------------------------------------------------------------------- #


def orch_dir(root_: Path) -> Path:
    return root_ / ".orchestrator"


def sub(root_: Path, name: str) -> Path:
    p = orch_dir(root_) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def halt_path(root_: Path) -> Path:
    return orch_dir(root_) / "HALT"


def halted(root_: Path) -> str | None:
    p = halt_path(root_)
    if not p.exists():
        return None
    return p.read_text().strip() or "(HALT present, no reason written)"


def halt(root_: Path, reason: str) -> None:
    """Write `.orchestrator/HALT`. The tick exits 0 at its first line while it
    exists; only the owner removes it (ops/README.md)."""
    orch_dir(root_).mkdir(parents=True, exist_ok=True)
    stamp = now().isoformat(timespec="seconds")
    with halt_path(root_).open("a") as f:
        f.write(f"{stamp} {reason}\n")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write_json(path: Path, obj) -> None:
    """Atomic: a reader never sees half a record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.strip()]


def night_path(root_: Path) -> Path:
    return sub(root_, "state") / "night.json"


def read_night(root_: Path) -> dict | None:
    return load_json(night_path(root_), None)


PIPELINE_DEFAULT: dict = {
    "items": {},  # item_id -> {brief, lane, slots, worktree, branch}
    "awaiting": [],  # item ids whose researcher returned; waiting on verification
    "collecting": {},  # job_id -> item_id, jobs finished and not yet collected
    "collected": [],  # job ids a collector has returned on
    "failures": {},  # item_id | "_manager" -> [cause, ...]
    "parked": {},  # item_id -> reason
    "merged": [],  # item ids merged into the night branch
    "reconciled": [],  # cycle numbers already routed
    "manager_due": True,
    "last_manager_end": None,
    "idle_signature": None,
}


def load_pipeline(root_: Path) -> dict:
    data = load_json(sub(root_, "state") / "pipeline.json", {})
    return {**json.loads(json.dumps(PIPELINE_DEFAULT)), **data}


def save_pipeline(root_: Path, p: dict) -> None:
    write_json(sub(root_, "state") / "pipeline.json", p)


# --------------------------------------------------------------------------- #
# cycle (claude session) records
# --------------------------------------------------------------------------- #


def new_cycle(root_: Path, record: dict) -> tuple[int, Path]:
    """Claim the next `.orchestrator/cycles/<n>.json` with O_EXCL."""
    d = sub(root_, "cycles")
    n = 1 + max((int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()), default=0)
    while True:
        path = d / f"{n}.json"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({**record, "n": n}, indent=2, sort_keys=True) + "\n")
        return n, path


def cycle_path(root_: Path, n: int) -> Path:
    return sub(root_, "cycles") / f"{n}.json"


def result_path(root_: Path, n: int) -> Path:
    """Written once, by `session.py`, when the session's process has exited. A
    cycle record with no result and a dead pid is a session that died unreported."""
    return sub(root_, "cycles") / f"{n}.result.json"


def cycles(root_: Path) -> list[tuple[int, dict, dict | None]]:
    """(n, record, result-or-None), ascending."""
    d = sub(root_, "cycles")
    out = []
    for p in sorted(
        (p for p in d.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem)
    ):
        out.append(
            (
                int(p.stem),
                load_json(p, {}),
                load_json(result_path(root_, int(p.stem)), None),
            )
        )
    return out


def live_sessions(root_: Path, kind: str | None = None) -> list[dict]:
    return [
        rec
        for _n, rec, res in cycles(root_)
        if res is None
        and pid_alive(rec.get("pid"))
        and (kind is None or rec.get("kind") == kind)
    ]


def spend_path(root_: Path) -> Path:
    return sub(root_, "state") / "spend.jsonl"


def night_spend(root_: Path, night_date: str) -> float:
    return sum(
        float(r.get("cost_usd") or 0.0)
        for r in read_jsonl(spend_path(root_))
        if r.get("night") == night_date
    )


# --------------------------------------------------------------------------- #
# config and clock
# --------------------------------------------------------------------------- #


def env_file(root_: Path) -> Path:
    return Path(os.environ.get("RSR_ORCH_ENV") or root_ / "ops" / "orchestrator.env")


def load_config(root_: Path) -> dict[str, str]:
    """`ops/orchestrator.env` (KEY=VALUE, `#` comments), overlaid by the process
    environment -- so an explicit env var always wins over the file."""
    cfg: dict[str, str] = {}
    with contextlib.suppress(FileNotFoundError):
        for raw in env_file(root_).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            cfg[key] = val.strip().strip('"').strip("'")
    for key, val in os.environ.items():
        if key.startswith("RSR_") or key == "DRY_RUN":
            cfg[key] = val
    return cfg


def cap(cfg: dict[str, str], key: str) -> float | None:
    """A spend cap, or None when it is `UNSET`, missing or not a positive number.
    The caps are an owner decision; None means the loop refuses to spend."""
    raw = cfg.get(key, "UNSET")
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val > 0 else None


def now() -> dt.datetime:
    """Local wall clock; `RSR_NOW` (ISO, naive local) pins it for tests."""
    env = os.environ.get("RSR_NOW")
    return dt.datetime.fromisoformat(env) if env else dt.datetime.now()


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass(frozen=True)
class Window:
    """The night, in local wall-clock minutes. It wraps midnight.

    `opens` is when a night may start; the stop rules of
    docs/lab-notes/dispatch-2026-09-21-overnight.md ("Wall clock") are the rest:
    no new brief after `last_dispatch`, everything merged or parked by `park_by`,
    the morning document final by `digest_by`.
    """

    opens: str = "21:00"
    last_dispatch: str = "06:30"
    park_by: str = "07:30"
    digest_by: str = "08:00"

    @classmethod
    def from_config(cls, cfg: dict[str, str]) -> Window:
        return cls(
            opens=cfg.get("RSR_WINDOW_OPENS", cls.opens),
            last_dispatch=cfg.get("RSR_LAST_DISPATCH", cls.last_dispatch),
            park_by=cfg.get("RSR_PARK_BY", cls.park_by),
            digest_by=cfg.get("RSR_DIGEST_BY", cls.digest_by),
        )

    def _since_open(self, hm: int) -> int:
        return (hm - _hm(self.opens)) % 1440

    def phase(self, t: dt.datetime) -> str:
        """`open` (dispatch allowed) · `draining` (after last_dispatch) ·
        `parking` (after park_by) · `closed` (after digest_by, before opens)."""
        m = self._since_open(t.hour * 60 + t.minute)
        if m >= self._since_open(_hm(self.digest_by)):
            return "closed"
        if m >= self._since_open(_hm(self.park_by)):
            return "parking"
        if m >= self._since_open(_hm(self.last_dispatch)):
            return "draining"
        return "open"

    def night_date(self, t: dt.datetime) -> str:
        d = t.date()
        if t.hour * 60 + t.minute < _hm(self.opens):
            d -= dt.timedelta(days=1)
        return d.isoformat()

    def as_json(self) -> dict[str, str]:
        return {
            "opens": self.opens,
            "last_dispatch": self.last_dispatch,
            "park_by": self.park_by,
            "digest_by": self.digest_by,
        }


@contextlib.contextmanager
def tick_lock(root_: Path):
    """Non-blocking flock. Yields True when held, False when another tick has it."""
    path = sub(root_, "locks") / "tick.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True
    finally:
        os.close(fd)


def log(msg: str) -> None:
    print(f"[{now().isoformat(timespec='seconds')}] {msg}", flush=True)


def claude_argv(cfg: dict[str, str], prompt: str, agent: str, settings: str) -> list[str]:
    """The one spelling of a claude invocation. `RSR_CLAUDE_BIN` is the binary --
    the tests point it at a stub; nothing here ever defaults to a live session in a
    test, because every test sets it."""
    return [
        cfg.get("RSR_CLAUDE_BIN") or "claude",
        "-p",
        prompt,
        "--agent",
        agent,
        "--settings",
        settings,
        "--permission-mode",
        "dontAsk",
        "--output-format",
        "json",
        "--max-budget-usd",
        cfg.get("RSR_CYCLE_CAP", "UNSET"),
    ]
