"""A throwaway git world for the loop-driver tests (tests/test_orch_*.py, agent D).

Builds, under `tmp_path`: a bare `origin.git`, a clone `repo/` whose `main` holds
the frozen paths the merge guard protects, and executable stubs for every external
the loop calls -- `claude`, `gh`, `uv`, and the other agents' CLIs (`workqueue`,
`lint_brief`, `lanes`, `slot`). Each stub appends its argv to a log so a test can
count calls.

🔴 **No test here can start a real claude session or post to GitHub.** The fixture
sets `RSR_CLAUDE_BIN` and `RSR_GH_BIN` to stubs and `Orch.__init__` asserts both
resolve inside `tmp_path` before any test body runs. `RSR_ORCH_ROOT` points at the
throwaway repo, so the real `.orchestrator/` is never written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from orchestrator import loopcore as lc  # noqa: E402

NOW = "2026-09-22T23:00:00"  # inside the window: phase "open", night 2026-09-22
NIGHT = "2026-09-22"

CONSTANTS_PY = """\
class Klass:
    FROZEN = "FROZEN"
    MEASURED = "MEASURED"


def Definition(**kw):
    return kw


_DEFINITIONS = (
    Definition(
        name="b_max",
        klass=Klass.FROZEN,
        spec_ref="4.5",
        value=1.0,
    ),
    Definition(
        name="tau",
        klass=Klass.MEASURED,
        spec_ref="4.5",
        source_experiment="E0e",
    ),
)
"""

_STUB_HEAD = f"#!{sys.executable}\nimport json, os, sys, pathlib\n"
_LOG = (
    "def log(name, obj):\n"
    "    d = pathlib.Path(os.environ['STUB_LOG_DIR'])\n"
    "    with (d / f'{name}.jsonl').open('a') as f:\n"
    "        f.write(json.dumps(obj) + '\\n')\n"
)

STUBS = {
    "claude": _LOG
    + """
log("claude", {"argv": sys.argv[1:], "cwd": os.getcwd(),
               "run_id": os.environ.get("RSR_RUN_ID")})
rc = int(os.environ.get("STUB_CLAUDE_RC", "0"))
print(json.dumps({
    "type": "result",
    "subtype": "success" if rc == 0 else "error_during_execution",
    "is_error": rc != 0,
    "total_cost_usd": float(os.environ.get("STUB_CLAUDE_COST", "0.25")),
    "result": os.environ.get("STUB_CLAUDE_RESULT", "status: RETURNED"),
}))
sys.exit(rc)
""",
    "gh": _LOG
    + """
log("gh", sys.argv[1:])
state = pathlib.Path(os.environ["STUB_LOG_DIR"]) / "gh_issue"
a = sys.argv[1:]
if a[:2] == ["issue", "list"]:
    print(json.dumps([{"number": 7}]) if state.exists() else "[]")
elif a[:2] == ["issue", "create"]:
    state.write_text("7")
    print("https://github.com/example/rsr/issues/7")
sys.exit(int(os.environ.get("STUB_GH_RC", "0")))
""",
    "uv": "sys.exit(int(os.environ.get('STUB_UV_RC', '0')))\n",
    "workqueue": _LOG
    + """
log("workqueue", sys.argv[1:])
q = pathlib.Path(os.environ["STUB_QUEUE"])
data = json.loads(q.read_text()) if q.exists() else {"items": []}
cmd = sys.argv[1]
if cmd == "ready":
    # The real workqueue's shape (workqueue.cmd_ready): {"base_sha", "ready": [rows]}.
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else None
    rows = [dict(i, ready=True) for i in data["items"] if i["status"] == "ready"]
    print(json.dumps({"base_sha": base, "ready": rows}))
elif cmd == "set-status":
    for i in data["items"]:
        if i["id"] == sys.argv[2]:
            i["status"] = sys.argv[3]
            if "--attempt" in sys.argv:
                i["attempt"] = i.get("attempt", 1) + 1
    q.write_text(json.dumps(data))
elif cmd == "digest":
    print("\\n".join(f"- {i['id']}: {i['status']}" for i in data["items"]))
""",
    "lint_brief": _LOG
    + """
log("lint_brief", sys.argv[1:])
sys.exit(int(os.environ.get("STUB_LINT_RC", "0")))
""",
    "lanes": "print(json.dumps({'cpu': {'free': 1}}))\n",
    "slot": _LOG
    + """
import subprocess
log("slot", sys.argv[1:])
a = sys.argv[1:]
cut = a.index("--")
opts, cmd = a[:cut], a[cut + 1:]
kv = {opts[i]: opts[i + 1] for i in range(1, len(opts) - 1, 2)}
root = pathlib.Path(os.environ["RSR_ORCH_ROOT"]) / ".orchestrator" / "jobs"
root.mkdir(parents=True, exist_ok=True)
rec = {"job_id": kv["--job-id"], "item_id": kv["--item"], "lane": kv["--lane"],
       "slots": int(kv["--slots"]), "cmd": cmd, "cwd": os.getcwd(),
       "pid": os.getpid(), "pinned_sha": kv["--pinned-sha"], "start": "t0",
       "end": None, "rc": None, "status": "running"}
path = root / (kv["--job-id"] + ".json")
path.write_text(json.dumps(rec))
rc = subprocess.run(cmd).returncode
rec.update(end="t1", rc=rc, status="done")
path.write_text(json.dumps(rec))
sys.exit(rc)
""",
}


class Orch:
    def __init__(self, tmp: Path, monkeypatch, *, caps: tuple[str, str] = ("5", "20")):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.logs = tmp / "logs"
        self.bin.mkdir()
        self.logs.mkdir()
        for name, body in STUBS.items():
            p = self.bin / name
            p.write_text(_STUB_HEAD + body)
            p.chmod(0o755)
        self.queue_path = tmp / "queue.json"
        self.env_path = tmp / "orchestrator.env"
        self.write_env(*caps)
        env = {
            "RSR_ORCH_ROOT": str(tmp / "repo"),
            "RSR_ORCH_ENV": str(self.env_path),
            "RSR_CLAUDE_BIN": str(self.bin / "claude"),
            "RSR_GH_BIN": str(self.bin / "gh"),
            "RSR_UV_BIN": str(self.bin / "uv"),
            "RSR_ORCH_CMD_WORKQUEUE": str(self.bin / "workqueue"),
            "RSR_ORCH_CMD_LINT_BRIEF": str(self.bin / "lint_brief"),
            "RSR_ORCH_CMD_LANES": str(self.bin / "lanes"),
            "RSR_ORCH_CMD_SLOT": str(self.bin / "slot"),
            "RSR_NO_CAFFEINATE": "1",
            "RSR_NOW": NOW,
            "STUB_LOG_DIR": str(self.logs),
            "STUB_QUEUE": str(self.queue_path),
            "PYTHONPATH": str(SCRIPTS),
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        for k in ("DRY_RUN", "RSR_CYCLE_CAP", "RSR_NIGHT_CAP", "RSR_MAX_ATTEMPTS"):
            monkeypatch.delenv(k, raising=False)
        # 🔴 the one guarantee every test here rests on
        for key in ("RSR_CLAUDE_BIN", "RSR_GH_BIN"):
            assert Path(os.environ[key]).resolve().is_relative_to(tmp.resolve()), key
        self.root = tmp / "repo"
        self._make_repo()
        self.set_queue([])

    # -- setup ----------------------------------------------------------- #

    def write_env(self, cycle: str, night: str) -> None:
        self.env_path.write_text(
            f"RSR_CYCLE_CAP={cycle}\nRSR_NIGHT_CAP={night}\n"
            "RSR_WINDOW_OPENS=21:00\nRSR_LAST_DISPATCH=06:30\n"
            "RSR_PARK_BY=07:30\nRSR_DIGEST_BY=08:00\n"
        )

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return lc.git_out(cwd or self.root, *args)

    def _make_repo(self) -> None:
        origin = self.tmp / "origin.git"
        lc.git(self.tmp, "init", "--bare", "-b", "main", str(origin), check=True)
        lc.git(self.tmp, "init", "-b", "main", str(self.root), check=True)
        for k, v in {
            "user.name": "Test",
            "user.email": "test@example.invalid",
            "commit.gpgsign": "false",
        }.items():
            self.git("config", k, v)
        files = {
            ".gitignore": "/.worktrees/\n/.orchestrator/\n",
            "preregistration/e3.md": "threshold: 0.05\n",
            "runs/old/ledger.json": '{"v": 1}\n',
            "runs/canary/baseline.json": '{"loss": 1.0}\n',
            "docs/spec/spec.md": "spec\n",
            "docs/owner/ruling.md": "ruling\n",
            "src/rsr/constants.py": CONSTANTS_PY,
            ".claude/settings.researcher.json": "{}\n",
            ".claude/settings.orchestrator.json": "{}\n",
            "docs/lab-notes/dispatch-a.md": "# brief a\n",
            "docs/lab-notes/dispatch-b.md": "# brief b\n",
        }
        self.write(self.root, files)
        self.git("add", "-A")
        self.git("commit", "-m", "base")
        self.git("remote", "add", "origin", str(origin))
        self.git("push", "-q", "origin", "main")
        self.base = self.git("rev-parse", "HEAD")

    @staticmethod
    def write(where: Path, files: dict[str, str]) -> None:
        for rel, text in files.items():
            p = where / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def commit(self, wt: Path, files: dict[str, str], msg: str = "work") -> str:
        self.write(wt, files)
        self.git("add", "-A", cwd=wt)
        self.git("commit", "-q", "-m", msg, cwd=wt)
        return self.git("rev-parse", "HEAD", cwd=wt)

    def set_queue(self, items: list[dict]) -> None:
        self.queue_path.write_text(json.dumps({"items": items}))

    def item(self, iid: str = "a", status: str = "ready") -> dict:
        return {
            "id": iid,
            "class": "experiment",
            "lane": "cpu",
            "slots": 1,
            "brief": f"docs/lab-notes/dispatch-{iid}.md",
            "status": status,
        }

    def queue(self) -> dict[str, dict]:
        return {i["id"]: i for i in json.loads(self.queue_path.read_text())["items"]}

    # -- running --------------------------------------------------------- #

    def run(self, module: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", f"orchestrator.{module}", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            check=False,
        )

    def calls(self, name: str) -> list:
        p = self.logs / f"{name}.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line]

    def posts(self) -> list:
        return [
            c
            for c in self.calls("gh")
            if c[:2] in (["issue", "create"], ["issue", "comment"])
        ]

    def wait_result(self, n: int, timeout: float = 20.0) -> dict:
        path = lc.result_path(self.root, n)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return json.loads(path.read_text())
            time.sleep(0.05)
        raise AssertionError(f"cycle {n} wrote no result in {timeout}s")

    def wait_job(self, job_id: str, timeout: float = 20.0) -> dict:
        path = self.root / ".orchestrator" / "jobs" / f"{job_id}.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                rec = json.loads(path.read_text())
                if rec.get("status") != "running":
                    return rec
            time.sleep(0.05)
        raise AssertionError(f"job {job_id} did not finish in {timeout}s")

    def main_refs(self) -> tuple[str, str]:
        return (
            self.git("rev-parse", "refs/heads/main"),
            lc.git_out(self.tmp / "origin.git", "rev-parse", "refs/heads/main"),
        )


def dead_pid() -> int:
    """A pid that existed and has been reaped -- dead, not a zombie."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid
