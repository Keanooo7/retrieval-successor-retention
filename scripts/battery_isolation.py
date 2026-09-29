"""I1 (P1.1, 2026-09-29): the mutation battery never mutates the tree it is run from.

Until this module the battery wrote each mutation into the live checkout and put the
original back in a ``finally``. That held for exceptions and failed for everything
else: a stopped battery left a mutated ``loop.py`` behind once and a mutated
``lanes.py`` once (memory ``battery-operations``). And on 2026-09-27 a suite ran on
the main checkout's venv from inside ``_mgr`` -- the editable install imported
**main's** ``src`` -- so a green census certified a tree it never imported
(DIGEST cycle 16b, red-team M-12). Both are closed here, not by care:

1. **Shards.** Every suite run -- the baseline and each mutation -- happens in a
   *shard*: a detached git worktree at the pinned SHA (the invoking tree's HEAD,
   which must be clean), with its own ``.venv`` from ``uv sync --frozen --extra
   dev``. The pool lives OUTSIDE the invoking tree (``default_pool_dir``), so no
   path under it is ever written. Between mutations a shard is reset with
   ``git checkout -- .`` and ``git clean -ffdxq -e /.venv/`` -- ignored files
   INCLUDED, only the venv kept -- and must then show nothing but ``.venv/`` in
   ``git status --porcelain --ignored`` at the pinned SHA, or the battery refuses.
   (Review MAJOR-2: without ``-x`` an ignored ``runs/`` or ``.orchestrator/``
   file one mutation's suite wrote changed the next mutation's verdict.)
2. **Path assertion.** The shard's pytest writes a probe (``tests/_battery_probe.py``,
   env ``RSR_BATTERY_PROBE``) recording where ``rsr`` resolves in the pytest
   process itself, in a child it spawns with ``sys.executable``, and in the
   ``python`` on its ``PATH``. ``probe_problem`` refuses any of them outside the
   shard root -- and a missing probe. A refused probe makes that suite run
   ``DID_NOT_RUN``: never ``PROVEN``, never ``LEAKS``. So is a suite that did
   not complete: pytest killed by a signal, or exiting 2 (interrupted), 3
   (internal error) or 4 (usage) -- by its process status or by the
   ``exitstatus`` the probe records (``suite_problem``; review MAJOR-1).
3. **Per-shard state.** ``RSR_ORCH_ROOT`` is the shard root in the suite env, so
   an orchestrator write from a test lands in the shard, not the main checkout.
4. Signal handlers (SIGTERM, SIGHUP, SIGINT) remain only as defence in depth: they
   kill the suite's whole process group and tear the pool down. The live tree does
   not depend on them -- SIGKILL cannot be handled, and it leaves only shard
   worktrees, which ``--prune-shards`` removes (under the pool lock).

**Pool policy.** The default pool is ONE per repository (``default_pool_dir`` is
keyed on the git common dir), so every worktree of the repo shares it, and its
``flock`` admits one battery at a time: a second battery on the default pool exits
3 ("another battery holds the shard pool"). Concurrent batteries -- I2's shards,
or two worktrees' subsets -- must each pass their own ``--shard-dir``
(``scripts/battery_subset.py`` passes it through). ``--prune-shards`` takes the same
lock, so it can never remove a running battery's shard (review MAJOR-3).

Shards are the tool's own scratch at a pinned, committed SHA: removing one loses
nothing that is not in git. ``--keep-shards`` leaves them for reuse.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

PROBE_ENV = "RSR_BATTERY_PROBE"
"""The file the shard's pytest session writes its import paths to (`_battery_probe`)."""

SYNC_ARGV = ("uv", "sync", "--frozen", "--extra", "dev")
"""What builds a shard's venv. ``--extra dev`` because pytest is a dev extra: bare
``uv sync --frozen`` installs no pytest (measured 2026-09-29), and
``orchestrator.dispatch.sync_venv`` uses the same line."""

_VENV_ENV = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX", "PYTHONHOME")

_SUITE_STRIP = ("PYTHONPATH", "PYTHONPYCACHEPREFIX")
"""Never inherited by a shard's suite: ``PYTHONPATH`` can put another checkout's
``scripts/`` or ``experiments/`` ahead of the shard's with ``rsr`` still passing the
probe, and ``PYTHONPYCACHEPREFIX`` moves bytecode outside the shard, where the
reset cannot remove it (review MINOR-1)."""

COMPLETED_EXIT = (0, 1)
"""pytest exit statuses of a suite that ran to the end: all passed, or some failed."""

CLEAN_ARGV = ("clean", "-ffdxq", "-e", "/.venv/")
"""The shard reset's clean: untracked AND ignored files, nested repos, only the
venv kept (review MAJOR-2)."""


class Unisolated(Exception):
    """Isolation could not be established or verified: DID NOT RUN, never a verdict."""


def _git_bin() -> str:
    for cand in (os.environ.get("RSR_GIT"), "/opt/homebrew/bin/git"):
        if cand and Path(cand).exists():
            return cand
    found = shutil.which("git")
    if found is None:
        raise Unisolated("no git binary found (set RSR_GIT)")
    return found


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_git_bin(), "-C", str(cwd), *args], capture_output=True, text=True
    )


def _ok(proc: subprocess.CompletedProcess, what: str) -> str:
    if proc.returncode != 0:
        raise Unisolated(f"{what} -> rc={proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def is_under(path: str | Path | None, root: Path) -> bool:
    if path is None:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def pinned_sha(root: Path) -> str:
    """The invoking tree's HEAD -- refused if the tree differs from it.

    A shard is a checkout of a commit. A tracked edit or an untracked file in the
    invoking tree would not be in the shard, so the battery would certify a tree
    other than the one it was run from. Commit first.
    """
    sha = _ok(git(root, "rev-parse", "--verify", "HEAD"), "git rev-parse HEAD").strip()
    dirty = _ok(git(root, "status", "--porcelain"), "git status --porcelain")
    if dirty.strip():
        raise Unisolated(
            f"{root} has uncommitted changes; the battery runs the committed HEAD "
            f"{sha[:12]} in a shard and would not see them. Commit first.\n"
            + dirty.rstrip()
        )
    return sha


def common_dir(root: Path) -> Path:
    out = _ok(
        git(root, "rev-parse", "--path-format=absolute", "--git-common-dir"),
        "git rev-parse --git-common-dir",
    )
    return Path(out.strip()).resolve()


def default_pool_dir(root: Path) -> Path:
    """``~/.cache/rsr-battery/<repo key>``: outside every worktree of the repo."""
    key = hashlib.sha256(str(common_dir(root)).encode()).hexdigest()[:12]
    return Path.home() / ".cache" / "rsr-battery" / key


@dataclass
class Shard:
    root: Path
    sha: str
    index: int
    probe: Path
    timings: dict[str, float] = field(default_factory=dict)


def sync_venv(shard_root: Path) -> None:
    """``uv sync --frozen --extra dev`` in the shard, blind to any active venv."""
    env = {k: v for k, v in os.environ.items() if k not in _VENV_ENV}
    try:
        proc = subprocess.run(
            list(SYNC_ARGV), cwd=shard_root, capture_output=True, text=True, env=env
        )
    except OSError as e:
        raise Unisolated(
            f"{' '.join(SYNC_ARGV)} not runnable in {shard_root}: {e}"
        ) from e
    _ok(proc, f"{' '.join(SYNC_ARGV)} in {shard_root}")


def shard_env(shard: Shard, base: dict[str, str]) -> dict[str, str]:
    """The suite's environment in ``shard``: its venv first, its own orch root."""
    env = {k: v for k, v in base.items() if k not in _VENV_ENV}
    for k in _SUITE_STRIP:
        env.pop(k, None)
    venv = shard.root / ".venv"
    path = [
        p
        for p in env.get("PATH", "").split(os.pathsep)
        if p and not p.endswith(f"{os.sep}.venv{os.sep}bin")
    ]
    env["PATH"] = os.pathsep.join([str(venv / "bin"), *path])
    env["VIRTUAL_ENV"] = str(venv)
    env["RSR_ORCH_ROOT"] = str(shard.root)
    env[PROBE_ENV] = str(shard.probe)
    # 🔴 No bytecode cache in a shard. A .pyc is trusted when the source's mtime
    # (whole seconds) and size match, and a same-length mutation restored within
    # the same second passes both: the NEXT mutation's suite then ran the previous
    # mutation's code (measured 2026-09-29 on the stub battery: an OTHER=5 mutation
    # scored PROVEN on a VALUE gate). The in-place battery escaped this only because
    # a real suite takes minutes.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def probe_problem(probe: dict | None, root: Path) -> str | None:
    """Why this probe does not show a suite that imported ``rsr`` from ``root``.

    ``in_process`` (the pytest process) and ``child`` (a ``sys.executable`` child
    of it) must both resolve under ``root``. ``path_python`` (the ``python`` on the
    suite's PATH) must too, if it resolves ``rsr`` at all.
    """
    if probe is None:
        return "no probe: the suite did not report where it imported rsr from"
    if probe.get("exitstatus") not in COMPLETED_EXIT:
        return (
            f"pytest's own exit status was {probe.get('exitstatus')!r} (2 interrupted, "
            f"3 internal error, 4 usage error): the suite did not complete"
        )
    for key in ("in_process", "child"):
        if not is_under(probe.get(key), root):
            return f"rsr.__file__ ({key}) = {probe.get(key)!r} is not under {root}"
    other = probe.get("path_python")
    if other is not None and not is_under(other, root):
        return f"rsr.__file__ (path_python) = {other!r} is not under {root}"
    return None


def suite_problem(returncode: int, probe: dict | None, root: Path) -> str | None:
    """Why this suite run is not evidence, or None. Review MAJOR-1.

    A pytest killed by a signal, or one that exited 2/3/4, did not run the suite
    to the end: a SIGINT that lands after the gate test failed would otherwise
    score PROVEN with every later test -- every possible leak -- unrun.
    """
    if returncode < 0:
        return f"pytest was killed by signal {-returncode}"
    if returncode not in COMPLETED_EXIT:
        return (
            f"pytest exited {returncode} (2 interrupted, 3 internal error, 4 usage "
            f"error): the suite did not complete"
        )
    return probe_problem(probe, root)


def read_probe(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def check_interpreter(shard: Shard, env: dict[str, str]) -> None:
    """Before any suite: the shard's own python must import ``rsr`` from the shard."""
    py = shard.root / ".venv" / "bin" / "python"
    code = (
        "import importlib.util as u, json;"
        "s = u.find_spec('rsr'); print(json.dumps(s.origin if s else None))"
    )
    try:
        proc = subprocess.run(
            [str(py), "-c", code], cwd=shard.root, capture_output=True, text=True, env=env
        )
    except OSError as e:
        raise Unisolated(f"{py} not runnable: {e}") from e
    origin = json.loads(_ok(proc, f"{py} -c find_spec('rsr')").strip() or "null")
    if not is_under(origin, shard.root):
        raise Unisolated(
            f"shard {shard.root}: its python resolves rsr to {origin!r}, not under "
            f"the shard (split-brain)"
        )


def reset_shard(shard: Shard) -> None:
    """Back to the pinned tree: tracked files restored, untracked AND ignored removed.

    Only ``.venv/`` survives (``CLEAN_ARGV``). Verified, not assumed: HEAD is the
    pinned SHA and ``git status --porcelain --ignored`` lists nothing but
    ``.venv/``, or the battery refuses.
    """
    _ok(git(shard.root, "checkout", "--", "."), f"git checkout -- . in {shard.root}")
    _ok(git(shard.root, *CLEAN_ARGV), f"git clean in {shard.root}")
    purge_bytecode(shard.root)
    verify_clean(shard)


def purge_bytecode(root: Path) -> None:
    """Remove every ``__pycache__`` in the tree outside ``.venv``/``.git``.

    Belt to ``PYTHONDONTWRITEBYTECODE``'s braces (`shard_env`): a cache written by
    anything else -- a test that sets its own env, a kept shard -- must not carry
    one mutation's code into the next suite.
    """
    for dirpath, dirnames, _files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".venv", ".git")]
        if "__pycache__" in dirnames:
            shutil.rmtree(Path(dirpath) / "__pycache__", ignore_errors=True)
            dirnames.remove("__pycache__")


def verify_clean(shard: Shard) -> None:
    head = _ok(git(shard.root, "rev-parse", "HEAD"), "git rev-parse HEAD").strip()
    if head != shard.sha:
        raise Unisolated(f"shard {shard.root} is at {head[:12]}, not {shard.sha[:12]}")
    out = _ok(
        git(shard.root, "status", "--porcelain", "--ignored"),
        "git status --porcelain --ignored",
    )
    dirty = [ln for ln in out.splitlines() if ln.strip() and ln != "!! .venv/"]
    if dirty:
        raise Unisolated(f"shard {shard.root} is not clean:\n" + "\n".join(dirty))


def _registered(source: Path) -> set[Path]:
    out = _ok(git(source, "worktree", "list", "--porcelain"), "git worktree list")
    return {
        Path(line.split(" ", 1)[1]).resolve()
        for line in out.splitlines()
        if line.startswith("worktree ")
    }


def make_shard(source: Path, pool: Path, index: int, sha: str) -> Shard:
    """A worktree at ``sha`` under ``pool``, synced and interpreter-checked.

    An existing directory is reused only if it is a registered worktree of this
    repository (``--keep-shards`` from an earlier run); anything else there is
    refused, never overwritten.
    """
    path = (pool / f"shard-{index}").resolve()
    shard = Shard(path, sha, index, pool / f"probe-{index}.json")
    t0 = time.perf_counter()
    if path.exists():
        if path not in _registered(source):
            raise Unisolated(f"{path} exists and is not a worktree of {source}")
        _ok(git(path, "checkout", "--detach", "--force", sha), f"checkout in {path}")
        _ok(git(path, *CLEAN_ARGV), f"git clean in {path}")
        purge_bytecode(path)
    else:
        _ok(
            git(source, "worktree", "add", "--detach", str(path), sha),
            f"git worktree add {path}",
        )
    t1 = time.perf_counter()
    sync_venv(path)
    t2 = time.perf_counter()
    verify_clean(shard)
    check_interpreter(shard, shard_env(shard, dict(os.environ)))
    shard.timings = {"worktree_s": t1 - t0, "sync_s": t2 - t1}
    return shard


def remove_shard(source: Path, path: Path) -> str | None:
    """Unregister and remove one shard worktree; returns an error line or None."""
    proc = git(source, "worktree", "remove", "--force", str(path))
    if proc.returncode != 0 and path.exists():
        return f"git worktree remove {path}: {proc.stderr.strip()}"
    return None


@contextlib.contextmanager
def pool_lock(pool: Path) -> Iterator[None]:
    """Exclusive ``flock`` on ``pool/pool.lock``, or ``Unisolated``. Dies with the
    process, so a SIGKILLed battery never wedges the pool."""
    pool.mkdir(parents=True, exist_ok=True)
    lock = (pool / "pool.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise Unisolated(f"another battery holds the shard pool {pool}") from None
    try:
        yield
    finally:
        lock.close()


def prune_shards(source: Path, pool: Path) -> list[str]:
    """Remove every worktree of this repo registered under ``pool`` -- under its lock.

    The documented cleanup after a SIGKILL (which no handler can see). Refused
    (``Unisolated``) while a battery holds the pool: the default pool is shared by
    every worktree of the repo, so an unlocked prune would delete another
    battery's live shard mid-suite (review MAJOR-3).
    """
    pool = pool.resolve()
    with pool_lock(pool):
        return _prune_unlocked(source, pool)


def _prune_unlocked(source: Path, pool: Path) -> list[str]:
    """``prune_shards`` for a caller that already holds the pool lock."""
    pool = pool.resolve()
    errors = []
    for wt in sorted(_registered(source)):
        if is_under(wt, pool) and wt != pool:
            err = remove_shard(source, wt)
            if err:
                errors.append(err)
    return errors


@contextlib.contextmanager
def open_pool(
    source: Path,
    sha: str,
    n: int = 1,
    pool: Path | None = None,
    keep: bool = False,
) -> Iterator[list[Shard]]:
    """``n`` shards at ``sha``, exclusively held; torn down on exit unless ``keep``.

    The pool is locked with ``flock`` so two batteries never share a shard; the
    lock dies with the process, so a SIGKILLed battery never wedges it.
    """
    pool = (pool or default_pool_dir(source)).resolve()
    if is_under(pool, source):
        raise Unisolated(f"shard pool {pool} is inside the invoking tree {source}")
    with pool_lock(pool):
        shards: list[Shard] = []
        try:
            for k in range(n):
                shards.append(make_shard(source, pool, k, sha))
            yield shards
        finally:
            if not keep:
                for s in shards:
                    remove_shard(source, s.root)
                _prune_unlocked(source, pool)
