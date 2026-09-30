"""I2 fix round 1 (review BLOCKER-1): git worktree administration under concurrency.

`git worktree add` writes ``.git/worktrees/<id>/commondir`` non-atomically. Any
``git worktree add``, ``list`` or ``remove`` that reads it while it is still empty
dies with ``fatal: failed to read .git/worktrees/<id>/commondir: Undefined error:
0`` (git 2.55.0; 2-7% of 6-way concurrent adds on a scratch repo, and the I2
end-to-end test's 1-in-~150 flake). Inside the battery's own suite that flake was
a false undeclared red -- a LEAKS verdict for an unrelated mutation.

The race is made DETERMINISTIC here: a holder process takes the battery's
worktree-admin lock, leaves exactly the state an in-flight ``worktree add`` leaves
(an admin dir with an empty ``commondir``), holds it briefly, then completes and
releases. Each battery worktree call made meanwhile must wait for the lock and then
succeed. Without the lock it reads the half-written entry and dies -- every time.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

import battery_isolation as iso  # noqa: E402

from _battery_stub import build_stub, stub_sync  # noqa: E402

LOCK_NAME = "rsr-worktree-admin.lock"

HOLDER = r"""
import fcntl, sys, time
from pathlib import Path
lock, half, ready, done = map(Path, sys.argv[1:5])
with lock.open("a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    # exactly what `git worktree add` leaves while it is still writing
    half.mkdir(parents=True)
    (half / "gitdir").write_text(str(half) + "-checkout/.git\n")
    (half / "commondir").write_text("")
    ready.write_text("")
    time.sleep(float(sys.argv[5]))
    # the add completes (here: is abandoned) before the lock is released
    for p in half.iterdir():
        p.unlink()
    half.rmdir()
    done.write_text("")
"""


@pytest.fixture
def stub(tmp_path):
    return build_stub(tmp_path / "stub").resolve()


class _AddInFlight:
    """A worktree add in flight, holding the admin lock, for ``hold`` seconds."""

    def __init__(self, repo: Path, tmp: Path, hold: float = 0.5):
        common = iso.common_dir(repo)
        self.args = [
            str(common / LOCK_NAME),
            str(common / "worktrees" / "inflight"),
            str(tmp / "holder.ready"),
            str(tmp / "holder.done"),
            str(hold),
        ]
        self.ready, self.done = Path(self.args[2]), Path(self.args[3])

    def __enter__(self):
        self.proc = subprocess.Popen([sys.executable, "-c", HOLDER, *self.args])
        end = time.monotonic() + 30
        while not self.ready.exists():
            assert self.proc.poll() is None, "the holder died before taking the lock"
            assert time.monotonic() < end, "the holder never took the lock"
            time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self.proc.wait(timeout=30)


def test_the_lock_file_is_in_the_git_common_dir(stub):
    assert iso.ADMIN_LOCK == LOCK_NAME
    wt = stub.parent / "other"
    iso.git(stub, "worktree", "add", "--detach", str(wt), "HEAD")
    # every worktree of the repository names the same lock
    assert iso.common_dir(wt) == iso.common_dir(stub)
    iso.git(stub, "worktree", "remove", "--force", str(wt))


def test_worktree_admin_list_waits_for_an_add_in_flight(stub, tmp_path):
    with _AddInFlight(stub, tmp_path) as held:
        registered = iso._registered(stub)  # Unisolated: `failed to read commondir`
        assert held.done.exists(), "the list ran while the add was in flight"
    assert registered == {stub}


def test_worktree_admin_add_waits_for_an_add_in_flight(stub, tmp_path, monkeypatch):
    monkeypatch.setattr(iso, "sync_venv", stub_sync("good"))
    sha = iso.pinned_sha(stub)
    pool = tmp_path / "pool"
    pool.mkdir()
    with _AddInFlight(stub, tmp_path) as held:
        shard = iso.make_shard(stub, pool, 0, sha)  # Unisolated on the same read
        assert held.done.exists(), "the add ran while another add was in flight"
    assert iso.remove_shard(stub, shard.root) is None
    assert iso._registered(stub) == {stub}


def test_worktree_admin_remove_waits_for_an_add_in_flight(stub, tmp_path):
    wt = tmp_path / "wt"
    assert iso.git(stub, "worktree", "add", "--detach", str(wt), "HEAD").returncode == 0
    with _AddInFlight(stub, tmp_path) as held:
        err = iso.remove_shard(stub, wt)  # `failed to read commondir`, left in place
        assert held.done.exists(), "the remove ran while an add was in flight"
    assert err is None and not wt.exists()
    assert iso._registered(stub) == {stub}
