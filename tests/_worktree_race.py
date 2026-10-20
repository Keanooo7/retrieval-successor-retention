"""The git worktree race at N=6, through the battery's own calls (review BLOCKER-1).

    .venv/bin/python tests/_worktree_race.py [ROUNDS] [K]

Builds a scratch repository in a temporary directory, with one permanent worktree
(the real repository's topology: ``.git/worktrees`` never empties). Each round,
``K`` threads (default 6, as N=6 shards starting at once) each add a detached
worktree through ``battery_isolation.admin_git`` and then remove it the same way,
all concurrently. ``admin_git`` opens the lock file afresh per call, and macOS
``flock`` locks an open file description, so the threads exclude each other
exactly as separate processes do.

Prints ``failed adds: F / ROUNDS*K`` and ``failed removes: F / ROUNDS*K``, then each
distinct git error with its count. With the lock the expected count is 0; with
the lock removed (the ``i1: the worktree admin lock is a no-op`` mutation) git
2.55.0 lost 2-7% of rounds in the review's plain-git simulation. Touches nothing
outside its temporary directory. Not collected by pytest: it is a measurement,
not a test.
"""

from __future__ import annotations

import collections
import re
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import battery_isolation as iso  # noqa: E402


def main(rounds: int, k: int) -> int:
    fails = {"add": 0, "remove": 0}
    errors: collections.Counter[str] = collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        repo = t / "r"
        repo.mkdir()
        (repo / "f").write_text("x\n")
        for args in (
            ("init", "-q"),
            ("add", "f"),
            ("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "x"),
            ("worktree", "add", "-q", "--detach", str(t / "perm"), "HEAD"),
        ):
            assert iso.git(repo, *args).returncode == 0, args

        def op(verb: str, *args: str) -> None:
            barrier.wait()  # all K start together, as N shards launched at once do
            proc = iso.admin_git(repo, "worktree", verb, *args)
            if proc.returncode != 0:
                with lock:
                    fails[verb] += 1
                    msg = re.sub(r"shard-\d+-\d+", "shard-R-K", proc.stderr.strip())
                    errors[re.sub(r"shard-R-K\d*", "shard-R-KN", msg)] += 1

        lock = threading.Lock()
        barrier = threading.Barrier(k)
        for r in range(rounds):
            paths = [t / "s" / f"shard-{r}-{j}" for j in range(k)]
            for verb, argv in (
                ("add", lambda p: ("--detach", str(p), "HEAD")),
                ("remove", lambda p: ("--force", str(p))),
            ):
                threads = [
                    threading.Thread(target=op, args=(verb, *argv(p))) for p in paths
                ]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
    print(f"failed adds: {fails['add']} / {rounds * k}")
    print(f"failed removes: {fails['remove']} / {rounds * k}")
    for msg, n in sorted(errors.items()):
        print(f"{n:5d} {msg.splitlines()[-1] if msg else '<no stderr>'}")
    return 1 if any(fails.values()) else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(int(a[0]) if a else 200, int(a[1]) if len(a) > 1 else 6))
