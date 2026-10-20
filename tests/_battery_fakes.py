"""A stand-in for the battery's shard workspace, for tests of its verdict logic.

`scripts/mutation_battery.py::main` runs every suite in a shard worktree (I1). The
tests of what it does with a suite's result -- rows, reasons, exit codes -- do not
need a real worktree, and must not create one under ``~/.cache`` from inside the
suite. This yields ``root`` as the shard and makes the reset a no-op; the real
isolation is tested end to end in ``tests/test_battery_isolation.py``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import battery_isolation as iso


def use_fake_shard(monkeypatch, mb, root: Path) -> iso.Shard:
    shard = iso.Shard(root, "0" * 40, 0, root / ".battery-probe.json")
    monkeypatch.setattr(
        mb, "open_workspace", lambda args: contextlib.nullcontext([shard])
    )
    monkeypatch.setattr(mb, "reset", lambda shard: None)
    return shard
