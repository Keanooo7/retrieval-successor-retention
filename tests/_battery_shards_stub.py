"""Run the REAL sharded driver (`scripts/battery_shards.py`) on the I1 stub repo.

``python _battery_shards_stub.py drive STUB MODE NAMES [driver args...]`` points
``mutation_battery.ROOT`` at the stub, sets ``MUTATIONS`` to the named rows of
``_battery_stub.TABLE`` and builds shard venvs with ``stub_sync`` -- exactly as
``_battery_stub.drive`` does for the serial battery -- and then runs
``battery_shards.main``. ``SHARD_PREFIX`` is pointed back at this file, so every
shard child the driver spawns is stub-configured the same way. Everything else --
the partition, the per-shard pools, the done records, the merge, the exit code --
is the real code (PLAN-v4 §4 I2).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from _battery_stub import TABLE, stub_sync  # noqa: E402


def drive(stub: Path, mode: str, names: list[str], argv: list[str]) -> None:
    import battery_isolation as iso
    import battery_shards as bs
    import mutation_battery as mb

    mb.ROOT = stub.resolve()
    mb.MUTATIONS = tuple(mb.Mutation(n, *TABLE[n], why=n) for n in names)
    iso.sync_venv = stub_sync(mode)
    bs.SHARD_PREFIX = [
        sys.executable,
        str(Path(__file__).resolve()),
        "drive",
        str(stub),
        mode,
        ",".join(names),
    ]
    bs.run_main(lambda: bs.main(argv))


if __name__ == "__main__":
    # drive STUB MODE NAMES(comma-separated) [driver args...]
    _, cmd, stub_, mode_, names_, *rest = sys.argv
    assert cmd == "drive", cmd
    drive(Path(stub_), mode_, [n for n in names_.split(",") if n], rest)
