"""Run scripts/mutation_battery.py's own main() with MUTATIONS filtered by name.

    python battery_subset.py ROOT OUT.json [--shard-dir DIR] NAME [NAME ...]

``--shard-dir`` is passed through to the battery (I1 pool policy): the default
shard pool is one per repository and admits one battery at a time, so a subset
run beside another battery -- another worktree's, or one of I2's shards -- needs
its own pool.
"""

import sys


def parse(argv: list[str]) -> tuple[str, str, list[str], list[str]]:
    """``(root, out, names, extra battery args)`` from this script's argv[1:]."""
    root, out, rest = argv[0], argv[1], list(argv[2:])
    extra: list[str] = []
    if rest[:1] == ["--shard-dir"]:
        extra, rest = ["--shard-dir", rest[1]], rest[2:]
    return root, out, rest, extra


if __name__ == "__main__":
    root, out, names, extra = parse(sys.argv[1:])
    sys.path.insert(0, root + "/scripts")
    import mutation_battery as mb

    sel = tuple(m for m in mb.MUTATIONS if m.name in names)
    missing = set(names) - {m.name for m in sel}
    if missing or len(sel) != len(names):
        sys.exit(f"subset does not resolve: missing={sorted(missing)} n={len(sel)}")
    mb.MUTATIONS = sel
    sys.argv = ["mutation_battery.py", "--json", out, *extra]
    mb.run_main(mb.main)
