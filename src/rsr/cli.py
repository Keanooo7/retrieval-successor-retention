"""`rsr` command line entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsr", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("constants", help="inspect the constants registry")
    c.add_argument("name", nargs="?", help="constant to read; omit to list all")
    c.add_argument("--scope", default=None)

    args = parser.parse_args(argv)

    if args.command == "constants":
        from rsr.constants import REGISTRY, ConstantError

        if args.name is None:
            for name, d in sorted(REGISTRY.definitions.items()):
                src = f" <- {d.source_experiment}" if d.source_experiment else ""
                print(f"{name:16s} {d.klass.value:12s} section {d.spec_ref}{src}")
            return 0
        try:
            print(REGISTRY.get(args.name, args.scope))
        except ConstantError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
