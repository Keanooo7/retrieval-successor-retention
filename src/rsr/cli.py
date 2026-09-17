"""``rsr`` -- the project's command line.

Every subcommand that reports a number **measures it**; none accepts one as an argument. A human
reading a terminal and retyping a number into a command is the step where 4180 becomes 4108.
"""

from __future__ import annotations

import argparse
import sys

from rsr.constants import REGISTRY
from rsr.gates.floors import Direction, Exit, check_floor, measure_pytest_count, record_floor


def _cmd_floor(args: argparse.Namespace) -> int:
    if args.record:
        key, _, raw = args.record.partition("=")
        record_floor(
            key, float(raw), direction=Direction.MAY_RISE,
            evidence=args.evidence or "rsr floor --record",
        )
        print(f"banked {key} = {raw}")
        return Exit.OK

    measured = measure_pytest_count()
    result = check_floor("pytest_count", measured, direction=Direction.MAY_RISE,
                         store="test-floor.json")
    print(result)
    if result.code is Exit.UNBANKED_RISE and args.bank:
        record_floor("pytest_count", float(measured), direction=Direction.MAY_RISE,
                     evidence="uv run pytest -q", store="test-floor.json")
        print(f"banked pytest_count = {measured}")
        return Exit.OK
    # An unbanked rise is not a CI failure: it is a rise. Report it and pass.
    return Exit.OK if result.code in (Exit.OK, Exit.UNBANKED_RISE, Exit.UNKNOWN) else result.code


def _cmd_constants(args: argparse.Namespace) -> int:
    unset = REGISTRY.unset()
    if args.check:
        result = check_floor("unmeasured_constants", float(len(unset)),
                             direction=Direction.MAY_FALL)
        print(result)
        if result.code is Exit.UNKNOWN:
            record_floor("unmeasured_constants", float(len(unset)),
                         direction=Direction.MAY_FALL, evidence="rsr constants --check")
            print(f"seeded floor at {len(unset)}")
            return Exit.OK
        return Exit.OK if result.code in (Exit.OK, Exit.UNBANKED_RISE) else result.code

    width = max(len(r["name"]) for r in REGISTRY.table())
    for row in REGISTRY.table():
        mark = "set" if row["set"] else "UNSET"
        owed = f"  <- owed by {row['experiment']}" if not row["set"] else ""
        print(f"  {row['name']:<{width}}  {row['class']:<12} {mark:<6} {row['source']}{owed}")
    print(f"\n  {len(unset)} still unset: {', '.join(unset) if unset else '(none)'}")
    return Exit.OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsr", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("floor", help="measure and check the pytest-count ratchet")
    f.add_argument("--check", action="store_true", help="(default) measure and compare")
    f.add_argument("--bank", action="store_true", help="update the floor on a rise")
    f.add_argument("--record", metavar="KEY=VALUE")
    f.add_argument("--evidence", metavar="TEXT")
    f.set_defaults(fn=_cmd_floor)

    c = sub.add_parser("constants", help="show the registry, or ratchet the unset count")
    c.add_argument("--check", action="store_true")
    c.set_defaults(fn=_cmd_constants)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
