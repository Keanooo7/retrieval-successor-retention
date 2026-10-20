"""Print one ledger row's value (optionally a dotted sub-path), in the register's
comparison format. Usage: ledger_get.py <ledger.json> <key> [dotted.path]

A key that occurs more than once resolves to its LAST occurrence. Exit 3 if the key
or path is missing (did not run is not found nothing)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    d = C.jload(argv[0])
    rows = {}
    for r in d.get("rows") or []:
        if isinstance(r, dict) and isinstance(r.get("key"), str):
            rows[r["key"]] = C.row_value(r)
    if argv[1] not in rows:
        print(f"MISSING key {argv[1]!r}", file=sys.stderr)
        return 3
    try:
        v = C.dig(rows[argv[1]], argv[2] if len(argv) == 3 else None)
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(f"MISSING path {argv[2:]!r}: {e!r}", file=sys.stderr)
        return 3
    print(C.fmt_exact(v))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
