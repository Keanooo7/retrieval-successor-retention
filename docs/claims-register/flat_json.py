"""Print a JSON file as ONE line of flattened JSON, for register checks that pick a key.

Usage: flat_json.py <file.json>

- A ledger with a ``rows`` list: each row becomes ``"<row key>|<dotted path into the
  row, minus 'key'>"`` (so ``"e0d.class|value"``, ``"primary.H|samples"``,
  ``"seed0.analysis|value.populations.all.n_cells"``). A key that occurs more than once
  resolves to its LAST occurrence. Top-level non-``rows`` fields are flattened plainly.
- Any other JSON: dicts flattened with ``.``; lists are leaves (not flattened).

Reads only; writes nothing."""

from __future__ import annotations

import json
import sys


def flatten(x, prefix: str, out: dict) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    else:
        out[prefix] = x


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[0]) as fh:
        d = json.load(fh)
    out: dict = {}
    if isinstance(d, dict) and isinstance(d.get("rows"), list):
        for k, v in d.items():
            if k != "rows":
                flatten(v, k, out)
        for r in d["rows"]:
            if isinstance(r, dict) and isinstance(r.get("key"), str):
                stale = [k for k in out if k.startswith(r["key"] + "|")]
                for k in stale:
                    del out[k]
                body = {k: v for k, v in r.items() if k != "key"}
                for k, v in body.items():
                    flatten(v, f"{r['key']}|{k}", out)
    else:
        flatten(d, "", out)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
