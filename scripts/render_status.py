"""Check and emit `docs/status/status.json`, the data behind the research map.

`status.json` is curated text -- what each item tests, where it stands, what blocks
it -- plus numbers. **The numbers are not the curation's to choose.** Every entry in
an item's `numbers` names a ledger and a key, and `--check` resolves that key and
refuses any value that is not *equal* to the ledger's (no rounding: the page may
round for display, the data may not). Every evidence path must exist in the repo.

    uv run python scripts/render_status.py --check
    uv run python scripts/render_status.py --emit-js /tmp/rsr-status.js

`--emit-js` writes `window.RSR_STATUS = {...};` for the HTML page, and refuses to
write if `--check` would fail: a page must never show a number its ledger does not
hold.

Key resolution is `render_scoreboard._flatten`, the same one the prose audit's
`{{run_id:key}}` tokens use: a row key gives an observation's `value` or a
statistic's `mean`; `<key>.sd` / `.n` / `.samples` address the rest; dotted paths
reach into fields (`verdict.outcome`, `config.iters`).

Two content tripwires, cheap and deliberate. No string may mention spec §15 (it
must not be filled or restated by a model -- CLAUDE.md), and no item may use a
status outside the five this page knows.

Exit codes (`rsr.exit_codes`): `0` every check passed · `1` a mismatch, listed ·
`3` status.json missing or unreadable (did not run).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from render_scoreboard import _flatten  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main  # noqa: E402

STATUS = _REPO / "docs" / "status" / "status.json"
STATUSES = ("done", "in_progress", "blocked", "not_started", "out_of_scope")
REQUIRED = (
    "generated_at_sha",
    "claim",
    "workstreams",
    "items",
    "retracted",
    "owner_digest",
    "timeline",
    "environment",
)
#: Spec §15 must not be filled, restated or pointed at by a model (CLAUDE.md).
FORBIDDEN = ("§15",)


def _strings(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for k, x in v.items() for s in [k, *_strings(x)]]
    if isinstance(v, list):
        return [s for x in v for s in _strings(x)]
    return []


def _equal(actual: Any, claimed: Any) -> bool:
    """Exact equality. A bool is not a number; a float must match to the bit."""
    if isinstance(actual, bool) or isinstance(claimed, bool):
        return actual is claimed
    if isinstance(actual, (int, float)) and isinstance(claimed, (int, float)):
        if isinstance(actual, float) and math.isnan(actual):
            return False
        return float(actual) == float(claimed)
    return actual == claimed


def _evidence_paths(doc: dict) -> list[tuple[str, str]]:
    """`(where, path)` for every evidence path the document cites."""
    out: list[tuple[str, str]] = []
    for it in doc.get("items", []):
        for e in it.get("evidence", []):
            out.append((f"item {it.get('id')}", e.get("path", "")))
    for section in ("owner_digest", "rulings"):
        for entry in doc.get(section, []):
            for e in entry.get("evidence", []):
                out.append(
                    (f"{section} {entry.get('id', entry.get('ruling'))}", e["path"])
                )
    for r in doc.get("retracted", []):
        out.append(("retracted", r.get("source", "").split(" §")[0]))
    return out


def check(doc: dict, repo: Path | None = None) -> list[str]:
    """Every problem with `doc`, as one line each. Empty means it checks."""
    repo = repo or _REPO
    problems: list[str] = []

    for k in REQUIRED:
        if k not in doc:
            problems.append(f"missing top-level field {k!r}")

    ws_ids = {w.get("id") for w in doc.get("workstreams", [])}
    item_ids = [it.get("id") for it in doc.get("items", [])]
    dup = sorted({i for i in item_ids if item_ids.count(i) > 1})
    if dup:
        problems.append(f"duplicate item ids: {dup}")

    for it in doc.get("items", []):
        iid = it.get("id")
        if it.get("status") not in STATUSES:
            problems.append(f"item {iid}: status {it.get('status')!r} not in {STATUSES}")
        if it.get("workstream") not in ws_ids:
            problems.append(f"item {iid}: workstream {it.get('workstream')!r} undeclared")
        if not isinstance(it.get("owner_decision"), bool):
            problems.append(f"item {iid}: owner_decision must be a bool")

    for where, path in _evidence_paths(doc):
        if not path or not (repo / path).exists():
            problems.append(f"{where}: evidence path {path!r} does not exist")

    flats: dict[str, dict[str, Any] | None] = {}
    for it in doc.get("items", []):
        for n in it.get("numbers", []):
            ledger, key = n.get("ledger", ""), n.get("key", "")
            where = f"item {it.get('id')} number {n.get('label')!r}"
            if ledger not in flats:
                try:
                    flats[ledger] = _flatten(json.loads((repo / ledger).read_text()))
                except (OSError, ValueError):
                    flats[ledger] = None
            flat = flats[ledger]
            if flat is None:
                problems.append(f"{where}: ledger {ledger!r} missing or unreadable")
                continue
            if key not in flat:
                problems.append(f"{where}: key {key!r} not in {ledger}")
                continue
            if not _equal(flat[key], n.get("value")):
                problems.append(
                    f"{where}: status.json says {n.get('value')!r}, "
                    f"{ledger}:{key} says {flat[key]!r}"
                )

    for tl in doc.get("timeline", []):
        for iid in tl.get("items", []):
            if iid not in item_ids:
                problems.append(f"timeline {tl.get('phase')!r}: unknown item {iid!r}")
        md = tl.get("measured_duration")
        if md is not None and not (isinstance(md, dict) and md.get("source")):
            problems.append(f"timeline {tl.get('phase')!r}: a duration needs a source")

    for s in _strings(doc):
        for bad in FORBIDDEN:
            if bad in s:
                problems.append(f"forbidden reference {bad!r} in: {s[:80]!r}")
    return problems


def emit_js(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(doc, indent=2, ensure_ascii=False)
    path.write_text(f"window.RSR_STATUS = {body};\n")


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description="check / emit docs/status/status.json")
    ap.add_argument("--status", type=Path, default=STATUS)
    ap.add_argument("--check", action="store_true", help="validate paths and numbers")
    ap.add_argument("--emit-js", type=Path, default=None, metavar="PATH")
    args = ap.parse_args(argv)
    if not (args.check or args.emit_js):
        return did_not_run("nothing to do: pass --check and/or --emit-js PATH")

    if not args.status.exists():
        return did_not_run(f"{args.status} does not exist")
    try:
        doc = json.loads(args.status.read_text())
    except (OSError, ValueError) as exc:
        return did_not_run(f"{args.status} is unreadable: {exc}")

    problems = check(doc)
    if problems:
        print(f"{len(problems)} problems in {args.status}:")
        for p in problems:
            print(f"  {p}")
        if args.emit_js:
            print(f"refusing to write {args.emit_js}: the data does not check")
        return Exit.FAIL

    n_numbers = sum(len(it.get("numbers", [])) for it in doc["items"])
    print(
        f"OK: {len(doc['items'])} items; every evidence path exists; "
        f"all {n_numbers} numbers equal their ledger keys"
    )
    if args.emit_js:
        emit_js(doc, args.emit_js)
        print(f"wrote {args.emit_js}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
