"""Render the scoreboard and the morning artefact **from `runs/*/ledger.json`**.

**What this is the fix for.** Last night's ledgers were trustworthy and its prose
was not, and every defect lived in the layer between them -- transcription. Four
cycle-4 headline numbers (`10.817072550456`, `8.130104700724`,
`507.4735412597656`) appear in no ledger; "50 escalations" was 54; "3 canaries, all
held" was 2 survived and 1 inconclusive, because `runs/canary/cycle-04/ledger.json`
wrote the baseline rather than comparing against one and says `inconclusive` in so
many words.

So the morning artefact is **generated, not written**. A number that is not a
ledger key cannot appear in it. Every count printed here is `len()` of something;
none is typed.

⚠️ **`cycle` is not a usable join key and this module refuses to use it.** The
experiment ledgers' `cycle` field runs one behind the scoreboard's numbering
(researchers could not see the manager's counter), `cycle-zscore-denominator`
collides with `canary/cycle-04` on `4`, and `cycle-arm-identity` has
`"cycle": null`. **The join key is `run_id`.**

    S=scripts/render_scoreboard.py
    uv run python $S --over runs/
    uv run python $S --over runs/ --artefact Projects/RSR/overnight-2026-09-19.md
    uv run python $S --over runs/ --claim canary/cycle-04:survived
    uv run python $S --over runs/ --audit Projects/RSR/overnight-2026-09-18.md

Exit codes follow the run protocol: `0` pass · `1` real failure · `2` nothing to
compare · `3` did not run. 🔴 `3` must never collapse to `0`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from ledger import _exists_at_sha, entry_point  # noqa: E402

OUTCOMES = ("survived", "falsified", "inconclusive")


@dataclass
class Board:
    rows: list[dict[str, Any]] = field(default_factory=list)
    tally: dict[str, int] = field(default_factory=dict)
    cycle_collisions: dict[int, list[str]] = field(default_factory=dict)
    unreadable: list[dict[str, str]] = field(default_factory=list)


def _ledger_paths(runs_dir: Path) -> list[Path]:
    """Every `ledger.json` under `runs/`, at any depth -- the canaries live one
    level deeper than the experiments."""
    return sorted(runs_dir.rglob("ledger.json"))


def _rel_run_id(path: Path, runs_dir: Path) -> str:
    return path.parent.relative_to(runs_dir).as_posix()


def _reproducible(cmd: dict[str, Any], sha: str | None) -> bool:
    """Whether this command can be run again from the sha it was measured at.

    🔴 **Re-derived for ledgers written before cycle 0**, which carry no
    `reproducible` field. Treating a missing field as "unreproducible" would make
    the count 13 of 13; treating it as "fine" would make it 0. Neither is a
    measurement. Resolving the recorded `argv` against the recorded sha is.
    """
    if "reproducible" in cmd:
        return bool(cmd["reproducible"])
    ep = entry_point(cmd.get("argv", ""))
    if ep is None or sha is None:
        return False
    return _exists_at_sha(ep, sha)


def build(runs_dir: Path) -> Board:
    """One row per ledger, joined on `run_id`. No number here is typed."""
    board = Board()
    seen: dict[str, str] = {}
    by_cycle: dict[int, list[str]] = {}

    for p in _ledger_paths(runs_dir):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            board.unreadable.append({"path": str(p), "error": str(exc)})
            continue

        # The directory is the fact; `run_id` inside the file is a claim about it.
        run_id = _rel_run_id(p, runs_dir)
        if doc.get("run_id") not in (None, run_id):
            board.unreadable.append({
                "path": str(p),
                "error": f"run_id {doc['run_id']!r} disagrees with its directory "
                         f"{run_id!r}; joining on the directory",
            })
        if run_id in seen:
            board.unreadable.append({"path": str(p), "error": "duplicate run_id"})
            continue
        seen[run_id] = str(p)

        verdict = doc.get("verdict") or {}
        cmds = doc.get("commands", [])
        prov = doc.get("provenance", {})
        cycle = doc.get("cycle")
        if isinstance(cycle, int):
            by_cycle.setdefault(cycle, []).append(run_id)

        board.rows.append({
            "run_id": run_id,
            "cycle": cycle,
            "question": doc.get("question"),
            "falsifier": verdict.get("falsifier"),
            "outcome": verdict.get("outcome"),
            "detail": verdict.get("detail"),
            "capped_from": verdict.get("capped_from"),
            "status": doc.get("status"),
            "device": doc.get("device"),
            "seeds_actually_run": doc.get("seeds_actually_run"),
            "steps_requested": doc.get("steps_requested"),
            "steps_done": doc.get("steps_done"),
            "config_hash": doc.get("config_hash"),
            "git_sha": prov.get("git_sha"),
            "git_dirty": prov.get("dirty", prov.get("git_dirty")),
            "provenance_error": prov.get("provenance_error"),
            "n_commands": len(cmds),
            "n_rows": len(doc.get("rows", [])),
            "first_unreproducible": next(
                (c.get("argv") for c in cmds
                 if not _reproducible(c, prov.get("git_sha"))), None
            ),
            "n_null_exit": len([c for c in cmds if c.get("exit_code") is None]),
            "n_unreproducible": len(
                [c for c in cmds if not _reproducible(c, prov.get("git_sha"))]
            ),
        })

    board.cycle_collisions = {c: sorted(v) for c, v in by_cycle.items() if len(v) > 1}
    board.tally = {
        "total": len(board.rows),
        **{o: len([r for r in board.rows if r["outcome"] == o]) for o in OUTCOMES},
        "no_verdict": len([r for r in board.rows if r["outcome"] is None]),
        "unreadable": len(board.unreadable),
        "with_null_exit_code": len([r for r in board.rows if r["n_null_exit"]]),
        "not_re_executable": len(
            [r for r in board.rows if r["n_unreproducible"]]
        ),
    }
    return board


# --------------------------------------------------------------------------- #
# claims
# --------------------------------------------------------------------------- #

_CYCLE_SHAPED = re.compile(r"^cycle[-_ ]?(\d+)$", re.IGNORECASE)


def check_claim(runs_dir: Path, ident: str, outcome: str) -> tuple[bool, str]:
    """Is "`ident` = `outcome`" what the ledgers say?

    🔴 A cycle-shaped identifier is **refused, not resolved.** "cycle 4 survived" is
    the prose verdict this script exists to stop: at cycle 4 the ledgers are
    `canary/cycle-04` (`inconclusive`) and `cycle-zscore-denominator` (`survived`),
    and collapsing them is exactly the transcription that produced "3 canaries, all
    held".
    """
    board = build(runs_dir)
    by_id = {r["run_id"]: r for r in board.rows}

    if ident in by_id:
        actual = by_id[ident]["outcome"]
        if actual == outcome:
            return True, f"{ident}: ledger says {actual!r}"
        return False, (f"{ident}: prose claims {outcome!r}, ledger says {actual!r}")

    m = _CYCLE_SHAPED.match(ident.strip())
    if m:
        n = int(m.group(1))
        hits = [r["run_id"] for r in board.rows if r["cycle"] == n]
        return False, (
            f"`cycle` is not a join key -- refusing to resolve {ident!r}. "
            f"Cycle {n} is ambiguous across {len(hits)} ledgers: {sorted(hits)}. "
            f"Name a run_id."
        )
    return False, f"{ident!r}: no ledger under {runs_dir} has that run_id"


# --------------------------------------------------------------------------- #
# the prose audit
# --------------------------------------------------------------------------- #

# Skipped: section refs (§4.8), versions (v0.5), ISO dates, and anything inside a
# markdown link target. These are references, not measurements.
_SKIP_CONTEXT = re.compile(r"(§|\bv|\bADR-|\bE0|\bcycle[- ]|\bp\.)\s*$", re.IGNORECASE)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])")


def _ledger_numbers(runs_dir: Path) -> set[float]:
    """Every number that appears anywhere in any ledger, at any nesting depth."""
    out: set[float] = set()

    def walk(v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.add(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for p in _ledger_paths(runs_dir):
        try:
            walk(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def audit_prose(runs_dir: Path, text: str) -> list[str]:
    """Numeric literals in `text` that no ledger backs, in order of appearance.

    A literal is backed when some ledger number rounds to it at the precision the
    prose printed -- so `1.5945` backs a prose `1.59`, and `1.5476` backs nothing,
    which is the case that motivated this function. The board's own tallies count
    as backing, because they are `len()` over ledgers rather than typed numbers.
    """
    board = build(runs_dir)
    numbers = _ledger_numbers(runs_dir) | {float(v) for v in board.tally.values()}
    masked = _DATE.sub(" ", text)

    unbacked: list[str] = []
    for m in _NUMBER.finditer(masked):
        lit = m.group(1)
        if _SKIP_CONTEXT.search(masked[max(0, m.start() - 12):m.start()]):
            continue
        try:
            val = float(lit)
        except ValueError:                       # pragma: no cover - regex-guarded
            continue
        dec = len(lit.split(".")[1]) if "." in lit and "e" not in lit.lower() else None
        if dec is None:
            backed = any(n == val for n in numbers)
        else:
            backed = any(round(n, dec) == val for n in numbers)
        if not backed and lit not in unbacked:
            unbacked.append(lit)
    return unbacked


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def _cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "—"
    return str(v).replace("|", "\\|").replace("\n", " ")


def render(board: Board, *, runs_dir: Path) -> str:
    t = board.tally
    out = [
        "<!-- GENERATED by scripts/render_scoreboard.py from "
        f"{runs_dir}/*/ledger.json. Do not type a number into this block. -->",
        "",
        "## Scoreboard",
        "",
        f"Ledgers read: **{t['total']}**"
        + (f" (+{t['unreadable']} unreadable)" if t["unreadable"] else ""),
        "",
        "| run_id | cycle | outcome | status | exit nulls | unrepro cmds "
        "| rows | falsifier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(board.rows, key=lambda r: r["run_id"]):
        out.append(
            "| `{run_id}` | {cycle} | **{outcome}** | {status} | {nulls} | {unrepro} | "
            "{rows} | {fals} |".format(
                run_id=r["run_id"],
                cycle=_cell(r["cycle"]),
                outcome=_cell(r["outcome"]),
                status=_cell(r["status"]),
                nulls=r["n_null_exit"],
                unrepro=r["n_unreproducible"],
                rows=r["n_rows"],
                fals=_cell(r["falsifier"])[:80],
            )
        )

    out += [
        "",
        "### Tally",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in t.items()],
        "",
    ]

    canaries = [r for r in board.rows if r["run_id"].startswith("canary/")]
    if canaries:
        out += [
            "### Canary",
            "",
            *[
                f"- `{r['run_id']}` — **{_cell(r['outcome'])}** — {_cell(r['detail'])}"
                for r in sorted(canaries, key=lambda r: r["run_id"])
            ],
            "",
            f"{len([r for r in canaries if r['outcome'] == 'survived'])} survived, "
            f"{len([r for r in canaries if r['outcome'] == 'inconclusive'])} "
            f"inconclusive, "
            f"{len([r for r in canaries if r['outcome'] == 'falsified'])} moved.",
            "",
        ]

    if board.cycle_collisions:
        out += [
            "### ⚠️ `cycle` is not a join key",
            "",
            "These cycle numbers resolve to more than one ledger, so no row here is "
            "keyed on `cycle`:",
            "",
            *[
                f"- cycle **{c}** → {', '.join('`' + x + '`' for x in ids)}"
                for c, ids in sorted(board.cycle_collisions.items())
            ],
            "",
        ]

    bad = [r for r in board.rows if r["n_unreproducible"]]
    if bad:
        out += [
            "### 🔴 Not re-executable from any commit",
            "",
            f"{len(bad)} of {t['total']} ledgers record at least one command that "
            "cannot be resolved to an entry point committed at its own sha.",
            "",
            *[f"- `{r['run_id']}` — {r['n_unreproducible']}"
              f" of {r['n_commands']} commands — "
              f"`{_cell(r['first_unreproducible'])[:90]}`"
              for r in sorted(bad, key=lambda r: r["run_id"])],
            "",
        ]

    if board.unreadable:
        out += ["### Unreadable", "",
                *[f"- `{u['path']}` — {u['error']}" for u in board.unreadable], ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--over", type=Path, default=_REPO / "runs",
                    help="the runs/ directory to read")
    ap.add_argument("--artefact", type=Path, default=None,
                    help="write the rendered markdown here")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--claim", default=None, metavar="RUN_ID:OUTCOME",
                    help="check one prose claim against its ledger")
    ap.add_argument("--audit", type=Path, default=None, metavar="FILE",
                    help="list numbers in FILE that no ledger backs")
    args = ap.parse_args(argv)

    if not args.over.exists():
        print(f"DID NOT RUN: {args.over} does not exist", file=sys.stderr)
        return 3

    if args.claim:
        if ":" not in args.claim:
            print("--claim takes RUN_ID:OUTCOME", file=sys.stderr)
            return 3
        ident, outcome = args.claim.rsplit(":", 1)
        ok, why = check_claim(args.over, ident, outcome)
        print(("OK      " if ok else "REFUSED ") + why)
        return 0 if ok else 1

    if args.audit:
        if not args.audit.exists():
            print(f"DID NOT RUN: {args.audit} does not exist", file=sys.stderr)
            return 3
        unbacked = audit_prose(args.over, args.audit.read_text())
        if not unbacked:
            print(f"OK: every number in {args.audit} resolves to a ledger key")
            return 0
        print(f"{len(unbacked)} numbers in {args.audit} resolve to no ledger key:")
        for lit in unbacked:
            print(f"  {lit}")
        print("\nDelete it or fetch the key. Those are the only two options.")
        return 1

    board = build(args.over)
    text = render(board, runs_dir=args.over)
    if args.artefact:
        args.artefact.parent.mkdir(parents=True, exist_ok=True)
        args.artefact.write_text(text + "\n")
        print(f"wrote {args.artefact}")
    else:
        print(text)
    if args.json:
        args.json.write_text(json.dumps(
            {"rows": board.rows, "tally": board.tally,
             "cycle_collisions": {str(k): v for k, v in board.cycle_collisions.items()},
             "unreadable": board.unreadable}, indent=2) + "\n")
    return 2 if not board.rows else 0


if __name__ == "__main__":
    sys.exit(main())
