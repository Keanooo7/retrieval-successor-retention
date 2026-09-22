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
    uv run python $S --over runs/ --artefact docs/lab-notes/scoreboard-2026-09-19.md
    uv run python $S --over runs/ --claim canary/cycle-04:survived
    uv run python $S --over runs/ --audit docs/lab-notes/overnight-2026-09-18.md
    uv run python $S --over runs/ --render-tokens draft.md   # {{run_id:key}} -> value

Exit codes follow the run protocol: `0` pass · `1` real failure · `2` nothing to
compare · `3` did not run. 🔴 `3` must never collapse to `0`.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from ledger import _exists_at_sha, entry_point  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main  # noqa: E402

OUTCOMES = ("survived", "falsified", "inconclusive")


@dataclass
class Board:
    rows: list[dict[str, Any]] = field(default_factory=list)
    tally: dict[str, int] = field(default_factory=dict)
    cycle_collisions: dict[int, list[str]] = field(default_factory=dict)
    unreadable: list[dict[str, str]] = field(default_factory=list)
    derived: dict[str, int] = field(default_factory=dict)
    """Named counts the rendered artefact prints outside the tally table -- each a
    `len()`, each with a name, so the audit can require the name beside the
    number (the small-integer rule in `audit_prose`)."""


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
            board.unreadable.append(
                {
                    "path": str(p),
                    "error": f"run_id {doc['run_id']!r} disagrees with its directory "
                    f"{run_id!r}; joining on the directory",
                }
            )
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

        board.rows.append(
            {
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
                    (
                        c.get("argv")
                        for c in cmds
                        if not _reproducible(c, prov.get("git_sha"))
                    ),
                    None,
                ),
                "n_null_exit": len([c for c in cmds if c.get("exit_code") is None]),
                "n_unreproducible": len(
                    [c for c in cmds if not _reproducible(c, prov.get("git_sha"))]
                ),
            }
        )

    board.cycle_collisions = {c: sorted(v) for c, v in by_cycle.items() if len(v) > 1}
    board.tally = {
        "total": len(board.rows),
        **{o: len([r for r in board.rows if r["outcome"] == o]) for o in OUTCOMES},
        "no_verdict": len([r for r in board.rows if r["outcome"] is None]),
        "unreadable": len(board.unreadable),
        "with_null_exit_code": len([r for r in board.rows if r["n_null_exit"]]),
        "not_re_executable": len([r for r in board.rows if r["n_unreproducible"]]),
    }
    canaries = [r for r in board.rows if r["run_id"].startswith("canary/")]
    board.derived = {
        f"canary_{o}": len([r for r in canaries if r["outcome"] == o]) for o in OUTCOMES
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
# §9 requires every wall-clock time to be read from `date` and written down. A
# timestamp is a fact about when, not a measurement, and flagging its digits would
# make the audit unusable on exactly the documents that obey that rule.
_CLOCK = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?(?:\s+[A-Z]{2,5})?(?:\s+\d{4})?"
)
_MONTHDAY = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b"
)
# `-` is in the lookbehind so a digit inside an identifier -- `claude-501`,
# `cycle-04`, `ADR-0006` -- is not read as a measurement. A genuinely negative
# number still matches, because the match then starts at the sign.
#
# ⚠️ Code spans are deliberately NOT masked. Both of the fabrications this exists
# to catch are written inside backticks in the 2026-09-18 documents
# (`1.5476 → 1.5336`, `10.817072550456`), so masking them would blind the audit to
# its own motivating case.
_NUMBER = re.compile(r"(?<![\w.-])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])")


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


#: An integer literal at or below this magnitude is not backed by its value alone.
#: Small integers -- seed counts, beat counts, bucket edges, `M`, `S` -- are in
#: nearly every ledger, so "some ledger somewhere holds a 3" backed every 3 in
#: every document: the audit passed small integers by coincidence, not evidence.
SMALL_INT = 1000

#: `{{run_id:key}}` -- a number written as its ledger key. `run_id` is the
#: directory under runs/ (it may contain `/`); `key` is a row key or a dotted
#: path into the ledger (`steps_done`, `armB.ratio`, `config.iters`,
#: `armB.ratio.sd`, `verdict.outcome`).
_TOKEN = re.compile(r"\{\{\s*([^{}\s:]+)\s*:\s*([^{}\s]+)\s*\}\}")
_WORD = re.compile(r"[A-Za-z_][\w./-]*")
_CODE_SPAN = re.compile(r"`([^`]+)`")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*(\[\"'{])")
_BLOCK_START = re.compile(r"^(#|[-*>] |\d+\. )")
#: Delimiters that end a verbatim quotation in rendered markdown.
_SEGMENT_SPLIT = re.compile(r"\s+\u2014\s+|(?<!\\)\||\*\*|`")
BOARD_ID = "scoreboard"
_MISSING = object()


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _flatten(doc: dict[str, Any]) -> dict[str, Any]:
    """Every addressable key of one ledger -> its value.

    Top-level fields and dotted paths into dicts (`verdict.outcome`,
    `provenance.git_sha`); each row by its `key` -- an observation's `value`
    (walked like a field, so `config.iters` resolves), a statistic's `mean`, with
    `.mean` / `.sd` / `.n` / `.samples` addressable explicitly.
    """
    out: dict[str, Any] = {}

    def walk(prefix: str, v: Any) -> None:
        out[prefix] = v
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}", x)

    for k, v in doc.items():
        if k not in ("rows", "commands"):
            walk(k, v)
    for row in doc.get("rows", []):
        key = row.get("key")
        if not isinstance(key, str):
            continue
        if "value" in row:
            walk(key, row["value"])
        else:
            out[key] = row.get("mean")
            for f in ("mean", "sd", "n", "samples"):
                if f in row:
                    out[f"{key}.{f}"] = row[f]
    return out


def _strings(v: Any) -> list[str]:
    """Every string value anywhere in a ledger -- what a verbatim quotation of it
    can be a substring of."""
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [s for x in v.values() for s in _strings(x)]
    if isinstance(v, (list, tuple)):
        return [s for x in v for s in _strings(x)]
    return []


def _matches(v: Any, n: float) -> bool:
    """Does the key's value equal `n`? A list-valued key (`seeds_actually_run`,
    `samples`) matches any of its scalar elements."""
    if _is_number(v):
        return float(v) == n
    if isinstance(v, (list, tuple)):
        return any(_is_number(x) and float(x) == n for x in v)
    return False


def _sentences(text: str) -> list[tuple[str, str]]:
    """`(scan, context)` pairs. A prose sentence is its own context; a markdown
    table row's context is the row plus its header row, so a column named for a
    key (`n_rows`) names it for every cell under it."""
    out: list[tuple[str, str]] = []
    para: list[str] = []
    header: str | None = None

    def flush() -> None:
        if para:
            for sent in _SENT_SPLIT.split(" ".join(para)):
                if sent.strip():
                    out.append((sent, sent))
            para.clear()

    lines = text.splitlines()
    for i, line in enumerate(lines):
        st = line.strip()
        if st.startswith("|"):
            flush()
            if _TABLE_SEP.match(st):
                continue
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _TABLE_SEP.match(nxt):
                header = st
                out.append((st, st))
            else:
                out.append((st, f"{st} {header or ''}"))
            continue
        header = None
        if not st or _BLOCK_START.match(st):
            flush()
        if st:
            para.append(st)
    flush()
    return out


def _quoted(scan: str, start: int, end: int, runs: set[str], strings: dict) -> bool:
    """Is the number inside a verbatim quotation of a string field of a ledger
    named in the same sentence? The renderer copies `verdict.detail`, `falsifier`
    and `argv` straight off a ledger, and a digit inside a copied string is that
    ledger's text, not a typed measurement. The segment is bounded by the
    markdown delimiters the renderer puts around a copied field."""
    pos, seg = 0, ""
    for m in [*_SEGMENT_SPLIT.finditer(scan), None]:
        seg_end = m.start() if m else len(scan)
        if pos <= start and end <= seg_end:
            seg = scan[pos:seg_end].strip().replace("\\|", "|")
            break
        pos = m.end() if m else len(scan)
    if len(seg) < 8:
        return False
    return any(seg in x for rid in runs for x in strings.get(rid, ()))


def _key_mentions(text: str) -> set[str]:
    """Words that can name a ledger key: anything in a code span, and any bare
    word that is identifier-shaped (`_`, `.`, `/` or `-` in it). A plain English
    word is not a key mention -- `seeds` is a key in four ledgers, and "3 seeds"
    must not be backed by one of them listing seed 3."""
    out: set[str] = set()
    for span in _CODE_SPAN.findall(text):
        out |= {w.rstrip(".,;:)!?") for w in _WORD.findall(span)}
    for w in _WORD.findall(_CODE_SPAN.sub(" ", text)):
        w = w.rstrip(".,;:)!?")
        if any(c in w for c in "_./-"):
            out.add(w)
    return out


def _index(runs_dir: Path, board: Board) -> tuple[dict, dict]:
    """`(flat, strings)`: run_id -> {key: value}, run_id -> [string values].

    The board is indexed too: each row's `len()`-derived fields under its own
    run_id, and the tally plus the named counts under `scoreboard`. Every one of
    them is `len()` of something or a field copied off a ledger -- exactly the
    numbers this script exists to stop anyone typing -- so they back the prose,
    but only beside their names, like any other key.
    """
    flat: dict[str, dict[str, Any]] = {}
    strings: dict[str, list[str]] = {}
    for p in _ledger_paths(runs_dir):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rid = _rel_run_id(p, runs_dir)
        flat[rid] = _flatten(doc)
        strings[rid] = _strings(doc)
    for row in board.rows:
        for k, v in row.items():
            if _is_number(v):
                flat.setdefault(row["run_id"], {}).setdefault(k, v)
    flat[BOARD_ID] = {**board.tally, **board.derived}
    return flat, strings


def resolve_token(runs_dir: Path, run_id: str, key: str) -> Any:
    """The value `{{run_id:key}}` names, or `KeyError`."""
    flat, _ = _index(runs_dir, build(runs_dir))
    v = flat.get(run_id, {}).get(key, _MISSING)
    if v is _MISSING:
        raise KeyError(f"{{{{{run_id}:{key}}}}} resolves to no ledger key")
    return v


def render_tokens(runs_dir: Path, text: str) -> tuple[str, list[str]]:
    """`(rendered, unresolved)`: every `{{run_id:key}}` replaced by its ledger
    value -- `repr` for a float, so nothing is rounded on the way out. An
    unresolved token is left in place and listed."""
    flat, _ = _index(runs_dir, build(runs_dir))
    unresolved: list[str] = []

    def sub(m: re.Match[str]) -> str:
        v = flat.get(m.group(1), {}).get(m.group(2), _MISSING)
        if v is _MISSING:
            unresolved.append(m.group(0))
            return m.group(0)
        return repr(v) if isinstance(v, float) else str(v)

    return _TOKEN.sub(sub, text), unresolved


def audit_prose(runs_dir: Path, text: str) -> list[str]:
    """Numeric literals in `text` that no ledger backs, in order of appearance.

    **Decimals and integers above `SMALL_INT`**: backed when some ledger number
    rounds to the literal at the precision the prose printed -- so `1.5945`
    backs a prose `1.59`, and `1.5476` backs nothing, which is the case that
    motivated this function.

    🔴 **An integer with `|n| <= SMALL_INT` is backed only by a named key.**
    Either (a) a ledger key -- or a `{{run_id:key}}` token -- appears in the same
    sentence and its value equals the integer (a list-valued key matches any
    element); if the sentence names a run_id, only that run's keys count, or
    (b) the number is written as `{{run_id:key}}` itself, which resolves by
    construction (`render_tokens`). A token that resolves to nothing is itself
    reported. Same-value matches across unrelated ledgers no longer count: they
    backed nearly every small integer in every document. A table row's sentence
    includes its header row; a digit inside a verbatim quotation of a string
    field of a ledger the sentence names (`verdict.detail`, `argv`) is that
    ledger's text and is backed as such.
    """
    board = build(runs_dir)
    flat, strings = _index(runs_dir, board)
    by_key: dict[str, list[str]] = {}
    for rid, keys in flat.items():
        for k in keys:
            by_key.setdefault(k, []).append(rid)
    numbers = _ledger_numbers(runs_dir) | {
        float(v) for keys in flat.values() for v in keys.values() if _is_number(v)
    }
    masked = _MONTHDAY.sub(" ", _CLOCK.sub(" ", _DATE.sub(" ", text)))

    unbacked: list[str] = []
    for scan, context in _sentences(masked):
        tokens = [m.groups() for m in _TOKEN.finditer(context)]
        values: list[Any] = []
        for m in _TOKEN.finditer(scan):
            dangling = flat.get(m.group(1), {}).get(m.group(2), _MISSING) is _MISSING
            if dangling and m.group(0) not in unbacked:
                unbacked.append(m.group(0))
        words = _key_mentions(_TOKEN.sub(" ", context))
        named = {w for w in words if w in flat and w != BOARD_ID}
        named |= {rid for rid, _ in tokens}
        for rid, key in tokens:
            v = flat.get(rid, {}).get(key, _MISSING)
            if v is not _MISSING:
                values.append(v)
        for w in words:
            for rid in by_key.get(w, ()):
                if named and rid not in named and rid != BOARD_ID:
                    continue
                values.append(flat[rid][w])
        body = _TOKEN.sub(lambda m: " " * len(m.group(0)), scan)
        for m in _NUMBER.finditer(body):
            lit = m.group(1)
            if _SKIP_CONTEXT.search(body[max(0, m.start() - 12) : m.start()]):
                continue
            try:
                val = float(lit)
            except ValueError:  # pragma: no cover - regex-guarded
                continue
            exp = "e" in lit.lower()
            dec = len(lit.split(".")[1]) if "." in lit and not exp else None
            integer = "." not in lit and not exp
            if integer and abs(val) <= SMALL_INT:
                backed = any(_matches(v, val) for v in values) or _quoted(
                    body, m.start(), m.end(), named, strings
                )
            elif dec is None:
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
        f"Ledgers read (`total`): **{t['total']}**"
        + (f" (+{t['unreadable']} `unreadable`)" if t["unreadable"] else ""),
        "",
        # Column names are the board's own keys: the audit reads a table row
        # with its header, so each small integer below stands beside its key.
        "| run_id | `cycle` | outcome | status | `n_null_exit` | `n_unreproducible` "
        "| `n_rows` | falsifier |",
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
        *[f"| `{k}` | {v} |" for k, v in t.items()],
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
            f"`canary_survived` {board.derived['canary_survived']}, "
            f"`canary_inconclusive` {board.derived['canary_inconclusive']}, "
            f"`canary_falsified` (moved) {board.derived['canary_falsified']}.",
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
                f"- `cycle` **{c}** → {', '.join('`' + x + '`' for x in ids)}"
                for c, ids in sorted(board.cycle_collisions.items())
            ],
            "",
        ]

    bad = [r for r in board.rows if r["n_unreproducible"]]
    if bad:
        out += [
            "### 🔴 Not re-executable from any commit",
            "",
            f"`not_re_executable` {t['not_re_executable']} of `total` {t['total']} "
            "ledgers record at least one command that cannot be resolved to an "
            "entry point committed at its own sha.",
            "",
            *[
                f"- `{r['run_id']}` — `n_unreproducible` {r['n_unreproducible']}"
                f" of `n_commands` {r['n_commands']} — "
                f"`{_cell(r['first_unreproducible'])[:90]}`"
                for r in sorted(bad, key=lambda r: r["run_id"])
            ],
            "",
        ]

    if board.unreadable:
        out += [
            "### Unreadable",
            "",
            *[f"- `{u['path']}` — {u['error']}" for u in board.unreadable],
            "",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> Exit:
    # A usage error exits 3 (did not run), not argparse's 2: in this script `2` is
    # reserved for "no rows" below, and giving it two meanings is the conflation
    # S0-05 exists to remove.
    ap = ArgumentParser(description=__doc__)
    ap.add_argument(
        "--over", type=Path, default=_REPO / "runs", help="the runs/ directory to read"
    )
    ap.add_argument(
        "--artefact", type=Path, default=None, help="write the rendered markdown here"
    )
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--claim",
        default=None,
        metavar="RUN_ID:OUTCOME",
        help="check one prose claim against its ledger",
    )
    ap.add_argument(
        "--audit",
        type=Path,
        default=None,
        metavar="FILE",
        help="list numbers in FILE that no ledger backs",
    )
    ap.add_argument(
        "--render-tokens",
        type=Path,
        default=None,
        metavar="FILE",
        help="print FILE with every {{run_id:key}} replaced by its ledger value",
    )
    args = ap.parse_args(argv)

    if not args.over.exists():
        return did_not_run(f"{args.over} does not exist")

    if args.claim:
        if ":" not in args.claim:
            return did_not_run("--claim takes RUN_ID:OUTCOME")
        ident, outcome = args.claim.rsplit(":", 1)
        ok, why = check_claim(args.over, ident, outcome)
        print(("OK      " if ok else "REFUSED ") + why)
        return Exit.OK if ok else Exit.FAIL

    if args.audit:
        if not args.audit.exists():
            return did_not_run(f"{args.audit} does not exist")
        unbacked = audit_prose(args.over, args.audit.read_text())
        if not unbacked:
            print(f"OK: every number in {args.audit} resolves to a ledger key")
            return Exit.OK
        print(f"{len(unbacked)} numbers in {args.audit} resolve to no ledger key:")
        for lit in unbacked:
            print(f"  {lit}")
        print("\nDelete it or fetch the key. Those are the only two options.")
        return Exit.FAIL

    if args.render_tokens:
        if not args.render_tokens.exists():
            return did_not_run(f"{args.render_tokens} does not exist")
        rendered, unresolved = render_tokens(args.over, args.render_tokens.read_text())
        print(rendered, end="")
        for tok in unresolved:
            print(f"UNRESOLVED: {tok}", file=sys.stderr)
        return Exit.FAIL if unresolved else Exit.OK

    board = build(args.over)
    text = render(board, runs_dir=args.over)
    if args.artefact:
        args.artefact.parent.mkdir(parents=True, exist_ok=True)
        args.artefact.write_text(text + "\n")
        print(f"wrote {args.artefact}")
    else:
        print(text)
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "rows": board.rows,
                    "tally": board.tally,
                    "cycle_collisions": {
                        str(k): v for k, v in board.cycle_collisions.items()
                    },
                    "unreadable": board.unreadable,
                },
                indent=2,
            )
            + "\n"
        )
    # The repo's one correct `2` before S0-05: ran, and there was nothing to render.
    return Exit.UNKNOWN if not board.rows else Exit.OK


if __name__ == "__main__":
    run_main(main)
