"""Semantic 3-way union for `scripts/mutation_battery.py` during a merge.

Every branch appends `Mutation(...)` entries to the one `MUTATIONS` tuple, so two
branches that each add a mutation conflict textually (append/append) while agreeing
semantically. This tool resolves exactly that case and **refuses everything else**.

Ported from `~/Documents/RSR-2026-09-26-day/archive/union.py` (untested, run by
hand on 09-26 and 09-29; the original is archived at
`~/Documents/RSR-2026-09-29-day/superseded/union-2026-09-26.py`). PLAN-v4 I4 plans
to retire it; until then it is tested (`tests/test_battery_union.py`).

The file is split into three parts:

* **head** -- everything before ``MUTATIONS: tuple[Mutation, ...] = (``. Inside it,
  module-level ``_NAME = (`` ... ``)`` constants -- annotated ones too,
  ``_NAME: tuple[str, ...] = (`` (added 2026-09-30: the i1 merge, 7eda821, was
  refused because an annotated constant travelled as head code) -- are keyed by
  name; the rest of the head ("head code") is compared as one string.
* **body** -- the tuple's entries. An entry starts at ``    Mutation(`` (4-space
  indent) and ends at the matching ``    ),``; comment lines before it belong to
  it. It is keyed by its first argument (the mutation's name). Text after the last
  entry is the body's "trailing text".
* **tail** -- from the tuple's closing ``)`` at column 0 to end of file.

Rules (``ours`` = the checked-out side, ``theirs`` = the side being merged in):

* entry only in theirs -> appended, in theirs' order;
* entry in all three: ours == base -> theirs; theirs == base -> ours; both changed
  and different -> REFUSE;
* entry in base and ours but missing from theirs -> REFUSE (theirs deleted it);
  entry in base and theirs but missing from ours -> REFUSE (ours deleted it);
* constants: the same one-sided rule, deletions included (a constant either side
  deleted and the other kept is REFUSED; review 7d90e27a400b MINOR-1); a constant
  new in theirs is inserted before the constant that follows it in theirs, if ours
  has that one, else appended to the head;
* head code: taken from theirs only if ours == base; both changed -> REFUSE;
* tail: taken from theirs only if ours == base; both changed -> REFUSE
  (`_merge_tail`; the one-sided rule was added 2026-09-30);
* trailing text in the body: theirs may not change it (REFUSE) unless ours made
  the same change.

A malformed side (an entry with no terminating ``    ),``, no ``MUTATIONS`` tuple)
is a refusal, not a crash; and the merged text must parse (`ast.parse`) and hold
exactly ours' entries plus the ones added, or it is refused before anything is
written (review 7d90e27a400b MINOR-2).

A refusal writes nothing and exits `3` (`Exit.DID_NOT_RUN`: the union did not run;
resolve by hand). Success writes the merged file and exits `0`.

Usage (inside a conflicted merge, from the repo root)::

    python scripts/battery_union.py                  # index stages :1: :2: :3:
    python scripts/battery_union.py --revs B O T --out F   # any three revisions
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.exit_codes import ArgumentParser, Exit, run_main  # noqa: E402

PATH = "scripts/mutation_battery.py"
GIT = "/opt/homebrew/bin/git"
OPEN = "MUTATIONS: tuple[Mutation, ...] = (\n"
# `_NAME = (` or `_NAME: <annotation> = (`, through the `)` at column 0.
_CONST = re.compile(r"^(_[A-Z0-9_]+)(?:: [^\n=]+)? = \(\n.*?^\)\n", re.S | re.M)
_CONST_STRIP = re.compile(r"^(_[A-Z0-9_]+)(?:: [^\n=]+)? = \(\n.*?^\)\n\n?", re.S | re.M)


class Refused(Exception):
    """The union is not safe to compute; `reasons` says why. Nothing was written."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


@dataclass
class Parts:
    head: str
    head_code: str
    consts: dict[str, str]
    const_order: list[str]
    entries: dict[str, str]
    order: list[str]
    trailing: str
    tail: str
    const_gap: dict[str, str] = field(default_factory=dict)
    """Whitespace between a constant and the next one, when only whitespace."""


@dataclass
class Result:
    text: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    new_consts: list[str] = field(default_factory=list)
    tail_from: str = "ours"


def split(src: str) -> Parts:
    start = src.find(OPEN)
    if start < 0:
        raise Refused([f"no {OPEN.strip()!r} line"])
    body_start = start + len(OPEN)
    end = src.find("\n)\n", body_start)  # the tuple's closing paren at col 0
    if end < 0:
        raise Refused(["MUTATIONS has no closing ')' at column 0"])
    head, body, tail = src[:start], src[body_start : end + 1], src[end + 1 :]
    entries: dict[str, str] = {}
    order: list[str] = []
    lines = body.splitlines(keepends=True)
    pre: list[str] = []  # comment lines preceding an entry, attached to it
    i = 0
    while i < len(lines):
        if lines[i] == "    Mutation(\n":
            j = i + 1
            while j < len(lines) and lines[j] not in ("    ),\n", "    Mutation(\n"):
                j += 1
            if j == len(lines) or lines[j] != "    ),\n":
                where = "the end of MUTATIONS" if j == len(lines) else "the next entry"
                first = lines[i + 1].strip() if i + 1 < len(lines) else ""
                raise Refused(
                    [
                        f"the entry at MUTATIONS line {i + 1} ({first!r}) has no "
                        f"terminating '    ),' before {where}"
                    ]
                )
            block = "".join(pre + lines[i : j + 1])
            pre = []
            k = i + 1
            key = lines[k]
            while not key.rstrip().endswith(",") and k + 1 < j:
                k += 1
                key += lines[k]
            name = re.sub(r"\s+", " ", key).strip()
            if name in entries:
                raise Refused([f"duplicate entry {name}"])
            entries[name] = block
            order.append(name)
            i = j + 1
        else:
            pre.append(lines[i])
            i += 1
    consts: dict[str, str] = {}
    const_order: list[str] = []
    const_gap: dict[str, str] = {}
    found = list(_CONST.finditer(head))
    for n, m in enumerate(found):
        if m.group(1) in consts:
            raise Refused([f"duplicate constant {m.group(1)}"])
        consts[m.group(1)] = m.group(0)
        const_order.append(m.group(1))
        if n + 1 < len(found):
            gap = head[m.end() : found[n + 1].start()]
            if not gap.strip():
                const_gap[m.group(1)] = gap
    head_code = _CONST_STRIP.sub("", head)
    return Parts(
        head,
        head_code,
        consts,
        const_order,
        entries,
        order,
        "".join(pre),
        tail,
        const_gap,
    )


def _split(side: str, src: str) -> Parts:
    try:
        return split(src)
    except Refused as e:
        raise Refused([f"{side}: {r}" for r in e.reasons]) from None


def _merge_tail(b: Parts, o: Parts, t: Parts, refuse: list[str]) -> tuple[str, str]:
    """The code after MUTATIONS. Returns (tail, which side it came from).

    * theirs == base, or theirs == ours: ours' tail;
    * ours == base and theirs differs: **theirs' tail** -- a one-sided change. Added
      2026-09-30: union.py refused this twice on 09-29 (main's I5 edit to
      run_suite/parse_failures, merged into run/e0d and run/newcomer-bakeoff);
    * both changed, differently: REFUSE. There is no line merge here.
    """
    if t.tail == b.tail or t.tail == o.tail:
        return o.tail, "ours"
    if o.tail == b.tail:
        return t.tail, "theirs"
    refuse.append("theirs changed code after MUTATIONS, and so did ours")
    return o.tail, "ours"


def union(base: str, ours: str, theirs: str) -> Result:
    """The merged file, or `Refused` naming every reason it is not safe."""
    b, o, t = _split("base", base), _split("ours", ours), _split("theirs", theirs)
    refuse: list[str] = []
    if (
        t.head_code != b.head_code
        and t.head_code != o.head_code
        and o.head_code != b.head_code
    ):
        refuse.append("both sides changed non-constant code before MUTATIONS")
    take_theirs_head = o.head_code == b.head_code and t.head_code != b.head_code
    tail, tail_from = _merge_tail(b, o, t, refuse)
    if (
        t.trailing.strip() != b.trailing.strip()
        and t.trailing.strip() != o.trailing.strip()
    ):
        refuse.append(f"theirs changed trailing text in MUTATIONS: {t.trailing[:120]!r}")

    consts = dict(o.consts)
    new_consts: list[str] = []
    for k in t.const_order:
        tv, bv, ov = t.consts[k], b.consts.get(k), o.consts.get(k)
        if ov is None:
            if bv is not None:  # MINOR-1: never resurrect what ours deleted
                refuse.append(f"constant {k}: ours deleted it, theirs kept it")
            else:
                new_consts.append(k)
                consts[k] = tv
        elif tv != ov:
            if ov == bv:
                consts[k] = tv
            elif tv != bv:
                refuse.append(f"constant {k} changed on both sides")
    for k in b.const_order:  # MINOR-1: never silently drop theirs' deletion
        if k in o.consts and k not in t.consts:
            refuse.append(f"constant {k} deleted by theirs")

    entries = dict(o.entries)
    order = list(o.order)
    added: list[str] = []
    updated: list[str] = []
    for k in t.order:
        tv, bv, ov = t.entries[k], b.entries.get(k), o.entries.get(k)
        if ov is None:
            if bv is not None:
                refuse.append(f"entry {k!r}: ours deleted it, theirs kept")
            else:
                entries[k] = tv
                order.append(k)
                added.append(k)
        elif tv != ov:
            if ov == bv:
                entries[k] = tv
                updated.append(k)
            elif tv != bv:
                refuse.append(f"entry {k!r} changed on both sides")
    for k in b.order:
        if k in o.entries and k not in t.entries:
            refuse.append(f"entry {k!r} deleted by theirs")
    if not o.tail.startswith(")\n"):
        refuse.append(f"ours' tail does not start at the tuple's ')': {o.tail[:20]!r}")
    if refuse:
        raise Refused(refuse)

    # template: theirs' head if only theirs changed the head code, else ours'
    if take_theirs_head:
        h = t.head
        for k, v in consts.items():  # merged constant values
            if k in t.consts:
                if t.consts[k] != v:
                    h = h.replace(t.consts[k], v)
            else:
                h = h + v + "\n"  # ours-only constant
    else:
        h = o.head
        for k, v in consts.items():
            if k in o.consts and o.consts[k] != v:
                h = h.replace(o.consts[k], v)
        for k in new_consts:
            h = _insert_new_const(h, k, t, o)
    body = "".join(entries[k] for k in order) + o.trailing
    text = h + OPEN + body + tail
    _check_output(text, len(o.entries) + len(added), len(consts))
    return Result(text, added, updated, new_consts, tail_from)


def _insert_new_const(h: str, k: str, t: Parts, o: Parts) -> str:
    """Put theirs-only constant `k` where theirs put it: before the first constant
    that follows it in theirs and that ours also has, with theirs' whitespace gap.
    With no such constant, append it to the head (the pre-09-30 behaviour)."""
    later = t.const_order[t.const_order.index(k) + 1 :]
    succ = next((s for s in later if s in o.consts), None)
    if succ is None:
        return h + t.consts[k] + "\n"
    m = re.search(rf"^{re.escape(succ)}\b", h, re.M)
    if m is None:  # cannot happen for a constant split() found; refuse, don't guess
        raise Refused([f"constant {succ} not found in the merged head"])
    text = t.consts[k] + t.const_gap.get(k, "\n")
    return h[: m.start()] + text + h[m.start() :]


def _check_output(text: str, n_entries: int, n_consts: int) -> None:
    """MINOR-2: the union must parse and hold exactly the entries and constants
    the merge computed, or nothing is written."""
    try:
        ast.parse(text)
    except SyntaxError as e:
        raise Refused([f"the union does not parse: {e.msg} (line {e.lineno})"]) from None
    got = _split("the union", text)
    if len(got.entries) != n_entries or len(got.consts) != n_consts:
        raise Refused(
            [
                f"the union holds {len(got.entries)} entries and {len(got.consts)} "
                f"constants; the merge computed {n_entries} and {n_consts}"
            ]
        )


def _show(spec: str) -> str:
    r = subprocess.run([GIT, "show", spec], capture_output=True, text=True)
    if r.returncode != 0:
        raise Refused([f"git show {spec} failed: {r.stderr.strip()}"])
    return r.stdout


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--path", default=PATH, help="repo-relative file (default %(default)s)"
    )
    ap.add_argument(
        "--revs",
        nargs=3,
        metavar=("BASE", "OURS", "THEIRS"),
        help="read the three sides from these revisions instead of the index stages",
    )
    ap.add_argument("--out", help="write here instead of the working file at --path")
    a = ap.parse_args()
    specs = (
        [f"{r}:{a.path}" for r in a.revs]
        if a.revs
        else [f":{n}:{a.path}" for n in (1, 2, 3)]
    )
    try:
        res = union(*(_show(s) for s in specs))
    except Refused as e:
        print("REFUSED:", file=sys.stderr)
        for r in e.reasons:
            print("  -", r, file=sys.stderr)
        return Exit.DID_NOT_RUN
    Path(a.out or a.path).write_text(res.text)
    print(
        f"added {len(res.added)} entries, updated {len(res.updated)} entries, "
        f"new constants {res.new_consts}, tail from {res.tail_from}"
    )
    print("updated:", res.updated)
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
