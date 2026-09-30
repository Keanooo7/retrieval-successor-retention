"""`scripts/battery_union.py`: the 3-way union of `scripts/mutation_battery.py`.

The tool resolves the append/append conflict every branch that adds a mutation
produces, and refuses everything it cannot resolve safely. Until 09-30 it was an
untested scratch script (`union.py`), and on 09-29 it refused twice on a change
only one side had made -- main's I5 edit to the code *after* `MUTATIONS` -- so a
second scratch copy (`union_tail.py`) resolved merges 13ea9ae and 74dec77. These
tests pin the rules, including that one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import battery_union as bu  # noqa: E402

_HEAD = '''"""doc"""
import os

_A_COUPLING = (
    "a",
)

MUTATIONS: tuple[Mutation, ...] = (
'''
_TAIL = """)


def run_suite():
    return 0
"""


def _entry(name: str, new: str = "y") -> str:
    return (
        f'    Mutation(\n        "{name}",\n        "test_x",\n        "{new}",\n    ),\n'
    )


def _src(entries=("m1", "m2"), head=_HEAD, tail=_TAIL, updates=None) -> str:
    updates = updates or {}
    body = "".join(_entry(n, updates.get(n, "y")) for n in entries)
    return head + body + tail


# -- the rule added 09-30: code after MUTATIONS ---------------------------------

_TAIL_THEIRS = _TAIL.replace("return 0", "return parse_failures()")
_TAIL_OURS = _TAIL.replace("return 0", "return 1")


def test_a_tail_change_only_theirs_made_is_taken_from_theirs():
    """ours' tail == base, theirs' differs: the 09-29 case. Take theirs'."""
    base = _src()
    ours = _src(("m1", "m2", "m3"))
    theirs = _src(("m1", "m2", "m4"), tail=_TAIL_THEIRS)
    res = bu.union(base, ours, theirs)
    assert res.text == _src(("m1", "m2", "m3", "m4"), tail=_TAIL_THEIRS)
    assert res.tail_from == "theirs"
    assert res.added == ['"m4",']


def test_a_tail_change_both_sides_made_differently_is_refused():
    base = _src()
    ours = _src(tail=_TAIL_OURS)
    theirs = _src(tail=_TAIL_THEIRS)
    with pytest.raises(bu.Refused) as e:
        bu.union(base, ours, theirs)
    assert "theirs changed code after MUTATIONS" in e.value.reasons[0]


def test_a_tail_change_only_ours_made_is_kept():
    base = _src()
    ours = _src(tail=_TAIL_OURS)
    res = bu.union(base, ours, _src())
    assert res.text == _src(tail=_TAIL_OURS)
    assert res.tail_from == "ours"


def test_the_same_tail_change_on_both_sides_is_kept_once():
    res = bu.union(_src(), _src(tail=_TAIL_THEIRS), _src(tail=_TAIL_THEIRS))
    assert res.text == _src(tail=_TAIL_THEIRS)


# -- the rules carried over from union.py ---------------------------------------


def test_entries_only_in_theirs_are_appended_in_theirs_order():
    base = _src(("m1",))
    res = bu.union(base, _src(("m1", "o1")), _src(("m1", "t2", "t1")))
    assert res.text == _src(("m1", "o1", "t2", "t1"))
    assert res.added == ['"t2",', '"t1",']


def test_an_entry_only_theirs_changed_is_updated():
    base = _src()
    res = bu.union(base, base, _src(updates={"m2": "z"}))
    assert res.text == _src(updates={"m2": "z"})
    assert res.updated == ['"m2",']


def test_an_entry_both_sides_changed_differently_is_refused():
    with pytest.raises(bu.Refused, match="changed on both sides"):
        bu.union(_src(), _src(updates={"m1": "o"}), _src(updates={"m1": "t"}))


def test_an_entry_theirs_deleted_is_refused():
    with pytest.raises(bu.Refused, match="deleted by theirs"):
        bu.union(_src(), _src(), _src(("m1",)))


def test_an_entry_ours_deleted_and_theirs_kept_is_refused():
    with pytest.raises(bu.Refused, match="ours deleted it"):
        bu.union(_src(), _src(("m1",)), _src())


def test_a_constant_new_in_theirs_is_added_before_mutations():
    head_t = _HEAD.replace("MUTATIONS:", '_B_COUPLING = (\n    "b",\n)\n\nMUTATIONS:')
    res = bu.union(_src(), _src(), _src(head=head_t))
    assert res.new_consts == ["_B_COUPLING"]
    assert '_B_COUPLING = (\n    "b",\n)\n' in res.text
    assert res.text.index("_B_COUPLING") < res.text.index("MUTATIONS:")


def test_a_constant_both_sides_changed_is_refused():
    ho = _HEAD.replace('"a",', '"o",')
    ht = _HEAD.replace('"a",', '"t",')
    with pytest.raises(bu.Refused, match="constant _A_COUPLING"):
        bu.union(_src(), _src(head=ho), _src(head=ht))


def test_head_code_both_sides_changed_is_refused():
    ho = _HEAD.replace("import os", "import re")
    ht = _HEAD.replace("import os", "import json")
    with pytest.raises(bu.Refused, match="before MUTATIONS"):
        bu.union(_src(), _src(head=ho), _src(head=ht))


def test_head_code_only_theirs_changed_is_taken():
    ht = _HEAD.replace("import os", "import os\nimport json")
    res = bu.union(_src(), _src(("m1", "m2", "o")), _src(head=ht))
    assert res.text == _src(("m1", "m2", "o"), head=ht)


def test_a_refusal_lists_every_reason_not_just_the_first():
    with pytest.raises(bu.Refused) as e:
        bu.union(
            _src(),
            _src(("m1",), tail=_TAIL_OURS),
            _src(tail=_TAIL_THEIRS, updates={"m2": "z"}),
        )
    assert len(e.value.reasons) >= 2


# -- the CLI: refusal writes nothing and is exit 3 ------------------------------


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        [bu.GIT, "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def _commit(repo: Path, text: str, msg: str) -> str:
    (repo / "mb.py").write_text(text)
    _git(repo, "add", "mb.py")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "battery_union.py"), *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    return tmp_path


def test_cli_refusal_exits_3_and_writes_nothing(repo):
    b = _commit(repo, _src(), "b")
    o = _commit(repo, _src(tail=_TAIL_OURS), "o")
    t = _commit(repo, _src(tail=_TAIL_THEIRS), "t")
    out = repo / "out.py"
    r = _cli(repo, "--path", "mb.py", "--revs", b, o, t, "--out", str(out))
    assert r.returncode == 3, r.stderr
    assert "REFUSED" in r.stderr
    assert not out.exists()


def test_cli_one_sided_tail_exits_0_and_writes_the_union(repo):
    b = _commit(repo, _src(), "b")
    o = _commit(repo, _src(("m1", "m2", "m3")), "o")
    t = _commit(repo, _src(("m1", "m2", "m4"), tail=_TAIL_THEIRS), "t")
    out = repo / "out.py"
    r = _cli(repo, "--path", "mb.py", "--revs", b, o, t, "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert out.read_text() == _src(("m1", "m2", "m3", "m4"), tail=_TAIL_THEIRS)


# -- the two real merges of 2026-09-29 ------------------------------------------

_MERGES_0929 = ("13ea9ae", "74dec77")


def _have(rev: str) -> bool:
    r = subprocess.run(
        [bu.GIT, "-C", str(_REPO), "cat-file", "-e", f"{rev}^{{commit}}"],
        capture_output=True,
    )
    return r.returncode == 0


@pytest.mark.parametrize("merge", _MERGES_0929)
def test_reproduces_the_mutation_battery_the_0929_merges_committed(merge):
    """Both merges were resolved by the scratch `union_tail.py`. The tested tool
    must produce byte-for-byte the `scripts/mutation_battery.py` they committed.
    Skips (and says so) only where the merge commits are not in this clone."""
    if not _have(merge):
        pytest.skip(f"{merge} is not in this clone (fetch run/e0d, run/newcomer-bakeoff)")
    ours, theirs = _git(_REPO, "log", "-1", "--format=%P", merge).split()
    base = _git(_REPO, "merge-base", ours, theirs)

    def show(rev: str) -> str:
        return subprocess.run(
            [bu.GIT, "-C", str(_REPO), "show", f"{rev}:{bu.PATH}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    res = bu.union(show(base), show(ours), show(theirs))
    assert res.text == show(merge)
    assert res.tail_from == "theirs"


# -- annotated constants (the i1 merge, 2026-09-30) ------------------------------
#
# `_REVIEW_GATE_REFUSALS: tuple[str, ...] = (` did not match the constant regex, so
# it travelled as head code; when ours also changed head code (i1's imports) the
# union refused a merge it should have resolved, and i1 was merged by hand
# (7eda821). ~/Documents/RSR-2026-09-29-day/reports/P1-I1.md.

_ANNOTATED = '_R: tuple[str, ...] = (\n    "r1",\n    "r2",\n)\n'
_HEAD_ANN = _HEAD.replace("_A_COUPLING = (", _ANNOTATED + "_A_COUPLING = (")


def test_an_annotated_constant_is_split_out_as_a_constant():
    p = bu.split(_src(head=_HEAD_ANN))
    assert p.const_order == ["_R", "_A_COUPLING"]
    assert p.consts["_R"] == _ANNOTATED
    assert "_R" not in p.head_code


def test_an_annotated_constant_new_in_theirs_merges_when_ours_changed_head_code():
    """The i1 case: ours changed imports, theirs added an annotated constant."""
    ho = _HEAD.replace("import os", "import os\nimport signal")
    res = bu.union(_src(), _src(("m1", "m2", "o"), head=ho), _src(head=_HEAD_ANN))
    assert res.new_consts == ["_R"]
    assert res.text == _src(
        ("m1", "m2", "o"), head=_HEAD_ANN.replace("import os", "import os\nimport signal")
    )


def test_an_annotated_constant_only_theirs_changed_is_updated():
    ho = _HEAD_ANN.replace("import os", "import os\nimport signal")
    ht = _HEAD_ANN.replace('"r2",', '"r2",\n    "r3",')
    res = bu.union(_src(head=_HEAD_ANN), _src(head=ho), _src(head=ht))
    assert res.text == _src(head=ht.replace("import os", "import os\nimport signal"))


def test_a_constant_new_in_theirs_lands_where_theirs_put_it():
    """Before the constant that follows it in theirs, when ours has that one."""
    ho = _HEAD.replace("import os", "import os\nimport signal")
    res = bu.union(_src(), _src(head=ho), _src(head=_HEAD_ANN))
    assert res.text.index("_R:") < res.text.index("_A_COUPLING")


# -- constant deletions (review MINOR-1) ------------------------------------------

_HEAD_NO_A = _HEAD.replace('_A_COUPLING = (\n    "a",\n)\n\n', "")


def test_a_constant_ours_deleted_and_theirs_kept_is_refused_not_resurrected():
    with pytest.raises(bu.Refused, match="constant _A_COUPLING: ours deleted it"):
        bu.union(_src(), _src(head=_HEAD_NO_A), _src())


def test_a_constant_theirs_deleted_and_ours_kept_is_refused_not_dropped():
    with pytest.raises(bu.Refused, match="constant _A_COUPLING deleted by theirs"):
        bu.union(_src(), _src(), _src(head=_HEAD_NO_A))


def test_a_constant_both_sides_deleted_stays_deleted():
    res = bu.union(_src(), _src(head=_HEAD_NO_A), _src(head=_HEAD_NO_A))
    assert "_A_COUPLING" not in res.text


# -- malformed input and output (review MINOR-2) ----------------------------------

_UNTERMINATED = '    Mutation(\n        "bad",\n        "test_x",\n        "y",\n'


def test_a_last_entry_without_its_terminator_is_refused_not_a_crash():
    bad = _HEAD + _entry("m1") + _UNTERMINATED + _TAIL
    with pytest.raises(bu.Refused, match="no terminating"):
        bu.union(_src(), _src(), bad)


def test_a_mid_entry_without_its_terminator_is_refused_for_that_reason():
    bad = _HEAD + _UNTERMINATED + _entry("m1") + _entry("m2") + _TAIL
    with pytest.raises(bu.Refused, match="no terminating"):
        bu.union(_src(), bad, _src())


def test_a_union_that_does_not_parse_is_refused():
    broken = '    Mutation(\n        "m9",\n        "test_x",\n        (,\n    ),\n'
    theirs = _HEAD + _entry("m1") + _entry("m2") + broken + _TAIL
    with pytest.raises(bu.Refused, match="does not parse"):
        bu.union(_src(), _src(), theirs)


def test_cli_malformed_entry_exits_3_and_writes_nothing(repo):
    b = _commit(repo, _src(), "b")
    t = _commit(repo, _HEAD + _entry("m1") + _UNTERMINATED + _TAIL, "t")
    o = b  # ours == base
    out = repo / "out.py"
    r = _cli(repo, "--path", "mb.py", "--revs", b, o, t, "--out", str(out))
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "REFUSED" in r.stderr
    assert not out.exists()


# -- the real i1 merge of 2026-09-30 ----------------------------------------------


def test_reproduces_the_mutation_battery_the_i1_merge_committed():
    """7eda821 merged night/2026-09-30 (1b8495e) into eng/i1-isolation (83bd910),
    base f7a6b10, by hand, because this tool refused. The fixed tool must produce
    its `scripts/mutation_battery.py` byte for byte."""
    merge = "7eda821"
    if not _have(merge):
        pytest.skip(f"{merge} is not in this clone (fetch eng/i1-isolation)")
    ours, theirs = _git(_REPO, "log", "-1", "--format=%P", merge).split()
    base = _git(_REPO, "merge-base", ours, theirs)

    def show(rev: str) -> str:
        return subprocess.run(
            [bu.GIT, "-C", str(_REPO), "show", f"{rev}:{bu.PATH}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    res = bu.union(show(base), show(ours), show(theirs))
    assert res.new_consts == ["_REVIEW_GATE_COUPLING", "_REVIEW_GATE_REFUSALS"]
    assert res.text == show(merge)
