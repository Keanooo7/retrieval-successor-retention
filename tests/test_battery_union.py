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
