"""`scripts/mutation_battery.py --check-anchors`: every anchor exactly once, in seconds.

On 2026-09-22 the full battery ran 87 minutes at 5ecda88 and then refused 3 on one
stale anchor (`'an unsigned ruling satisfies the queue'`) that a string search finds
at once. The anchor check now runs before the first suite run, and on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import mutation_battery as mb  # noqa: E402


def _table(tmp_path, monkeypatch, text: str, old: str) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(text)
    monkeypatch.setattr(mb, "ROOT", tmp_path)
    monkeypatch.setattr(
        mb, "MUTATIONS", (mb.Mutation("m", "test_x", "src/a.py", old, "y", "why"),)
    )


def _no_suite(*a, **k):
    raise AssertionError("the suite ran: the anchor check must come first")


# ⚠️ No test asserts that the REAL table's anchors all resolve. Under the battery,
# every mutation removes its own anchor from its file, so such a test goes red on
# every one of them and every entry LEAKS (measured 2026-09-22). The real table is
# checked where mutation cannot reach it: at battery start, and by
# `mutation_battery.py --check-anchors`.


@pytest.mark.parametrize(
    ("text", "count"),
    [("x = 1\n", 0), ("x = 2\nx = 2\n", 2)],
    ids=["stale", "duplicated"],
)
def test_check_anchors_refuses_a_stale_or_duplicated_anchor(
    tmp_path, monkeypatch, capsys, text, count
):
    _table(tmp_path, monkeypatch, text, "x = 2")
    monkeypatch.setattr(mb, "run_suite", _no_suite)
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--check-anchors"])
    assert mb.main() == mb.Exit.DID_NOT_RUN
    assert f"occurs {count} times" in capsys.readouterr().err


def test_check_anchors_battery_start_refuses_before_any_mutation(tmp_path, monkeypatch):
    _table(tmp_path, monkeypatch, "x = 1\n", "x = 2")
    monkeypatch.setattr(mb, "run_suite", _no_suite)
    monkeypatch.setattr(mb, "apply", _no_suite)
    monkeypatch.setattr(sys, "argv", ["mutation_battery.py", "--check"])
    with pytest.raises(SystemExit) as e:
        mb.main()
    assert e.value.code == 3
