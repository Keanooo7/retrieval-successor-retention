"""`audit_prose`'s small-integer rule (2026-09-22).

**The hole.** A literal was "backed" when it equalled any number in any ledger. The
committed ledgers hold nearly every integer from 0 to a few hundred somewhere -- seed
lists, beat counts, bucket edges, `M`, `S`, row counts -- so a typed "3 seeds" or
"16 slots" passed whether or not any ledger said so *about that thing*.

**The rule.** An integer with `|n| <= 1000` passes only if (a) a ledger key -- or a
`{{run_id:key}}` token -- appears in the same sentence with that value (restricted to
the run named in the sentence, if one is), or (b) it is written as a
`{{run_id:key}}` token that resolves. Decimals and large integers keep the old
value rule, which is specific enough to mean something.

Every number used below is read off a committed ledger and named here:
`runs/canary/cycle-08/ledger.json` row `config`, field `iters` = 6;
`runs/decisive-shuffle/ledger.json` `steps_done` = 300, `steps_requested` = 300.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import render_scoreboard as rs  # noqa: E402

RUNS = _REPO / "runs"


def test_the_audit_refuses_a_small_integer_matched_only_by_an_unrelated_ledger():
    """The hole itself: `3` is in dozens of ledgers, about dozens of things."""
    assert rs.audit_prose(RUNS, "The run used 3 seeds and 16 slots.") == ["3", "16"]


def test_the_audit_refuses_a_small_integer_beside_its_key_with_the_wrong_value():
    text = "`canary/cycle-08` ran with `config.iters` 7."
    assert rs.audit_prose(RUNS, text) == ["7"]


def test_the_audit_refuses_a_small_integer_from_another_run():
    """`decisive-shuffle` has `steps_done` 300. A sentence about the canary may not
    borrow it: the canary ledger has no such value."""
    text = "`canary/cycle-08` recorded `steps_done` 300."
    assert rs.audit_prose(RUNS, text) == ["300"]


def test_a_small_integer_beside_its_key_is_backed():
    assert rs.audit_prose(RUNS, "`canary/cycle-08` ran with `config.iters` 6.") == []
    assert rs.audit_prose(RUNS, "`decisive-shuffle` ran `steps_done` 300.") == []


def test_a_small_integer_written_as_a_token_is_backed_and_renders():
    text = "It ran {{decisive-shuffle:steps_done}} steps."
    assert rs.audit_prose(RUNS, text) == []
    rendered, unresolved = rs.render_tokens(RUNS, text)
    assert rendered == "It ran 300 steps." and unresolved == []
    # a literal beside a token in the same sentence is checked against it
    ok = "It ran 300 of {{decisive-shuffle:steps_requested}} requested."
    assert rs.audit_prose(RUNS, ok) == []


def test_a_token_that_resolves_to_no_key_is_flagged():
    text = "It ran {{decisive-shuffle:no_such_key}} steps."
    assert rs.audit_prose(RUNS, text) == ["{{decisive-shuffle:no_such_key}}"]
    _, unresolved = rs.render_tokens(RUNS, text)
    assert unresolved == ["{{decisive-shuffle:no_such_key}}"]


def test_the_audit_refuses_a_small_integer_in_a_table_cell_its_header_key_denies():
    """A table row is read with its header: `6` under `config.iters` for
    `canary/cycle-08` is backed, `5` for `canary/cycle-12` is not."""
    table = (
        "| run | `config.iters` |\n|---|---|\n| `canary/cycle-08` | 6 |\n"
        "| `canary/cycle-12` | 5 |\n"
    )
    assert rs.audit_prose(RUNS, table) == ["5"]


def test_decimals_keep_the_value_rule():
    """Unchanged: a decimal is specific enough that a value match means something."""
    assert rs.audit_prose(RUNS, "rel_tol was 0.0001, NLL 1.5476.") == ["1.5476"]
