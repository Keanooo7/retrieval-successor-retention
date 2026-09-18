"""The committed fidelity tolerances (ADR-0002 Part B, D-H, gauntlet 2.2).

These tests exist so the numbers cannot drift after the fixtures are generated.
They are cheap and they are the entire mechanism: a tolerance is only a tolerance
if changing it is visible.
"""

from __future__ import annotations

import pytest

from rsr.model.tg.tolerances import FORWARD, GRADIENT, for_quantity


def test_the_committed_forward_tolerance():
    assert (FORWARD.rtol, FORWARD.atol) == (1e-4, 1e-5)


def test_the_committed_gradient_tolerance():
    assert (GRADIENT.rtol, GRADIENT.atol) == (1e-3, 1e-4)


def test_gradients_are_exactly_one_order_looser():
    """ADR-0002 says one order and gives the reason. Two orders would start
    absorbing the retained-graph divergence the fixtures exist to catch."""
    assert GRADIENT.rtol == pytest.approx(FORWARD.rtol * 10)
    assert GRADIENT.atol == pytest.approx(FORWARD.atol * 10)


@pytest.mark.parametrize(
    "quantity,expected",
    [
        ("activations", FORWARD),
        ("gestalts", FORWARD),
        ("cross_attention", FORWARD),
        ("logits", FORWARD),
        ("grad_w_sent", GRADIENT),
        ("grad_transformer", GRADIENT),
    ],
)
def test_every_fixture_quantity_has_a_committed_tolerance(quantity, expected):
    assert for_quantity(quantity) is expected


def test_an_unknown_quantity_raises_rather_than_defaulting():
    """A new quantity needs a decision in ADR-0002, not whichever branch it lands
    in."""
    with pytest.raises(KeyError, match="ADR-0002"):
        for_quantity("some_new_tensor")


def test_the_reasons_are_recorded_beside_the_numbers():
    """ "Say how much looser and why, in advance." The why travels with the value so
    it cannot be lost when the table is copied."""
    assert "reduce in different orders" in GRADIENT.why
    assert "float32" in FORWARD.why


def test_the_tolerance_commit_precedes_the_fixture_commit():
    """Gauntlet 2.2's ordering, now asserted against git rather than against the
    absence of the fixtures.

    `test_no_fixture_exists_yet` lived here and was deleted in the commit that
    added them, exactly as its docstring said it would be. The ordering claim it
    stood for is now checked directly: the commit that introduced
    `src/rsr/model/tg/tolerances.py` must be an ancestor of the one that
    introduced `tests/fixtures/`.
    """
    import shutil
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    git = next(
        (g for g in ("/opt/homebrew/bin/git", "/usr/local/bin/git") if Path(g).exists()),
        shutil.which("git"),
    )
    if git is None:
        pytest.skip("git not available; the ordering claim lives in ADR-0002")

    def first_commit(path: str) -> str | None:
        out = subprocess.run(
            [git, "-C", str(root), "log", "--reverse", "--format=%H", "--", path],
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in out.stdout.splitlines() if ln]
        return lines[0] if lines else None

    tol = first_commit("src/rsr/model/tg/tolerances.py")
    fix = first_commit("tests/fixtures")
    if tol is None or fix is None:
        pytest.skip("shallow clone or unborn history; ordering lives in ADR-0002")
    if tol == fix:
        pytest.fail(
            "tolerance and fixtures landed in one commit; the order is unprovable"
        )
    merge_base = subprocess.run(
        [git, "-C", str(root), "merge-base", "--is-ancestor", tol, fix]
    )
    assert merge_base.returncode == 0, (
        f"the tolerance commit {tol[:8]} is not an ancestor of the fixture commit "
        f"{fix[:8]}; ADR-0002's claim that the tolerance was committed first no "
        f"longer holds"
    )
