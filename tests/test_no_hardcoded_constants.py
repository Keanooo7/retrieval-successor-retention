"""Gauntlet 0.3 -- no MEASURED or DERIVED constant may be written as a literal.

`RSRConfig` hardcoded `nu = 0.0` and `beta = 1.0`. Both are MEASURED by E1, both
are refused by the registry by name, and `nu = 0.0` is additionally section 3.7's
*disabled* value -- so the default arm shipped with the redundancy term switched
off. **This is defect D-1's exact shape, one import from the module built to
prevent it**, which is why it gets a mechanical check rather than a code review.

The scan is an AST walk, not a grep: a grep for `nu = 0.0` misses
`nu=float(0)`, `nu: float = 0.0` and a dict entry, and flags the string "nu" in
prose. It looks for a literal number bound to a registry name in any of the four
places a hyperparameter actually enters a program.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from rsr import constants as C

SRC = Path(__file__).resolve().parents[1] / "src" / "rsr"

GUARDED = {
    name
    for name, defn in C.REGISTRY.definitions.items()
    if defn.klass in (C.Klass.MEASURED, C.Klass.DERIVED, C.Klass.CONDITIONAL)
}

EXEMPT_FUNCTIONS = {"reduction_to_tg"}
"""Section 3.7's off-switch values are not hyperparameters.

`nu = 0.0` inside `reduction_to_tg()` is the documented off-switch the spec
*requires* -- "every term added to the eviction score must have a documented
off-switch, or §3.7 silently stops being a reduction." The exemption is narrow, by
function name, and `tests/test_reduction.py` asserts that function does set them,
so the exemption cannot be used to hide a default.
"""

EXEMPT_ASSIGNMENTS = {"REDUCTION_SWITCHES"}
"""The same exemption, for the module-level table `reduction_to_tg()` is built from.

Section 3.7's off-switch set has to be written down *somewhere* as literals, and
one enumeration that a test checks is strictly safer than the values being repeated
inside a constructor. Narrow by name, and `tests/test_reduction.py` asserts the
table covers every field of `RSRConfig`."""

EXEMPT_FILES = {"constants.py"}
"""The registry defines the constants; that is its job."""


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _literal(node.operand)
    return False


def _offences(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    exempt_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in EXEMPT_FUNCTIONS:
            exempt_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(t, ast.Name) and t.id in EXEMPT_ASSIGNMENTS for t in targets
            ):
                exempt_lines.update(
                    range(node.lineno, (node.end_lineno or node.lineno) + 1)
                )

    found: list[str] = []

    def flag(name: str, value: ast.AST, lineno: int) -> None:
        if name in GUARDED and _literal(value) and lineno not in exempt_lines:
            found.append(f"{path.name}:{lineno} {name} = {ast.unparse(value)}")

    for node in ast.walk(tree):
        # x = 0.0   /   x: float = 0.0
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    flag(target.id, node.value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                flag(node.target.id, node.value, node.lineno)
        # def f(nu=0.0)
        elif isinstance(node, ast.FunctionDef):
            args = node.args
            for arg, default in zip(
                args.posonlyargs + args.args, args.defaults or [], strict=False
            ):
                flag(arg.arg, default, node.lineno)
            for arg, default in zip(
                args.kwonlyargs, args.kw_defaults or [], strict=False
            ):
                if default is not None:
                    flag(arg.arg, default, node.lineno)
        # {"nu": 0.0}
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    flag(key.value, value, getattr(key, "lineno", node.lineno))
    return found


@pytest.mark.parametrize(
    "path",
    sorted(p for p in SRC.rglob("*.py") if p.name not in EXEMPT_FILES),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_no_literal_assignment_of_a_registry_constant(path):
    offences = _offences(path)
    assert not offences, (
        "MEASURED/DERIVED/CONDITIONAL constants must come from rsr.constants, which "
        "refuses the read until the experiment logs a value (section 4.5). Found:\n  "
        + "\n  ".join(offences)
    )


def test_the_scan_catches_the_defect_it_was_written_for(tmp_path):
    """Mutation. A check nothing reddens adds nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text("class Cfg:\n    nu: float = 0.0\n    beta: float = 1.0\n")
    assert len(_offences(bad)) == 2


def test_the_scan_does_not_flag_a_non_constant(tmp_path):
    good = tmp_path / "good.py"
    good.write_text("learning_rate = 2.5e-4\nd_model = 128\n")
    assert _offences(good) == []


def test_the_switch_table_exemption_is_narrow(tmp_path):
    """Only the named table is exempt; a second dict of the same shape is not."""
    f = tmp_path / "f.py"
    f.write_text('REDUCTION_SWITCHES = {"nu": 0.0}\nDEFAULTS = {"nu": 0.5}\n')
    offences = _offences(f)
    assert len(offences) == 1 and offences[0].endswith("nu = 0.5")


def test_the_reduction_exemption_is_narrow(tmp_path):
    """The §3.7 off-switch exemption must not extend to a neighbouring function."""
    f = tmp_path / "f.py"
    f.write_text(
        "def reduction_to_tg():\n    nu = 0.0\n    return nu\n"
        "def some_other_builder():\n    nu = 0.0\n    return nu\n"
    )
    offences = _offences(f)
    assert len(offences) == 1 and "5" in offences[0]
