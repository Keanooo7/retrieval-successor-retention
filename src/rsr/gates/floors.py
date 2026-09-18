"""Ratchets with a direction, and five exit codes that mean five different things.

A floor that can move both ways is not a floor, so every metric here declares a ``Direction``.

The exit codes matter more than they look:

``0`` at or above the floor · ``1`` FLOOR DROP · ``2`` floor UNKNOWN ·
``3`` **ENVIRONMENT / DID NOT RUN** · ``4`` UNBANKED RISE

🔴 ``2`` and ``3`` are separate deliberately, and ``3`` is the one this design can be silently
defeated on. *"The gate did not run"* and *"the gate found nothing"* are different facts, and a
caller that checks only ``!= 0``, or only ``== 1``, merges them. **Exit 3 is not a pass.** In ML
this is the sharpest version of the problem it can be: a training run that produced no metric looks
exactly like a run that produced a bad one.

``4`` exists so that a rise the runner is structurally unable to bank does not fire as a failure at
the wrong moment. A gate that cries wolf gets relaxed rather than debugged.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from enum import StrEnum as _StrEnum
from pathlib import Path

__all__ = ["Direction", "Exit", "FloorResult", "check_floor", "record_floor", "repo_root"]


class Exit(IntEnum):
    OK = 0
    DROP = 1
    UNKNOWN = 2
    DID_NOT_RUN = 3
    UNBANKED_RISE = 4


class Direction(_StrEnum):
    MAY_RISE = "may-rise-never-fall"
    MAY_FALL = "may-fall-never-rise"


@dataclass
class FloorResult:
    key: str
    measured: float | None
    floor: float | None
    direction: Direction
    code: Exit
    message: str

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.key}: {self.message}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _store(name: str) -> Path:
    return repo_root() / ".rsr" / name


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def check_floor(
    key: str,
    measured: float | None,
    *,
    direction: Direction,
    store: str = "gate-floors.json",
    hard_limit: float | None = None,
) -> FloorResult:
    """``measured=None`` means the measurement did not happen -- exit 3, never exit 0."""
    data = _load(_store(store))
    entry = data.get(key)

    if measured is None:
        return FloorResult(
            key, None, (entry or {}).get("value"), direction, Exit.DID_NOT_RUN,
            "measurement did not run. This is NOT a pass -- it is the absence of an answer.",
        )

    if hard_limit is not None:
        over = measured > hard_limit if direction is Direction.MAY_FALL else measured < hard_limit
        if over:
            return FloorResult(
                key, measured, (entry or {}).get("value"), direction, Exit.DROP,
                f"{measured:g} breaches the HARD LIMIT {hard_limit:g}. "
                "This is a spec constraint, not a ratchet -- it does not move.",
            )

    if entry is None:
        return FloorResult(
            key, measured, None, direction, Exit.UNKNOWN,
            f"no recorded floor. Measured {measured:g}. "
            f"Bank it with: rsr floor --record {key}={measured:g}",
        )

    floor = float(entry["value"])
    if direction is Direction.MAY_RISE:
        if measured < floor:
            return FloorResult(
                key, measured, floor, direction, Exit.DROP,
                f"{measured:g} is BELOW the floor {floor:g}. The floor may rise, never fall.",
            )
        if measured > floor:
            return FloorResult(
                key, measured, floor, direction, Exit.UNBANKED_RISE,
                f"{measured:g} is above the floor {floor:g} and the floor was not updated.",
            )
    else:
        if measured > floor:
            return FloorResult(
                key, measured, floor, direction, Exit.DROP,
                f"{measured:g} is ABOVE the ceiling {floor:g}. The floor may fall, never rise.",
            )
        if measured < floor:
            return FloorResult(
                key, measured, floor, direction, Exit.UNBANKED_RISE,
                f"{measured:g} improves on {floor:g} and the floor was not updated.",
            )

    return FloorResult(key, measured, floor, direction, Exit.OK, f"at the floor ({floor:g}).")


def record_floor(
    key: str,
    value: float,
    *,
    direction: Direction,
    evidence: str,
    store: str = "gate-floors.json",
) -> None:
    """Bank a measurement. ``evidence`` names the command or file that produced it.

    A number whose provenance is wrong is harder to correct than one that is simply old, so the
    commit is stamped from git **here**, inside the tree that was measured -- never passed in.
    """
    if not evidence.strip():
        raise ValueError("recording a floor needs evidence: the command or file that produced it")
    path = _store(store)
    data = _load(path)
    data[key] = {
        "value": value,
        "direction": str(direction),
        "evidence": evidence.strip(),
        "commit": _head_sha(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def measure_pytest_count() -> int | None:
    """Run the suite and parse **pytest's own** summary. Returns ``None`` if it did not run.

    The number is never typed by a human and never passed as an argument -- that is the step where
    one digit changes. If the suite reports failures the count is refused outright: a passing count
    measured on a red suite is not a floor.
    """
    # Parse pytest's JUnit XML, not its human summary. pyproject sets addopts="-q" and an
    # explicit -q made -qq, which suppresses the summary line entirely -- so this function
    # returned None on every machine and the ratchet reported DID_NOT_RUN forever. The XML
    # attributes exist at any verbosity, and `skipped` becomes a first-class field instead
    # of being invisible.
    import tempfile
    import xml.etree.ElementTree as ET

    with tempfile.TemporaryDirectory() as td:
        xml = Path(td) / "report.xml"
        try:
            subprocess.run(
                ["uv", "run", "pytest", "--tb=no", f"--junitxml={xml}"],
                cwd=repo_root(), capture_output=True, text=True, timeout=1800,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if not xml.exists():
            return None
        suite = ET.parse(xml).getroot().find("testsuite")
        if suite is None:
            return None
        g = lambda k: int(suite.get(k, 0))  # noqa: E731
        if g("failures") or g("errors"):
            return None  # a passing count measured on a red suite is not a floor
        return g("tests") - g("skipped") - g("failures") - g("errors")
