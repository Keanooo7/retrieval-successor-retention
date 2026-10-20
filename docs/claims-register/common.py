"""Shared helpers for the claims register (PLAN-2026-10-19 phase 0.2).

Every value in ``docs/claims-register.md`` is read here from a committed ledger or a
raw record; nothing is typed. Each register row carries one or more ``Check``s: a
command, runnable from a clean checkout of this tree, whose printed output is compared
(string-equal) against the ledger value formatted the same way. ``build.py rederive``
runs the checks and writes their literal output to ``rederive/<experiment>.json``;
``build.py render`` joins rows and outputs into the register.

Raw ``runs/`` artefacts are gitignored. They are read from the read-only archive
``~/rsr-runs-archive/<worktree>/runs/`` (sha256 manifest ``MANIFEST.sha256``); nothing
here writes into the archive or into any other worktree.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # the worktree / clean checkout root
HERE = Path(__file__).resolve().parent
ARCHIVE = Path.home() / "rsr-runs-archive"
PY = ".venv/bin/python"
# Compute (anything that loads records and bootstraps) goes through a lane slot.
SLOT = (
    "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run "
    "--lane cpu-det --slots 1 --wait 3600 -- "
)
LEDGER_GET = f"{PY} docs/claims-register/ledger_get.py"

PREREG = "PREREG'd"
EXPLORATORY = "EXPLORATORY"
SEEN = "SEEN"
UNSEEN = "UNSEEN"


@dataclass
class Check:
    """One re-derivation: run ``cmd`` (from the tree root), take its last stdout line,
    optionally parse it as JSON and pick ``pick`` (a key of that object), format it with
    ``fmt_exact`` and compare string-equal to ``expected``.

    ``kind``: ``raw`` = recomputed from per-document / per-seed raw records via the run's
    own analysis code; ``archive-ledger`` = read from the run's own out-root ledger in the
    archive (not the committed copy); ``ledger`` = read from the committed ledger key (a
    read, not a recomputation; it proves only that the register reads the right key)."""

    label: str
    cmd: str
    expected: str
    kind: str = "raw"
    pick: str | None = None


@dataclass
class Row:
    id: str
    experiment: str
    tier: str  # "headline" | "secondary"
    claim: str
    seeds: list[int]
    values: list[Any]
    source: str
    prereg: str
    seen: str
    seen_why: str
    ci: list[tuple[float, float] | None] | None = None
    ci_method: str = "none computed"
    checks: list[Check] = field(default_factory=list)
    rederive_note: str | None = (
        None  # why there is no check, e.g. "not re-derived (cost)"
    )
    note: str | None = None


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

_LEDGERS: dict[str, dict] = {}


def ledger_path(run_id: str) -> Path:
    return ROOT / "runs" / run_id / "ledger.json"


def ledger(run_id: str) -> dict:
    if run_id not in _LEDGERS:
        _LEDGERS[run_id] = json.loads(ledger_path(run_id).read_text())
    return _LEDGERS[run_id]


def lrows(run_id: str) -> dict[str, Any]:
    """key -> value of the committed ledger's rows; a key that occurs more than once
    resolves to its LAST occurrence (the e0d ledger appends; its own claims read [-1])."""
    out: dict[str, Any] = {}
    for r in ledger(run_id).get("rows") or []:
        if isinstance(r, dict) and isinstance(r.get("key"), str):
            out[r["key"]] = row_value(r)
    return out


def row_value(r: dict) -> Any:
    """A ledger row's value. ``scripts/ledger.py`` writes two row shapes: scalar rows
    ``{key, kind, value, how}`` and statistic rows ``{key, kind, samples, mean, sd, n,
    sd_exactly_zero, how}`` with no ``value``. A statistic row resolves to the row
    minus ``key`` (so ``samples.<s>`` is the per-seed value); reading ``value`` there
    would silently give None."""
    if "value" in r:
        return r["value"]
    return {k: v for k, v in r.items() if k != "key"}


def dig(x: Any, path: str | list[str] | None) -> Any:
    """Follow a dotted path (or list of keys) into nested dicts/lists."""
    if path is None or path == "":
        return x
    parts = path.split(".") if isinstance(path, str) else path
    for p in parts:
        x = x[int(p)] if isinstance(x, list) else x[p]
    return x


def lget(run_id: str, key: str, path: str | None = None) -> Any:
    return dig(lrows(run_id)[key], path)


def jload(p: Path | str) -> Any:
    return json.loads(Path(p).read_text())


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #


def fmt_exact(x: Any) -> str:
    """The comparison format: full-precision repr for floats, JSON for containers."""
    if isinstance(x, bool) or x is None:
        return json.dumps(x)
    if isinstance(x, float):
        return repr(x)
    if isinstance(x, (int, str)):
        return str(x)
    return json.dumps(x, sort_keys=True)


def fmt(x: Any, nd: int = 4) -> str:
    """The display format in the register (the full value is at the cited key)."""
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        if x != 0 and abs(x) < 10 ** (-nd + 1):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def ledger_check(run_id: str, key: str, path: str | None, label: str) -> Check:
    """A committed-ledger read (kind ``ledger``)."""
    p = f" {path}" if path else ""
    return Check(
        label=label,
        cmd=f"{LEDGER_GET} runs/{run_id}/ledger.json '{key}'{p}",
        expected=fmt_exact(lget(run_id, key, path)),
        kind="ledger",
    )


def archive_ledger_check(
    archive_ledger: str, run_id: str, key: str, path: str | None, label: str
) -> Check:
    """Read the same key from the run's own ledger in the archive (kind
    ``archive-ledger``); expected is the committed value."""
    p = f" {path}" if path else ""
    return Check(
        label=label,
        cmd=f"{LEDGER_GET} {archive_ledger} '{key}'{p}",
        expected=fmt_exact(lget(run_id, key, path)),
        kind="archive-ledger",
    )
