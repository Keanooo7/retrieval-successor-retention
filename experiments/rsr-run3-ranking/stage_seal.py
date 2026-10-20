"""R-SEAL (PREREG rsr-run3-ranking §1): phase 1's G-C1 seal, extended to stages.

PREREG ``experiments/rsr-run3-ranking/PREREG.md`` (committed alone at 32475b5) §1:
every document constructor phase 2 can reach is wrapped for the whole process by a
guard that raises ``SealedRange`` for any id in a range **sealed at that stage**:

    stage                                   TUNE     EVAL2
    2a fit                                  sealed   sealed
    2a select (TUNE deploys)                OPEN     sealed
    2a deploy (EVAL), finalize              sealed   sealed
    2b smoke, train                         sealed   sealed
    2b measure (after every footer)         sealed   OPEN
    2b finalize                             sealed   sealed

**Phase 1's guard is reused, never copied** (``experiments/post-fails-controls/
common.py``, loaded under its one shared name ``pfc_common``): its ``install_seal``
wraps ``rsr.data.synthetic._generate_document`` and ``generate`` and rebinds every
module-level reference to them; the bounds are machine-read from the phase-1 PREREG
front matter (``sealed_ranges_source``, R-SEAL (c)). That guard seals TUNE and EVAL2
together; this module then **narrows** the guarded set to the stage's sealed ranges by
setting the guard's live range table (``pfc_common._SEAL["ranges"]``, which the guard
reads at call time) to a SUBSET of the file's ranges. It can only remove the range the
stage opens; it can never add a range, widen one, or seal less than the stage says.

**One stage per process.** A second ``install`` for a different stage in the same
process raises: the seal cannot be re-widened once narrowed, and two stages sharing a
process would let a range opened for one leak into the other. Every stage is its own
subcommand, so its own process.

G-C1(e) / R-SEAL (d): phase 1's branch search is re-run unchanged
(``pfc_common.branch_search``); its hits are re-classified against phase 1's allowlist
plus phase 2's own registration and generated outputs (the PREREG names the sealed
bounds in its table, so a search that did not allow it would flag this file's own
registration). Nothing in phase 1's module is modified.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PFC_COMMON = ROOT / "experiments" / "post-fails-controls" / "common.py"
SEALED_SOURCE = "experiments/post-fails-controls/PREREG.md"  # front matter `sealed:`

#: PREREG §1's table. A stage absent here cannot install a seal.
STAGES: dict[str, dict[str, tuple[str, ...]]] = {
    "2a": {
        "launch-checks": ("TUNE", "EVAL2"),
        "time": ("TUNE", "EVAL2"),
        "slot-check": ("TUNE", "EVAL2"),
        "fit": ("TUNE", "EVAL2"),
        "select": ("EVAL2",),
        "deploy": ("TUNE", "EVAL2"),
        "finalize": ("TUNE", "EVAL2"),
    },
    "2b": {
        "plan": ("TUNE", "EVAL2"),
        "launch-checks": ("TUNE", "EVAL2"),
        "smoke": ("TUNE", "EVAL2"),
        "train": ("TUNE", "EVAL2"),
        "measure": ("TUNE",),
        "finalize": ("TUNE", "EVAL2"),
    },
}

#: R-SEAL (d): phase 2's own registration and generated outputs, on top of phase 1's
#: ``SEARCH_ALLOWED``. Everything else is a new hit.
PHASE2_ALLOWED = (
    re.compile(r"^experiments/rsr-run3-ranking/PREREG\.md$"),
    re.compile(r"^experiments/rsr-run3-ranking/RESULTS[^/]*\.md$"),
    re.compile(r"^runs/rsr-run3-[^/]+/.*\.json$"),
    re.compile(r"^runs/rsr-run3-[^/]+/RESULTS[^/]*\.md$"),
)

_STATE: dict[str, Any] = {"stage": None, "sealed": None, "opened": None}


def common():
    """Phase 1's ``common.py`` under its one shared module name (the seal is
    process-wide state; two instances would split it)."""
    name = "pfc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PFC_COMMON)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


C = common()
SealedRange = C.SealedRange


def sealed_ranges() -> dict[str, tuple[int, int]]:
    """TUNE and EVAL2, machine-read from the phase-1 PREREG front matter."""
    return C.sealed_ranges(ROOT / SEALED_SOURCE)


def install(run: str, stage: str) -> dict:
    """Install the process-wide guard and narrow it to ``(run, stage)``'s sealed set.
    Idempotent for the same stage; raises for a different one."""
    if run not in STAGES or stage not in STAGES[run]:
        raise C.ControlFailure(f"R-SEAL: no seal is registered for {run} {stage!r}")
    key = (run, stage)
    if _STATE["stage"] is not None and _STATE["stage"] != key:
        raise C.ControlFailure(
            f"R-SEAL: this process already runs stage {_STATE['stage']}; one stage "
            f"per process ({key} refused)"
        )
    info = C.install_seal(ROOT / SEALED_SOURCE)
    allr = sealed_ranges()
    keep = STAGES[run][stage]
    missing = [k for k in keep if k not in allr]
    if missing:
        raise C.ControlFailure(f"R-SEAL: ranges {missing} are not in {SEALED_SOURCE}")
    narrowed = {k: allr[k] for k in keep}
    if not set(narrowed.items()) <= set(allr.items()):  # never adds or widens
        raise C.ControlFailure(f"R-SEAL: {narrowed} is not a subset of {allr}")
    C._SEAL["ranges"] = narrowed
    _STATE.update(
        stage=key,
        sealed=dict(narrowed),
        opened={k: v for k, v in allr.items() if k not in keep},
    )
    return {
        "run": run,
        "stage": stage,
        "source": SEALED_SOURCE,
        "file_ranges": {k: list(v) for k, v in allr.items()},
        "sealed": {k: list(v) for k, v in narrowed.items()},
        "opened": {k: list(v) for k, v in _STATE["opened"].items()},
        "rebound": info.get("rebound_now", []),
    }


def rebind() -> list[str]:
    """Re-run the rebinding after loading more modules (``from ... import`` copies)."""
    return C._rebind_everywhere()


def census(named: dict[str, tuple[int, int]] | None = None) -> dict:
    """R-SEAL (b): ids generated in this process by range; 0 in every range sealed at
    this stage; the opened range listed with its counts."""
    named = dict(named or {})
    for k, v in (_STATE["opened"] or {}).items():
        named.setdefault(f"{k} (opened at this stage)", v)
    out = C.seal_census(named)
    out["stage"] = list(_STATE["stage"]) if _STATE["stage"] else None
    out["ok"] = bool(out["ok"]) and _STATE["stage"] is not None
    return out


def allowed(path: str) -> bool:
    return C.search_allowed(path) or any(p.match(path) for p in PHASE2_ALLOWED)


def branch_search(**kw) -> dict:
    """R-SEAL (d): phase 1's §1.1 search, every local branch, re-classified with
    phase 2's allowlist added. A git failure stays did-not-run."""
    r = C.branch_search(sealed_ranges(), **kw)
    hits = r.get("hits") or []
    for h in hits:
        h["allowed"] = allowed(h["path"])
    new = [h for h in hits if not h["allowed"]]
    r.update(
        allowed_phase2=[p.pattern for p in PHASE2_ALLOWED],
        n_new_hits=len(new),
        new_hits=new,
    )
    r["ok"] = (
        r.get("for_each_ref_rc") == 0
        and not r.get("errors")
        and not new
        and r.get("n_refs", 0) > 0
    )
    return r
