"""The per-run measurement ledger: `runs/<id>/ledger.json`.

**Why this exists.** The overnight protocol's rule is that *every number in a report
must trace to a ledger row*. A number typed into prose is a claim; a number a
process wrote next to the command, the git sha and the wall clock that produced it
is a measurement. This module is the only supported way to create the second kind,
and it deliberately makes the first kind inconvenient.

Distinct from `measurements/ledger.json` (§4.5, `rsr.constants.Registry`), which
holds *constants* a named experiment has measured and frozen. This one holds the
raw observations of a single run. A constant graduates from here to there; nothing
travels the other way.

Three refusals, each because the alternative lets a bad number through:

* **`record()` stamps the git sha from inside this tree.** Never accepted as an
  argument -- corpus-sheet defect 8 is a caller-typed sha, which is a provenance
  *claim*. It also records whether the tree was dirty, because a number measured on
  an uncommitted tree is not reproducible from its sha.
* **A multi-seed statistic must carry its per-seed values.** `stat()` computes mean
  and sd from the samples it is given and stores the samples alongside. An `sd` that
  came from nowhere, or an `sd` of exactly 0.0 across seeds that were supposed to
  differ, is the signature of a statistic over one number reported as many.
* **`sd` of a single sample is `None`, not `0.0`.** Zero reads as "measured, and it
  did not vary". None reads as "you have one sample".
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["Ledger", "SingleSample"]

_REPO = Path(__file__).resolve().parents[1]


class SingleSample(ValueError):
    """Raised when a statistic is asked for across fewer than two samples."""


def _git(*args: str) -> str:
    """Homebrew git, explicitly. Apple's refuses to run until an Xcode licence is
    accepted and fails as an *empty answer rather than an error* -- which has
    already produced one false "everything is clean" report on this machine."""
    for exe in ("/opt/homebrew/bin/git", "git"):
        try:
            r = subprocess.run(
                [exe, *args], cwd=_REPO, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return r.stdout.strip()
    return ""


class Ledger:
    """Append-only JSON record for one run.

    >>> led = Ledger("my-run", question="does X differ from Y?")
    >>> led.note("victims_rsr", [3, 1, 0], how="policy.records victim field")
    >>> led.stat("loss_final", [1.10, 1.14, 1.09], how="heartbeat last beat, 3 seeds")
    >>> p = led.write()
    """

    def __init__(self, run_id: str, *, question: str, cycle: int | None = None) -> None:
        self.run_id = run_id
        self.path = _REPO / "runs" / run_id / "ledger.json"
        sha = _git("rev-parse", "HEAD")
        self.doc: dict[str, Any] = {
            "run_id": run_id,
            "cycle": cycle,
            "question": question,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": {
                "git_sha": sha or "unknown",
                "git_dirty": bool(_git("status", "--porcelain")),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "commands": [],
            "rows": [],
            "verdict": None,
        }

    # -- what was run -------------------------------------------------------- #

    def command(
        self, argv: list[str] | str, *, exit_code: int | None = None, note: str = ""
    ) -> None:
        """The literal command line. A ledger without one cannot be re-executed,
        and re-execution is how the manager verifies a claim."""
        self.doc["commands"].append(
            {
                "argv": argv if isinstance(argv, str) else " ".join(argv),
                "exit_code": exit_code,
                "note": note,
                "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    # -- numbers ------------------------------------------------------------- #

    def note(self, key: str, value: Any, *, how: str) -> None:
        """One observed value. `how` says which artefact it was read out of, so a
        reader can go and look at that artefact instead of trusting the row."""
        self.doc["rows"].append(
            {"key": key, "kind": "observation", "value": value, "how": how}
        )

    def stat(
        self, key: str, samples: list[float], *, how: str, allow_single: bool = False
    ) -> dict[str, Any]:
        """Mean and sample sd over `samples`, with the samples kept.

        Refuses one sample unless asked explicitly, and never reports `sd = 0.0`
        for it. A zero sd over seeds that should differ is reported as-is and
        flagged -- it is real, and it usually means the seed did not reach the RNG.
        """
        xs = [float(x) for x in samples]
        if len(xs) < 2 and not allow_single:
            raise SingleSample(
                f"{key!r}: a mean/sd over {len(xs)} sample(s) is not a statistic. "
                f"Pass >=2 samples, or allow_single=True and accept sd=None."
            )
        mean = sum(xs) / len(xs)
        if len(xs) < 2:
            sd: float | None = None
        else:
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
        row = {
            "key": key,
            "kind": "statistic",
            "n": len(xs),
            "samples": xs,
            "mean": mean,
            "sd": sd,
            "how": how,
            "sd_exactly_zero": sd is not None and sd == 0.0,
        }
        self.doc["rows"].append(row)
        return row

    # -- the answer ---------------------------------------------------------- #

    def verdict(self, *, falsifier: str, outcome: str, detail: str) -> None:
        """`outcome` is one of `survived`, `falsified`, `inconclusive`.

        `survived` means the falsifier was run and the claim did not die. It is not
        a synonym for "it worked" -- a falsifier that could not run at all is
        `inconclusive`, and saying so is the honest row.
        """
        if outcome not in {"survived", "falsified", "inconclusive"}:
            raise ValueError(
                f"outcome={outcome!r} is not one of survived / falsified / inconclusive"
            )
        self.doc["verdict"] = {
            "falsifier": falsifier,
            "outcome": outcome,
            "detail": detail,
        }

    def write(self) -> Path:
        self.doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=2, default=str) + "\n")
        return self.path
