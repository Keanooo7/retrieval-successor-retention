"""The run heartbeat (gauntlet 3.8).

One JSONL line per interval, appended and fsynced, so that **the whole night is reconstructable
from this file alone, with no agent present.** That is the point: an unattended run whose only
record is an agent's summary is an unattended run you cannot audit.

Four things every beat carries, because §3.4 and D-D require them *from the first policy run* and
not as an afterthought:

``loss``
    The thing being optimised.
``gini``
    Attention-share Gini across live slots -- the collapse monitor. §7.4: *rising Gini with flat
    loss is collapse in progress*, and it is invisible in the loss alone.
``attribution``
    §3.4: the fraction of eviction argmins that ``b`` and ``nu`` each flipped. *"If ``b`` flips a
    large share, the balance controller is the policy"* -- the v0.2 objection D-1's fix reopened.
``rank_shift``
    D-D. Under FIFO a slot's positional encoding means recency; under RSR, evicting a middle slot
    shifts the ranks of everything behind it, so ``P^(sent)`` becomes a function of which slots the
    policy killed. This is displacement from the FIFO counterfactual, which is 0 under FIFO by
    construction. **E0b cannot see this confound** -- under §3.7 the policy *is* FIFO -- so the only
    way it is ever observed is by logging it here.

🔴 **A crash lands in this file.** ``crash()`` writes a terminal record with the traceback before
the exception propagates. A run that dies at 03:00 must not be indistinguishable in the morning
from a run that never started.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

__all__ = ["Heartbeat"]


class Heartbeat:
    """Append-only JSONL run record. Every write is flushed and fsynced."""

    def __init__(self, path: Path | str, *, run_id: str, provenance: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.provenance = provenance
        self._t0 = time.time()
        self._fh = self.path.open("a", buffering=1)
        self._closed = False

    # -- the write primitive -------------------------------------------------------------------

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        rec = {
            "kind": kind,
            "run_id": self.run_id,
            "elapsed_s": round(time.time() - self._t0, 3),
            **payload,
        }
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())  # survive a SIGKILL, not just a clean exit

    # -- the three record kinds ----------------------------------------------------------------

    def header(self, **fields: Any) -> None:
        """First line. Carries provenance, so no number in this file is orphaned from its origin.

        A metric without its git sha, device, seed and config hash is not reproducible and not
        locatable -- which is the whole reason the return contract demands a provenance line.
        """
        self._write("header", {"provenance": self.provenance, **fields})

    def beat(
        self,
        step: int,
        *,
        loss: float | None = None,
        gini: float | None = None,
        attribution: dict[str, float] | None = None,
        rank_shift: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """One interval.

        ``loss=None`` is written as null rather than skipped. **A missing measurement is a fact and
        must look like one** -- a beat with no loss is a beat that did not measure a loss, and that
        is different from a beat that measured zero.
        """
        self._write(
            "beat",
            {
                "step": step,
                "loss": loss,
                "gini": gini,
                "attribution": attribution,
                "rank_shift": rank_shift,
                **extra,
            },
        )

    def crash(self, exc: BaseException) -> None:
        """Terminal record for an exception. Call this BEFORE re-raising."""
        self._write(
            "crash",
            {
                "status": "crashed",
                "exc_type": type(exc).__name__,
                "exc": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        self.close()

    def footer(self, status: str = "completed", **fields: Any) -> None:
        self._write("footer", {"status": status, **fields})
        self.close()

    # -- lifecycle -----------------------------------------------------------------------------

    def close(self) -> None:
        if not self._closed:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> Heartbeat:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.crash(exc)
        elif not self._closed:
            self.footer("completed")
        return False  # never swallow
