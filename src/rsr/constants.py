"""The constants registry.

§4.5 of the spec says "Never freeze an unmeasured constant", and records that freezing one is how
the predecessor project died. This module makes that a mechanical guarantee rather than a
discipline: reading a ``MEASURED`` or ``DERIVED`` constant before its source experiment has logged a
value **raises**, and the exception names the experiment you owe.

Four classes:

``FROZEN``
    The spec fixes the value. Readable immediately.
``MEASURED``
    An experiment measures it. Unreadable until that experiment records a value with evidence.
``DERIVED``
    Computed from measured values by a stated formula. Unreadable until its inputs exist.
``CONDITIONAL``
    Scope-dependent and not yet decided anywhere. Unreadable until explicitly set, with a reason.

Measurements persist to ``.rsr/measurements.json`` so the registry survives a process boundary --
an experiment that measures a constant in one run and a training job that reads it in another are
the normal case, not the exception.

The count of still-unset constants is a **ratchet**: it may fall, never rise. See
``rsr.gates.floors``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "REGISTRY",
    "Class",
    "Constant",
    "ConstantError",
    "Registry",
    "UnknownScopeError",
    "UnmeasuredConstantError",
    "get",
    "record",
]

# Sentinel scope for constants that take a single value everywhere.
GLOBAL = "global"


class Class(StrEnum):
    FROZEN = "FROZEN"
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    CONDITIONAL = "CONDITIONAL"


class ConstantError(RuntimeError):
    """Base for every registry refusal."""


class UnmeasuredConstantError(ConstantError):
    """Raised when a MEASURED/DERIVED/CONDITIONAL constant is read before it has a value.

    The message names the experiment that owes the value, because the whole point of this class is
    that the reader should not have to go looking.
    """


class UnknownScopeError(ConstantError):
    """Raised when a scoped constant is read without a scope, or with one it does not define."""


@dataclass(frozen=True)
class Constant:
    name: str
    cls: Class
    source: str
    """Spec section that fixes or assigns this constant."""
    note: str = ""
    experiment: str | None = None
    """Which experiment measures or derives it. Required for MEASURED and DERIVED."""
    frozen_values: dict[str, Any] = field(default_factory=dict)
    """scope -> value, for FROZEN constants."""
    scoped: bool = False
    """True if the constant takes a different value per corpus/experiment."""
    derives_from: tuple[str, ...] = ()
    """Names of the constants this one is computed from, for DERIVED."""
    formula: str = ""

    @property
    def needs_measurement(self) -> bool:
        return self.cls in (Class.MEASURED, Class.DERIVED, Class.CONDITIONAL)


# --------------------------------------------------------------------------------------------
# The registry contents. Every entry cites the spec section that put it there.
# Corrections that override the spec body are marked and cross-referenced to
# docs/spec-corrections.md -- do not "fix" them back to the body's reading.
# --------------------------------------------------------------------------------------------

_CONSTANTS: tuple[Constant, ...] = (
    Constant(
        name="M",
        cls=Class.FROZEN,
        source="§5.1",
        scoped=True,
        frozen_values={"synthetic": 16, "corpora": 40, "e7": 8},
        note="Split, not global. M=16 on corpora was a v0.1 error; E7 gets its own trained model.",
    ),
    Constant(
        name="S",
        cls=Class.FROZEN,
        source="§5.2",
        scoped=True,
        frozen_values={
            "diagnostics": 30,
            "synthetic": 48,
            "e3": 80,
            "e4_start": 30,
        },
        note="E3 is 80 fixed, no curriculum. See spec-corrections S-3 -- that is [P2]'s worst "
        "measured ablation and 'required' is not established.",
    ),
    Constant(
        name="L_max",
        cls=Class.FROZEN,
        source="§5.1 / [P2]",
        frozen_values={GLOBAL: 64},
        note="Tokens per sentence. Verified against [P2].",
    ),
    Constant(
        name="n_layers",
        cls=Class.FROZEN,
        source="§5.1 / [P2]",
        frozen_values={GLOBAL: 12},
    ),
    Constant(
        name="gestalt_layer",
        cls=Class.FROZEN,
        source="§5.1 / [P2]",
        frozen_values={GLOBAL: 7},
        note="Contextualised hidden state of <EOS> at layer 7. Verified against [P2].",
    ),
    Constant(
        name="b_max",
        cls=Class.FROZEN,
        source="§3.5",
        frozen_values={GLOBAL: 1.0},
        note="One SD, because psi-hat is z-scored across live slots.",
    ),
    Constant(
        name="lambda_return",
        cls=Class.FROZEN,
        source="§3.3 / correction C-2",
        frozen_values={GLOBAL: 1.0},
        note="1.0 = Monte Carlo, the DEFAULT. TD(0) is ablation A8. The spec body still composes "
        "L_TD; read it as L_MC.",
    ),
    Constant(
        name="lambda_shadow",
        cls=Class.FROZEN,
        source="§4.5",
        frozen_values={GLOBAL: 0.5},
        note="A stated judgement, not a measurement. Checked once in A7.",
    ),
    Constant(
        name="K",
        cls=Class.FROZEN,
        source="§3.4 self-audit + §4.5 / correction C-1",
        scoped=True,
        frozen_values={"pg19": 64, "synthetic": 40},
        note="K >= max target gap. NOT M. §3.4's closing line still says K=M and is superseded -- "
        "K=M=40 censors exactly the (40,64] events E3 measures.",
    ),
    Constant(
        name="T_warm",
        cls=Class.FROZEN,
        source="§3.4",
        frozen_values={GLOBAL: 1},
        note="One epoch. MUST be reported as a fraction of total epochs -- one of three is a "
        "different experiment from one of twelve.",
    ),
    # ---- measured -------------------------------------------------------------------------
    Constant(
        name="tau",
        cls=Class.MEASURED,
        source="§3.5 / §4.5",
        experiment="E0e",
        note="Dead-band half-width for the bias loop. Measured from the u-bar distribution on a "
        "FIFO run. Attention over slots is heavy-tailed; a frozen +/-25% band may never be "
        "occupied, leaving b saturated.",
    ),
    Constant(
        name="E_lifetime",
        cls=Class.MEASURED,
        source="§3.5",
        experiment="E0e",
        note="Expected slot lifetime, from the same FIFO run that measures u-bar.",
    ),
    Constant(
        name="ema_half_life",
        cls=Class.DERIVED,
        source="§3.5",
        experiment="E0e",
        derives_from=("E_lifetime",),
        formula="E_lifetime / 4",
        note="u-bar must be an EMA RATE, never a cumulative sum -- a running sum accumulates with "
        "age by construction and smuggles recency back in. NB spec-corrections S-2: H2O App. B.2 "
        "reports that replacing accumulation with an average DEGRADED performance for them.",
    ),
    Constant(
        name="gamma_b",
        cls=Class.DERIVED,
        source="§3.5 item 4 / D-1",
        experiment="E0e",
        derives_from=("b_max", "E_lifetime"),
        formula="b_max / (0.25 * E_lifetime)",
        note="Order 0.05-0.1, NOT 0.001. At 0.001 the whole anti-collapse section is inert: "
        "0.001*80 = 0.08 SD over a full stream, which cannot move the argmin.",
    ),
    Constant(
        name="gamma",
        cls=Class.MEASURED,
        source="§4.5",
        experiment="E1",
        note="Swept {0.5, 0.9, 0.97} on synthetic then FROZEN at the week-4 gate. gamma=0 is a "
        "CONTROL ARM, not a sweep value. Constrained by 1/(1-gamma) <= min(S, A_max) -- see "
        "A_max_synthetic (correction B-5), which binds this sweep.",
    ),
    Constant(
        name="beta",
        cls=Class.MEASURED,
        source="§4.5",
        experiment="E1",
        note="Swept {0.01, 0.1, 1.0} on synthetic then frozen.",
    ),
    Constant(
        name="nu",
        cls=Class.MEASURED,
        source="§3.4 / §4.5",
        experiment="E1",
        note="Redundancy penalty. nu=0 is part of the §3.7 reduction condition.",
    ),
    Constant(
        name="sent_per_sec",
        cls=Class.MEASURED,
        source="correction B-2",
        experiment="E0c",
        note="MUST be measured at the widths that will actually run. The spec's budget scales from "
        "[P2]'s 21 sent/sec, which is at d_model=768 / 85.6M params -- ~4x wider than anything RSR "
        "trains. Do not inherit 21.",
    ),
    Constant(
        name="e0i_events_per_bucket_floor",
        cls=Class.MEASURED,
        source="§6 E0i / §16 condition 1",
        experiment="T2 pre-registration",
        note="PRE-REGISTERED, not measured from the data. Must be committed BEFORE the histogram "
        "is computed, by someone who has not seen it. Registering it afterwards makes the gate "
        "unfalsifiable, and this gate stands in front of 576 GPU-hours.",
    ),
    # ---- conditional ----------------------------------------------------------------------
    Constant(
        name="A_max_synthetic",
        cls=Class.CONDITIONAL,
        source="§3.6a / §4.5 / correction B-5",
        experiment="an owner decision, before E1",
        note="UNSET IN THE SPEC AND BINDING. Synthetic is S=48, M=16. If A_max defaults to M=16, "
        "gamma=0.97 (horizon 33) violates 1/(1-gamma) <= min(S, A_max) -- the top value of the "
        "gamma sweep is illegal on the only corpus it is swept on. Set A_max >= 33, or drop "
        "gamma=0.97 and say so.",
    ),
)

_BY_NAME = {c.name: c for c in _CONSTANTS}


@dataclass
class Measurement:
    value: Any
    experiment: str
    evidence: str
    recorded_at: str


def _default_store() -> Path:
    override = os.environ.get("RSR_MEASUREMENTS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / ".rsr" / "measurements.json"


class Registry:
    def __init__(self, store: Path | None = None) -> None:
        self._store = store or _default_store()
        self._measured: dict[str, Measurement] = {}
        self._load()

    # -- persistence -------------------------------------------------------------------------

    def _load(self) -> None:
        if not self._store.exists():
            return
        raw = json.loads(self._store.read_text())
        for key, m in raw.items():
            self._measured[key] = Measurement(**m)

    def _save(self) -> None:
        self._store.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: vars(m) for k, m in sorted(self._measured.items())}
        self._store.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _key(name: str, scope: str | None) -> str:
        return f"{name}@{scope}" if scope else name

    # -- reads -------------------------------------------------------------------------------

    def spec(self, name: str) -> Constant:
        try:
            return _BY_NAME[name]
        except KeyError:
            known = ", ".join(sorted(_BY_NAME))
            raise ConstantError(f"{name!r} is not a registered constant. Known: {known}") from None

    def get(self, name: str, scope: str | None = None) -> Any:
        c = self.spec(name)

        if c.cls is Class.FROZEN:
            if c.scoped:
                if scope is None:
                    raise UnknownScopeError(
                        f"{name!r} is scoped ({c.source}) and takes a different value per scope: "
                        f"{sorted(c.frozen_values)}. Pass scope=."
                    )
                if scope not in c.frozen_values:
                    raise UnknownScopeError(
                        f"{name!r} has no value for scope {scope!r}. "
                        f"Defined: {sorted(c.frozen_values)}."
                    )
                return c.frozen_values[scope]
            return c.frozen_values[GLOBAL]

        if c.cls is Class.DERIVED:
            return self._derive(c, scope)

        key = self._key(name, scope)
        if key in self._measured:
            return self._measured[key].value
        if scope is not None and name in self._measured:
            return self._measured[name].value
        raise UnmeasuredConstantError(self._refusal(c, scope))

    def _derive(self, c: Constant, scope: str | None) -> Any:
        key = self._key(c.name, scope)
        if key in self._measured:
            return self._measured[key].value
        missing = [d for d in c.derives_from if not self.is_set(d, scope)]
        if missing:
            raise UnmeasuredConstantError(self._refusal(c, scope, missing_inputs=missing))
        inputs = {d: self.get(d, scope) for d in c.derives_from}
        if c.name == "gamma_b":
            return inputs["b_max"] / (0.25 * inputs["E_lifetime"])
        if c.name == "ema_half_life":
            return inputs["E_lifetime"] / 4.0
        raise ConstantError(f"{c.name!r} is DERIVED but has no derivation implemented.")

    def _refusal(
        self, c: Constant, scope: str | None, missing_inputs: list[str] | None = None
    ) -> str:
        where = f" (scope {scope!r})" if scope else ""
        lines = [
            f"{c.name!r}{where} is {c.cls} and has no value yet.",
            f"  spec:       {c.source}",
            f"  owed by:    {c.experiment}",
        ]
        if c.formula:
            lines.append(f"  formula:    {c.formula}")
        if missing_inputs:
            lines.append(f"  missing:    {', '.join(missing_inputs)}")
        if c.note:
            lines.append(f"  why:        {c.note}")
        lines.append(
            f"  to resolve: rsr.constants.record({c.name!r}, <value>, "
            f"experiment={c.experiment!r}, evidence=<path or command>)"
        )
        return "\n".join(lines)

    def is_set(self, name: str, scope: str | None = None) -> bool:
        try:
            self.get(name, scope)
        except ConstantError:
            return False
        return True

    # -- writes ------------------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: Any,
        *,
        experiment: str,
        evidence: str,
        scope: str | None = None,
        recorded_at: str | None = None,
    ) -> None:
        """Log a measured value. ``evidence`` must name a file or a command, never a claim."""
        c = self.spec(name)
        if c.cls is Class.FROZEN:
            raise ConstantError(
                f"{name!r} is FROZEN by {c.source}; it is not measured. "
                "If the spec is wrong, correct the spec, not the registry."
            )
        if not evidence or not evidence.strip():
            raise ConstantError(
                f"recording {name!r} needs evidence -- a path or a command that produced the "
                "number. 'measured it' is a claim, not evidence."
            )
        from datetime import datetime

        self._measured[self._key(name, scope)] = Measurement(
            value=value,
            experiment=experiment,
            evidence=evidence.strip(),
            recorded_at=recorded_at or datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._save()

    # -- the ratchet -------------------------------------------------------------------------

    def unset(self) -> list[str]:
        """Names of constants that still owe a value. This count is a floor: it may fall,
        never rise."""
        out = []
        for c in _CONSTANTS:
            if not c.needs_measurement:
                continue
            if not self.is_set(c.name):
                out.append(c.name)
        return sorted(out)

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for c in _CONSTANTS:
            rows.append(
                {
                    "name": c.name,
                    "class": str(c.cls),
                    "source": c.source,
                    "experiment": c.experiment or "",
                    "set": not c.needs_measurement or self.is_set(c.name),
                }
            )
        return rows


REGISTRY = Registry()


def get(name: str, scope: str | None = None) -> Any:
    return REGISTRY.get(name, scope)


def record(name: str, value: Any, **kw: Any) -> None:
    REGISTRY.record(name, value, **kw)
