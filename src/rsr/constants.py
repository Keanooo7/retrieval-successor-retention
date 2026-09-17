"""The constants registry (spec section 4.5).

Every constant in the project is classified FROZEN, MEASURED, DERIVED or
CONDITIONAL. Reading a MEASURED or DERIVED constant before its source experiment
has logged a value **raises**, with a message naming the experiment.

This exists because section 4.5's rule -- "never freeze an unmeasured constant" --
is stated as discipline, and discipline is what failed in v0.4. Defect D-1 was a
frozen `gamma_b = 0.001` that could not move the argmin it governed: not a tuned
hyperparameter, a disabled mechanism, and one that would have reported "no effect"
in week 7. Defect "self-audit" was the same disease in `K`. A registry that
refuses the read turns the rule into a mechanical guarantee.

Two refinements beyond the kickoff's seed table, both forced by the spec:

* `M` and `S` are FROZEN *per scope*, not globally (sections 5.1, 5.2). A bare
  `get("M")` raises rather than guessing between 16, 40 and 8. Section 13 is
  explicit that no cross-corpus comparison of absolute numbers is valid, so a
  scope-free read of `M` is always a bug.
* `A_max` is CONDITIONAL on `S` (section 3.6a): `A_max <= S`, and it is swept only
  where `S` makes it bind.

See `docs/spec-corrections.md` -- corrections 1, 4 and 9 all land here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "REGISTRY",
    "ConstantError",
    "ConstraintViolation",
    "Klass",
    "Registry",
    "ScopeRequired",
    "UnknownScope",
    "UnmeasuredConstant",
    "check_gamma_horizon",
    "get",
    "record",
]


class Klass(StrEnum):
    """How a constant acquires its value."""

    FROZEN = "FROZEN"
    """Fixed by the spec. Safe to read at any time."""

    MEASURED = "MEASURED"
    """Produced by an experiment. Reading before that experiment logs raises."""

    DERIVED = "DERIVED"
    """Computed from other constants. Raises if any dependency is unmeasured."""

    CONDITIONAL = "CONDITIONAL"
    """Valid only against a context that must be supplied and is validated."""


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ConstantError(Exception):
    """Base for every registry refusal."""


class UnmeasuredConstant(ConstantError):
    """A MEASURED or DERIVED constant was read before its experiment logged it."""


class ScopeRequired(ConstantError):
    """A per-scope constant was read without a scope."""


class UnknownScope(ConstantError):
    """A per-scope constant was read with a scope it does not define."""


class ConstraintViolation(ConstantError):
    """A CONDITIONAL constant's constraint failed against the supplied context."""


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Definition:
    name: str
    klass: Klass
    spec_ref: str
    note: str
    value: Any = None
    scoped: Mapping[str, Any] | None = None
    source_experiment: str | None = None
    depends_on: tuple[str, ...] = ()
    derive: Callable[..., Any] | None = None
    constraint: Callable[[Any, Mapping[str, Any]], str | None] | None = None
    context_keys: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def _a_max_le_s(value: Any, ctx: Mapping[str, Any]) -> str | None:
    """Section 3.6(a): `A_max <= S`, and `A_max` is swept only where `S` binds."""
    s = ctx["S"]
    if value > s:
        return (
            f"A_max = {value} exceeds S = {s}. Section 3.6(a): gaps longer than S do "
            f"not exist in training data, so an A_max above S is inert and produces "
            f"a flat curve that is an artifact, not a finding."
        )
    return None


def check_gamma_horizon(gamma: float, s: int, a_max: int) -> None:
    """Section 4.5: `1/(1-gamma) <= min(S, A_max)`. Raises on violation.

    Not a registry read -- a standalone check, because `gamma` is swept before it
    is frozen and the sweep values must each be validated against the
    configuration they will run in.

    See `docs/spec-corrections.md` correction 9. Section 3.6(a)'s
    `A_max in {16, 32, 64}` is an *illustration of a failure at S = 30*, not a
    prescribed sweep set; reading it as one makes `gamma = 0.97` look illegal on
    synthetic, and it is not.
    """
    if not 0.0 <= gamma < 1.0:
        raise ConstraintViolation(f"gamma = {gamma} is outside [0, 1).")
    if gamma == 0.0:
        return  # The separating control arm (section 2). Horizon is 1 by definition.
    horizon = 1.0 / (1.0 - gamma)
    bound = min(s, a_max)
    if horizon > bound:
        raise ConstraintViolation(
            f"gamma = {gamma} has horizon 1/(1-gamma) = {horizon:.1f}, which exceeds "
            f"min(S, A_max) = min({s}, {a_max}) = {bound}. Section 4.5."
        )


# --------------------------------------------------------------------------- #
# Derivations
# --------------------------------------------------------------------------- #


def _derive_gamma_b(b_max: float, e_lifetime: float) -> float:
    """Section 3.5 item 4: `gamma_b ~= b_max / (0.25 * E[lifetime])`.

    Order 0.05-0.1, **never 0.001**. Defect D-1: at 0.001 with a stream-bounded
    lifetime, the maximum attainable |b| over a slot's entire life is 0.08 SD of a
    z-scored psi_hat -- operationally identical to `b == 0`, which is also the
    section 3.7 reduction condition. The loop would have been called unnecessary
    rather than absent.

    Updates fire only when `u_bar` is outside the `tau` band, empirically on order
    a quarter of steps -- hence the 0.25.
    """
    return b_max / (0.25 * e_lifetime)


# --------------------------------------------------------------------------- #
# The registry contents
# --------------------------------------------------------------------------- #

_DEFINITIONS: tuple[Definition, ...] = (
    Definition(
        name="M",
        klass=Klass.FROZEN,
        spec_ref="5.1",
        note=(
            "Memory capacity. Split by corpus, not global. 16 synthetic (maximum "
            "eviction pressure, exact gap control); 40 corpora ([P2]'s value, "
            "40 * ~25 tokens ~= 1024 ~= GPT-2's context); 8 for E7 with its own "
            "trained model, because at 40 a 10-30 sentence passage never fills "
            "memory and every arm retains byte-identical contents (defect C1)."
        ),
        scoped={"synthetic": 16, "corpora": 40, "e7": 8},
    ),
    Definition(
        name="S",
        klass=Klass.FROZEN,
        spec_ref="5.2",
        note=(
            "Stream length. Gaps longer than S do not exist in training data, so S "
            "silently caps the headline experiment -- this was the largest hole in "
            "v0.1. E3 needs 80 for gaps in (40, 64] to exist at all. E7 is "
            "evaluation only, at passage length, uncut."
        ),
        scoped={
            "diagnostics": 30,
            "synthetic": 48,
            "pg19_e3": 80,
            "wikitext_e4": 30,
            "e7": None,
        },
    ),
    Definition(
        name="K",
        klass=Klass.DERIVED,
        spec_ref="3.4, 4.5",
        note=(
            "Shadow-buffer depth. `K >= max target gap`, NOT `K = M`. Correction 1: "
            "section 3.4's closing line still says `K = M`; the self-audit above it "
            "and the section 4.5 table override that. At M = 40 with E3's target "
            "window (40, 64], `K = 40` censors exactly the events E3 exists to "
            "measure. Cost is O(K*d), negligible."
        ),
        scoped={"synthetic": 40, "pg19": 64},
    ),
    Definition(
        name="b_max",
        klass=Klass.FROZEN,
        spec_ref="4.5",
        note=(
            "Bias clip. Interpretable as one standard deviation because psi_hat is "
            "z-scored across live slots each step (section 3.5 item 1)."
        ),
        value=1.0,
    ),
    Definition(
        name="lambda_return",
        klass=Klass.FROZEN,
        spec_ref="3.3, 4.5",
        note=(
            "1.0 = Monte Carlo, the default (correction 2 / defect D-8). TD(0) is "
            "ablation A8. Sweep lambda only if MC variance is limiting, and only "
            "on synthetic."
        ),
        value=1.0,
    ),
    Definition(
        name="lambda_shadow",
        klass=Klass.FROZEN,
        spec_ref="4.5",
        note=(
            "Down-weight on counterfactual shadow targets. A stated judgement, not "
            "a measurement -- shadow targets are estimates, not observations "
            "(section 7.2). Checked once in A7."
        ),
        value=0.5,
    ),
    Definition(
        name="T_warm",
        klass=Klass.FROZEN,
        spec_ref="3.4",
        note=(
            "Eviction warmup: FIFO until T_warm while psi_hat trains passively on "
            "realized r_i. One epoch, and **reported as a fraction of total "
            "epochs** -- one of three is a different experiment from one of twelve. "
            "Also the clean on-policy/off-policy boundary for section 7.2."
        ),
        value="one_epoch",
    ),
    Definition(
        name="tau",
        klass=Klass.MEASURED,
        spec_ref="3.5 item 3, 4.5",
        note=(
            "Anti-collapse dead band on the EMA utilisation rate. Attention over "
            "slots is heavy-tailed; u_bar may essentially never sit inside a +/-25% "
            "band, leaving b saturated for most slots most of the time. Measure the "
            "distribution on a FIFO run before freezing. Freezing an unmeasured "
            "constant is how v0.2-NP died."
        ),
        source_experiment="E0e",
    ),
    Definition(
        name="E_lifetime",
        klass=Klass.MEASURED,
        spec_ref="3.5 item 4, 4.5",
        note=(
            "Expected slot lifetime under FIFO. Comes free from the same E0e run "
            "that measures the u_bar distribution. Also sets the EMA half-life at "
            "E[lifetime]/4."
        ),
        source_experiment="E0e",
    ),
    Definition(
        name="gamma_b",
        klass=Klass.DERIVED,
        spec_ref="3.5 item 4, 4.5",
        note=(
            "Anti-collapse step size, a timescale rather than a magic number. "
            "Require b to reach O(b_max) within the expected slot lifetime, not "
            "within training. Order 0.05-0.1. Then swept in A5 -- a genuine sweep, "
            "not a sensitivity check around a frozen value."
        ),
        depends_on=("b_max", "E_lifetime"),
        derive=_derive_gamma_b,
        source_experiment="E0e",
    ),
    Definition(
        name="gamma",
        klass=Klass.MEASURED,
        spec_ref="4.5",
        note=(
            "Discount. Swept {0.5, 0.9, 0.97} on synthetic only, then frozen at the "
            "week-4 gate. `gamma = 0` is a CONTROL ARM, not a sweep value -- it "
            "isolates context conditioning from lookahead (section 2) and is the "
            "only arm that licenses the word 'future' in the abstract. 0.99 stays "
            "dropped: horizon 100 exceeds S = 80. Constrained by "
            "`1/(1-gamma) <= min(S, A_max)`; use check_gamma_horizon()."
        ),
        source_experiment="E1",
    ),
    Definition(
        name="beta",
        klass=Klass.MEASURED,
        spec_ref="4.5",
        note=(
            "Retention-loss weight in `L = L_NTP + beta * L_MC`. Swept "
            "{0.01, 0.1, 1.0} on synthetic only, then frozen. Collapses the corpus "
            "sweep from 27 configurations to 3."
        ),
        source_experiment="E1",
    ),
    Definition(
        name="nu",
        klass=Klass.MEASURED,
        spec_ref="3.4, 4.5",
        note=(
            "Redundancy penalty in the eviction rule (defect D-3). Set value under "
            "leave-one-out is submodular, so greedy argmin over independent scores "
            "is not an approximation to marginal value -- under redundancy it is "
            "anti-correlated with it. Swept once on synthetic. `nu = 0` is also "
            "part of the section 3.7 reduction condition."
        ),
        source_experiment="E1",
    ),
    Definition(
        name="A_max",
        klass=Klass.CONDITIONAL,
        spec_ref="3.6(a), 4.5",
        note=(
            "Maximum slot age. Constrained `A_max <= S`. Correction 9: section "
            "3.6(a)'s `{16, 32, 64}` is an illustration of a failure at S = 30, not "
            "a prescribed sweep set. On synthetic A_max = S = 48, non-binding "
            "(ADR-0004). On E3 it is 64, forced by the target gap window. A2 is the "
            "`A_max = M` ablation."
        ),
        scoped={"synthetic": 48, "pg19_e3": 64, "a2_ablation": 40},
        constraint=_a_max_le_s,
        context_keys=("S",),
    ),
)


# --------------------------------------------------------------------------- #
# Ledger + registry
# --------------------------------------------------------------------------- #

_LEDGER_ENV = "RSR_LEDGER"
_DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "measurements" / "ledger.json"

_UNSET = object()


@dataclass
class Registry:
    """Constant definitions plus the measurement ledger they read from."""

    definitions: Mapping[str, Definition]
    ledger_path: Path
    _cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False)

    # -- ledger ------------------------------------------------------------- #

    def _load(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        with self.ledger_path.open() as fh:
            return json.load(fh)

    def measurements(self, name: str, scope: str | None = None) -> list[dict[str, Any]]:
        """Every logged measurement for `name`, oldest first."""
        return [
            r
            for r in self._load()
            if r["name"] == name and (scope is None or r.get("scope") == scope)
        ]

    def record(
        self,
        name: str,
        value: Any,
        *,
        experiment: str,
        git_sha: str,
        run_id: str,
        scope: str | None = None,
        note: str = "",
    ) -> None:
        """Log a measured value. Appends; never overwrites.

        Section 12.4: documented configuration is not evidence of what was
        actually run. Every entry carries the experiment, the commit and the run
        that produced it, so a value can always be traced to a RESULTS.md.
        """
        if name not in self.definitions:
            raise ConstantError(f"{name!r} is not a registered constant.")
        defn = self.definitions[name]
        if defn.klass is Klass.FROZEN:
            raise ConstantError(
                f"{name!r} is FROZEN at {defn.value!r} by spec section "
                f"{defn.spec_ref}. It is not measured. If the spec changed, change "
                f"the definition and say so in docs/spec-corrections.md."
            )
        if defn.source_experiment and experiment != defn.source_experiment:
            raise ConstantError(
                f"{name!r} is sourced from {defn.source_experiment}, not "
                f"{experiment!r} (spec section {defn.spec_ref})."
            )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load()
        records.append(
            {
                "name": name,
                "value": value,
                "scope": scope,
                "experiment": experiment,
                "git_sha": git_sha,
                "run_id": run_id,
                "note": note,
            }
        )
        with self.ledger_path.open("w") as fh:
            json.dump(records, fh, indent=2)
            fh.write("\n")

    # -- reads -------------------------------------------------------------- #

    def definition(self, name: str) -> Definition:
        try:
            return self.definitions[name]
        except KeyError:
            known = ", ".join(sorted(self.definitions))
            raise ConstantError(
                f"{name!r} is not a registered constant. Known: {known}"
            ) from None

    def get(self, name: str, scope: str | None = None, **ctx: Any) -> Any:
        defn = self.definition(name)

        if defn.klass is Klass.FROZEN:
            return self._scoped_or_plain(defn, scope)

        if defn.klass is Klass.CONDITIONAL:
            value = self._scoped_or_plain(defn, scope)
            missing = [k for k in defn.context_keys if k not in ctx]
            if missing:
                raise ScopeRequired(
                    f"{name!r} is CONDITIONAL (spec section {defn.spec_ref}) and "
                    f"needs {missing} supplied as keyword context, e.g. "
                    f"get({name!r}, scope=..., {missing[0]}=...)."
                )
            if defn.constraint is not None:
                problem = defn.constraint(value, ctx)
                if problem:
                    raise ConstraintViolation(problem)
            return value

        if defn.klass is Klass.MEASURED:
            return self._measured(defn, scope)

        if defn.klass is Klass.DERIVED:
            # A DERIVED constant may be a pure lookup (K) or a computation
            # (gamma_b). The lookup form carries `scoped`; the computed form
            # carries `derive` + `depends_on`.
            if defn.derive is None:
                return self._scoped_or_plain(defn, scope)
            args = [self.get(dep) for dep in defn.depends_on]
            return defn.derive(*args)

        raise ConstantError(f"unhandled class {defn.klass!r} for {name!r}")

    # -- helpers ------------------------------------------------------------ #

    def _scoped_or_plain(self, defn: Definition, scope: str | None) -> Any:
        if defn.scoped is None:
            return defn.value
        if scope is None:
            scopes = ", ".join(sorted(defn.scoped))
            raise ScopeRequired(
                f"{defn.name!r} is defined per scope (spec section "
                f"{defn.spec_ref}) and has no global value. Pass one of: {scopes}. "
                f"Section 13: no cross-corpus comparison of absolute numbers is "
                f"valid, so a scope-free read of {defn.name!r} is always a bug."
            )
        if scope not in defn.scoped:
            scopes = ", ".join(sorted(defn.scoped))
            raise UnknownScope(
                f"{defn.name!r} has no scope {scope!r}. Known scopes: {scopes}."
            )
        value = defn.scoped[scope]
        if value is None:
            raise ScopeRequired(
                f"{defn.name!r} in scope {scope!r} is set by the data, not by the "
                f"registry (spec section {defn.spec_ref}). {defn.note}"
            )
        return value

    def _measured(self, defn: Definition, scope: str | None) -> Any:
        records = self.measurements(defn.name, scope)
        if not records:
            where = f" in scope {scope!r}" if scope else ""
            raise UnmeasuredConstant(
                f"{defn.name!r} is {defn.klass.value} and has no logged value"
                f"{where}. It is produced by **{defn.source_experiment}** (spec "
                f"section {defn.spec_ref}).\n\n"
                f"  {defn.note}\n\n"
                f"Run {defn.source_experiment}, then log the result with "
                f"rsr.constants.record(). Do not hardcode a value here -- section "
                f"4.5: never freeze an unmeasured constant."
            )
        return records[-1]["value"]


def _ledger_path() -> Path:
    override = os.environ.get(_LEDGER_ENV)
    return Path(override) if override else _DEFAULT_LEDGER


REGISTRY = Registry(
    definitions={d.name: d for d in _DEFINITIONS},
    ledger_path=_ledger_path(),
)


def get(name: str, scope: str | None = None, **ctx: Any) -> Any:
    """Read a constant. Raises if it is not yet measurable. See `Registry.get`."""
    return REGISTRY.get(name, scope, **ctx)


def record(name: str, value: Any, **kw: Any) -> None:
    """Log a measured value. See `Registry.record`."""
    REGISTRY.record(name, value, **kw)
