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

## Gauntlet 0.7 -- seven defects, all fixed here

Recorded so the guarantees are legible rather than archaeological:

1. **Scope-free MEASURED reads leaked across scopes.** `get("tau")` returned a
   value recorded under `scope="synthetic"`. Section 13 forbids exactly that
   comparison. A scope-free read now raises once any scoped measurement exists.
2. **`record()` accepted DERIVED names that `get()` then ignored.** Logging
   `gamma_b = 999.0` succeeded and changed nothing -- the worst shape of defect,
   since the ledger then carries a number no code reads. DERIVED is now refused
   at `record()`.
3. **`A_max`'s guard trusted a caller-supplied `S`.** `get("A_max", "synthetic",
   S=10**9)` passed. `S` is now cross-checked against the registry's own value for
   the paired scope.
4. **`A_max` with `S=None` raised a bare `TypeError`**, outside the
   `ConstantError` hierarchy, so a caller catching registry refusals missed it.
5. **`K` accepted a measurement from any experiment name** -- it is DERIVED with
   no `source_experiment`, so the experiment check did not fire. Subsumed by (2).
6. **The scope vocabulary had no entry for E4, E7 or diagnostics.** A scope is now
   either a value or an `Unspecified` refusal that names the spec gap, and
   `SCOPES` is closed: a test asserts every scoped definition covers it.
7. **`git_sha` was caller-typed and never verified.** `record()` no longer accepts
   it; the sha is stamped from git inside the measured tree, with the dirty flag.

## D-I -- `T_warm` has one representation

The registry held the string `"one_epoch"` and `RSRConfig` held a float, with
nothing converting between them. `T_warm` is now CONDITIONAL on `steps_per_epoch`
and **returns a float number of steps**. `T_warm_epochs` holds section 3.4's "one
epoch" as the FROZEN quantity it actually is, and the conversion lives in one
place. The string is deleted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "REGISTRY",
    "SCOPES",
    "ConstantError",
    "ConstraintViolation",
    "Klass",
    "Registry",
    "ScopeRequired",
    "UnknownScope",
    "UnmeasuredConstant",
    "Unspecified",
    "check_gamma_horizon",
    "get",
    "git_provenance",
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
    """Base for every registry refusal.

    **Every** refusal derives from this. Gauntlet 0.7 defect 4 was an `A_max` read
    escaping as a bare `TypeError`, which a caller written to catch registry
    refusals would miss -- and a missed refusal is a silently defaulted constant,
    which is the one failure this module exists to prevent.
    """


class UnmeasuredConstant(ConstantError):
    """A MEASURED or DERIVED constant was read before its experiment logged it."""


class ScopeRequired(ConstantError):
    """A per-scope constant was read without a scope."""


class UnknownScope(ConstantError):
    """A per-scope constant was read with a scope it does not define."""


class ConstraintViolation(ConstantError):
    """A CONDITIONAL constant's constraint failed against the supplied context."""


# --------------------------------------------------------------------------- #
# The scope vocabulary
# --------------------------------------------------------------------------- #

SCOPES: frozenset[str] = frozenset(
    {
        "diagnostics",  # section 5.2's S = 30 debugging scope
        "synthetic",  # section 5.3's demand-controlled generator
        "corpora",  # section 5.1's shared corpus value for M
        "pg19",  # PG-19, the E3 corpus
        "pg19_e3",  # PG-19 under E3's stream schedule specifically
        "wikitext_e4",  # E4
        "e7",  # E7, its own trained model (defect C1)
        "a2_ablation",  # the A_max = M ablation
    }
)
"""The closed scope vocabulary.

Closed deliberately. Gauntlet 0.7 defect 6: E4, E7 and diagnostics had no entry in
several definitions, so `get("K", "wikitext_e4")` raised `UnknownScope` -- which
reads as "you typed it wrong" when the truth is "the spec never says." Those are
different problems with different fixes, and conflating them sends you looking for
a typo.

`tests/test_constants.py` asserts every scoped definition covers this set, so a new
scope cannot be added to one constant and forgotten in three others.
"""


@dataclass(frozen=True)
class Unspecified:
    """A scope the spec does not define for this constant. Reading it raises.

    Distinct from a missing entry (a typo) and from `None` (set by the data, not by
    the registry). `reason` is shown to the caller, so the refusal names the spec
    gap instead of implying a mistake.
    """

    reason: str


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def _a_max_le_s(value: Any, ctx: Mapping[str, Any]) -> str | None:
    """Section 3.6(a): `A_max <= S`, and `A_max` is swept only where `S` binds."""
    s = ctx["S"]
    if s is None:
        return (
            "A_max was read with S=None. Section 3.6(a) constrains A_max <= S, so "
            "S must be a concrete stream length. If S is set by the data rather "
            "than by the registry (E7), A_max is not readable there either."
        )
    if not isinstance(s, int | float) or isinstance(s, bool):
        return f"A_max was read with S={s!r}, which is not a number."
    if value > s:
        return (
            f"A_max = {value} exceeds S = {s}. Section 3.6(a): gaps longer than S do "
            f"not exist in training data, so an A_max above S is inert and produces "
            f"a flat curve that is an artifact, not a finding."
        )
    return None


def _positive_steps_per_epoch(value: Any, ctx: Mapping[str, Any]) -> str | None:
    """`T_warm` in steps needs a real steps-per-epoch (D-I)."""
    spe = ctx["steps_per_epoch"]
    if not isinstance(spe, int | float) or isinstance(spe, bool) or spe <= 0:
        return (
            f"T_warm needs steps_per_epoch > 0 to express section 3.4's one epoch "
            f"as a float number of steps; got {spe!r}. D-I: T_warm has one "
            f"representation and this is the only place the conversion happens."
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
# Derivations and conversions
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


def _t_warm_steps(epochs: float, ctx: Mapping[str, Any]) -> float:
    """D-I: section 3.4's warmup, as a float number of steps. The only conversion.

    Section 3.4 states the warmup as **one epoch**, and requires it be *reported*
    as a fraction of total epochs -- one of three is a different experiment from
    one of twelve. The dispatch compares it against the optimizer step `k` (not the
    sentence index `t`; correction 31), so the registry returns optimizer steps and
    the epoch figure stays readable as `T_warm_epochs`.
    """
    return float(epochs) * float(ctx["steps_per_epoch"])


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
    convert: Callable[[Any, Mapping[str, Any]], Any] | None = None
    """CONDITIONAL only: maps the stored value plus context to the returned value."""
    context_scope: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    """`{context_key: {this_scope: that_constant's_scope}}`.

    Gauntlet 0.7 defect 3: `A_max`'s guard trusted whatever `S` the caller passed,
    so `get("A_max", "synthetic", S=10**9)` returned 48 and validated nothing. This
    names, per scope, which scope of the *other* constant the context must agree
    with, and `Registry.get` cross-checks it.
    """


_UNSPEC_M = Unspecified(
    "Section 5.1 states M for synthetic (16), corpora (40) and e7 (8) only. "
    "A diagnostics or E4 run borrows one of those -- name the scope it borrows."
)
_UNSPEC_K = Unspecified(
    "K >= max target gap (correction 1). The spec states it for synthetic (40) and "
    "pg19 (64); any other scope needs its own target window stated first."
)
_UNSPEC_A = Unspecified(
    "Section 3.6(a) states A_max where S makes it bind: synthetic, pg19_e3, and the "
    "A2 ablation. Elsewhere A_max is not part of the design and must not be guessed."
)

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
        scoped={
            "diagnostics": _UNSPEC_M,
            "synthetic": 16,
            "corpora": 40,
            "pg19": 40,
            "pg19_e3": 40,
            "wikitext_e4": 40,
            "e7": 8,
            "a2_ablation": 40,
        },
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
            "corpora": Unspecified(
                "Section 5.2 schedules S per experiment, not per corpus. Use "
                "pg19_e3, wikitext_e4 or diagnostics."
            ),
            "pg19": Unspecified(
                "Section 5.2 schedules S per experiment. E3's PG-19 stream is "
                "pg19_e3 (80)."
            ),
            "pg19_e3": 80,
            "wikitext_e4": 30,
            "e7": None,
            "a2_ablation": 80,
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
        scoped={
            "diagnostics": _UNSPEC_K,
            "synthetic": 40,
            "corpora": _UNSPEC_K,
            "pg19": 64,
            "pg19_e3": 64,
            "wikitext_e4": _UNSPEC_K,
            "e7": _UNSPEC_K,
            "a2_ablation": 64,
        },
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
        name="T_warm_epochs",
        klass=Klass.FROZEN,
        spec_ref="3.4",
        note=(
            "Eviction warmup in EPOCHS: one epoch, per section 3.4, and **reported "
            "as a fraction of total epochs** -- one of three is a different "
            "experiment from one of twelve. This is the reportable figure; T_warm "
            "is the same quantity in steps (D-I)."
        ),
        value=1.0,
    ),
    Definition(
        name="T_warm",
        klass=Klass.CONDITIONAL,
        spec_ref="3.4",
        note=(
            "Eviction warmup as a **float number of steps** (D-I). FIFO until "
            "T_warm while psi_hat trains passively on realized r_i; also the clean "
            "on-policy/off-policy boundary for section 7.2. One representation: the "
            "registry previously held the string 'one_epoch' and RSRConfig held a "
            "float, with nothing converting between them."
        ),
        value=1.0,  # epochs; converted to steps below
        context_keys=("steps_per_epoch",),
        constraint=_positive_steps_per_epoch,
        convert=_t_warm_steps,
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
        scoped={
            "diagnostics": _UNSPEC_A,
            "synthetic": 48,
            "corpora": _UNSPEC_A,
            "pg19": _UNSPEC_A,
            "pg19_e3": 64,
            "wikitext_e4": _UNSPEC_A,
            "e7": _UNSPEC_A,
            "a2_ablation": 40,
        },
        constraint=_a_max_le_s,
        context_keys=("S",),
        context_scope={
            "S": {
                "synthetic": "synthetic",
                "pg19_e3": "pg19_e3",
                "a2_ablation": "a2_ablation",
            }
        },
    ),
)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git() -> str | None:
    """Apple's `git` refuses to run in some environments and fails silently; prefer
    Homebrew's when it is there, fall back to whatever is on PATH."""
    for candidate in ("/opt/homebrew/bin/git", "/usr/local/bin/git"):
        if Path(candidate).exists():
            return candidate
    return shutil.which("git")


def git_provenance(tree: Path | None = None) -> dict[str, Any]:
    """Stamp the sha from git **inside the measured tree** (gauntlet 0.7 defect 7).

    Never accept a sha as an argument. A caller-supplied sha records what the
    caller believed, and section 12.4's whole point is that a documented
    configuration is not evidence of what was actually run.

    `dirty` is not cosmetic: a measurement taken against an uncommitted working
    tree is not reproducible from the sha, and the ledger has to say so.
    """
    tree = tree or _REPO_ROOT
    git = _git()
    if git is None:
        return {"git_sha": None, "dirty": None, "provenance_error": "git not found"}
    try:
        sha = subprocess.run(
            [git, "-C", str(tree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [git, "-C", str(tree), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"git_sha": None, "dirty": None, "provenance_error": str(exc)}
    return {"git_sha": sha, "dirty": bool(status)}


# --------------------------------------------------------------------------- #
# Ledger + registry
# --------------------------------------------------------------------------- #

_LEDGER_ENV = "RSR_LEDGER"
_DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "measurements" / "ledger.json"


@dataclass
class Registry:
    """Constant definitions plus the measurement ledger they read from."""

    definitions: Mapping[str, Definition]
    ledger_path: Path
    tree: Path = _REPO_ROOT

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
        run_id: str,
        scope: str | None = None,
        note: str = "",
    ) -> None:
        """Log a measured value. Appends; never overwrites.

        Section 12.4: documented configuration is not evidence of what was
        actually run. Every entry carries the experiment, the commit and the run
        that produced it, so a value can always be traced to a RESULTS.md.

        **`git_sha` is not a parameter** (gauntlet 0.7 defect 7). It is stamped
        from git inside the measured tree, with the dirty flag, because a
        caller-typed sha records a belief rather than a fact.
        """
        if name not in self.definitions:
            known = ", ".join(sorted(self.definitions))
            raise ConstantError(f"{name!r} is not a registered constant. Known: {known}")
        defn = self.definitions[name]
        if defn.klass is Klass.FROZEN:
            raise ConstantError(
                f"{name!r} is FROZEN at {defn.value!r} by spec section "
                f"{defn.spec_ref}. It is not measured. If the spec changed, change "
                f"the definition and say so in docs/spec-corrections.md."
            )
        if defn.klass is Klass.CONDITIONAL:
            raise ConstantError(
                f"{name!r} is CONDITIONAL (spec section {defn.spec_ref}): it is "
                f"computed from the context it is read with, not measured. "
                f"Recording it would write a number nothing reads."
            )
        if defn.klass is Klass.DERIVED:
            # Gauntlet 0.7 defects 2 and 5. `record("gamma_b", 999.0)` used to
            # succeed and change nothing, and `K` accepted any experiment name
            # because it has no source_experiment to check against. A ledger row
            # no `get()` will ever read is worse than a refused write: it looks
            # like provenance.
            source = (
                f" It is derived from {', '.join(defn.depends_on)}."
                if defn.depends_on
                else " It is a lookup fixed by spec section " + defn.spec_ref + "."
            )
            raise ConstantError(
                f"{name!r} is DERIVED and cannot be recorded.{source} Record its "
                f"dependencies instead; `get({name!r})` computes it."
            )
        if defn.source_experiment is None:
            raise ConstantError(
                f"{name!r} is MEASURED but names no source experiment. Fix the "
                f"definition before logging against it."
            )
        if experiment != defn.source_experiment:
            raise ConstantError(
                f"{name!r} is sourced from {defn.source_experiment}, not "
                f"{experiment!r} (spec section {defn.spec_ref})."
            )
        if scope is not None and scope not in SCOPES:
            raise UnknownScope(
                f"{scope!r} is not in the scope vocabulary. Known: "
                f"{', '.join(sorted(SCOPES))}."
            )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load()
        records.append(
            {
                "name": name,
                "value": value,
                "scope": scope,
                "experiment": experiment,
                "run_id": run_id,
                "note": note,
                **git_provenance(self.tree),
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
            self._cross_check_context(defn, scope, ctx)
            if defn.convert is not None:
                return defn.convert(value, ctx)
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

    def _cross_check_context(
        self, defn: Definition, scope: str | None, ctx: Mapping[str, Any]
    ) -> None:
        """Gauntlet 0.7 defect 3: a CONDITIONAL guard must not trust its caller.

        `get("A_max", "synthetic", S=10**9)` returned 48 and validated nothing,
        because `_a_max_le_s` compared against whatever `S` arrived. Where the
        registry knows the paired scope's own value, the supplied context must
        agree with it.
        """
        for key, scope_map in defn.context_scope.items():
            if scope is None or scope not in scope_map or key not in ctx:
                continue
            expected = self.get(key, scope_map[scope])
            if expected is None:
                continue
            if ctx[key] != expected:
                raise ConstraintViolation(
                    f"{defn.name!r} in scope {scope!r} was read with {key}="
                    f"{ctx[key]!r}, but the registry holds {key}={expected!r} for "
                    f"scope {scope_map[scope]!r} (spec section "
                    f"{self.definition(key).spec_ref}). A CONDITIONAL guard that "
                    f"trusts a caller-supplied {key} validates nothing."
                )

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
        if isinstance(value, Unspecified):
            raise UnknownScope(
                f"{defn.name!r} is not specified for scope {scope!r}. "
                f"{value.reason} (spec section {defn.spec_ref})"
            )
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
        if scope is None:
            # Gauntlet 0.7 defect 1. A scope-free read used to return the most
            # recent record whatever scope it carried, so a tau measured on
            # synthetic silently became tau on PG-19 -- precisely the cross-corpus
            # transfer section 13 forbids.
            scoped = sorted({r["scope"] for r in records if r.get("scope") is not None})
            if scoped:
                raise ScopeRequired(
                    f"{defn.name!r} has measurements recorded per scope "
                    f"({', '.join(scoped)}) and was read without one. Section 13: "
                    f"no cross-corpus comparison of absolute numbers is valid, so "
                    f"the most recent row is not a safe default. Pass a scope."
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
