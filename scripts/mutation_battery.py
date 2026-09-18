"""Gauntlet 1.7 -- the mutation battery.

> **A new check is not believed until a mutation has shown it red** -- and the
> mutation must redden *only* it. If nothing reddens it, **the check adds nothing
> and that is the finding.**

Each entry below breaks one thing on purpose, runs the suite, and records which
tests went red. A mutation that reddens nothing is reported as a gate that adds
nothing. A mutation that reddens tests outside its own gate is reported too -- that
is a coupling worth knowing about, not automatically a fault.

Run:

    .venv/bin/python scripts/mutation_battery.py            # the table
    .venv/bin/python scripts/mutation_battery.py --check    # non-zero if any gate
                                                            # is unproven

The source tree is restored after every mutation, including on failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTEST = ROOT / ".venv" / "bin" / "pytest"


@dataclass(frozen=True)
class Mutation:
    name: str
    gate: str
    """The check this mutation exists to prove. Substring-matched against node ids."""

    path: str
    old: str
    new: str
    why: str


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "t_warm back to inf",
        "test_reduction",
        "src/rsr/retention/rsr.py",
        '    "t_warm": 0.0,',
        '    "t_warm": float("inf"),',
        "gauntlet 0.1: the reduction stops reaching the score path",
    ),
    Mutation(
        "reduction uses the learned head",
        "test_reduction",
        "src/rsr/retention/rsr.py",
        '    "psi_override": "neg_age",',
        '    "psi_override": None,',
        "the §3.7 reduction is no longer argmin(-a_i)",
    ),
    Mutation(
        "drop nu from the off-switch table",
        "test_every_config_field_has_an_off_switch",
        "src/rsr/retention/rsr.py",
        '    "nu": 0.0,\n    "beta": 0.0,',
        '    "beta": 0.0,',
        "a term with no documented off-switch; the brief records this exact hole",
    ),
    Mutation(
        "RSRConfig.nu gets a default",
        "test_no_registry_owned_field_has_a_default",
        "src/rsr/retention/rsr.py",
        "    a_max: int\n",
        "    a_max: int = 16\n",
        "gauntlet 0.3: a frozen unmeasured constant, D-1's exact shape. Defaulting "
        "`nu` instead makes the dataclass itself invalid (a default before a "
        "non-default), so the module fails to import and the gate never runs -- a "
        "stronger guarantee, but not one this test can demonstrate.",
    ),
    Mutation(
        "LRU state back in MemoryState.accum",
        "test_lru_and_fifo_choose_different_slots_in_a_real_eviction_loop",
        "src/rsr/baselines/lru.py",
        "        self._last_used[winner] = step",
        "        slots.accum.setdefault('lru', {})[winner] = step",
        "gauntlet 0.4: LRU silently becomes FIFO",
    ),
    Mutation(
        "an accumulating policy skips the admission hook",
        "test_an_accumulating_policy_is_corrupted_without_the_admission_hook",
        "tests/test_policies.py",
        "        if self.use_hook:\n            self.total[slot] = 0.0",
        "        if False:\n            self.total[slot] = 0.0",
        "gauntlet 0.4 root cause, on the policy shape that needs it. NOTE: "
        "neutering LRU's own on_write reddens NOTHING -- its select_eviction takes "
        "max(written_at, last_used) and a stale record is always older than the new "
        "occupant's write step, so the hook is defensive for LRU and load-bearing "
        "for H2O. Recorded in the table rather than papered over.",
    ),
    Mutation(
        "scope-free MEASURED read leaks again",
        "test_scope_free_measured_read_does_not_leak_across_scopes",
        "src/rsr/constants.py",
        "            if scoped:",
        "            if False:",
        "gauntlet 0.7(1): §13's cross-corpus prohibition stops being enforced",
    ),
    Mutation(
        "record() accepts DERIVED again",
        "test_derived_constants_cannot_be_recorded",
        "test_derived_constants_cannot_be_recorded",
        "src/rsr/constants.py",
        "",
        "gauntlet 0.7(2): a ledger row nothing reads",
    ),
    Mutation(
        "A_max trusts the caller's S",
        "test_A_max_does_not_trust_a_caller_supplied_S",
        "src/rsr/constants.py",
        "            self._cross_check_context(defn, scope, ctx)",
        "            pass",
        "gauntlet 0.7(3): a CONDITIONAL guard that validates nothing",
    ),
    Mutation(
        "bilinear multiplier dropped",
        "test_the_bilinear_multiplier_is_one_over_d_in_the_output",
        "src/rsr/retention/value_head.py",
        "        self.bilinear_multiplier = 1.0 / d_model",
        "        self.bilinear_multiplier = 1.0",
        "gauntlet 1.6: §4.3's 1/d stored but not applied",
    ),
    Mutation(
        "W transposed",
        "test_W_is_not_silently_transposed",
        "src/rsr/retention/value_head.py",
        "        h = context @ self.W.T",
        "        h = context @ self.W",
        "gauntlet 1.4: every shape test stays green",
    ),
    Mutation(
        "linear term dropped",
        "test_both_terms_are_present",
        "src/rsr/retention/value_head.py",
        "        out = bilinear + linear",
        "        out = bilinear",
        "gauntlet 1.4: ψ̂ silently becomes purely bilinear",
    ),
    Mutation(
        "value head shares the transformer group",
        "test_the_value_head_has_its_own_group",
        "src/rsr/mup/param_groups.py",
        '"name": VALUE_HEAD_GROUP',
        '"name": "transformer.hidden"',
        "gauntlet 1.6: mis-grouping surfaces in week 9 with no error message",
    ),
    Mutation(
        "r_i drops the memory gate",
        "test_the_memory_gate_changes_the_answer",
        "src/rsr/retention/reward.py",
        "        scaled = scaled * attn.gate.reshape(-1, 1, 1, 1)",
        "        pass",
        "D-E: layer-specific rescaling silently omitted",
    ),
    Mutation(
        "r_i accepts a train-mode trace",
        "test_a_train_mode_trace_is_refused",
        "src/rsr/retention/reward.py",
        "    if not attn.eval_mode:",
        "    if False:",
        "D-F: the policy learns from dropout masks",
    ),
    Mutation(
        "rank shift counts the sliding window",
        "test_evicting_the_oldest_displaces_nothing",
        "src/rsr/retention/instrumentation.py",
        "    return victim_rank, victim_rank - 1",
        "    return victim_rank, int(((ranks > victim_rank) & (ranks > 0)).sum().item())",
        "ADR-0006: the metric measures the window, not the policy",
    ),
    Mutation(
        "underfull steps are not rescaled",
        "test_underfull_steps_are_rescaled_not_masked",
        "src/rsr/retention/reward.py",
        "    return raw / total * (n_live / capacity)",
        "    return raw / total",
        "§3.2.1: the estimator learns 'stream-initial content is valuable'",
    ),
)

# The DERIVED-record mutation needs a different shape (deleting a block), handled
# by path/old/new below rather than by the tuple above.
_DERIVED_BLOCK = (
    "src/rsr/constants.py",
    "        if defn.klass is Klass.DERIVED:\n"
    "            # Gauntlet 0.7 defects 2 and 5.",
    "        if False:\n            # Gauntlet 0.7 defects 2 and 5.",
)


def _fix_derived_entry() -> tuple[Mutation, ...]:
    out = []
    for m in MUTATIONS:
        if m.name == "record() accepts DERIVED again":
            out.append(
                Mutation(
                    m.name,
                    "test_derived_constants_cannot_be_recorded",
                    _DERIVED_BLOCK[0],
                    _DERIVED_BLOCK[1],
                    _DERIVED_BLOCK[2],
                    "gauntlet 0.7(2): a ledger row nothing reads",
                )
            )
        else:
            out.append(m)
    return tuple(out)


def run_suite() -> set[str]:
    """Return the set of failing test node ids."""
    proc = subprocess.run(
        [str(PYTEST), "-p", "no:cacheprovider", "--tb=no", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    failing = set()
    for line in (proc.stdout + proc.stderr).splitlines():
        line = line.strip()
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            failing.add(line.split(" ", 1)[1].split(" ")[0])
    if not failing and proc.returncode not in (0, 5):
        failing.add(f"<collection/exit {proc.returncode}>")
    return failing


def apply(mutation: Mutation) -> str:
    path = ROOT / mutation.path
    original = path.read_text()
    if mutation.old not in original:
        raise SystemExit(
            f"mutation {mutation.name!r}: anchor not found in {mutation.path}.\n"
            f"The battery is stale -- fix the anchor, do not drop the mutation."
        )
    path.write_text(original.replace(mutation.old, mutation.new, 1))
    return original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="exit non-zero on any unproven gate"
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    baseline = run_suite()
    if baseline:
        raise SystemExit(f"the suite is not green before mutating: {sorted(baseline)}")

    rows = []
    for m in _fix_derived_entry():
        path = ROOT / m.path
        original = apply(m)
        try:
            failing = run_suite()
        finally:
            path.write_text(original)
        on_gate = sorted(f for f in failing if m.gate in f)
        off_gate = sorted(f for f in failing if m.gate not in f)
        rows.append(
            {
                "mutation": m.name,
                "gate": m.gate,
                "why": m.why,
                "reddened_gate": bool(on_gate),
                "n_on_gate": len(on_gate),
                "off_gate": off_gate,
                "verdict": "PROVEN" if on_gate else "ADDS NOTHING",
            }
        )
        print(
            f"{rows[-1]['verdict']:13s} {m.name:42s} "
            f"-> {len(on_gate)} on gate, {len(off_gate)} off"
        )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")

    unproven = [r for r in rows if not r["reddened_gate"]]
    print(f"\n{len(rows) - len(unproven)}/{len(rows)} gates proven by mutation")
    if unproven:
        print("UNPROVEN -- these checks add nothing until a mutation reddens them:")
        for r in unproven:
            print(f"  {r['gate']}  ({r['mutation']})")
    return 1 if (args.check and unproven) else 0


if __name__ == "__main__":
    sys.exit(main())
