"""Gauntlet 1.7 -- the mutation battery.

> **A new check is not believed until a mutation has shown it red** -- and the
> mutation must redden *only* it. If nothing reddens it, **the check adds nothing
> and that is the finding.**

Each entry below breaks one thing on purpose, runs the suite, and records which
tests went red. A mutation that reddens nothing is reported as a gate that adds
nothing.

🔴 **Clause 2 is enforced here, as of cycle 0 of the 2026-09-19 run.** Until then
`off_gate` was computed, stored and printed -- and never entered the `unproven`
filter, so *"the mutation must redden only it"* was enforced by nothing: a mutation
reddening 11 unrelated tests still scored `PROVEN`. Off-gate failures now make a
mutation unproven unless they are **declared** in that mutation's
`off_gate_allowed`, each with a reason. A coupling worth knowing about is one
somebody wrote down; an undeclared one is indistinguishable from a leak.

Run:

    uv run python scripts/mutation_battery.py            # the table
    uv run python scripts/mutation_battery.py --check     # non-zero if any gate
                                                          # is unproven or leaks
    uv run python scripts/mutation_battery.py \
        --markdown docs/mutation-battery.md               # regenerate the record

`docs/mutation-battery.md` is this script's **output**, committed so a reviewer can
read the verdicts without running it. Regenerate it rather than editing it: its
table stood at "17/17, 11 off-gate" from 0a9f5d8 while the suite had grown to 19
off-gate on that same mutation, and a record that has to be retyped is a record
that goes stale.

The source tree is restored after every mutation, including on failure.
"""

from __future__ import annotations

import argparse
import json
import os
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

    off_gate_allowed: tuple[tuple[str, str], ...] = ()
    """Node ids this mutation is *expected* to redden besides its gate, each with
    a reason. Declared, not tolerated: anything red and not listed here makes the
    mutation unproven. Recorded in `docs/mutation-battery.md` under "Known
    couplings"."""


# Reasons shared by several declared couplings. Written once so the table below
# stays readable and so a reader can see that the same argument is being made.
_REDUCTION_TABLE_COUPLING = (
    "`reduction_to_tg()` is built *from* the off-switch table, so removing an "
    "entry makes every reduction test fail to construct a config. The coupling is "
    "the design: one enumeration, not two."
)
_MUP_COUPLING = (
    "the muP multipliers are checked both on the attribute and on the output, "
    "deliberately -- an attribute set correctly and never applied is precisely the "
    "silent failure §4.3 warns about, so one edit must redden both."
)
_DISPLACEMENT_COUPLING = (
    "the displacement statistic is asserted in test_instrumentation.py and in "
    "test_reduction.py because it is both a property of the metric and a property "
    "of the reduction (ADR-0006)."
)

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
        off_gate_allowed=tuple(
            (f"tests/test_reduction.py::{t}", _REDUCTION_TABLE_COUPLING)
            for t in (
                "test_a_max_tracks_capacity",
                "test_is_reduction_detects_a_single_flipped_switch",
                "test_reduction_agrees_with_fifo_on_every_eviction",
                "test_reduction_config_disables_every_added_term",
                "test_reduction_runs_through_the_score_path",
                "test_switching_psi_source_to_the_learned_head_reddens_the_reduction",
                "test_t_warm_is_zero_not_infinity",
                "test_the_learned_head_does_shift_ranks",
                "test_the_reduction_pins_the_context_source",
                "test_the_reduction_pins_the_positional_index_to_rank",
                "test_the_reduction_shifts_no_ranks",
                # 📌 Added 2026-09-18. docs/mutation-battery.md recorded 11 here
                # at 0a9f5d8; the suite has grown and the same coupling now
                # reaches 19. The count in that table was stale, not wrong.
                "test_a_learned_head_changes_the_loss",
                "test_loss_curve_is_bit_exact_against_stock_tg",
                "test_phi_construction_does_not_perturb_the_global_rng",
                "test_the_learned_head_does_evict_middle_slots",
                "test_the_reduction_displaces_no_ranks_on_the_real_model",
                "test_the_reduction_ran_through_the_score_path",
            )
        ) + tuple(
            (f"tests/test_device_placement.py::{t}", _REDUCTION_TABLE_COUPLING)
            for t in (
                "test_rsr_policy_exposes_a_to_method",
                "test_rsr_policy_runs_where_the_memory_lives[neg_age]",
            )
        ),
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
        off_gate_allowed=(
            ("tests/test_policies.py::test_lru_remembers_an_attention_event_older_"
             "than_one_step",
             "the same defect: moving LRU's recency into MemoryState.accum is what "
             "both tests assert it does not do"),
            ("tests/test_checkpoint.py::test_lru_policy_state_round_trips",
             "the round-trip asserts LRU's recency lives in the policy's own "
             "state; moving it into MemoryState.accum is exactly what that test "
             "is checking cannot happen"),
        ),
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
        # 📌 Field-shifted until 2026-09-19 cycle 0: `gate` was duplicated into
        # `path`, and `_fix_derived_entry()` patched it back at runtime. The
        # literal is fixed and the repair is gone -- both together is how a table
        # stops being readable as the thing that runs.
        "record() accepts DERIVED again",
        "test_derived_constants_cannot_be_recorded",
        "src/rsr/constants.py",
        "        if defn.klass is Klass.DERIVED:\n"
        "            # Gauntlet 0.7 defects 2 and 5.",
        "        if False:\n            # Gauntlet 0.7 defects 2 and 5.",
        "gauntlet 0.7(2): a ledger row nothing reads",
        off_gate_allowed=(
            ("tests/test_constants.py::"
             "test_K_cannot_be_recorded_against_an_arbitrary_experiment",
             "K is DERIVED, so the same refusal is what stops an arbitrary "
             "experiment recording it; one guard, two assertions"),
        ),
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
        off_gate_allowed=(
            ("tests/test_value_head.py::test_multipliers_follow_the_muP_table",
             _MUP_COUPLING),
            ("tests/test_value_head_arithmetic.py::test_W_is_not_silently_transposed",
             _MUP_COUPLING),
        ),
    ),
    Mutation(
        "W transposed",
        "test_W_is_not_silently_transposed",
        "src/rsr/retention/value_head.py",
        "        h = context @ self.W.T",
        "        h = context @ self.W",
        "gauntlet 1.4: every shape test stays green",
        off_gate_allowed=(
            ("tests/test_value_head_arithmetic.py::"
             "test_forward_matches_the_formula_computed_by_hand",
             "the hand-computed forward is the same arithmetic the transpose gate "
             "checks; one edit cannot break one and not the other"),
        ),
    ),
    Mutation(
        "linear term dropped",
        "test_both_terms_are_present",
        "src/rsr/retention/value_head.py",
        "        out = bilinear + linear",
        "        out = bilinear",
        "gauntlet 1.4: ψ̂ silently becomes purely bilinear",
        off_gate_allowed=(
            ("tests/test_value_head_arithmetic.py::"
             "test_forward_matches_the_formula_computed_by_hand", _MUP_COUPLING),
            ("tests/test_value_head_arithmetic.py::"
             "test_the_linear_multiplier_is_one_over_fan_in_in_the_output",
             _MUP_COUPLING),
        ),
    ),
    Mutation(
        "value head shares the transformer group",
        "test_the_value_head_has_its_own_group",
        "src/rsr/mup/param_groups.py",
        '"name": VALUE_HEAD_GROUP',
        '"name": "transformer.hidden"',
        "gauntlet 1.6: mis-grouping surfaces in week 9 with no error message",
        off_gate_allowed=(
            ("tests/test_param_groups.py::test_phi_is_not_also_in_a_transformer_group",
             _MUP_COUPLING),
            ("tests/test_param_groups.py::"
             "test_the_value_head_lr_follows_its_own_one_over_d_rule", _MUP_COUPLING),
        ),
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
        off_gate_allowed=(
            ("tests/test_instrumentation.py::"
             "test_evicting_a_middle_slot_displaces_the_slots_older_than_it",
             _DISPLACEMENT_COUPLING),
            ("tests/test_instrumentation.py::test_evicting_the_newest_displaces_the_most",
             _DISPLACEMENT_COUPLING),
            ("tests/test_reduction.py::test_the_reduction_shifts_no_ranks",
             _DISPLACEMENT_COUPLING),
            ("tests/test_reduction.py::"
             "test_the_reduction_displaces_no_ranks_on_the_real_model",
             _DISPLACEMENT_COUPLING + " Added 2026-09-18: the on-the-real-model "
             "variant post-dates docs/mutation-battery.md's table."),
        ),
    ),
    Mutation(
        "underfull steps are not rescaled",
        "test_underfull_steps_are_rescaled_not_masked",
        "src/rsr/retention/reward.py",
        "    return raw / total * (n_live / capacity)",
        "    return raw / total",
        "§3.2.1: the estimator learns 'stream-initial content is valuable'",
    ),
    # -- cycle 0 of the 2026-09-19 run: the evidence machinery itself ---------- #
    # Each of the four repairs in the dispatch's section 5 gets a mutation. A
    # repair with no mutation is a repair nobody has shown to be load-bearing,
    # which is the same standing the thing it replaced had.
    Mutation(
        "ledger accepts an unreproducible entry point",
        "test_command_refuses",
        "scripts/ledger.py",
        "        if why and not allow_unreproducible:",
        "        if False:",
        "5.1: the scratch-dir argv goes back in, and 9 of 13 ledgers become "
        "un-re-executable again with nothing saying so",
    ),
    Mutation(
        "any console script counts as a committed entry point",
        "test_command_refuses_an_undeclared_tool_entry_point",
        "scripts/ledger.py",
        "        if ep in TOOL_ENTRY_POINTS:",
        "        if ep is not None and not ep.endswith(\".py\"):",
        "5.1: the declared-tool allowlist stops being an allowlist and every bare "
        "command name passes",
    ),
    Mutation(
        "an unreproducible command stops capping the verdict",
        "test_the_escape_hatch_caps_the_verdict_at_inconclusive",
        "scripts/ledger.py",
        '        if unrepro and outcome != "inconclusive":',
        "        if False:",
        "5.1: a number nobody can re-run gets to report `survived`",
    ),
    Mutation(
        "a git failure reads as a clean tree again",
        "test_a_git_failure_is_not_recorded_as_a_clean_tree",
        "scripts/ledger.py",
        '        return {"git_sha": None, "dirty": None, '
        '"provenance_error": "git not found"}',
        '        return {"git_sha": "", "dirty": bool("")}',
        "5.1: `bool(\"\")` is `False` -- the exact shape that produced two false "
        "clean bills of health on this project",
    ),
    Mutation(
        "write() tolerates uncommitted source",
        "test_write_refuses_a_dirty_source_tree",
        "scripts/ledger.py",
        "        dirty = _dirty_source_paths()\n        if dirty:",
        "        dirty = _dirty_source_paths()\n        if False:",
        "5.1: a measurement against an uncommitted tree is not reproducible from "
        "its sha",
    ),
    Mutation(
        "runs/ starts gating the ledger write",
        "test_a_dirty_runs_directory_is_fine",
        "scripts/ledger.py",
        'SOURCE_ROOTS = ("src/", "scripts/", "experiments/")',
        'SOURCE_ROOTS = ("src/", "scripts/", "experiments/", "runs/")',
        "5.1: runs/ churns by design; gating on it makes the gate unusable and it "
        "gets switched off",
        off_gate_allowed=(
            ("tests/test_evidence_machinery.py::"
             "test_a_dirty_source_path_is_detected_through_porcelain",
             "SOURCE_ROOTS is one enumeration: what gates and what does not are "
             "the same list, so widening it necessarily moves both assertions"),
        ),
    ),
    Mutation(
        "status becomes optional again",
        "test_write_refuses_without_a_status",
        "scripts/ledger.py",
        '        if self.doc["status"] is None:',
        "        if False:",
        "5.1: a run that never recorded whether it finished",
    ),
    Mutation(
        "exit_code gets a default",
        "test_exit_code_is_required",
        "scripts/ledger.py",
        "    def command(self, argv: list[str] | str, *, exit_code: int | None,",
        "    def command(self, argv: list[str] | str, *, exit_code: int | None = None,",
        "5.1: seven of 13 ledgers carried `exit_code: null` by omission",
    ),
    Mutation(
        "the scoreboard resolves a cycle number",
        "test_the_scoreboard_refuses_the_prose_verdict_for_cycle_4",
        "scripts/render_scoreboard.py",
        '_CYCLE_SHAPED = re.compile(r"^cycle[-_ ]?(\\d+)$", re.IGNORECASE)',
        '_CYCLE_SHAPED = re.compile(r"^$")',
        "5.2: cycle 4 is two ledgers that disagree; collapsing them is how "
        '"3 canaries, all held" was written over 2 survived and 1 inconclusive',
    ),
    Mutation(
        "a scoreboard count is typed rather than counted",
        "test_every_scoreboard_count_is_a_len_not_a_typed_number",
        "scripts/render_scoreboard.py",
        '        "total": len(board.rows),',
        '        "total": 999,',
        "5.2: the whole prose/ledger gap in one line -- every count must be len() "
        "of something",
    ),
    Mutation(
        "the prose audit reads timestamps as measurements",
        "test_the_audit_does_not_read_a_wall_clock_time_as_a_measurement",
        "scripts/render_scoreboard.py",
        "    masked = _MONTHDAY.sub(\" \", _CLOCK.sub(\" \", _DATE.sub(\" \", text)))",
        "    masked = _DATE.sub(\" \", text)",
        "5.2: an audit that flags the wall clock is an audit nobody runs on a "
        "document that obeys §9",
    ),
    Mutation(
        "the prose audit backs every number",
        "test_the_audit_flags_a_number_that_is_in_no_ledger",
        "scripts/render_scoreboard.py",
        "        if not backed and lit not in unbacked:",
        "        if False:",
        "5.2: `1.5476` goes back to passing an audit it should fail",
        off_gate_allowed=(
            ("tests/test_evidence_machinery.py::"
             "test_the_audit_does_not_read_a_wall_clock_time_as_a_measurement",
             "the timestamp test asserts both halves of the same behaviour -- a "
             "clock is skipped AND a real number beside it is still flagged -- so "
             "an audit that flags nothing necessarily reddens it too"),
        ),
    ),
    Mutation(
        "a mutated run writes the real census file",
        "test_a_mutated_suite_run_does_not_clobber_the_census",
        "scripts/mutation_battery.py",
        '    return {**os.environ, "RSR_TEST_COUN' + 'T": SCRATCH_COUNT}',
        "    return {**os.environ}",
        "5.3: every mutation overwrites the file CI asserts on, and the last "
        "mutated run is what survives on disk",
    ),
    Mutation(
        "the board's own counts stop backing the prose",
        "test_the_rendered_artefact_passes_its_own_audit",
        "scripts/render_scoreboard.py",
        "    for row in board.rows:\n        for v in row.values():",
        "    for row in []:\n        for v in row.values():",
        "5.2: the generated artefact fails its own audit on the per-run row "
        "counts -- exactly the numbers the script exists to stop anyone typing",
    ),
    Mutation(
        "off-gate failures stop counting",
        "test_an_off_gate_failure_makes_a_mutation_unproven",
        "scripts/mutation_battery.py",
        # 📌 split so this literal is not itself the first match in this file
        '        if not r["reddened_gate"]' + " or leaked:",
        '        if not r["reddened_gate"]' + ":",
        "5.3: clause 2 of the stated discipline goes back to being enforced by "
        "nothing",
    ),
    Mutation(
        "the coupling allowlist is ignored",
        "test_a_declared_coupling_does_not_make_a_mutation_unproven",
        "scripts/mutation_battery.py",
        '        allowed = {node for node, _reason in '
        'r.get("off_gate_allo' + 'wed", ())}',
        "        allowed = set()",
        "5.3: a declared coupling has to be distinguishable from a leak, or the "
        "check is unusable and gets switched off",
    ),
    Mutation(
        "the mutation table is field-shifted again",
        "test_the_mutation_table_is_well_formed_without_a_runtime_repair",
        "scripts/mutation_battery.py",
        '        "record() accepts DERIVED again",\n'
        '        "test_derived_constants_cannot_be_recorded",\n'
        '        "src/rsr/constants.py",',
        '        "record() accepts DERIVED again",\n'
        '        "test_derived_constants_cannot_be_recorded",\n'
        '        "test_derived_constants_cannot_be_recorded",',
        "5.3: the table stops describing what runs, and a runtime repair hides it",
    ),
    Mutation(
        "a first canary reading exits 0 again",
        "test_a_first_canary_reading_exits_3_not_0",
        "scripts/canary.py",
        'EXIT_CODES = {"held": 0, "MOVED": 1, "baseline": 3}',
        'EXIT_CODES = {"held": 0, "MOVED": 1, "baseline": 0}',
        "5.4: `3` collapsing to `0` -- DID NOT RUN reported as FOUND NOTHING",
    ),
    Mutation(
        "a truncated canary run is compared over the overlap",
        "test_a_length_mismatch_is_a_move",
        "scripts/canary.py",
        "    if len(baseline) != len(losses):",
        "    if False:",
        "5.4: a run that produced 2 of 6 beats reports \"held\"",
    ),
)

def _markdown(rows: list[dict]) -> str:
    """The committed record, rendered. Every count here is `len()` of something."""
    bad = unproven(rows)
    out = [
        "# The mutation battery — gauntlet 1.7",
        "",
        "> **A new check is not believed until a mutation has shown it red** — and "
        "the",
        "> mutation must redden *only* it. If nothing reddens it, **the check adds "
        "nothing",
        "> and that is the finding.**",
        "",
        "<!-- GENERATED by `scripts/mutation_battery.py --markdown`. Do not edit: "
        "regenerate. -->",
        "",
        "**Clause 2 is enforced**, as of cycle 0 of the 2026-09-19 run. An off-gate "
        "failure makes a mutation unproven unless it is declared in that mutation's "
        "`off_gate_allowed` with a reason. Until then `off_gate` was computed, "
        "printed and never filtered on, so a mutation reddening 11 unrelated tests "
        "still scored `PROVEN`.",
        "",
        f"**{len(rows) - len(bad)}/{len(rows)} gates proven.**",
        "",
        "| Mutation | Gate it must redden | Verdict | Off-gate | Declared |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['mutation']} | `{r['gate']}` | **{r['verdict']}** | "
            f"{len(r['off_gate'])} | {len(r['off_gate_allowed'])} |"
        )
    out += ["", "## What each mutation breaks, and what else went red", ""]
    for r in rows:
        out += [f"### {r['mutation']}", "",
                f"**Gate:** `{r['gate']}` — **{r['verdict']}**", "", r["why"], ""]
        if r["off_gate"]:
            out += [f"Also reddened ({len(r['off_gate'])}):", ""]
            declared = {n: why for n, why in
                        (tuple(x) for x in r["off_gate_allowed"])}
            for f in r["off_gate"]:
                mark = "✔ declared" if f in declared else "🔴 UNDECLARED"
                out.append(f"- `{f}` — {mark}")
            out.append("")
        else:
            out += ["Reddened nothing else.", ""]
    out += [
        "## Declared couplings",
        "",
        "A coupling worth knowing about is one somebody wrote down. These are the "
        "reasons carried in the table itself, not in prose beside it:",
        "",
    ]
    seen: set[str] = set()
    for r in rows:
        for _node, why in (tuple(x) for x in r["off_gate_allowed"]):
            if why in seen:
                continue
            seen.add(why)
            out.append(f"- {why}")
    out += [
        "",
        "## Not covered here",
        "",
        "- **`tests/conftest.py`'s all-skipped tripwire** cannot be mutated from "
        "inside the suite it guards. Proved separately, by running a suite in which "
        "every test skips.",
        "- **An anchor-freshness check cannot live in this suite.** The battery runs "
        "the suite against a mutated tree, so a test asserting every `old` anchor is "
        "present reddens under every mutation by construction — it made all 32 score "
        "`LEAKS` the first time clause 2 was enforced. `apply()` raises on a stale "
        "anchor instead.",
        "",
    ]
    return "\n".join(out)


def unproven(rows: list[dict]) -> list[dict]:
    """The mutations that prove nothing.

    Two ways to fail, and **both clauses of the stated discipline count**:

    * the mutation reddened **nothing** on its gate -- the check adds nothing;
    * the mutation reddened something **off** its gate that is not declared in
      `off_gate_allowed` -- the check is not specific, so a green suite after the
      fix does not isolate the defect.

    The second clause was enforced by nothing until 2026-09-19 cycle 0.
    """
    out = []
    for r in rows:
        allowed = {node for node, _reason in r.get("off_gate_allowed", ())}
        leaked = [f for f in r.get("off_gate", []) if f not in allowed]
        if not r["reddened_gate"] or leaked:
            out.append(r)
    return out


#: Where a mutated run is allowed to write its census.
SCRATCH_COUNT = "runs/.mutation-census.json"


def _suite_env() -> dict[str, str]:
    """🔴 A mutated run must not write `test-count.json`.

    `tests/conftest.py` writes the census on every session, so each mutation
    overwrote the file CI asserts on -- and the battery's last mutated run is the
    one that survives on disk. On 2026-09-18 that file read
    `passed=315 failed=1` after a green 316-test suite, and a ledger row was
    written from it before the mismatch was noticed. The census has to come from
    the run that claims it.
    """
    return {**os.environ, "RSR_TEST_COUNT": SCRATCH_COUNT}


def run_suite() -> set[str]:
    """Return the set of failing test node ids."""
    proc = subprocess.run(
        [str(PYTEST), "-p", "no:cacheprovider", "--tb=no", "-q", "--no-header"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_suite_env(),
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
    ap.add_argument("--markdown", type=Path, default=None,
                    help="regenerate docs/mutation-battery.md from this run")
    args = ap.parse_args()

    baseline = run_suite()
    if baseline:
        raise SystemExit(f"the suite is not green before mutating: {sorted(baseline)}")

    rows = []
    for m in MUTATIONS:
        path = ROOT / m.path
        original = apply(m)
        try:
            failing = run_suite()
        finally:
            path.write_text(original)
        on_gate = sorted(f for f in failing if m.gate in f)
        off_gate = sorted(f for f in failing if m.gate not in f)
        declared = {node for node, _reason in m.off_gate_allowed}
        leaked = [f for f in off_gate if f not in declared]
        if on_gate and leaked:
            verdict = "LEAKS"
        elif on_gate:
            verdict = "PROVEN"
        else:
            verdict = "ADDS NOTHING"
        rows.append(
            {
                "mutation": m.name,
                "gate": m.gate,
                "why": m.why,
                "reddened_gate": bool(on_gate),
                "n_on_gate": len(on_gate),
                "off_gate": off_gate,
                "off_gate_allowed": [list(x) for x in m.off_gate_allowed],
                "off_gate_undeclared": leaked,
                "verdict": verdict,
            }
        )
        print(
            f"{rows[-1]['verdict']:13s} {m.name:42s} "
            f"-> {len(on_gate)} on gate, {len(off_gate)} off "
            f"({len(leaked)} undeclared)"
        )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(_markdown(rows))

    bad = unproven(rows)
    print(f"\n{len(rows) - len(bad)}/{len(rows)} gates proven by mutation")
    if bad:
        print("UNPROVEN:")
        for r in bad:
            if not r["reddened_gate"]:
                print(f"  {r['gate']}  ({r['mutation']}) -- reddens nothing")
            else:
                print(f"  {r['gate']}  ({r['mutation']}) -- reddens "
                      f"{len(r['off_gate_undeclared'])} undeclared tests off its "
                      f"gate:")
                for f in r["off_gate_undeclared"]:
                    print(f"      {f}")
        print("\nA gate whose mutation reddens nothing adds nothing. A mutation "
              "that reddens tests it did not declare has not isolated the defect. "
              "Declare the coupling with a reason, or narrow the mutation.")
    return 1 if (args.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
