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

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from orchestrator import lanes  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main  # noqa: E402

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
_CAPTURE_BRIDGE_COUPLING = (
    "`r_i` is asserted twice on purpose, and S0-02 is the reason: "
    "tests/test_reward.py checks the property on a hand-built `AttentionTrace`, "
    "tests/test_capture_bridge.py checks the same property end to end on a real "
    "`TGModel` forward pass. Those were two disconnected claims until the capture "
    "bridge existed -- `reward.py` had 11 passing tests and zero callers in "
    "`src/` precisely because nothing joined them -- so one edit to `reward.py` "
    "reddening both is the join working. A `reward.py` mutation that reddened "
    "ONLY the fixture test would mean the bridge does not actually reach the "
    "reward, which is the failure this file was written to detect."
)

_S003_REFUSAL_COUPLING = (
    "S0-03: `rsr.train.loop.answer_targets` REFUSES a corpus whose answers are out "
    "of band rather than return an empty supervision mask, and `train()` calls it, "
    "so every test that trains reddens when the default corpus reverts to the "
    "pre-S0-03 one. The refusal is the design: an empty mask would make every "
    "answer-token loss a mean over nothing and pass silently."
)
_S003_CORPUS_COUPLING = (
    "S0-03: the test reads a property of the in-stream answer token itself (its "
    "position, its id under the supervision mask, the vocabulary it adds), so it "
    "cannot hold on a corpus that has no such token."
)

_DISPLACEMENT_COUPLING = (
    "the displacement statistic is asserted in test_instrumentation.py and in "
    "test_reduction.py because it is both a property of the metric and a property "
    "of the reduction (ADR-0006)."
)

_LANE_FLOCK_COUPLING = (
    "orchestrator: the flock is the only thing that makes a slot exclusive, and "
    "every one of these asserts that a held slot is held -- against a second "
    "process, a killed holder, a reservation, the status probe, a full lane, or an "
    "orphaned child. One mechanism, observed from six places."
)
_LANES_ABSENT_COUPLING = (
    "orchestrator: every CLI path loads ops/lanes.json through load_config, so the "
    "refusal naming C0 is observed through `lanes status` and `slot run` as well."
)
_SLOT_RC_COUPLING = (
    "orchestrator: a signalled child's 128+N is passed through the same verbatim "
    "exit as any other rc, so collapsing it reddens the signal tests too."
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
        )
        + tuple(
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
            (
                "tests/test_policies.py::test_lru_remembers_an_attention_event_older_"
                "than_one_step",
                "the same defect: moving LRU's recency into MemoryState.accum is what "
                "both tests assert it does not do",
            ),
            (
                "tests/test_checkpoint.py::test_lru_policy_state_round_trips",
                "the round-trip asserts LRU's recency lives in the policy's own "
                "state; moving it into MemoryState.accum is exactly what that test "
                "is checking cannot happen",
            ),
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
            (
                "tests/test_constants.py::"
                "test_K_cannot_be_recorded_against_an_arbitrary_experiment",
                "K is DERIVED, so the same refusal is what stops an arbitrary "
                "experiment recording it; one guard, two assertions",
            ),
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
            (
                "tests/test_value_head.py::test_multipliers_follow_the_muP_table",
                _MUP_COUPLING,
            ),
            (
                "tests/test_value_head_arithmetic.py::test_W_is_not_silently_transposed",
                _MUP_COUPLING,
            ),
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
            (
                "tests/test_value_head_arithmetic.py::"
                "test_forward_matches_the_formula_computed_by_hand",
                "the hand-computed forward is the same arithmetic the transpose gate "
                "checks; one edit cannot break one and not the other",
            ),
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
            (
                "tests/test_value_head_arithmetic.py::"
                "test_forward_matches_the_formula_computed_by_hand",
                _MUP_COUPLING,
            ),
            (
                "tests/test_value_head_arithmetic.py::"
                "test_the_linear_multiplier_is_one_over_fan_in_in_the_output",
                _MUP_COUPLING,
            ),
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
            (
                "tests/test_param_groups.py::test_phi_is_not_also_in_a_transformer_group",
                _MUP_COUPLING,
            ),
            (
                "tests/test_param_groups.py::"
                "test_the_value_head_lr_follows_its_own_one_over_d_rule",
                _MUP_COUPLING,
            ),
        ),
    ),
    Mutation(
        "r_i drops the memory gate",
        "test_the_memory_gate_changes_the_answer",
        "src/rsr/retention/reward.py",
        "        scaled = scaled * attn.gate.reshape(-1, 1, 1, 1)",
        "        pass",
        "D-E: layer-specific rescaling silently omitted",
        off_gate_allowed=(
            (
                "tests/test_capture_bridge.py::"
                "test_the_captured_gate_is_the_models_memory_gate",
                _CAPTURE_BRIDGE_COUPLING,
            ),
        ),
    ),
    Mutation(
        "r_i accepts a train-mode trace",
        "test_a_train_mode_trace_is_refused",
        "src/rsr/retention/reward.py",
        "    if not attn.eval_mode:",
        "    if False:",
        "D-F: the policy learns from dropout masks",
        off_gate_allowed=(
            (
                "tests/test_capture_bridge.py::"
                "test_a_train_mode_capture_is_refused_downstream",
                _CAPTURE_BRIDGE_COUPLING,
            ),
        ),
    ),
    Mutation(
        "rank shift counts the sliding window",
        "test_evicting_the_oldest_displaces_nothing",
        "src/rsr/retention/instrumentation.py",
        "    return victim_rank, victim_rank - 1",
        "    return victim_rank, int(((ranks > victim_rank) & (ranks > 0)).sum().item())",
        "ADR-0006: the metric measures the window, not the policy",
        off_gate_allowed=(
            (
                "tests/test_instrumentation.py::"
                "test_evicting_a_middle_slot_displaces_the_slots_older_than_it",
                _DISPLACEMENT_COUPLING,
            ),
            (
                "tests/test_instrumentation.py::test_evicting_the_newest_displaces_the_most",
                _DISPLACEMENT_COUPLING,
            ),
            (
                "tests/test_reduction.py::test_the_reduction_shifts_no_ranks",
                _DISPLACEMENT_COUPLING,
            ),
            (
                "tests/test_reduction.py::"
                "test_the_reduction_displaces_no_ranks_on_the_real_model",
                _DISPLACEMENT_COUPLING + " Added 2026-09-18: the on-the-real-model "
                "variant post-dates docs/mutation-battery.md's table.",
            ),
        ),
    ),
    Mutation(
        "underfull steps are not rescaled",
        "test_underfull_steps_are_rescaled_not_masked",
        "src/rsr/retention/reward.py",
        "    return raw / total * (n_live / capacity)",
        "    return raw / total",
        "§3.2.1: the estimator learns 'stream-initial content is valuable'",
        off_gate_allowed=(
            (
                "tests/test_capture_bridge.py::"
                "test_retrieval_demand_runs_on_a_real_forward_pass",
                _CAPTURE_BRIDGE_COUPLING,
            ),
        ),
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
        '        if ep is not None and not ep.endswith(".py"):',
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
        '5.1: `bool("")` is `False` -- the exact shape that produced two false '
        "clean bills of health on this project",
    ),
    Mutation(
        "write() tolerates uncommitted source",
        "test_write_refuses_a_dirty_source_tree",
        "scripts/ledger.py",
        "        dirty = _dirty_source_paths()\n        if dirty:",
        "        dirty = _dirty_source_paths()\n        if False:",
        "5.1: a measurement against an uncommitted tree is not reproducible from its sha",
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
            (
                "tests/test_evidence_machinery.py::"
                "test_a_dirty_source_path_is_detected_through_porcelain",
                "SOURCE_ROOTS is one enumeration: what gates and what does not are "
                "the same list, so widening it necessarily moves both assertions",
            ),
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
        # 📌 Re-anchored 2026-09-20 (cycle 1). The old one-line form of this
        # signature was reflowed by the Studio/trunk merge (4ece429), and because
        # `apply()` raises on a stale anchor the battery **aborted at this entry**
        # -- so every mutation after it, including the 13 that follow, had not
        # been running since that merge. The brief's revalidation checked that
        # `off_gate_allowed` existed and declared bar item 2 "executable as
        # written"; it did not run the battery.
        '        *,\n        exit_code: int | None,\n        note: str = "",',
        '        *,\n        exit_code: int | None = None,\n        note: str = "",',
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
        '    masked = _MONTHDAY.sub(" ", _CLOCK.sub(" ", _DATE.sub(" ", text)))',
        '    masked = _DATE.sub(" ", text)',
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
            (
                "tests/test_evidence_machinery.py::"
                "test_the_audit_does_not_read_a_wall_clock_time_as_a_measurement",
                "the timestamp test asserts both halves of the same behaviour -- a "
                "clock is skipped AND a real number beside it is still flagged -- so "
                "an audit that flags nothing necessarily reddens it too",
            ),
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
        "5.3: clause 2 of the stated discipline goes back to being enforced by nothing",
    ),
    Mutation(
        "the coupling allowlist is ignored",
        "test_a_declared_coupling_does_not_make_a_mutation_unproven",
        "scripts/mutation_battery.py",
        "        allowed = {node for node, _reason in "
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
        "test_a_first_canary_reading_exits_2_nothing_to_compare",
        "scripts/canary.py",
        '"baseline": Exit.UNKNOWN}',
        '"baseline": Exit.OK}',
        "5.4: the ORIGINAL defect -- a first reading, which compared nothing, "
        "reported as a pass",
    ),
    Mutation(
        "a first canary reading exits 3 again",
        "test_a_first_canary_reading_exits_2_nothing_to_compare",
        "scripts/canary.py",
        '"baseline": Exit.UNKNOWN}',
        '"baseline": Exit.DID_NOT_RUN}',
        "S0-05: the FIX's defect, on the right axis. Cycle 0 mapped a first reading "
        "to 3; it ran and had nothing to compare, which is 2. The 0-mutation above "
        "only proves the old defect stays dead -- it cannot see the value the fixer "
        "actually wrote. This one can: `2` collapsing to `3`.",
    ),
    Mutation(
        "a canary ledger row typed again",
        "test_the_canary_ledger_row_records_the_exit_code_it_returns",
        "scripts/canary.py",
        "        exit_code=int(code),",
        "        exit_code=0,",
        "S0-05 bar 6: the row was the literal `exit_code=0`, written before the "
        "verdict existed, so a MOVED canary exiting 1 filed a ledger saying 0. A "
        "hardcoded 0 is worse than null: it looks measured.",
    ),
    Mutation(
        "unimplemented experiments raise again",
        "test_an_unimplemented_experiment_exits_3",
        "src/rsr/exit_codes.py",
        '    return did_not_run(f"{experiment} is not implemented yet.")',
        '    raise NotImplementedError(f"{experiment} is not implemented yet.")',
        "S0-05 bar 5: seven e0* stubs exiting 1 (real failure) for the state that "
        "defines did-not-run. One mutation for the class: all seven go through "
        "`not_implemented()`, and all seven parametrised cases redden.",
    ),
    Mutation(
        "a checker returns a bare boolean",
        "test_no_checker_returns_a_bare_boolean",
        "scripts/render_scoreboard.py",
        "        return Exit.OK if ok else Exit.FAIL",
        "        return ok",
        "ROADMAP §6 conversion row: `True` exits 1 and `False` exits 0 -- a claim "
        "check that passed would report failure, and a bool has no did-not-run.",
    ),
    Mutation(
        "status() accepts a bool",
        "test_status_refuses_a_bool",
        "src/rsr/exit_codes.py",
        "    if isinstance(code, bool):",
        "    if False:",
        "S0-05: the runtime half of the bare-boolean rule; run_main() is the "
        "last line a bool would pass through on its way to sys.exit.",
    ),
    Mutation(
        "a stale battery anchor exits 1 again",
        "test_a_stale_battery_anchor_exits_3",
        "scripts/mutation_battery.py",
        "        refuse(\n"
        "            Exit.DID_NOT_RUN,\n"
        '            f"mutation {mutation',
        '        refuse(\n            Exit.FAIL,\n            f"mutation {mutation',
        "S0-05: nothing was mutated, so the battery did not run; a bare "
        "`raise SystemExit(msg)` reported that as 1.",
    ),
    Mutation(
        "a red baseline exits 1 again",
        "test_a_red_baseline_exits_3",
        "scripts/mutation_battery.py",
        "        refuse(\n"
        "            Exit.DID_NOT_RUN,\n"
        '            f"the suite is not green',
        '        refuse(\n            Exit.FAIL,\n            f"the suite is not green',
        "S0-05: a suite red before mutating means no mutation ran.",
    ),
    Mutation(
        "an empty battery passes",
        "test_a_battery_with_no_mutations_exits_2",
        "scripts/mutation_battery.py",
        "        return Exit.UNKNOWN\n\n    baseline = run_suite()",
        "        return Exit.OK\n\n    baseline = run_suite()",
        "S0-05: '0/0 proven' has no unproven gate in it and exited 0. Nothing to "
        "compare is 2.",
    ),
    Mutation(
        "a usage error exits 2 again",
        "test_a_usage_error_exits_3_not_2",
        "src/rsr/exit_codes.py",
        '        refuse(Exit.DID_NOT_RUN, f"{self.prog}: {message}")',
        "        raise SystemExit(2)",
        "S0-05: argparse's 2 gave render_scoreboard's `2` two meanings, bad "
        "arguments and an empty board.",
    ),
    Mutation(
        "extract_golden_tensors exits 1 without JAX again",
        "test_extract_golden_tensors_without_jax_exits_3",
        "scripts/extract_golden_tensors.py",
        '        Exit.DID_NOT_RUN,\n        f"{_exc}.',
        '        Exit.FAIL,\n        f"{_exc}.',
        "S0-05: the ImportError the project venv guarantees (ADR-0001) is "
        "did-not-run, not a failed extraction.",
    ),
    Mutation(
        "a checker bypasses run_main",
        "test_every_converted_checker_exits_through_the_protocol",
        "experiments/e0c/run.py",
        "    run_main(main)",
        "    sys.exit(main())",
        "S0-05: the enum is only the deliverable if the entry points use it; "
        "`sys.exit(main())` skips status()'s bool/None refusal.",
    ),
    Mutation(
        "a truncated canary run is compared over the overlap",
        "test_a_length_mismatch_is_a_move",
        "scripts/canary.py",
        "    if len(baseline) != len(losses):",
        "    if False:",
        '5.4: a run that produced 2 of 6 beats reports "held"',
    ),
    Mutation(
        "the training loss scores padding again",
        "test_lm_loss",
        "src/rsr/train/loop.py",
        "    per = lm_token_losses(logits, ids_t)\n"
        "    valid = mask_t[:, 1:].reshape(-1)\n"
        "    n = valid.sum()\n"
        "    return (per * valid.to(per.dtype)).sum() / n.clamp(min=1)",
        "    per = lm_token_losses(logits, ids_t)\n    return per.mean()",
        "cycle 1 defect 1: the objective goes back to averaging over every target, "
        "93.4% of which are PAD on the committed synthetic corpus -- so the number "
        "minimised, reported and exponentiated into a perplexity is mostly the "
        "model's skill at predicting zeros",
    ),
    Mutation(
        "W_O dropped from the capture path",
        "test_capture_bridge",
        "src/rsr/model/tg/policy_loop.py",
        '            wo_vs.append(torch.einsum("bmhk,hkd->bhmd", v, wo))',
        "            wo_vs.append(v.permute(0, 2, 1, 3))",
        'S0-02 bar item 3, and defect D-6. §3.2.1: *"`W_O` is not optional... '
        "dropping it reintroduces the confound the norm-weighting was adopted to "
        'remove."* The reason this needs a mutation rather than a code review is '
        "that dropping `W_O` is **shape-compatible**: `reward.contribution` norms "
        "over the last axis, and `[L, H, M, Dh]` norms just as happily as "
        "`[L, H, M, D]`. Nothing downstream raises, no shape assertion fires, and "
        "`r_i` becomes norm-weighted raw attention -- which is v0.1's rejected "
        "definition wearing the new one's name.",
    ),
    Mutation(
        "observe() is never reached",
        "test_observe",
        "src/rsr/model/tg/policy_loop.py",
        "        if observe:\n            # Pre-write memory, deliberately:",
        "        if False:\n            # Pre-write memory, deliberately:",
        "S0-02 bar item 4. `git grep '\\.observe(' -- src/` returned **zero hits** "
        "before this cycle: `reward.py` had 160 lines and 11 passing tests and no "
        "path from a forward pass to any of it. A call site with no test that "
        "notices its removal is the same condition with an extra line of code.",
    ),
    # ----------------------------------------------------------------------- #
    # S0-01 -- the training-loop defects. One mutation per fix, each reverting
    # exactly that fix and nothing else, per the brief's Bar.
    # ----------------------------------------------------------------------- #
    Mutation(
        "policy built unconditionally again",
        "test_train_does_not_stamp_a_policy_it_did_not_build",
        "src/rsr/train/loop.py",
        "    policy = build_policy(\n"
        "        policy_name, d_model=d, steps_per_epoch=float(iters), generator=gen\n"
        "    )",
        "    policy = FIFOPolicy()",
        "S0-01 defect (b), the original line. `policy_name` still flows into the "
        "frozen config and the `run_id`, so `train(policy_name='rsr')` completes "
        "and returns `run_id='rsr-d32-...'` for a stream FIFO evicted. Nothing "
        "in the run contradicts anything else in it, which is what made the "
        "defect silent and what makes the test necessary: no assertion about the "
        "loss curve could ever have caught this, because the loss curve is "
        "genuine.",
    ),
    Mutation(
        "--policy stops reaching train()",
        "test_the_policy_is_selectable_from_the_command_line",
        "src/rsr/train/loop.py",
        "        policy_name=a.policy,\n",
        "        # policy_name=a.policy,  # MUTATED\n",
        "S0-01 defect (b), CLI half. The flag still parses and still appears in "
        "`--help`; it simply does not arrive. A flag that is accepted and "
        "discarded is worse than an absent one -- the absent one is an error at "
        "the shell.",
    ),
    Mutation(
        "srep-norm hinge back out of the objective",
        "test_the_hinge",
        "src/rsr/train/loop.py",
        "        loss = (lm + w_srep * hinge) if w_srep else lm",
        "        loss = lm",
        "S0-01 defect (c). `o.srep_norm_penalty` goes back to being computed at "
        "`model.py:424` and discarded, which is the state in which "
        "`grep -c srep_norm src/rsr/train/loop.py` returned 0. The hinge is "
        "still *reported*, so this mutation also checks that reporting a term is "
        "not mistaken for optimising it.",
    ),
    Mutation(
        "ppl computed from the penalised loss",
        "test_perplexity_is_a_perplexity",
        "src/rsr/train/loop.py",
        '                    ppl=float(torch.exp(torch.tensor(last["loss_lm"]))),',
        "                    ppl=float(torch.exp(loss.detach() / steps_per_stream)),",
        "Not one of the brief's defects -- it is the defect the FIX for (c) would "
        "have introduced. `exp(loss / steps)` is a perplexity only while `loss` "
        "is the LM loss; with a regulariser in it the field keeps its name and "
        "stops being the thing the name says. Pinned so the next person to add a "
        "term to the objective is told.",
    ),
    Mutation(
        "--vocab default back to 50257",
        "test_the_cli_vocab_default_reaches_the_derived_path",
        "src/rsr/train/loop.py",
        '        "--vocab",\n        type=int,\n        default=None,',
        '        "--vocab",\n        type=int,\n        default=50257,',
        "S0-01 defect (e). 50257 is truthy, so `V = vocab if vocab else 4 + "
        "len(build_vocab(probe))` never derives from the CLI at the default and "
        "every run allocates a 50257-row embedding for a 156-word corpus. Note "
        "the claim this proves is the SMALLER one the manager corrected the "
        "brief to: unreachable *at the default*, not from the CLI -- "
        "`--vocab 0` always reached it.",
    ),
    Mutation(
        "from_registry reads every field eagerly again",
        "test_from_registry",
        "src/rsr/retention/rsr.py",
        "        kw: dict[str, Any] = {\n"
        "            k: read() for k, read in sources.items() if k not in overrides\n"
        "        }",
        "        kw: dict[str, Any] = {k: read() for k, read in sources.items()}",
        "S0-01's second 'less certain' item, which measured as real. Every "
        "registry read fires before `kw.update(overrides)` discards it, so a "
        "caller who supplied `nu` is refused for not having measured `nu`. The "
        "two arms the docstring names as the whole reason `overrides` exists -- "
        "the `gamma = 0` control and A2's `A_max = M` -- are unbuildable until "
        "E1 logs constants neither of them uses.",
    ),
    Mutation(
        "the shuffle control hands each row its own memory",
        "test_shuffle_control.py::",
        "src/rsr/metrics/memory_liveness.py",
        "    return torch.roll(torch.arange(n, device=device), 1)",
        "    return torch.arange(n, device=device)",
        "S0-04 Bar 1: an instrument that cannot read non-zero. With the identity "
        "permutation the control reads exactly 0.0 on ANY model, live or dead -- "
        "the reading §10.3's null was, and the one cycle 1 was rejected for not "
        "having ruled out. Only the live-memory decoy can see it.",
    ),
    Mutation(
        "the shuffle control never applies its permutation",
        "test_live_memory_moves_the_loss",
        "src/rsr/metrics/memory_liveness.py",
        "out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)",
        "out = model(ids_t, mask_t, kv, valid, bc, bv)",
        "the same no-op with a correct `derangement()`: the replay reaches the "
        "forward un-permuted. The derangement test stays green, so only the "
        "live-memory reading catches it.",
    ),
    Mutation(
        "the shuffle replay perturbs the memory it replays",
        "test_shuffle_control.py::",
        "src/rsr/metrics/memory_liveness.py",
        "out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)",
        "out = model(ids_t, mask_t, kv[perm] + 1e-3, valid[perm], bc, bv)",
        "the replay hands over memory that is not the memory the reference pass "
        "read: a 1e-3 offset on every slot. The derangement is still correct and "
        "the live-memory decoy still moves, so neither of the other two entries "
        "sees it -- only a replay of each row's OWN memory, which must read "
        "exactly 0.0, can tell a faithful replay from a perturbed one.",
    ),
    Mutation(
        "the shuffle replay hands over the bos gestalt too",
        "test_shuffle_control.py::",
        "src/rsr/metrics/memory_liveness.py",
        "out = model(ids_t, mask_t, kv[perm], valid[perm], bc, bv)",
        "out = model(ids_t, mask_t, kv[perm], valid[perm], bc[perm], bv[perm])",
        "the control permutes the bos-copy path along with the memory, so its "
        "delta measures memory PLUS the bos gestalt rather than memory alone. With "
        "memory disabled the kv swap is a no-op but the bos swap is not, so the "
        "disabled-memory reading stops being exactly 0.0.",
    ),
    # -- decisive run (experiments/decisive-shuffle/PREREG.md, Mutation bar) -- #
    Mutation(
        "decisive: two arms swapped in the script",
        "test_arms_are_the_preregistered_table",
        "experiments/decisive-shuffle/run.py",
        '    "A": {"masked_loss": False, "srep_norm_reg_weight": 0.0},\n'
        '    "B": {"masked_loss": True, "srep_norm_reg_weight": 0.0},',
        '    "A": {"masked_loss": True, "srep_norm_reg_weight": 0.0},\n'
        '    "B": {"masked_loss": False, "srep_norm_reg_weight": 0.0},',
        "PREREG Mutation bar, arm swap: arm A would train B's objective and be "
        "reported as 'the corpus alone'. The manifest is frozen from ARMS, so it "
        "would agree with the swap; only the hand-copied PREREG table sees it.",
    ),
    Mutation(
        "decisive: the manifest-hash arm check never refuses",
        "test_an_arm_swap_is_refused_by_the_manifest_check",
        "experiments/decisive-shuffle/run.py",
        "    if why:\n        raise ArmMismatch(",
        "    if False:\n        raise ArmMismatch(",
        "PREREG Mutation bar, arm swap: a checkpoint trained under another arm's "
        "config would be measured and reported as this arm's.",
        off_gate_allowed=(
            (
                "tests/test_decisive_shuffle.py::"
                "test_a_config_that_does_not_hash_to_its_own_stamp_is_refused",
                "the same refusal statement guards both the arm hash and the "
                "config's own stamp; disabling it must redden both, by design.",
            ),
        ),
    ),
    Mutation(
        "decisive: the aliasing clause dropped from the decision rule",
        "test_decisive_shuffle.py::",
        "experiments/decisive-shuffle/run.py",
        "    if any(x == 1.0 for x in r.values()):",
        "    if False:",
        "PREREG Mutation bar, decoy aliasing: with the decoy pointed at the trained "
        "checkpoint ratio == 1.0 >= 0.1, and the rule would call the arm 'live' "
        "on a reading that compares the model with itself.",
    ),
    Mutation(
        "decisive: measure() ignores the decoy checkpoint it is handed",
        "test_the_decoy_pointed_at_the_trained_checkpoint",
        "experiments/decisive-shuffle/run.py",
        "        ck.load(decoy_ckpt, model=decoy, restore_rng=False)",
        "        pass",
        "the aliasing test must exercise the decoy-loading path end to end; if "
        "measure() silently kept the untrained decoy, the aliasing mutation "
        "could never be run against the real code.",
    ),
    Mutation(
        "random replacement with self perturbs the memory",
        "test_random_replacement_with_self_reads_exactly_zero",
        "src/rsr/metrics/memory_liveness.py",
        "            return kv.clone()",
        "            return kv.clone() + 1e-3",
        "PREREG Secondary 2: the replacement path's own control. A path that moves "
        "tokens by itself would read as 'memory is read' on any model.",
    ),
    Mutation(
        "random replacement is not norm-matched",
        "test_random_replacement_preserves_each_slots_norm",
        "src/rsr/metrics/memory_liveness.py",
        "        return r / norm_r * kv.norm(dim=-1, keepdim=True)",
        "        return r",
        "PREREG Secondary 2 requires the Gaussian rescaled to the replaced slot's "
        "L2 norm; an unscaled draw changes magnitude as well as content, and the "
        "live-memory reading still moves, so only the norm check sees it.",
    ),
    Mutation(
        "the answer-token mask is ignored",
        "test_token_mask_on_padding_raises_and_default_adds_no_key",
        "src/rsr/metrics/memory_liveness.py",
        "                        sel.setdefault(name, []).append(sel_full[real])",
        "                        sel.setdefault(name, []).append(real[real])",
        "PREREG Secondary 1: the answer-token split would silently score every "
        "real target. A full mask reads the same either way; only an empty mask "
        "tells them apart.",
    ),
    Mutation(
        "cross-row cosine reports a constant",
        "test_cross_row_cosine_is_one_for_identical_rows_and_bounded_otherwise",
        "src/rsr/metrics/memory_liveness.py",
        "            vals.append(float(c[off].mean()))",
        "            vals.append(1.0)",
        "the descriptive cosine would read 'rows collinear' whatever the memory "
        "holds, which is exactly the rival hypothesis it exists to test.",
    ),
    Mutation(
        "the oracle evicts the sentence most needed",
        "test_oracle.py::",
        "src/rsr/baselines/oracle.py",
        "            v = self._future(int(slots.written_at[k]), step)",
        "            v = -self._future(int(slots.written_at[k]), step)",
        "E-feas reads oracle - FIFO as an upper bound on what retention can buy. An "
        "oracle that is not optimal makes that bound a lower number than the truth, "
        "and 'oracle ~= FIFO' would then be a finding about the oracle.",
    ),
    Mutation(
        "checkpoints written straight to the final path",
        "test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint",
        "src/rsr/train/checkpoint.py",
        '    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")',
        "    tmp = path",
        "gauntlet 3.6: the non-atomic write the SIGKILL test exists to catch. The "
        "child is killed while parked in `checkpoint._write_hook` (synchronised over "
        "a pipe, no wall-clock delay), so it reddens [mid_write] (torn file) and "
        "[before_replace] (old checkpoint replaced pre-rename) on every run. Until "
        "Brief 0b (2026-09-21) it was a kill-delay sweep that caught this in 1 of 7 "
        "battery observations.",
    ),
    Mutation(
        "the synthetic answer goes back out of band",
        "test_every_query_carries_its_answer_as_its_final_token",
        "src/rsr/data/synthetic.py",
        "    answer_in_stream: bool = True",
        "    answer_in_stream: bool = False",
        "S0-03's fixture mutation: the default corpus reverts to the pre-S0-03 "
        "generator, whose answer lived only in `Sentence.answer` and never entered "
        "the token stream -- so no next-token target required retrieval and 'the "
        "memory is inert' was a finding about the corpus. The named gate must "
        "redden; the rest are the declared consequences of the refusal.",
        off_gate_allowed=(
            (
                "tests/test_synthetic.py::"
                "test_the_answer_is_the_only_difference_between_the_two_corpora",
                _S003_CORPUS_COUPLING,
            ),
            (
                "tests/test_synthetic.py::"
                "test_the_target_mask_marks_exactly_the_answer_token_of_every_query",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_synthetic.py::"
                "test_the_gap_tensor_is_the_pairs_gap_at_each_query",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_synthetic.py::"
                "test_the_answer_targets_are_a_minority_that_a_pooled_loss_would_hide",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_train_loop.py::test_the_hinge_reaches_the_objective",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_train_loop.py::test_the_hinge_has_a_documented_off_switch",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_train_loop.py::"
                "test_perplexity_is_a_perplexity_and_not_a_penalised_loss",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_train_loop.py::test_the_hinge_is_on_by_default",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_decisive_shuffle.py::"
                "test_the_decoy_pointed_at_the_trained_checkpoint_reads_ratio_one_"
                "and_inconclusive",
                _S003_REFUSAL_COUPLING,
            ),
            (
                "tests/test_train_loop.py::test_the_corpus_holds_156_unique_words",
                _S003_CORPUS_COUPLING,
            ),
            (
                "tests/test_train_loop.py::"
                "test_the_derived_vocabulary_is_what_the_model_is_built_with",
                _S003_REFUSAL_COUPLING,
            ),
        ),
    ),
    # --- orchestrator: workqueue ---
    Mutation(
        "the work queue offers owner items",
        "test_owner_item_",
        "scripts/orchestrator/workqueue.py",
        '            return Readiness(False, ["owner decision: never dispatched"])',
        "            pass",
        "an owner decision (RESEARCH-CONTEXT §12, sprint gates) becomes dispatchable "
        "work: the queue would hand an agent a decision only Brendan may make",
    ),
    Mutation(
        "a PREREG co-committed with code satisfies the queue",
        "test_prereg_cocommitted_",
        "scripts/orchestrator/workqueue.py",
        "        if touched != {path}:",
        "        if path not in touched:",
        "CLAUDE.md: pre-registration commits land in their own commit, ahead of the "
        "experiment; a threshold committed alongside its experiment's code is not "
        "evidence it came first",
    ),
    Mutation(
        "the work queue's §16 sprint cap moves to 5",
        "test_sprint_cap_",
        "scripts/orchestrator/workqueue.py",
        "MAX_SPRINT = 4\n",
        "MAX_SPRINT = 5\n",
        "§16 approves weeks 1-4 only; a sprint-5 item would validate and be scheduled",
    ),
    Mutation(
        "an unsigned ruling satisfies the queue",
        "test_ruling_unsigned_",
        "scripts/orchestrator/workqueue.py",
        '        for k in ("date", "stated_in"):\n            if not meta.get(k):',
        "        for k in ():\n            if not meta.get(k):",
        "a ruling file with no date or stated_in (docs/owner/rulings/README.md) "
        "unblocks work as if Brendan had stated it",
    ),
    Mutation(
        "a blocked item can be claimed",
        "test_claim_of_",
        "scripts/orchestrator/workqueue.py",
        '    if args.status == "claimed":',
        "    if False:",
        "set-status is the only writer; without the check a caller claims an item "
        "the deterministic rule says is not ready, and the LLM decides readiness",
    ),
    # --- orchestrator: lanes ---
    Mutation(
        "lane flock becomes a no-op",
        "test_processes_cannot_hold_more_than_N_slots",
        "scripts/orchestrator/lanes.py",
        "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n",
        "        pass\n",
        "the semaphore stops counting: every process is admitted, so nine 1-thread "
        "CPU jobs become as many as ask, on 16 cores",
        off_gate_allowed=tuple(
            (node, _LANE_FLOCK_COUPLING)
            for node in (
                "tests/test_orch_lanes.py::test_a_killed_holder_releases_its_slots",
                "tests/test_orch_lanes.py::test_a_multi_slot_admission_is_all_or_nothing",
                "tests/test_orch_lanes.py::test_battery_reserves_cpu_det_slots",
                "tests/test_orch_lanes.py::test_status_json_names_the_holder",
                "tests/test_orch_slot.py::test_a_full_lane_refuses_3_after_the_wait",
                "tests/test_orch_slot.py::"
                "test_slots_stay_held_while_an_orphaned_child_lives",
            )
        ),
    ),
    Mutation(
        "absent ops/lanes.json falls back to typed defaults",
        "test_missing_lanes_file_is_refused_naming_C0",
        "scripts/orchestrator/lanes.py",
        "        raise Refused(_absent_message(path))",
        "        return LanesConfig(9, 2, 2, 8.0, 1)",
        "capacities stop being MEASURED by C0: a hand-typed default is D-1's shape "
        "applied to the scheduler",
        off_gate_allowed=(
            (
                "tests/test_orch_lanes.py::test_status_cli_without_lanes_file_exits_3",
                _LANES_ABSENT_COUPLING,
            ),
            (
                "tests/test_orch_slot.py::"
                "test_missing_lanes_file_refuses_3_and_does_not_run",
                _LANES_ABSENT_COUPLING,
            ),
        ),
    ),
    Mutation(
        "mps admission skips the memory check",
        "test_mps_admission_refuses_when_free_memory_is_insufficient",
        "scripts/orchestrator/lanes.py",
        "            if free_gb - peak_gb >= cfg.reserve_gb:",
        "            if True:",
        "an MPS job is admitted into unified memory it does not fit, and swaps the "
        "CPU lanes it shares 64 GB with",
    ),
    # --- orchestrator: slot ---
    Mutation(
        "slot collapses the child's rc to a bool",
        "test_the_childs_rc_passes_through_verbatim",
        "scripts/orchestrator/slot.py",
        "    raise SystemExit(rc)",
        "    raise SystemExit(bool(rc))",
        "2/3/5/137 all become 1: 'nothing to compare', 'did not run' and 'killed' "
        "collapse into 'real failure'",
        off_gate_allowed=(
            (
                "tests/test_orch_slot.py::test_a_child_killed_by_a_signal_exits_128_plus_n",
                _SLOT_RC_COUPLING,
            ),
            (
                "tests/test_orch_slot.py::test_sigterm_is_forwarded_and_recorded",
                _SLOT_RC_COUPLING,
            ),
        ),
    ),
    Mutation(
        "slot does not force the thread env",
        "test_cpu_det_forces_exactly_slots_threads",
        "scripts/orchestrator/slot.py",
        "        env.update(lanes.thread_env(threads))",
        "        pass",
        "a 1-slot cpu-det job spawns a BLAS/OpenMP pool per core; the slot count "
        "stops meaning cores",
    ),
    Mutation(
        "slot does not pass its lock fds to the child",
        "test_slots_stay_held_while_an_orphaned_child_lives",
        "scripts/orchestrator/slot.py",
        "pass_fds=lease.fds()",
        "pass_fds=()",
        "a SIGKILLed wrapper frees the slots of a job that is still running",
    ),
    # --- orchestrator: battery threads ---
    Mutation(
        "battery suite threads not capped",
        "test_the_env_override_caps_every_thread_pool",
        "scripts/mutation_battery.py",
        # Multi-line on purpose: the one-line form also occurs in THIS table (the
        # slot entry above), and apply() replaces the first occurrence.
        "    if threads is not None:\n        env.update(lanes.thread_env(threads))\n",
        "    if threads is not None:\n        pass\n",
        "the battery's pytest ignores RSR_BATTERY_THREADS and battery_cpu_slots and "
        "takes every core from the cpu-det lane",
        off_gate_allowed=(
            (
                "tests/test_orch_battery.py::test_lanes_file_supplies_battery_cpu_slots",
                "both sources of the cap reach the suite through the one "
                "env.update; one edit, both sources lost",
            ),
        ),
    ),
    # --- orchestrator: lint_brief ---
    # Brief errors are the dominant failure (overnight-2026-09-21.md §6). Each
    # mutation removes one of lint_brief's checks; the test that plants that class
    # of brief error must redden, and nothing else.
    Mutation(
        "lint_brief: a drifted anchor passes",
        "test_a_drifted_anchor_is_a_finding",
        "scripts/orchestrator/lint_brief.py",
        "        if actual is not None and expect in actual:\n",
        "        if True:\n",
        "The rsr.py:433-vs-:448 class. Accepting any line as the anchor means a "
        "brief citing a line that no longer holds its text lints clean.",
    ),
    Mutation(
        "lint_brief: a writer premise is run",
        "test_a_writer_command_is_rejected_and_not_run",
        "scripts/orchestrator/lint_brief.py",
        "    for pat, label in WRITER_PATTERNS:\n",
        "    for pat, label in ():\n",
        "A premise is a READ of the base. With no writer filter, a redirect, `rm`, "
        "a commit and a push are executed rather than rejected.",
    ),
    Mutation(
        "lint_brief: a wrong baseline_sha passes",
        "test_a_wrong_baseline_sha_is_a_finding",
        "scripts/orchestrator/lint_brief.py",
        "    if sha != base:\n",
        "    if False:\n",
        "The baseline off by the brief's own commit (Brief 0, S0-03, 2026-09-21) "
        "lints clean.",
    ),
    Mutation(
        "lint_brief: no base exits 0",
        "test_no_base_exits_3",
        "scripts/orchestrator/lint_brief.py",
        "        return did_not_run(str(e))\n",
        "        return Exit.OK\n",
        "3 collapsing to 0: a lint that never ran -- no base to check against -- "
        "reported as a clean brief.",
        off_gate_allowed=(
            (
                "tests/test_orch_lint_brief.py::"
                "test_the_cli_exits_through_the_protocol_with_json",
                "the CLI test asserts the same refusal end to end through "
                "`python -m`, so one edit to the refusal reddens both: the unit "
                "and the process exit status.",
            ),
        ),
    ),
    Mutation(
        "lint_brief: the throwaway worktree is kept",
        "test_premises_run_in_a_throwaway_worktree_at_base",
        "scripts/orchestrator/lint_brief.py",
        '        removed = _git(self.root, "worktree", "remove", "--force", '
        "str(self.path))\n",
        "        removed = subprocess.CompletedProcess([], 0)\n",
        "Premise checks run in a detached worktree at base; if it is not removed "
        "every lint leaves a registered worktree behind in the shared repository.",
    ),
    # --- orchestrator: loop driver (reconcile, merge, tick, notify) --- #
    # Agent D, 2026-09-22. Each gate is a test in tests/test_orch_*.py run
    # against a throwaway repo with stubbed claude/gh (tests/_orch_loop_helpers.py).
    Mutation(
        "reconcile: a repeated cause no longer parks",
        "test_two_failures_with_the_same_cause_park_the_item",
        "scripts/orchestrator/reconcile.py",
        "        repeat = cause in hist\n",
        "        repeat = False\n",
        "the stop rule 'two failures with the same cause' (dispatch-2026-09-21-"
        "overnight) stops being a mechanism: the item goes back to ready and the "
        "same failure is re-dispatched until RSR_MAX_ATTEMPTS.",
    ),
    Mutation(
        "reconcile: a dead job pid is read as alive",
        "test_a_running_job_whose_pid_is_dead_becomes_crashed_and_its_item_collecting",
        "scripts/orchestrator/reconcile.py",
        '                if lc.pid_alive(rec.get("pid")):\n'
        "                    continue\n"
        '                rec["status"] = "crashed"',
        "                if True:\n"
        "                    continue\n"
        '                rec["status"] = "crashed"',
        "a job whose process died stays `running` forever: no collector is ever "
        "spawned and the night reads as busy until park_by.",
    ),
    Mutation(
        "merge: preregistration/ dropped from the frozen globs",
        "test_the_guard_trips_on_a_preregistration_edit",
        "scripts/orchestrator/merge.py",
        '    "preregistration/*",\n',
        '    # "preregistration/*",\n',
        "an unattended branch that edits a signed threshold merges into the night "
        "branch -- 'a threshold registered after seeing the data is not a threshold'.",
    ),
    Mutation(
        "merge: a changed FROZEN definition is not compared",
        "test_the_guard_trips_on_a_frozen_constant_edit",
        "scripts/orchestrator/merge.py",
        "                elif before[name] != after[name]:",
        "                elif False:",
        "defect D-1's shape: a FROZEN constant's value edited on a run branch passes "
        "the guard, because only additions and removals are checked.",
    ),
    Mutation(
        "merge: no verification record is not a refusal",
        "test_merge_is_refused_without_a_verification_record",
        "scripts/orchestrator/merge.py",
        "    if rec is None:\n        return Exit.DID_NOT_RUN,",
        "    if rec is None and False:\n        return Exit.DID_NOT_RUN,",
        "the verification gate stops being a gate: with no record the merge crashes "
        "(1) instead of refusing (3) -- 'did not run' collapsing into 'real failure'.",
    ),
    Mutation(
        "tick: HALT ignored",
        "test_halt_is_respected_before_anything_else",
        "scripts/orchestrator/tick.py",
        "        h = lc.halted(self.root)\n        if h:",
        "        h = lc.halted(self.root)\n        if False:",
        "the owner's stop switch, and the loop's own (guard trip, spend cap), no "
        "longer stop the python pass; only tick.zsh's first line still would.",
    ),
    Mutation(
        "tick: the idle path falls through to launching",
        "test_the_idle_path_launches_nothing_and_notifies_once",
        "scripts/orchestrator/tick.py",
        "            self.idle()\n            return Exit.OK",
        "            self.idle()",
        "an idle night starts a paid manager cycle every ten minutes with nothing "
        "to review.",
    ),
    Mutation(
        "tick: UNSET spend caps are not refused",
        "test_caps_unset_refuse_3_and_notify_once",
        "scripts/orchestrator/tick.py",
        "        if cycle_cap is None or night_cap is None:",
        "        if False:",
        "the spend caps are an owner decision; without the refusal the loop runs "
        "with no cap at all (here it crashes on the None cap, exit 1 not 3).",
        off_gate_allowed=(
            (
                "tests/test_orch_tick.py::"
                "test_tick_zsh_reports_the_pass_status_unborrowed",
                "the zsh wrapper's test drives the same UNSET refusal through "
                "tick.zsh to prove the status is reported unborrowed; it cannot "
                "hold when the refusal it reports is gone. Skipped where zsh is "
                "absent (CI), so it may or may not redden there.",
            ),
        ),
    ),
    Mutation(
        "notify: the content-hash dedupe removed",
        "test_the_same_content_is_posted_once",
        "scripts/orchestrator/notify.py",
        "    if h in seen:",
        "    if False:",
        "the owner is posted the same notice on every tick "
        "(R-2026-09-22-owner-out-of-loop).",
        off_gate_allowed=(
            (
                "tests/test_orch_tick.py::test_caps_unset_refuse_3_and_notify_once",
                "the tick's refusal notice is posted once BECAUSE notify dedupes "
                "it -- the tick deliberately has no second guard for a refusal.",
            ),
        ),
    ),
)


def _markdown(rows: list[dict]) -> str:
    """The committed record, rendered. Every count here is `len()` of something."""
    bad = unproven(rows)
    out = [
        "# The mutation battery — gauntlet 1.7",
        "",
        "> **A new check is not believed until a mutation has shown it red** — and the",
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
        out += [
            f"### {r['mutation']}",
            "",
            f"**Gate:** `{r['gate']}` — **{r['verdict']}**",
            "",
            r["why"],
            "",
        ]
        if r["off_gate"]:
            out += [f"Also reddened ({len(r['off_gate'])}):", ""]
            declared = {n: why for n, why in (tuple(x) for x in r["off_gate_allowed"])}
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
    threads, _source = suite_threads()
    env = {**os.environ, "RSR_TEST_COUNT": SCRATCH_COUNT}
    if threads is not None:
        env.update(lanes.thread_env(threads))
    return env


def suite_threads() -> tuple[int | None, str]:
    """How many threads the mutated suite may use, and where that number came from.

    The battery runs beside the lane scheduler's other jobs on one Mac Studio
    (ADR-0007), and an uncapped pytest spawns a BLAS/OpenMP pool per core. So:

    1. ``RSR_BATTERY_THREADS`` if set (a positive int, else DID NOT RUN);
    2. else ``battery_cpu_slots`` from ``ops/lanes.json`` -- the cpu-det slots the
       battery lane reserves (`scripts/orchestrator/lanes.py`) -- when that file
       exists; a present but invalid file is DID NOT RUN, never ignored;
    3. else ``None``: the environment is left as it was, which is the behaviour
       before the scheduler existed. ``ops/lanes.json`` is absent until capacity
       experiment C0 runs.

    `main()` prints which, so a battery record says what it ran under.
    """
    raw = os.environ.get("RSR_BATTERY_THREADS")
    if raw is not None:
        try:
            n = int(raw)
        except ValueError:
            n = 0
        if n < 1:
            refuse(Exit.DID_NOT_RUN, f"RSR_BATTERY_THREADS={raw!r} is not an int >= 1")
        return n, "RSR_BATTERY_THREADS"
    if not (ROOT / lanes.LANES_FILE).exists():
        return None, f"unset ({lanes.LANES_FILE} absent; environment unchanged)"
    try:
        cfg = lanes.load_config(ROOT)
    except lanes.Refused as e:
        refuse(Exit.DID_NOT_RUN, str(e))
    return max(cfg.battery_cpu_slots, 1), f"{lanes.LANES_FILE} battery_cpu_slots"


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
        # 🔴 DID NOT RUN (3), not 1: nothing was mutated, so nothing was tested.
        # A bare `raise SystemExit("...")` exits 1 -- a stale anchor reported as a
        # failed gate (S0-05).
        refuse(
            Exit.DID_NOT_RUN,
            f"mutation {mutation.name!r}: anchor not found in {mutation.path}.\n"
            f"The battery is stale -- fix the anchor, do not drop the mutation.",
        )
    path.write_text(original.replace(mutation.old, mutation.new, 1))
    return original


def main() -> Exit:
    ap = ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="exit non-zero on any unproven gate"
    )
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="regenerate docs/mutation-battery.md from this run",
    )
    args = ap.parse_args()

    if not MUTATIONS:
        # Zero mutations is NOTHING TO COMPARE (2), not a pass: "0/0 proven" has no
        # unproven gate in it and would otherwise exit 0.
        print("UNKNOWN: the battery has no mutations to run", file=sys.stderr)
        return Exit.UNKNOWN

    threads, source = suite_threads()
    print(
        f"suite threads: {threads if threads is not None else 'uncapped'} "
        f"(source: {source})"
    )
    baseline = run_suite()
    if baseline:
        # DID NOT RUN (3): nothing was mutated. Used to exit 1 via a bare
        # `raise SystemExit(msg)` (S0-05).
        refuse(
            Exit.DID_NOT_RUN,
            f"the suite is not green before mutating: {sorted(baseline)}",
        )

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
                print(
                    f"  {r['gate']}  ({r['mutation']}) -- reddens "
                    f"{len(r['off_gate_undeclared'])} undeclared tests off its "
                    f"gate:"
                )
                for f in r["off_gate_undeclared"]:
                    print(f"      {f}")
        print(
            "\nA gate whose mutation reddens nothing adds nothing. A mutation "
            "that reddens tests it did not declare has not isolated the defect. "
            "Declare the coupling with a reason, or narrow the mutation."
        )
    # Without --check this is the table renderer and exits 0 once the table is
    # rendered; --check is the gate. Documented in the module docstring, and
    # decided in S0-05's RESULTS.md.
    return Exit.FAIL if (args.check and bad) else Exit.OK


if __name__ == "__main__":
    run_main(main)
