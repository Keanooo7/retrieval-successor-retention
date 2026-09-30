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

🔴 **The tree you run this from is never mutated** (I1, 2026-09-29). Every suite
run -- the baseline and each mutation -- happens in a shard: a git worktree of the
invoking tree's HEAD (which must be clean) with its own venv, outside the invoking
tree (`scripts/battery_isolation.py`). A suite whose `rsr` does not resolve under
its shard, in the pytest process or in a child of it, is `DID_NOT_RUN` -- never
`PROVEN` or `LEAKS`. After a SIGKILL, which no handler sees, the only leftovers
are shard worktrees:

    uv run python scripts/mutation_battery.py --prune-shards
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import battery_isolation as iso  # noqa: E402
from orchestrator import lanes  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status  # noqa: E402

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
_REVIEW_GATE_COUPLING = (
    "every T5(b) refusal is returned by the one review_gate() call; skipping the "
    "call turns each of them into a merge. The coupling is the design: one gate."
)
_REVIEW_GATE_REFUSALS: tuple[str, ...] = (
    "test_a_record_for_a_different_head_is_refused",
    "test_a_record_for_another_branch_is_refused",
    "test_a_record_older_than_head_with_a_code_change_since_is_refused",
    "test_a_record_whose_filename_is_not_its_reviewed_head_is_refused",
    "test_a_reviewer_who_is_a_git_author_of_the_branch_is_refused",
    "test_a_self_review_is_refused",
    "test_an_unknown_verdict_is_refused",
    "test_do_not_merge_is_refused_as_a_failure",
    "test_merge_with_fixes_verified_at_not_an_ancestor_is_refused",
    "test_merge_with_fixes_with_code_after_the_fix_check_is_refused",
    "test_merge_with_fixes_without_fixes_verified_at_is_refused",
    "test_other_prefixes_are_refused_without_a_review[docs/x]",
    "test_other_prefixes_are_refused_without_a_review[eng/x]",
    # added with the review fixes (MAJOR-1, MAJOR-2, MINOR-2), 2026-09-30
    "test_merge_with_fixes_is_never_mergeable_on_its_own",
    "test_fixes_verified_at_must_be_a_full_sha_of_a_real_commit[run/a]",
    "test_fixes_verified_at_must_be_a_full_sha_of_a_real_commit[HEAD]",
    "test_fixes_verified_at_must_be_a_full_sha_of_a_real_commit[" + "0" * 40 + "]",
    "test_only_the_merging_runs_own_verification_is_exempt",
    "test_no_verification_json_is_exempt_on_other_prefixes",
)
_REDUCTION_TABLE_COUPLING = (
    "`reduction_to_tg()` is built *from* the off-switch table, so removing an "
    "entry makes every reduction test fail to construct a config. The coupling is "
    "the design: one enumeration, not two."
)
_REDUCTION_WARMUP_LOUD_COUPLING = (
    "since correction 31 a policy with T_warm > 0 and no optimizer step set raises "
    "instead of evicting FIFO. Under `t_warm = inf` every caller of the reduction "
    "that sets no step -- the accelerator placement test, the reduction's own "
    "no-counter test -- now fails loudly: the gauntlet 0.1 defect surfacing as an "
    "error rather than as a silently FIFO reduction. Found by the 2026-09-26 "
    "review (the raise confirmed in-process); the full battery is the measurement."
)
_BUILD_POLICY_INJECTION_COUPLING = (
    "correction 31's train-driven tests put their RSRPolicy into the real train() "
    "by patching build_policy -- the one path S0-01 made the only path. A train() "
    "that builds FIFOPolicy unconditionally never consults it, so those tests see "
    "zero RSR evictions: the same defect, observed from the warmup side."
)
_T_WARM_COUPLING = (
    "correction 31 states one defect three ways -- below S (a FIFO prefix), above "
    "S (FIFO forever), and the invariant that warm never flips inside one stream. "
    "Restoring the sentence-index counter or dropping the loop's hand-off breaks "
    "all of them at once, by design."
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
_FRESH_STREAM_S003_COUPLING = (
    "fresh-stream trains and hashes on the S0-03 corpus: its default-path bit "
    "identity, its init and resume exactness, and its vocabulary-closure preflight "
    "all build documents through the generator this mutation reverts. They fail "
    "for the corpus change, not for a defect of their own."
)
_FRESH_STREAM_AUDIT_COUPLING = (
    "fresh-stream's RESULTS renderer backs each small integer by naming its key in "
    "the same sentence -- the rule this mutation removes -- so its failed-control "
    "page fails the audit: the same rule seen from fresh-stream, as from C0."
)
_DEFAULT_PATH_HASH_COUPLING = (
    "fresh-stream's control 1 asserts the default path's config is S0-03's, byte "
    "for byte. Stamping n_documents on the default path moves that hash: bar (1) "
    "read from the experiment that reproduces it."
)
_N64_CALL_COUPLING = (
    "fresh-stream arm B and scaffold-dose's single run are defined as the "
    "corpus-size N=64 call. If the control arm passes n_documents, that call leaves "
    "train()'s default path, so both identity tests fail: bar (1) seen from its two "
    "consumers."
)
_EVERY_SEED_COUPLING = (
    "fresh-stream's verdict and scaffold-dose's U call scaffold-timing's `holds` "
    "(imported: experiments/fresh-stream/run.py `ST.holds`), so weakening it to "
    "any-seed weakens their every-seed readouts too; fresh-escape reads R through "
    "fresh-stream's `arm_readout` (experiments/fresh-escape/run.py `FS.arm_readout`), "
    "so it is the fourth reader. One rule, four readers."
)
_S003_CORPUS_COUPLING = (
    "S0-03: the test reads a property of the in-stream answer token itself (its "
    "position, its id under the supervision mask, the vocabulary it adds), so it "
    "cannot hold on a corpus that has no such token."
)

_C0_S003_WORKLOAD_COUPLING = (
    "capacity-c0: C0's workloads are S0-03's CONFIG and rewardable corpus, imported "
    "(PREREG training_workload / core_workload), and this test really trains the "
    "core workload twice; a corpus with no in-stream answer token cannot train."
)

#: scaffold-dose tests that assert a classification reached through k* and the
#: table, end to end: a mutation of either reddens them by design.
_SCAFFOLD_DOSE_CLASSIFIED = (
    "test_U_needs_every_seed",
    "test_k_star_is_the_smallest",
    "test_monotonicity_is_reported",
    "test_render_results_passes_the_audit",
    "test_run_all_order_and_classification",
    "test_verdict_every_row_from_tables[unlocked2-AT_MEMORISATION]",
    "test_verdict_every_row_from_tables[unlocked3-AT_MEMORISATION]",
    "test_verdict_every_row_from_tables[unlocked4-EARLY]",
    "test_verdict_every_row_from_tables[unlocked5-EARLY]",
)

#: fresh-escape tests that assert a classification reached through P, R(P), the
#: stream-loss windows and the table, end to end: a mutation of the table reddens
#: them by design.
_FRESH_ESCAPE_CLASSIFIED = (
    "test_P_is_the_largest_checkpoint_on_every_seed",
    "test_R_needs_every_seed_and_reads_P_only",
    "test_deadline_stops_the_children_and_P_is_what_exists",
    "test_render_results_passes_the_audit",
    "test_run_all_stirring_from_the_heartbeats",
    "test_stirring_window_must_end_at_or_before_P",
    "test_verdict_every_row_from_tables[stirring]",
    "test_verdict_every_row_from_tables[stirring_R_before_P]",
    "test_verdict_every_row_from_tables[none]",
    "test_post_deadline_measurement_is_excluded_from_P",
)
#: fresh-escape tests that read P = 9000 (or 6000 / 7000) from a full or partial
#: run: P the smallest checkpoint makes every one of them read P = 4000.
_FRESH_ESCAPE_READS_P = (
    "test_P_below_6000_is_inconclusive",
    "test_P_below_6000_under_the_deadline_filter",
    "test_post_deadline_measurement_is_excluded_from_P",
    "test_run_all_under_the_stamped_deadline",
    "test_R_needs_every_seed_and_reads_P_only",
    "test_deadline_before_6000_is_inconclusive",
    "test_deadline_stops_the_children_and_P_is_what_exists",
    *(
        f"test_failed_controls_are_inconclusive[{c}]"
        for c in (
            "control_1_absent",
            "control_1_failed",
            "control_1_raised",
            "measurement_raised",
            "preflight_absent",
            "preflight_failed",
            "resume_failed",
            "resume_missing",
        )
    ),
    "test_render_results_passes_the_audit",
    "test_run_all_order_and_classification",
    "test_run_all_stirring_from_the_heartbeats",
    "test_stirring_window_must_end_at_or_before_P",
    "test_verdict_every_row_from_tables[escapes]",
    "test_verdict_every_row_from_tables[escapes_also_stirring]",
    "test_verdict_every_row_from_tables[stirring]",
    "test_verdict_every_row_from_tables[stirring_R_before_P]",
    "test_verdict_every_row_from_tables[none]",
)

_DISPLACEMENT_COUPLING = (
    "the displacement statistic is asserted in test_instrumentation.py and in "
    "test_reduction.py because it is both a property of the metric and a property "
    "of the reduction (ADR-0006)."
)

#: liveness-wiring: tests that read the measurement a real train() run wrote.
_E0E_READS_R_COUPLING = (
    "E0e reads r_i through reward.retrieval_demand (gated, rescaled), imported, not "
    "retyped; its hand-built gated/underfull trace reddens with the target's own "
    "tests. One target, two readers."
)
_LIVENESS_HOOK_READERS = (
    "test_the_controls_read_what_they_must_on_a_real_run",
    "test_the_decoy_is_the_untrained_model_at_the_same_seed",
    "test_an_inert_run_quarantines_its_checkpoints",
    "test_the_loader_refuses_a_quarantined_checkpoint",
)
_LIVENESS_HOOK_COUPLING = (
    "liveness-wiring: it reads what the real train() measured (the controls, the "
    "decoy, or the band that triggers the quarantine). With the hook skipped there "
    "is no measurement to read: the same defect, seen from each reader."
)
_LIVENESS_MAPPING_COUPLING = (
    "liveness-wiring: main() and the quarantine test read the exit through the one "
    "band->exit table (loop.LIVENESS_EXIT), so its mapping reddens them too, by "
    "design: one table, not two."
)
_LIVENESS_INSTRUMENT_COUPLING = (
    "liveness-wiring: every train() now ends with memory_liveness's controls, and "
    "these tests read them off a real run. Breaking the instrument breaks the "
    "controls those tests assert: the same instrument, seen from the wiring."
)
_LIVENESS_TRAIN_S003_COUPLING = (
    "liveness-wiring: the test trains a real run on the default corpus (and the "
    "liveness batch is built from it), so a corpus answer_targets refuses cannot "
    "train: _S003_REFUSAL_COUPLING, seen from the liveness tests."
)
_T_WARM_TRAIN_S003_COUPLING = (
    "correction 31's train-driven tests put an RSRPolicy into the real train() on "
    "the default corpus, so a corpus answer_targets refuses cannot train: "
    "_S003_REFUSAL_COUPLING, seen from the warmup tests. Undeclared since PR #47; "
    "found by liveness-wiring's subset battery (2026-09-26) and confirmed on "
    "cd41d4e with the two tests alone."
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
_USAGE_ERROR_COUPLING = (
    "all three tests exercise the shared rsr.exit_codes.ArgumentParser usage-error "
    "path; one edit to its refusal reddens each CLI that asserts it. Confirmed "
    "2026-09-22 by applying the mutation at 5ecda88 (owner proxy, MacBook)."
)
_AUDIT_FLAGS_COUPLING = (
    "the test asserts that `audit_prose` flags a specific literal; a mutation that "
    "makes the audit flag nothing necessarily reddens every such assertion. The "
    "small-integer rule's own mutations (under `# --- orchestrator: "
    "render_scoreboard ---`) each redden only their gate."
)

_C0_RESTART_COUPLING = (
    "capacity-c0 server safety (2026-09-22): a signal (S2) and a kill that raises "
    "mid-stop (S1) are both exceptions out of the run with servers down, and each "
    "test asserts they come back -- through the same finally this mutation empties."
)

_LOOKAHEAD_READS_R_COUPLING = (
    "lookahead-room (W10) reads r_i through the same retrieval_demand(gated=True) "
    "and pins it on E0e's hand-built trace; an edit to reward.py reddening that "
    "test is the target seen from the experiment that consumes it."
)

_LOOKAHEAD_S003_COUPLING = (
    "lookahead-room's tiny-model tests build answer targets from the default "
    "synthetic corpus via answer_targets, which refuses an out-of-band corpus: the "
    "S0-03 refusal, seen from W10's harness."
)

# W11 (2026-09-26): couplings found by the final integration battery, each looked at
# under its mutation before being declared (the failing line is named in the reason).
_LOO_S003_COUPLING = (
    "loo: the instrument's no-knockout path IS S0-03's answer_readout, and its "
    "tests encode S0-03 documents through answer_targets (loop.py refuses an "
    "out-of-band corpus: 27 tests) or re-encode a twin whose rewritten answers "
    "must be in the vocabulary (encode KeyError: 3 tests). A corpus with no "
    "in-stream answer has nothing to read: _S003_REFUSAL_COUPLING, seen from loo."
)
_LOO_S003_READERS = (
    "test_all_slots_resample_donor_excludes_the_queried_key",
    "test_all_slots_resample_swaps_whole_rows_for_a_different_documents_memory",
    "test_batched_equals_row_by_row_for_row_local_conditions",
    "test_bos_off_variants_drop_only_the_bos_context",
    "test_control_is_never_of_the_queried_key",
    "test_deterministic_under_a_seed_and_seed_moves_donors",
    "test_donor_choice_does_not_depend_on_global_rng",
    "test_duplicate_key_in_document_is_flagged",
    "test_every_condition_matches_a_hand_recomputation",
    "test_every_forward_is_eval_mode_and_no_grad_and_mode_is_restored",
    "test_live_path_is_bit_exact_to_answer_readout",
    "test_loo_delta_loss_matches_a_direct_single_knockout",
    "test_loo_delta_loss_padding_step_is_nan_not_zero",
    "test_loo_delta_loss_resample_excludes_the_slots_own_key",
    "test_loo_delta_loss_resample_records_donor_or_nan",
    "test_loo_delta_loss_step_one_by_hand",
    "test_loo_delta_loss_zero_mode_needs_no_annotations",
    "test_memory_off_masks_every_slot_for_the_query_forward_only",
    "test_missing_control_is_recorded_not_substituted",
    "test_missing_donor_is_recorded_not_substituted",
    "test_no_knockout_is_written_back",
    "test_object_flags_and_the_donor_object_exclusion",
    "test_own_donor_excludes_the_queried_key",
    "test_own_slot_is_the_slot_holding_t_minus_gap",
    "test_own_status_distinguishes_evicted_from_never_written",
    "test_pick_donor_is_called_with_the_required_exclusions",
    "test_readout_knockout_forwards_differ_from_live_only_at_the_target",
    "test_records_are_ordered_step_major_then_row",
    "test_rejects_a_model_without_memory",
    "test_resample_donor_is_same_kind_different_doc_same_rank",
)
_READABILITY_S003_COUPLING = (
    "retention-readability: every one of these tests takes the module fixture "
    "`world`, which maps ANSWER_SYMBOLS through the S0-03 corpus's vocabulary to "
    "read answers under the 16-way mask; an out-of-band corpus adds no answer "
    "token, so the fixture errors at setup (KeyError) for all of them: "
    "_S003_CORPUS_COUPLING, seen from the readability harness."
)
_READABILITY_WORLD_READERS = (
    "test_batched_rows_are_refused",
    "test_document_id_mismatch_raises",
    "test_documents_are_E_then_P_and_closed",
    "test_factfiller_evicts_no_assert_while_a_filler_is_live",
    "test_fifo_harness_bit_exact_to_answer_readout",
    "test_oracle_demand_of_another_document_raises",
    "test_oracle_keeps_every_queried_fact",
    "test_rank_index_is_M_minus_gap_under_fifo",
    "test_residency_matches_simulate_for_every_arm",
)
_CARRY_FORWARD_S003_COUPLING = (
    "carry-forward's end-to-end test encodes H64 through encode_set, which calls "
    "answer_targets; an out-of-band corpus is refused there: "
    "_S003_REFUSAL_COUPLING, seen from carry-forward."
)
_CARRY_FORWARD_BITEXACT_COUPLING = (
    "carry-forward's end-to-end test asserts its bit-exactness control on a real "
    "measure_one: loo_readout's no-knockout path must equal S0-03's "
    "answer_readout(cond='live') on H64 and an EXT chunk, field by field. A defect "
    "on either side of that equality (the readout's Brier16, the live forward's "
    "bos context, a knockout left in the live memory) fails the control by design: "
    "loo's own exactness gate, seen from the experiment that relies on it."
)
_HELDOUT_READERS_COUPLING = (
    "carry-forward and retention-readability take their held-out set from "
    "corpus-size-curve (CSC.HELDOUT, CSC.doc_sets) and check it by document id "
    "(ids 4096..4159); held-out drawn from docs 64..127 fails those id checks "
    "(carry-forward: 'H64 by id differs from doc_sets' H64'). One held-out set, "
    "three readers."
)
_LIVENESS_BATCH_STAMP_COUPLING = (
    "liveness-wiring: loop.liveness_batch rebuilds the measurement corpus from the "
    "frozen config and reads n_documents / stream only when the key is PRESENT -- "
    "the stamped-only-when-set convention this mutation breaks. A None stamped on "
    "the default path reaches generate() as n_documents=None (or None['vocab_"
    "documents']), so every real train's liveness measurement is invalid and each "
    "test that reads it reddens: the default-path config, seen from liveness."
)

_I1_E2E_WIRING = (
    "I1: the end-to-end stub tests run the real battery; each depends on the "
    "guard this mutation disables for its own assertion, so one edit reddens the "
    "unit gate and every end-to-end path through the same guard. Asserted both "
    "ways on purpose: a unit test alone would pass a guard that is never called."
)
_I1_SHARD_MUTATION = (
    "I1: every end-to-end stub test either asserts the invoking tree stays clean "
    "mid-run and after, or needs the mutation to land in the shard to get its "
    "verdict; a battery that mutates the invoking tree breaks each by construction."
)
_I1_TEARDOWN = (
    "I1: every end-to-end stub test that finishes or stops a battery asserts no "
    "worktree registration is left; a pool never torn down leaves one in each."
)
_I1_DIRTY = (
    "I1: tracked edits and untracked files are the two halves of 'dirty'; one "
    "check covers both, so one edit reddens both tests."
)


_I1_RESET_E2E = (
    "I1 review: every end-to-end stub test runs more than one suite in one shard, "
    "so the reset between them is on its path; a reset that leaves state behind "
    "fails verify_clean or changes the next verdict in each."
)
_I1_POOL_LOCK = (
    "I1 review: --prune-shards and a battery take the same pool lock; one edit to "
    "the lock reddens both tests."
)
_I1_GROUP = (
    "I1 review: every test that stops a suite -- the battery or the pytest child -- "
    "asserts the suite's grandchild is gone; one edit to the group kill reddens "
    "each."
)


def _i1_declare(reason: str, *nodes: str) -> tuple[tuple[str, str], ...]:
    """Declared couplings on tests/test_battery_isolation.py nodes, one reason."""
    return tuple((f"tests/test_battery_isolation.py::{n}", reason) for n in nodes)


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "t_warm back to inf",
        "test_reduction",
        "src/rsr/retention/rsr.py",
        '    "t_warm": 0.0,',
        '    "t_warm": float("inf"),',
        "gauntlet 0.1: the reduction stops reaching the score path",
        off_gate_allowed=(
            (
                "tests/test_t_warm_dispatch.py::test_the_reduction_needs_no_training_st"
                "ep",
                _REDUCTION_WARMUP_LOUD_COUPLING,
            ),
            (
                "tests/test_device_placement.py::"
                "test_rsr_policy_runs_where_the_memory_lives[neg_age]",
                _REDUCTION_WARMUP_LOUD_COUPLING,
            ),
        ),
    ),
    Mutation(
        "reduction uses the learned head",
        "test_reduction",
        "src/rsr/retention/rsr.py",
        '    "psi_override": "neg_age",',
        '    "psi_override": None,',
        "the §3.7 reduction is no longer argmin(-a_i)",
        off_gate_allowed=(
            (
                "tests/test_t_warm_dispatch.py::test_the_reduction_needs_no_training_st"
                "ep",
                "it builds RSRConfig.reduction_to_tg() and asserts the one eviction is "
                "attributed to neg_age -- the psi_override this mutation clears "
                "(measured under it, W11: {'psi': 1} != {'neg_age': 1}). The "
                "reduction, seen from correction 16's no-counter test.",
            ),
        ),
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
        )
        + (
            # W11: RSRConfig(...) missing 'nu' at construction, measured under it.
            (
                "tests/test_t_warm_dispatch.py::test_the_reduction_needs_no_training_st"
                "ep",
                _REDUCTION_TABLE_COUPLING,
            ),
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
        "        self._last_used.setdefault(slots.row, {})[occupant] = step",
        "        slots.accum.setdefault('lru', {})[occupant] = step",
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
            (
                "tests/test_on_write_slot.py::test_lru_state_is_per_batch_row",
                "2026-09-29: with recency in the per-step accum no record survives "
                "to select_eviction, so row B falls back to write order and the "
                "per-row assertion fails -- LRU losing all cross-step state, seen "
                "from the per-row test",
            ),
            (
                "tests/test_on_write_slot.py::"
                "test_lru_recency_follows_the_occupant_through_compaction",
                "2026-09-29: the step-3 use of occupant 1 is forgotten by step 4, "
                "so LRU evicts it as FIFO would -- the same lost cross-step state, "
                "seen from the compaction test",
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
            (
                "tests/test_e0e.py::test_r_i_is_the_gated_rescaled_target",
                _E0E_READS_R_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::test_r_i_is_e0es_gated_rescaled_target",
                _LOOKAHEAD_READS_R_COUPLING,
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
                "tests/test_instrumentation.py::test_evicting_the_newest_displaces_the_"
                "most",
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
            (
                "tests/test_e0e.py::test_r_i_is_the_gated_rescaled_target",
                _E0E_READS_R_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::test_r_i_is_e0es_gated_rescaled_target",
                _LOOKAHEAD_READS_R_COUPLING,
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
            # 2026-09-22: every small-integer refusal test asserts that the audit
            # flags something, so an audit that flags nothing reddens each of them.
            *(
                (f"tests/test_audit_small_ints.py::{t}", _AUDIT_FLAGS_COUPLING)
                for t in (
                    "test_the_audit_refuses_a_small_integer_matched_only_by_an_"
                    "unrelated_ledger",
                    "test_the_audit_refuses_a_small_integer_beside_its_key_with_the_"
                    "wrong_value",
                    "test_the_audit_refuses_a_small_integer_from_another_run",
                    "test_the_audit_refuses_a_small_integer_in_a_table_cell_its_"
                    "header_key_denies",
                    "test_decimals_keep_the_value_rule",
                )
            ),
        ),
    ),
    Mutation(
        "a mutated run writes the real census file",
        "test_a_mutated_suite_run_does_not_clobber_the_census",
        "scripts/mutation_battery.py",
        # 📌 Re-anchored 2026-09-22: B's thread cap turned `_suite_env`'s one-line
        # return into `env = ...`. The split literal keeps this table row from
        # being the first occurrence apply() replaces.
        '    env = {**os.environ, "RSR_TEST_COUN' + 'T": SCRATCH_COUNT}',
        "    env = {**os.environ}",
        "5.3: every mutation overwrites the file CI asserts on, and the last "
        "mutated run is what survives on disk",
    ),
    Mutation(
        "the board's own counts stop backing the prose",
        "test_the_rendered_artefact_passes_its_own_audit",
        "scripts/render_scoreboard.py",
        # 📌 Re-anchored 2026-09-22 (small-integer audit): the board's per-run
        # fields now back a number only beside their own key, via `_index()`.
        "    for row in board.rows:\n        for k, v in row.items():",
        "    for row in []:\n        for k, v in row.items():",
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
        # 📌 Re-anchored 2026-09-22 (canary split): EXIT_CODES became a multi-line
        # dict when `not_comparable` was added, so the old `...UNKNOWN}` anchor
        # would have aborted the battery at this entry.
        '    "baseline": Exit.UNKNOWN,\n',
        '    "baseline": Exit.OK,\n',
        "5.4: the ORIGINAL defect -- a first reading, which compared nothing, "
        "reported as a pass",
    ),
    Mutation(
        "a first canary reading exits 3 again",
        "test_a_first_canary_reading_exits_2_nothing_to_compare",
        "scripts/canary.py",
        '    "baseline": Exit.UNKNOWN,\n',
        '    "baseline": Exit.DID_NOT_RUN,\n',
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
        "defines did-not-run. One mutation for the class: every stub goes through "
        "`not_implemented()`, and every parametrised case reddens (six since E0e "
        "was implemented, 2026-09-26).",
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
        # 📌 Re-anchored 2026-09-29 (I1): the baseline now runs inside the shard's
        # `with`, four columns deeper. Same edit.
        "            refuse(\n"
        "                Exit.DID_NOT_RUN,\n"
        '                f"the suite is not green',
        "            refuse(\n"
        "                Exit.FAIL,\n"
        '                f"the suite is not green',
        "S0-05: a suite red before mutating means no mutation ran.",
    ),
    Mutation(
        "an empty battery passes",
        "test_a_battery_with_no_mutations_exits_2",
        "scripts/mutation_battery.py",
        # 📌 Re-anchored 2026-09-22: the thread-cap report now sits between the
        # empty-table check and the baseline run.
        "        return Exit.UNKNOWN\n\n    threads, source = suite_threads()",
        "        return Exit.OK\n\n    threads, source = suite_threads()",
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
        off_gate_allowed=tuple(
            (node, _USAGE_ERROR_COUPLING)
            for node in (
                "tests/test_canary_split.py::test_an_unknown_canary_device_did_not_run",
                "tests/test_orch_outbox.py::test_the_cli_usage_error_exits_3",
            )
        ),
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
        off_gate_allowed=(
            (
                "tests/test_e0e.py::"
                "test_fifo_lifetimes_on_a_tiny_live_model_are_the_analytic_ones",
                "E0e's recorder counts lifetimes through FIFOPolicy.observe; with "
                "observe never reached no sentence is seen live and every lifetime "
                "reads 0: the same call site, seen from E0e.",
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_probes_fill_demand_for_every_past_sentence_and_the_identity_probe"
                "_holds",
                "lookahead-room records r_i and runs its probes and online_g0 rule from "
                "observe(); never reached, no demand is recorded: the same call site, "
                "seen from W10.",
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_a_probe_reads_the_swapped_in_sentence_not_the_resident_one",
                "lookahead-room records r_i and runs its probes and online_g0 rule from "
                "observe(); never reached, no demand is recorded: the same call site, "
                "seen from W10.",
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_online_g0_evicts_its_own_argmin_and_replays_model_free",
                "lookahead-room records r_i and runs its probes and online_g0 rule from "
                "observe(); never reached, no demand is recorded: the same call site, "
                "seen from W10.",
            ),
        ),
    ),
    # ----------------------------------------------------------------------- #
    # S0-01 -- the training-loop defects. One mutation per fix, each reverting
    # exactly that fix and nothing else, per the brief's Bar.
    # ----------------------------------------------------------------------- #
    Mutation(
        "policy built unconditionally again",
        "test_train_does_not_stamp_a_policy_it_did_not_build",
        "src/rsr/train/loop.py",
        "    policy = build_policy(policy_name, d_model=d, steps_per_epoch=None, "
        "generator=gen)",
        "    policy = FIFOPolicy()",
        "S0-01 defect (b), the original line. `policy_name` still flows into the "
        "frozen config and the `run_id`, so `train(policy_name='rsr')` completes "
        "and returns `run_id='rsr-d32-...'` for a stream FIFO evicted. Nothing "
        "in the run contradicts anything else in it, which is what made the "
        "defect silent and what makes the test necessary: no assertion about the "
        "loss curve could ever have caught this, because the loss curve is "
        "genuine.",
        off_gate_allowed=(
            (
                "tests/test_t_warm_dispatch.py::test_warmup_below_S_covers_whole_traini"
                "ng_steps",
                _BUILD_POLICY_INJECTION_COUPLING,
            ),
            (
                "tests/test_t_warm_dispatch.py::test_warm_status_never_flips_inside_one"
                "_stream",
                _BUILD_POLICY_INJECTION_COUPLING,
            ),
        ),
    ),
    # Correction 31: T_warm counts optimizer steps, not sentences.
    Mutation(
        "T_warm compared against the sentence index again",
        "test_warmup_below_S_covers_whole_training_steps",
        "src/rsr/retention/rsr.py",
        "        warm = self.config.t_warm > 0 and "
        "self._train_step < self.config.t_warm\n",
        "        warm = step < self.config.t_warm\n",
        "Correction 31 (a), the original line. `step` is the sentence index inside "
        "one stream, so below S the warmup becomes a FIFO prefix of every stream "
        "from optimizer step 0, and at T_warm >= S the arm is FIFO forever -- "
        "gauntlet 0.1 by mixed units. Nothing crashes and the attribution field "
        "still reads plausibly, which is why it survived a registry, a correction "
        "and two warmup tests that fed `select_eviction` the sentence index.",
        off_gate_allowed=(
            (
                "tests/test_t_warm_dispatch.py::test_warm_status_never_flips_inside_one"
                "_stream",
                _T_WARM_COUPLING,
            ),
            (
                "tests/test_t_warm_dispatch.py::test_warmup_longer_than_a_stream_ends",
                _T_WARM_COUPLING,
            ),
        ),
    ),
    Mutation(
        "train() stops handing the policy its optimizer step",
        "test_warm_status_never_flips_inside_one_stream",
        "src/rsr/train/loop.py",
        "                policy.set_train_step(it)\n",
        "                pass  # MUTATED\n",
        "Correction 31 (a), the call-site half. A policy fixed to count optimizer "
        "steps is useless if the loop never tells it which step it is on; here the "
        "unset-step guard is what turns that into a failure instead of a silent "
        "FIFO arm.",
        off_gate_allowed=(
            (
                "tests/test_t_warm_dispatch.py::test_warmup_below_S_covers_whole_traini"
                "ng_steps",
                _T_WARM_COUPLING,
            ),
        ),
    ),
    Mutation(
        "an unset training step defaults to 0",
        "test_a_warmup_policy_refuses_to_guess_its_training_step",
        "src/rsr/retention/rsr.py",
        "        self._train_step: int | None = None\n",
        "        self._train_step: int | None = 0\n",
        "Correction 31 (a). The natural 'harmless' default: a policy nobody told "
        "the step is at step 0, so it is warm, so it is FIFO -- every eval-time "
        "use of a trained RSR policy would silently report FIFO numbers as RSR's.",
    ),
    Mutation(
        "build_policy accepts rsr without an epoch",
        "test_build_policy_refuses_rsr_without_an_epoch",
        "src/rsr/train/loop.py",
        "        if steps_per_epoch is None:\n",
        "        if False:  # MUTATED\n",
        "Correction 31 (b). With the check gone, None reaches the registry, which "
        "then fails for a reason unrelated to the open epoch decision -- or, once "
        "a ledger exists, not at all.",
        off_gate_allowed=(
            (
                "tests/test_train_loop.py::test_train_does_not_stamp_a_policy_it_did_no"
                "t_build",
                "train() reaches the epoch refusal through build_policy: one "
                "refusal, two call sites, and the S0-01 test matches its message "
                "to prove `iters` is no longer passed.",
            ),
        ),
    ),
    Mutation(
        "train() passes iters as steps_per_epoch again",
        "test_train_does_not_stamp_a_policy_it_did_not_build",
        "src/rsr/train/loop.py",
        "    policy = build_policy(policy_name, d_model=d, steps_per_epoch=None, "
        "generator=gen)",
        "    policy = build_policy(policy_name, d_model=d, "
        "steps_per_epoch=float(iters), generator=gen)",
        "Correction 31 (b), the original line. `iters` makes section 3.4's one "
        "epoch the whole run, so the warmup never ends and a future 'rsr' run is "
        "FIFO throughout. Today the registry would still refuse (E1 has not "
        "logged nu/beta/gamma); the day it does not, this line is the defect.",
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
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_INSTRUMENT_COUPLING)
            for t in (
                "test_the_controls_read_what_they_must_on_a_real_run",
                "test_a_decoy_aliased_to_the_trained_model_is_invalid",
            )
        ),
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
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_INSTRUMENT_COUPLING)
            for t in (
                "test_the_controls_read_what_they_must_on_a_real_run",
                "test_a_decoy_aliased_to_the_trained_model_is_invalid",
            )
        ),
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
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_INSTRUMENT_COUPLING)
            for t in ("test_the_controls_read_what_they_must_on_a_real_run",)
        ),
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
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_INSTRUMENT_COUPLING)
            for t in ("test_the_controls_read_what_they_must_on_a_real_run",)
        ),
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
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_INSTRUMENT_COUPLING)
            for t in ("test_the_controls_read_what_they_must_on_a_real_run",)
        ),
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
        off_gate_allowed=(
            (
                "tests/test_lookahead_room.py::"
                "test_control2_reproduces_the_red_team_on_seed_0",
                "W10's control C2 re-derives the red team's oracle - FIFO headroom "
                "and the oracle / pending-FIFO identity through OraclePolicy: an "
                "anti-oracle fails both, the oracle seen from that control.",
            ),
            (
                "tests/test_retention_readability.py::test_oracle_keeps_every_queried_f"
                "act",
                "retention-readability's arm A is OraclePolicy(discounted_demand(doc)); "
                "the test asserts it keeps every queried fact resident, which an "
                "oracle evicting the most-needed sentence cannot: the oracle, seen "
                "from readability's upper-bound arm.",
            ),
        ),
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
                "tests/test_fresh_stream.py::test_default_path_config_is_unchanged",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_probes_fill_demand_for_every_past_sentence_and_the_identity_probe"
                "_holds",
                _LOOKAHEAD_S003_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_the_probes_never_feed_back_into_the_fifo_rollout",
                _LOOKAHEAD_S003_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_a_probe_reads_the_swapped_in_sentence_not_the_resident_one",
                _LOOKAHEAD_S003_COUPLING,
            ),
            (
                "tests/test_lookahead_room.py::"
                "test_online_g0_evicts_its_own_argmin_and_replays_model_free",
                _LOOKAHEAD_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::"
                "test_initial_parameters_identical_with_and_without_the_stream[0]",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::"
                "test_initial_parameters_identical_with_and_without_the_stream[2]",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::test_preflight_reports_a_closure_failure",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::test_resume_check_is_exact",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::test_resume_check_passes_on_a_real_resume",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::"
                "test_resume_continues_the_stream_at_the_resumed_step",
                _FRESH_STREAM_S003_COUPLING,
            ),
            (
                "tests/test_fresh_stream.py::"
                "test_train_default_path_is_bit_identical_with_explicit_none",
                _FRESH_STREAM_S003_COUPLING,
            ),
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
                "tests/test_capacity_c0.py::"
                "test_output_hash_identical_runs_equal_while_checkpoint_files_differ",
                _C0_S003_WORKLOAD_COUPLING,
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
            *(
                (f"tests/test_corpus_size_curve.py::{t}", _S003_REFUSAL_COUPLING)
                for t in (
                    "test_default_corpus_call_is_unchanged",
                    "test_default_path_config_hash_is_s003s",
                )
            ),
            *(
                (f"tests/test_t_warm_dispatch.py::{t}", _T_WARM_TRAIN_S003_COUPLING)
                for t in (
                    "test_warm_status_never_flips_inside_one_stream",
                    "test_warmup_below_S_covers_whole_training_steps",
                )
            ),
            *(
                (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_TRAIN_S003_COUPLING)
                for t in (
                    "test_a_decoy_aliased_to_the_trained_model_is_invalid",
                    "test_a_live_run_is_not_quarantined",
                    "test_an_inert_run_quarantines_its_checkpoints",
                    "test_every_train_ends_with_the_liveness_measurement",
                    "test_the_controls_read_what_they_must_on_a_real_run",
                    "test_the_decoy_is_the_untrained_model_at_the_same_seed",
                    "test_the_loader_refuses_a_quarantined_checkpoint",
                )
            ),
            *((f"tests/test_loo.py::{t}", _LOO_S003_COUPLING) for t in _LOO_S003_READERS),
            *(
                (f"tests/test_retention_readability.py::{t}", _READABILITY_S003_COUPLING)
                for t in _READABILITY_WORLD_READERS
            ),
            (
                "tests/test_carry_forward.py::test_measure_one_end_to_end_on_a_tiny_mod"
                "el",
                _CARRY_FORWARD_S003_COUPLING,
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
        # 📌 Re-anchored 2026-09-22: the check moved into the module-level
        # ruling_problem() (one indent shallower) when unparseable rulings were
        # made loud (2e2063f). The battery at 5ecda88 refused 3 on this anchor.
        '    for k in ("date", "stated_in"):\n        if not meta.get(k):',
        "    for k in ():\n        if not meta.get(k):",
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
                "tests/test_orch_slot.py::test_a_child_killed_by_a_signal_exits_128_plu"
                "s_n",
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
        "lint_brief: a baseline the base does not contain passes",
        "test_a_baseline_the_base_does_not_contain_is_a_finding",
        "scripts/orchestrator/lint_brief.py",
        '    if _git(root, "merge-base", "--is-ancestor", sha, base).returncode != 0:\n',
        "    if False:\n",
        "A brief written against a tree this night's base does not contain lints "
        "clean; its anchors were derived somewhere else.",
    ),
    Mutation(
        "lint_brief: the brief's own commit passes as its baseline",
        "test_a_wrong_baseline_sha_is_a_finding",
        "scripts/orchestrator/lint_brief.py",
        '    if rel is not None and _git(root, "cat-file", "-e", f"{sha}:{rel}")'
        ".returncode == 0:\n",
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
    # --- orchestrator: hooks --- #
    Mutation(
        "hooks: .claude/** is no longer owner-only",
        "test_orch_hooks.py::test_owner_only_",
        "scripts/orchestrator/hooks.py",
        '            if s == ".claude" and nxt != "worktrees":',
        "            if False:",
        "A headless agent could rewrite its own role file or the settings that load "
        "its hooks -- the one edit that switches every other guardrail off. Only the "
        ".claude cases of the owner-only tests can see it.",
    ),
    Mutation(
        "hooks: a push that names main is not recognised",
        "test_orch_hooks.py::test_push_to_main_",
        "scripts/orchestrator/hooks.py",
        '    return dst in ("main", "refs/heads/main") or dst.endswith('
        '"/refs/heads/main")',
        "    return False",
        "R-2026-09-22-night-branch: main never moves in a night. With the refspec "
        "check gone every spelling of a push to main -- main, HEAD:main, x:refs/heads/"
        "main, :main, via -C/-c/bash -c, or `git update-ref` -- goes through.",
    ),
    Mutation(
        "hooks: a researcher may read every outbox file",
        "test_orch_hooks.py::test_blinding_",
        "scripts/orchestrator/hooks.py",
        '    if ctx.profile == "researcher" and ctx.run_id and kind == '
        'f"{ctx.run_id}.md":',
        '    if ctx.profile == "researcher":',
        "The blinding failure of 09-21: the decisive researcher read S0-03's numbers "
        "through the shared outbox. The own-file exemption widened to 'any file' "
        "un-blinds every researcher while its own report still works.",
    ),
    Mutation(
        "hooks: bare pytest is no longer slot-wrapped",
        "test_orch_hooks.py::test_unwrapped_compute_",
        "scripts/orchestrator/hooks.py",
        '    if prog in ("pytest", "py.test") and not in_slot:',
        "    if False:",
        "Compute outside lane slots: two sessions each running the full suite on the "
        "Studio at once is the contention the slot lanes exist to prevent.",
    ),
    Mutation(
        "hooks: an internal error lets the tool call through",
        "test_orch_hooks.py::test_fail_closed_",
        "scripts/orchestrator/hooks.py",
        '            reason = f"orchestrator hook error, failing closed: '
        '{type(e).__name__}: {e}"',
        "            reason = None",
        "Claude Code treats any hook exit other than 0/2 as a non-blocking error and "
        "runs the tool call. A parser bug would then silently disable every rule.",
    ),
    # --- orchestrator: outbox --- #
    Mutation(
        "outbox: the template placeholder counts as an answer",
        "test_orch_outbox.py::test_template_placeholder_is_not_an_answer",
        "scripts/orchestrator/outbox.py",
        '    return content.startswith("<")',
        "    return False",
        "`BELIEVED, NOT VERIFIED: <list>` would pass the Stop hook: a report can end "
        "with the template's own text in a required field, which is an absent field "
        "that looks present.",
        off_gate_allowed=(
            (
                "tests/test_orch_outbox.py::test_new_writes_the_template_with_the_run_id",
                "a freshly generated template must fail on every required field; the "
                "placeholder rule is exactly what makes it fail.",
            ),
        ),
    ),
    # --- orchestrator: verify --- #
    Mutation(
        "verify: the claim draw ignores the run id",
        "test_orch_verify.py::test_draw_is_seeded_by_run_id",
        "scripts/orchestrator/verify.py",
        "    rng = random.Random(int(seed_for(run_id), 16))",
        "    rng = random.Random(0)",
        "Every run would re-execute the same claim positions, so a researcher who "
        "knew the draw could put the claims it trusts least where it is never drawn. "
        "Consistent everywhere else, so only the seed test sees it.",
    ),
    # --- orchestrator: canary ---
    Mutation(
        "a changed loss path is compared anyway",
        "test_canary_not_comparable",
        "scripts/canary.py",
        "        if not ok:\n"
        '            verdict, moved, detail = "not_comparable", [], why',
        "        if False:\n"
        '            verdict, moved, detail = "not_comparable", [], why',
        "canary split 2026-09-22: at 945b501 a code change read MOVED on 6 of 6 "
        "beats and exited 1 -- a code change reported as an environment move. "
        "Skipping the comparability check restores exactly that.",
    ),
    Mutation(
        "the CPU canary gets a tolerance",
        "test_the_cpu_canary_is_bit_exact",
        "scripts/canary.py",
        "REL_TOL_CPU = 0.0",
        "REL_TOL_CPU = 1e-4",
        "canary split: the CPU variant exists to be bit-exact; a tolerance "
        "silently makes it a second MPS canary.",
    ),
    Mutation(
        "the CPU canary runs on every thread",
        "test_the_cpu_canary_trains_on_one_thread",
        "scripts/canary.py",
        "        torch.set_num_threads(CPU_THREADS)\n",
        "        pass\n",
        "canary split: multi-threaded CPU reductions are the nondeterminism a "
        "bit-exact canary cannot absorb.",
    ),
    Mutation(
        "loss_path_hash ignores file contents",
        "test_loss_path_hash_moves_with_the_code",
        "scripts/canary.py",
        "        h.update((repo / rel).read_bytes())\n",
        "        pass\n",
        "canary split: a hash over names only reads an edited loop.py as the same "
        "code, and a code change goes back to reading as a MOVED environment.",
    ),
    # --- orchestrator: render_scoreboard ---
    Mutation(
        "small integers are backed by any ledger again",
        "test_the_audit_refuses_a_small_integer",
        "scripts/render_scoreboard.py",
        "            if integer and abs(val) <= SMALL_INT:",
        "            if False:",
        "2026-09-22 audit fix: any literal equal to any number in any ledger "
        "passed, so '3 seeds' was backed by whichever ledger held a 3. This is "
        "the hole itself.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::"
                "test_render_results_after_a_failed_control_passes_the_audit",
                _FRESH_STREAM_AUDIT_COUPLING,
            ),
            (
                "tests/test_capacity_c0.py::test_rendered_results_pass_the_prose_audit",
                "C0's RESULTS renderer backs each small integer by naming its key in "
                "the same sentence -- the rule this mutation removes. Under the old "
                "value rule those integers have no equal value in C0's ledger, so the "
                "rendered page fails the audit: the same rule, seen from C0.",
            ),
        ),
    ),
    Mutation(
        "a key from another run backs the sentence",
        "test_the_audit_refuses_a_small_integer_from_another_run",
        "scripts/render_scoreboard.py",
        "                if named and rid not in named and rid != BOARD_ID:",
        "                if False:",
        "2026-09-22 audit fix: a sentence about canary/cycle-08 borrowing "
        "decisive-shuffle's steps_done is a same-value match across unrelated "
        "ledgers wearing a key's name.",
    ),
    Mutation(
        "a plain English word names a key",
        "test_the_audit_refuses_a_small_integer_matched_only_by_an_unrelated_ledger",
        "scripts/render_scoreboard.py",
        '        if any(c in w for c in "_./-"):',
        "        if True:",
        "2026-09-22 audit fix: `seeds` is a key in four ledgers, so the prose word "
        "'seeds' backed '3 seeds' by one of them listing seed 3.",
    ),
    Mutation(
        "a dangling token passes",
        "test_a_token_that_resolves_to_no_key_is_flagged",
        "scripts/render_scoreboard.py",
        "            if dangling and m.group(0) not in unbacked:",
        "            if False:",
        "2026-09-22 audit fix: a {{run_id:key}} that resolves to nothing would "
        "read as backed prose while naming no measurement.",
    ),
    # --- orchestrator: render_status ---
    Mutation(
        "status.json numbers are not compared to their ledgers",
        "test_render_status_refuses_a_number",
        "scripts/render_status.py",
        '            if not _equal(flat[key], n.get("value")):',
        "            if False:",
        "2026-09-22 research map: a number typed into status.json that its ledger "
        "does not hold would reach the page -- the transcription layer every "
        "retracted number on this project came from (RESEARCH-CONTEXT §11).",
    ),
    Mutation(
        "status.json evidence paths are not checked",
        "test_render_status_flags_a_missing_evidence_path",
        "scripts/render_status.py",
        "        if not path or not (repo / path).exists():",
        "        if False:",
        "2026-09-22 research map: an item citing a RESULTS.md that does not exist "
        "reads as evidenced.",
    ),
    # --- orchestrator: workqueue rulings (2026-09-22) ---
    Mutation(
        "an unreadable ruling at base passes as absent again",
        "test_unreadable_ruling_at_base_stops_ready",
        "scripts/orchestrator/workqueue.py",
        "    unreadable = q.unreadable_rulings(q.base())\n    if unreadable:\n",
        "    unreadable = q.unreadable_rulings(q.base())\n    if False:\n",
        "Three 09-22 rulings with an unquoted ': ' were invalid YAML and read as "
        "'unsigned' with no error: a decision silently not in force.",
    ),
    Mutation(
        "validate ignores ruling files that do not parse",
        "test_unreadable_ruling_in_tree_fails_validate",
        "scripts/orchestrator/workqueue.py",
        "    return Exit.FAIL if (bad or rbad) else Exit.OK\n",
        "    return Exit.FAIL if bad else Exit.OK\n",
        "The one pre-commit check that would have caught the 09-22 YAML defect "
        "before a ruling reached any base.",
    ),
    # --- battery: anchor check (2026-09-22) ---
    Mutation(
        "the anchor check counts nothing",
        "test_check_anchors_",
        "scripts/mutation_battery.py",
        "        if n != 1:\n            out.append(",
        "        if False:\n            out.append(",
        "5ecda88's battery ran 87 minutes, then refused on one stale anchor a "
        "string search finds at once; the check is what makes that a 0-second exit.",
    ),
    # --- orchestrator: verifier write guard (PR #36, 2026-09-22) ---
    Mutation(
        "the verifier's blanket /tmp allowance restored",
        "test_verifier_confined_under_a_tmp_prefix",
        "scripts/orchestrator/hooks.py",
        '    allowed = [payload.get("scratchpad_dir")]\n',
        '    allowed = [payload.get("scratchpad_dir"), "/tmp", "/private/tmp"]\n',
        "PR #36 was red on Linux CI and green on the Mac: with /tmp allowed, a "
        "repo checked out under /tmp let the verifier write src/.",
    ),
    Mutation(
        "the verifier's work-tree check dropped",
        "test_verifier_scratchpad_inside_a_work_tree_is_denied",
        "scripts/orchestrator/hooks.py",
        "    return not _inside_work_tree(p)\n",
        "    return True\n",
        "A scratchpad_dir pointing into a repo would become a way around "
        "'the verifier never edits code'.",
    ),
    # --- capacity-c0 ---
    Mutation(
        "C0: the determinism comparison skipped (every job matches)",
        "test_det_verdict",
        "experiments/capacity-c0/run.py",
        "        match = job.output_hash == ref\n",
        "        match = True\n",
        "capacity-c0 brief bar, mutation (1). The falsifier is the one gate in C0 "
        "that fails where nothing else does; a verdict that never compares writes "
        "capacities for a lane whose premise (CPU runs are bit-exact) is false.",
    ),
    Mutation(
        "C0: cpu_det_slots from the faster repetition",
        "test_cpu_det_slots_slower_rep",
        "experiments/capacity-c0/run.py",
        "    return max(per_rep)\n",
        "    return min(per_rep)\n",
        "capacity-c0 brief bar, mutation (2); PREREG repetitions: 'the SLOWER of "
        "the two repetitions is used'. The faster one overstates the core ceiling.",
    ),
    Mutation(
        "C0: the checkpoint file hashed instead of the state_dict",
        "test_output_hash",
        "experiments/capacity-c0/run.py",
        '    digest = state_dict_sha256(payload["model"])\n',
        "    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()\n",
        "capacity-c0 brief bar, mutation (3); PREREG output_hash. The file carries "
        "unseeded python/numpy RNG state, so identical runs would read as "
        "non-deterministic and falsify C0 spuriously.",
    ),
    Mutation(
        "C0: two jobs given the same out_dir",
        "test_job_out_dirs",
        "experiments/capacity-c0/run.py",
        '        / f"rep{spec.rep}"\n        / f"job{spec.j}"\n',
        '        / f"rep{spec.rep}"\n',
        "capacity-c0 brief bar, mutation (4); PREREG job_isolation. Concurrent "
        "jobs sharing an out_dir overwrite each other's checkpoint and heartbeat.",
        off_gate_allowed=(
            (
                "tests/test_capacity_c0.py::"
                "test_output_hash_identical_runs_equal_while_checkpoint_files_differ",
                "capacity-c0: `Wave` refuses a job whose out_dir already exists, so "
                "the real two-job wave in the output-hash test is refused under the "
                "same edit. Defence in depth for job_isolation, by design.",
            ),
        ),
    ),
    Mutation(
        "C0: the servers not restarted when the run raises",
        "test_servers_restarted",
        "experiments/capacity-c0/run.py",
        "    except BaseException as e:\n        if led.doc.get(",
        "    except BaseException as e:\n        stopped = []\n        if led.doc.get(",
        "R-2026-09-22-mlx-servers: the owner's servers are restarted when C0 "
        "completes, however it completes. A restart that runs only on success "
        "leaves them down after a deadline, a refusal or a crash.",
        off_gate_allowed=(
            (
                "tests/test_capacity_c0.py::test_partial_on_deadline",
                "capacity-c0: a deadline is an exception out of the measurement, and "
                "the test asserts the servers come back after it.",
            ),
            (
                "tests/test_capacity_c0.py::"
                "test_quiet_yield_stops_only_the_servers_and_records_them",
                "capacity-c0: 'machine not quiet' after a YIELD stop is a refusal "
                "raised with the servers down, and the test asserts they come back.",
            ),
            *(
                (node, _C0_RESTART_COUPLING)
                for node in (
                    "tests/test_capacity_c0.py::"
                    "test_servers_signal_restarts_and_writes_partial[SIGTERM]",
                    "tests/test_capacity_c0.py::"
                    "test_servers_signal_restarts_and_writes_partial[SIGHUP]",
                    "tests/test_capacity_c0.py::"
                    "test_servers_stop_marks_each_kill_so_a_later_failure_still_restarts",
                )
            ),
        ),
    ),
    # --- capacity-c0: server safety (2026-09-22) ---
    Mutation(
        "C0: a server is not marked stopped when its kill succeeds",
        "test_servers_stop_marks_each_kill",
        "experiments/capacity-c0/run.py",
        '        rec["stopped"] = True  # the kill succeeded: '
        "from here on it is restarted\n",
        "        pass\n",
        "S1. A kill that raised at the NEXT server left every earlier record "
        "stopped=False, so the finally skipped them as 'never stopped': the owner's "
        "servers stayed down (R-2026-09-22-mlx-servers: restart after C0).",
    ),
    Mutation(
        "C0: SIGTERM/SIGHUP handlers not installed",
        "test_servers_signal_restarts",
        "experiments/capacity-c0/run.py",
        "    with interrupt_on_signals():\n",
        "    with contextlib.nullcontext():\n",
        "S2. A SIGTERM or a closed terminal (SIGHUP) killed C0 without running any "
        "finally: servers left down, no ledger, not even 'partial'.",
    ),
    Mutation(
        "C0: a raising restart skips the ledger write",
        "test_servers_restart_failure_still_writes_ledger",
        "experiments/capacity-c0/run.py",
        "        try:\n            restarted = restart_servers(sysm, stopped, log_dir)\n"
        "        except BaseException as e:  # S2: a failed restart never skips "
        "led.write()\n",
        "        if True:\n"
        "            restarted = restart_servers(sysm, stopped, log_dir)\n"
        "        if False:\n",
        "S2. An exception out of restart_servers escaped the finally before "
        "led.write(): the run left no ledger at all, the one record of what it "
        "stopped.",
    ),
    Mutation(
        "C0: a restarted server counted without the alive check",
        "test_servers_restart_alive_check",
        "experiments/capacity-c0/run.py",
        "            alive = rc is None\n",
        "            alive = True\n",
        "S3. Popen succeeding is not a server running: one that exits at once (bad "
        "venv, port taken) was recorded 'restarted' and nobody looked.",
    ),
    Mutation(
        "C0: a caffeinate helper restarted like a server",
        "test_servers_measured_layout",
        "experiments/capacity-c0/run.py",
        '        if rec.get("role") == "helper":\n',
        "        if False:\n",
        "S4. The measured helpers are `caffeinate -s <server argv>`: restarting one "
        "launches a DUPLICATE server beside the restarted one.",
    ),
    Mutation(
        "C0: a name/port disagreement not refused",
        "test_servers_disagreement_refuses",
        "experiments/capacity-c0/run.py",
        "    if problems:\n        raise Refusal(\n"
        '            "servers: name and port disagree',
        "    if False:\n        raise Refusal(\n"
        '            "servers: name and port disagree',
        "S4. A name-matched process that is neither a listening server nor its "
        "helper, or a foreign owner of a server's declared port, means the "
        "identification is wrong; stopping anyway stops the wrong thing.",
    ),
    Mutation(
        "C0: the start barrier released before every job is ready",
        "test_barrier_",
        "experiments/capacity-c0/run.py",
        "        if not missing:\n            break\n",
        "        if True:\n            break\n",
        "S5. Without the barrier the jobs of a wave start their timed loops at "
        "different times, so at high k the early ones run partly alone: "
        "makespan(k) is biased down and cpu_det_slots up.",
    ),
    # --- retrieval-curve (docs/lab-notes/dispatch-retrieval-curve.md, Bar) ---
    Mutation(
        "retrieval-curve: decide() fed the train population",
        "test_decide_is_fed_heldout_not_train",
        "experiments/retrieval-curve/run.py",
        '    feed = {s: {"heldout": m["heldout"]} for s, m in per_seed.items()}\n',
        '    feed = {s: {"heldout": m["train"]} for s, m in per_seed.items()}\n',
        "retrieval-curve brief bar, mutation (1). The training set is 64 documents "
        "seen ~750 times by 3000 iterations: fed the train population, decide() "
        "reads memorisation as retrieval and the curve reports 'falsified'.",
    ),
    Mutation(
        "retrieval-curve: the reproduction control's tolerance widened",
        "test_reproduction_control_tolerance",
        "experiments/retrieval-curve/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-3\n",
        "retrieval-curve brief bar, mutation (2). The control is the one gate that "
        "fails where nothing else does: a curve that silently is not S0-03's "
        "configuration (another thread count, another interpreter) would pass "
        "every other check here.",
        off_gate_allowed=(
            (
                "tests/test_corpus_size_curve.py::"
                "test_control_failure_stops_every_later_arm",
                "corpus-size-curve imports this reproduction_control, so the "
                "tolerance it applies is this REPRO_TOL; widened, its 1e-3 fake "
                "miss passes and every later arm runs.",
            ),
        ),
    ),
    Mutation(
        "retrieval-curve: the reproduction control's comparison ledger swapped",
        "test_reproduction_control_reads_the_s003_ledger",
        "experiments/retrieval-curve/run.py",
        "    path = path or ROOT / S003_LEDGER\n",
        '    path = path or ROOT / "runs/decisive-shuffle/ledger.json"\n',
        "retrieval-curve brief bar, mutation (2), the swap half (PR #39 listed it "
        "unproven). A control read against another run's ledger compares the "
        "curve with the wrong reference; the gate reads the S0-03 row by key from "
        "the real ledger and must go red.",
    ),
    Mutation(
        "retrieval-curve: every checkpoint measured from the final checkpoint",
        "test_each_checkpoint_label_",
        "experiments/retrieval-curve/run.py",
        "    ckpt = ckpt_path(seed_dir, label)\n",
        "    ckpt = ckpt_path(seed_dir, ITERS)\n",
        "retrieval-curve brief bar, mutation (3). Every label read off the last "
        "checkpoint turns the curve into three copies of one point; the step "
        "stored in the checkpoint must equal its label.",
    ),
    # --- corpus-size-curve (docs/lab-notes/dispatch-corpus-size-curve.md, Bar) ---
    Mutation(
        "corpus-size-curve: n_documents stamped into the default path's config",
        "test_default_path_config_hash_is_s003s",
        "src/rsr/train/loop.py",
        "    if n_documents is not None:\n        # Stamped only when set",
        "    if True:\n        # Stamped only when set",
        "corpus-size-curve brief bar (1). The default path must stay byte for byte "
        "S0-03's: a config_hash that moves on the default call means the N=64 arm "
        "is not the configuration its reproduction control compares against.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::test_default_path_config_is_unchanged",
                _DEFAULT_PATH_HASH_COUPLING,
            ),
            *(
                (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_BATCH_STAMP_COUPLING)
                for t in (
                    "test_a_decoy_aliased_to_the_trained_model_is_invalid",
                    "test_every_train_ends_with_the_liveness_measurement",
                    *_LIVENESS_HOOK_READERS,
                )
            ),
        ),
    ),
    Mutation(
        "corpus-size-curve: held-out drawn from S0-03's docs 64..127",
        "test_heldout_is_disjoint_from_every_arm",
        "experiments/corpus-size-curve/run.py",
        "    ho = docs[HELDOUT[0] : HELDOUT[1]]\n",
        "    ho = docs[64:128]\n",
        "corpus-size-curve brief bar (2). S0-03's held-out documents are training "
        "data for every N >= 128: measured there, the large arms read memorisation "
        "as retrieval.",
        off_gate_allowed=(
            *(
                (f"tests/test_carry_forward.py::{t}", _HELDOUT_READERS_COUPLING)
                for t in (
                    "test_measure_one_end_to_end_on_a_tiny_model",
                    "test_seed_sets_closure_and_prefix_stability",
                )
            ),
            (
                "tests/test_retention_readability.py::test_documents_are_E_then_P_and_c"
                "losed",
                _HELDOUT_READERS_COUPLING,
            ),
        ),
    ),
    Mutation(
        "corpus-size-curve: Brier16 computed as an absolute distance",
        "test_brier16_",
        "experiments/s0-03-rewardable-corpus/run.py",
        "out_b16.append((lp16.exp() - onehot).pow(2).sum(-1))",
        "out_b16.append((lp16.exp() - onehot).abs().sum(-1))",
        "corpus-size-curve brief bar (3). 'Squaring is what makes it proper; "
        "absolute distance would not be' (research corpus 03_Calibration:142): the "
        "primary readout must be the squared distance.",
        off_gate_allowed=(
            (
                "tests/test_carry_forward.py::test_measure_one_end_to_end_on_a_tiny_mod"
                "el",
                _CARRY_FORWARD_BITEXACT_COUPLING,
            ),
            (
                "tests/test_loo.py::test_live_path_is_bit_exact_to_answer_readout",
                "loo's exactness gate compares its own no-knockout Brier16 (squared) "
                "with S0-03's answer_readout field by field; an absolute-distance "
                "Brier16 on the S0-03 side breaks that equality (measured max |diff| "
                "0.937, W11): the readout, seen from the instrument that reproduces it.",
            ),
        ),
    ),
    Mutation(
        "corpus-size-curve: the verdict read off the N=64 arm",
        "test_verdict_reads_the_n4096_arm_only",
        "experiments/corpus-size-curve/run.py",
        "    prim = arms.get(PRIMARY_ARM)\n",
        "    prim = arms.get(CONTROL_ARM)\n",
        "corpus-size-curve brief bar (4). The N=64 arm is the memorised one; the "
        "question is the N=4096 arm's.",
        off_gate_allowed=tuple(
            (
                f"tests/test_corpus_size_curve.py::{t}",
                "the decision-table rows are built with the N=4096 arm only; read off "
                "another arm, every row sees no arm and reads 'stopped before "
                "ckpt1000'.",
            )
            for t in (
                "test_B_needs_every_seed",
                "test_row_bar1_fails_where_B_would_be_read",
                "test_row_falsified_at_any_checkpoint[1000-gaps0]",
                "test_row_falsified_at_any_checkpoint[300-gaps1]",
                "test_row_stopped_before_ckpt1000",
                "test_row_survived",
            )
        ),
    ),
    Mutation(
        "corpus-size-curve: the reproduction control read on the common held-out set",
        "test_control_reads_s003s_own_heldout",
        "experiments/corpus-size-curve/run.py",
        '                    s, per[s][label]["s003_default"], reference\n',
        '                    s, per[s][label]["s003"], reference\n',
        "corpus-size-curve brief bar (5). S0-03's ledger was measured on its own "
        "held-out docs 64..127; compared against the common set [4096, 4160) the "
        "control compares two different populations.",
    ),
    Mutation(
        "corpus-size-curve: the control arm passes n_documents",
        "test_control_arm_omits_n_documents",
        "experiments/corpus-size-curve/run.py",
        '    kw = {} if n == CONTROL_ARM else {"n_documents": n}\n',
        '    kw = {"n_documents": n}\n',
        "corpus-size-curve brief bar (1), the call-site half: the N=64 arm must take "
        "train()'s default path, or its config_hash is not S0-03's.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::test_arms_are_the_n64_call_plus_the_stream",
                _N64_CALL_COUPLING,
            ),
            (
                "tests/test_scaffold_dose.py::"
                "test_single_is_fresh_streams_arm_B_from_ckpt_k",
                _N64_CALL_COUPLING,
            ),
        ),
    ),
    Mutation(
        "corpus-size-curve: the reproduction control's tolerance widened",
        "test_reproduction_control_tolerance",
        "experiments/corpus-size-curve/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-3\n",
        "corpus-size-curve brief bar (5). The run's REPRO_TOL is the PREREG's; the "
        "control itself is the retrieval curve's, imported.",
        off_gate_allowed=(
            (
                "tests/test_corpus_size_curve.py::test_thresholds_are_the_preregs",
                "it also transcribes REPRO_TOL from the PREREG front matter.",
            ),
        ),
    ),
    # --- scaffold-timing (experiments/scaffold-timing/PREREG.md, 167d650) ---
    Mutation(
        "scaffold-timing: the sustained requirement dropped (first crossing = onset)",
        "test_single_noisy_crossing_that_lapses_is_not_an_onset",
        "experiments/scaffold-timing/run.py",
        "        if not flags[c]:\n            break\n",
        "        if not flags[c]:\n            continue\n",
        "scaffold-timing PREREG 'Primary readout': onset is the earliest checkpoint "
        "from which R (or M) holds at that and EVERY later checkpoint -- a single "
        "noisy crossing does not count.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_timing.py::{t}",
                "it reads an onset off flags that lapse after an early crossing.",
            )
            for t in (
                "test_onset_absent_when_the_last_checkpoint_fails",
                "test_R_and_M_need_every_seed",
                "test_noisy_crossing_in_tables_does_not_move_the_onset",
            )
        ),
    ),
    Mutation(
        "scaffold-timing: the reproduction control's tolerance widened",
        "test_reproduction_control_fails_at_2e_6",
        "experiments/scaffold-timing/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-5\n",
        "scaffold-timing PREREG 'The reproduction control': every ledger key within "
        "1e-6 absolute; a 2e-6 difference must fail.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_timing.py::{t}",
                "it transcribes or exercises the same tolerance.",
            )
            for t in (
                "test_thresholds_are_the_preregs",
                "test_reproduction_control_exact_passes",
                "test_run_measures_the_control_first_and_stops_on_failure",
                "test_render_results_after_a_failed_control_passes_the_audit",
            )
        ),
    ),
    Mutation(
        "scaffold-timing: the onset comparison swapped (BEFORE <-> AFTER)",
        "test_classify_table",
        "experiments/scaffold-timing/run.py",
        "    if onset_r < onset_m:\n",
        "    if onset_r > onset_m:\n",
        "scaffold-timing PREREG decision table: BEFORE iff onset_R < onset_M, AFTER "
        "iff onset_R > onset_M.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_timing.py::{t}",
                "it asserts a BEFORE or AFTER classification end to end.",
            )
            for t in (
                "test_R_and_M_need_every_seed",
                "test_after_and_before_from_tables",
                "test_noisy_crossing_in_tables_does_not_move_the_onset",
                "test_verdict_reads_the_n64_arm_only",
                "test_run_passes_the_control_then_measures_everything",
                "test_render_results_passes_the_audit",
            )
        ),
    ),
    Mutation(
        "scaffold-timing: R and M hold on ANY seed instead of every seed",
        "test_R_and_M_need_every_seed",
        "experiments/scaffold-timing/run.py",
        "all(v >= DELTA for v in values)",
        "any(v >= DELTA for v in values)",
        "scaffold-timing PREREG 'Primary readout': R(c) and M(c) hold only when the "
        "quantity is >= DELTA on every seed.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::test_verdict_reads_ckpt3000_only",
                _EVERY_SEED_COUPLING,
            ),
            (
                "tests/test_scaffold_dose.py::test_U_needs_every_seed",
                _EVERY_SEED_COUPLING,
            ),
            (
                "tests/test_fresh_escape.py::test_R_needs_every_seed_and_reads_P_only",
                _EVERY_SEED_COUPLING,
            ),
        ),
    ),
    Mutation(
        "scaffold-timing: the measurement retyped as a local wrapper, not imported",
        "test_measurement_is_the_corpus_size_function",
        "experiments/scaffold-timing/run.py",
        "measure_checkpoint = CSC.measure_checkpoint\n",
        "def measure_checkpoint(seed_dir, seed, label, n):\n"
        "    return CSC.measure_checkpoint(seed_dir, seed, label, n)\n",
        "scaffold-timing PREREG 'instrument': the corpus-size curve's measurement path, "
        "imported -- the object called must be that function, not a copy.",
    ),
    # --- fresh-stream (experiments/fresh-stream/PREREG.md, 83128ee) ---
    Mutation(
        "fresh-stream: the stream stamped on the default path",
        "test_default_path_config_is_unchanged",
        "src/rsr/train/loop.py",
        "    if stream is not None:\n"
        "        # Stamped only when set, like n_documents: the default path's "
        "config_hash\n"
        "        # is unchanged and a stream run can never share it.\n"
        '        frozen["stream"] = asdict(stream)\n',
        '    frozen["stream"] = None if stream is None else asdict(stream)\n',
        "fresh-stream PREREG 'Condition': the stream argument's default (None) is "
        "today's path, byte for byte -- nothing stamped, S0-03's config_hash.",
        off_gate_allowed=(
            (
                "tests/test_corpus_size_curve.py::test_default_path_config_hash_is_s003s",
                "it checks the same default-path config_hash against S0-03's ledger.",
            ),
            *(
                (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_BATCH_STAMP_COUPLING)
                for t in (
                    "test_a_decoy_aliased_to_the_trained_model_is_invalid",
                    "test_every_train_ends_with_the_liveness_measurement",
                    *_LIVENESS_HOOK_READERS,
                )
            ),
        ),
    ),
    Mutation(
        "fresh-stream: a global-RNG draw before model init in stream mode",
        "test_initial_parameters_identical_with_and_without_the_stream",
        "src/rsr/train/loop.py",
        '        corpus_kw = {"n_documents": stream.vocab_documents}\n',
        '        corpus_kw = {"n_documents": stream.vocab_documents}\n'
        "        torch.rand(1)  # a draw from the global RNG before model init\n",
        "fresh-stream PREREG 'Arm A': the same initialisation as the corpus-size "
        "N = 64 arm of that seed -- anything drawn from the global RNG before model "
        "init shifts it (CLAUDE.md, test_reduction.py's RNG-ordering note).",
    ),
    Mutation(
        "fresh-stream: a resumed run restarts the stream at its first window",
        "test_resume_continues_the_stream_at_the_resumed_step",
        "src/rsr/train/loop.py",
        "                sdocs = stream_documents(stream, it, batch, stream_cfg)\n",
        "                sdocs = stream_documents(\n"
        "                    stream, it - start, batch, stream_cfg\n"
        "                )\n",
        "fresh-stream PREREG 'stream': A and B see identical documents at every step "
        "t in [1000, 3000) -- arm B, resumed at 1000, must read window 1000, not 0.",
    ),
    Mutation(
        "fresh-stream: SCAFFOLD and TRAP swapped",
        "test_classify_table",
        "experiments/fresh-stream/run.py",
        '    if r_b:\n        return "SCAFFOLD"\n    if r_a:\n        return "TRAP"\n',
        '    if r_a:\n        return "SCAFFOLD"\n    if r_b:\n        return "TRAP"\n',
        "fresh-stream PREREG classification table: R3000(B) only is SCAFFOLD, "
        "R3000(A) only is TRAP.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_stream.py::{t}",
                "it asserts a SCAFFOLD or TRAP classification end to end.",
            )
            for t in (
                "test_render_results_passes_the_audit",
                "test_run_all_happy_path_order_and_scaffold",
                "test_verdict_all_four_cells_from_tables[0.015625-0.0625-SCAFFOLD]",
                "test_verdict_all_four_cells_from_tables[0.0625-0.015625-TRAP]",
                "test_verdict_reads_ckpt3000_only",
            )
        ),
    ),
    Mutation(
        "fresh-stream: control 1's tolerance widened",
        "test_control_1_fails_at_2e_6",
        "experiments/fresh-stream/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-5\n",
        "fresh-stream PREREG control 1: every n64.ckpt100 statistic key within 1e-6 "
        "absolute; a 2e-6 difference must fail.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_stream.py::{t}",
                "it transcribes or exercises the same tolerance.",
            )
            for t in (
                "test_thresholds_are_the_preregs",
                "test_control_1_failure_stops_before_any_arm",
                "test_render_results_after_a_failed_control_passes_the_audit",
            )
        ),
    ),
    Mutation(
        "fresh-stream: the vocabulary closure disabled",
        "test_vocabulary_closure_raises_on_an_out_of_vocab_word",
        "src/rsr/train/loop.py",
        "                if w not in vocab:\n",
        "                if w not in vocab and False:\n",
        "fresh-stream PREREG 'Vocabulary': a word outside the [0, 64) map must be "
        "refused before any step, not met as a KeyError mid-run.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::test_preflight_reports_a_closure_failure",
                "the run's preflight proves the closure with the same function.",
            ),
        ),
    ),
    Mutation(
        "fresh-stream: control 3 drops the held-out overlap",
        "test_disjointness_catches_overlap",
        "experiments/fresh-stream/run.py",
        '"ok": bool(ids) and not (in_probe or in_vocab or in_heldout or repeats),',
        '"ok": bool(ids) and not (in_probe or in_vocab or repeats),',
        "fresh-stream PREREG control 3: no stream id in [0, 64) or [4096, 4160), and "
        "no id repeats.",
    ),
    Mutation(
        "fresh-stream: C holds on ANY seed instead of every seed",
        "test_C_needs_every_seed",
        "experiments/fresh-stream/run.py",
        "all(v <= C_THRESHOLD for v in values)",
        "any(v <= C_THRESHOLD for v in values)",
        "fresh-stream PREREG 'Primary readout': C(c) holds only when Brier16(live) "
        "<= 0.8875 on every seed.",
    ),
    Mutation(
        "fresh-stream: the measurement retyped as a local wrapper, not imported",
        "test_measurement_is_the_corpus_size_function",
        "experiments/fresh-stream/run.py",
        "measure_checkpoint = CSC.measure_checkpoint\n",
        "def measure_checkpoint(seed_dir, seed, label, n):\n"
        "    return CSC.measure_checkpoint(seed_dir, seed, label, n)\n",
        "fresh-stream PREREG 'Instrument': the corpus-size curve's measure_checkpoint, "
        "imported -- the object called must be that function, not a copy.",
    ),
    Mutation(
        "fresh-stream: control 2 made approximate",
        "test_resume_check_is_exact",
        "experiments/fresh-stream/run.py",
        "            and torch.equal(a, b)\n",
        "            and torch.allclose(a.double(), b.double(), atol=1e-3)\n",
        "fresh-stream PREREG control 2: arm B's model and optimizer state after load "
        "must equal ckpt-001000.pt exactly.",
    ),
    # --- scaffold-dose (experiments/scaffold-dose/PREREG.md, 4249567) ---
    Mutation(
        "scaffold-dose: AT_MEMORISATION and EARLY swapped",
        "test_classification_table",
        "experiments/scaffold-dose/run.py",
        '    if kstar in MEMORISATION_K:\n        return "AT_MEMORISATION"\n'
        '    if kstar in EARLY_K:\n        return "EARLY"\n',
        '    if kstar in MEMORISATION_K:\n        return "EARLY"\n'
        '    if kstar in EARLY_K:\n        return "AT_MEMORISATION"\n',
        "scaffold-dose PREREG classification table: k* in {300, 400} is "
        "AT_MEMORISATION, k* in {100, 200} is EARLY.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_dose.py::{t}",
                "it asserts an AT_MEMORISATION or EARLY classification end to end.",
            )
            for t in _SCAFFOLD_DOSE_CLASSIFIED
        ),
    ),
    Mutation(
        "scaffold-dose: k* the largest k with U(k)",
        "test_k_star_is_the_smallest",
        "experiments/scaffold-dose/run.py",
        "    return min(hits) if hits else None\n",
        "    return max(hits) if hits else None\n",
        "scaffold-dose PREREG decision rule: k* is the SMALLEST k in the sweep with "
        "U(k), also when U is not monotone in k.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_dose.py::{t}",
                "it asserts the classification that follows from k*, end to end.",
            )
            for t in _SCAFFOLD_DOSE_CLASSIFIED
        ),
    ),
    Mutation(
        "scaffold-dose: U(k) on ANY seed instead of every seed",
        "test_U_needs_every_seed",
        "experiments/scaffold-dose/run.py",
        '        "U": ro["R"] if (ro and not why) else None,\n',
        '        "U": any(v >= DELTA for v in ro["R_quantity"]) '
        "if (ro and not why) else None,\n",
        "scaffold-dose PREREG decision rule: U(k) holds when R holds at ckpt k + 500 "
        "on EVERY seed.",
    ),
    Mutation(
        "scaffold-dose: control 2's tolerance widened",
        "test_control_2_fails_at_2e_6",
        "experiments/scaffold-dose/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-5\n",
        "scaffold-dose PREREG control 2: every n64.ckpt600 statistic key within 1e-6 "
        "of the scaffold-timing ledger; a 2e-6 difference must fail.",
        off_gate_allowed=tuple(
            (
                f"tests/test_scaffold_dose.py::{t}",
                "it transcribes or exercises the same tolerance.",
            )
            for t in (
                "test_thresholds_are_the_preregs",
                "test_render_results_after_a_failed_control_passes_the_audit",
            )
        ),
    ),
    Mutation(
        "scaffold-dose: control 3 drops the held-out overlap",
        "test_arm_disjointness_catches_overlap",
        "experiments/scaffold-dose/run.py",
        '"ok": bool(ids) and not (in_probe or in_vocab or in_heldout or repeats),',
        '"ok": bool(ids) and not (in_probe or in_vocab or repeats),',
        "scaffold-dose PREREG control 3: no id of [4160 + 16 k, 4160 + 16 (k + 500)) "
        "in [0, 64) or [4096, 4160), and no id repeats.",
    ),
    Mutation(
        "scaffold-dose: the start-checkpoint sha256 comparison dropped",
        "test_start_checkpoint_sha_mismatch_refuses",
        "experiments/scaffold-dose/run.py",
        "            if recorded.get(rel) != sha:\n",
        "            if False and recorded.get(rel) != sha:\n",
        "scaffold-dose PREREG 'Condition': the start checkpoints' sha256 must match "
        "runs/scaffold-timing/manifest.json, or the run is refused.",
    ),
    Mutation(
        "scaffold-dose: a monotonicity violation not named",
        "test_monotonicity_is_reported",
        "experiments/scaffold-dose/run.py",
        "if k1 < k2 and u[k1] and not u[k2]",
        "if k1 < k2 and u[k1] and not u[k2] and False",
        "scaffold-dose PREREG 'Monotonicity is reported, not assumed': U(k) at some k "
        "and not at a larger k is named.",
        off_gate_allowed=(
            (
                "tests/test_scaffold_dose.py::test_render_results_passes_the_audit",
                "it asserts the monotonicity rows the ledger writes.",
            ),
        ),
    ),
    Mutation(
        "scaffold-dose: the measurement retyped as a local wrapper, not imported",
        "test_measurement_is_the_corpus_size_function",
        "experiments/scaffold-dose/run.py",
        "measure_checkpoint = FS.measure_checkpoint\n",
        "def measure_checkpoint(seed_dir, seed, label, n):\n"
        "    return FS.measure_checkpoint(seed_dir, seed, label, n)\n",
        "scaffold-dose PREREG 'instrument': the corpus-size curve's measure_checkpoint, "
        "imported -- the object called must be that function, not a copy.",
    ),
    # --- fresh-escape (experiments/fresh-escape/PREREG.md, d5b9a23) ---
    Mutation(
        "fresh-escape: STIRRING and NO_ESCAPE swapped",
        "test_classification_table",
        "experiments/fresh-escape/run.py",
        '        return "STIRRING"\n    return "NO_ESCAPE"\n',
        '        return "NO_ESCAPE"\n    return "STIRRING"\n',
        "fresh-escape PREREG classification table: not R(P) with a stream-loss window "
        "below 2.6726 at or before P is STIRRING; neither is NO_ESCAPE by P.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_escape.py::{t}",
                "it asserts a STIRRING or NO_ESCAPE classification end to end.",
            )
            for t in _FRESH_ESCAPE_CLASSIFIED
        ),
    ),
    Mutation(
        "fresh-escape: P the smallest checkpoint measured on every seed",
        "test_P_is_the_largest_checkpoint_on_every_seed",
        "experiments/fresh-escape/run.py",
        "    return max(done) if done else None\n",
        "    return min(done) if done else None\n",
        "fresh-escape PREREG decision rule: P is the LARGEST listed checkpoint "
        "measured on every seed before the deadline.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_escape.py::{t}",
                "it reads the classification or P of a run that reached ckpt >= 6000.",
            )
            for t in _FRESH_ESCAPE_READS_P
            if t != "test_P_is_the_largest_checkpoint_on_every_seed"
        ),
    ),
    Mutation(
        "fresh-escape: the 6000 minimum on P lowered",
        "test_P_below_6000_is_inconclusive",
        "experiments/fresh-escape/run.py",
        "MIN_PRIMARY_CKPT = 6000\n",
        "MIN_PRIMARY_CKPT = 4000\n",
        "fresh-escape PREREG decision rule: if P < 6000 the result is inconclusive.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_escape.py::{t}",
                "it transcribes or exercises the same minimum.",
            )
            for t in (
                "test_classification_table[4000-True-True-inconclusive]",
                "test_classification_table[5000-True-False-inconclusive]",
                "test_deadline_before_6000_is_inconclusive",
                "test_P_below_6000_under_the_deadline_filter",
                "test_manifest_records_the_start_checkpoints",
                "test_thresholds_are_the_preregs",
            )
        ),
    ),
    Mutation(
        "fresh-escape: control 1's tolerance widened",
        "test_control_1_fails_at_2e_6",
        "experiments/fresh-escape/run.py",
        "REPRO_TOL = 1e-6\n",
        "REPRO_TOL = 1e-5\n",
        "fresh-escape PREREG control 1: every A.ckpt3000 statistic key within 1e-6 "
        "of runs/fresh-stream/ledger.json; a 2e-6 difference must fail.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_escape.py::{t}",
                "it transcribes or exercises the same tolerance.",
            )
            for t in (
                "test_thresholds_are_the_preregs",
                "test_render_results_after_a_failed_control_passes_the_audit",
            )
        ),
    ),
    Mutation(
        "fresh-escape: control 3 drops the held-out overlap",
        "test_stream_disjointness_catches_overlap",
        "experiments/fresh-escape/run.py",
        '"ok": bool(ids) and not (in_probe or in_vocab or in_heldout or repeats),',
        '"ok": bool(ids) and not (in_probe or in_vocab or repeats),',
        "fresh-escape PREREG control 3: no id of [4160 + 16*3000, 4160 + 16*9000) in "
        "[0, 64) or [4096, 4160), and no id repeats.",
    ),
    Mutation(
        "fresh-escape: the start-checkpoint sha256 comparison dropped",
        "test_start_checkpoint_sha_mismatch_refuses",
        "experiments/fresh-escape/run.py",
        "        if sha != START_SHA256[s]:\n",
        "        if False and sha != START_SHA256[s]:\n",
        "fresh-escape PREREG control 3: the sha256 of each start checkpoint equals "
        "the front matter, or the run is refused.",
    ),
    Mutation(
        "fresh-escape: stream-loss windows after P counted",
        "test_stirring_window_must_end_at_or_before_P",
        "experiments/fresh-escape/run.py",
        "if int(w0) + STREAM_LOSS_WINDOW <= p}",
        "if int(w0) <= p}",
        "fresh-escape PREREG decision rule: STIRRING needs a 100-step window below "
        "2.6726 AT OR BEFORE P; ckpt-P holds steps [0, P), so [P, P + 100) is after.",
    ),
    Mutation(
        "fresh-escape: post-deadline measurements counted toward P",
        "test_post_deadline_measurement_is_excluded_from_P",
        "experiments/fresh-escape/run.py",
        '    if "s003" not in rec or rec.get("post_deadline"):\n'
        "        return False\n"
        '    at = rec.get("measured_at")\n'
        "    return at is None or _dt.datetime.fromisoformat(at).timestamp() <= "
        "deadline\n",
        '    return "s003" in rec\n',
        "fresh-escape PREREG decision rule: P is the largest listed checkpoint "
        "measured on every seed BEFORE the deadline, which is absolute; a measurement "
        "completed after 08:30 is secondary and never feeds P.",
        off_gate_allowed=tuple(
            (
                f"tests/test_fresh_escape.py::{t}",
                "it exercises the same deadline filter end to end.",
            )
            for t in (
                "test_P_below_6000_under_the_deadline_filter",
                "test_deadline_stamped_stamps_and_refuses_to_start_after_the_deadline",
                "test_run_all_under_the_stamped_deadline",
            )
        ),
    ),
    Mutation(
        "fresh-escape: the measurement retyped as a local wrapper, not imported",
        "test_measurement_is_the_corpus_size_function",
        "experiments/fresh-escape/run.py",
        "measure_checkpoint = FS.measure_checkpoint\n",
        "def measure_checkpoint(seed_dir, seed, label, n):\n"
        "    return FS.measure_checkpoint(seed_dir, seed, label, n)\n",
        "fresh-escape PREREG 'instrument': the corpus-size curve's measure_checkpoint, "
        "imported -- the object called must be that function, not a copy.",
    ),
    # -- W4: rsr.metrics.loo, the LOO slot-knockout instrument (spec §3.2.1) --
    Mutation(
        "loo: a knockout clobbers the whole row, not one slot",
        "test_knockout_kv_zero_changes_exactly_the_target_slot_of_the_target_row",
        "src/rsr/metrics/loo.py",
        "sel = sel & (slot >= 0).unsqueeze(-1)  # [B, M]",
        "sel = (slot >= 0).unsqueeze(-1).expand(-1, M)  # [B, M]",
        "§3.2.1 LOO ablates slot i: a knockout must change exactly the target slot "
        "of the target row and pass every other element through bit-exactly.",
        off_gate_allowed=(
            (
                "tests/test_loo.py::test_knockout_kv_replace_writes_the_replacement_"
                "only_there",
                "the same locality property for mode='replace': one selection mask "
                "serves both modes.",
            ),
            (
                "tests/test_loo.py::test_readout_knockout_forwards_differ_from_live_"
                "only_at_the_target",
                "the same property observed through loo_readout's forward passes; "
                "knockout_kv is the one code path, so it must go red there too.",
            ),
            (
                "tests/test_loo.py::test_every_condition_matches_a_hand_recomputation",
                "it recomputes every condition by direct indexing; a whole-row "
                "knockout is not the stated single-slot one.",
            ),
        ),
    ),
    Mutation(
        "loo: loo_readout runs with autograd enabled",
        "test_every_forward_is_eval_mode_and_no_grad_and_mode_is_restored",
        "src/rsr/metrics/loo.py",
        "@torch.no_grad()\ndef loo_readout(",
        "def loo_readout(",
        "LOO is a readout, not a training path (correction 20: eval-mode "
        "measurement); every forward runs under no_grad.",
    ),
    Mutation(
        "loo: a resample donor may come from the same document",
        "test_pick_donor_filters_kind_doc_rank_and_key",
        "src/rsr/metrics/loo.py",
        "if b == row or int(doc_id[b]) == me or not bool(valid[b, rank]):",
        "if b == row or not bool(valid[b, rank]):",
        "W4 resample semantics: the donor gestalt comes from a DIFFERENT document, "
        "or the 'knockout' can re-insert the target document's own content.",
    ),
    Mutation(
        "loo: a resample donor's sentence kind is not checked",
        "test_resample_donor_is_same_kind_different_doc_same_rank",
        "src/rsr/metrics/loo.py",
        'if int(ann["kind"][b, s]) != want_kind:',
        "if False:",
        "W4 resample semantics: same kind (assert vs filler) keeps the knocked-out "
        "input in-distribution; a filler donor for an assert is a different "
        "intervention.",
        off_gate_allowed=(
            (
                "tests/test_loo.py::test_pick_donor_filters_kind_doc_rank_and_key",
                "the unit test of the same filter: its no-candidate case (a kind no "
                "row holds) returns a donor once kind is unchecked.",
            ),
        ),
    ),
    Mutation(
        "loo: a missing control is substituted by any pending assert",
        "test_missing_control_is_recorded_not_substituted",
        "src/rsr/metrics/loo.py",
        "for rk in (r - 1, r + 1):",
        "for rk in range(M):",
        "W4 control target: an ADJACENT-rank pending assert or recorded missing -- "
        "never silently another slot.",
    ),
    Mutation(
        "loo: the live forward drops the bos-copy context",
        "test_live_path_is_bit_exact_to_answer_readout",
        "src/rsr/metrics/loo.py",
        "out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)\n"
        "            am = tmask",
        "out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid & False)\n"
        "            am = tmask",
        "The exactness control: with no knockout, loo_readout must reproduce S0-03's "
        "answer_readout(cond='live') bit-exactly, or its deltas are against a "
        "different readout.",
        off_gate_allowed=(
            *(
                (
                    "tests/test_loo.py::" + node,
                    "it compares a condition's bos flag or the live trajectory with "
                    "the honest loop's, which this mutation changes.",
                )
                for node in (
                    "test_memory_off_masks_every_slot_for_the_query_forward_only",
                    "test_bos_off_variants_drop_only_the_bos_context",
                    "test_no_knockout_is_written_back",
                    "test_every_condition_matches_a_hand_recomputation",
                )
            ),
            (
                "tests/test_carry_forward.py::test_measure_one_end_to_end_on_a_tiny_mod"
                "el",
                _CARRY_FORWARD_BITEXACT_COUPLING,
            ),
        ),
    ),
    # -- W4-fix: PR #48 review (rsr.metrics.loo, spec §3.2.1) --
    Mutation(
        "loo: the whole-memory donor may ask the queried question",
        "test_all_slots_resample_donor_excludes_the_queried_key",
        "src/rsr/metrics/loo.py",
        'and int(ann["fact_key"][b, s]) == exclude_key',
        "and False",
        (
            "PR #48 M1: an all_slots_resample donor memory holding an ass"
            "ert of the queried key can answer the query itself."
        ),
    ),
    Mutation(
        "loo: all_slots_resample keeps the row's own memory",
        ("test_all_slots_resample_swaps_whole_rows_for_a_different_documents_memory"),
        "src/rsr/metrics/loo.py",
        "            all_kv[b] = mem.kv[d]\n",
        "            all_kv[b] = mem.kv[b]\n",
        (
            "PR #48 M1: all_slots_resample replaces the row's whole memor"
            "y with the donor's."
        ),
        off_gate_allowed=(
            (
                ("tests/test_loo.py::test_every_condition_matches_a_hand_recomputation"),
                (
                    "it recomputes every condition by direct indexing of the live"
                    " memory, so it goes red for any condition's plumbing defect "
                    "by design."
                ),
            ),
        ),
    ),
    Mutation(
        "loo: memory_off leaves the memory valid",
        "test_memory_off_masks_every_slot_for_the_query_forward_only",
        "src/rsr/metrics/loo.py",
        '"memory_off": (mem.kv, no_mem, everyone),',
        '"memory_off": (mem.kv, mem.valid, everyone),',
        (
            "PR #48 M1: memory_off is valid=False for every slot, for the"
            " query forward only."
        ),
        off_gate_allowed=(
            (
                ("tests/test_loo.py::test_every_condition_matches_a_hand_recomputation"),
                (
                    "it recomputes every condition by direct indexing of the live"
                    " memory, so it goes red for any condition's plumbing defect "
                    "by design."
                ),
            ),
        ),
    ),
    Mutation(
        "loo: a bos_off condition keeps the bos-copy context",
        "test_bos_off_variants_drop_only_the_bos_context",
        "src/rsr/metrics/loo.py",
        'bv = bos_off if cond.endswith("_bos_off") else bos_valid',
        "bv = bos_valid",
        (
            "PR #48 M2: at gap 1 the bos-copy context is the assert's ges"
            "talt; *_bos_off must switch it off."
        ),
        off_gate_allowed=(
            (
                ("tests/test_loo.py::test_every_condition_matches_a_hand_recomputation"),
                (
                    "it recomputes every condition by direct indexing of the live"
                    " memory, so it goes red for any condition's plumbing defect "
                    "by design."
                ),
            ),
        ),
    ),
    Mutation(
        "loo: own donors may carry the answer object",
        "test_pick_donor_is_called_with_the_required_exclusions",
        "src/rsr/metrics/loo.py",
        (
            "            exclude_objects={qobj},\n            seed=seed,\n "
            "           t=t,\n            role=_ROLE_OWN,"
        ),
        ("            seed=seed,\n            t=t,\n            role=_ROLE_OWN,"),
        (
            "PR #48 M3: a donor carrying the queried answer object re-sup"
            "plies the answer the knockout removed."
        ),
        off_gate_allowed=(
            (
                ("tests/test_loo.py::test_object_flags_and_the_donor_object_exclusion"),
                (
                    "it checks that no donor carries the answer object: the same "
                    "exclusion, read from the records."
                ),
            ),
        ),
    ),
    Mutation(
        "loo: own donors may ask the queried question",
        "test_own_donor_excludes_the_queried_key",
        "src/rsr/metrics/loo.py",
        ("            exclude_keys={qkey},\n            exclude_objects={qobj},"),
        ("            exclude_keys=set(),\n            exclude_objects={qobj},"),
        (
            "PR #48 minor: the own donor must not ask the queried questio"
            "n (it survived at :411 before)."
        ),
        off_gate_allowed=(
            (
                (
                    "tests/test_loo.py::test_pick_donor_is_called_with_the_requir"
                    "ed_exclusions"
                ),
                "it checks the same exclusion at the pick_donor call site.",
            ),
        ),
    ),
    Mutation(
        "loo: control donors may ask the control's question",
        "test_pick_donor_is_called_with_the_required_exclusions",
        "src/rsr/metrics/loo.py",
        "exclude_keys={qkey, int(key[b, s])},",
        "exclude_keys={qkey},",
        (
            "A control donor asking the control's question re-supplies th"
            "e knocked-out fact."
        ),
    ),
    Mutation(
        "loo: the control may ask the queried question",
        "test_control_is_never_of_the_queried_key",
        "src/rsr/metrics/loo.py",
        "if pending and int(key[b, s]) != qkey:",
        "if pending:",
        (
            "PR #48 minor: a control of the queried key is not a control "
            "-- it can answer the query."
        ),
    ),
    Mutation(
        "loo: a knockout is written back into the live memory",
        "test_no_knockout_is_written_back",
        "src/rsr/metrics/loo.py",
        "    for k, v in flags.items():\n        put(k, v[idx])\n",
        (
            "    for k, v in flags.items():\n        put(k, v[idx])\n    me"
            'm.kv.copy_(inputs["own_zero"][0])\n'
        ),
        (
            "Single-step interventions: no knockout may reach the memory "
            "later steps attend over."
        ),
        off_gate_allowed=(
            (
                "tests/test_carry_forward.py::test_measure_one_end_to_end_on_a_tiny_mod"
                "el",
                _CARRY_FORWARD_BITEXACT_COUPLING,
            ),
            (
                ("tests/test_loo.py::test_live_path_is_bit_exact_to_answer_readout"),
                (
                    "a write-back changes the live memory of every later step, so"
                    " the live answers stop matching answer_readout."
                ),
            ),
            (
                ("tests/test_loo.py::test_every_condition_matches_a_hand_recomputation"),
                (
                    "it recomputes every condition by direct indexing of the live"
                    " memory, so it goes red for any condition's plumbing defect "
                    "by design."
                ),
            ),
        ),
    ),
    Mutation(
        ("loo: delta-loss resample donors may ask the slot's own question"),
        "test_loo_delta_loss_resample_excludes_the_slots_own_key",
        "src/rsr/metrics/loo.py",
        "exclude_keys={own_key} if own_key >= 0 else set(),",
        "exclude_keys=set(),",
        (
            "PR #48 minor: an E0d resample donor asking the displaced sen"
            "tence's question is not a knockout of it."
        ),
    ),
    Mutation(
        "loo: a padding step scores 0.0 instead of NaN",
        "test_loo_delta_loss_padding_step_is_nan_not_zero",
        "src/rsr/metrics/loo.py",
        'return torch.where(n > 0, mean, float("nan")).cpu()',
        "return mean.cpu()",
        (
            "PR #48 minor: a sentence with no real target has no loss; 0."
            "0 would enter E0d's correlation as data."
        ),
    ),
    Mutation(
        "loo: zero-mode delta loss requires synthetic annotations",
        "test_loo_delta_loss_zero_mode_needs_no_annotations",
        "src/rsr/metrics/loo.py",
        ('ann = stream_annotations(docs, steps=S) if mode == "resample" else None'),
        "ann = stream_annotations(docs, steps=S)",
        ("PR #48 minor: E0d's zero mode must run on non-synthetic text."),
    ),
    Mutation(
        "loo: never-written asserts reported as evicted",
        "test_own_status_distinguishes_evicted_from_never_written",
        "src/rsr/metrics/loo.py",
        ("own_status[b] = OWN_EVICTED if bool(written[b, a]) else OWN_NEVER_WRITTEN"),
        "own_status[b] = OWN_EVICTED",
        ("PR #48 minor: an assert that never entered memory was not evicted."),
    ),
    Mutation(
        "loo: the seed mix collides at role >= 131 again",
        "test_seed_mix_has_no_role_collision",
        "src/rsr/metrics/loo.py",
        '    return int.from_bytes(h[:8], "little") % (2**63)',
        ("    return (((seed * 1_000_003 + doc_id) * 8_191 + t) * 131 + role) % (2**63)"),
        ("PR #48 minor: (seed, doc, t, role) must not alias; E0d's roles are 16 + rank."),
    ),
    Mutation(
        "loo: a duplicate key in a document is not flagged",
        "test_duplicate_key_in_document_is_flagged",
        "src/rsr/metrics/loo.py",
        "dup_key[di, q] = n_asking[per_doc[di][q]] > 1",
        "dup_key[di, q] = n_asking[per_doc[di][q]] > 2",
        ("PR #48 minor: record documents where two asserts ask the queried question."),
    ),
    Mutation(
        "loo: ctrl_same_object never fires",
        "test_object_flags_and_the_donor_object_exclusion",
        "src/rsr/metrics/loo.py",
        'flags["ctrl_same_object"][b] = int(obj[b, s]) == qobj',
        'flags["ctrl_same_object"][b] = False',
        (
            "PR #48 M3: a control carrying the answer object must be flag"
            "ged for W5 to stratify."
        ),
    ),
    Mutation(
        "loo: all_donor_has_object never fires",
        "test_object_flags_and_the_donor_object_exclusion",
        "src/rsr/metrics/loo.py",
        'int(kind[d, s]) == KIND["assert"] and int(obj[d, s]) == qobj',
        "False",
        ("PR #48 M3: a whole-memory donor holding the answer object must be flagged."),
    ),
    # --- retention-readability (experiments/retention-readability/PREREG.md, ab7cbd2) ---
    Mutation(
        "retention-readability: the per-document id assertion dropped",
        "test_document_id_mismatch_raises",
        "experiments/retention-readability/run.py",
        "    if policy.doc_id != doc.doc_id:\n",
        "    if policy.doc_id != doc.doc_id and False:\n",
        "retention-readability PREREG control 3: a policy built for one document must "
        "never run on another (batched OraclePolicy applies row 0's demand to every "
        "row, policy_loop.py:358-365).",
    ),
    Mutation(
        "retention-readability: the oracle-demand check dropped",
        "test_oracle_demand_of_another_document_raises",
        "experiments/retention-readability/run.py",
        "        if policy.inner.demand != own:\n",
        "        if policy.inner.demand != own and False:\n",
        "retention-readability PREREG control 3: the oracle's demand matrix must be "
        "discounted_demand(doc, 0.97) of the document being run.",
    ),
    Mutation(
        "retention-readability: control 2 passes a NaN again",
        "test_control2_fails_on_nan_on_either_or_both_sides",
        "experiments/retention-readability/run.py",
        "            if not (math.isfinite(float(mine)) and "
        "math.isfinite(float(theirs))):",
        "            if False:",
        "retention-readability PREREG control 2: abs(nan - nan) > tol is False, so "
        "without the finiteness guard a NaN on both sides reproduces the reference "
        "and max_abs_diff stays 0.0.",
    ),
    Mutation(
        "retention-readability: rank index read newest-first",
        "test_rank_index_is_M_minus_gap_under_fifo",
        "experiments/retention-readability/run.py",
        '"rank": prefix.index(a) if resident else -1,',
        '"rank": len(prefix) - 1 - prefix.index(a) if resident else -1,',
        "retention-readability PREREG 'Instrument': rank index is the position in the "
        "OLDEST-first prefix (ADR-0006), so under FIFO a full memory puts gap g at "
        "rank M - g.",
    ),
    Mutation(
        "retention-readability: the bootstrap unpaired across arms",
        "test_paired_bootstrap_identical_arms_have_zero_width",
        "experiments/retention-readability/run.py",
        '    rep = {a: torch.einsum("rd,dbk->rbk", W, x) for a, x in sums.items()}\n',
        "    rep = {\n"
        "        a: torch.einsum(\n"
        '            "rd,dbk->rbk", W[torch.randperm(n_boot, generator=g)], x\n'
        "        )\n"
        "        for a, x in sums.items()\n"
        "    }\n",
        "retention-readability PREREG: one resampled multiset of documents per "
        "replicate, used for every arm (paired). Unpaired replicates widen every "
        "difference's CI.",
    ),
    Mutation(
        "retention-readability: PENALTY on any seed",
        "test_classify_a_table",
        "experiments/retention-readability/run.py",
        '    if all(v["point"] >= RP_MAX and v["lo"] > 0 for v in per_seed.values()):\n',
        '    if any(v["point"] >= RP_MAX and v["lo"] > 0 for v in per_seed.values()):\n',
        "retention-readability PREREG (a): PENALTY needs point >= 0.03 and CI lower "
        "bound > 0 on EVERY seed.",
    ),
    Mutation(
        "retention-readability: control 1's tolerance loosened",
        "test_control1_tolerance_is_exact",
        "experiments/retention-readability/run.py",
        "CONTROL1_TOL = 0.0\n",
        "CONTROL1_TOL = 1e-6\n",
        "retention-readability PREREG control 1: the harness's FIFO NLL equals "
        "answer_readout's at B = 1 EXACTLY (max |diff| == 0.0).",
    ),
    Mutation(
        "retention-readability: the arm A null ignored",
        "test_decide_a_null_failure_is_inconclusive",
        "experiments/retention-readability/run.py",
        "        if not a_null_ok(null):\n",
        "        if False and not a_null_ok(null):\n",
        "retention-readability PREREG control 6: arm A's oracle - FIFO CI outside "
        "(-0.03, +0.03) voids the arm-B readouts (inconclusive).",
    ),
    # ----------------------------------------------------------------------- #
    # liveness-wiring (docs/lab-notes/dispatch-liveness-wiring.md): every run
    # measures its memory's liveness; INERT exits 5 and is quarantined.
    # ----------------------------------------------------------------------- #
    Mutation(
        "liveness-wiring: the liveness hook skipped",
        "test_every_train_ends_with_the_liveness_measurement",
        "src/rsr/train/loop.py",
        "    liveness = _measure_liveness(model, init_state, cfg, frozen, "
        "device=device)\n",
        '    liveness = {"band": "live"}\n',
        "Bar 2: the falsifier itself -- a run that never measures and reports live "
        "exits 0 with nothing behind it, so an inert run is again indistinguishable "
        "from a live one.",
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_HOOK_COUPLING)
            for t in _LIVENESS_HOOK_READERS
        ),
    ),
    Mutation(
        "liveness-wiring: an inert run reported as 0",
        "test_an_inert_ratio_is_inert_and_exits_5",
        "src/rsr/train/loop.py",
        '    "inert": Exit.INERT,\n',
        '    "inert": Exit.OK,\n',
        "R-2026-09-22-inert-exit-5: an inert run exiting 0 is consumed downstream "
        "like a live one -- the state before this brief.",
        off_gate_allowed=tuple(
            (f"tests/test_liveness_wiring.py::{t}", _LIVENESS_MAPPING_COUPLING)
            for t in (
                "test_main_exits_on_the_liveness_band[inert-5]",
                "test_an_inert_run_quarantines_its_checkpoints",
            )
        ),
    ),
    Mutation(
        "liveness-wiring: the decoy pointed at the trained model",
        "test_the_decoy_is_the_untrained_model_at_the_same_seed",
        "src/rsr/train/loop.py",
        "        decoy.load_state_dict(init_state)\n",
        "        decoy.load_state_dict(model.state_dict())\n",
        "decisive PREREG *Mutation bar* (decoy aliasing): A_decoy == A_trained, "
        "ratio == 1.0 exactly, which the rule makes invalid (3), never 0.",
    ),
    Mutation(
        "liveness-wiring: an inconclusive ratio rounded to inert",
        "test_an_inconclusive_ratio_exits_1_never_0_or_5",
        "src/rsr/metrics/memory_liveness.py",
        '    return "inconclusive", f"{INERT_MAX_RATIO} < ratio',
        '    return "inert", f"{INERT_MAX_RATIO} < ratio',
        "decisive PREREG :45: the band between is 'reported as such, not rounded "
        "to either outcome'. Folding it into inert is the ruling's own erratum.",
        off_gate_allowed=(
            (
                "tests/test_fresh_stream.py::"
                "test_resume_continues_the_stream_at_the_resumed_step",
                "that test's tiny stream run measures inconclusive (ratio 0.0506, "
                "read from its footer 2026-09-26); rounded to inert, train() "
                "quarantines the checkpoint the test resumes from, so the resume "
                "cannot find it. The quarantine, seen from a checkpoint consumer.",
            ),
        ),
    ),
    Mutation(
        "liveness-wiring: an inconclusive run exits 0",
        "test_an_inconclusive_ratio_exits_1_never_0_or_5",
        "src/rsr/train/loop.py",
        '    "inconclusive": Exit.FAIL,\n',
        '    "inconclusive": Exit.OK,\n',
        "R-2026-09-22-inert-exit-5: inconclusive is 1, liveness not demonstrated; "
        "0 would report an undemonstrated memory as live.",
        off_gate_allowed=(
            (
                "tests/test_liveness_wiring.py::"
                "test_main_exits_on_the_liveness_band[inconclusive-1]",
                _LIVENESS_MAPPING_COUPLING,
            ),
        ),
    ),
    Mutation(
        "liveness-wiring: an inert run's checkpoints left in place",
        "test_an_inert_run_quarantines_its_checkpoints",
        "src/rsr/train/loop.py",
        '    if liveness["band"] == "inert":\n',
        "    if False:\n",
        "R-2026-09-22-inert-checkpoint-quarantine: an inert checkpoint left beside "
        "the live ones is read by any loader, silently.",
        off_gate_allowed=(
            (
                "tests/test_liveness_wiring.py::"
                "test_the_loader_refuses_a_quarantined_checkpoint",
                "it loads the checkpoint the quarantine put under quarantine/; "
                "with no quarantine there is no such file to refuse.",
            ),
        ),
    ),
    Mutation(
        "liveness-wiring: the loader override defaults to allow",
        "test_the_loader_refuses_a_",
        "src/rsr/train/checkpoint.py",
        "    allow_quarantined: bool = False,\n",
        "    allow_quarantined: bool = True,\n",
        "R-2026-09-22-inert-checkpoint-quarantine: 'loaders refuse a quarantined "
        "checkpoint unless explicitly overridden'. A default of allow is no refusal.",
    ),
    Mutation(
        "liveness-wiring: status() refuses 5",
        "test_status_accepts_5_as_inert",
        "src/rsr/exit_codes.py",
        "    try:\n        return Exit(code)\n",
        "    try:\n        if code == 5:\n            raise ValueError\n"
        "        return Exit(code)\n",
        "R-2026-09-22-inert-exit-5: a status() still bounded at 0-4 turns every "
        "INERT run into a ValueError at run_main.",
        off_gate_allowed=(
            (
                "tests/test_liveness_wiring.py::"
                "test_main_exits_on_the_liveness_band[inert-5]",
                "main() returns status(liveness_exit(...)); an INERT band reaches "
                "status(5) there -- the same refusal seen from the entry point.",
            ),
        ),
    ),
    # ----------------------------------------------------------------------- #
    # E0e (experiments/e0e/PREREG.md, Mutation bar): the u_bar distribution and
    # E[lifetime] on a FIFO run.
    # ----------------------------------------------------------------------- #
    Mutation(
        "e0e: r_i read ungated",
        "test_r_i_is_the_gated_rescaled_target",
        "experiments/e0e/run.py",
        "    return retrieval_demand(trace, n_live=n_live, capacity=M, gated=True)",
        "    return retrieval_demand(trace, n_live=n_live, capacity=M, gated=False)",
        "correction 17 / D-E: r_i includes TG's memory gate; ungated, every tau and "
        "u_bar is measured on a different quantity than the one b will act on.",
    ),
    Mutation(
        "e0e: the half-life frozen at v0.4's 16",
        "test_the_half_life_is_a_quarter_of_the_lifetime",
        "experiments/e0e/run.py",
        "    return e_lifetime / HALF_LIFE_DIVISOR",
        "    return 16.0",
        "§3.5 item 2: half-life E[lifetime]/4, 'not the frozen 16 of v0.4'.",
    ),
    Mutation(
        "e0e: lifetime counted over evicted sentences only",
        "test_lifetime_counts_every_written_sentence",
        "experiments/e0e/run.py",
        "    lt = [s.lifetime for s in sentences]\n",
        "    lt = [s.lifetime for s in sentences if s.evicted]\n",
        "PREREG Definitions 1: stream-end survivors are the lifetime b has to act "
        "within (b resets at the boundary). Evicted-only reads M under FIFO, the "
        "§12.2 audit's E[lt] = M, and moves gamma_b.",
        off_gate_allowed=(
            (
                "tests/test_e0e.py::"
                "test_fifo_lifetimes_on_a_tiny_live_model_are_the_analytic_ones",
                "the end-to-end test asserts the same analytic E_lifetime on a real "
                "FIFO stream: the definition, seen from the recorder.",
            ),
        ),
    ),
    Mutation(
        "e0e: the tau quantile moved to the median",
        "test_tau_is_the_75th_percentile_of_the_relative_deviation",
        "experiments/e0e/run.py",
        "TAU_QUANTILE = 0.75\n",
        "TAU_QUANTILE = 0.5\n",
        "PREREG Definitions 4: the 0.75 quantile makes b fire on the 25% of steps "
        "§3.5 item 4's gamma_b derivation assumes. Moving it after data is choosing "
        "a threshold after seeing it.",
    ),
    Mutation(
        "e0e: a non-live seed admitted to the values",
        "test_a_seed_that_is_not_live_is_excluded",
        "experiments/e0e/run.py",
        '    return sorted(s for s, b in bands.items() if b == "live")',
        '    return sorted(s for s, b in bands.items() if b != "invalid")',
        "PREREG precondition: an inert or inconclusive memory's shares are not the "
        "retrieval demand of a memory that retrieves.",
    ),
    Mutation(
        "e0e: a consistency failure ledgered as ok",
        "test_a_consistency_failure_is_ledgered_as_failed_not_ok",
        "experiments/e0e/run.py",
        '        code, status = Exit.FAIL, "failed"',
        '        code, status = Exit.FAIL, "ok"',
        "the exit-1 path wrote status ok: a failure the ledger calls a pass, read "
        "as one by anything that reads the ledger rather than the exit code "
        "(orchestrator/lanes.py refuses only status != ok).",
    ),
    # -- W5: experiments/carry-forward (PREREG + amendment 1) --
    Mutation(
        "carry-forward: the bootstrap resamples targets, not documents",
        "test_cluster_bootstrap_resamples_documents_not_targets",
        "experiments/carry-forward/run.py",
        "uniq, inv = torch.unique(doc, return_inverse=True)",
        "uniq, inv = torch.arange(doc.numel()), torch.arange(doc.numel())",
        "carry-forward PREREG 'bootstrap': per-document cluster bootstrap; targets "
        "of one document are correlated, and resampling them narrows every CI.",
        off_gate_allowed=(
            (
                "tests/test_carry_forward.py::test_cluster_bootstrap_point_is_the_"
                "ratio_of_sums",
                "the same function's n_docs field counts clusters; with targets as "
                "clusters it counts targets.",
            ),
        ),
    ),
    Mutation(
        "carry-forward: CARRY ignores the CI lower bound",
        "test_carry_needs_the_ci_lower_bound_above_zero",
        "experiments/carry-forward/run.py",
        'return bool(ci["point"] >= REACH_DELTA and ci["lo"] == ci["lo"] '
        'and ci["lo"] > 0)',
        'return bool(ci["point"] >= REACH_DELTA)',
        "carry-forward PREREG decision rule: CARRY needs excess >= 0.03 AND its 95% "
        "CI lower bound > 0.",
    ),
    Mutation(
        "carry-forward: MIXED needs two carrying seeds, not one",
        "test_classification_table",
        "experiments/carry-forward/run.py",
        "    elif carriers:\n",
        "    elif len(carriers) >= 2:\n",
        "carry-forward PREREG table: CARRY on one or two seeds is MIXED.",
        off_gate_allowed=(
            (
                "tests/test_carry_forward.py::test_inconclusive_l_without_carry_is_"
                "inconclusive_exit_3",
                "its second case asserts that ONE carrying seed with an "
                "L_INCONCLUSIVE seed is still MIXED (amendment 2: MIXED does not "
                "rest on L) -- the same one-seed MIXED row of the table.",
            ),
        ),
    ),
    Mutation(
        "carry-forward: EXT disjointness forgets the fresh-escape stream",
        "test_disjointness_catches_the_fresh_escape_stream",
        "experiments/carry-forward/run.py",
        '    "fresh_escape": (52160, 148160),\n',
        "    # fresh_escape dropped\n",
        "carry-forward PREREG control 4: EXT is disjoint from fresh-escape's "
        "continuation of the stream [52160, 148160).",
    ),
    Mutation(
        "carry-forward: the reproduction tolerance widened tenfold",
        "test_reproduction_fails_at_2e_6",
        "experiments/carry-forward/run.py",
        "                max_diff = max(max_diff, d)\n                if d > tol:",
        "                max_diff = max(max_diff, d)\n                if d > 10 * tol:",
        "carry-forward PREREG control 2: within 1e-6 absolute.",
    ),
    Mutation(
        "carry-forward: the bit-exact control accepts allclose",
        "test_bitexact_control_refuses_a_one_ulp_difference",
        "experiments/carry-forward/run.py",
        "torch.equal(a.cpu(), b.cpu())",
        "torch.allclose(a.cpu(), b.cpu())",
        "carry-forward PREREG control 3: loo live == answer_readout live under "
        "torch.equal, not a tolerance.",
    ),
    Mutation(
        "carry-forward: L's window includes gap 1",
        "test_l_population_is_gap_2_to_M",
        "experiments/carry-forward/run.py",
        "return (g >= 2) & (g <= M)",
        "return (g >= 1) & (g <= M)",
        "PR #48 review M2 / amendment 1 A2: at gap 1 the bos-copy context is the "
        "assert's own gestalt, so L is read on gap 2..M only.",
    ),
    Mutation(
        "carry-forward: L's denominator is all_slots_zeroed again",
        "test_l_uses_the_like_for_like_denominator",
        "experiments/carry-forward/run.py",
        'out[f"L.{m}"] = ratio_ci(rec, pop, "own_resample", "all_slots_resample", m)',
        'out[f"L.{m}"] = ratio_ci(rec, pop, "own_resample", "all_slots_zeroed", m)',
        "PR #48 review M1 / amendment 1 A1: own_resample pairs with "
        "all_slots_resample, like for like.",
    ),
    Mutation(
        "carry-forward: the reach baseline is all_slots_zeroed again",
        "test_reach_baseline_is_all_slots_resample_on_evicted_targets",
        "experiments/carry-forward/run.py",
        'b["excess.acc"] = drop_ci(rec, pop, "all_slots_resample", "acc")',
        'b["excess.acc"] = drop_ci(rec, pop, "all_slots_zeroed", "acc")',
        "amendment 1 A4: reach is read over all_slots_resample (in distribution).",
    ),
    Mutation(
        "carry-forward: a sensitivity label change is not a disagreement",
        "test_sensitivity_disagreement_threshold",
        "experiments/carry-forward/run.py",
        "return bool(lab_p != lab_s or d != d or d > L_SENSITIVITY_MAX_DIFF)",
        "return bool(d != d or d > L_SENSITIVITY_MAX_DIFF)",
        "carry-forward amendment 2 ruling 3: a label change between primary and "
        "sensitivity L is a material disagreement.",
        off_gate_allowed=(
            (
                "tests/test_carry_forward.py::test_l_full_marks_a_disagreeing_"
                "sensitivity_inconclusive",
                "the same rule observed through l_full: its disagreement is a label "
                "change with the points 1.0 apart only on a subset.",
            ),
        ),
    ),
    Mutation(
        "carry-forward: an inconclusive L no longer blocks a no-carry outcome",
        "test_inconclusive_l_without_carry_is_inconclusive_exit_3",
        "experiments/carry-forward/run.py",
        "if not carriers and any("
        'per_seed[s]["L_label"] == "L_INCONCLUSIVE" for s in SEEDS):',
        'if False and any(per_seed[s]["L_label"] == "L_INCONCLUSIVE" for s in SEEDS):',
        "carry-forward amendment 2 ruling 3: with no seed carrying, an "
        "L_INCONCLUSIVE seed makes the classification inconclusive (exit 3).",
    ),
    # ----------------------------------------------------------------------- #
    # lookahead-room (W10, experiments/lookahead-room/PREREG.md): is there room
    # for 3b (gamma 0.9 vs gamma 0) at M = 16?
    # ----------------------------------------------------------------------- #
    Mutation(
        "lookahead-room: r_i read ungated",
        "test_r_i_is_e0es_gated_rescaled_target",
        "experiments/lookahead-room/run.py",
        "    return retrieval_demand(trace, n_live=n_live, capacity=m, gated=True)",
        "    return retrieval_demand(trace, n_live=n_live, capacity=m, gated=False)",
        "correction 17 / D-E: the gamma = 0 and gamma > 0 targets are both built "
        "from r_i; ungated, room_3b compares targets RSR would never train on.",
    ),
    Mutation(
        "lookahead-room: every probe reads the resident rank-0 sentence",
        "test_a_probe_reads_the_swapped_in_sentence_not_the_resident_one",
        "experiments/lookahead-room/run.py",
        "            kvP[p, 0] = self.gest[i]",
        "            kvP[p, 0] = self.gest[cand[0]]",
        "PREREG demand-if-resident: a probe that never swaps the sentence in gives "
        "every evicted sentence the rank-0 resident's demand -- a target that "
        "cannot tell a pending fact from filler by construction.",
    ),
    Mutation(
        "lookahead-room: the return drops its discount",
        "test_the_discounted_return_sums_the_future_within_the_stream",
        "experiments/lookahead-room/run.py",
        "        acc = row + gamma * acc",
        "        acc = row + acc",
        "§3.4 / correction 2: G = sum gamma^k r(t+k). Undiscounted, 0.9 and 0.97 "
        "become the same horizon-to-stream-end sum and room_3b measures neither.",
        off_gate_allowed=(
            (
                "tests/test_lookahead_room.py::"
                "test_the_literal_target_is_zero_where_fifo_did_not_hold_the_sentence",
                "targets() builds rule_g09 through discounted_returns; that test "
                "pins rule_g09 = 0.4 + 0.9 * 0.7 on the same matrix -- the return, "
                "seen from the target table.",
            ),
        ),
    ),
    Mutation(
        "lookahead-room: target-rule ties go to the newest slot",
        "test_a_target_rule_evicts_the_argmin_and_ties_go_to_the_oldest",
        "experiments/lookahead-room/run.py",
        "            if v < best_v:",
        "            if v <= best_v:",
        "PREREG target rules: ties to the lowest slot (the oldest), as OraclePolicy. "
        "Ties to the newest turn every all-zero literal target into evict-newest "
        "and every tie into an age preference nobody registered.",
    ),
    Mutation(
        "lookahead-room: literal target keeps the probed demand",
        "test_the_literal_target_is_zero_where_fifo_did_not_hold_the_sentence",
        "experiments/lookahead-room/run.py",
        "        resident, torch.nan_to_num(D, nan=0.0), torch.zeros_like(D)",
        "        resident | True, torch.nan_to_num(D, nan=0.0), torch.zeros_like(D)",
        "PREREG secondary variant: the literal FIFO-rollout target is zero where "
        "FIFO did not hold the sentence; keeping the probe makes it the primary "
        "and the caveat it exists to size disappears.",
    ),
    Mutation(
        "lookahead-room: NO_ROOM read from the point, not the CI",
        "test_the_rule_is_the_prereg_rule",
        "experiments/lookahead-room/run.py",
        '    if all(v["hi"] < NO_ROOM_MAX for v in per_seed.values()):',
        '    if all(v["point"] < NO_ROOM_MAX for v in per_seed.values()):',
        "PREREG decision rule: NO_ROOM needs the CI upper bound below 0.02 on "
        "every seed; a point estimate under 0.02 with a wide CI is PARTIAL.",
    ),
    Mutation(
        "lookahead-room: a failed control still exits 0",
        "test_a_failed_control_makes_the_run_inconclusive",
        "experiments/lookahead-room/run.py",
        '        "exit": int(Exit.DID_NOT_RUN if failed else Exit.OK),',
        '        "exit": int(Exit.OK),',
        "PREREG: any control failure is INCONCLUSIVE, exit 3. 'Did not run' and "
        "'found something' are different facts.",
    ),
    Mutation(
        "lookahead-room: C1 compared in float64 again",
        "test_c1_passes_on_float32_exact_accuracy_that_float64_misses",
        "experiments/lookahead-room/run.py",
        "                if float(okb[msk].float().mean()) != float(theirs):",
        "                if float(ok64[msk].mean()) != float(theirs):",
        "PREREG Amendment 1: the reference ledger stores float32 means, so a float64 "
        "comparison fails C1 whatever the rollout did -- run 1's INCONCLUSIVE.",
    ),
    Mutation(
        "lookahead-room: C1 ignores argmax mismatches",
        "test_c1_fails_on_one_argmax_mismatch_or_a_wrong_accuracy",
        "experiments/lookahead-room/run.py",
        '        "ok": not bad and not missing and mism == 0 and missing_keys == 0,',
        '        "ok": not bad and not missing,',
        "PREREG Amendment 1: argmax identity is half of the amended C1; bucket means "
        "can agree while individual answers flip in opposite directions.",
    ),
    Mutation(
        "the battery drops the failure reason",
        "test_a_mutation_row_carries_the_failure_reasons",
        "scripts/mutation_battery.py",
        # 📌 split so this literal is not itself the first match in this file.
        # Re-anchored 2026-09-29 (I1): the row is built in `_row`, eight columns
        # shallower. Same edit.
        '        "failure_reasons": {f: reasons[f] ' + "for f in sorted(reasons)},",
        '        "failure_reasons": dict.fromkeys(' + "sorted(reasons)),",
        "I5: the 09-26 dispatch red left only a node id because the battery ran "
        "--tb=no. A row whose reasons are all null is that blindness back.",
    ),
    Mutation(
        "the stub slot writes its job record in place again",
        "test_the_stub_slot_never_exposes_a_half_written_job_record",
        "tests/_orch_loop_helpers.py",
        "    tmp.write_text(json.dumps(rec))\n    os.replace(tmp, path)\n",
        "    path.write_text(json.dumps(rec))\n",
        "I5b: write_text truncates then writes, and wait_job polls json.loads("
        "read_text()) every 50 ms -- a demonstrated torn read (not established as "
        "the cause of the 09-26 dispatch red).",
    ),
    Mutation(
        "b0-ceilings: kind-oracle ties go to the oldest",
        "test_random_ties_use_b2s_a1_10_generator_exactly",
        "experiments/b0-ceilings/run.py",
        '    return random.Random(f"ko:{seed}:{doc_id}:{t}").choice(ks)',
        "    return ks[0]",
        "B0 PREREG §4 / B2 A1.10: an age tie-break gives the kind-oracle age "
        "information psi_hat is barred from; B0's kind-oracle must break ties with "
        "B2's own random draw, or its ceiling describes a different comparator.",
        off_gate_allowed=(
            (
                "tests/test_b0_ceilings.py::"
                "test_the_kind_oracle_evicts_the_lowest_class_and_ties_randomly",
                "the tie rule is asserted twice on purpose: once on the bare draw, once "
                "through the kind-oracle policy on a real document, so that a policy "
                "that bypassed tie_break would be caught. One edit to the draw must "
                "redden both.",
            ),
        ),
    ),
    Mutation(
        "b0-ceilings: the T0 manifest check ignores a changed D.pt",
        "test_the_manifest_check_catches_a_changed_or_unlisted_file",
        "experiments/b0-ceilings/run.py",
        '    ok = all(v["want"] is not None and v["got"] == v["want"] for v in '
        "files.values())",
        '    ok = all(v["want"] is not None for v in files.values())',
        "B0 PREREG §6 C1 / PLAN-v4 T0: every later run re-checks the substrate "
        "manifest; a check that only asks whether the file is listed would pass a "
        "D.pt rewritten in place.",
    ),
    Mutation(
        "b0-ceilings: class-mean rows include under-full steps",
        "test_rows_are_full_memory_steps_and_past_sentences_only",
        "experiments/b0-ceilings/run.py",
        "    return (t >= m) & (i < t)",
        "    return (t >= 1) & (i < t)",
        "B0 PREREG §3: rows are full-memory steps t >= M (B2 §6's kind-oracle rows); "
        "under-full steps decide no eviction and would shift every class mean.",
        off_gate_allowed=(
            (
                "tests/test_b0_ceilings.py::"
                "test_class_means_pool_rows_by_kind_band_and_age",
                "the class means are computed over row_mask's rows; the hand-computed "
                "means use t >= 16, so widening the rows must change them. The "
                "coupling is the point: the means are only as right as their rows.",
            ),
        ),
    ),
    Mutation(
        "b0-ceilings: C3 tolerates a float difference",
        "test_c3_is_exact_equality_against_the_reference_ledger",
        "experiments/b0-ceilings/run.py",
        "            if got is None or float(got) != float(theirs):",
        "            if got is None or abs(float(got) - float(theirs)) > 1e-9:",
        "B0 PREREG §6 C3: the reproduction of lookahead-room-r2's U.hit keys is exact "
        "(same tensors, same documents, same simulate); a tolerance would hide a "
        "different reading of D.pt.",
    ),
    # ------------------------------------------------------------------ I1
    # P1.1 (2026-09-29): battery isolation. Each guard in
    # scripts/battery_isolation.py and in this file's shard wiring, disabled.
    # 📌 anchors in THIS file are split so the table's own text never matches.
    Mutation(
        "i1: the battery mutates the invoking tree again",
        "test_verdicts_come_from_the_shard_and_the_live_tree_is_untouched",
        "scripts/mutation_battery.py",
        "            original = apply(m, " + "shard.root)",
        "            original = apply(" + "m)",
        "I1: the live checkout the battery is invoked from is never mutated; a "
        "stopped battery left mutated loop.py and lanes.py behind twice.",
        off_gate_allowed=tuple(
            (n, _I1_SHARD_MUTATION)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_check_still_fails_on_an_unproven_gate",
                "tests/test_battery_isolation.py::"
                "test_a_mutation_whose_suite_reports_no_probe_did_not_run",
                "tests/test_battery_isolation.py::"
                "test_a_stopped_battery_leaves_the_live_tree_and_git_clean[SIGTERM]",
                "tests/test_battery_isolation.py::"
                "test_a_stopped_battery_leaves_the_live_tree_and_git_clean[SIGHUP]",
                "tests/test_battery_isolation.py::"
                "test_a_sigkilled_battery_leaves_only_shards_and_prune_removes_them",
                "tests/test_battery_isolation.py::"
                "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
            )
        )
        + _i1_declare(
            _I1_SHARD_MUTATION,
            "test_markdown_is_not_written_while_a_row_did_not_run",
            "test_markdown_is_written_when_every_row_ran",
            "test_an_ignored_file_one_mutation_writes_cannot_change_the_next_verdict",
            "test_bytecode_from_one_mutation_cannot_run_under_the_next",
            "test_a_stopped_battery_leaves_the_live_tree_and_git_clean[SIGINT]",
            "test_a_stopped_battery_kills_its_suites_grandchildren",
            "test_a_sigint_to_the_pytest_child_alone_is_did_not_run",
        ),
    ),
    Mutation(
        "i1: the path assertion accepts any rsr",
        "test_the_path_assertion_requires_rsr_under_the_shard",
        "scripts/battery_isolation.py",
        "        if not is_under(probe.get(key), root):",
        "        if False:",
        "I1: rsr must resolve under the shard or the suite DID NOT RUN; the 09-27 "
        "suite ran on the main checkout's venv (DIGEST cycle 16b).",
        off_gate_allowed=tuple(
            (n, _I1_E2E_WIRING)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_a_suite_on_another_checkouts_venv_did_not_run",
            )
        ),
    ),
    Mutation(
        "i1: the path assertion skips the pytest child",
        "test_the_path_assertion_requires_rsr_under_the_shard",
        "scripts/battery_isolation.py",
        '    for key in ("in_process", "child"):',
        '    for key in ("in_process",):',
        "I1: a test's sys.path.insert can put the shard's src first in the pytest "
        "process while every child it spawns imports another checkout's rsr.",
        off_gate_allowed=tuple(
            (f"tests/test_battery_isolation.py::{t}", _I1_E2E_WIRING)
            for t in ("test_a_suite_on_another_checkouts_venv_did_not_run",)
        ),
    ),
    Mutation(
        "i1: a missing probe is accepted",
        "test_the_path_assertion_requires_rsr_under_the_shard",
        "scripts/battery_isolation.py",
        '        return "no probe: the suite did not report where it imported rsr from"',
        "        return None",
        "I1: a suite that says nothing about where it imported rsr from has not "
        "shown that it ran in the shard.",
        off_gate_allowed=tuple(
            (n, _I1_E2E_WIRING)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_a_mutation_whose_suite_reports_no_probe_did_not_run",
            )
        )
        + _i1_declare(
            _I1_E2E_WIRING,
            "test_a_completed_suite_is_judged_by_its_probe",
            "test_a_baseline_that_did_not_run_exits_3",
            "test_markdown_is_not_written_while_a_row_did_not_run",
        ),
    ),
    Mutation(
        "i1: run_suite never reads the probe",
        "test_a_suite_on_another_checkouts_venv_did_not_run",
        "scripts/mutation_battery.py",
        # 📌 Re-anchored for the review fixes: run_suite consults suite_problem
        # (completion AND probe) now. Same edit, and it disables every layer at
        # once, so it is also the mutation that gates the end-to-end tests.
        "    problem = iso.suite_problem(proc.returncode, "
        + "iso.read_probe(shard.probe), shard.root)",
        "    problem = None",
        "I1: the path assertion and the completion check exist only if the battery "
        "consults them.",
        off_gate_allowed=tuple(
            (n, _I1_E2E_WIRING)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_a_mutation_whose_suite_reports_no_probe_did_not_run",
            )
        )
        + _i1_declare(
            _I1_E2E_WIRING,
            "test_a_killed_pytest_is_did_not_run_not_a_failure",
            "test_a_baseline_that_did_not_run_exits_3",
            "test_markdown_is_not_written_while_a_row_did_not_run",
            "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
            "test_a_sigint_to_the_pytest_child_alone_is_did_not_run",
        ),
    ),
    Mutation(
        "i1: a killed pytest is scored as a verdict",
        "test_a_killed_pytest_is_did_not_run_not_a_failure",
        # 📌 Re-anchored for the review fixes: the check moved into
        # battery_isolation.suite_problem. Same edit.
        "scripts/battery_isolation.py",
        "    if returncode < 0:",
        "    if False:",
        "I1: a pytest killed by a signal became a '<collection/exit -9>' failure and "
        "was scored LEAKS or ADDS NOTHING; it is DID NOT RUN.",
        off_gate_allowed=tuple(
            (n, _I1_E2E_WIRING)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
            )
        ),
    ),
    Mutation(
        "i1: a DID_NOT_RUN row exits 0",
        "test_a_mutation_whose_suite_reports_no_probe_did_not_run",
        "scripts/mutation_battery.py",
        '    if any(r["verdict"] == "DID_NOT_' + 'RUN" for r in rows):',
        "    if False:",
        "I1 / exit-code protocol: a mutation that did not run is not a table entry; "
        "'did not run' never becomes exit 0.",
        off_gate_allowed=tuple(
            (n, _I1_E2E_WIRING)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
            )
        )
        + _i1_declare(
            _I1_E2E_WIRING,
            "test_markdown_is_not_written_while_a_row_did_not_run",
            "test_a_sigint_to_the_pytest_child_alone_is_did_not_run",
        ),
    ),
    Mutation(
        "i1: RSR_ORCH_ROOT is not the shard's",
        "test_the_suite_env_is_the_shards_own",
        "scripts/battery_isolation.py",
        '    env["RSR_ORCH_ROOT"] = str(shard.root)\n',
        "    pass\n",
        "I1: without it an orchestrator write from a test resolves the MAIN "
        "checkout (lanes.orch_root), the stray-write class of 2026-09-27.",
    ),
    Mutation(
        "i1: a dirty invoking tree is battery'd at HEAD",
        "test_a_dirty_invoking_tree_is_refused",
        "scripts/battery_isolation.py",
        '    if dirty.strip():\n        raise Unisolated(\n            f"{root} has',
        '    if False:\n        raise Unisolated(\n            f"{root} has',
        "I1: a shard is a commit; uncommitted edits in the invoking tree would not "
        "be in it, and the record would certify a tree the battery never ran.",
        off_gate_allowed=tuple(
            (n, _I1_DIRTY)
            for n in (
                "tests/test_battery_isolation.py::"
                "test_an_untracked_file_in_the_invoking_tree_is_refused",
            )
        ),
    ),
    Mutation(
        "i1: a shard pool inside the invoking tree is accepted",
        "test_a_pool_inside_the_invoking_tree_is_refused",
        "scripts/battery_isolation.py",
        "    if is_under(pool, source):",
        "    if False:",
        "I1: shards live outside the invoking tree, so no path under it is written.",
    ),
    Mutation(
        "i1: the shard reset leaves untracked files",
        "test_reset_restores_the_pinned_tree",
        "scripts/battery_isolation.py",
        # 📌 Re-anchored for the review fixes: the clean is CLEAN_ARGV now.
        '    _ok(git(shard.root, *CLEAN_ARGV), f"git clean in {shard.root}")\n',
        "    pass\n",
        "I1: a file one mutated suite created must not be in the next one's tree.",
        off_gate_allowed=_i1_declare(
            _I1_RESET_E2E,
            "test_an_ignored_file_left_in_a_shard_is_not_clean",
            "test_verdicts_come_from_the_shard_and_the_live_tree_is_untouched",
            "test_check_still_fails_on_an_unproven_gate",
            "test_an_ignored_file_one_mutation_writes_cannot_change_the_next_verdict",
            "test_bytecode_from_one_mutation_cannot_run_under_the_next",
        ),
    ),
    Mutation(
        "i1: a shard suite writes bytecode again",
        "test_a_shard_suite_writes_no_bytecode",
        "scripts/battery_isolation.py",
        '    env["PYTHONDONTWRITEBYTECODE"] = "1"\n',
        "    pass\n",
        "I1: a same-size mutation restored within one second passes the .pyc "
        "mtime+size check, so the next suite ran the previous mutation's code "
        "(observed on the stub battery, 2026-09-29).",
    ),
    Mutation(
        "i1: the reset keeps __pycache__",
        "test_reset_purges_bytecode_outside_the_venv",
        "scripts/battery_isolation.py",
        '            shutil.rmtree(Path(dirpath) / "__pycache__", ignore_errors=True)\n',
        "            pass\n",
        "I1: the second layer under PYTHONDONTWRITEBYTECODE, for caches written by "
        "anything that sets its own env.",
    ),
    Mutation(
        "i1: the shard interpreter is not checked",
        "test_a_shard_venv_that_imports_another_checkout_is_refused_before_any_suite",
        "scripts/battery_isolation.py",
        "    check_interpreter(shard, shard_env(shard, dict(os.environ)))\n",
        "    pass\n",
        "I1: an editable install pointing at another checkout is refused before the "
        "first (multi-minute) suite, not after it.",
    ),
    Mutation(
        "i1: the shard pool is never torn down",
        "test_a_stopped_battery_leaves_the_live_tree_and_git_clean",
        "scripts/battery_isolation.py",
        "            if not keep:",
        "            if keep:",
        "I1: a finished or stopped battery leaves no worktree registration behind.",
        off_gate_allowed=tuple(
            (n, _I1_TEARDOWN)
            for n in (
                "tests/test_battery_isolation.py::test_reset_restores_the_pinned_tree",
                "tests/test_battery_isolation.py::"
                "test_verdicts_come_from_the_shard_and_the_live_tree_is_untouched",
                "tests/test_battery_isolation.py::"
                "test_a_suite_on_another_checkouts_venv_did_not_run",
                "tests/test_battery_isolation.py::"
                "test_a_shard_venv_that_imports_another_checkout_is_refused_before_any_suite",
                "tests/test_battery_isolation.py::"
                "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
                "tests/test_battery_isolation.py::"
                "test_a_kept_shard_is_reused_at_the_new_pinned_sha",
                "tests/test_battery_isolation.py::test_two_batteries_never_share_a_pool",
            )
        )
        + _i1_declare(
            _I1_TEARDOWN,
            "test_prune_refuses_while_a_battery_holds_the_pool",
            "test_a_baseline_that_did_not_run_exits_3",
            "test_a_sigint_to_the_pytest_child_alone_is_did_not_run",
        ),
    ),
    Mutation(
        "i1: two batteries share a shard pool",
        "test_two_batteries_never_share_a_pool",
        "scripts/battery_isolation.py",
        "        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)",
        "        pass",
        "I1: a shard is reset between mutations; a second battery in the same pool "
        "would reset or mutate the tree under the first one's suite.",
        off_gate_allowed=_i1_declare(
            _I1_POOL_LOCK,
            "test_prune_refuses_while_a_battery_holds_the_pool",
        ),
    ),
    Mutation(
        "i1: a foreign directory in the pool is taken over",
        "test_a_directory_in_the_pool_that_is_not_our_worktree_is_refused",
        "scripts/battery_isolation.py",
        "        if path not in _registered(source):",
        "        if False:",
        "I1: the pool only ever resets worktrees of this repository; anything else "
        "at a shard path is refused, never checked out over or cleaned.",
    ),
    # ------------------------------------------------ I1 review fixes (09-29)
    Mutation(
        "i1: an incomplete pytest run is scored",
        "test_a_suite_that_did_not_complete_did_not_run",
        "scripts/battery_isolation.py",
        "    if returncode not in COMPLETED_EXIT:",
        "    if False:",
        "Review MAJOR-1: pytest exiting 2 (SIGINT), 3 or 4 did not run the suite to "
        "the end; it was scored, and an interrupt after the gate failed is a "
        "false PROVEN.",
    ),
    Mutation(
        "i1: pytest's own exit status is not checked",
        "test_the_path_assertion_requires_rsr_under_the_shard",
        "scripts/battery_isolation.py",
        '    if probe.get("exitstatus") not in COMPLETED_EXIT:',
        "    if False:",
        "Review MAJOR-1: the probe's exitstatus is pytest's own verdict on whether "
        "the session completed, independent of how the process ended.",
    ),
    Mutation(
        "i1: the reset keeps ignored files",
        "test_an_ignored_file_one_mutation_writes_cannot_change_the_next_verdict",
        "scripts/battery_isolation.py",
        'CLEAN_ARGV = ("clean", "-ffdxq", "-e", "/.venv/")',
        'CLEAN_ARGV = ("clean", "-ffdq", "-e", "/.venv/")',
        "Review MAJOR-2: an ignored runs/ or .orchestrator/ file one mutation's "
        "suite wrote changed the next mutation's verdict.",
        off_gate_allowed=_i1_declare(
            _I1_RESET_E2E,
            "test_a_kept_shard_is_reused_at_the_new_pinned_sha",
            "test_an_ignored_file_left_in_a_shard_is_not_clean",
            "test_verdicts_come_from_the_shard_and_the_live_tree_is_untouched",
            "test_check_still_fails_on_an_unproven_gate",
            "test_bytecode_from_one_mutation_cannot_run_under_the_next",
        ),
    ),
    Mutation(
        "i1: the clean check cannot see ignored files",
        "test_an_ignored_file_left_in_a_shard_is_not_clean",
        "scripts/battery_isolation.py",
        '        git(shard.root, "status", "--porcelain", "--ignored"),',
        '        git(shard.root, "status", "--porcelain"),',
        "Review MAJOR-2: `verify_clean` claimed a pinned tree while ignored files "
        "from a previous suite were in it.",
    ),
    Mutation(
        "i1: a reused shard keeps ignored files",
        "test_a_kept_shard_is_reused_at_the_new_pinned_sha",
        "scripts/battery_isolation.py",
        '        _ok(git(path, *CLEAN_ARGV), f"git clean in {path}")',
        '        _ok(git(path, "clean", "-fdq"), f"git clean in {path}")',
        "Review MAJOR-2: a --keep-shards pool carries one battery's ignored state "
        "into the next battery.",
    ),
    Mutation(
        "i1: --prune-shards takes no lock",
        "test_prune_refuses_while_a_battery_holds_the_pool",
        "scripts/battery_isolation.py",
        "    with pool_lock(pool):\n        return _prune_unlocked(source, pool)",
        "    return _prune_unlocked(source, pool)",
        "Review MAJOR-3: the default pool is shared by every worktree; an unlocked "
        "prune removes another battery's live shard mid-suite.",
    ),
    Mutation(
        "i1: the suite inherits PYTHONPATH",
        "test_the_suite_never_inherits_pythonpath_or_a_pycache_prefix",
        "scripts/battery_isolation.py",
        '_SUITE_STRIP = ("PYTHONPATH", "PYTHONPYCACHEPREFIX")',
        "_SUITE_STRIP = ()",
        "Review MINOR-1: another checkout's scripts/ via PYTHONPATH passes the rsr "
        "probe; a pycache prefix escapes the reset.",
    ),
    Mutation(
        "i1: the suite's process group outlives it",
        "test_a_stopped_battery_kills_its_suites_grandchildren",
        "scripts/mutation_battery.py",
        "    finally:\n        _kill_" + "group(proc)\n",
        "    finally:\n        proc.kill()\n        proc.wait()\n",
        "Review MINOR-2: a grandchild of the suite kept writing into the shard under "
        "the next suite or the teardown.",
        off_gate_allowed=_i1_declare(
            _I1_GROUP,
            "test_a_killed_pytest_child_is_did_not_run_and_the_tree_stays_clean",
            "test_a_sigint_to_the_pytest_child_alone_is_did_not_run",
        ),
    ),
    Mutation(
        "i1: SIGINT is not handled",
        "test_a_stopped_battery_leaves_the_live_tree_and_git_clean[SIGINT]",
        "scripts/mutation_battery.py",
        "_HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGHUP, " + "signal.SIGINT)",
        "_HANDLED_SIGNALS = (signal.SIGTERM, " + "signal.SIGHUP)",
        "Review MINOR-3: a Ctrl-C exited with a traceback, and a second Ctrl-C cut "
        "the teardown short.",
    ),
    Mutation(
        "i1: --markdown records DID_NOT_RUN rows",
        "test_markdown_is_not_written_while_a_row_did_not_run",
        "scripts/mutation_battery.py",
        "    if args.markdown and " + "did_not_run:",
        "    if False:",
        "Review MINOR-4: the committed table records verdicts; a mutation that did "
        "not run has none.",
    ),
    Mutation(
        "i1: a baseline that did not run exits 1",
        "test_a_baseline_that_did_not_run_exits_3",
        "scripts/mutation_battery.py",
        "            refuse(Exit.DID_NOT_RUN, "
        + 'f"the baseline suite did not run: {e}")',
        '            refuse(Exit.FAIL, f"the baseline suite did not ' + 'run: {e}")',
        "Review MINOR-8: nothing was mutated, so nothing was tested: 3, not 1.",
        off_gate_allowed=_i1_declare(
            _I1_E2E_WIRING,
            "test_a_suite_on_another_checkouts_venv_did_not_run",
        ),
    ),
    Mutation(
        "i1: battery_subset drops --shard-dir",
        "test_battery_subset_passes_the_shard_dir_through",
        "scripts/battery_subset.py",
        '        extra, rest = ["--shard-dir", rest[1]], rest[2:]',
        "        rest = rest[2:]",
        "Review MINOR-7 / pool policy: a subset beside another battery needs its "
        "own pool, or it exits 3.",
    ),
    Mutation(
        "i1: no SIGTERM/SIGHUP handler",
        "test_a_stopped_battery_leaves_the_live_tree_and_git_clean",
        "scripts/mutation_battery.py",
        "    old = {s: signal.signal(s, _raise_"
        + "signalled) for s in _HANDLED_SIGNALS}",
        "    old = {}",
        "I1 defence in depth: a stopped battery kills its pytest child and tears its "
        "shards down; without the handler both outlive it.",
        off_gate_allowed=_i1_declare(
            _I1_E2E_WIRING,
            "test_a_stopped_battery_kills_its_suites_grandchildren",
        ),
    ),
    Mutation(
        "on_write is told the victim index again",
        "test_on_write_receives_the_newcomers_slot_after_an_eviction",
        "src/rsr/model/tg/policy_loop.py",
        "                    policy.on_write(memory_state(mem, t, row), slot, t)",
        "                    policy.on_write(memory_state(mem, t, row), "
        "int(victim[row]), t)",
        "2026-09-29, gauntlet 0.4 one level down: `write_at` compacts behind the "
        "victim and writes the newcomer at M-1 (§3.1, ADR-0006), so after the "
        "write the victim index names a different occupant -- and on an underfull "
        "row it is the placeholder 0. Any policy keying state on `slot` (LRU, "
        "H2O, E1's per-slot baselines) would update the wrong slot. The loop's "
        "own written_at guard checks `slot`, not the argument actually passed, so "
        "it cannot catch this mutation; only the recording-policy test does.",
        off_gate_allowed=(
            (
                "tests/test_on_write_slot.py::"
                "test_on_write_receives_the_newly_filled_slot_on_a_non_full_row",
                "the same call site: the old call passed the placeholder 0 on an "
                "underfull row, which is the other half of the defect this gate "
                "names",
            ),
        ),
    ),
    # T5(b), 2026-09-30: the review gate (docs/review-records.md). Proven by hand
    # at authoring time against tests/test_orch_merge*.py only (battery lane busy).
    Mutation(
        "merge: the review gate is skipped",
        "test_merge_is_refused_without_a_review_record",
        "scripts/orchestrator/merge.py",
        "    code, record = review_gate(root, branch, base, head_sha, run_item)\n"
        "    if code != Exit.OK:\n",
        "    code, record = review_gate(root, branch, base, head_sha, run_item)\n"
        "    if False:\n",
        "PLAN-v4 §4 T5(b) stops being a mechanism: any verified run/ branch, and "
        "any fix/ eng/ feat/ docs/ branch at all, merges into the night branch with "
        "no adversarial review on record.",
        off_gate_allowed=tuple(
            (f"tests/test_orch_merge_review.py::{t}", _REVIEW_GATE_COUPLING)
            for t in _REVIEW_GATE_REFUSALS
        ),
    ),
    Mutation(
        "merge: a stale review record is accepted",
        "test_a_record_older_than_head_with_a_code_change_since_is_refused",
        "scripts/orchestrator/merge.py",
        "    unreviewed = [\n"
        "        p for p in diff.stdout.splitlines() if p and not review_exempt(p, "
        "run_item)\n"
        "    ]\n",
        "    unreviewed = []\n",
        "a review of an early head waves through every commit made after it -- "
        "the census shape of 2026-09-29 (eng/i1-isolation reviewed at ff6dccc, "
        "7 files changed since).",
        off_gate_allowed=tuple(
            (
                f"tests/test_orch_merge_review.py::{t}",
                "a forged verification.json is refused by the same staleness diff; "
                "no diff, no refusal. One check, two path classes.",
            )
            for t in (
                "test_only_the_merging_runs_own_verification_is_exempt",
                "test_no_verification_json_is_exempt_on_other_prefixes",
            )
        ),
    ),
    # T5(b) review fixes, 2026-09-30 (docs/reviews/eng-t5b-review-gate/
    # bbfa38c9bade.md). Proven by hand, targeted files only (B5 holds the CPU).
    Mutation(
        "merge: MERGE WITH FIXES merges on its own",
        "test_merge_with_fixes_is_never_mergeable_on_its_own",
        "scripts/orchestrator/merge.py",
        '    if fm["verdict"] == "MERGE WITH FIXES":\n'
        "        # Review MAJOR-2, PM decision",
        "    if False:\n        # Review MAJOR-2, PM decision",
        "review MAJOR-2 (probe P3): the author commits a 'fix', edits the "
        "reviewer's record to attest it, and the unreviewed fix merges.",
        off_gate_allowed=tuple(
            (
                f"tests/test_orch_merge_review.py::{t}",
                "every MERGE WITH FIXES record is refused by this one branch; "
                "without it each falls through to the MERGE path.",
            )
            for t in (
                "test_merge_with_fixes_without_fixes_verified_at_is_refused",
                "test_merge_with_fixes_with_code_after_the_fix_check_is_refused",
                "test_merge_with_fixes_verified_at_not_an_ancestor_is_refused",
            )
        ),
    ),
    Mutation(
        "merge: fixes_verified_at is not validated",
        "test_fixes_verified_at_must_be_a_full_sha_of_a_real_commit",
        "scripts/orchestrator/merge.py",
        "        if fixes is not None and not (_full_sha(fixes) and _commit(root, "
        "fixes)):",
        "        if False:",
        "review MAJOR-1 (probe P1): a record naming a moving ref (`fix/x`, `HEAD`) "
        "is accepted as if it named a commit.",
    ),
    Mutation(
        "merge: every runs/*/verification.json is staleness-exempt",
        "test_only_the_merging_runs_own_verification_is_exempt",
        "scripts/orchestrator/merge.py",
        '    return run_item is not None and path == f"runs/{run_item}/'
        'verification.json"',
        '    return path.startswith("runs/") and path.endswith("/verification.json")',
        "review MINOR-2 (probe P4): a forged verification.json for another run id "
        "rides in after review and then satisfies gate 1 for that run from the "
        "night branch.",
        off_gate_allowed=(
            (
                "tests/test_orch_merge_review.py::"
                "test_no_verification_json_is_exempt_on_other_prefixes",
                "the same exemption, reached from a non-run/ prefix.",
            ),
        ),
    ),
    Mutation(
        "battery-union: a one-sided tail change keeps ours",
        "test_a_tail_change_only_theirs_made_is_taken_from_theirs",
        "scripts/battery_union.py",
        '        return t.tail, "theirs"',
        '        return o.tail, "ours"',
        "the 2026-09-30 rule: theirs' post-MUTATIONS tail is taken iff ours' tail == "
        "base. Keeping ours silently drops main's I5 edit, which is what 13ea9ae and "
        "74dec77 would have lost.",
        off_gate_allowed=(
            (
                "tests/test_battery_union.py::"
                "test_cli_one_sided_tail_exits_0_and_writes_the_union",
                "the same rule through the CLI: the written file keeps ours' tail",
            ),
            *(
                (
                    "tests/test_battery_union.py::"
                    f"test_reproduces_the_mutation_battery_the_0929_merges_committed[{m}]",
                    "both real 09-29 merges took theirs' tail; keeping ours cannot "
                    "reproduce the committed file",
                )
                for m in ("13ea9ae", "74dec77")
            ),
        ),
    ),
    Mutation(
        "battery-union: an annotated constant is not a constant",
        "test_an_annotated_constant_is_split_out_as_a_constant",
        "scripts/battery_union.py",
        '_CONST = re.compile(r"^(_[A-Z0-9_]+)(?:: [^\\n=]+)? = ',
        '_CONST = re.compile(r"^(_[A-Z0-9_]+) = ',
        "2026-09-30: `_X: tuple[str, ...] = (` must be keyed as a constant. The "
        "pre-fix regex made it head code, and the union refused the i1 merge "
        "(7eda821 was resolved by hand). Reverting only _CONST is worse than the "
        "original defect: _CONST_STRIP still removes the constant from head code, "
        "so a theirs-only annotated constant vanishes from the union silently.",
        off_gate_allowed=(
            *(
                (
                    f"tests/test_battery_union.py::{t}",
                    "the same regex seen through union(): the annotated constant is "
                    "neither a constant nor head code, so it is dropped (measured by "
                    "hand 2026-09-30 on tests/test_battery_union.py: 5 red, 25 green)",
                )
                for t in (
                    "test_an_annotated_constant_new_in_theirs_merges_when_ours_changed_"
                    "head_code",
                    "test_an_annotated_constant_only_theirs_changed_is_updated",
                    "test_a_constant_new_in_theirs_lands_where_theirs_put_it",
                )
            ),
            (
                "tests/test_battery_union.py::"
                "test_reproduces_the_mutation_battery_the_i1_merge_committed",
                "the real i1 merge: theirs' _REVIEW_GATE_REFUSALS is annotated, so "
                "the union cannot reproduce 7eda821 without it",
            ),
        ),
    ),
    Mutation(
        "b2-psi-probe: age leaks into psi-hat",
        "test_no_age_in_psi_hat_permuting_written_at_and_step_changes_nothing",
        "experiments/b2-psi-probe/run.py",
        "            return X @ self.w\n",
        "            return X @ self.w"
        " + 1e-3 * (step - slots.written_at[live]).double()\n",
        "PREREG §4, §6, §11 control 9 (CLAUDE.md: age is excluded from psi-hat): the "
        "bilinear score reads (s_i, c_t) only. A score that also reads written_at and "
        "the step collapses onto recency and makes the vacuity failure invisible.",
        off_gate_allowed=(
            (
                "tests/test_b2_psi_probe.py::"
                "test_the_probe_evicts_the_argmin_ties_to_the_oldest_and_logs_it",
                "inspected: that test ties slots 1 and 2 at psi 0.1 with ages 3 and 2; "
                "an age term (+1e-3 * age) makes them 0.103 and 0.102, so the argmin "
                "moves to slot 2 and the logged psi vector changes -- the same leak "
                "seen through the tie-break and the per-eviction log.",
            ),
        ),
    ),
    Mutation(
        "b2-psi-probe: a captured s_i / c_t keeps its graph",
        "test_stopgrad_cuts_the_graph",
        "experiments/b2-psi-probe/run.py",
        "    return x.detach().clone()\n",
        "    return x.clone()\n",
        "PREREG §1, §5 (CLAUDE.md: no gradient into the transformer or W_sent; c_t "
        "enters psi-hat with a stop-gradient): `_stopgrad` is the one place a captured "
        "gestalt or context leaves the transformer.",
    ),
    Mutation(
        "b2-psi-probe: the ridge lambda is chosen on whatever rows it is handed",
        "test_lambda_is_chosen_on_fit_val_and_never_on_eval",
        "experiments/b2-psi-probe/run.py",
        # fit_heads's pair: r2_split_by_index (review F11) repeats the FIT_VAL line alone,
        # so the FIT_TRAIN line above it keeps this anchor unique (--check-anchors).
        '    require_range([c.doc_id for c in train_caps], "FIT_TRAIN")\n'
        '    require_range([c.doc_id for c in val_caps], "FIT_VAL")\n',
        '    require_range([c.doc_id for c in train_caps], "FIT_TRAIN")\n    pass\n',
        "PREREG §6 / A1.7: lambda minimises the FIT_VAL validation MSE. Without the "
        "range guard an EVAL capture selects lambda silently -- a fit tuned on the "
        "data it is scored on.",
    ),
    Mutation(
        "b2-psi-probe: ref is chosen on whatever accuracies it is handed",
        "test_ref_is_never_chosen_on_eval",
        "experiments/b2-psi-probe/run.py",
        '    require_range(doc_ids, "FIT_VAL")  # §9.3: ref on FIT_VAL, never EVAL\n',
        "    pass\n",
        "PREREG §9.3 / PLAN-v4: ref (FIFO or the age-only head) is chosen per seed on "
        "FIT_VAL, never on EVAL. Choosing it on EVAL picks the easier comparator after "
        "seeing the contrast.",
    ),
    Mutation(
        "b2-psi-probe: kind-oracle ties go to the oldest",
        "test_kind_oracle_ties_are_uniform_random_seeded_per_step",
        "experiments/b2-psi-probe/run.py",
        "        return rng.choice(ks)\n",
        "        return ks[0]\n",
        "PREREG A1.10: ties inside the lowest class are uniformly random, seeded "
        "ko:seed:doc:t. An oldest tie-break hands Q2's comparator the age information "
        "psi-hat is barred from.",
    ),
    Mutation(
        "b2-psi-probe: the random floor is its best seed, not its 5-seed mean",
        "test_the_bootstrap_is_paired_and_the_random_floor_is_the_five_seed_mean",
        "experiments/b2-psi-probe/run.py",
        "    return torch.stack(xs).mean(0)\n",
        "    return torch.stack(xs).min(0).values\n",
        "PREREG §9.2: the random arm's accuracy per replicate is the mean over its 5 "
        "seeds. Its minimum lowers the floor that WIN's clause (4) must clear.",
    ),
    Mutation(
        "b2-psi-probe: WIN no longer has to beat random",
        "test_the_random_floor_is_part_of_win",
        "experiments/b2-psi-probe/run.py",
        "and noninf_lo > -delta and rand_lo > 0:",
        "and noninf_lo > -delta:",
        "PREREG §9.5 WIN clause (4) (REDTEAM-v3 edit 5): psi-hat must beat the random "
        "arm's 5-seed mean with a paired CI lower bound > 0.",
        off_gate_allowed=(
            (
                "tests/test_b2_psi_probe.py::test_the_outcome_rule",
                "the outcome rule's table carries the same clause as one of its rows "
                "(rand_lo = -0.01 must not be WIN); the dedicated test isolates it. "
                "Inspected: both fail on that one row and nothing else.",
            ),
        ),
    ),
    Mutation(
        "b2-psi-probe: E0h reads its class before the thresholds are ratified",
        "test_e0h_rc_is_its_own_and_is_2_until_ratified",
        "experiments/b2-psi-probe/run.py",
        "    if not ratified:\n        return 2\n",
        "    pass\n",
        "PREREG A1.3: until a ruling ratifies the proposed 0.90 / 0.49, E0h exits 2 "
        "whatever the R^2; otherwise an unratified COLLINEAR is a kill (rc 1).",
    ),
    Mutation(
        "b2-psi-probe: an exception inside E0h exits 1",
        "test_e0h_without_its_fits_or_on_an_exception_exits_3",
        "experiments/b2-psi-probe/run.py",
        "            pass\n        return 3\n",
        "            pass\n        return 1\n",
        "PREREG A1.3: an exception inside run.py e0h is DID NOT RUN (3); 1 means "
        "COLLINEAR under a ratifying ruling, and a crash must never read as a kill.",
    ),
    Mutation(
        "b2-psi-probe: the range check skips the used ranges",
        "test_an_overlap_with_any_used_range_is_a_problem",
        "experiments/b2-psi-probe/run.py",
        "        for uname, urng in used.items():\n",
        "        for uname, urng in {}.items():\n",
        "PREREG §3 / A1.15: B2's ranges are asserted disjoint from every used range, "
        "E0d's [262144, 263168) included, before any model is loaded (exit 3).",
    ),
    Mutation(
        "b2-psi-probe: the selected lambda's residual is not checked",
        "test_a_selected_lambda_over_1e8_residual_or_no_eligible_point_exits_1",
        "experiments/b2-psi-probe/run.py",
        '    if path[best]["resid"] > SELECTED_RESID:\n',
        '    if path[best]["resid"] > SELECTED_RESID * 1e3:\n',
        "PREREG A1.6 / §11 control 10: the selected lambda's relative residual must "
        "be <= 1e-8, else exit 1.",
    ),
    Mutation(
        "b2-psi-probe: psi-hat ties go to the newest",
        "test_the_probe_evicts_the_argmin_ties_to_the_oldest_and_logs_it",
        "experiments/b2-psi-probe/run.py",
        "            if vals[q] < vals[j]:  # strict: ties to the lowest slot, "
        "the oldest\n",
        "            if vals[q] <= vals[j]:  # strict: ties to the lowest slot, "
        "the oldest\n",
        "PREREG §4: argmin with ties to the lowest slot index (the oldest), as "
        "OraclePolicy. Ties to the newest is an age preference nobody registered.",
    ),
    Mutation(
        "b2-psi-probe: a CI that excludes its estimate is read",
        "test_a_ci_excluding_the_estimate_is_unresolved",
        "experiments/b2-psi-probe/run.py",
        '    if outcome_flags(point, lo, hi):\n        return "UNRESOLVED"\n',
        "    pass\n",
        "PREREG A1.13: if the estimate is outside its CI the contrast is UNRESOLVED "
        "(CI_EXCLUDES_ESTIMATE); it removes the only WIN and EQUIV overlap.",
    ),
    Mutation(
        "b2-psi-probe: Q2's contrast sizes N_E",
        "test_power_takes_the_max_over_gating_contrasts_only_and_clips",
        "experiments/b2-psi-probe/run.py",
        "        if c in GATING_CONTRASTS\n",
        "        if True\n",
        "PREREG A1.12: only the gating contrasts enter the max; psi-U - kind-oracle "
        "(Q2, outside the truth table) is reported with the N_E that results.",
    ),
    Mutation(
        "b2-psi-probe: the FIT_VAL R² multiplies a summed SSE by n_val again",
        "test_val_r2_raw_is_one_minus_sse_over_sst",
        "experiments/b2-psi-probe/run.py",
        '            "val_r2_raw": [1.0 - float(x) / sst if sst > 0 else None'
        " for x in sse_raw[j]],\n",
        '            "val_r2_raw": [1.0 - float(x) * n_val / sst if sst > 0 else None'
        " for x in sse_raw[j]],\n",
        "PREREG §6 / §8.1, A1.3: phase A (280a2ad) logged 1 - n_val*SSE/SST, so every "
        "pooled R² read near -2e5 and E0h's R² > 0 informativeness gate read every "
        "seed UNINFORMATIVE. The old test asserted only r2 <= 1 and passed on it.",
    ),
    Mutation(
        "b2-psi-probe: the E0h pre-hook changes the forward pass",
        "test_the_logit_prehook_does_not_change_the_forward",
        "experiments/b2-psi-probe/run.py",
        "        x, mem_kv, mem_valid = args[:3]\n        with torch.no_grad():\n"
        "            lg = recompute_logits(module, x, mem_kv, mem_valid)\n"
        "        self._pending.append(lg)\n        self._valid = mem_valid\n"
        "        self.calls.append(tuple(lg.shape))\n        return None\n",
        "        x, mem_kv, mem_valid = args[:3]\n        x = x * 1.001\n"
        "        with torch.no_grad():\n"
        "            lg = recompute_logits(module, x, mem_kv, mem_valid)\n"
        "        self._pending.append(lg)\n        self._valid = mem_valid\n"
        "        self.calls.append(tuple(lg.shape))\n        return (x, *args[1:])\n",
        "PREREG A1.5: the logits come from a forward pre-hook and the forward pass is "
        "unchanged (test_fidelity / test_reduction untouched). A hook that edits its "
        "input passes the softmax control, since the recompute sees the same edit.",
    ),
    Mutation(
        "b2-psi-probe: EVAL units are held in RAM until the tier ends",
        "test_eval_units_reach_disk_before_the_next_document_runs",
        "experiments/b2-psi-probe/run.py",
        "            _flush(pending)  # F1: every unit reaches disk before the next "
        "document runs\n",
        "            pass  # units wait in `pending` until the tier ends\n",
        "B2 build review F1 (BLOCKER): EVAL's per-eviction logs are about 1 kB a "
        "record; held in RAM for a tier they reach tens of GB at the N_E cap. Each "
        "document's unit must be on disk before the next runs, which is also what "
        "makes a killed child resumable.",
    ),
    Mutation(
        "b2-psi-probe: an E0h logit-control failure stops B2's EVAL child",
        "test_a_failed_logit_control_is_e0h_exit_3_and_b2_still_completes",
        "experiments/b2-psi-probe/run.py",
        '            if not u["residency_ok"] or u["sum_worst"] > SUM_TOL:\n',
        '            if not u["residency_ok"] or u["sum_worst"] > SUM_TOL or '
        '(u["logit_control_worst"] or 0.0) > LOGIT_TOL:\n',
        "PREREG A1.3 / A1.5 (build review F3): a failed logit control is E0h exit 3, "
        "and B2's rc covers B2 alone. Raising it from the EVAL child kills B2 after "
        "its cost is paid.",
    ),
    Mutation(
        "b2-psi-probe: non-inferiority reads the all-query CI",
        "test_win_needs_non_inferiority_on_gap_2_to_m",
        "experiments/b2-psi-probe/run.py",
        '            boot["gap_2_to_M"][k],\n',
        '            boot["all"][k],\n',
        "PREREG §9.5 WIN clause (3): the non-inferiority bound is read on "
        "gap_2_to_M. Reading it on all queries lets an arm that loses the "
        "short-gap bucket win (build review F4, M1).",
    ),
    Mutation(
        "b2-psi-probe: E0h pairs each slot's logits with another slot's psi-hat",
        "test_e0h_rows_pair_each_slot_with_its_own_sentence",
        "experiments/b2-psi-probe/run.py",
        "        i_idx = torch.arange(t - m, t)  # FIFO memory at t, oldest first\n",
        "        i_idx = torch.arange(t - m, t).flip(0)"
        "  # FIFO memory at t, oldest first\n",
        "PREREG §10 / A1.4: E0h's rows pair FIFO slot j's logits with "
        "psi-hat(s_{t-m+j}, c_t) and D[t, t-m+j]. A reversed index regresses one "
        "slot's psi-hat on another's logits (build review F4, M2).",
    ),
    Mutation(
        "b2-psi-probe: delta is measured from a random seed, not FIFO",
        "test_val_decisions_delta_is_a_quarter_of_oracle_minus_fifo",
        "experiments/b2-psi-probe/run.py",
        '    d09 = delta_of(out["acc"][0.9]["oracle"], out["acc"][0.9]["fifo"])\n',
        '    d09 = delta_of(out["acc"][0.9]["oracle"], out["acc"][0.9]["random0"])\n',
        "PREREG §9.3: delta = 0.25 (acc_oracle - acc_FIFO) on FIT_VAL. Any other "
        "baseline rescales every WIN / EQUIV / LOSS threshold (build review F4, M3).",
    ),
    Mutation(
        "b2-psi-probe: the gamma = 0 contrasts use the gamma = 0.9 ref",
        "test_the_gamma_zero_contrasts_use_the_gamma_zero_ref",
        "experiments/b2-psi-probe/run.py",
        '        ref = {"U": decisions["ref"][f"U@{g}"], '
        '"C": decisions["ref"][f"C@{g}"]}\n',
        '        ref = {"U": decisions["ref"]["U@0.9"], '
        '"C": decisions["ref"]["C@0.9"]}\n',
        "PREREG §9.3 / §9.8: ref is chosen per (seed, arm, gamma) on FIT_VAL; gamma "
        "= 0 is read against its own ref (build review F4, M4).",
    ),
    Mutation(
        "b2-psi-probe: the bootstrap draws from the global RNG",
        "test_the_bootstrap_is_seeded_and_leaves_the_global_rng_alone",
        "experiments/b2-psi-probe/run.py",
        "    idx = torch.randint(0, D, (n_boot, D), generator=g)\n",
        "    idx = torch.randint(0, D, (n_boot, D))\n",
        "PREREG §9.2: the paired bootstrap uses torch.Generator(20260927 + seed). "
        "Unseeded, every CI changes run to run and consumes the global stream "
        "(build review F4, M7).",
    ),
    Mutation(
        "b2-psi-probe: WIN's random clause reads the ref contrast",
        "test_win_needs_the_random_floor_not_the_ref",
        "experiments/b2-psi-probe/run.py",
        'noninf_lo=n["lo"], rand_lo=rn["lo"]',
        'noninf_lo=n["lo"], rand_lo=a["lo"]',
        "PREREG §9.5 WIN clause (4): psi-hat - random's CI lower bound > 0. Reading "
        "the ref contrast's bound instead drops the random floor from the wiring "
        "(build review F4, M9).",
    ),
    Mutation(
        "newcomer-bakeoff: grace protects the newest slot even when switched off",
        "test_grace_g0_is_the_inner_policy",
        "experiments/newcomer-bakeoff/run.py",
        "        ok = [a > self.g for a in ages]\n",
        "        ok = [a > max(self.g, 1) for a in ages]\n",
        "PREREG §4: g = 0 is the grace rule's off-switch and must return exactly the "
        "inner psi-U victim (the pack's reduction requirement, stated for this "
        "experiment-local wrapper).",
        off_gate_allowed=(
            (
                "tests/test_newcomer_bakeoff.py::test_grace_protects_the_g_newest",
                "its first assertion is the g = 0 case (the newest slot, the inner "
                "argmin), which the mutation protects",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_passes_every_control_on_b2_s_own_fits",
                "the module fixture `fitted` runs fit_core, whose run-time control C9 "
                "asserts the g = 0 off-switch reproduces psi-U; the mutation trips C9, "
                "the fixture raises and every test that uses it errors. By design: the "
                "off-switch is guarded twice",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_refuses_when_the_c_refit_is_not_b2_s",
                "the module fixture `fitted` runs fit_core, whose run-time control C9 "
                "asserts the g = 0 off-switch reproduces psi-U; the mutation trips C9, "
                "the fixture raises and every test that uses it errors. By design: the "
                "off-switch is guarded twice",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_refuses_when_the_tree_does_not_reproduce_b2_s_fit_val",
                "the module fixture `fitted` runs fit_core, whose run-time control C9 "
                "asserts the g = 0 off-switch reproduces psi-U; the mutation trips C9, "
                "the fixture raises and every test that uses it errors. By design: the "
                "off-switch is guarded twice",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_a_unit_runs_every_arm_and_logs_content_and_victim_position",
                "the module fixture `fitted` runs fit_core, whose run-time control C9 "
                "asserts the g = 0 off-switch reproduces psi-U; the mutation trips C9, "
                "the fixture raises and every test that uses it errors. By design: the "
                "off-switch is guarded twice",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_units_are_atomic_resumable_and_keyed",
                "the module fixture `fitted` runs fit_core, whose run-time control C9 "
                "asserts the g = 0 off-switch reproduces psi-U; the mutation trips C9, "
                "the fixture raises and every test that uses it errors. By design: the "
                "off-switch is guarded twice",
            ),
        ),
    ),
    Mutation(
        "newcomer-bakeoff: grace lets the g-th newest slot be evicted",
        "test_grace_protects_the_g_newest",
        "experiments/newcomer-bakeoff/run.py",
        "        ok = [a > self.g for a in ages]\n",
        "        ok = [a >= self.g for a in ages]\n",
        "PREREG §4: a slot of age <= g is ineligible (the g newest are protected; B2's "
        "convention, newest = age 1).",
        off_gate_allowed=(
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_grace_falls_back_when_every_slot_is_protected",
                "at g = 4 with ages 1..4, `>=` makes the age-4 slot eligible, so the "
                "fallback is never reached and the victim changes",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_a_unit_runs_every_arm_and_logs_content_and_victim_position",
                "it asserts graceU1 never evicts age 1 and graceU2 never age <= 2 on "
                "the tiny model; `>=` lets exactly those through",
            ),
        ),
    ),
    Mutation(
        "newcomer-bakeoff: grace has no fallback when every slot is protected",
        "test_grace_falls_back_when_every_slot_is_protected",
        "experiments/newcomer-bakeoff/run.py",
        '        return ok if any(ok) else [True] * len(ages)  # "unless all are"\n',
        '        return ok  # "unless all are"\n',
        "PREREG §4: 'ineligible unless all are' -- with every slot protected the rule "
        "falls back to the unmasked argmin.",
        off_gate_allowed=(
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_passes_every_control_on_b2_s_own_fits",
                "the module fixture `fitted` runs fit_core, whose C8 determinism "
                "control runs graceU4 at the tiny model's M = 4, where every slot has "
                "age <= 4 on some steps; without the fallback the argmin has no "
                "eligible slot and raises, so the fixture and every test using it error",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_refuses_when_the_c_refit_is_not_b2_s",
                "the module fixture `fitted` runs fit_core, whose C8 determinism "
                "control runs graceU4 at the tiny model's M = 4, where every slot has "
                "age <= 4 on some steps; without the fallback the argmin has no "
                "eligible slot and raises, so the fixture and every test using it error",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_fit_core_refuses_when_the_tree_does_not_reproduce_b2_s_fit_val",
                "the module fixture `fitted` runs fit_core, whose C8 determinism "
                "control runs graceU4 at the tiny model's M = 4, where every slot has "
                "age <= 4 on some steps; without the fallback the argmin has no "
                "eligible slot and raises, so the fixture and every test using it error",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_a_unit_runs_every_arm_and_logs_content_and_victim_position",
                "the module fixture `fitted` runs fit_core, whose C8 determinism "
                "control runs graceU4 at the tiny model's M = 4, where every slot has "
                "age <= 4 on some steps; without the fallback the argmin has no "
                "eligible slot and raises, so the fixture and every test using it error",
            ),
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_units_are_atomic_resumable_and_keyed",
                "the module fixture `fitted` runs fit_core, whose C8 determinism "
                "control runs graceU4 at the tiny model's M = 4, where every slot has "
                "age <= 4 on some steps; without the fallback the argmin has no "
                "eligible slot and raises, so the fixture and every test using it error",
            ),
        ),
    ),
    Mutation(
        "newcomer-bakeoff: the R1 shadow part is not down-weighted",
        "test_r1_at_half_weight_is_resident_plus_half_the_shadow",
        "experiments/newcomer-bakeoff/run.py",
        "torch.where(in_win, lam * base, torch.zeros_like(base))",
        "torch.where(in_win, base, torch.zeros_like(base))",
        "PREREG §5 / ADR-0009 L3 R1: the shadow r~ enters the return at weight "
        "lambda_shadow (FROZEN 0.5, read from the registry).",
        off_gate_allowed=(
            (
                "tests/test_newcomer_bakeoff.py::"
                "test_r1_at_zero_weight_is_the_censored_target",
                "at lambda 0 R1 must equal B2's censored target; ignoring lambda gives "
                "U's demand instead",
            ),
            (
                "tests/test_newcomer_bakeoff.py::test_r1_window_truncates_at_k",
                "it checks the in-window cells equal 0.5 * D0, which the mutation makes"
                " D0",
            ),
        ),
    ),
    Mutation(
        "newcomer-bakeoff: the R1 shadow window is one step too deep",
        "test_r1_window_truncates_at_k",
        "experiments/newcomer-bakeoff/run.py",
        "    in_win = after & ((t - t_e.unsqueeze(0)) < K)\n",
        "    in_win = after & ((t - t_e.unsqueeze(0)) <= K)\n",
        "PREREG §5: a slot stays in the depth-K shadow buffer while fewer than K "
        "evictions followed its own (t - t_e < K).",
    ),
    Mutation(
        "newcomer-bakeoff: partial rho does not partial out content",
        "test_partial_rho_removes_an_age_signal_carried_by_content",
        "experiments/newcomer-bakeoff/analysis.py",
        "    P = torch.linalg.pinv(dZ.T @ dZ)\n",
        "    P = 0 * torch.linalg.pinv(dZ.T @ dZ)\n",
        "PREREG §6.3: the §7.1 vacuity statistic on the effective rule is a PARTIAL rho"
        " given content; without the content regression it is the raw within-step "
        "Spearman.",
    ),
    Mutation(
        "b1: L_MC means over rows, not over (doc, t)",
        "test_l_mc_sums_over_live_slots_and_means_over_doc_t",
        "experiments/b1-beta-inertness/run.py",
        "    return (diff * diff).sum() / (N * (S - 1))\n",
        "    return (diff * diff).sum() / rows.sum()\n",
        "PREREG §3 / ADR-0009 L9: sum over resident slots, then mean over (doc, t).",
    ),
    Mutation(
        "b1: the rows are rowset U, not C",
        "test_rows_are_b2_rowset_c_and_target_is_c_at_09",
        "experiments/b1-beta-inertness/run.py",
        '        t, i = B2.row_index(cap, "C", m)\n',
        '        t, i = B2.row_index(cap, "U", m)\n',
        "PREREG §2: B2 rowset C, the FIFO-resident slots online L_MC sums over.",
        off_gate_allowed=(
            (
                (
                    "tests/test_b1_beta_inertness.py::test_full_memory_steps_are_those_with_m_resident"
                ),
                (
                    "full is derived from rows (|rows[t]| = M); under rowset U every i "
                    "< t is a row, so the full steps become t = M only instead of t >= "
                    "M -- the same defect seen from the decision set"
                ),
            ),
        ),
    ),
    Mutation(
        "b1: full-memory steps include M-1 resident",
        "test_full_memory_steps_are_those_with_m_resident",
        "experiments/b1-beta-inertness/run.py",
        '        "full": rows_t.sum(-1) == m,\n',
        '        "full": rows_t.sum(-1) >= m - 1,\n',
        "PREREG §6 m2: decisions only where |resident[t]| = M.",
    ),
    Mutation(
        "b1: pack accepts a captured input carrying graph",
        "test_pack_refuses_inputs_that_carry_graph",
        "experiments/b1-beta-inertness/run.py",
        "            if x.requires_grad or x.grad_fn is not None:\n",
        "            if False:\n",
        "PREREG §2 / CLAUDE.md isolation: no gradient reaches the transformer.",
    ),
    Mutation(
        "b1: psi_all pairs s_t with c_i",
        "test_psi_all_is_the_head_at_every_t_and_i",
        "experiments/b1-beta-inertness/run.py",
        "    return head(g, ctx.reshape(N * S, d)).reshape(N, S, S)\n",
        "    return head(g, ctx.reshape(N * S, d)).reshape(N, S, S).transpose(1, 2)\n",
        "PREREG §3: psi[n, t, i] = head(s_i, c_t).",
    ),
    Mutation(
        "b1: beta does not scale the loss",
        "test_beta_scales_the_loss_and_never_the_lr_in_the_four_arms",
        "experiments/b1-beta-inertness/run.py",
        "    return 1.0 if arm == CONTROL_P else beta\n",
        "    return 1.0\n",
        "PREREG §5: phi's loss is beta * L_MC in the four arms.",
    ),
    Mutation(
        "b1: P does not scale the LR",
        "test_positive_control_scales_the_lr_and_not_the_loss",
        "experiments/b1-beta-inertness/run.py",
        "    lr_eff = beta * lr if arm == CONTROL_P else lr\n",
        "    lr_eff = lr\n",
        "PREREG §5 P: L7(a)'s reading, lr = beta * lr_phi.",
        off_gate_allowed=(
            (
                (
                    "tests/test_b1_beta_inertness.py::test_train_one_is_deterministic_and_beta_inert_without_eps_but_p_is_not"
                ),
                (
                    "its last assertion is that P at beta = 0.01 moves phi away from "
                    "beta = 1; with the LR unscaled and the loss unscaled, P is the "
                    "beta = 1 run exactly, so the end-to-end test sees the same defect"
                ),
            ),
        ),
    ),
    Mutation(
        "b1: coupled_l2 is AdamW",
        "test_coupled_l2_is_adam_with_l2_and_the_others_are_adamw_decoupled",
        "experiments/b1-beta-inertness/run.py",
        (
            "        return torch.optim.Adam(params, lr=lr, betas=ADAM_BETAS, eps=eps, "
            "weight_decay=L2)\n"
        ),
        (
            "        return torch.optim.AdamW(params, lr=lr, betas=ADAM_BETAS, eps=eps,"
            " weight_decay=L2)\n"
        ),
        "PREREG §5 arm 4: coupled L2 is Adam with the L2 term in the gradient.",
    ),
    Mutation(
        "b1: the phi clip is never applied",
        "test_phi_clip_binds_on_phi_norm_at_one",
        "experiments/b1-beta-inertness/run.py",
        "        torch.nn.utils.clip_grad_norm_(params, PHI_CLIP)\n",
        "        pass\n",
        "PREREG §5 arm 2: clip_grad_norm_(phi, 1.0).",
    ),
    Mutation(
        "b1: the joint clip ignores phi's norm",
        "test_joint_clip_uses_the_joint_norm_with_the_transformers",
        "experiments/b1-beta-inertness/run.py",
        "math.sqrt(g_T * g_T + n * n)",
        "g_T",
        "PREREG §5 arm 3: the joint norm is sqrt(g_T^2 + |g_phi|^2).",
    ),
    Mutation(
        "b1: the batch order samples with replacement",
        "test_batch_order_is_a_permutation_per_epoch_and_seed_determined",
        "experiments/b1-beta-inertness/run.py",
        "        perm = torch.randperm(n, generator=g)\n",
        "        perm = torch.randint(n, (n,), generator=g)\n",
        "PREREG §3: without replacement within an epoch.",
    ),
    Mutation(
        "b1: the argmin reads non-resident slots",
        "test_argmins_only_over_resident_slots_at_full_steps",
        "experiments/b1-beta-inertness/run.py",
        '    x = psi.masked_fill(~rows, float("inf"))\n',
        "    x = psi\n",
        "PREREG §6 m2: argmin over resident slots only.",
    ),
    Mutation(
        "b1: eps ratio without bias correction",
        "test_eps_ratio_uses_the_bias_corrected_second_moment",
        "experiments/b1-beta-inertness/run.py",
        '            vhat = st["exp_avg_sq"].double() / (1.0 - b2**k)\n',
        '            vhat = st["exp_avg_sq"].double()\n',
        "PREREG §6 m3: eps / sqrt(v / (1 - 0.95^k)).",
    ),
    Mutation(
        "b1: INERT needs agreement strictly above 0.99",
        "test_classify_thresholds_are_the_prereg_ones",
        "experiments/b1-beta-inertness/run.py",
        "    if min(agreements) >= INERT_AGREE and max(rels) <= INERT_REL:\n",
        "    if min(agreements) > INERT_AGREE and max(rels) <= INERT_REL:\n",
        "PREREG §7: INERT is agreement >= 0.99 and rel <= 0.01.",
    ),
    Mutation(
        "b1: every cell is compared against decoupled_wd",
        "test_compare_reads_every_cell_against_its_own_beta_one",
        "experiments/b1-beta-inertness/run.py",
        (
            '    return "decoupled_wd" if arm in (CONTROL_P, CONTROL_N, CONTROL_I) else'
            " arm\n"
        ),
        '    return "decoupled_wd"\n',
        "PREREG §6: against beta = 1 of the same seed, arm and eps.",
    ),
    Mutation(
        "b1: g_T accepts a missing step",
        "test_read_g_t_takes_steps_2000_to_2999_and_refuses_a_gap",
        "experiments/b1-beta-inertness/run.py",
        "    if sorted(got) != list(range(lo, hi)):\n",
        "    if not got:\n",
        "PREREG §4: exactly one logged transformer norm per step 2000-2999.",
    ),
    Mutation(
        "b1: phi is initialised from the global generator",
        "test_train_one_is_deterministic_and_beta_inert_without_eps_but_p_is_not",
        "experiments/b1-beta-inertness/run.py",
        (
            "        d, base_width=BASE_WIDTH, "
            "generator=torch.Generator().manual_seed(init_seed)\n"
        ),
        "        d, base_width=BASE_WIDTH, generator=None\n",
        (
            "PREREG §2 / ADR-0009 L11: a dedicated generator, so every beta/arm/eps of "
            "a seed starts from the same phi (N is bit-identical)."
        ),
    ),
    # ------------------------------------------------------------------ #
    # Expire-Span [P7] (spec §5.4, D-4, falsifier 6; release condition 3).
    # P7:L<n> = line of ~/research-corpus/sources/memory-retention/
    # sukhbaatar-2021-expire-span.md
    # ------------------------------------------------------------------ #
    Mutation(
        "expire-span: eviction takes the MOST remaining span",
        "test_expire_span_evicts_the_slot_with_least_remaining_span",
        "src/rsr/baselines/expire_span.py",
        "            victim = int(r.argmin().item())",
        "            victim = int(r.argmax().item())",
        "ADR-0010 q6 (a port choice; [P7] has no capacity): a full memory "
        "drops the slot whose longest remaining span over layers is least. "
        "argmax keeps the dying slot.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_a_slot_one_layer_still_reads_is_not_the_victim",
                "the cross-layer eviction test asserts the same argmin, over "
                "max-over-layers.",
            ),
            (
                "tests/test_expire_span.py::test_at_zero_weight_init_expire_span_evicts_exactly_fifo",
                "at the zero-weight init least-remaining IS oldest; argmax is "
                "newest-first.",
            ),
            (
                "tests/test_expire_span.py::test_eviction_ignores_dead_slots",
                "dead slots are masked to +inf for the argmin; argmax picks exactly "
                "those.",
            ),
            (
                "tests/test_expire_span.py::test_span_resets_on_admission",
                "ends by asserting the fresh newcomer is not the victim; under argmax "
                "it is.",
            ),
        ),
    ),
    Mutation(
        "expire-span: newcomer inherits the row's oldest age",
        "test_span_resets_on_admission",
        "src/rsr/baselines/expire_span.py",
        "        age = (step - mem_step).to(e.dtype)",
        "        age = (step - mem_step.min(dim=-1, keepdim=True).values.clamp(min=0))"
        ".to(e.dtype)",
        "gauntlet 0.4: a new occupant is aged from its own write (P7:L267 r = "
        "e - (t-i)). Aging every slot from the row's oldest tenant makes the "
        "newcomer born expired.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_memory_weight_is_zero_on_dead_slots_and_follows_age",
                "pins age = step - written_at per slot by hand; any other age reddens "
                "it.",
            ),
            (
                "tests/test_expire_span.py::test_each_cross_layer_gets_its_own_span_and_mask",
                "compares per-layer masks at hand-set ages; wrong ages collapse them "
                "to 0.",
            ),
            (
                "tests/test_expire_span.py::test_structured_dropout_drops_every_memory_older_than_one_cutoff",
                "the cutoff drops by the same age; wrong ages expire everything at eval.",
            ),
            (
                "tests/test_expire_span.py::test_the_span_loss_charges_memories_on_the_ramp_only",
                "the ramp window is an age window; wrong ages move memories off it.",
            ),
            (
                "tests/test_expire_span.py::test_the_span_loss_carries_gradient_only_through_the_span",
                "its one charged memory is chosen by age; wrong ages charge none.",
            ),
        ),
    ),
    Mutation(
        "expire-span: predictor path leaks into the gestalt",
        "test_predictor_path_trains_the_predictor_but_not_the_gestalt",
        "src/rsr/baselines/expire_span.py",
        "        return kv.detach()",
        "        return kv",
        "ADR-0010 q1: `predictor` must stop span gradient at w, b. Without the "
        "detach it silently becomes `through_gestalt` under a config that says "
        "otherwise.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_model_gradients_are_identical_under_none_and_predictor",
                "the end-to-end form of the same guard.",
            ),
            (
                "tests/test_expire_span.py::test_through_gestalt_changes_the_model_gradients",
                "with the detach gone `predictor` and `through_gestalt` are one setting.",
            ),
        ),
    ),
    Mutation(
        "expire-span: structured dropout never fires",
        "test_structured_dropout_drops_every_memory_older_than_one_cutoff",
        "src/rsr/baselines/expire_span.py",
        "        if training and self.cfg.structured_dropout:",
        "        if False and self.cfg.structured_dropout:",
        "[P7] §4.2 (P7:L387-392): sample l ~ U(0, L) per step and zero every "
        "memory older than l in training. A switch that is on and never "
        "applied is an unregularised comparator passing as a regularised one.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_structured_dropout_is_reproducible_under_its_seed",
                "with no draws two seeds give the same all-ones mask; the test asserts "
                "they differ.",
            ),
        ),
    ),
    Mutation(
        "expire-span: construction draws from the global RNG",
        "test_construction_does_not_consume_the_global_rng",
        "src/rsr/baselines/expire_span.py",
        "        self.weight = nn.Parameter(torch.zeros(n_layers, d_model))",
        "        self.weight = nn.Parameter(torch.randn(n_layers, d_model) * 0.0)",
        "tests/test_reduction.py's RNG trap: identical values, but the "
        "constructor consumed global draws, shifting data order and dropout "
        "for every later arm.",
    ),
    Mutation(
        "model: Expire-Span mask never reaches cross-attention",
        "test_a_zero_weight_slot_receives_no_attention_and_rows_renormalise",
        "src/rsr/model/tg/model.py",
        "            att = _reweight_attention(att, mem_weight)",
        "            att = att",
        "Eq. 4 (P7:L272-279): the mask in attention is the only route by which "
        "the LM loss trains the spans. Dropped, Expire-Span is hard eviction "
        "by an untrained predictor.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_each_c_block_reads_its_own_layer_of_the_mask",
                "asserts a per-layer zero shows in that layer's attention; no mask, no "
                "zero.",
            ),
            (
                "tests/test_expire_span.py::test_the_lm_loss_reaches_the_span_predictor_through_the_model",
                "with alpha = 0 the LM loss reaches w only through the mask.",
            ),
            (
                "tests/test_expire_span.py::test_the_loop_passes_the_mask_to_the_model",
                "fully-expired spans must change the loss vs FIFO; unmasked, it is "
                "FIFO's.",
            ),
        ),
    ),
    Mutation(
        "loop: Expire-Span's span loss is dropped",
        "test_the_loop_adds_the_span_loss",
        "src/rsr/model/tg/policy_loop.py",
        "            contrib = contrib + span_loss",
        "            contrib = contrib",
        "Eq. 7 (P7:L332-335): the alpha term must enter the objective the arm "
        "trains on; a loop that discards it runs alpha = 0 under a config "
        "stamped otherwise.",
    ),
    Mutation(
        "expire-span: the mask ramp is not clamped at 1",
        "test_the_mask_is_the_clamped_ramp",
        "src/rsr/baselines/expire_span.py",
        "        return (1.0 + remaining / self.cfg.ramp).clamp(0.0, 1.0)",
        "        return (1.0 + remaining / self.cfg.ramp).clamp(0.0, None)",
        "Eq. 5 (P7:L284) m = max(0, min(1, 1 + r/R)): unclamped above, an "
        "unexpired slot's weight grows with its span -- an attention bias, not "
        "expiry.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_span_resets_on_admission",
                "asserts the newcomer's mask is exactly 1.",
            ),
            (
                "tests/test_expire_span.py::test_structured_dropout_drops_every_memory_older_than_one_cutoff",
                "asserts the eval mask is exactly 1 before testing the cutoff.",
            ),
            (
                "tests/test_expire_span.py::test_structured_dropout_is_off_at_eval_and_when_switched_off",
                "asserts every weight is exactly 1 at long spans.",
            ),
        ),
    ),
    Mutation(
        "expire-span: span loss charged once at admission",
        "test_the_span_loss_charges_memories_on_the_ramp_only",
        "src/rsr/baselines/expire_span.py",
        "            charged = (m > 0.0) & (m < 1.0) & mem_valid & row_valid.view(-1, 1)",
        "            charged = (mem_step == step - 1) & mem_valid"
        " & row_valid.view(-1, 1)",
        "[P7] §4.2 Loss Computation (P7:L368-382): charging at admission "
        "'empirically results in poor performance'; the loss is charged while "
        "0 < m < 1. This mutation restores this branch's own first (wrong) "
        "implementation.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::"
                "test_rows_past_their_stream_length_are_not_charged",
                "its memories sit on the ramp (age 3) and must be charged before the "
                "row mask can halve the charge; charged only at admission they are "
                "never charged, so its 'charge > 0' precondition fails.",
            ),
        ),
    ),
    Mutation(
        "model: every C block reads layer 0's mask",
        "test_each_c_block_reads_its_own_layer_of_the_mask",
        "src/rsr/model/tg/model.py",
        "                layer_weight = mem_weight[c_idx]",
        "                layer_weight = mem_weight[0]",
        "[P7] computes spans 'independently for each layer' (P7:L81-82). "
        "Routing one layer's mask to every block silently shares one span "
        "across layers.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_the_lm_loss_reaches_the_span_predictor_through_the_model",
                "asserts more than one layer's predictor receives LM gradient; with "
                "layer 0's mask everywhere only layer 0's does.",
            ),
        ),
    ),
    Mutation(
        "expire-span: eviction uses the SHORTEST-lived layer",
        "test_a_slot_one_layer_still_reads_is_not_the_victim",
        "src/rsr/baselines/expire_span.py",
        "            r = (e - age).amax(dim=0)",
        "            r = (e - age).amin(dim=0)",
        "ADR-0010 q6: a slot any layer still reads is alive (as P7:L353-355 "
        "says for heads). min over layers evicts a slot a deep layer is still "
        "retaining.",
    ),
    Mutation(
        "expire-span: span loss charged at eval",
        "test_the_span_loss_is_not_charged_at_eval",
        "src/rsr/baselines/expire_span.py",
        "        if training:\n            charged",
        "        if True:\n            charged",
        "review MAJOR-1 (feat-expire-span @ 49027dd): Eq. 7 is a training objective "
        "(P7:L330-335). Charged at eval, every loss read from the loop bills "
        "Expire-Span alone for its spans -- biasing the referendum against the "
        "baseline whose win ends the project, the direction B-3 forbids.",
        off_gate_allowed=(
            (
                "tests/test_expire_span.py::test_memory_weight_returns_no_span_loss_at_eval",
                "the unit form of the same guard: the policy returns None at eval.",
            ),
        ),
    ),
    Mutation(
        "expire-span: rows past their stream are charged",
        "test_rows_past_their_stream_length_are_not_charged",
        "src/rsr/baselines/expire_span.py",
        "            charged = (m > 0.0) & (m < 1.0) & mem_valid & row_valid.view(-1, 1)",
        "            charged = (m > 0.0) & (m < 1.0) & mem_valid",
        "review MINOR-1: after a row's stream ends step_fn masks it, so no LM "
        "gradient opposes a span charge on its still-ageing memories; charging it "
        "pushes spans down for nothing.",
    ),
    # --- I6: RSR_* environment (PLAN-v4 §4; red-team m-2) ---
    Mutation(
        "battery-env: a stray RSR_* variable is not refused",
        "test_stray_variable",
        "scripts/battery_env.py",
        "    if stray:\n        named = []",
        "    if False:\n        named = []",
        "red-team m-2: the battery hands its whole environment to pytest. An "
        "RSR_* name nothing declares is a typo of a real control or a leftover; "
        "on 09-26/27 the verdicts moved with ambient state nobody recorded.",
    ),
    Mutation(
        "battery: main() records the RSR_* environment without checking it",
        "test_stray_variable_stops_main",
        "scripts/mutation_battery.py",
        # split literals: this row must not be the occurrence apply() replaces
        "    print(battery_env.header(battery_env.che" + "ck()))",
        "    print(battery_env.header(battery_env.present()))",
        "I6: the refusal is one call in main(). Without it battery_env.check is "
        "correct and unused, and a stray override runs every suite.",
    ),
    Mutation(
        "battery: a driver that skips main() is not checked",
        "test_stray_variable_stops_a_driver",
        "scripts/mutation_battery.py",
        "        own = battery_env.che" + "ck()",
        "        own = battery_env.present()",
        "I6 fix round 1: I2's sharded driver runs shards through _run_isolated "
        "without main(); the check there is the only one it passes.",
    ),
    Mutation(
        "battery-env: RSR_ORCH_CMD_ with any suffix is declared",
        "test_stray_variable_orch_cmd",
        "scripts/battery_env.py",
        "    return name in DECLARED or name in ORCH_CMD_NAMES",
        "    return name in DECLARED or name.startswith(ORCH_CMD_PREFIX)",
        "review MINOR-5: orch_cmd reads RSR_ORCH_CMD_<module.upper()>; a misspelled "
        "module stubs nothing, the typo class the gate exists for.",
    ),
    Mutation(
        "battery: the --json rows drop the RSR_* environment",
        "test_declared_variable_is_recorded",
        "scripts/mutation_battery.py",
        "rows.append(battery_env.sta" + "mp(_row(m, failing, did_not_run), own, suite))",
        "rows.append(_row(m, failing, did_not_run))",
        "I6: the json record is what a later reader has. Until 09-30 it carried "
        "nothing of the environment the verdicts were issued under.",
    ),
    Mutation(
        "battery-env: the json record carries names without values",
        "test_declared_variable_is_recorded",
        "scripts/battery_env.py",
        '    out.setdefault("rsr_env", dict(own))',
        '    out.setdefault("rsr_env", sorted(own))',
        "I6: RSR_BATTERY_THREADS=2 and =14 are the same name; the 09-26/27 "
        "verdicts differed by the value.",
    ),
    Mutation(
        "battery-env: the json record carries the suite's names without values",
        "test_declared_variable_is_recorded",
        "scripts/battery_env.py",
        '    out.setdefault("rsr_env_suite", dict(suite))',
        '    out.setdefault("rsr_env_suite", sorted(suite))',
        "review MAJOR-3: the suite's values are the ones pytest ran under.",
    ),
    Mutation(
        "battery-env: the header prints names without values",
        "test_declared_variable_is_recorded",
        "scripts/battery_env.py",
        '        lines.append(f"  {name}={value!r} -- {_role(name)}")',
        '        lines.append(f"  {name} -- {_role(name)}")',
        "I6: the stdout header is the record of a battery run without --json.",
    ),
    Mutation(
        "battery-env: the suite header does not mark a replaced value",
        "test_suite_header_marks",
        "scripts/battery_env.py",
        '            note = f" [replaced by the battery; its own value is '
        '{own[name]!r}]"',
        '            note = ""',
        "review MAJOR-3/MINOR-6: ambient RSR_TORCH_THREADS=14 under a cap of 2 was "
        "recorded as 14 with nothing to say the suite ran with 2.",
    ),
    Mutation(
        "battery-env: stamp overwrites a row's recorded environment",
        "test_stamp_keeps",
        "scripts/battery_env.py",
        '    out.setdefault("rsr_env", dict(own))',
        '    out["rsr_env"] = dict(own)',
        "review MAJOR-4: a driver merging per-shard rows would record its own "
        "environment on every row instead of each shard's.",
    ),
    Mutation(
        "battery-env: a name the tree reads is not declared",
        "test_inventory_every_rsr_name",
        "scripts/battery_env.py",
        '    "RSR_NB_B2_FITS": "experiments/newcomer-bakeoff/run.py: where B2\'s '
        'fits are",\n',
        "    # (declaration removed)\n",
        "I6: the red team counted twelve RSR_* names; this tree has 36 and a "
        "family of per-module names. A declared list nothing holds to the tree is "
        "stale the day after it is written, and a legitimate launch is then "
        "refused. (The mutation replaces the line with a comment rather than "
        "renaming the key: a renamed key would itself be a new RSR_* token in "
        "this file.)",
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
        "- **The probe writer (`tests/_battery_probe.py`) cannot be mutated here.** "
        "A suite that writes no probe, or a wrong one, is `DID_NOT_RUN` by design "
        "(I1), so its mutation could never score `PROVEN`. Proved on a stub "
        "repository instead: `test_a_mutation_whose_suite_reports_no_probe_did_not_run` "
        "and `test_a_suite_on_another_checkouts_venv_did_not_run`.",
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


REASON_COLUMNS = "4000"
"""Terminal width for the mutated suite. pytest trims the short-summary message to
the width, and drops it altogether when the node id alone fills 80 columns."""

_SUMMARY_HEADER = "short test summary info"


class SuiteDidNotRun(Exception):
    """This suite run is not evidence: killed by a signal, or not isolated (I1)."""


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL pytest's whole process group, then reap pytest.

    Called on EVERY exit from a shard suite, normal or not (review MINOR-2): the
    real suite spawns grandchildren -- nested stub batteries, ``python -m venv``,
    experiment ``run.py`` -- and one left alive would keep writing into the shard
    while the next mutation's suite runs there, or under a teardown's
    ``git worktree remove``.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _spawn(argv: list[str], cwd: Path, env: dict[str, str]):
    """Run one shard suite as the leader of its own session; group-killed on exit.

    Its own session also means a terminal Ctrl-C reaches only the battery, whose
    SIGINT handler then stops the suite -- not pytest first, behind its back.
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate()
    finally:
        _kill_group(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def run_suite(shard: iso.Shard | None = None) -> dict[str, str | None]:
    """Return ``{failing node id: one-line reason}`` (``None`` if pytest printed none).

    🔴 I1: with a ``shard``, the suite runs there on the shard's own pytest, and
    raises ``SuiteDidNotRun`` if the suite did not complete -- pytest killed by a
    signal, or exiting 2/3/4 -- or if its probe does not put ``rsr`` under the shard
    root (`battery_isolation.suite_problem`). A killed pytest used to become a
    ``<collection/exit -9>`` "failure" and be scored; a SIGINTed one still was,
    as ``<collection/exit 2>`` (review MAJOR-1).

    📌 I5 (PLAN-v4 §4): this ran ``--tb=no`` until 2026-09-27, so the 09-26 red on
    ``test_submit_launches_the_slot_detached_at_the_pinned_sha`` left only a node
    id. The reason is recorded on the verdict row; it never enters a verdict --
    the node set is parsed exactly as before (`parse_failures`).
    """
    env = {**_suite_env(), "COLUMNS": REASON_COLUMNS}
    argv = ["-p", "no:cacheprovider", "-rfE", "--tb=line", "-q", "--no-header"]
    if shard is None:
        proc = subprocess.run(
            [str(PYTEST), *argv], cwd=ROOT, capture_output=True, text=True, env=env
        )
        return parse_failures(proc.stdout + proc.stderr, proc.returncode)
    env = iso.shard_env(shard, env)
    with contextlib.suppress(FileNotFoundError):
        shard.probe.unlink()
    proc = _spawn([str(shard.root / ".venv" / "bin" / "pytest"), *argv], shard.root, env)
    problem = iso.suite_problem(proc.returncode, iso.read_probe(shard.probe), shard.root)
    if problem:
        raise SuiteDidNotRun(f"{problem} (in {shard.root})")
    return parse_failures(proc.stdout + proc.stderr, proc.returncode)


def parse_failures(text: str, returncode: int) -> dict[str, str | None]:
    """Failing node ids from pytest's short summary, each with its one-line reason.

    The node id is cut at the first space exactly as the ``--tb=no`` parse did, so
    the node SET -- the only thing a verdict reads -- is unchanged. Only lines
    after the short-summary header count: ``--tb=line`` prints message
    continuation lines, and a message may contain ``ERROR ``.
    """
    failing: dict[str, str | None] = {}
    in_summary = False
    for line in text.splitlines():
        if _SUMMARY_HEADER in line and line.startswith("="):
            in_summary = True
            continue
        if not in_summary:
            continue
        line = line.strip()
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            rest = line.split(" ", 1)[1]
            node = rest.split(" ")[0]
            _, sep, reason = rest.partition(" - ")
            failing[node] = reason if sep else None
    if not failing and returncode not in (0, 5):
        last = [ln.strip() for ln in text.splitlines() if ln.strip()]
        failing[f"<collection/exit {returncode}>"] = last[-1] if last else None
    return failing


def _reasons(failing) -> dict[str, str | None]:
    """A stubbed ``run_suite`` may still return a bare set: no reasons, not a crash."""
    return failing if isinstance(failing, dict) else dict.fromkeys(failing)


def anchor_problems() -> list[str]:
    """One line per mutation whose anchor does not occur EXACTLY once in its file.

    Zero occurrences is a stale anchor (apply() would refuse mid-run). More than
    one means apply() replaces the first, which may not be the code the gate is
    about -- the table's own text, for a mutation that targets this file.
    """
    out = []
    for m in MUTATIONS:
        path = ROOT / m.path
        if not path.is_file():
            out.append(f"{m.name!r}: {m.path} does not exist")
            continue
        n = path.read_text().count(m.old)
        if n != 1:
            out.append(f"{m.name!r}: anchor occurs {n} times in {m.path} (must be 1)")
    return out


def apply(mutation: Mutation, root: Path | None = None) -> str:
    """Write ``mutation`` into ``root`` (a shard; ``ROOT`` only for the anchor tests)."""
    path = (ROOT if root is None else root) / mutation.path
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


_HANDLED_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)
"""SIGINT too (review MINOR-3): a Ctrl-C tears down exactly as SIGTERM does, and a
second Ctrl-C during the teardown is ignored rather than cutting it short."""


class _Signalled(BaseException):
    """Raised by the SIGTERM/SIGHUP/SIGINT handler so every ``finally`` runs."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _raise_signalled(signum, _frame):
    # One signal is enough: ignore repeats so the teardown is not interrupted too.
    for s in _HANDLED_SIGNALS:
        signal.signal(s, signal.SIG_IGN)
    raise _Signalled(signum)


@contextlib.contextmanager
def signal_handlers():
    """Defence in depth (I1). The live tree does not depend on these -- no suite
    ever runs in it. They let a stopped battery kill its pytest child
    (``subprocess.run`` does on any exception) and tear its shards down. The
    previous handlers come back on exit, so an in-process ``main()`` (the unit
    tests) leaves its host's handlers as it found them."""
    old = {s: signal.signal(s, _raise_signalled) for s in _HANDLED_SIGNALS}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def open_workspace(args) -> contextlib.AbstractContextManager[list[iso.Shard]]:
    """The shard(s) every suite run happens in: HEAD of the clean invoking tree."""
    sha = iso.pinned_sha(ROOT)
    return iso.open_pool(ROOT, sha, n=1, pool=args.shard_dir, keep=args.keep_shards)


def reset(shard: iso.Shard) -> None:
    iso.reset_shard(shard)


# I6 (PLAN-v4 §4). Imported here, not with the imports above MUTATIONS, so that the
# whole I6 diff outside the table is in this file's tail: `battery_union` refuses a
# merge in which both sides changed the head.
import battery_env  # noqa: E402


def main() -> Exit:
    ap = ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="exit non-zero on any unproven gate"
    )
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--check-anchors",
        action="store_true",
        help="only verify every anchor occurs exactly once in its file (seconds); "
        "exit 3 if not",
    )
    ap.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="regenerate docs/mutation-battery.md from this run",
    )
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="where the shard worktrees live (default ~/.cache/rsr-battery/<repo>); "
        "must be outside this tree",
    )
    ap.add_argument(
        "--keep-shards",
        action="store_true",
        help="leave the shard worktrees for reuse instead of removing them",
    )
    ap.add_argument(
        "--prune-shards",
        action="store_true",
        help="only remove this repo's shard worktrees under --shard-dir (the cleanup "
        "after a SIGKILL); exit 3 if any could not be removed",
    )
    args = ap.parse_args()

    if args.prune_shards:
        pool = (args.shard_dir or iso.default_pool_dir(ROOT)).resolve()
        try:
            errors = iso.prune_shards(ROOT, pool)
        except iso.Unisolated as e:
            refuse(Exit.DID_NOT_RUN, f"--prune-shards: {e}; nothing removed")
        for e in errors:
            print(e, file=sys.stderr)
        if errors:
            return Exit.DID_NOT_RUN
        print(f"no shard worktrees registered under {pool}")
        return Exit.OK

    # 🔴 I6: every RSR_* variable is recorded, and an undeclared one is DID NOT RUN
    # (3) here -- before the anchor check and before any suite.
    print(battery_env.header(battery_env.check()))

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
    # 🔴 Every anchor is checked BEFORE the first suite run. On 2026-09-22 the
    # full battery ran 87 minutes at 5ecda88 and then refused 3 on one stale
    # anchor that a string search finds in under a second.
    problems = anchor_problems()
    if args.check_anchors:
        for p in problems:
            print(f"STALE ANCHOR {p}", file=sys.stderr)
        if problems:
            return Exit.DID_NOT_RUN
        print(f"{len(MUTATIONS)}/{len(MUTATIONS)} anchors occur exactly once")
        return Exit.OK
    if problems:
        for p in problems:
            print(f"STALE ANCHOR {p}", file=sys.stderr)
        refuse(
            Exit.DID_NOT_RUN,
            f"{len(problems)} stale anchor(s); nothing was mutated. Fix the anchors "
            f"(`--check-anchors`), do not drop the mutations.",
        )

    try:
        with signal_handlers():
            return status(_run_isolated(args))
    except iso.Unisolated as e:
        refuse(Exit.DID_NOT_RUN, f"not isolated: {e}")
    except _Signalled as e:
        name = signal.Signals(e.signum).name
        print(
            f"INTERRUPTED by {name}: shards torn down, no verdicts; the invoking "
            f"tree was never mutated",
            file=sys.stderr,
        )
        raise SystemExit(128 + e.signum) from None


def _run_isolated(args) -> Exit:
    with open_workspace(args) as shards:
        shard = shards[0]
        print(
            f"shard: {shard.root} at {shard.sha[:12]} "
            + " ".join(f"{k}={v:.2f}" for k, v in shard.timings.items())
        )
        # 🔴 I6 (review MAJOR-3): the RSR_* environment pytest receives, beside the
        # battery's own. check() again, so a driver that enters here without main()
        # is gated too; the suite env is built by the functions run_suite uses.
        own = battery_env.check()
        suite = battery_env.present(iso.shard_env(shard, _suite_env()))
        print(battery_env.suite_header(own, suite))
        try:
            baseline = run_suite(shard)
        except SuiteDidNotRun as e:
            refuse(Exit.DID_NOT_RUN, f"the baseline suite did not run: {e}")
        if baseline:
            # DID NOT RUN (3): nothing was mutated. Used to exit 1 via a bare
            # `raise SystemExit(msg)` (S0-05).
            refuse(
                Exit.DID_NOT_RUN,
                f"the suite is not green before mutating: {sorted(baseline)}",
            )

        rows = []
        resets: list[float] = []
        for m in MUTATIONS:
            path = shard.root / m.path
            original = apply(m, shard.root)
            did_not_run = None
            try:
                failing = run_suite(shard)
            except SuiteDidNotRun as e:
                failing, did_not_run = {}, str(e)
            finally:
                path.write_text(original)
                t0 = time.perf_counter()
                reset(shard)
                resets.append(time.perf_counter() - t0)
            rows.append(battery_env.stamp(_row(m, failing, did_not_run), own, suite))
            print(
                f"{rows[-1]['verdict']:13s} {m.name:42s} "
                f"-> {rows[-1]['n_on_gate']} on gate, {len(rows[-1]['off_gate'])} off "
                f"({len(rows[-1]['off_gate_undeclared'])} undeclared)"
            )
        if resets:
            print(
                f"shard reset: {len(resets)} x, mean {sum(resets) / len(resets):.3f} s, "
                f"max {max(resets):.3f} s"
            )
    return _report(args, rows)


def _row(m: Mutation, failing, did_not_run: str | None) -> dict:
    reasons = _reasons(failing)
    on_gate = sorted(f for f in failing if m.gate in f)
    off_gate = sorted(f for f in failing if m.gate not in f)
    declared = {node for node, _reason in m.off_gate_allowed}
    leaked = [f for f in off_gate if f not in declared]
    if did_not_run is not None:
        # 🔴 I1: not isolated, or killed. Nothing here is evidence either way.
        verdict = "DID_NOT_RUN"
    elif on_gate and leaked:
        verdict = "LEAKS"
    elif on_gate:
        verdict = "PROVEN"
    else:
        verdict = "ADDS NOTHING"
    row = {
        "mutation": m.name,
        "gate": m.gate,
        "why": m.why,
        "reddened_gate": bool(on_gate),
        "n_on_gate": len(on_gate),
        "off_gate": off_gate,
        "off_gate_allowed": [list(x) for x in m.off_gate_allowed],
        "off_gate_undeclared": leaked,
        "verdict": verdict,
        # I5: one line per failing node. Data, never a verdict input.
        "failure_reasons": {f: reasons[f] for f in sorted(reasons)},
    }
    if did_not_run is not None:
        row["did_not_run"] = did_not_run
    return row


def _report(args, rows: list[dict]) -> Exit:
    if args.json:
        # A run record: DID_NOT_RUN rows belong in it.
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    did_not_run = [r for r in rows if r["verdict"] == "DID_NOT_RUN"]
    if args.markdown and did_not_run:
        # 🔴 The committed table is a record of verdicts; a mutation that did not
        # run has none (review MINOR-4). Leave the old record in place.
        print(
            f"NOT WRITING {args.markdown}: {len(did_not_run)} mutation(s) did not run",
            file=sys.stderr,
        )
    elif args.markdown:
        args.markdown.write_text(_markdown(rows))

    bad = unproven(rows)
    print(f"\n{len(rows) - len(bad)}/{len(rows)} gates proven by mutation")
    if bad:
        print("UNPROVEN:")
        for r in bad:
            if r["verdict"] == "DID_NOT_RUN":
                print(
                    f"  {r['gate']}  ({r['mutation']}) -- DID NOT RUN: {r['did_not_run']}"
                )
            elif not r["reddened_gate"]:
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
    # 🔴 I1: a mutation that did not run is not a table entry, with or without
    # --check. "Did not run" never becomes exit 0.
    if any(r["verdict"] == "DID_NOT_RUN" for r in rows):
        return Exit.DID_NOT_RUN
    # Without --check this is the table renderer and exits 0 once the table is
    # rendered; --check is the gate. Documented in the module docstring, and
    # decided in S0-05's RESULTS.md.
    return Exit.FAIL if (args.check and bad) else Exit.OK


if __name__ == "__main__":
    run_main(main)
