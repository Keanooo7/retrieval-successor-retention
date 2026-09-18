# The mutation battery — gauntlet 1.7

> **A new check is not believed until a mutation has shown it red** — and the
> mutation must redden *only* it. If nothing reddens it, **the check adds nothing
> and that is the finding.**

Produced by `.venv/bin/python scripts/mutation_battery.py` at `0a9f5d8` on 2026-09-17.
The script is the artifact; this table is its output, committed so a reviewer can
see the verdicts without running it. Re-run it after adding a gate — a gate with no
mutation is not in this table and is therefore unproven.

**17/17 gates proven.**

| Mutation | Gate it must redden | Verdict | Off-gate |
|---|---|---|---|
| t_warm back to inf | `test_reduction` | **PROVEN** | 0 |
| reduction uses the learned head | `test_reduction` | **PROVEN** | 0 |
| drop nu from the off-switch table | `test_every_config_field_has_an_off_switch` | **PROVEN** | 11 |
| RSRConfig.nu gets a default | `test_no_registry_owned_field_has_a_default` | **PROVEN** | 0 |
| LRU state back in MemoryState.accum | `test_lru_and_fifo_choose_different_slots_in_a_real_eviction_loop` | **PROVEN** | 1 |
| an accumulating policy skips the admission hook | `test_an_accumulating_policy_is_corrupted_without_the_admission_hook` | **PROVEN** | 0 |
| scope-free MEASURED read leaks again | `test_scope_free_measured_read_does_not_leak_across_scopes` | **PROVEN** | 0 |
| record() accepts DERIVED again | `test_derived_constants_cannot_be_recorded` | **PROVEN** | 1 |
| A_max trusts the caller's S | `test_A_max_does_not_trust_a_caller_supplied_S` | **PROVEN** | 0 |
| bilinear multiplier dropped | `test_the_bilinear_multiplier_is_one_over_d_in_the_output` | **PROVEN** | 2 |
| W transposed | `test_W_is_not_silently_transposed` | **PROVEN** | 1 |
| linear term dropped | `test_both_terms_are_present` | **PROVEN** | 2 |
| value head shares the transformer group | `test_the_value_head_has_its_own_group` | **PROVEN** | 2 |
| r_i drops the memory gate | `test_the_memory_gate_changes_the_answer` | **PROVEN** | 0 |
| r_i accepts a train-mode trace | `test_a_train_mode_trace_is_refused` | **PROVEN** | 0 |
| rank shift counts the sliding window | `test_evicting_the_oldest_displaces_nothing` | **PROVEN** | 3 |
| underfull steps are not rescaled | `test_underfull_steps_are_rescaled_not_masked` | **PROVEN** | 0 |

## What each mutation breaks, and what else went red

### t_warm back to inf

**Gate:** `test_reduction` — **PROVEN**

gauntlet 0.1: the reduction stops reaching the score path

Reddened nothing else.

### reduction uses the learned head

**Gate:** `test_reduction` — **PROVEN**

the §3.7 reduction is no longer argmin(-a_i)

Reddened nothing else.

### drop nu from the off-switch table

**Gate:** `test_every_config_field_has_an_off_switch` — **PROVEN**

a term with no documented off-switch; the brief records this exact hole

Also reddened (11):

- `tests/test_reduction.py::test_a_max_tracks_capacity`
- `tests/test_reduction.py::test_is_reduction_detects_a_single_flipped_switch`
- `tests/test_reduction.py::test_reduction_agrees_with_fifo_on_every_eviction`
- `tests/test_reduction.py::test_reduction_config_disables_every_added_term`
- `tests/test_reduction.py::test_reduction_runs_through_the_score_path`
- `tests/test_reduction.py::test_switching_psi_source_to_the_learned_head_reddens_the_reduction`
- `tests/test_reduction.py::test_t_warm_is_zero_not_infinity`
- `tests/test_reduction.py::test_the_learned_head_does_shift_ranks`
- `tests/test_reduction.py::test_the_reduction_pins_the_context_source`
- `tests/test_reduction.py::test_the_reduction_pins_the_positional_index_to_rank`
- `tests/test_reduction.py::test_the_reduction_shifts_no_ranks`

### RSRConfig.nu gets a default

**Gate:** `test_no_registry_owned_field_has_a_default` — **PROVEN**

gauntlet 0.3: a frozen unmeasured constant, D-1's exact shape. Defaulting `nu` instead makes the dataclass itself invalid (a default before a non-default), so the module fails to import and the gate never runs -- a stronger guarantee, but not one this test can demonstrate.

Reddened nothing else.

### LRU state back in MemoryState.accum

**Gate:** `test_lru_and_fifo_choose_different_slots_in_a_real_eviction_loop` — **PROVEN**

gauntlet 0.4: LRU silently becomes FIFO

Also reddened (1):

- `tests/test_policies.py::test_lru_remembers_an_attention_event_older_than_one_step`

### an accumulating policy skips the admission hook

**Gate:** `test_an_accumulating_policy_is_corrupted_without_the_admission_hook` — **PROVEN**

gauntlet 0.4 root cause, on the policy shape that needs it. NOTE: neutering LRU's own on_write reddens NOTHING -- its select_eviction takes max(written_at, last_used) and a stale record is always older than the new occupant's write step, so the hook is defensive for LRU and load-bearing for H2O. Recorded in the table rather than papered over.

Reddened nothing else.

### scope-free MEASURED read leaks again

**Gate:** `test_scope_free_measured_read_does_not_leak_across_scopes` — **PROVEN**

gauntlet 0.7(1): §13's cross-corpus prohibition stops being enforced

Reddened nothing else.

### record() accepts DERIVED again

**Gate:** `test_derived_constants_cannot_be_recorded` — **PROVEN**

gauntlet 0.7(2): a ledger row nothing reads

Also reddened (1):

- `tests/test_constants.py::test_K_cannot_be_recorded_against_an_arbitrary_experiment`

### A_max trusts the caller's S

**Gate:** `test_A_max_does_not_trust_a_caller_supplied_S` — **PROVEN**

gauntlet 0.7(3): a CONDITIONAL guard that validates nothing

Reddened nothing else.

### bilinear multiplier dropped

**Gate:** `test_the_bilinear_multiplier_is_one_over_d_in_the_output` — **PROVEN**

gauntlet 1.6: §4.3's 1/d stored but not applied

Also reddened (2):

- `tests/test_value_head.py::test_multipliers_follow_the_muP_table`
- `tests/test_value_head_arithmetic.py::test_W_is_not_silently_transposed`

### W transposed

**Gate:** `test_W_is_not_silently_transposed` — **PROVEN**

gauntlet 1.4: every shape test stays green

Also reddened (1):

- `tests/test_value_head_arithmetic.py::test_forward_matches_the_formula_computed_by_hand`

### linear term dropped

**Gate:** `test_both_terms_are_present` — **PROVEN**

gauntlet 1.4: ψ̂ silently becomes purely bilinear

Also reddened (2):

- `tests/test_value_head_arithmetic.py::test_forward_matches_the_formula_computed_by_hand`
- `tests/test_value_head_arithmetic.py::test_the_linear_multiplier_is_one_over_fan_in_in_the_output`

### value head shares the transformer group

**Gate:** `test_the_value_head_has_its_own_group` — **PROVEN**

gauntlet 1.6: mis-grouping surfaces in week 9 with no error message

Also reddened (2):

- `tests/test_param_groups.py::test_phi_is_not_also_in_a_transformer_group`
- `tests/test_param_groups.py::test_the_value_head_lr_follows_its_own_one_over_d_rule`

### r_i drops the memory gate

**Gate:** `test_the_memory_gate_changes_the_answer` — **PROVEN**

D-E: layer-specific rescaling silently omitted

Reddened nothing else.

### r_i accepts a train-mode trace

**Gate:** `test_a_train_mode_trace_is_refused` — **PROVEN**

D-F: the policy learns from dropout masks

Reddened nothing else.

### rank shift counts the sliding window

**Gate:** `test_evicting_the_oldest_displaces_nothing` — **PROVEN**

ADR-0006: the metric measures the window, not the policy

Also reddened (3):

- `tests/test_instrumentation.py::test_evicting_a_middle_slot_displaces_the_slots_older_than_it`
- `tests/test_instrumentation.py::test_evicting_the_newest_displaces_the_most`
- `tests/test_reduction.py::test_the_reduction_shifts_no_ranks`

### underfull steps are not rescaled

**Gate:** `test_underfull_steps_are_rescaled_not_masked` — **PROVEN**

§3.2.1: the estimator learns 'stream-initial content is valuable'

Reddened nothing else.

## Two findings the battery produced, recorded rather than fixed away

**1. `LRUPolicy.on_write` is defensive, not load-bearing.** Neutering it reddened
**nothing**. `select_eviction` takes `max(written_at[i], last_used[i])`, and a stale
`last_used` is by construction older than the new occupant's write step — a slot can
only be attended at or after the step it was written — so the `max` already discards
the previous tenant's history. The hook stays because **H2O needs it**: accumulated
attention is not monotone in write time, so a stale accumulator survives the `max`
trick. That is now proved on a stand-in accumulator in `tests/test_policies.py`
rather than asserted about code that does not exist yet. The original test asserted
the hook was load-bearing and was vacuous; it was replaced, not deleted.

**2. Defaulting `nu` breaks the module, not the test.** `RSRConfig` lists `nu` before
fields that do have defaults, so giving it one makes the dataclass itself invalid and
the module fails to import — a stronger guarantee than the test provides, but not one
the test can demonstrate. The battery therefore mutates `a_max`, the last
non-defaulted field, where the gate can actually fire.

## Known couplings

Three mutations redden well beyond their own gate, and that is expected rather than a
fault:

- **drop `nu` from the off-switch table** (11 off-gate) — `reduction_to_tg()` is built
  *from* that table, so removing an entry makes every reduction test fail to
  construct a config. The coupling is the design: one enumeration, not two.
- **rank shift counts the sliding window** (3 off-gate) — the displacement statistic
  is asserted in both `test_instrumentation.py` and `test_reduction.py`, because it is
  both a property of the metric and a property of the reduction.
- **bilinear multiplier / linear term / value-head group** (2 each) — the μP
  multipliers are checked on the output in `test_value_head_arithmetic.py` and on the
  attribute in `test_value_head.py`. Both deliberately: an attribute set correctly and
  never applied is precisely the silent failure §4.3 warns about.

## Not covered here

- **`tests/conftest.py`'s all-skipped tripwire** cannot be mutated from inside the
  suite it guards. Proved separately, by running a suite in which every test skips:

  ```
  passed=0 failed=0 skipped=2 errors=0
  FAILED: only 0 tests passed, floor is 100.
  all-skipped suite exit code = 1
  real suite exit code       = 0
  ```

- **`tests/test_fidelity.py`** and the E0b loss-curve test are skipped pending the
  transcription. They are unrun, not passing, and no mutation can prove a gate that
  does not execute.
