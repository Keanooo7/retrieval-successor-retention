# PREREG — the §10.3 decisive run: is the working memory live on the rewardable corpus?

**Written 2026-09-21 by the manager**, before any S0-03 number existed. At the moment of writing, S0-03 had not been spawned, so no model had been trained on the new corpus by anyone. **This file is committed before any run it governs.** Amendments are appended below, dated, with the data that prompted them named. The text above an amendment is never edited.

**run_id:** `decisive-shuffle` · **device:** CPU (overnight dispatch H2: on CPU the within-seed replicate floor is exactly `0.0`; on MPS it is ~`2.5e-6`, the same order as #17's readout) · **seeds:** `0, 1, 2` per arm.

## Question

`docs/RESEARCH-CONTEXT.md:879`: *"Repair the loop, make the corpus rewardable, re-run the ablation with the shuffle control as the primary readout."* #17 (`runs/shuffle-control/`) found the memory inert under the audit's model: unmasked, hinge off, old corpus. It could not tell whether (a) the readout ignores memory, or (b) memory carries no row-specific content. This run changes the corpus (S0-03), and then the objective, one factor at a time.

**Falsifier addressed:** "the TG working memory is inert on a corpus whose objective requires retrieval."

## Condition

`experiments/shuffle-control/run.py`'s `CONFIG` (`d=128`, `steps_per_stream=48`, `batch=16`, `max_tokens=64`, `memory_slots=16`, `iters=300`, `lr=1e-3`, `policy="fifo"`, `measure_documents=8`), with the corpus replaced by S0-03's rewardable synthetic corpus at `sentences_per_document=48, seed=<seed>`. Arms:

| arm | `masked_loss` | `srep_norm_reg_weight` | isolates |
|---|---|---|---|
| **A** | `False` | `0.0` | the corpus alone: exactly #17's config except the corpus |
| **B** | `True` | `0.0` | the repaired objective |
| **C** | `True` | `None` → `TGConfig` default `0.01` | the hinge (an open owner decision; running it informs that decision without taking it) |

**Decoy:** the untrained model at the same seed and the same config (same `V`, same corpus).

## Instrument

`src/rsr/metrics/memory_liveness.py::shuffle_control`, reused. **No third implementation.** Any new function the discriminators need is added beside it, with its own controls, in the same file.

## Primary readout and decision rule: #17's Amendment 1, unchanged

`experiments/shuffle-control/PREREG.md`, *Amendment 1*. `A` = mean |per-token Δ| over real tokens (`mean_abs_token_delta`), and `ratio = A_trained / A_decoy` at the same seed. **Per arm:**

| condition | verdict for the arm |
|---|---|
| any control fails on any seed (memory-disabled ≠ 0, own-memory ≠ 0, decoy `A` = 0), **or** `ratio == 1.0` exactly on any seed (the decoy is the trained model: an aliasing control, see *Mutation bar*) | `inconclusive` |
| every seed `ratio ≤ 0.01` | **inert** (Amendment 1's `survived`) |
| any seed `ratio ≥ 0.1` | **live** (Amendment 1's `falsified`) |
| otherwise | `inconclusive` |

These thresholds were fixed on 2026-09-20 in `ba71e86`, before any of tonight's data existed. **No threshold is chosen in this file.**

**Which §10.3 outcome occurred** is read off B and C:
- **Inert on B and C:** the memory path itself is broken (wiring or gate). Escalate. Sprint 2 does not start.
- **Live on any arm:** the cause was the objective or the corpus. Arm A's verdict says whether the corpus alone was enough.
- Anything else (for example, B or C `inconclusive`) is reported as such, **not** rounded to either outcome.

## Secondary readouts (reported, not the bar)

1. **Answer tokens only.** The same `A`/`ratio` computation restricted to S0-03's answer-token supervision mask, split by `gap > M` vs `gap < M`. Amendment 1's bands are applied **descriptively**; they do not change the arm's verdict.
2. **Rival-hypothesis discriminators**, which #17 could not separate:
   - **Cross-row cosine of trained memory contents.** Take the mean off-diagonal cosine similarity between rows' memory vectors at matched slot and step, over the measurement batch. **Descriptive only; no verdict attached.** Near-collinear memories make any swap a near no-op, whatever the readout does.
   - **Matched-norm random replacement.** Replace each row's memory with i.i.d. Gaussian vectors rescaled to the replaced vector's L2 norm (per slot, per step), from a generator seeded independently of the model. Read `A_random / A_decoy` and interpret it with Amendment 1's bands, **descriptively**:
     - random `≤ 0.01` → the readout ignores memory;
     - random `≥ 0.1` while the shuffle `≤ 0.01` → memory is read but carries no row-specific content.
   - The replacement function needs its own control: replacing each row's memory with **itself** must read exactly `0.0`.
3. Signed mean Δ, honest real-token NLL, memory gates, and tokens moved, as in #17.

## Mutation bar (the experiment script is new)

- **Arm swap:** swap two arms' configs in the script. The manifest-hash check must catch it, meaning each arm's checkpoint config must be verified against that arm's frozen manifest entry, and the run must refuse.
- **Decoy aliasing:** point the decoy at the trained checkpoint. Then `A_decoy == A_trained`, `ratio == 1.0` exactly, and the rule above makes the arm `inconclusive`. A test must redden.

## What this does not establish

Nothing here tests whether RSR discovers Kintsch & van Dijk's leading-edge strategy. It establishes whether TG has a working memory that **any** retention policy could act on. That is the precondition for E1/E3/E7 meaning anything.

---

## Amendment 1 — 2026-09-21, manager, before the decisive run started, prompted by S0-03's return (PR #30)

**What prompted it.** These are three methodological findings in S0-03's return. They are not its effect sizes.
1. Scoring on training documents is contaminated. 300 iters × 16 streams is ~75 passes over 64 docs, and answer NLL with memory zeroed was **below chance on train docs**, so memorisation reads as retrieval.
2. At `gap = 1`, `bos_replacement_mode="copy"` carries the previous sentence outside the memory, so zeroing or shuffling memory does not remove the assert there.
3. "`gap > M` vs `gap < M`" leaves `gap == M` unassigned, and under FIFO a `gap == M` assert is still resident.

The manager had read S0-03's numbers when writing this. **For that reason this amendment changes no threshold, no arm, no condition, and not the primary readout's population.**

**Changes (secondary readouts only):**
- **Secondary 1 (answer tokens)** is scored on **held-out documents**: S0-03's held-out set, docs 64..127 of each seed's generator stream, using its first `measure_documents = 8`. It is bucketed **`gap = 1` (reported separately)**, **`2 ≤ gap ≤ M`** (the assert is resident under FIFO), and **`gap > M`** (evicted).
- The **primary** readout stays on the #17 measurement batch, for comparability with `runs/shuffle-control/`. It is **also** reported on the held-out batch as a descriptive row, and that row carries no verdict.
- The discriminators (cross-row cosine, matched-norm random replacement) are reported on both batches.
