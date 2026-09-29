# Review: E0d Amendment 2 v2 (adversarial, pre-commit)

- **Reviewer:** Claude Opus 5.5 subagent, 2026-09-29. Read-only. No repo file was edited, nothing was committed, nothing was trained.
- **Target:** `~/Documents/RSR-2026-09-27-plan/reviews/E0D-AMENDMENT-2-FINAL-v2.md`, cited below as `A2v2:<line>`.
- **Base:** `run/e0d` @ `4918f2d` (`.worktrees/e0d`), `experiments/e0d/PREREG.md` (598 lines, base + A1 + A1.10 erratum).
- **Rulings:** `docs/owner/rulings/R-2026-09-27-e0d-statistic.md` (committed on `main` at `7f9bb10`; byte-identical to the `owner-drafts/` copy, `diff` rc 0). `owner-drafts/R-2026-09-27-e0d-statistic-amended.md` (not committed anywhere).
- **Scratch script:** `/private/tmp/claude-501/-Users-keanooo7-retrieval-successor-retention/e135e824-a1b9-4e3e-bdda-24db6dc3af6d/scratchpad/indep.py`. This is an independent reimplementation. It does not import m5 or m6. Output is in `indep.out` next to it. rc 0.

## What I checked myself, with literal output

### 1. Pre-data status and disjointness (verified from files)

- **No E0d gate data exists.**
  - No `runs/e0d*` directory exists in the main checkout or in any of the 36 worktrees. The glob returned "no matches found" for each.
  - `find / -name e0d.class` returned nothing.
  - `LOOP-STATE.json` shows E0D-AMD2 "not committed; independent review is next-session P0". No run is recorded.
- **Document keys are disjoint.**
  - `src/rsr/data/synthetic.py:237` keys each document `cfg.seed * 1_000_003 + doc_id`.
  - Set E seed s gives keys `s·1000003 + [64,128)`. `D_E0d` seed s' gives keys `s'·1000003 + [262144,263168)`.
  - Because 262144 > 128 and 263168 − 64 < 1000003, no key is shared within a seed or across seeds.
- **The scripts guard the range.**
  - `m5_tau_seeds.py:34-38` asserts `HI=128 ≤ 262144`. It calls `e0d_documents(..., cleared=False)`, and `run.py:468` refuses any range that touches `D_E0D` unless cleared.
  - `m6` reads only the three `.npz` files and `m5_out.json`.
- **Cell files.** `shasum -a 256` gives `b683dd1b…f6cd6c`, `4c55cc60…156197` and `1338de4b…65f15c`. All three equal A2.2's table and `m5_out.json`.
- **No real `r_i` was joined to LOO cells.**
  - I grepped every script in `e0d-a2/` and `e0d-a2-final/`, and the predecessor's scratch `dry.py`, for `lookahead|D.pt|retrieval_demand|capture`.
  - The only hits are docstrings that say no `r_i` is computed.
  - **But real gated arm-B `r_i` does exist on set E.** It is in `lookahead-room-r2` `D.pt`, docs `[0,1088)`, ckpt 2500/3000, all seeds. Its age and kind profile was read before the statistic was chosen (brief §3.4). See m-3.

### 2. τ, recomputed independently (`indep.py`, same rule: |Δ_resample|, full-memory Q-steps, non-A, non-NaN, numpy type 7, q = 0.995)

```
seed 0: tau 0.4157434984576128  n_off 13376
seed 1: tau 0.4315741845071321  n_off 12842
seed 2: tau 0.4884946896703897  n_off 13398
```

- All three are bit-equal to A2v2:122-124. The `crit_at_tau` counts and bootstrap CIs in `m5_out.json` equal the A2.2 tables.
- τ uses Δ only. No `r_i` enters it, so τ does not leak the gate's data.
- q = 0.995 was picked after `m4.log` showed where each candidate q lands on the labels (m-4). That is label-only information.

### 3. Calibration, recomputed independently (A1.3 population; brute-force Mann-Whitney, ties ½; midrank pct over the eligible slots)

```
            steps n_t positives  binary_y true_demand age_only(log age) recency perfect_cont(own eps)
seed 0:     720  {15}  517        1.0      0.9672      0.5               0.5     0.9872
seed 1:     724  {15}  506        1.0      0.9731      0.5               0.5     0.9861
seed 2:     740  {15}  512        1.0      0.9750      0.5               0.5     0.9867
C11 structure: rank == M - age on every full-memory cell, all seeds; min age in population 2 (bin 1 empty)
```

- The following claims are reproduced exactly, or to within noise for the continuous proxy, which uses a different ε draw:
  - binary perfect 1.0;
  - age-only and recency 0.5 exactly;
  - `true_demand` 0.9672 / 0.9731 / 0.9750;
  - continuous perfect ≈ 0.987;
  - positives 517 / 506 / 512;
  - `n_t = 15`;
  - bin 1 excluded.
- Pure step-concentration ≈ 0.50: **not re-run**. It follows analytically. On k ≥ 1 steps a flat score ties every slot at pct 0.5. On k = 0 steps every cell is a negative. So each bin's AUROC is ½ in expectation. The point values 0.5008 / 0.4963 / 0.4995 are **UNVERIFIED** by me.
- The content family, age-term rows, within-step shuffle and every sd: **UNVERIFIED**. I did not re-run m6's bootstrap.
- Power arithmetic: **verified from `m7.log`**.
  - z = 2.8016 (one seed) and 3.4234 (three seeds).
  - 0.85 + 3.4234 × 0.0026 = 0.8589.
  - 0.85 + 3.4234 × 0.0036 = 0.8623.
  - These match A2v2:371-373 and :391.

### 4. C8 regex (A2.13), run on both ruling texts

```
R-2026-09-27-e0d-statistic-amended.md unanchored: ['0.85'] anchored: []
R-2026-09-27-e0d-statistic.md         unanchored: ['0.85'] anchored: []
```

D.2 item 5 is confirmed.

## Findings

### BLOCKER

None.

### MAJOR

**M-1. C8 does not require the ruling that authorises the gated statistic.**
- **Evidence.**
  - A2v2:415-417 requires "every `R-*-e0d-statistic*` file". A glob cannot know which files ought to exist.
  - Suppose the amended ruling is never committed, or Brendan declines it. `R-2026-09-27-e0d-statistic.md` alone matches the glob, yields `AUROC_STAR 0.85`, and C8 passes.
  - The runner would then gate on `AUROC_strat,pct` and ignore H. The only committed ruling names neither choice: it names raw `AUROC_strat` plus the H harm gate (ruling lines 11-12, 19).
  - So the runner would enforce a PM decision that has no owner-committed authority, with no error.
- **Edit.** In A2.13, after "which **replace `R-*-rho-star*`**", add:
  > "C8 additionally requires at least one committed, unmodified file matching `R-*-e0d-statistic-amended*`; without it C8 exits 3, because the committed `R-2026-09-27-e0d-statistic` alone names raw `AUROC_strat` and the H gate, not A2.4's statistic."
- **Test.** Add a case to T14: "the first ruling alone → exit 3", and a battery mutation "C8 accepts the first ruling alone".

**M-2. The pct gate can produce a kill that is an artefact of age in `r_i`, and the author left the decision on this open.**
- **Evidence.** The effect itself is stated.
  - A2v2:394 and D.2 item 3 (A2v2:565-575) report that adding `u(age)` lowers pct by 0.03-0.05 at a = 2.0, while raw rises.
  - Real gated arm-B `r_i` is known to carry age: demand is U-shaped in age, and the pending/filler gap varies with age (brief §3.4; lookahead-room-r2 `D.pt`).
- **Why it matters for the kill.** D.2 item 3 calls the pct form "more conservative… It does not make it permissive." That holds only for the pass.
  - For the kill it is the opposite. Row 6 fires on pct CI upper < 0.85.
  - A seed can therefore read `DISAGREE` while raw `AUROC_strat`, which is also exactly age-controlled (D.2 item 2), reads `AGREE`.
  - Such a seed contributes to CONFOUND, exit 1. `STEP_SENSITIVE` is only a report (A2v2:237-239).
- **The decision is still open.** A2v2:573-575 says "Whether that is intended is the PM's to confirm. This draft does not change it." A pre-registration cannot be committed with an unresolved decision about its own kill rule.
- **Edit (preferred).** In A2.8, insert a row between rows 5 and 6:
  > `5b | AUROC_strat,pct CI upper < A* and raw AUROC_strat CI lower ≥ A* | STEP_OR_AGE_AMBIGUOUS`
  - It is counted like `AGREE_VIA_RANK` in the §8 table: row 5 becomes "…at least one `AGREE_VIA_RANK` or `STEP_OR_AGE_AMBIGUOUS`", class `RECENCY_ONLY`, or a new class, exit 2.
  - A pass still needs pct. A kill then needs both forms to reject.
  - Add a T12 case and the mutation "row 5b removed".
- **Alternative.** Keep the rule, and write the PM's confirmation into A2.8 in so many words:
  > "A kill may be carried by age attenuation of the pct form; this is accepted."
  - Delete "Whether that is intended is the PM's to confirm" from D.2 item 3.
  - Either way, the question must be closed before commit.

**M-3. What a pass means is overstated by omission: the gated statistic is at-query agreement, not future-demand agreement.**
- **Evidence.** In the A1.3 population on seed 0, the 517 positives break down as follows (`m3.log`):
  - 475 are A-cells, the queried assert at its own query step;
  - 10 (1.9%) are pending asserts;
  - 12 are queries.
- **Where the limit is stated.** A2v2:250-255 states this limit for H only. A2.4 and A2.12 do not state it for `AUROC_strat,pct`, and it applies equally there.
- **Why it matters.** The C10′ control that simply flags the answer slot scores 0.967-0.975, and `R-2026-09-27-retrieval-shown` has already ruled that arm B retrieves. So `AGREE` is close to implied by the retrieval ruling.
- **The exploit.** A pass would be read as "`r_i` tracks causal retention value", which RSR needs for *future* demand.
- **Edit.** Add a bullet to A2.4 after "Properties":
  > "**Scope of an AGREE.** On set E the positives are ≈ 92% A-cells (the queried assert at its own query step); ≈ 2% are needed after `t` (`m3.log`: 10 / 517 on seed 0). `AUROC_strat,pct` therefore certifies that `r_i` ranks the slot being retrieved *now* above same-age peers. It is not evidence that `r_i` predicts later demand; A2.6's limit applies to this statistic as well."

### MINOR

**m-1. The ruling status is misstated (A2v2:25, :430-431).**
- **What the draft says.** "Neither is committed in `docs/owner/rulings/`" and "Both files are in `owner-drafts/`, not in `docs/owner/rulings/`."
- **What git shows.** `R-2026-09-27-e0d-statistic.md` is committed on `main` at `7f9bb10` (`git ls-files` lists it).
- **Also missing:** `7f9bb10` is not an ancestor of `run/e0d` (`merge-base --is-ancestor` rc 1). The e0d worktree's rulings directory has none of the 09-27 rulings, so C8 on `run/e0d` would exit 3 until `main` is merged.
- **Edit.** Replace A2v2:430-431 with:
  > "The first ruling is committed on `main` (`7f9bb10`); the amended ruling is in `owner-drafts/` and is committed by Brendan. `run/e0d` must contain both (merge `main`) before the run; C8 checks this."
- Fix A2v2:25 in the same way.

**m-2. The date and authorship in Part A are wrong (A2v2:49, :51).**
- **Problem.** The heading says "Amendment 2 (2026-09-27, pre-data)" and the authorship line names only the 09-27 session. But m6 and m7, and every A2.4 and A2.11 number, were produced on 2026-09-29.
- **Edit.** Change the heading to "Amendment 2 (drafted 2026-09-27, completed 2026-09-29, pre-data)". Add to the authorship line:
  > "Completed 2026-09-29 by a second session of the same model (m6, m7)."

**m-3. The pre-data statement is narrower than a hostile reader will check (A2v2:78).**
- **Problem.** "No arm-B `r_i` has been computed on any document for this amendment" is true, but gated arm-B ckpt3000 `r_i` on docs `[0,1088)` (which includes set E) exists in `lookahead-room-r2` `D.pt`. Its age and kind profile was read before this design (brief §3.4). Anyone could compute the gate on set E from existing files.
- **Edit.** Add:
  > "Real gated arm-B `r_i` on `[0,1088)` exists (`lookahead-room-r2` `D.pt`; B0; E0e) and its age/kind profile was known when this amendment was written. No script joined it to any LOO cell, and no `AUROC` of real `r_i` was computed on any document (checked by grep over `reviews/e0d-a2*/`)."

**m-4. The τ population and the choice of q are under-declared (A2v2:105-107).**
- **Population.** The m5 code (`m5_tau_seeds.py:100`) takes off-A cells on all full-memory Q-steps, including rank M−1 and gap-1 steps, which is before the A1.3 exclusions. τ is then applied to the A1.3 population. The draft does not say so.
- **Choice of q.** q = 0.995 was fixed after `m4.log` showed critical counts at five candidate τ values on seed 0.
- **Edit.** Append to "The rule":
  > "(before the A1.3 exclusions; the same τ is applied to every population)"
- Add a sentence saying q was chosen from the label composition on seed 0 (`m4.log`), and that no `r_i` was involved.

**m-5. Cross-references are wrong.**
- A2v2:210 says the ceiling test is "(T1b)". It is T1c; T1b pins exactly 1.0.
- A2v2:210, :212 and :522 say the ceiling or the full calibration table "is in A2.11". It is in A2.4's own table (A2v2:214-231). A2.11 holds power only.
- **Edit.** Replace these references with "A2.4 table (T1c)".

**m-6. Scoring the §9 prediction against A2.8 is meaningless (A2v2:439-441).**
- **Problem.** §9 predicted CONFOUND on the ρ statistic with ρ\* = 0.5, a rule whose ceiling of 0.363 made that class near-certain. Scoring that prediction against the AUROC classification scores a different claim.
- **Edit.**
  > "§9's class prediction is scored against `label_A1` only; no class prediction is registered for the A2.8 classification."

**m-7. The bootstrap has no rule for an undefined replicate (A2v2:195-200).**
- **Problem.** "Undefined" is specified for the point estimate only. A resample could empty a bin's class. That is practically unreachable at n = 1024, but a rule is still needed.
- **Edit.** Add:
  > "A resample in which the statistic is undefined is counted and reported; if any occurs, the seed's label is computed on the defined resamples and the count is reported beside it."
- Alternatively, exit 3. Either works, as long as one is chosen now.

**m-8. The fixture's NaN assumption contradicts the data (A2v2:458-459).**
- **Problem.** B.1 says "0.8% NaN `d_resample` independent of `y`". On seed 0, A-cells are NaN at 11/663 = 1.7%, against 105/13,481 = 0.8% for non-A cells (`m5_out.json`). Seed 1 is 14/615.
- **Edit.** Make the fixture's NaN rate about 2× higher on A-cells. Keep a separate case where NaN is independent of `y`.

**m-9. The percentile is over 15 slots, not the 16 eviction candidates (A2v2:180-182).**
- **Problem.** The phrase "a property of the eviction candidates' scores" is imprecise. Under A1.3 the newest slot is excluded from the ranking, but FIFO or argmin eviction ranges over all 16.
- **Edit.** Say "of the A1.3-eligible slots' scores".

## Consistency checks (no finding)

- **Brendan's ruling.**
  - The 0.85 bar holds, per seed, through the 95% CI.
  - Every seed must pass, and pooled results never gate: A2.8 row 2, A2.10.
  - Gated `r_i` and the resample knockout are primary: A2.9.
  - "τ rule and age bins fixed in Amendment 2, before any data": A2.2 and A2.3. These are E0d data only; see m-3.
  - Moving 0.85 onto the pct form and demoting H rest on the PM's amended ruling under Brendan's delegation. The draft states that honestly (A2v2:64). M-1 closes the gap in enforcement.
- **CLAUDE.md.**
  - Nothing touches ψ̂. Age is not supplied to ψ̂.
  - There is no §15 content.
  - `docs/spec-corrections.md` is not edited (the `main` working tree is clean). Part E is a draft for Brendan and is labelled "not applied".
  - Nothing owner-only is enacted.
- **Cluster bootstrap.**
  - The unit is the document, which is correct: steps within a document share memory state, and seeds use disjoint generator keys.
  - One weight vector is shared by every statistic of a seed.
  - `pct` is fixed per step and invariant under whole-document resampling.
  - The per-seed pass rule (row 4, CI lower ≥ A\*, first match wins) is unambiguous.
  - The seed-level inference limit is stated at A2.10.
- **Exits.** The §8 table is exhaustive and first-match-wins. `AGREE, AGREE, DISAGREE` gives exit 2. CEILING counts as UNRESOLVED. The A1.4 exit discipline is kept.
- **Age control.** It is exact: the recomputation gives exactly 0.5 for any age-only score. D.2 item 2 is correct that the raw form also has this property, and that the pct form's distinguishing property is removing step-level concentration.

## Edits required before commit

1. **M-1:** C8 requires `R-*-e0d-statistic-amended*`. Add the T14 case and the mutation.
2. **M-2:** add row 5b (preferred), or record the PM's acceptance in A2.8 and delete the open question in D.2 item 3.
3. **M-3:** add the "Scope of an AGREE" bullet to A2.4.
4. **m-1:** correct the ruling status (A2v2:25, :430-431). Note that `run/e0d` must merge `main`.
5. **m-2:** heading date and authorship.
6. **m-3:** disclose the existing real `r_i` (`D.pt`).
7. **m-4:** state the τ population (before A1.3) and the provenance of q.
8. **m-5:** fix the cross-references (T1c; A2.4 table).
9. **m-6:** score §9 against `label_A1` only.
10. **m-7:** a rule for undefined bootstrap replicates.
11. **m-8:** the fixture's NaN rate depends on `y`.
12. **m-9:** "A1.3-eligible slots" wording.

VERDICT: MERGE-READY WITH EDITS (edits 1-12 above; 1-3 are required, 4-12 are required text fixes of low risk)
