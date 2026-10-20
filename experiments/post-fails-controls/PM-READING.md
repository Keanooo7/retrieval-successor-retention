# PM reading: phase 1 (post-FAILS controls), 2026-10-20

Written by a model, the PM session executing PLAN-2026-10-19. It is not written by Brendan. It is
committed separately from the results, per handoff §4 step 4.

This reading applies each item's **registered** rule from `PREREG.md` (committed alone at `9f22d87`;
amendments 1a-S, 1bcde-S and G-C1e) and adds nothing the rule does not license. This file types no
number: every value is in the cited RESULTS file or ledger.

| item | exit | registered outcome | verified | RESULTS |
|---|---|---|---|---|
| 1a | 0 | FactFiller **WIN on all 3 seeds**, so the registered reading is "the failure is in learning". Kind-oracle LOSS on seed 1, which never gates. | `verify/post-fails-controls-1a` 2f62a93 | `RESULTS-1a.md` (branch `build/pfc-1a`) |
| 1b | 3 | **NOT READ.** Control B-C5 failed on seed 0. | `verify/post-fails-controls-1b` f589114 (the failure is reproduced on an independent path) | `RESULTS-1b.md` |
| 1c | 0 | **DROPPED.** "Finding 1" does not survive. | `verify/post-fails-controls-1c` 848ee00 | `RESULTS-1c.md` |
| 1d | 0 | Descriptive. No label. | `verify/post-fails-controls-1d` c39df18 | `RESULTS-1d.md` |
| 1e | 0 | Descriptive. No label. | `verify/post-fails-controls-1e` 72090da | `RESULTS-1e.md` |

Tier D review: `review/post-fails-controls-b746139` e472120, MERGE WITH FIXES. LB-1 (1b printed its
labels despite exiting 3) was fixed in 45b5436 / 0a3ed09 / 777d8d5, and the PM checked it narrowly in
round 2. FU-1 was fixed in c17f741. The other 13 follow-ups are batched.

## Item by item

**1a.**
- Gate P2 was applied mechanically by the PM: no FactFiller LOSS on any seed, so **phase 2a runs**
  (DIGEST, "GATE P2").
- Under the registered reading, a legal content-only fixed rule beats FIFO through this frozen reader
  on EVAL on every seed. So run 1's and run 2's failure is located on the learning side, not in the
  reader (D2).
- Caveats that bind any use of this:
  - EVAL is already used.
  - The U analogue was already SEEN (retention-readability). This is a replication on new documents,
    not a discovery (PREREG §11.1).
  - It does not say which part of learning failed.

**1b.**
- No label is read.
- For Gate A, as facts, not readings:
  - B-C5's shuffled-label probe lies **below** 0.5, on two independent paths. Leakage does not push
    in that direction.
  - The shuffled fit hit the iteration cap without converging.
  - A descriptive check by the verifier, outside the registered control, found other shuffles whose
    CIs contain 0.5.
- **Owner question:** re-run 1b under a re-specified B-C5 (a permutation null over several
  shuffles), labelled as a post-data amendment, or leave 1b NOT READ.

**1c.**
- DROPPED.
- 15 of the 18 conditions hold. All three that fail are "r_i's bottom-k CI upper bound ≤ chance",
  and they fail on seed 0 (k = 1 and k = 3) and seed 1 (k = 1).
- That is, on those seeds r_i's bottom-k agrees with demand strictly after the decision step
  **above** chance. The registered consequence is that finding 1 ("attention ranks the top well and
  misranks the bottom") is **dropped from the paper**.
- The prior-art survey (`docs/lab-notes/prior-art-eviction-signals.md`, branch
  `docs/prior-art-eviction-signals` 7ec78ca) found the asymmetry NOT FOUND as a direct measurement in
  the literature. 1c does not supply it.
- **Kill rule:** 1c DROPPED is one half of PLAN §2's kill condition. The other half is phase 2's
  outcome.

**1d (descriptive; gates nothing).**
- FIFO's model-free strength depends on the question-spacing mix.
- Across the registered grid, FIFO's hit rate falls as `heavy_tail_weight` rises, and FactFiller's
  advantage over FIFO grows with it.
- At least one grid point with no heavy tail has FactFiller below FIFO.
- Base-point and per-cell values: `RESULTS-1d.md`. This is residency, not accuracy: the reader was
  trained on the base mix only.

**1e (descriptive; gates nothing).**
- The interaction share, Σ|I| / Σ|Δ_G|, is **largest for G2 fact pairs**, i.e. an assert together
  with its query sentence. It is larger at query steps than at other steps.
- It is smallest for duplicate fillers. Random sets of 4 or 8 exceed random pairs.
- Single leave-one-out therefore does not price a fact pair as the sum of its two singles. Table:
  `RESULTS-1e.md`.
- Single-step Δ only. No claim about trajectories or natural text.

## What goes to Gate A from phase 1

- 1a: learning-side, with its caveats.
- 1b: NOT READ, plus the B-C5 owner question.
- 1c: DROPPED.
- 1d and 1e: descriptive tables.
- Follow-ups list: review file `docs/reviews/post-fails-controls/b7461397ce69.md`.
