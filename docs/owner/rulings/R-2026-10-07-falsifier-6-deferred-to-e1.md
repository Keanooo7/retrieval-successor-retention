---
id: R-2026-10-07-falsifier-6-deferred-to-e1
date: 2026-10-07
stated_in: 'Interactive PM session, 2026-10-07, through the PM question form. Brendan, verbatim, in the free-text answer: "Recorded outcome: SUPERSEDES against Run 1 $\psi C$3. Ruling on Falsifier 6 is DEFERRED to Gate E1 per specification §161. The Expire-Span baseline is pinned to configuration L48-k1e-2 on the pre-registered EVAL dataset5. At Gate E1, RSR Run 2 (LOO target) must beat both FIFO and pinned L48-k1e-2 across all seeds". The "3", "§161" and "5" are pasted citation markers. The spec places falsifier 6 in spec §6, E1 row (rsr_model_spec_v0.5.md:555), and in §5.4, Expire-Span row (:516).'
evidence: 'Expire-Span referendum, Amendment 2 re-run at 99646cb: class SUPERSEDES, rsr-verifier CONFIRMED 4/4, record 3a241fc on verify/expire-span-referendum. ES − ψC(run 1) +0.211 / +0.182 / +0.165; ES − FIFO +0.006 / −0.046 / −0.004.'
drafted_by: 'Claude Opus 5.5 (PM session). NOT written by Brendan.'
---
# Falsifier 6: SUPERSEDES recorded, ruling deferred to E1, baseline pinned

## Ruling

1. **Recorded outcome.** The Expire-Span referendum's class is **SUPERSEDES**, against **RSR run 1's ψC**, the gated `r_i` target. That class stands as measured and is not reinterpreted.
2. **Falsifier 6 is not ruled now.** It is **deferred to gate E1**, where the spec places it (§6, E1 row: "If Expire-Span ≥ RSR, stop and write a different thesis in week 5").
3. **The Expire-Span baseline is pinned before any run-2 data.**
   - **Config:** `L48-k1e-2`, which is max span 48 and α 1e-2. It was selected on VAL by the referendum's pre-registered rule.
   - **EVAL:** the referendum's pre-registered EVAL, documents `[995000, 997048)`. This is run 1's EVAL: 2048 documents per seed, held-out batch 16, eval mode.
   - **Checkpoints** (step 3000, the referendum's trained weights). Pinned copies are read-only at `~/rsr-substrate/2026-10-07-es-pinned/` with `MANIFEST.sha256`.
     - seed 0: `f98411a469c4a00615adb74de6b892088dcdb45425e723f309e25705326ede6d`
     - seed 1: `0bd50bb07a9e30efac9fc291a35591720b9c2c1be94937731556cddcbc360861`
     - seed 2: `8d1f4f52ee78b34d35306bdf7e46bf84a8476ff254aa267f80d6e06437e70fd2`
     - Source: `.worktrees/es-referendum-a2-run/runs/expire-span-referendum/L48-k1e-2/seed{0,1,2}/ckpt-003000.pt`, from run head `99646cb`.
   - E1 reads these. It does not retune or reselect Expire-Span.
4. **The bar at E1.** RSR run 2 (C3, the LOO target; `R-2026-10-07-c3-loo-target`) must **beat both FIFO and the pinned L48-k1e-2 on every seed**.
   - E1's PREREG defines "beat" numerically before E1 runs. It must not be weaker than the spec's "RSR ≥ Expire-Span".

## What this ruling does not do

- It does not say falsifier 6 failed to fire. If RSR run 2 does not beat the pinned Expire-Span at E1, falsifier 6 applies there in full.
- It does not change the referendum's PREREG, its class, or its numbers.
- It does not touch §15.
