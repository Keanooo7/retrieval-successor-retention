# RSR — the gauntlet to first training (complete; no pending decisions)

**For:** the Mac Studio session · **Revised:** 2026-09-17 (supersedes the earlier copy)
**Standing context:** `~/rsr-studio-brief-2026-09-17.md`

Your objective is a **valid** first training run, not a fast one. **Every decision that was
outstanding has been made below.** Nothing here should stop you to ask a question. Where a decision
is the owner's to overrule, a default is given and the work proceeds on it — flag it in the return,
do not block on it.

## The contract — how to report, every time

- 🔴 **Literal command output, or it did not happen.** `39 passed, 8 skipped` is a result. "Tests pass" is an adjective.
- 🔴 **A skipped test is not a passing test.** Report them separately, always.
- 🔴 **"Did not run" ≠ "found nothing".** Exit 3 is not exit 0. Two false clean bills of health have already been produced on this project by ignoring that.
- 🔴 **A new check is not believed until a mutation has shown it red** — and the mutation must redden *only* it. If nothing reddens it, **the check adds nothing and that is the finding.**
- 🔴 **A negative result must say where it searched, and with what command.** One confident "not found" on this project proposed a 3–6 week reimplementation for code that existed.
- **Stop at the first BLOCKER.** Do not continue past a red gate.
- **A deferred item is not a failure and must not be dressed as a success.**
- Use `/opt/homebrew/bin/git`, never bare `git` — Apple's refuses to run and fails silently.

---

# DECISIONS — all made. Do not reopen; implement.

**D-A · Build against the CODE, not the paper.** They diverge, and the README says so: *"This version
differs slightly from the version of the model described in (arXiv:2512.25026)."* The fidelity harness
compares against the released code's tensors — **you cannot validate fidelity against a paper.** So the
code is the reference by construction. Record every divergence you find in `docs/code-vs-paper.md`.

**D-B · The gestalt is L2-normalised to unit length. This is settled by reading the source.**
`tg/models/tg_srep_head.py`: `srep_BxD = raw_BxD / denom_Bx1 * cfg.srep_norm_target`, with
`srep_norm_target: float = 1.0` in `tg_config.py`. The file's own comment calls it equivalent to
`F.normalize(v, p=2, dim=0)`. **‖s‖₂ = 1.0 exactly; coordinates are O(1/√d), not Θ(1).**

⚠️ The paper describes **no** normalization — `s_t = W_sent H^(ℓs)_iEOS`, and zero hits for any
normalization of the gestalt in 20 pages. The paper's line *"s_t lives in the same d-dimensional
space as token hidden states"* is a claim about **dimensionality, not norm**; do not read a magnitude
guarantee out of it. This is a genuine code-vs-paper divergence and D-A resolves which side wins.

**Consequence:** §4.3's `1/d` multiplier was derived assuming Θ(1) coordinates. Under unit norm the
trained-regime output is Θ(1/d) — it decays with width. **Do NOT change the multiplier.** §15.3 makes
that derivation the author's. E0a measures it under both input regimes and reports; the ADR follows
the measurement.

**D-C · `c_t` is the CURRENT SENTENCE GESTALT.** Not a running context vector.
Three reasons: it is the simplest reading of §3.2.2; it makes both arguments to the bilinear form
unit-norm, so the μP analysis is single-valued instead of forked; and the running-context variant
becomes a clean ablation rather than a branch in the critical path. **Write it into the config as a
named enum with one value implemented and the other raising.** Owner may overrule; the work proceeds.

**D-D · `P^(sent)` stays RANK-indexed, and the confound it creates is MEASURED, not argued.**
🔴 **This is the most important item in this document. Read it twice.**

§2.2: `K_M = s_{t−Mt},…,s_{t−1} + P^(sent)_{1:Mt}` — sinusoidal, **keys only**, `V_M` gets none — over
a memory *"ordered from oldest to most recent."* **So `P` indexes rank in the memory ordering, not
absolute age.**

Under FIFO rank and age coincide, so `P` means recency. **Under RSR, evicting a middle slot shifts
the ranks of every slot behind it**, so a slot's positional encoding becomes a function of *which
other slots the policy killed*. RSR and FIFO then differ in more than the eviction rule — the exact
property E0b exists to guarantee.

🔴 **And E0b structurally cannot catch it.** Under §3.7's reduction the policy *is* FIFO, so rank and
age-order coincide and the test passes. **The confound is invisible in the reduction and present in
every arm you care about.**

**Decision:** keep rank-indexing (it matches TG, which the fidelity harness compares against), and
make the confound observable:

1. **Log, per eviction, how many slots' ranks shifted.** Report the distribution in every RSR run.
2. **Add a `P^(sent)`-ablated arm** — positional encoding zeroed — to E1. It bounds the confound's size.
3. **Add an absolute-age-indexed variant behind a config switch**, default off. It must be off in the
   §3.7 reduction or the reduction breaks.
4. Record all of this in `docs/decisions/ADR-0006-sentence-positional-encoding.md` **before the
   policy is written.**

Two consequences that follow and must go in the writeup:
- The cross-attention logit contains a positional term ψ̂ **structurally cannot have** (§3.2.2 excludes
  age). This **lowers the E0h collinearity prior** and is independent support for D-C.
- `r_i` is a softmax over logits that already contain `P^(sent)`, so **the retention target is itself
  partly a function of slot position** — while §7.1 gates on whether the *score* is a function of age.
  **State this in §7.1's write-up**: the gate must not convict the estimator of a contamination the
  architecture put in the target.

**D-E · `r_i` includes the memory gate, and the profile has SIX entries.**
```
r_i(t) = Σ_{l,h} ‖ g_mem^(l) · α_{l,h,i} · W_O^(l,h) v_{l,h,i} ‖₂
```
Cross-attention layers are `ℓ ∈ {2,4,6,8,10,12}` — **six, not twelve.** D-6's argument for `W_O`
("precisely where head-specific rescaling lives") applies verbatim to `g_mem`, which is where
*layer*-specific rescaling lives, and App. C measures the gates **growing over training and larger in
deeper layers** — so the weighting is non-stationary. **Compute `r_i` both ways** (gated and raw) and
report both against LOO Δloss in E0d. §3.2.1's truth rule is unchanged: if they disagree, **LOO is truth.**

**D-F · `r_i` is collected in EVAL mode.** `attn_dropout: float = 0.2` in `tg_config.py`, so `α` is
stochastically zeroed during training and an on-policy `r_i` would be noisy in a policy-relevant way.
Eval mode for the retention target; train mode for the LM loss. **Write it into the code as an
explicit mode switch, not an ambient default.**

**D-G · The E0i threshold is reconciled. Use this.**
Both derivations agreed the coref attenuation exponent is **`p²`, not `p`**. They disagreed on the
number; the per-bin constraint binds, and the pooled figure is subsumed:

> **PASS: `n_raw × p² ≥ 150` in EVERY ONE of `(40,48]`, `(48,56]`, `(56,64]`, AND `n_raw × p² ≥ 600`
> pooled across `(40,64]`.**

At `σ_d = 0.30` that is a **2.0% per-bin** minimum detectable effect and **1.0% pooled** — against
[P2]'s own 2–4% headline effect, which is the only benchmark available. The Studio's pooled `≥ 400`
(1.2%) is subsumed by the pooled 600. **A pooled-only gate can hide an empty top bin**, and the top
bin is where the claim lives — that is why per-bin binds.
`p` is measured on 100 hand-annotated reintroductions. **Unmeasured `p` → exit 3, did not run. Not a pass.**
`σ_d = 0.30` is an assumption; when a real paired run produces one, recompute the MDE and **report it
beside the headline.** The event threshold does not move retroactively.

**D-H · Fidelity tolerances, committed before any fixture exists.**
Float32. Forward: `atol = 1e-5, rtol = 1e-4`. Gradients: `atol = 1e-4, rtol = 1e-3` — gradients
accumulate more error and JAX/PyTorch reduction orders differ. **If the transcription cannot meet
these, the ADR records the achieved value and why. It does not silently relax.**
`git log` must show the tolerance commit **precedes** the fixture commit.

**D-I · `T_warm` has one representation: a float number of steps.** The registry currently holds a
string and `RSRConfig` a float, with nothing converting between them. Pick the float, make the
registry hold it, and delete the string.

---

# PHASE 0 — Repair. Four of these would invalidate the science silently.

From a 10-agent audit of `83bdf57`, 2026-09-17. **Verify each before fixing — do not take it on trust.**

**0.1 🔴 `reduction_to_tg()` sets `t_warm = inf`, making E0b vacuous.** Dispatch is *`t < T_warm` →
FIFO*; `inf` makes that true forever, so under the §3.7 reduction **no eviction ever reaches the
score**. E0b would certify FIFO against FIFO.
→ Set `t_warm = 0.0`. **PASS:** a test asserting the reduction agrees with FIFO **through the score
path**, plus a mutation (switch `psi_source` to the learned head) that reddens it.

**0.2 🔴 The DEFAULT `RSRConfig` is also `t_warm = inf`** — a default-constructed RSR policy is stock
TG forever while reporting `name = "rsr"`. Arm and control silently become one arm.
→ **PASS:** 100 evictions from the default policy produce at least one decision attributable to the score.

**0.3 🔴 `RSRConfig` hardcodes `nu = 0.0` and `beta = 1.0`** — both MEASURED (E1), both refused by the
registry by name, and `nu = 0.0` is the §3.7 *disabled* value. **This is D-1's exact shape one import
from the module built to prevent it.**
→ **PASS:** `grep -rn` over `src/` finds no literal assignment of any MEASURED/DERIVED constant, and
building a training config on an empty ledger **raises `UnmeasuredConstant`**.

**0.4 🔴 LRU is silently FIFO** — its state lives in `MemoryState`, so a fresh state per step erases
it. §7.1 makes "RSR must beat LRU" the behavioural vacuity test and §10.1 makes LRU one of E7's two
legitimate controls. **The comparator is crippled in RSR's favour.**
→ **PASS:** a real eviction loop where LRU and FIFO **choose different slots.** If they cannot be made
to differ, LRU is not implemented.
→ Root cause: the protocol has no write/admission hook, so no policy is told a slot was overwritten.
**H2O will inherit this** — its accumulated attention must zero for a new occupant. Fix the protocol.

**0.5 The registry has no traffic** — nothing in `src/` calls `constants.get()`.
→ **PASS:** at least one non-test module reads it, and a test proves the config path raises on an empty ledger.

**0.6 `test_reduction.py`'s off-switch test is skipped under a reason false for it** — it needs no TG
and would pass today, so the §3.7 off-switch contract is enforced by nothing that runs.
→ **PASS:** un-skipped, green, and a mutation adding a term without an off-switch reddens it.

**0.7 Registry defects**, all verified: scope-free MEASURED reads leak across scopes; `record()` accepts
DERIVED names `get()` then ignores; `A_max`'s guard trusts a caller-supplied `S`; `A_max` with `S=None`
raises a bare `TypeError` outside the `ConstantError` hierarchy; `K` accepts a measurement from any
experiment name; the scope vocabulary has no entry for E4/E7/diagnostics; `git_sha` is caller-typed and
never verified. **Fix all seven. Stamp the sha from git inside the measured tree — never accept it as
an argument.**

**0.8 `preregistration/e0i_threshold.md` does not exist here**, though `GATE-1.md` cites it as evidence.
→ Take the MacBook repo's copy, including its `p²` errata, and apply **D-G**.

---

# PHASE 1 — Prove what exists. No new code until green.

1.1 `ruff check` and `ruff format --check` both clean; report file counts.
1.2 Full suite with an explicit skip census (`pytest -q -rs`). Every skip reason true *for that test*. **Zero unexplained skips.**
1.3 CI emits a machine-readable count, and **an all-skipped suite is RED.** Today `pytest -q` exits 0 with no live tests.
1.4 `ψ̂.forward()` matches §3.2.2's formula computed by hand, with §4.3's multipliers applied. **No test exercises the arithmetic today** — `ψ̂` could return zeros and stay green.
1.5 Age cannot reach `ψ̂` — by signature **and** numerically.
1.6 μP: the `1/d` multiplier is applied **and** the value head has its **own parameter group**. The group does not exist and `build_param_groups` raises.
1.7 **Mutation battery.** Every gate gets the mutation that reddens only it. Record the table. Any gate no mutation reddens is reported as **adding nothing**.

---

# PHASE 2 — Build what is missing, each with its acceptance criterion

**2.1 Vendor the pinned TG reference.** Read-only. **JAX is never installed on the Mac** — no Metal path; `jax-metal`'s last release was 2024-10-08.
→ **PASS:** the vendored tree's commit **equals the pin**, verified.

**2.2 Commit D-H's tolerances** before generating any fixture. → **PASS:** `git log` shows the ordering.

**2.3 Golden tensors, forward AND gradients.** TG writes gestalts **without detaching the graph**; JAX functional autodiff and PyTorch retained-graph semantics diverge exactly there.
→ **PASS:** both, plus a **positive control** — a deliberately detached variant must **fail** the gradient check. A forward-only match with wrong graph retention survives to week 7.

**2.4 Transcribe TG → PyTorch at `d = 128`.** The long pole, 3–5 days. → **PASS:** `test_fidelity.py` green on forward and gradients within D-H.

**2.5 The synthetic corpus generator.** 🔴 **It exists in neither repo**, and E1/E2 are built on it. The Studio handoff's file tree lists `src/rsr/data/{synthetic,coref,pg19}.py` — **that directory is not there.**
→ **PASS:** a fact asserted at *i* and queried at *i+k*, max gap 40, ground-truth demand known per slot per step; **byte-identical across two separate processes** at one seed; a different seed gives different bytes.

**2.6 E0b — bit-exact reduction against the PyTorch TG.**
→ **PASS:** identical loss curve, `φ` constructed **after** the model or from a separate RNG stream.
⚠️ On failure check the RNG trap first — instantiating the head consumes draws and shifts data order, failing for a reason unrelated to the mechanism.

**2.7 E0c — capacity, on the RENTED 48 GB card.** §4.2 sizes against 64 GB (the Mac); E3 runs on 48.
→ **PASS:** a committed `(S, d, batch)` triple measured at full memory **on the target device**, plus sent/sec at the widths that will run. **Do not inherit the spec's 21 sent/sec** — measured at `d_model=768` / 85.6M.
⚠️ The Studio has **more** memory than the rental card. **Do not let a Mac-side batch size leak into the E3 config.**

**2.8 Implement D-D's instrumentation** — rank-shift logging, the `P^(sent)`-ablated arm, the config-gated absolute-age variant.

---

# PHASE 3 — The hour before the first long run

3.1 Tree committed; the run stamped with a real sha, the device, and the seed set.
3.2 Interpreter and torch version match the pinned ones.
3.3 Suite green with **zero skips** and an asserted count.
3.4 Every MEASURED/DERIVED constant **logged with provenance** or **defaulted with a written reason.** No silent defaults.
3.5 The run config has **traffic through the registry** — zero hardcoded rows. Dump it and read it.
3.6 Checkpoint completeness: every stateful object round-trips, **policy state included**, and the write survives `SIGKILL` mid-save.
3.7 Mid-stream resume reproduces the uninterrupted run exactly.
3.8 Heartbeat logs enough that hour 1 is informative — loss, attention-share Gini (the collapse monitor), per-eviction decision attribution (§3.4 requires it **from the first policy run**), and D-D's rank-shift distribution.
3.9 **Gauntlet self-test.** Every check above observed red at least once. Report the full list with verdicts. **A check never seen red is reported as unproven, not passed.**

---

# The only two things that are not yours

1. **GPU procurement** — it spends money. Research and draft; **Brendan sends and signs.** Note for the
   brief: RunPod's savings plans are **prepaid and non-refundable**, so the "cancellable hold" the plan
   assumes may not be a purchasable product. An enterprise quote is free and is the only route where a
   deferrable term is negotiable. **This blocks 2.3 and 2.7 only** — everything else proceeds without it.
2. **§15 stays untouched.** Not filled, not restated, no candidates suggested. It says it must not be
   filled by a model. Neither session has. Keep it that way.

**Everything else above is decided. Work Phase 0 → 1 → 2 → 3. Stop at the first BLOCKER and report.
Do not start a training run until 3.9 is green.**
