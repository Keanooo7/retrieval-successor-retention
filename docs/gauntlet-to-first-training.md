# RSR — the gauntlet to first training

**For:** the Mac Studio session · **Written:** 2026-09-17
**Standing context:** `~/rsr-studio-brief-2026-09-17.md` (read it first if you have not)

Your objective is a **valid** first training run, not a fast one. Every check below exists because
something specific would otherwise be discovered in week 6 instead of today.

## The contract — how to report, every time

- 🔴 **Literal command output, or it did not happen.** `39 passed, 8 skipped` is a result. "Tests
  pass" is an adjective.
- 🔴 **A skipped test is not a passing test.** Report them separately, always.
- 🔴 **"Did not run" ≠ "found nothing".** If a check could not execute, say so and stop. Exit 3 is
  not exit 0. This has already produced two false clean bills of health on this project.
- 🔴 **A new check is not believed until a mutation has shown it red** — and the mutation must redden
  *only* it. If nothing reddens it, **the check adds nothing and that is the finding.**
- 🔴 **A negative result must say where it searched.** "Found nothing" is unfalsifiable.
- **Stop at the first BLOCKER.** Do not continue past a red gate and do not fix it later.
- **A deferred item is not a failure and must not be dressed as a success.**
- Use `/opt/homebrew/bin/git`, never bare `git`. Apple's refuses to run and fails silently.

---

# PHASE 0 — Repair. Four of these would invalidate the science silently.

A 10-agent audit ran against commit `83bdf57` on 2026-09-17. These are its findings. Verify each
before fixing — do not take them on trust.

**0.1 🔴 `reduction_to_tg()` sets `t_warm = inf`, which makes E0b vacuous.**
The dispatch is *`t < T_warm` → eviction is FIFO*. `inf` makes that true for every step, so under the
§3.7 reduction config **no eviction ever reaches the score** `argmin[z(ψ̂) + b − ν·cos]`. E0b would
certify FIFO against FIFO and pass.
→ Set `t_warm = 0.0` in the reduction config.
→ **PASS:** a test that asserts the reduction agrees with FIFO **through the score path**, plus a
  mutation (set `psi_source` to the learned head) that reddens it.

**0.2 🔴 The default `RSRConfig` is also `t_warm = inf`.**
A default-constructed RSR policy is stock TG forever while reporting `name = "rsr"`. The arm and its
control silently become the same arm.
→ **PASS:** constructing the default policy and running 100 evictions produces at least one decision
  attributable to the score, not the warmup branch.

**0.3 🔴 `RSRConfig` hardcodes `nu = 0.0` and `beta = 1.0`.**
Both are **MEASURED** (from E1) and the registry refuses them by name. `nu = 0.0` is also the §3.7
*disabled* value. This is D-1's exact shape one import away from the module built to prevent it.
→ **PASS:** `grep -rn` over `src/` finds no literal assignment of any MEASURED or DERIVED constant,
  and constructing a training config on an empty ledger **raises** `UnmeasuredConstant`.

**0.4 🔴 LRU is silently FIFO.** Its state lives in `MemoryState`, so a fresh state per step erases
it. §7.1 makes "RSR must beat LRU" the behavioural vacuity test and §10.1 makes LRU one of E7's two
legitimate controls — so the comparator is crippled in RSR's favour.
→ **PASS:** a test driving a real eviction loop where LRU and FIFO **choose different slots**. If
  they cannot be made to differ, LRU is not implemented.
→ Root cause: the shared protocol has no write/admission hook, so no policy is told a slot was
  overwritten. **H2O will inherit this** — its accumulated attention must zero for a new occupant.

**0.5 The registry has no traffic.** Nothing in `src/` calls `constants.get()`.
→ **PASS:** at least one non-test module reads it, and a test proves the config path raises on an
  empty ledger.

**0.6 `test_reduction.py`'s off-switch test is skipped under a reason false for it.** It needs no TG
and would pass today, so the §3.7 off-switch contract is enforced by nothing that runs.
→ **PASS:** un-skipped, green, and a mutation adding an eviction term without an off-switch reddens it.

**0.7 `preregistration/e0i_threshold.md` does not exist here**, though `GATE-1.md` cites it as
evidence. It **does** exist in the MacBook repo — take that copy, including its `p²` errata.

---

# PHASE 1 — Prove what already exists. No new code until this is green.

**1.1** `ruff check` and `ruff format --check` both clean. Report the file counts.
**1.2** Full suite with an **explicit skip census**: `pytest -q -rs`. Every skip has a reason that is
true *for that test*. **Zero unexplained skips.**
**1.3** CI emits a machine-readable test count, and **an all-skipped suite is RED.** Today
`pytest -q` exits 0 with no live tests, so the gate cannot notice the suite emptying.
**1.4** `ψ̂`'s `forward()` matches §3.2.2's formula computed by hand, with §4.3's multipliers applied.
Currently **no test exercises the arithmetic at all** — `ψ̂` could return zeros and stay green.
**1.5** Age cannot reach `ψ̂` — by signature *and* numerically (§3.2.2).
**1.6** μP: the `1/d` multiplier is applied **and** the value head has its **own parameter group**
(§4.3). The group does not exist today and `build_param_groups` raises.
**1.7 Mutation battery.** For every gate, name the mutation that reddens only it. Record the table.
Any gate no mutation reddens is reported as **adding nothing**.

---

# PHASE 2 — Build the missing, each with its acceptance criterion

**2.1 Vendor the pinned TG reference** (`third_party/PINS.md`). Read-only. **JAX is never installed
on the Mac** — there is no Metal path; its last release was 2024-10-08.
→ **PASS:** the vendored tree's commit **equals the pin**, verified, not assumed.

**2.2 Commit ADR-0002's fidelity tolerance BEFORE generating any fixture.**
→ **PASS:** `git log` shows the tolerance commit **precedes** the fixture commit. That ordering is
  the whole point; a tolerance chosen after seeing the numbers is not a tolerance.

**2.3 Golden tensors, including GRADIENTS.** TG writes gestalts to memory *without detaching the
graph*; JAX functional autodiff and PyTorch retained-graph semantics diverge exactly there.
→ **PASS:** forward *and* gradient fixtures, plus a **positive control** — a deliberately detached
  variant must **fail** the gradient check. A forward-only match with wrong graph retention survives
  to week 7.

**2.4 Transcribe TG → PyTorch at `d = 128`.** The long pole, 3–5 days.
→ **PASS:** `test_fidelity.py` green on forward and gradients, within the pre-committed tolerance.

**2.5 The synthetic corpus generator.** 🔴 **It does not exist in either repo**, and E1/E2 are built
on it. The Studio handoff's file tree lists `src/rsr/data/{synthetic,coref,pg19}.py` — **that
directory is not there.**
→ **PASS:** generates documents with a fact at *i* queried at *i+k*, max gap 40, ground-truth demand
  known per slot per step; **byte-identical across two separate processes** at the same seed; and a
  different seed produces different bytes (the negative control).

**2.6 E0b — bit-exact reduction against the PyTorch TG.**
→ **PASS:** identical loss curve, with `φ` constructed **after** the model or from a separate RNG
  stream. ⚠️ If it fails, check the RNG trap first — instantiating the head consumes draws, shifts
  data order, and fails for a reason unrelated to the mechanism.

**2.7 E0c — capacity, on the RENTED 48 GB card, not the Studio's 64.**
→ **PASS:** a committed `(S, d, batch)` triple measured at full memory on the target device, plus
  measured sentences/sec **at the widths that will actually run**. Do not inherit the spec's 21
  sent/sec — that was measured at `d_model=768` / 85.6M params.
⚠️ The Studio has *more* memory than the rental card. **Do not let a Mac-side batch size leak into
the E3 config.**

---

# PHASE 3 — The hour before the first long run

**3.1** Tree committed; the run stamped with a real sha, the device, and the seed set.
**3.2** Interpreter and torch version match the pinned ones.
**3.3** Suite green with **zero skips** and an asserted count.
**3.4** Every MEASURED/DERIVED constant is either **logged with provenance** or **defaulted with a
written reason**. No silent defaults.
**3.5** The run config has **traffic through the registry** — zero hardcoded rows. Dump it and read it.
**3.6** Checkpoint completeness: every stateful object round-trips, **policy state included**, and
the write survives a `SIGKILL` mid-save.
**3.7** Mid-stream resume reproduces the uninterrupted run exactly.
**3.8** Heartbeat logs enough that hour 1 is informative — loss, the attention-share Gini collapse
monitor, and the per-eviction decision attribution (§3.4 requires it **from the first policy run**).
**3.9 Gauntlet self-test.** Every check above has been observed red at least once. Report the full
list with its verdict. **A check never seen red is reported as unproven, not as passed.**

---

# What needs Brendan — do not decide these

1. **Pin `c_t`: raw gestalt or running context vector?** TG L2-normalises gestalts to unit norm, so
   the two have different norms and therefore **different correct μP multipliers**. **E0a cannot run
   until this is pinned** — it would measure an underspecified object.
2. **ADR-0002** — every decision cell is empty, and it gates 2.2, 2.7 and §16 condition 8.
3. **Sign the pre-registration**, after reconciling the two derivations (both agree the coref
   attenuation exponent is `p²`; they disagree on the number).
4. **§15 stays untouched.** It must not be filled by a model. Neither session has. Keep it that way.

---

**Work Phase 0 → 1 → 2 → 3 in order. Stop at the first BLOCKER and report. Do not start a training
run until 3.9 is green.**
