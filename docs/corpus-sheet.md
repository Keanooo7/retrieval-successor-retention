# RSR corpus sheet — current state, for the advising window

**Written:** 2026-09-17 · **For:** a MacBook Pro session helping Brendan question and prompt the
Mac Studio session. **Everything below was executed or read, not inferred.** Where something was
taken on another session's word it says so.

---

## 1. What the project is, in one paragraph

Thought Gestalt (TG) is a language model that reads one sentence at a time and keeps a small
notebook of past sentences. When the notebook is full it throws out the oldest entry. **RSR replaces
that one rule** with a learned one: throw out whatever the model predicts it will need least, given
where the story currently is. The scientific question is whether a model trained only to predict the
next sentence rediscovers a 1978 theory of how people remember narrative. **The engineering delta is
explicitly secondary**, and the spec says so.

---

## 2. Machines and where everything lives

| | |
|---|---|
| **Mac Studio** — M4 Max, 64 GB, 16 cores, 736 GB free | `keanooo7@100.81.77.20`. The training machine. Repo at `~/retrieval-successor-retention`, pushed to `github.com/Keanooo7/retrieval-successor-retention` (private). |
| **MacBook Pro** — 16 GB | This machine. Advisory only. **No memory or throughput number measured here transfers.** Second repo at `~/Documents/GitHub/retrieval-successor-retention` — ✅ **given a remote 2026-09-18**; its 13 commits are pushed to branch `macbook-local-2026-09-18`. It has an **unrelated history** to `main` and is not mergeable by fast-forward. |
| Link | Tailscale + SSH key installed. `ssh keanooo7@100.81.77.20` works. Thunderbolt cable is plugged in but **unconfigured** (no address), so traffic relays the long way — ~218 ms between two machines on one desk. |

🔴 **Use `/opt/homebrew/bin/git` on the Studio, never bare `git`.** Apple's git refuses to run until
an Xcode licence prompt is accepted, and it fails in a way that looks like an **empty answer rather
than an error**. This has already produced one false "everything is clean" report.

---

## 3. Current state — measured

> 🔄 **Re-measured 2026-09-18** by the MacBook co-researcher session, against Studio
> `431b347` and MacBook `5c0fda7`. The figures below replace the 2026-09-17 ones, which
> were taken before the Studio's overnight run and understated it by a wide margin
> (they said Studio *1 commit / 66 files / 39 passed, 8 skipped*). Every number here is
> from a command run on the machine it describes, not carried across.

| | Studio | MacBook |
|---|---|---|
| tracked files | **176** | **50** |
| commits | **37** | **13** |
| tests | **287 passed, 0 failed, 0 skipped** | **43 passed, 1 skipped** |
| on GitHub | yes — `main`, in sync | **yes, as of 2026-09-18** — branch `macbook-local-2026-09-18` |
| TG reference vendored | **yes** — `third_party/ThoughtGestaltCode/`, 43 tracked files | no |
| a trained model | **no** | no |
| corpus / data module | **yes** — `src/rsr/data/{synthetic,pg19,coref}.py` | no |

**The two repos have unrelated histories.** Studio root `83bdf57`, MacBook root `05ec79c`;
`git rev-list --left-right --count origin/main...HEAD` returns `37  13` with no common
ancestor. They were scaffolded independently and **cannot be merged by fast-forward.** The
MacBook's 13 commits are preserved as their own branch; trunk is the Studio's `main`.

🔴 **The Studio can now train. The line "nothing in either repository can train anything
today" was true on 2026-09-17 and is false now** — `src/rsr/train/loop.py` exists and runs.
What it trains is a separate question, and a bad one: see the five confirmed defects in
`Projects/RSR/HANDOFF-2026-09-18.md` §5.

### What is real vs stub, on the Studio — re-measured 2026-09-18

Measured by `wc -l` and `grep -c NotImplementedError` on each file at `431b347`.

| Area | State |
|---|---|
| `constants.py` registry | **REAL.** 872 lines. Reading an unmeasured constant raises and names the experiment that owes it. |
| `retention/value_head.py` | **REAL.** 173 lines. |
| `retention/policy.py` | **REAL.** 260 lines. |
| `retention/rsr.py` | **REAL**, 456 lines, 3 `NotImplementedError` remaining. Was a stub on 2026-09-17. |
| `retention/reward.py` | **REAL.** 160 lines, 0 raises. |
| `retention/{shadow,bias}.py` | **STILL STUBS.** 46 and 72 lines, 3 and 4 raises. No shadow buffer, no bias loop. |
| `baselines/` | fifo / lru / random implemented. **h2o, expire_span, leading_edge, oracle are STILL STUBS** — 36–46 lines each, 4 raises each. |
| `mup/param_groups.py` | **REAL.** 113 lines, 0 raises. `build_param_groups` no longer raises. |
| `mup/coord_check.py` | **STUB** that raises — but its docstring carries the load-bearing E0a reading rule (read the coordinate scale after real optimizer steps, never at init). |
| `data/synthetic.py` | **REAL.** 315 lines. ⚠️ But it stores each query's answer out of band and never puts it in the token stream — HANDOFF §4. |
| `train/loop.py` | **REAL**, 228 lines, 0 raises — **and carries five confirmed defects** (HANDOFF §5). Nothing in `tests/` imports it except `checkpoint`. |
| `experiments/e0*/run.py` | **e0a, e0f, e0g still 11-line stubs that raise.** e0c has `measure.py`, `sweep.py`, `confirm.py`. |

---

## 4. 🔴 Known defects — ranked by what they would cost

Found by a 10-agent survey, 2026-09-17. **The top four would invalidate the science silently**, which
is worse than crashing.

> ⚠️ **Re-read 2026-09-18: this ranking predates the Studio's overnight run and has NOT been re-verified as a whole.** Defect 7 is measured-and-fixed (above). The rest were true of the 2026-09-17 trees; several name files that have since been rewritten (`retention/rsr.py` went 165 → 456 lines). **Re-measure before acting on any row here.** Defect 10 *was* reproduced live on 2026-09-18 and is real: `uv run pytest -q` on the MacBook prints a progress line and **no summary**, because `addopts = "-q"` plus an explicit `-q` makes `-qq`.

| # | Defect | Why it matters |
|---|---|---|
| **1** | `RSRConfig.reduction_to_tg()` sets `t_warm = inf`. The dispatch is *`t < T_warm` → eviction is FIFO*, so **inf makes that true forever.** | **E0b — a named kill gate — would certify FIFO against FIFO** without the eviction score ever executing. Green and meaningless. |
| **2** | The **default** `RSRConfig` also has `t_warm = inf`. | A default-constructed RSR policy is **stock TG forever while still reporting `name = "rsr"`.** The experimental arm and its control become the same arm, silently. |
| **3** | `RSRConfig` hardcodes `nu = 0.0` and `beta = 1.0` — both **MEASURED** constants the registry refuses by name. `nu = 0.0` is also the §3.7 *disabled* value. | This is defect **D-1's exact shape** (a frozen unmeasured constant governing a mechanism) reintroduced one import away from the module built to prevent it. |
| **4** | **LRU is silently FIFO** in both repos, by different causes. On the Studio its state lives in `MemoryState`, so a fresh state per step erases it. | §7.1 makes "RSR must beat LRU" the behavioural vacuity test, and §10.1 names LRU as one of E7's two legitimate controls. **The comparator is crippled in RSR's favour.** |
| **5** | **The registry has no traffic.** Nothing in `src/` calls `constants.get()` — only the CLI and tests. | A gate with nothing passing through it. The mechanical guarantee is opt-in, and the training loop can bypass it by simply not calling it. |
| 6 | `test_reduction.py`'s off-switch test is skipped under a reason that is **false for it** — it needs no TG and would pass today. | The §3.7 off-switch contract is enforced by nothing that runs. |
| 7 | ~~`preregistration/e0i_threshold.md` does not exist on the Studio~~ — 🔄 **FIXED, re-measured 2026-09-18.** It exists on the Studio at 337 lines and is the **current** version: it reconciles the two derivations (D-G), adds a pooled `≥ 600` floor, and tightens `p̂` to `p_LCB` (one-sided 95% Wilson bound). The MacBook's 191-line copy is the **stale** one. 🔴 It is **UNSIGNED** and says so — *"NOT IN FORCE UNTIL SIGNED"*. | Was a release condition with no artefact. Now an artefact awaiting a signature that only Brendan can give. |
| 8 | Studio's `Registry.record()` accepts `git_sha` as a caller-typed string and never verifies it. | Provenance can be fabricated. The MacBook version stamps the sha from git inside the measured tree. |
| 9 | Registry: scope-free reads of MEASURED constants leak across scopes; `record()` accepts DERIVED names that `get()` then ignores; the scope vocabulary has no entry for E4/E7/diagnostics. | Two sources of truth for `gamma_b` — the constant D-1 is *about*. |
| 10 | **MacBook only:** `measure_pytest_count()` can never return a number. `addopts = "-q"` plus an explicit `-q` makes `-qq`, which suppresses the summary line the parser reads. | **That ratchet is permanently dead**, and CI also fails `ruff format --check` on 6 files. Never noticed because the repo has no remote, so CI has never run once. |

---

## 5. Settled — do not reopen

All raised by Brendan, all examined, all resolve **in favour of the plan as written**.

| Question | Answer |
|---|---|
| Is 21M parameters enough? | **Yes. Stay at ≤21.3M (`d ≤ 384`).** The base paper's own largest model is 85.6M; there is no large TG to scale to. The comparison arms are the expense, not the model. |
| μP is width-only — what about depth? | **Correct, and it never bites.** Depth is fixed at 12 layers throughout. Every sweep is width-only, which is μP's proven regime. Depth-μP exists (Tensor Programs VI) and is not needed. |
| Is 3× a transfer limit? | **No — different claim.** 3× is too narrow to *fit a scaling law*. μP *transfer* went 40M → 6.7B in the original paper. |
| Could LoRA/QLoRA use a bigger base? | **No.** No pretrained TG exists to adapt, and RSR changes the forward-pass data path, not weights. 📌 The design already *is* parameter-efficient: only the ~150K-parameter value head gets the retention gradient. |
| How to use the Studio's 64 GB fully? | **Longer streams and bigger batches, not more parameters.** The memory bill is the retained computation graph across the stream. |

---

## 6. Owner decisions still outstanding — no agent may make these

1. **Pin `c_t`: raw sentence gestalt, or running context vector?** §3.2.2 allows either. TG L2-normalises
   gestalts to unit length, so the two have **different norms and therefore different correct μP
   multipliers**. **E0a cannot run until this is pinned** — it would be measuring an underspecified object.
2. ~~ADR-0002 (rented hardware)~~ — **RESOLVED 2026-09-17, no longer a decision.** The fidelity
   tolerance is fixed by D-H (float32; forward `atol=1e-5, rtol=1e-4`; gradients `atol=1e-4,
   rtol=1e-3`), the VRAM number is now simply the Studio's, and §16 condition 8 is moot because
   **no compute is rented.** The ADR should be superseded, not filled in.
3. **Sign the E0i pre-registration** — and reconcile first: the MacBook derivation says ≥150 events per
   bin from a stated `σ_d`; the Studio says `N × p² ≥ 400`. They agree the attenuation exponent is
   **`p²` not `p`** (the MacBook version was wrong and has an errata). They disagree on the number.
4. **§15 is the author's and must never be filled by a model.** Neither session has touched it. Keep it that way.

---

## 7. The path to a first training run

```
ADR-0002 tolerance committed  (owner)
  -> vendor the pinned TG reference (JAX, read-only; never installed on the Mac)
    -> extract golden tensors ON THIS MACHINE, CPU — INCLUDING GRADIENTS
       (jaxlib ships macOS arm64 CPU wheels; jax-metal's death kills only the GPU
        backend. ~8MB from a 2.36M-param model: seconds of CPU work, in a
        throwaway venv so JAX never becomes a project dependency)
      -> transcribe TG to PyTorch at d=128        <- THE LONG POLE, 3-5 days
        -> fidelity green (forward AND gradients)
          -> E0b, E0c, E0d, E0a
            -> write the synthetic corpus generator   <- DOES NOT EXIST YET
              -> E1 / E2 -> the week-4 gate
```

🔴 **ALL TRAINING IS ON THE MAC STUDIO. There is no rented compute and none is planned.** Outsourcing
is reconsidered only after the model has demonstrated its effect locally. E0c measures the
`(S, d, batch)` ceiling **on the Studio**, for the widths actually being trained.

Nothing is lost: §16's approved scope is weeks 1–4, and E1/E2 are synthetic at `M=16, S=48` — which
fits on this machine several times over. The rental was only ever attached to the unapproved block.

📌 **Gradient fixtures are mandatory.** TG writes sentences to memory *without detaching the gradient
graph*; JAX and PyTorch diverge exactly there, so a forward-only match with wrong graph retention
passes and survives to week 7.

---

## 8. How to tell a good answer from a bad one

The single most useful question: **"show me the command you ran and what it printed."**

| Tell | What it means |
|---|---|
| "Tests pass" with no number | An adjective, not a result. `39 passed, 8 skipped` is a result. |
| A skipped test counted as passing | The Studio's suite is 39+8. If a report ever says "47 pass", that is the failure — and its own skip messages warn against exactly this. |
| A zero that means "didn't run" | Has already happened twice here. Exit code 3 (*did not run*) must never read as 0 (*passed*). |
| "I searched and found nothing" | Worthless without a positive control. This produced the largest error of the project — a 3–6 week reimplementation proposed for code that existed. |
| A new green check | Ask: *what did you break to prove it can go red?* |
| Confidence that grew with no new measurement | Drift, not progress. |

Quick independent checks:

```bash
ssh keanooo7@100.81.77.20 'cd ~/retrieval-successor-retention && /opt/homebrew/bin/git log --oneline | head'
ssh keanooo7@100.81.77.20 'cd ~/retrieval-successor-retention && .venv/bin/pytest -q -rs --tb=no | tail -5'
```

---

## 9. Where the facts live

| Need | Read |
|---|---|
| Corrections that **override** the spec | `docs/spec-corrections.md` — **both repos, and they differ.** Studio has 15; MacBook has B-1…B-5, S-1…S-6, D-10…D-15. |
| The spec itself | `docs/spec/rsr_model_spec_v0.5.md` (identical in both) |
| Why the two sessions differ, and who was right | `docs/decision-review.md` (MacBook only) |
| Gate discipline and mutation evidence | `docs/gates.md` (MacBook only) |
| Decisions taken | `docs/decisions/ADR-000*.md` — **numbering collides:** Studio's ADR-0002 is rented hardware, MacBook's is the TG-code correction |
| The TG reference pin | `third_party/PINS.md` (Studio only) |
