# GATE-1 — Sprint 1 report

| | |
|---|---|
| Sprint | 1 · week of 2026-09-17 |
| Sprint goal | **Kill the project cheaply this week if it deserves killing.** |
| Verdict | 🟡 **CONTINUE — nothing found kills it, and three gates are DEFERRED on one blocker** |
| Repo @ | `a81a568` |
| Suite | `.........................................s..                             [100%]` |
| Constants |   10 still unset: A_max_synthetic, E_lifetime, beta, e0i_events_per_bucket_floor, ema_half_life, gamma, gamma_b, nu, sent_per_sec, tau |

---

## Exit criteria

| # | Task | Status | Evidence |
|---|---|---|---|
| **T1** | Scaffold, CLAUDE.md, CI, constants registry, policy protocol | ✅ **PASS** | `uv run pytest -q` · `uv run ruff check` clean · `docs/spec-corrections.md` committed |
| **T2** | Pre-register E0i's threshold | ✅ **PASS** | `preregistration/e0i_threshold.md`, committed **before** any PG-19 data was loaded — `git log` is the check |
| **T3** | E0i coref histogram + coref precision | ⬜ **NOT STARTED** | Pipeline design and threshold are in place; needs the PG-19 subset |
| **T4** | E0g — name and obtain the E7 stimulus set | ✅ **PASS** | `experiments/e0g/RESULTS.md` — NFRD, CC0 |
| **T5** | E0c — memory/throughput | ⚠️ **DEFERRED** | Blocker: ADR-0001. Procurement half done: `experiments/e0c/PROCUREMENT.md` |
| **T6** | E0d — `r_i` vs LOO Δloss | ⚠️ **DEFERRED** | Blocker: ADR-0001 (needs a briefly-trained TG) |
| **T7** | E0a — μP coordinate check ×2 | ⚠️ **PARTIAL** | Value-head half complete, 5 seeds: `experiments/e0a/RESULTS.md`. Bare-TG half blocked on ADR-0001 |
| **T8** | GPU procurement | ⚠️ **PARTIAL** | Comparison and decision recorded; **no account, no quote, no hold** — owner action |
| **T9** | This report | ✅ | |

🔴 **No deferred gate is reported as a pass.** Three gates could not run, one ran half. That is the
honest state and it is the useful one.

---

## The one blocker

> 🔴 **SUPERSEDED 2026-09-17 by `docs/decisions/ADR-0002-tg-base-found.md`. The verdict below is
> WRONG. The code exists: `github.com/jlmcc94303/ThoughtGestaltCode` — JAX/Flax, Apache-2.0 — found
> by session `d23e3dcd` and verified here by direct fetch.**
>
> The search of the *paper* below is accurate and stands. The conclusion drawn from it is not. The
> failure was the instrument: a web index was searched for a repository, returned the 2018 *Sentence
> Gestalt* model, and was treated as an answer instead of as evidence about the index. **No positive
> control was run.**
>
> **What changes:** the blocker becomes a **3–5 day PyTorch transcription** of a pinned JAX
> reference, not a 3–6 week from-paper reimplementation. The 29.8 PPL smoke test is **withdrawn** —
> the release differs from the paper's model by the authors' own README. The author email is
> **withdrawn** from the critical path. E0a / E0c / E0d remain deferred, on a smaller and better-
> bounded blocker.
>
> This section is left unedited below. An ADR or a gate report that gets quietly corrected teaches
> nothing, and this one has something to teach.

**ADR-0001: [P2]'s Thought Gestalt implementation is not publicly available.**

Zero hits for `github`, `we release`, `code available`, `available at/upon` or `open source` across
all 20 pages of arXiv:2512.25026v2. The single hit for `reproduc` is the word "reproduce" inside an
ablation sentence. No Code / Papers-with-Code / Hugging Face link on the abstract page.

The spec tags TG **[E]** — *"demonstrated in a cited source."* It does not mean *obtainable*, and
that distinction is invisible in the tag. It is the largest unpriced item in the project: the
delivery plan prices reimplementation at **3–6 weeks**, and no branch of the schedule can absorb it.

**Decision taken:** email the authors day 1 (draft ready, **not sent** — that is the owner's), and
open a from-paper reimplementation in parallel rather than waiting. If the code arrives, the
reimplementation becomes E0b's **independent cross-check**, which makes a bit-exactness claim far
stronger than one implementation can.

---

## What Sprint 1 actually established

### The mechanism reduces to stock TG, and the test is not vacuous
200/200 agreement with FIFO on **both** the warmup path and the score path. The score path is tested
separately on purpose: under the full §3.7 config every call routes through the warmup's FIFO
shortcut, so a suite testing only that config would pass **without ever evaluating the eviction
score** — green, and vacuous.

### A mutation found a hole in the reduction gate before it could hide anything
Dropping the `nu` clause from `is_reduction` reddened **nothing**, while four other mutations
reddened the reduction tests. `is_reduction` gates whether `observe` may no-op, so that hole would
have let a policy configured with `ν = 0.5` report itself as stock TG and **silently not learn**.
Two tests added; all seven mutations now redden. Table in `docs/gates.md`.

### §4.3's μP footnote is confirmed quantitatively
Across five seeds, `ψ̂`'s RMS ratio `d=384/d=128` **at init** is **0.581** against **0.577** predicted
by `Θ(1/√d)`. A 0.7% match.

🔴 **Operational consequence, new:** E0a's verdict must be read at `t ≥ 1`, never at step 0. A check
at init shows a *correct* parameterization shrinking with width and reads as a μP violation; "fixing"
the multiplier to `1/√d` tunes it for the init regime and breaks the trained one — §4.3's own failure
mode, reached from the opposite direction by an entirely ordinary measurement.

### E0f, pulled forward, found five things that change code or a number
[P2], [P5], [P6] and [P7] verified against the primary sources. **Most of the spec holds exactly.**
Five do not: **B-1** `r_i` omits TG's memory gate · **B-2** the GPU budget's anchor is
off-configuration · **B-3** the baselines get no tuning budget while falsifier 6 is the stop
condition · **B-4** §5.1's parameter range · **B-5** synthetic `A_max`. Plus six scientific
objections. All in `docs/spec-corrections.md`; evidence in `docs/citation-audit.md`.

### E0g may take a release condition off the budget
NFRD is **CC0** — no data-use agreement, so E0g's real schedule risk is gone. And **C1/D-5's premise
is false for it**: three of the four stories run ~85–95 sentences, over 2× `M = 40`, so E7 may run on
E3's own model. That would void the second model (~72 GPU-h), dissolve §10.1's capacity-mismatch
precondition, and satisfy **§16 condition 4 at zero cost**. Estimated from word counts — **measure
under SaT before banking it.**

🔴 And **`baseball` is a likely PG-19 contaminant** — *Baseball Joe in the Big League* (Chadwick,
**1915**), Gutenberg **#27584**, inside PG-19's pre-1919 window, and the story with the **highest**
mean recall rate of the four. Check the subset before training.

---

## §16 conditions

| # | Condition | State |
|---|---|---|
| 1 | E0i histogram vs a pre-registered threshold | 🟡 **threshold registered**, histogram not run |
| 2 | `γ_b` rederived; §3.5 moves the argmin | ⬜ E0e. Registry holds `gamma_b` DERIVED and unset |
| 3 | Expire-Span at the week-4 gate | ⬜ Sprint 3. ⚠️ **B-3: it must be *tuned*, or the referendum is rigged** |
| 4 | E7 model funded, or E7 cut | 🟢 **E7 survives**, and may need no second model — see E0g |
| 5 | Redundancy scoring + ρ with/without | 🟡 `ν` term implemented; ρ needs E0d |
| 6 | §11 with leading-edge as implemented baseline | ⬜ Sprint 3. ⚠️ [P11] still unverified |
| 7 | §15.2 non-empty, defended in person | 🔴 **unsatisfiable as written** — see `docs/release-conditions.md` |
| 8 | Parallel GPU capacity reserved | 🟡 ⚠️ **the instrument may not exist** — RunPod savings plans are prepaid and non-refundable. See `experiments/e0c/PROCUREMENT.md` |

---

## Decisions owed to the owner

1. **TG.** Send the email (`docs/decisions/ADR-0001-email-draft.md`). Accept or overrule the
   parallel-reimplementation decision, knowing it is 3–6 weeks that no schedule holds.
2. **B-3.** Approve the equal-budget protocol for E1, or accept that falsifier 6's verdict is
   provisional. **This is the one that can hand you a wrong answer to the question that ends the
   project**, and it is the first thing I would fix.
3. **B-5.** Set synthetic `A_max ≥ 33`, or drop `γ = 0.97`. Before E1. The registry refuses to
   proceed on a default.
4. **Condition 7.** Decide how §15.2 gets adjudicated with one person. Not reinterpreted quietly.
5. **T8.** Whether to pursue an enterprise quote, or to accept on-demand availability risk.

## Not started, and why

**T3 (E0i)** needs the PG-19 30M-token subset and a coreference pipeline — a real download and a
tool choice that deserves its own ADR. The threshold is registered and the gate's design is fixed,
which is the part that had to happen before the data, and did.
