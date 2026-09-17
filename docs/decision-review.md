# Decision review — Sprint 1, and a comparison against session `d23e3dcd`

**Written:** 2026-09-17
**This session:** `09d11f93-c0a5-4a27-bfc6-7fea75b51f58` · MacBook Pro 18,3 · **16 GB RAM** ·
with the Claude Memory Palace attached
**Compared against:** `d23e3dcd-1fa1-4db1-8bed-07a52030c52a` · Mac Studio, M4 Max, 64 GB ·
no memory palace · **with the TG source**
**Their evidence:** `~/Downloads/rsr-sprint-1-handoff-2026-09-17.md`

**Two readers.** Brendan, and session `d23e3dcd`. Written for the second as much as the first —
which is why §3 comes before §4.

---

## 0. The verdict, first

The premise I was handed was that this session's work is better because it had more context.

**That is half right, and the wrong half is the expensive half.**

| Axis | Ahead |
|---|---|
| Instrumentation, gates, process, orchestration | **this session**, clearly |
| The science, the sources, and Task 0 | **`d23e3dcd`**, decisively |

**I got the single most consequential question of Sprint 1 wrong.** I concluded the Thought Gestalt
implementation did not exist, recommended emailing the authors, and opened a 3–6 week reimplementation
that sits in no schedule. `d23e3dcd` found the code. I verified their finding: it is real.

The right outcome is a merge, not a winner. §6 says what goes which way.

---

## 1. What I built, and why

### 1.1 §3.7's reduction is reachable by configuration alone

Every term in the eviction score has an off-switch in `RetentionConfig`, and
`RetentionConfig.reduction_to_tg()` sets all of them. `test_every_score_term_has_an_off_switch`
asserts the switch set is **complete** — it fails the moment someone adds a scoring term without a
switch, which is the moment §3.7 silently stops being a reduction.

**The decision worth transferring is a test-design one.**
`test_reduction_matches_fifo_via_the_score_path` sets `T_warm = 0` **deliberately**. Under the full
§3.7 config, `t_warm = ∞` routes every call through the warmup's FIFO shortcut — so a suite that
tested only that config would pass **without ever evaluating the eviction score.** Green, and
vacuous. Both paths are tested separately; each agrees with FIFO 200/200.

*(`d23e3dcd` reached the same architecture — protocol, `reduction_to_tg()`, off-switches. Their split
of E0b into reduction + fidelity is better than mine; see §3.7.)*

### 1.2 A constants registry that refuses

`FROZEN | MEASURED | DERIVED | CONDITIONAL`. Reading a `MEASURED`/`DERIVED`/`CONDITIONAL` constant
before its source experiment logs a value **raises**, and the message names the experiment that owes
it. §4.5's *"never freeze an unmeasured constant"* becomes mechanical rather than remembered.

`record()` refuses a value with no evidence string — "measured it" is a claim, not evidence.

*(Convergent: `d23e3dcd` built the same thing, same four classes, same refusal. Two sessions reading
the same §4.5 and reaching the same mechanism is a signal the spec earned it.)*

### 1.3 Ratchets with a direction, and five exit codes

A floor that can move both ways is not a floor. Seven metrics, each with a declared direction, and
five exit codes that mean five different things:

```
0 at/above · 1 FLOOR DROP · 2 floor UNKNOWN · 3 DID NOT RUN · 4 UNBANKED RISE
```

🔴 **`3` is the one this design can be silently defeated on.** "The gate did not run" and "the gate
found nothing" are different facts, and a caller testing only `!= 0` merges them. In ML that is the
sharpest version of the problem it can be: **a run that produced no metric looks exactly like a run
that produced a bad one.**

All five were exercised end-to-end, in both directions, before the gate was wired. A threshold
crossed in only one direction indicts the gate.

The row I would keep above all the others is **unmeasured-constants count, may-fall-never-rise** —
the mechanical form of the rule that killed v0.2-NP. It sat at 10; adopting `A_max = S = 48` moved it
to 9, and the ratchet correctly reported `UNBANKED_RISE` before it was banked.

### 1.4 Mutation-testing my own gates

**The decision I would defend hardest, because it caught me.**

A gate is not believed green until a mutation has shown it red — and the mutation must redden *only*
it. Seven mutations against `RetentionConfig`. Four reddened the reduction tests as expected.

🔴 **One reddened nothing: dropping the `nu` clause from `is_reduction`.** That function gates whether
`RSRPolicy.observe` may no-op. With the hole, a policy configured with `ν = 0.5` would have reported
itself as the §3.7 reduction, `observe` would have silently returned `None`, and **an arm that was
supposed to be learning would have learned nothing while still being called RSR.**

Two tests added; all seven mutations now redden. The table is in `docs/gates.md`.

I do not know whether that bug would have shipped. I know the suite was green before the mutation and
green after, and only the mutation distinguished those two greens.

### 1.5 The verification pass (E0f, pulled forward from week 2)

§1 records that ten of fourteen references were unverified at drafting. The GPU budget, two kill
gates and every baseline definition descend from those claims. I read arXiv:2512.25026v2 and the H2O
NeurIPS paper end to end against the spec.

**Most of the spec survived exactly** — App. A verbatim, 29.8→35.0 and 21→24 exact, the +12/5-epoch
curriculum exact, [P5] does include `W_O`, H2O's 50/50 split supported by its own `H2O-256-256`
notation. Five claims did not, and are B-1…B-5 / S-1…S-6 in `docs/spec-corrections.md`.

### 1.6 The W0 → ML orchestration port

The half of the ask `d23e3dcd` had no way to do — it is a vault artefact, and they did not have the
vault.

One new adapter file, **no orchestrator script forked**, per the vault's own rule that *"a second
project is a second file in this directory, never a second script."* Five lanes, split **by gate, not
by topic**, because the app project's transferable insight is that the gate boundary is what makes
lanes parallel and the file boundary is not. Here the scarce resource is **the device**, so exactly
one lane holds it and the CPU-only lane runs free beside it.

`device:mps0` claims as a **pseudo-path**. `claim.cjs`'s matching is *"prefix + glob, deliberately
coarse"* — so a resource claims exactly like a file, and the existing pre-commit hook arbitrates it
with zero code change.

**A sixth retire trigger, new for ML:** *a lane that has inspected data it must later pre-register a
threshold against is retired before it writes the pre-registration.* It cannot pre-register any more
— it has seen the data. That gate stands in front of 576 GPU-hours.

And the port found a real defect in the layer it ported from: `brief-status.cjs`,
`check-merge-diff.cjs` and `premise.cjs` were hardcoded to `cleaning.json` while every sibling honors
`ORCH_PROJECT`. Measured: `ORCH_PROJECT=rsr node brief-status.cjs` reported **785 Cleaning briefs**.
`premise.cjs` defaults its cwd to `cfg.repo`, so an RSR brief's premise assertions would have run
against the Cleaning tree, silently — in the script that exists *because* briefs were dispatched on
false premises. Patched; Cleaning output proven byte-identical with the variable unset.

### 1.7 Pre-registering E0i before any data existed

`preregistration/e0i_threshold.md`, committed in the first commit, before any PG-19 subset was
loaded. `git log` is the check, and that is the entire point: the ordering claim is falsifiable after
the fact.

**One of its two halves is wrong.** See §3.6.

---

## 2. Where the Memory Palace actually changed a decision

The hypothesis was that palace access made this work better. Tested honestly:

**It did — and only on process. Never once on science.**

| Palace rule | What it produced here | In `d23e3dcd`? |
|---|---|---|
| *A gate not shown red by a mutation adds nothing* | §1.4 — found a real hole | **no** |
| *"Did not run" ≠ "found nothing"; exit 3 is not a pass* | §1.3 | **no** |
| *A floor needs a direction* | §1.3 | **no** |
| *Numbers are measured, never typed* | `rsr floor` parses pytest's own summary | **no** |
| *Split lanes by gate, not topic* · single-writer · retire-and-respawn | §1.6 — the whole port | **no** |
| *An absolute path is not provenance, a sha is* | the `provenance:` return line | **no** |
| *A negative result must say where it searched* | ADR-0001's evidence section | **had the rule, broke it — §3.1** |

And the other direction:

> **Zero of my substantive spec findings came from the palace.** B-1 through S-6 came from reading
> PDFs — something both sessions could do, and which `d23e3dcd` did better because it also read the
> source code.

So the honest form of the hypothesis: **the palace made my instrumentation better and my scholarship
no better.** It gave me the exact rule that would have prevented my worst error, and I broke it
anyway. A rule you hold and do not execute is not an advantage.

---

## 3. Where session `d23e3dcd` is ahead

Stated before §4, because it is the more useful half of this document.

### 3.1 🔴 Task 0 — they found the code. I was confidently wrong.

**`https://github.com/jlmcc94303/ThoughtGestaltCode`** — JAX/Flax, Apache-2.0, 0 stars, on James
McClelland's personal account. **Verified here by direct fetch**, including the README sentence:
*"This version differs slightly from the version of the model described in (arXiv:2512.25026)."*

My ADR-0001 concluded *"the answer from published sources alone is no"* and proposed a 3–6 week
reimplementation plus an author email.

**The instrument was the failure, not the effort.** I searched the paper exhaustively and correctly —
that part stands. Then I searched a **web index** for a repository, got `milenarabovsky/SG_model` (the
2018 *Sentence Gestalt* model), and stopped. They searched **GitHub's own repository index for "thought
gestalt"** and found it immediately.

What was missing is a **positive control**. A search instrument never shown to find something you
know exists cannot support a negative result. The single `SG_model` hit should have read as *"this
index resolves model names, not this model"* — information about the instrument — rather than as an
answer. One query.

Superseded in `docs/decisions/ADR-0002-tg-base-found.md`. ADR-0001 is left standing, unedited.

**Three qualifications that are theirs and that reshape the plan:** the release is JAX/Flax with no
Mac GPU path (`jax-metal` dead since 2024-10); it is **not the paper's model**, so my proposed 29.8
PPL smoke test is unreachable by the authors' own statement; and `src_recurrent` is missing so the
corpus path does not run.

### 3.2 🔴 Correction 15 — unit-norm gestalts break §4.3's μP premise

**The deepest finding either session produced, and it needed the source to see.**

`tg_srep_head.py` L2-normalises the sentence representation to `srep_norm_target = 1.0`. So
`‖s_i‖₂ = 1` exactly, and its coordinates are of order `1/√d` — **not `Θ(1)`, which is what §4.3's
derivation of the `1/d` multiplier assumes.** Recounted under unit norm, the trained-regime output
after the prescribed multiplier is `Θ(1/d)`: it *decays* with width in exactly the regime μP governs.

**And it invalidates the input premise of my own E0a.** `experiments/e0a/run.py` feeds
`torch.randn(N, d)` gestalts — `Θ(1)` coordinates, norm `~√d`. My 0.581-vs-0.577 match is correct
arithmetic **under an assumption the real system breaks.** It confirms §4.3's reasoning; it does not
confirm §4.3's prescription for TG's actual gestalts. Caveat added to `experiments/e0a/RESULTS.md`.

**They flagged it and did not resolve it, which is right** — §15.3 identifies that derivation as the
author's. And they extracted the actionable consequence I would have missed even with the finding:
§3.2.2 allows `c_t` to be *"the current sentence gestalt **or a running context vector**"* — different
norms, different correct multipliers. **Pin it before E0a runs.**

### 3.3 Correction 10 — activation memory sized against the wrong card

§4.2 fixes `(S, d, batch)` against **64 GB** (the Mac). **E3 runs on 48 GB rentals.** E0c can pass at
64 and E3 OOM in week 6 — §7.6's exact failure, arriving through the experiment meant to prevent it.

**I wrote an entire procurement document comparing 48 GB cards and never noticed.** Their D4 goes
further: run E0c *on* the rented card, *before* the hold, so it doubles as the provider smoke test.

### 3.4 Correction 13 — [P2] trains on WikiText and never touches PG-19

**Verified here.** Across all 20 pages: `PG-19` 0, `Gutenberg` 0, `WikiText` 3.

§5.3's *"PG-19 — where the claim lives"* is RSR's own corpus choice. §13's *"`S = 80` exceeds [P2]'s
curriculum"* understates it: E3 differs from [P2] in stream length **and corpus**. I read that paper
closely enough to quote its ablation table and missed the corpus it was trained on.

### 3.5 The γ-freeze consequence, pre-registered

The one I most wish I had had:

```
γ = 0.90    0.90^40 = 0.015     0.90^64 = 0.0012    ->   0.1-1.5% of the return
γ = 0.97    0.97^40 = 0.296     0.97^64 = 0.142     ->    14-30% of the return
```

**If E1 freezes γ below 0.97, E3's `(40,64]` window is known in advance to be under-weighted.** That
turns a possible mystery null into a limitation registered before the run. Excellent work.

*(We also both misread §3.6(a)'s `A_max ∈ {16,32,64}` as a prescribed sweep set. Brendan caught
theirs. Mine survived into `docs/spec-corrections.md` B-5 until now — resolved to their
`A_max = S = 48`. **Two agents misreading the same sentence identically is a signal about the
sentence**; §3.6(a) is worth rewording in v0.6.)*

### 3.6 🔴 The coref attenuation exponent — my pre-registration is wrong

My threshold adjusts as `n_raw × p`. **It should be `n_raw × p²`.**

At precision `p`, false positives attenuate the effect (`δ_obs ≈ p·δ_true`); power goes as `δ²·n`; so
required `n` scales as `1/p²`. At `p = 0.7` my linear form asks for **30% fewer events than are
actually needed** — and it relaxes exactly where measurement error is worst.

Errata appended to `preregistration/e0i_threshold.md`. **The threshold is not silently moved** —
re-registering a number after the fact is the failure that file exists to prevent. The error is
stated, the corrected form derived, and it awaits a signature. Their absolute target (`N × p² ≥ 400`)
and my per-bin 150 still disagree and need reconciling; mine at least shows its `σ_d` assumption.

### 3.7 E0b split into reduction **and** fidelity

Mine is one test: RSR-under-§3.7 vs FIFO, intra-repo. Theirs is two:

- `test_reduction.py` — intra-repo, bit-exact, guaranteeing **E3's FIFO and RSR arms differ only in
  the eviction rule.** Same as mine.
- `test_fidelity.py` — the **external** claim against golden tensors from the pinned JAX reference,
  to a tolerance committed in an ADR **before the fixtures are generated.**

And the insight I could not have had without knowing the reference is JAX: **gradient fixtures are
mandatory**, because gestalts are written without detaching the graph and backward depth is bounded
by `S` — JAX's functional autodiff and PyTorch's retained-graph semantics diverge exactly there, so
*a forward-only match with wrong graph retention passes and survives to week 7.*

### 3.8 E0g — they met §5.3's criterion; I said it was unmeetable

We independently picked the same dataset (NFRD, CC0). But §5.3's actual criterion is *"whether
importance is scored independently of serial position,"* and my write-up concluded **"No — and no
naturalistic free-recall corpus does."**

**Wrong.** NFRD ships a **semantic centrality** measure — Universal Sentence Encoder similarity graph,
per Lee & Chen — with *"a significantly positive effect … on the likelihood of recall (p < 0.001 …
β = 0.27)"* and published code. **Verified here.** Centrality is position-independent *by
construction*: reorder the events and it is unchanged.

I read Methods and Data Records and stopped before the usage notes, then asserted a claim about an
entire class of corpora. Correction appended to `experiments/e0g/RESULTS.md`.

*(One precision that is mine: the paper does not state that the reported β controlled **for** serial
position, so their phrasing is slightly stronger than the text supports. E7 still partials position
out. The gain is that E7 now has a second, independent importance regressor.)*

### 3.9 Practical findings

`jax-metal` last released v0.1.1 on 2024-10-08 · `src_recurrent` missing from the release ·
`mlx_lm.server` routinely holding ~25 GB of the Studio's 64, which independently disqualifies the Mac
for E0c · `git init.defaultBranch` unset on that machine · the git identity mismatch.

They also found, and recorded as not-their-problem, that `EXPERIMENTS.md` admits the paper's gist
baseline *"indexes the gist flag on the query axis instead of the key axis and so grants no
cross-sentence access at all"* — **an admission of a bug in a published baseline.**

---

## 4. Where this session is ahead

1. **The orchestration port** (§1.6). They have none, and could not have — it is a vault artefact.
2. **Mutation testing** (§1.4), which found a real hole in my own gate.
3. **Directional ratchets and the five exit codes** (§1.3), with exit 3 ≠ pass.
4. **S-1 — H2O already formalises eviction as dynamic submodular maximisation** with a near-optimality
   theorem for greedy (Lemma 3.1, Thm 4.4, proofs in App. D) — while §3.4's D-3 argues *against*
   greedy independent scoring on submodularity grounds and §11's H2O entry says only "evicts by
   accumulated attention." This is the S-1 scholarship failure repeated on a second paper, and
   falsifier 5's *"regardless of how good ψ̂ is"* is stated more strongly than published theory
   supports. Not in their list.
5. **S-2 — H2O App. B.2 documents the accumulated-attention age bias *and* reports that the averaged-
   score fix degraded performance for them.** §3.5 makes exactly that move for `ū`. Also: H2O's bias
   runs *anti*-recency while FIFO/LRU run pro-recency, so **E2's single-arm vacuity ρ is
   uninterpretable** — report it for every arm. Not in their list.
6. 🔴 **B-3 — Expire-Span gets no tuning budget while falsifier 6 is the stop condition.** [P7] has a
   ramp length `R`, a loss coefficient `α`, and *requires* structured dropout; the spec runs it once,
   untuned, against RSR's 9-config grid plus a `ν` sweep. **An untuned Expire-Span losing is
   uninformative; a tuned one winning ends the project.** The asymmetry protects the project, which is
   the wrong direction. **This is the finding I would move to the top of the merge list** — it is the
   only one that can hand a wrong answer to the question that stops the work.
7. **B-1 — `r_i` omits TG's learnable `g_mem`**, which App. C measures as growing over training and
   larger in deeper layers, plus Kobayashi et al. 2021 as the named successor instrument to [P5].
   *(§3.2 is deeper and in adjacent territory; these compose rather than compete.)*
8. **S-3 / S-4** — E3 adopts [P2]'s worst measured design ablation (no curriculum, 30.5 vs 29.8, "the
   largest drop") on a premise never checked, since a curriculum reaches 78 then 90; and E5 partially
   re-runs [P2]'s own published width sweep over a *wider* range with the same conclusion.
9. **Pre-registration committed**, so `git log` proves it preceded the data. Theirs is untracked
   pending signature — better governance, weaker provenance. **Do both:** commit unsigned and
   timestamped, then commit the signature.

---

## 5. Independent convergence

Where two agents, from the same sources, reached the same place — which is the closest thing to
corroboration available here:

- §4.1's 768 GPU-h for weeks 5–7 is an undercount; the real load is **842**, breaking the stated floor of 3
- The falsifier numbering `1, 2, 3, 3b, 3c, 5, 6, 4` is a citation hazard
- Procurement is a **cancellable hold, movable start, week 6, floor 4** — not a week-5 reservation at floor 3
- **NFRD** as the E7 stimulus set, over the spec's own ranked list
- **The μP init-decay trap, and that E0a's verdict must be read after real optimizer steps.** They
  measured 0.0895/0.0701/0.0647/0.0512 across `d = 128…384`; I measured the ratio at 0.581 against
  0.577 predicted. Same phenomenon, same reading rule, arrived at independently
- A constants registry that **raises** on an unmeasured read, with the same four classes
- §3.7's reduction reachable **by configuration alone**, every scoring term carrying an off-switch
- **§15 untouched by both.** Neither drafted it, restated its pointer, or suggested a candidate

---

## 6. What to merge, which way

**`d23e3dcd`'s repo is the trunk.** It is on the training machine, it has the private remote, and it
carries the better science and the actual TG reference.

**From here into it**, in priority order:

| Take | Why |
|---|---|
| **B-3** — Expire-Span tuning budget | the only finding that can corrupt the stop condition |
| `docs/gates.md` + the mutation discipline | it found a real bug; adopt the practice, not just the file |
| Directional ratchets + the five exit codes, `rsr floor` / `rsr constants --check` | exit 3 ≠ pass is worth more in ML than anywhere else |
| **S-1, S-2** | §11 must engage H2O's Thm 4.4; E2 must report age-ρ for every arm |
| **B-1** (`g_mem` in `r_i`), **S-3**, **S-4** | each changes an experiment's design or its writeup |
| Commit the pre-registration unsigned-and-timestamped, then the signature | provenance and governance are not exclusive |

**From it into here** — already done, in this commit: ADR-0002 supersedes ADR-0001 · E0g corrected ·
pre-registration errata · B-5 resolved to `A_max = S = 48` · their corrections 10, 12, 13, 15 adopted
as D-10, D-12, D-13, D-15 · the compute plan corrected to Mac-Studio-first.

**The port does not merge.** It lives in the vault and works against whichever checkout is the trunk;
only `projects/rsr.json`'s `repo` path needs repointing.

---

## 7. Disclosures

- 🔴 **Everything here was built and measured on a MacBook Pro 18,3 with 16 GB of RAM**, not the
  64 GB Mac Studio. E0a is a CPU width-scaling check so memory never bound it and those numbers
  stand — but **no memory or throughput claim from this machine transfers**, and `PROCUREMENT.md` was
  written without ever touching a GPU. That was true at the time and was not stated. It is now.
- Three of `d23e3dcd`'s claims are **adopted unverified** and labelled as such: `jax-metal`'s status,
  the missing `src_recurrent`, and the "fork-by-import of nanodo" provenance (GitHub reports the repo
  as not a fork; a fork-by-import does present that way, so it is plausible and unconfirmed).
- **No email was sent to anyone.** ADR-0001's draft remains a draft, and ADR-0002 withdraws the
  recommendation.
- **§15 is untouched.** Not filled, not restated, no candidates suggested — in this document or
  anywhere else this session wrote. It says it must not be filled by a model.
