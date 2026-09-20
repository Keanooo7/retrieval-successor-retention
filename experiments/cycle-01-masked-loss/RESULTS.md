# Cycle 1 — the training loss scores padding

Brief: `docs/lab-notes/dispatch-cycle-01-masked-loss.md` (body at `46c208e`,
revalidation at `3beb2b7`, first correction at `8a8a116`, second correction at
`eedc0c0`).

---

## 🔒 Pre-registration — written before the measurement, and not edited afterwards

> **Expected result.** Masking the loss to real tokens should **raise** the
> reported per-token loss, because PAD is trivially predictable and currently
> dominates the mean. **A masked loss that is *lower* is a finding, not a bug to
> tune away**, and it would falsify the reading.

**Falsifier.** The reading dies if **either**:

1. the fraction of scored targets that are PAD is **below 0.50**; or
2. masking the loss moves the **final training loss** by **no more than the
   run-to-run floor** at identical seed and config.

**The floor statistic, pinned.** `|final-beat difference|` between two runs
differing in nothing — same seed, same config, same code, separate processes.
This is deliberately **not** the `3.1789143850602386e-07` of
`runs/cycle-arm-identity/`, whose `how` field says it is
`max_t |loss_x[t] - loss_y[t]| over the 10 beats` — a **trajectory sup-norm**,
which is `≥` the final-beat difference by construction. The sup-norm is reported
beside the floor for continuity and is **not** the bar.

**Device, pinned: `mps`.** On CPU the run-to-run floor is `0.0`
(`runs/cycle-arm-identity/RESULTS.md`), which reduces the falsifier's half 2 to
*"the delta exceeds zero"* — satisfied by one float ulp. That is a vacuous
discriminator, and `decide()` in `run.py` returns **`inconclusive`**, not
`survived`, whenever the measured floor is exactly `0.0`. That rule is in the
commit that precedes the run.

**`iters = 50`, stated because the falsifier does not name it.** It is `train()`'s
own committed default. The manager's `iters = 8` reconstruction was retracted
before this run (`8a8a116`) as a coincidence of factorisation, so adopting it
would have been adopting arithmetic.

**Decision rule, pre-registered** — `experiments/cycle-01-masked-loss/run.py::decide`:

| condition | verdict |
|---|---|
| `pad_fraction ≤ 0.50` | `falsified` |
| floor `== 0.0` | `inconclusive` (half 2 is vacuous on this device) |
| `delta ≤ floor` | `falsified` (the defect is real and inert) |
| otherwise | `survived` |

---

## Provenance

| | |
|---|---|
| git sha | `e76c42e` (measured; `git_provenance()` from inside the tree, Homebrew git) |
| device | `mps` (headline) and `cpu` (replication) — two ledgers, one per device |
| manifest | `runs/cycle-01-masked-loss/manifest.json`, config hash `557e9cb0ea200a4d` (MPS) · `6de5c204904983a3` (CPU) |
| manifest frozen at | `2026-09-20T21:01:28Z`, before the first run started |
| seeds actually run | `[0, 1, 2]`, plus a **second replicate at seed 0** per arm for the floor |
| steps requested / done | `50 / 50`, every one of the 8 runs per device |
| runs | 8 per device, one subprocess each, every exit code read from the process |
| gates | `uv run pytest -rs` → **exit 0**, `passed=323 failed=0 skipped=0 errors=0` · `uv run python scripts/mutation_battery.py --check` → **exit 0**, 37/37 proven |

Commands: `uv run python experiments/cycle-01-masked-loss/run.py --device mps`
and `... --device cpu --run-id cycle-01-masked-loss-cpu`.

---

## Verdict

| device | outcome | why |
|---|---|---|
| **mps** | **`survived`** | PAD fraction `0.9343` > 0.50, and masking moves the final training loss by `1.749`, **698,587×** the measured run-to-run floor `2.503e-06` |
| **cpu** | **`inconclusive`** | half 1 survives identically; the floor measured **exactly `0.0`** (all four replicate pairs bit-identical), so half 2 reduces to *"the delta exceeds zero"* and discriminates nothing |

The CPU `inconclusive` is the pre-registered rule firing, not a failure. A floor of
`0.0` **here is a correct determinism result, not the `sd = 0.0000` shape** — the
*seeds* do vary (`sd = 9.602e-02` on `masked.loss_final`); it is the *replicates at
one seed* that are bit-identical, which is what CPU determinism means.

---

## 1 — The PAD census

Two populations, counted separately, because they do not agree to the last digit
and a reader who sees only one will read the disagreement as a contradiction.

| | denominator | targets | PAD | fraction |
|---|---|---|---|---|
| **corpus-shaped** (whole committed corpus, encoded once) | `64 × 48 × (64−1)` | `193,536` | `180,812` | **`0.9342551256613757`** |
| **run-shaped** (one training iteration's batch, MPS) | `8 × 48 × (64−1)` | `24,192` | `22,613` | `0.9347304894179894` |
| **run-shaped** (same, CPU) | `8 × 48 × (64−1)` | `24,192` | — | `0.9351025132275133` |

**The handoff's "93.4% of the 193,536 scored targets are PAD" reproduces exactly,
and its denominator is corpus-shaped.** The manager's retraction of the `iters = 8`
inference is confirmed: `64 × 48 × 63 = 193,536` needs no `iters` at all.

**The two sources of PAD, counted separately as instructed:**

| source | count | share of all targets |
|---|---|---|
| **A** — trailing PAD inside a sentence slot holding at least one real token | `180,812` | `0.93426` |
| **B** — every position of an entirely empty sentence slot | **`0`** | `0.0` |

Source B is **empty**, and the revalidation's instruction not to assume both are
non-empty was the right one. `SyntheticConfig(sentences_per_document=48)` emits
exactly 48 sentences in all 64 documents (`corpus.empty_sentence_slots = 0`), so
every PAD target is trailing padding inside a real sentence. Sentences are 3–7
tokens (EOS included) inside a 64-token frame.

`mask_t` and `ids_t != 0` identify the same positions on this corpus
(`test_the_mask_and_the_pad_id_identify_the_same_positions`), because
`build_vocab` reserves ids 0–3.

---

## 2 — The run-to-run floor

Two runs differing in **nothing** — same seed, same config, same code, **separate
processes**. The statistic is `|final-beat difference|`, which is the quantity
this cycle claims moved.

| device | arm | `|final-beat diff|` | trajectory sup-norm | bit-identical |
|---|---|---|---|---|
| mps | masked, `loss` | `1.748402913559488e-06` | `1.748402913559488e-06` | no |
| mps | unmasked, `loss` | `2.5033950805664062e-06` | `6.079673767089844e-06` | no |
| mps | masked, `loss_real_tokens` | `1.9073486328125e-06` | `1.9073486328125e-06` | no |
| mps | unmasked, `loss_real_tokens` | `2.9558936755158527e-05` | `8.997321128845215e-05` | no |
| **cpu** | **all four** | **`0.0`** | **`0.0`** | **yes** |

**The `3.1789143850602386e-07` of `runs/cycle-arm-identity/` was not reproduced and
was not the right comparator.** Its `how` says it is `max_t |loss_x[t] - loss_y[t]|
over the 10 beats` — a trajectory sup-norm, `≥` the final-beat difference by
construction, at a beat count this run does not use. Mine is ~5–10× larger, which
is the expected direction for 50 beats of accumulated MPS non-determinism rather
than 10.

---

## 3 — The arms

`n = 3` seeds each, replicate `a`. **Spread is reported for every headline number
and no sd is zero.**

### MPS

| quantity | masked arm | unmasked arm | Δ |
|---|---|---|---|
| **reported** final loss (`loss`) | `2.010773 ± 0.132726` | `0.261934 ± 0.003367` | **`1.7488391002019248`** |
| **real-token NLL** (common yardstick) | `2.010772 ± 0.132726` | `3.967117 ± 0.015492` | **`1.956344069706069`** |
| loss at beat 0 (before learning) | `5.074075 ± 0.047075` | `4.982288 ± 0.027627` | |

### CPU

| quantity | masked arm | unmasked arm | Δ |
|---|---|---|---|
| **reported** final loss | `2.028117 ± 0.096020` | `0.258454 ± 0.009724` | `1.7696625855233936` |
| **real-token NLL** | `2.028117 ± 0.096020` | `3.914553 ± 0.125977` | `1.8864357479744491` |
| loss at beat 0 | `5.070791 ± 0.048257` | `4.980208 ± 0.020710` | |

**Pre-registration held: masking raises the reported per-token loss**, by `1.749`
nats on MPS — 698,587× the floor, and 6.7× the largest seed-to-seed sd. The
direction is the one written down before the run.

### 🔑 The result that is not tautological

The row above compares two different objectives, so a nonzero difference is
nearly guaranteed. **The common yardstick — real-token NLL, the same quantity in
both arms — is the one that carries information, and it is the bigger number.**

Derived from the ledger rows above (`exp` of the means; arithmetic, not new
measurements):

| | reported perplexity | **real-token perplexity** |
|---|---|---|
| unmasked arm (the pre-fix pipeline) | **`1.2994`** | **`52.8320`** |
| masked arm (after the fix) | `7.4691` | **`7.4691`** |

So the pre-fix pipeline reported a perplexity of **1.30** for a model whose actual
perplexity on real tokens was **52.8**. And training on the masked objective does
not merely *report* differently — it **cuts real-token NLL by 49.3%**
(`3.967117 → 2.010772`), `66,184×` the yardstick's floor. **The defect was not
inert. It was changing what the model learned.**

📌 **Instrument check that the numbers pass:** `masked.loss_first_beat = 5.074075`
against `ln(V) = ln(160) = 5.075174`. At initialization the masked loss is the
uniform-vocabulary entropy to four significant figures, which is what it must be
and what the unmasked loss (`4.982`) already is not.

---

## 4 — The shuffle control 🔴

Required of every training run here. Each row is handed **another document's entire
16-slot memory** (roll-by-1 derangement); the memory *contents* are replayed
unchanged from the honest pass, so only ownership changes. `model.eval()`
(correction 20). `bos_ctx` is left honest on purpose — `bos_replacement_mode ==
"copy"` sits outside the `use_memory` guard.

| device | arm | honest real-token NLL | Δ from shuffling | relative | max per-step \|Δ\| |
|---|---|---|---|---|---|
| mps | masked | `1.8492362573742867` | `+3.499289353614543e-06` | `1.9e-06` | `8.86e-05` |
| mps | unmasked | `3.9210285544395447` | `-6.457169852147615e-07` | `-1.6e-07` | `7.39e-06` |
| cpu | masked | `1.9213201329112053` | **`-7.651746273040771e-06`** | `-4.0e-06` | `1.50e-04` |
| cpu | unmasked | `3.664038414756457` | `+1.8874804208479645e-07` | `5.2e-08` | `1.10e-05` |

🔴 **The memory is still inert, in both arms, on both devices.** On MPS the delta
is `1.8×` the run-to-run floor — indistinguishable from numerical noise. On CPU,
where the floor is exactly `0.0` and the delta is therefore real, it is
`7.7e-06` nats/token against an honest loss of `1.92` — and in the **masked** arm
it is **negative**: the model does very slightly *better* when handed the wrong
document's memory.

**What this retires.** `docs/lab-notes/for-brendan-2026-09-18.md` lists three
candidate causes for the inert memory, only one of them a code defect: the missing
`srep_norm` hinge; a ~15k-token corpus memorisable without memory; and *"a loss
dominated by ~58 pad positions per sentence."* **This cycle removes the third at
50 iterations.** Fixing the loss makes the model twice as good at real tokens and
leaves the memory exactly as dead.

⚠️ **It does not retire it at 300 iterations**, which is where the original
inert-memory finding was made. See "What this run does not establish".

---

## What this run does not establish

- **50 iterations is not 300.** The prior inert-memory result was measured after
  300 iterations; this shuffle control is on a 50-iteration model. The retirement
  of the pad-dominated-loss hypothesis is **conditional on that**, and closing it
  needs the ablation re-run at the original depth.
- **`iters` is a choice, not a measurement.** 50 is `train()`'s committed default.
  The falsifier does not name it, and a different value could move the arm deltas
  (not the PAD fraction, and not the floor's order).
- **One policy, one width, one corpus.** `FIFOPolicy`, `d = 128`, `M = 16`,
  synthetic. §13: no cross-corpus comparison of absolute numbers is valid.
- **The manifest's `device_note` is imprecise.** It says the CPU floor "is measured
  here too"; it is measured in the *sibling* run
  `runs/cycle-01-masked-loss-cpu/`, not in the MPS one. The manifest is frozen and
  has deliberately not been edited.

---

## BRIEF ERRORS

Against `docs/lab-notes/dispatch-cycle-01-masked-loss.md` and its two corrections.

1. 🔴 **`--check` was not green at the baseline, and bar item 2 was not
   executable as written.** The revalidation block checked that `off_gate_allowed`
   existed at `scripts/mutation_battery.py:61` and concluded *"Bar item 2 is
   executable as written."* It is not. `uv run python scripts/mutation_battery.py
   --check` **exited 1** at the baseline tree, aborting 23 mutations in on a stale
   anchor in the `exit_code gets a default` entry — `Ledger.command`'s signature
   was reflowed onto separate lines by the Studio/trunk merge `4ece429`. It is
   entry **24 of 37**, so **12 mutations — that one and the 11 after it, including
   every canary and scoreboard gate — had not run since that merge** (counted, not
   estimated: `MUTATIONS.index(...) == 24`, `len(MUTATIONS) == 37`, minus this
   cycle's new entry). `docs/mutation-battery.md` went on asserting "36/36
   gates proven" from a pre-merge run. Verified at four shas: the anchor is present
   at `46c208e` and absent at `4ece429`, `dab4f08` and `eedc0c0`. Fixed in
   `e76c42e` (re-anchored, not dropped, per `apply()`'s own instruction); battery
   now 37/37, exit 0. *The revalidation re-executed five premises and this was not
   one of them — checking that a flag exists is not checking that the gate runs.*
2. **Body: "`TGConfig` sets `pad_id=0`" is false**, under a heading that reads
   *"The premise, verified at source."* `src/rsr/model/tg/config.py:96` is
   `pad_id: int = 50257`; the `0` is passed by `loop.py` when it builds the config.
   Confirmed independently. The revalidation caught this; recorded again because
   the body still carries it above the correction.
3. **Body: "365 lines of that package have zero tests" does not reconcile**, as the
   revalidation says. `wc -l` before this cycle: `__init__ 25`, `checkpoint 361`,
   `heartbeat 159`, `loop 307`. The untested part was `loop + heartbeat = 466`. The
   revalidation's own list of non-reconciling subsets is correct.
4. **Revalidation item 2's `iters = 8` inference was wrong** — retracted by the
   manager at `8a8a116` before the run, and the retraction is confirmed by
   measurement: the denominator is corpus-shaped, `64 × 48 × 63 = 193,536`, with no
   `iters` in it. Recorded here as the brief itself asks.
5. **Both the brief and the dispatch that commissioned it told the researcher to
   reproduce `3.1789143850602386e-07` without stating that it is a trajectory
   sup-norm at 10 beats on MPS, and neither pinned a device.** Retracted by the
   manager at `eedc0c0`. The correction is load-bearing and correct: on CPU the
   floor is exactly `0.0` and half 2 of the falsifier is vacuous, which this run
   measured rather than quoted.
6. **"Files in scope: … Nothing else" cannot produce the deliverables the same
   brief requires.** The bar asks for a `RESULTS.md`, a ledger written by
   `scripts/ledger.py`, a measured PAD fraction, a two-run floor and a shuffle
   control — none of which any of the three in-scope files can produce. This cycle
   added `experiments/cycle-01-masked-loss/{run.py,RESULTS.md}`, which is CLAUDE.md's
   own convention ("every experiment writes a `RESULTS.md` next to its `run.py`"),
   and regenerated `docs/mutation-battery.md`, which is the battery's committed
   output and is stale otherwise.
7. **"One commit on `main`" contradicts CLAUDE.md.** CLAUDE.md requires
   *"pre-registration commits land before the experiment they govern, in their own
   commit, ordered ahead in git history"*, and `scripts/ledger.py` **refuses to
   write over an uncommitted `src/`, `scripts/` or `experiments/`** — so the code
   must be committed before the ledger can exist, and the ledger cannot be in the
   same commit as the code it measures without an amend that would invalidate the
   sha the ledger records. Three commits, pre-registration first.
8. **Neither the brief nor the revalidation says which branch.** The brief says
   `main`; the dispatch put the working tree on `merge-studio-trunk`. Committed on
   `merge-studio-trunk` — switching branches is not a researcher's call.
9. **The falsifier's second half is ambiguous and the ambiguity matters.**
   *"Masking the loss … moves the final training loss"* can mean the arms' own
   reported numbers (different objectives, so tautologically different) or the same
   quantity measured in both arms. Both are reported; the second is the one with
   content, and it is the larger effect.
10. **The shuffle-control requirement and the "one defect, one cycle" rule
    collide.** *"A training run that does not report it, or reports it at ≈0, is
    refused rather than filed"* would refuse this cycle's runs, since the delta is
    ≈0 in every arm on both devices. That cannot be the intent of a cycle whose own
    hypothesis was that the loss defect might be what makes memory inert. Reported
    prominently instead of filed silently — **and the manager should treat every
    training number here as being about a model whose memory does nothing.**

---

## UNANSWERED BY THE BRIEF

- **`iters`.** Named as missing, never supplied. Chose `train()`'s committed default
  of 50 and said so.
- **Which `loss` the falsifier means** (see BRIEF ERRORS 9).
- **What to do when the shuffle control is ≈0 on the run that was supposed to fix
  it.** The rule says "refuse rather than file"; the cycle's purpose says "measure".
- **Whether the CPU floor of `0.0` should have been measured or quoted.** Measured.
- **Whether `masked_loss=False` should exist at all.** The brief says fix `step_fn`;
  it does not say how to compare arms at one sha. A documented off-switch was the
  project's own idiom, but an owner may want the defective branch deleted once the
  comparison is banked.

## BELIEVED, NOT VERIFIED

- That `iters = 50` is deep enough for the arm comparison to be the one an owner
  cares about. The direction is unambiguous; the magnitude is not established at
  any other depth.
- That the shuffle control implemented here matches the one in
  `docs/lab-notes/dispatch-S0-04-shuffle-control.md`. **S0-04 was not read**
  (out of scope, and a different cycle's brief). This implementation has **no
  positive control**: it has not been shown to return a large number on a model
  whose memory is deliberately live, so "≈0" is not yet distinguishable from "the
  instrument does not work". S0-04's bar item 1 exists for exactly this.
- That `train(masked_loss=False)` reproduces the pre-fix objective *exactly*. It
  is byte-equal in intent (`lm_token_losses(...).mean()` vs
  `F.cross_entropy(..., reduction="mean")`) but was not asserted bit-exact against
  the parent commit's trainer.
- That `docs/mutation-battery.md`'s pre-existing rows are still accurate. They were
  regenerated from the `--json` of the same 37/37 run that `--check` scored, but the
  12 mutations from the stale anchor onward had not executed since `4ece429` and
  their previously committed verdicts were therefore unwitnessed for that period.
  ⚠️ **Erratum against commit `e76c42e`'s own message, which says "13".** The count
  is **12**; the message was written before the index was counted. The commit is
  not amended because `runs/cycle-01-masked-loss/ledger.json` records that sha.
- That `_sha()` in `loop.py` is safe. It shells out to **bare `git`**, which
  CLAUDE.md forbids because Apple's git fails as an empty answer. It happens to
  resolve to a working git on this machine (verified: it returned the correct sha).
  Out of scope to fix; it is a latent false-provenance path.

## NEXT (proposed, not decided)

1. **Re-run the memory ablation at 300 iterations on the repaired loss**, with the
   shuffle control as the primary readout. This cycle retires the
   pad-dominated-loss hypothesis at 50 iterations only, and 300 is where the
   original finding lives. Cheapest decisive thing available (~10 min of CPU).
2. **Give the shuffle control a positive control and promote it out of this
   experiment script** (`docs/lab-notes/dispatch-S0-04-shuffle-control.md`). Until
   a deliberately-live memory makes it read large, every `≈0` in this project is
   uninterpretable in the same way the perplexities were.
3. **Reconsider the cycle order.** The brief fixes cycle 3 as the `srep_norm` hinge.
   With the loss defect closed and the memory still dead, the hinge is now the
   leading remaining candidate for *why*, and it outranks cycle 2's policy
   selection — which sits above a memory that does nothing.
4. **An anchor-freshness check that can run outside the battery.** The
   battery cannot host one (its own docstring says so), but a 6-line loop over
   `MUTATIONS` asserting `path.read_text().count(m.old) == 1` runs in
   milliseconds from a script and would have caught a 14-mutation silent outage
   at merge time. It is not a test in the suite, so it does not have the
   self-reference problem.
