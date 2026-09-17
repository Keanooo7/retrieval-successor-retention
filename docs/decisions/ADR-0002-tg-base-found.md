# ADR-0002 — The Thought Gestalt implementation exists. ADR-0001 was wrong.

**Status:** Accepted · **supersedes ADR-0001**
**Date:** 2026-09-17
**Found by:** session `d23e3dcd-1fa1-4db1-8bed-07a52030c52a` (Mac Studio), reported in
`~/Downloads/rsr-sprint-1-handoff-2026-09-17.md` §2. **Independently verified here** before acceptance.

> ADR-0001 is **not edited**. It stands as the record of what this session concluded on its date,
> and of how it concluded it wrongly. An ADR that gets quietly corrected teaches nothing.

---

## The correction

**`https://github.com/jlmcc94303/ThoughtGestaltCode`**

Verified 2026-09-17 by fetching the repository directly:

| | |
|---|---|
| Description | *"A JAX/Flax implementation of the Thought Gestalt (TG) model — a recurrent transformer that models language at two levels of abstraction: tokens and sentence-level 'thought' states."* |
| Owner | `jlmcc94303` — James McClelland's personal account |
| Licence | Apache-2.0 |
| Stars | 0 · 43 commits on main |
| Framework | **JAX / Flax**, not PyTorch |

ADR-0001's conclusion — *"the answer from published sources alone is no"* — was correct about the
*paper* and wrong about the *world*. The paper genuinely carries no availability statement; the code
was still findable.

## Why the search failed, which matters more than the miss

ADR-0001's evidence section is accurate and its instrument was wrong.

- It searched the **paper** exhaustively: `pdftotext | grep -iE "github|we release|code available|
  available at|open.?sourc"` across all 20 pages → one hit, the word "reproduce" in an ablation. That
  part stands.
- It then searched a **web index** for a repository and got `milenarabovsky/SG_model` — the 2018
  *Sentence Gestalt* model — and stopped.
- Session `d23e3dcd` searched **GitHub's own repository index for "thought gestalt"** and found it
  immediately. Their note: GitHub returns 0 results for `borazjanizadeh` and 0 for `2512.25026`, and
  the repo was released ~8 months after the paper and never back-linked. **Effectively
  undiscoverable by author name or arXiv id — but trivially discoverable by model name.**

🔴 **The failure is not "the search returned nothing." It is that the search returned nothing
confidently, and a 3–6 week reimplementation was proposed on the strength of it.**

**What was missing was a positive control.** A search instrument that has never been shown to find a
thing you know exists cannot support a negative result. One `SG_model`-shaped hit should have been
read as *"this index resolves model names, not this model"* — a signal about the instrument — rather
than as an answer. It costs one query to check.

## The three qualifications — all from `d23e3dcd` §2, recorded here because they change the plan

1. **JAX/Flax, and there is no Mac GPU path.** Deps are `jax`, `jaxlib`, `flax`, `optax`, `orbax`,
   `grain`, `tensorflow`, `tensorflow_datasets`; Python 3.10/3.11 only. `jax-metal`'s last release was
   **v0.1.1, 2024-10-08**, and current JAX requires Python ≥3.12 — so on the Mac Studio the reference
   would be **CPU-only**. *(Not re-verified here; taken from their handoff.)*

2. 🔴 **The release is not the paper's model.** README, verbatim and verified by fetch:
   > *"This version differs slightly from the version of the model described in (arXiv:2512.25026).
   > It achieves comparable results to those reported in the paper when trained with 12M text tokens."*

   **This withdraws ADR-0001's acceptance bar.** ADR-0001 proposed reproducing *"29.8 test PPL and
   21 sentence-steps/sec on a single A40 at `S ≈ 30`"* as the smoke test. By the authors' own
   statement that target is **unreachable from this release**. `d23e3dcd` replaces it with a
   golden-tensor fidelity harness against the pinned reference, which is the right instrument: it
   tests agreement with *the thing you actually have*, not with a number from a different model.

3. **The corpus path does not run.** `tg/Rough/preprocess_corpus.py` imports `src_recurrent.*`, which
   does not exist anywhere in the tree. RSR writes its own preprocessing regardless. The release does
   name the segmenter — **SaT `sat-3l-sm`**, with an explicit refusal to fall back to regex.
   *(Not re-verified here.)*

⚠️ Their handoff describes the repo as a *"fork-by-import of `google-deepmind/nanodo`."* **I could
not verify that** — GitHub reports it as not a fork. A fork-by-import does present as an independent
repo, so the claim is plausible; it is recorded as theirs and unconfirmed.

## What this changes

| ADR-0001 said | Now |
|---|---|
| No code; email the authors day 1 | **Withdrawn.** The code exists. An email may still be worth sending about the paper-vs-release gap, but it is no longer on the critical path and nothing waits on it. |
| Open a from-paper reimplementation, 3–6 weeks, in no schedule | **Withdrawn as framed.** The work becomes a **transcription** of a pinned JAX reference into PyTorch at `d = 128` — `d23e3dcd` estimates 3–5 days and calls it the long pole. Bounded, and checkable against golden tensors. |
| Reproduce 29.8 PPL as the acceptance bar | **Withdrawn — unreachable.** Golden-tensor fidelity against the pin, with the tolerance committed *before* the fixtures are generated. |
| E0a / E0c / E0d deferred with a named blocker | Still deferred, but the blocker is now **days of transcription**, not weeks of reimplementation. |

📌 **Gradient fixtures are mandatory in that harness**, and the reason is `d23e3dcd`'s, not mine:
gestalts are written to memory **without detaching the graph** and backward depth is bounded by `S`
([P2] App. A). JAX's functional autodiff and PyTorch's retained-graph semantics diverge exactly
there — so a forward-only match with wrong graph retention passes fidelity and survives to week 7.

## Consequence for GATE-1

`experiments/GATE-1.md` reports Task 0 as "answered: no." **That verdict is superseded by this ADR.**
The gate's other findings are unaffected. The schedule hole ADR-0001 opened — 3–6 unbudgeted weeks —
is smaller than stated and better bounded, which is the single largest schedule change either session
produced.
