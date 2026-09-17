# ADR-0001 — Is the Thought Gestalt implementation available?

**Status:** Accepted
**Date:** 2026-09-17
**Decides:** Task 0 of the Sprint 1 kickoff prompt. Blocks E0a, E0c, E0d.

---

## Context

`rsr_model_spec_v0.5` §3.1 treats TG as given — *"Base model (unmodified TG) **[E]**"* — and budgets
zero hours anywhere in §4.1, §6 or §8 for building or reproducing it. The entire project starts from
a working 12-layer recurrent transformer with sentence-level cross-attention, gestalt writes that
retain their computation graph, depth-stratified memory gates, and [P2]'s reported 29.8 test PPL.

[P2] is arXiv:2512.25026v2, submitted 2025-12-31, revised 2026-01-12, **under review**. The spec
never states whether its code is available.

## Decision

**No code-availability statement exists in the paper.** The answer from published sources alone is
**no**. We email the authors on day 1 and, in parallel and without waiting, begin a from-paper TG
reimplementation.

## Evidence

The negative result, with where it searched — a search that does not name its scope is not
falsifiable.

**Source:** `https://arxiv.org/pdf/2512.25026` (v2), fetched 2026-09-17, 4,987,936 bytes,
**20 pages**, 81,427 characters of extracted text.

```
$ pdftotext -layout 2512.25026v2.pdf - | grep -n -iE \
    "github|code (is |will be )?(available|released)|we release|\
     availab(le|ility) (at|upon)|open.?sourc|reproduc"
422:  mechanism can reproduce TG's gains without recurrence or an external memory, we implement a
```

One hit across the whole paper, and it is the word "reproduce" inside a sentence about an ablation.
**Zero** hits for `github`, `we release`, `code available`, `available at`, `available upon request`,
or `open source`.

Also checked, all negative:
- The arXiv abstract page (`arxiv.org/abs/2512.25026`) — no Code, Papers-with-Code, or Hugging Face
  link in its listing; no code statement in the abstract or comments; no venue in the comments.
- Web search for a TG repository. The nearest hit is `github.com/milenarabovsky/SG_model` — the
  **Sentence Gestalt** model (Rabovsky, Hansen & McClelland 2018), which is earlier related work by
  an overlapping group and **is not this architecture**. Do not vendor it by mistake.

## What this costs

The plan artifact prices reimplementation at **3–6 weeks**, and correctly identifies it as the
largest unpriced item in the project. Nothing in the 12-week schedule (or the amended 14-week one)
has a branch that can absorb it.

Two of week 1's four kill gates need a running TG:

| Gate | Needs TG | Status |
|---|---|---|
| E0i reintroduction histogram | no — CPU, corpus only | **proceeds** |
| E0g stimulus set | no — literature and licensing | **proceeds** |
| E0c memory/throughput | **yes** | deferred, named blocker |
| E0d `r_i` vs LOO Δloss | **yes** (needs a briefly-trained TG) | deferred, named blocker |
| E0a μP coordinate check | **yes** | deferred, named blocker |

## Why parallel rather than sequential

Waiting on the email stalls three gates for an unknown number of days against a correspondent with
no obligation to reply. Building immediately costs agent time, not owner time.

And the reimplementation is **not wasted if the code arrives**. E0b's entire job is to prove
bit-exact reduction to stock TG. A bit-exactness claim is far stronger against two independent
implementations than against one, and [P2] is under review and may change materially (§13 says so).
If the authors' code lands, the from-paper build becomes E0b's independent cross-check and the
reference for any divergence.

## Consequences

1. Sprint 1 returns E0a, E0c and E0d as **deferred with a named blocker** — not as passes, and not
   as failures. GATE-1 records them that way.
2. A TG reimplementation lane opens now. Its acceptance bar is [P2]'s own reported numbers:
   **29.8 test PPL** and **21 sentence-steps/sec on a single A40 at `S ≈ 30`**, at
   `d_model = 768`, 12 layers, `M = 40`, `L_max = 64`, gestalt from layer 7 `<EOS>`.
   ⚠️ Those numbers are at **85.6M non-embedding parameters** — see `docs/spec-corrections.md` B-2.
   They are the *reproduction* target, not the configuration RSR runs at.
3. `src/rsr/model/tg/` holds the reimplementation. If vendored code arrives it goes to
   `src/rsr/model/tg_upstream/` with a pinned commit, and this ADR is superseded, never edited.
4. The schedule consequence is real and is not absorbed anywhere. It is stated in GATE-1 for the
   owner to decide against, which is what the kickoff prompt asked for.

## The email

Draft at `docs/decisions/ADR-0001-email-draft.md`. Send on day 1. If no reply within 5 business
days, the reimplementation is the path and no further decision is needed — that is the point of
starting it now.
