# ADR-0001 — TG code request, draft

⚠️ **DRAFT ONLY. NOT SENT.** Sending is an outward-facing action to real researchers and is the
owner's to take. Nothing in this repo sends it.

**To:** `nasimb@stanford.edu`, `jlmcc@stanford.edu`
*(published on the title page of arXiv:2512.25026v2 — not guessed)*

**Subject:** Thought Gestalt — code availability for a follow-on retention-policy study

---

Dear Nasim Borazjanizadeh and Professor McClelland,

I'm a student working on a follow-on study to *Modeling Language as a Sequence of Thoughts*
(arXiv:2512.25026). I'm building a learned replacement for TG's FIFO memory eviction — scoring each
gestalt by predicted future retrieval demand conditioned on the current discourse state — and
testing whether a policy trained only on next-sentence prediction converges on something like
Kintsch & van Dijk's leading-edge strategy.

The design depends on TG being reproducible closely enough that I can demonstrate an exact reduction:
with the retention head switched off, my system must produce a bit-identical loss curve to stock TG.
That test is what keeps the comparison honest, and it only means something against a faithful base.

Two questions:

1. **Is the implementation available**, or planned for release? I could not find a repository or a
   code statement in v2 and wanted to ask rather than assume.
2. If not, would you be willing to confirm a few details I am reconstructing from the paper? Chiefly
   the sentence-head and memory-gate initialisation, the exact cross-attention position encoding
   over memory slots, and the `<EOS>` down-weighting schedule. I have the reported ablation numbers
   (29.8 test PPL, 21 sent./sec on one A40) as my reproduction target.

I am reimplementing from the paper in the meantime and expect to continue either way — access would
mainly let me spend the time on the retention question instead of on the base model, and would let
me validate my reimplementation against yours.

Very happy to share what I find, and to send the reduction test back if it's useful to you.

With thanks,
Brendan Keane

---

## Notes for the sender

- Do not attach the spec. It is on its fifth revision, carries an unfilled §15, and is an internal
  working document.
- The reproduction target and the ablation numbers in this draft are **verified against the paper**
  (`docs/citation-audit.md`), so they can be quoted safely.
- If they reply with code: vendor to `src/rsr/model/tg_upstream/`, pin the commit, supersede
  ADR-0001 with a new ADR rather than editing it, and keep the reimplementation as E0b's independent
  cross-check.
- If no reply within 5 business days: no decision is needed. The reimplementation lane is already
  the path.
