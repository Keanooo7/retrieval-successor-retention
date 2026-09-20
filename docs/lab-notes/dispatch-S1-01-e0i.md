# S1-01 — E0i: the reintroduction histogram

**Baseline:** `c647af4290f7a45be2edf947b7973816dcc2af8b`. **Lane:** data · 🔴 **CPU-only — do not claim the device.**
**Sprint:** 1 — runs in parallel with Sprint 0.

## 🔴 Blocked until the owner signs

`preregistration/e0i_threshold.md` is **final and unsigned**. It says *"AWAITING SIGNATURE (§6).
NOT IN FORCE UNTIL SIGNED"* and *"it must be signed by a person, and that person is not a model."*
**No agent may sign it, and this brief does not start before it is signed.** A signature records
that a number was committed to before the data; it supplies no measurement, and **E0i stays exit 3
until T3 reports `p`.**

## Hypothesis (not instruction)

E0i is the cheapest kill gate in the project — one CPU-day against 576 GPU-hours — and it stands in
front of E3, which is where the claim lives. At `S = 80` only **16 stream positions** can host a
`k = 64` reintroduction, so the events E3 measures may simply not be there in useful numbers.

## Files in scope

`src/rsr/data/pg19.py` (stub) · `src/rsr/data/coref.py` (stub) · `experiments/e0i/run.py`
(stub) · `experiments/e0i/RESULTS.md` · `tests/test_coref_pipeline.py` (new).

`pyproject.toml` already declares the `coref` extra — fastcoref, wtpsplit, datasets, spacy — kept
out of the default install on purpose. **fastcoref is the pipeline of record** and maverick-coref is
an agreement check only, per ADR-0003; maverick is CC BY-NC-SA and never ships.

## The gate, from the pre-registration — do not restate it from memory

`n_raw × p_LCB² ≥ 150` in **every one** of `(40,48]`, `(48,56]`, `(56,64]`, **and**
`n_raw × p_LCB² ≥ 600` pooled, from **≥ 30 distinct documents**.

`p_LCB` is the **one-sided 95% Wilson lower bound** on coref precision over 100 hand-annotated
reintroductions — not the point estimate — and it enters **squared**. At `p̂ = 0.90`,
`p_LCB = 0.840` and the per-bin floor needs **213 raw events, not 186**.

🔴 **The window may not be re-sliced to pass, and the threshold does not move retroactively.**
Proposing a threshold change after seeing the histogram is not a negotiation, it is un-registering
the gate — and if you have seen the histogram you are structurally disqualified from touching the
threshold. That is retire trigger 6 and it applies to this lane specifically.

## Bar

**`p` must be measured, not assumed.** Hand-annotate the 100 reintroductions and report precision
with its Wilson lower bound. 🔴 **If `p` is not measured, E0i returns exit 3 — did not run. Not a
pass.**

Also required: a determinism hash over the pipeline output, and the per-bin counts reported
separately from the pooled count, because **a pooled-only gate passes on an empty top bin** and
`(56,64]` is where E3's claim lives.

## Done when

`experiments/e0i/RESULTS.md` carries the histogram, `p`, `p_LCB`, per-bin and pooled
`n_raw × p_LCB²`, the document count, the determinism hash, and a **PASS / FAIL / exit 3** verdict
against the signed threshold. PR merged.

## Do NOT

- Do not claim `device:mps0`. If a task here seems to need it, the brief is mis-assigned.
- Do not touch `preregistration/`. R0 owns it and the signature is the owner's.
- Do not quote a coref accuracy figure from a paper. ADR-0003 is explicit that **no published CPU
  throughput or precision figure applies here** — the pilot measures it.
- Do not report a shortfall as a failure of the project. A thin histogram has a **response ladder**
  in the pre-registration; read it before concluding anything.

## Report

Standard block, `BRIEF ERRORS`, every number above, and the wall-clock the pipeline actually took —
that figure is currently unmeasured and it is an input to Sprint 5's budget.
