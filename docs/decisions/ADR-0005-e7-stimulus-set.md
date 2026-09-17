# ADR-0005 — The E7 stimulus set

- **Status:** **proposed** — pending Brendan's sign-off and the "in hand" exit
- **Date:** 2026-09-17
- **Relates to:** kickoff T4, spec §5.3, §10.1, E0g; §16 condition 4

## Context

§5.3 requires the stimulus set be **named and confirmed in week 1**, and states the
decision criterion:

> "Verify availability, licensing, and **whether importance is scored independently
> of serial position** before week 2, and report what was found regardless."

If nothing suitable exists, **E7 is cut and falsifier 4 is withdrawn** rather than
left unfalsifiable (§16 condition 4).

## The spec's candidates all fail the spec's own criterion

| Candidate | Available? | Position-independent importance? |
|---|---|---|
| **Sherlock** (Chen et al. 2017) — [ds001132](https://openneuro.org/datasets/ds001132); recall transcripts Born et al. 2023 (Zenodo, CC BY) | Yes, but ds001132's `License` field is **empty** and ds001110 has **no License field at all** | **No.** Units are ~50 ordered, human-annotated scene segments with timestamps. No per-unit importance score ships with it |
| **Narratives** (Nastase et al.) — [ds002345](https://openneuro.org/datasets/ds002345), **CC0** verified in `dataset_description.json` | Yes, cleanest licence of the four | **No.** It is a *listening* dataset — 28 stories, 345 subjects. **No recall protocols and no importance scores at all** |
| **Thorndyke (1977)** | **No.** Paywalled at Elsevier; ERIC EJ154360 is metadata-only; no OSF/Zenodo deposit | **Yes** — plot-hierarchy level is assigned by the story grammar, and the central result is that recall tracked structural level "independent of passage content". Would require re-keying from the PDF |
| **Kintsch & van Dijk (1978)** | **No** public archive found | **Yes** in construct; protocols and propositional analyses are not downloadable |

So the two with the right construct are unavailable, and the two available ones
lack the construct.

## Decision (proposed)

**Raccah, Chen, Gureckis, Poeppel & Vo (2024), "The Naturalistic Free Recall
dataset", *Scientific Data*.**

| | |
|---|---|
| OSF | https://osf.io/h2pkv/ |
| Code | https://github.com/phoebsc/Narrative-Memory-Dataset |
| **Licence** | **CC0 1.0** |
| Scale | 4 spoken narratives, **229 participants** |
| Content | High-fidelity timestamped transcripts of **both** stimuli and recalls |
| **Position-independent importance** | **Yes.** Ships per-event probability of recall (% of participants recalling each event) **and a semantic-centrality measure the authors show predicts recall independently of the temporal-contiguity / serial-position effect** |

It is the only candidate that meets §5.3's criterion out of the box, and it is the
most permissively licensed of any of them.

**This is off the spec's ordered preference list**, which is why it is an ADR and
not a silent substitution. The spec's list is ordered by *preference*; its
*criterion* is position-independence, and the criterion selects this dataset.

## Still required before E7 is green

1. **Brendan's sign-off** on departing from the named list.
2. **"In hand", not just "identified"** — the kickoff treats these as separate
   exits. CC0 means no data-use agreement, so there is no calendar risk here, which
   is itself a reason to prefer it.
3. **Unit mapping, pre-registered** (§10.1). Human norms are scored per event;
   TG slots are SaT-segmented sentences. Aggregate to the containing sentence (max
   or mean importance — **pre-register which**) and report the fraction of
   sentences carrying mixed high/low-importance content. **Above ~30%, the
   correlation is between two different things and must be reported as such.**
4. **The two §10.1 preconditions**, both checked before the result is interpreted:
   - **Capacity mismatch.** Run TG/FIFO on the passages at `M ∈ {8, 16, 40}` and
     confirm perplexity degrades gracefully. D-5 funded the separately-trained
     `M = 8, d = 128` model, so this is a robustness report rather than a gate —
     but it is still checked.
   - **Comprehension floor.** Does next-sentence loss distinguish plot-central from
     peripheral content at all? **If not, E7 measures noise and the null is
     reported as uninformative, not as evidence against the levels effect.**
5. **Controls are H2O and LRU, not FIFO** (defect C3). Under FIFO survival time is
   `min(M, S−i)`, a deterministic function of serial position; the raw correlation
   re-measures position and the partial correlation zeroes it out by construction.
   "RSR beats FIFO" here is true trivially and means nothing.
