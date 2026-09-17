# ADR-0003 — Coreference tooling for E0i: fastcoref ships, maverick checks

- **Status:** accepted
- **Date:** 2026-09-17
- **Relates to:** kickoff T3, spec §6 E0i, defect D-2

## Context

D-2 names coreference as *"an unnamed dependency with its own error rate feeding
straight into the dependent measure"*, then prescribes only a **population** check
(the gap histogram), which cannot see precision. E0i feeds the **primary dependent
measure** — reintroduction loss vs gap `k` — so model quality and *measured* error
rate matter more than which tool is picked.

## Decision

**fastcoref is the pipeline of record. maverick-coref is an agreement/ceiling check
only.**

| Tool | Licence | Role |
|---|---|---|
| [fastcoref](https://github.com/shon-otmazgin/fastcoref) | **MIT** | **Runs the full 30M-token subset.** The pipeline of record |
| [maverick-coref](https://github.com/SapienzaNLP/maverick-coref) | **CC BY-NC-SA 4.0** | The hand-annotated sample and a ~1% slice. Ceiling check only |
| [coreferee](https://github.com/richardpaulhudson/coreferee) | MIT | Floor / fallback. Rule+ML hybrid, genuinely CPU-oriented |
| AllenNLP | Apache-2.0 | **Not used — archived since 2022-11** |

Rationale:

- **fastcoref's permissive licence leaves the reproducibility story
  unencumbered** for anyone re-running the thesis, and it is fast enough to re-run
  when the eval split grows — which is rung 1 of the pre-registered response ladder,
  so re-running is an expected operation, not an exception.
- **maverick is more accurate (SOTA on CoNLL-2012, ACL 2024) but non-commercial**,
  and the ShareAlike term reaches outputs. That does not need deciding, because it
  is only ever an evaluation check and never ships.
- If maverick **materially beats** fastcoref on the annotated reintroductions, that
  is itself a finding about how load-bearing the coref dependency is, and it goes in
  GATE-1. If they agree, fastcoref ships and the question closes in week 1.
- Non-commercial use is acceptable here — academic research, results published, no
  product. *Not legal advice; the point is that the question does not need to be
  decided, because of how maverick is used.*

**Both licences are recorded verbatim above with the specific use of each.**
"Which coref model, under what licence, with what measured precision" is a
first-ten-minutes committee question, and the answer should be written in week 1,
not reconstructed in week 12.

## Sentence segmentation

**SaT** (`wtpsplit`), matching §10.1's "SaT-segmented sentences" and the TG
reference's `sat-3l-sm`. The reference's preprocessing script explicitly refuses a
regex fallback; we keep that refusal.

## Throughput

**No published CPU tokens/sec or docs/sec figure exists for any of these tools.**
Every published number is GPU-relative (maverick's "170x faster inference";
fastcoref's "2.8K OntoNotes docs in 25 s on a V100"). **The 1M-token pilot measures
it rather than citing one** — §12.4.

## Precision and recall

T3 hand-annotates **100 reintroductions** and reports precision and recall against
both systems. Annotation is by the agent, with a random 25 spot-checked by Brendan;
agreement is reported in `experiments/e0i/RESULTS.md`. The provenance caveat is
stated there regardless of the outcome.
