# ADR-0001 — The TG base model: vendor the JAX reference, reimplement in PyTorch

- **Status:** accepted
- **Date:** 2026-09-17
- **Answers:** Sprint 1 kickoff, Task 0
- **Supersedes:** nothing. **Superseded by:** nothing.

## Question

The spec treats TG as given — §3.1 is headed *"Base model (unmodified TG) **[E]**"*
— and budgets zero hours for building it. Is [P2]'s implementation public?

The kickoff's instruction: *"stop and report before proceeding if the answer is no.
If TG's code is not available, reproducing it to the point where E0b's
bit-exactness test means anything is 3–6 weeks that exist in no schedule."*

## Answer: yes, with three qualifications that change the plan

### The paper

| Field | Value |
|---|---|
| arXiv | [2512.25026](https://arxiv.org/abs/2512.25026) |
| Title | Modeling Language as a Sequence of Thoughts |
| Authors | **Nasim** Borazjanizadeh, James McClelland (Stanford; "authors contributed equally") |
| v1 | 2025-12-31 |
| v2 | 2026-01-12 |
| Categories | cs.CL (primary), cs.AI |

Verified via the arXiv API (`export.arxiv.org/api/query?id_list=2512.25026`), not
only the abs page. The spec's "under review" status is **not stated on arXiv** and
remains unverified.

The abs page carries **no code link**. Its "Code, Data and Media" tab is arXiv's
generic widget and has no entries. alphaXiv has none either. PapersWithCode is
defunct — `paperswithcode.com/api/...` now redirects to HuggingFace — so any
instruction to "check PWC" is dead.

### The code

**https://github.com/jlmcc94303/ThoughtGestaltCode**

| Field | Value |
|---|---|
| Owner | `jlmcc94303` — Jay McClelland's personal account |
| Created | 2026-08-20 |
| Last push | 2026-09-02 |
| Licence | **Apache-2.0** |
| Language | **Python / JAX / Flax** |
| Stars | **0** |
| Upstream | fork-by-import of [`google-deepmind/nanodo`](https://github.com/google-deepmind/nanodo) (Apache-2.0) |
| Release commit | `a0079d08` "Release Thought Gestalt: sentence-level recurrent LM built on NanoDO" (2026-08-21, `utkrisht12`) |
| Head at pin time | `f220b109` "Add corpus preprocessing and dataset subset scripts." (2026-09-02) |

Below the release commit, the history is Peter J. Liu / Pierre Marcenac Copybara
commits from 2024 — the NanoDO import. **Nasim Borazjanizadeh is not a committer.**

**It is effectively undiscoverable.** GitHub returns 0 results for a user or repo
search on `borazjanizadeh`, and 0 for the string `2512.25026` across repos. It
surfaced only on a repo search for `"thought gestalt"` (1 result). It was released
~8 months after the paper and never back-linked into the arXiv listing.

### Qualification 1 — it is JAX/Flax, and the kickoff mandates PyTorch

From `pyproject.toml`: `jax`, `jaxlib`, `flax`, `optax`, `orbax`, `grain`,
`sentencepiece`, `tensorflow`, `tensorflow_datasets`, `clu`, `ml-collections`.
`requires-python = ">=3.10"`; the README says Python 3.10 or 3.11.

The kickoff mandates *"Python 3.11+, PyTorch"* and *"Target hardware for this
sprint is a 64 GB Mac Studio on MPS."* These are incompatible with the reference:

- **`jax-metal` is abandoned.** Last PyPI release v0.1.1, **2024-10-08**. Current
  JAX is 0.11.1 and requires Python ≥ 3.12, which the reference forbids.
- So a vendored JAX TG runs **CPU-only** on the target machine. There is no GPU
  path for it on Apple silicon.

### Qualification 2 — the release is not the paper's model

README, verbatim:

> "This version differs slightly from the version of the model described in
> ([arXiv:2512.25026](https://arxiv.org/abs/2512.25026)). It achieves comparable
> results to those reported in the paper when trained with 12M text tokens."

The kickoff's smoke test — *"reproduce its reported numbers (29.8 test PPL, 21
sentence-steps/sec on one A40 at `S ≈ 30`)"* — is therefore **unreachable by the
authors' own statement**. See `docs/spec-corrections.md` correction 14.

`EXPERIMENTS.md` adds, on the gist-masking baseline:

> "The gist mask here is derived from the paper's description rather than from the
> original implementation, whose mask indexes the gist flag on the query axis
> instead of the key axis and so grants no cross-sentence access at all. Numbers
> from this config are therefore not expected to match published ones."

That is an admission of a bug in a *published* baseline. It does not affect RSR
directly — TG itself is what we transcribe — but it is a caution about treating
[P2]'s reported comparisons as settled.

### Qualification 3 — the corpus path does not run

`tg/Rough/preprocess_corpus.py` imports `src_recurrent.pipelines.data.io`,
`src_recurrent.core.tokenizer_setup`, and `src_recurrent.core.data.sentence_splitter`.
**`src_recurrent` does not exist anywhere in the repo tree.** It uses SaT
(`sat-3l-sm`) for sentence splitting and explicitly refuses a regex fallback.

RSR writes its own preprocessing. We keep SaT to match both this and §10.1's
"SaT-segmented sentences".

## Decision

**Vendor the JAX reference pinned and read-only. Reimplement TG in PyTorch.
Validate the reimplementation against golden tensors extracted once from the
reference.**

1. **PyTorch is the only project stack.** JAX is **not** installed on the Mac and
   is **not** a project dependency. `third_party/ThoughtGestaltCode/` is a pinned
   source reference for reading, plus a one-time tensor extraction.
2. **The extraction runs on rented hardware.** ~1 h, inside the ~$100 weeks-1–4
   envelope. Nothing JAX touches the development machine.
3. **MPS is best-effort, not required.** §4.4 gives the Mac only the coordinate
   check, the reduction test, the synthetic corpus and interactive debugging. CPU
   serves all four, and is better for E0b's determinism.
4. **Transcribe `d = 128` only this sprint.** `d` remains a config axis; other
   widths instantiate at E0a and E5.

## Why not the alternatives

**Build RSR in JAX/Flax.** Strongest argument: E0b's bit-exactness would be against
the actual reference, and T5/T6/T7 could start with no transcription — the best
shot at hitting the week-1 gates on time. Rejected because it abandons the
mandated stack, puts every RSR component (bilinear head, μP groups, shadow buffer)
in Flax, drags TensorFlow + TFDS into the dependency tree on Python 3.11, and
leaves the development machine CPU-only with no GPU path at all.

**Full PyTorch reimplementation with no reference extraction.** Rejected: it
discards the one artifact that makes the port checkable. Without golden tensors,
"my reimplementation matches my reimplementation" is the only test available, and
a transcription error in the cross-attention or the gestalt write survives
undetected.

**The hybrid is cheap precisely because the reference exists.** The kickoff's
3–6 week estimate is for reverse-engineering an architecture from a paper. We have
24 KB of `tg_model.py`, 7 KB of `tg_cross_attention.py` and 4 KB of
`tg_srep_head.py` to transcribe from. Estimate: **3–5 days at `d = 128`.**

## Consequences

- E0b changes meaning and is split in two. See ADR-0002 and the docstrings of
  `tests/test_reduction.py` and `tests/test_fidelity.py`.
  - `test_reduction.py` (E0b proper) is **intra-repo by design**: RSR under the
    §3.7 switches vs the PyTorch TG, bit-exact. It guarantees that E3's FIFO and
    RSR arms differ only in the eviction rule. That is what §3.7 actually needs.
  - `test_fidelity.py` is new and carries the external claim, to a tolerance.
- The critical path for Sprint 1 becomes
  `T8 (provider) → golden tensors → transcription → fidelity green → {E0b, E0c, E0d, E0a}`.
  T5/T6/T7 sit behind a 3–5 day transcription and may report *in progress* at
  GATE-1.
- §13's evidence-stops list gains an entry: **the PyTorch TG is a transcription of
  a JAX reference that is itself not the paper's model.** Two hops from the
  published numbers, and the writeup says so.

## Evidence

```
gh api repos/jlmcc94303/ThoughtGestaltCode
gh api repos/jlmcc94303/ThoughtGestaltCode/commits
gh api "repos/jlmcc94303/ThoughtGestaltCode/git/trees/main?recursive=1"
gh api repos/jlmcc94303/ThoughtGestaltCode/contents/{pyproject.toml,README.md,EXPERIMENTS.md}
curl -s https://pypi.org/pypi/jax-metal/json      # latest 0.1.1, 2024-10-08
curl -s https://pypi.org/pypi/jax/json            # latest 0.11.1, requires-python >=3.12
curl -s "http://export.arxiv.org/api/query?id_list=2512.25026"
```

Pin recorded in `third_party/PINS.md` once the vendor step runs.
