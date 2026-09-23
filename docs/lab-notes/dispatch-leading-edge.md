---
id: leading-edge
item: eng-leading-edge
baseline_sha: 22d66bb216e08b3ff68cef08228d5c57a46b0b9b
falsifier: >-
  The implemented leading-edge policy selects differently from Kintsch & Vipond's verbatim
  rule (Kintsch & van Dijk 1978, p. 379) on a hand-built coherence graph, or its evictions on
  the synthetic corpus cannot be told apart from FIFO's.

anchors:
  # the stub, and the docstring correction 26 says is wrong
  - {path: src/rsr/baselines/leading_edge.py, line: 4, expect: "highest in the macrostructure"}
  - {path: src/rsr/baselines/leading_edge.py, line: 26, expect: "def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:"}
  # the source, as audited
  - {path: docs/citation-audit.md, line: 205, expect: "https://www.cl.cam.ac.uk/teaching/1516/R216/Towards.pdf"}
  - {path: docs/citation-audit.md, line: 209, expect: "The 'leading-edge strategy,' originally proposed by Kintsch and Vipond (1978)"}
  - {path: docs/citation-audit.md, line: 211, expect: "up all propositions along the graph's lower edge"}
  - {path: docs/spec-corrections.md, line: 626, expect: "## 26 — the leading-edge strategy runs on the MICROstructure"}
  - {path: docs/spec-corrections.md, line: 641, expect: "This is an implementation bug waiting to happen."}
  - {path: docs/spec-corrections.md, line: 647, expect: "`s = 10` fails outright"}
  # the interface the policy plugs into
  - {path: src/rsr/retention/policy.py, line: 49, expect: "written_at: Tensor"}
  - {path: src/rsr/retention/policy.py, line: 204, expect: "def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:"}
  - {path: src/rsr/baselines/fifo.py, line: 29, expect: "def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:"}
  # the synthetic corpus's arguments
  - {path: src/rsr/data/synthetic.py, line: 128, expect: "_FILLERS: tuple[str, ...] = ("}
  - {path: src/rsr/data/synthetic.py, line: 192, expect: "class Sentence:"}
  - {path: src/rsr/data/synthetic.py, line: 255, expect: "entity = _ENTITIES[rng.randrange(len(_ENTITIES))]"}

premises:
  - claim: the policy is still a stub with four raises
    check: "git grep -c NotImplementedError -- src/rsr/baselines/leading_edge.py"
    expect_rc: 0
    expect_stdout_contains: ":4"
  - claim: the stub docstring still carries the macrostructure reading correction 26 retracts
    check: "git grep -n 'highest in the macrostructure' -- src/rsr/baselines/leading_edge.py"
    expect_rc: 0
  - claim: no coherence-graph module exists
    check: "test -e src/rsr/baselines/coherence.py"
    expect_rc: 1
  - claim: no leading-edge test exists
    check: "test -e tests/test_leading_edge.py"
    expect_rc: 1
  - claim: synthetic sentences carry no structured argument field (arguments must be derived)
    check: "git grep -n -E 'arguments|entity:' -- src/rsr/data/synthetic.py"
    expect_rc: 1

files_in_scope:
  - src/rsr/baselines/leading_edge.py
  - path: src/rsr/baselines/coherence.py
    new: true
  - path: tests/test_leading_edge.py
    new: true
  - path: tests/fixtures/kvd1978_figure1.json
    new: true
  - src/rsr/data/synthetic.py
  - scripts/mutation_battery.py

bar:
  - "Fidelity: on a KvD 1978 Figure 1 fixture (transcribed with page numbers), the selection at s=4 equals the verbatim p.379 rule applied by hand, the hand derivation written in the test's docstring. If the paper prints the selected set, the test asserts the printed set and says so."
  - "Distinguishable from FIFO: on generate(SyntheticConfig(seed=s)) for s in 0..2 at M=16, the number of evictions where the leading-edge victim differs from FIFO's victim is > 0 for every seed. The test reports the count, measured, never assumed. And the current graph's top unit is never the victim (it is picked first, so it is never lowest priority)."
  - "Protocol: isinstance(policy, RetentionPolicy); select_eviction returns a live slot; reset() clears the graph; on_write registers the new unit; observe() is a documented no-op (the 1978 rule reads no attention)."
  - "uv run pytest -rs --tb=no census line and $? quoted; ruff check and ruff format --check exit 0; scripts/mutation_battery.py N/N line quoted."
  - "Three mutations, each reddening only its declared test: (1) drop 'as long as each is more recent' on the lower edge. Under this brief's own construction a child always postdates its parent, so this clause is vacuous on synthetic graphs. The test for it must use a hand-built fixture graph whose lower edge has a non-increasing step (KvD's Figure 1, where propositions of one cycle are numbered independently of attachment, is the natural candidate). If no such fixture can be justified from the source, drop mutation (1) and say so; (2) phase 2 picks oldest-first; (3) an empty argument set counts as overlapping everything."
done_when:
  - "The module docstring quotes p.379 verbatim and states the graph construction used, with the page it comes from, or says it is an interpretation where the source is silent."
  - "leading_edge.py:4's macrostructure sentence is gone (correction 26)."
  - "The report lists every interpretive choice made where the source is silent, each as a BRIEF ERROR candidate."
  - "BRIEF ERRORS written out, 'none' if none."
do_not:
  - "Build a macrostructure or use macro-operators (correction 26)."
  - "Simulate reinstatement search. Count the trigger (a new unit sharing no argument with the buffer); do not act on it."
  - "Run a training run or report an eviction-quality number. A hit-rate comparison against FIFO/oracle is a measurement and needs its own PREREG (next item, not this one)."
  - "Tune s to a result. s = M in the policy; the fixture uses KvD's s = 4."
  - "Touch retention/rsr.py, the model, or the loop."
---

# Brief: the leading-edge baseline, on the microstructure coherence graph

**Status:** written, not started. **Written at:** `22d66bb216e08b3ff68cef08228d5c57a46b0b9b`.
**Lane:** researcher (code only, CPU, no training). **Queue item:** `eng-leading-edge`.

## Why

The project's primary comparator is not FIFO but the leading-edge strategy
(`docs/ROADMAP.md` §4, correction 25). E7's headline number is the agreement between RSR's
survival ordering and this strategy's. The policy is a stub
(`src/rsr/baselines/leading_edge.py:26`), and its docstring carries the macrostructure reading
that correction 26 retracts (`:4`; `docs/spec-corrections.md:626`). Correction 26 says the brief
that implements it **must specify the coherence graph** (`:641`), and this brief does below.

This runs on CPU with no model, so it proceeds in parallel with `retrieval-curve` without
contending for the device.

## The rule, verbatim (p. 379, `docs/citation-audit.md:209`)

> *"Start with the top proposition in Figure 1 and pick up all propositions along the graph's
> lower edge, as long as each is more recent than the previous one (i.e., the index numbers
> increase); next, go to the highest level possible and pick propositions in order of their
> recency (i.e., highest numbers first); stop whenever s propositions have been selected."*

## The coherence graph (specified here, per correction 26)

`src/rsr/baselines/coherence.py`, pure Python, no torch.

- **Unit:** one sentence = one slot. On the synthetic corpus a sentence is exactly one
  proposition, so the mapping is 1:1. On PG-19 a sentence holds several propositions and this is
  an approximation. Say so in the docstring.
- **Arguments:** a sentence's argument set. For the synthetic corpus, add
  `sentence_arguments(s: Sentence) -> frozenset[str]` to `src/rsr/data/synthetic.py`, derived
  from the generator's own tables (`_ENTITIES`, `_OBJECTS`), not by guessing at text:
  - assert `"E P O."` → `{E, O}`
  - query `"What does E P?"` → `{E}` (the object is the answer)
  - filler (`:128`) → `∅`
- **Construction:** the first unit of a stream is the top (level 1). A new unit attaches one
  level below the **highest-level** unit (smallest level number) in the current graph with which
  it shares an argument. A unit sharing no argument with any unit in the graph is a **coherence
  failure**: count it (the reinstatement trigger, `docs/citation-audit.md`, the "minor" note),
  place it at level 1, and do nothing more.
  - 🔴 **This construction is this brief's reading of KvD pp. 377–379, not a quotation.** Fetch
    the source (URL at `docs/citation-audit.md:205`) and compare. Where the source differs, follow
    the source and file the difference as a BRIEF ERROR with the quote. If the source cannot be
    fetched, say so; the fidelity bar then becomes a check of the transcribed rule.
- **Selection** (the verbatim rule, operationalised):
  - Phase 1: from the top, walk the **lower edge**. At each level, take the most recent unit
    attached under the current one (**interpretation**: the quote says "the graph's lower edge"
    and does not define it). Continue only while index numbers increase.
  - Phase 2: from the highest level with unpicked units, take units in descending index
    ("highest numbers first"), moving down a level when one is exhausted (**interpretation**:
    the quote does not say what happens when a level is exhausted).
  - Stop at `s`.
  - "The highest level possible" is ambiguous between "closest to the top" and "the highest
    level not yet exhausted". Choose one, state it, and file it as an interpretive choice.

## The policy

`LeadingEdgePolicy(arguments: Callable[[int], frozenset[str]], s: int | None = None)`.
`arguments(step)` is the argument set of the sentence written at `step`. The policy maps slots to
steps with `MemoryState.written_at` (`src/rsr/retention/policy.py:49`).

On `select_eviction` (called only when memory is full, `:204`):
1. Build the priority order: run the selection over the graph of the live units plus the
   incoming one, with `s = M`. The pick order is the priority.
2. The incoming unit is always written, so evict the live slot with the **lowest** priority.
   Units never picked rank below all picked units. Ties go to the oldest.

`s` defaults to `M`. KvD's fitted `s ≈ 1–4` (`docs/spec-corrections.md:647`) is for the fixture
and for E7 at `M = 8`, not a default here.

## Falsifier

In the front matter. Two halves, and either falsifies: fidelity to the verbatim rule, and
distinguishability from FIFO.

## Files in scope

- `src/rsr/baselines/leading_edge.py`: implement the policy, and replace the `:4` docstring.
- `src/rsr/baselines/coherence.py` (**new**): the graph and the selection.
- `src/rsr/data/synthetic.py`: add `sentence_arguments` only. The generator is untouched, so
  the corpus stays byte-identical.
- `tests/test_leading_edge.py` and `tests/fixtures/kvd1978_figure1.json` (**new**).
- `scripts/mutation_battery.py`: three entries.

## Bar

In the front matter. The synthetic property test is the one gate that fails where nothing else
does: a policy that is FIFO under another name passes the protocol and fidelity checks on any
graph where recency and the lower edge agree.

⚠️ **An earlier draft's bar was false and was caught in pre-push review.** It said the policy never evicts an assert while a filler is live. Under this construction fillers and non-overlapping asserts all sit at level 1, and phase 2 picks by recency whatever the kind. An independent simulation over seeds 0–2 at M=16 counted 741 such evictions for leading-edge against 605 for FIFO, out of 6,144 each. **The leading-edge strategy is not an answer-aware oracle, and no test may assume it is.**

## Done when

In the front matter.

## Do NOT

In the front matter. Also (`CLAUDE.md`): stage by explicit path; literal output or it did not
happen.
