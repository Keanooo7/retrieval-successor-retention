"""Demand-controlled synthetic corpus -- the gate (spec section 5.3).

Documents of `L` sentences; a fact asserted at `i` is queried at `i + k`, with `k`
from a mixture of short-range geometric and a heavy tail, **max gap 40**.

Ground-truth discounted demand is known per slot per step, so `psi_hat` is
measurable against truth and eviction regret against oracle -- directly, not
through perplexity. **The cheapest and most diagnostic artifact in the project.**

`S = 48` here, because `S` must exceed the max gap (section 5.2). With
`A_max = S = 48` (ADR-0004) every swept `gamma` including 0.97 is legal.

**Caveat to state in the writeup:** at `gamma = 0.97` the horizon (33) is shorter
than the generator's max gap (40), so the longest synthetic gaps are attenuated to
~0.30 weight. Attenuated, not invisible.

**E1 on this corpus is a kill gate, not a result** (defect D-9). The generator
constructs the gap structure that makes lookahead pay, so `RSR > RSR(gamma=0)`
here is evidence about the optimizer, not about language.

---

## Determinism is a requirement, not a nicety

E0b asserts a **bit-exact** loss curve, and E1/E2 compare arms that must differ only
in the eviction rule. A corpus that is not byte-identical across processes turns
every one of those comparisons into a comparison of two datasets.

Three hazards this module avoids, all of which produce "reproducible" data that is
not:

1. **The global RNG.** Nothing here touches `random` or `np.random` module state.
   Every draw comes from a `random.Random` seeded per document, so generating
   document 7 does not depend on whether documents 0-6 were generated first.
2. **Set and dict iteration.** No output ordering derives from a `set`, and no key
   is a string whose hash varies with `PYTHONHASHSEED`. Entity and predicate choice
   is by integer index into a fixed tuple.
3. **Float formatting.** The serialized form carries integers and short strings
   only; the demand matrix is derived, not stored, so no float repr enters the
   bytes that are compared.

`generate(...)` is a pure function of `SyntheticConfig`. `to_bytes` is its canonical
encoding, and `tests/test_synthetic.py` runs it in two separate interpreters.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

__all__ = [
    "ANSWER_SYMBOLS",
    "Document",
    "Sentence",
    "SyntheticConfig",
    "answer_symbol",
    "discounted_demand",
    "fraction_of_pairs_beyond",
    "generate",
    "to_bytes",
    "true_demand",
]

SentenceKind = Literal["assert", "query", "filler"]

# Fixed vocabularies. Tuples, indexed by integer -- never a set, never hashed.
_ENTITIES: tuple[str, ...] = tuple(
    f"{a}{b}"
    for a in ("Ada", "Bo", "Cy", "Del", "Eli", "Fen", "Gus", "Hal")
    for b in ("", "-1", "-2", "-3", "-4", "-5", "-6", "-7")
)
_PREDICATES: tuple[str, ...] = (
    "lives in",
    "owns",
    "studies",
    "avoids",
    "recalls",
    "repairs",
    "visits",
    "collects",
    "translates",
    "abandons",
    "buys",
    "teaches",
    "measures",
    "plants",
    "sells",
    "drafts",
)
_OBJECTS: tuple[str, ...] = (
    "the harbour",
    "a brass key",
    "old maps",
    "the north road",
    "a grey cat",
    "the ledger",
    "salt marshes",
    "an oak chest",
    "the ferry",
    "wire rope",
    "dry timber",
    "a cracked bell",
    "the west wing",
    "lamp oil",
    "the tide table",
    "a bone comb",
)


def answer_symbol(obj: str) -> str:
    """The single token a query's answer is emitted as (S0-03).

    **Single-symbol, not word-level**, so that chance on the answer is one number
    -- `ln(len(ANSWER_SYMBOLS)) = ln 16` -- rather than a different figure at each
    word position (`"the ..."` leaves 6 continuations, `"a ..."` 4). Joined with
    `_`, which no other word in the corpus contains, so a symbol never collides
    with a word of the assert sentence it must be retrieved from.
    """
    return obj.replace(" ", "_")


ANSWER_SYMBOLS: tuple[str, ...] = tuple(answer_symbol(o) for o in _OBJECTS)
"""The 16 answer tokens, in `_OBJECTS` order. Chance on an answer is
`ln(len(ANSWER_SYMBOLS))` for a model that knows only that one of these comes next."""

_FILLERS: tuple[str, ...] = (
    "The weather turned.",
    "Nothing else happened that day.",
    "The room stayed quiet.",
    "Time passed.",
    "A door closed somewhere.",
    "The light changed.",
    "No one spoke.",
    "The hour grew late.",
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Section 5.3's generator, with every knob that affects the bytes.

    `max_gap = 40` and `sentences_per_document = 48` are section 5.2's: `S` must
    exceed the max gap, or the longest gaps do not exist in the data at all -- the
    largest hole in v0.1, restated as a config invariant here.
    """

    n_documents: int = 64
    sentences_per_document: int = 48  # S, synthetic scope
    max_gap: int = 40
    geometric_p: float = 0.30
    """Short-range component. Mean gap ~1/p, so ~3.3 sentences."""

    heavy_tail_weight: float = 0.35
    """Probability a gap is drawn from the long component instead. Section 5.3
    wants the long gaps to be common enough to train on, not merely present."""

    heavy_tail_min: int = 12
    seed: int = 0

    answer_in_stream: bool = True
    """S0-03. `True` appends the query's answer to the query sentence as one token
    (`answer_symbol`), so the next-token objective has a target that can only be
    predicted by retrieving the asserted fact. `False` is the **pre-S0-03 corpus,
    byte for byte**: the answer lived only in `Sentence.answer`, out of band, and
    no cross-entropy target required retrieval at all -- which made "the memory is
    inert" a finding about the corpus rather than about TG. It is the documented
    off-switch, kept so the old corpus stays reproducible at this sha; it draws
    nothing from the RNG either way, so the two corpora have identical facts and
    gaps and differ only in the appended token."""

    def __post_init__(self) -> None:
        if self.max_gap >= self.sentences_per_document:
            raise ValueError(
                f"max_gap={self.max_gap} must be < sentences_per_document="
                f"{self.sentences_per_document}. Section 5.2: gaps longer than S do "
                f"not exist in the data, so a generator that cannot fit its own "
                f"longest gap silently produces a shorter-tailed corpus."
            )
        if not 0.0 < self.geometric_p <= 1.0:
            raise ValueError(f"geometric_p={self.geometric_p} must be in (0, 1]")
        if not 0.0 <= self.heavy_tail_weight <= 1.0:
            raise ValueError(f"heavy_tail_weight={self.heavy_tail_weight} outside [0, 1]")
        if not 1 <= self.heavy_tail_min <= self.max_gap:
            raise ValueError(
                f"heavy_tail_min={self.heavy_tail_min} must be in [1, {self.max_gap}]"
            )


@dataclass(frozen=True)
class Sentence:
    index: int
    kind: SentenceKind
    text: str
    fact_id: int | None = None
    """The fact this sentence asserts (`kind="assert"`) or queries
    (`kind="query"`). `None` for filler."""

    answer: str | None = None
    """The queried fact's object, as asserted. With `answer_in_stream` (S0-03) it
    is ALSO the query's final token, as `answer_symbol(answer)`; before S0-03 it
    existed only here, out of band, and never entered the token stream."""


@dataclass(frozen=True)
class Document:
    doc_id: int
    sentences: tuple[Sentence, ...] = field(default_factory=tuple)
    pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    """`(assert_index, query_index)` per fact, in assert order. The ground truth."""

    @property
    def gaps(self) -> tuple[int, ...]:
        return tuple(q - a for a, q in self.pairs)


def _draw_gap(rng: random.Random, cfg: SyntheticConfig) -> int:
    """Mixture of short-range geometric and a heavy tail, truncated at `max_gap`.

    Rejection-free: the geometric draw is clamped, and the tail is uniform over
    `[heavy_tail_min, max_gap]`. Rejection sampling would make the number of draws
    data-dependent, which is fine for determinism but makes the stream position
    depend on the config -- and that is how two "same seed" corpora diverge.
    """
    if rng.random() < cfg.heavy_tail_weight:
        return rng.randint(cfg.heavy_tail_min, cfg.max_gap)
    k = 1
    while k < cfg.max_gap and rng.random() > cfg.geometric_p:
        k += 1
    return k


def _generate_document(doc_id: int, cfg: SyntheticConfig) -> Document:
    # Seeded per document, so document i does not depend on documents 0..i-1.
    rng = random.Random(
        (cfg.seed, doc_id).__hash__() if False else cfg.seed * 1_000_003 + doc_id
    )
    n = cfg.sentences_per_document

    kinds: list[SentenceKind] = ["filler"] * n
    fact_of: list[int | None] = [None] * n
    answers: list[str | None] = [None] * n
    texts: list[str] = [""] * n
    pairs: list[tuple[int, int]] = []

    fact_id = 0
    for i in range(n):
        if kinds[i] != "filler":
            continue
        gap = _draw_gap(rng, cfg)
        j = i + gap
        if j >= n or kinds[j] != "filler":
            continue  # no room, or the query slot is taken; leave i as filler
        entity = _ENTITIES[rng.randrange(len(_ENTITIES))]
        predicate = _PREDICATES[rng.randrange(len(_PREDICATES))]
        obj = _OBJECTS[rng.randrange(len(_OBJECTS))]

        kinds[i], fact_of[i] = "assert", fact_id
        texts[i] = f"{entity} {predicate} {obj}."
        kinds[j], fact_of[j], answers[j] = "query", fact_id, obj
        texts[j] = f"What does {entity} {predicate}?"
        if cfg.answer_in_stream:
            # S0-03: the answer as the query's last token. No RNG draw, so the
            # corpus with and without it has the same facts, gaps and fillers.
            texts[j] += f" {answer_symbol(obj)}"
        pairs.append((i, j))
        fact_id += 1

    for i in range(n):
        if kinds[i] == "filler":
            texts[i] = _FILLERS[rng.randrange(len(_FILLERS))]

    sentences = tuple(
        Sentence(
            index=i, kind=kinds[i], text=texts[i], fact_id=fact_of[i], answer=answers[i]
        )
        for i in range(n)
    )
    return Document(doc_id=doc_id, sentences=sentences, pairs=tuple(pairs))


def generate(cfg: SyntheticConfig | None = None) -> tuple[Document, ...]:
    """The corpus. A pure function of `cfg` -- no global RNG, no ambient state."""
    cfg = cfg or SyntheticConfig()
    return tuple(_generate_document(i, cfg) for i in range(cfg.n_documents))


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def true_demand(doc: Document) -> list[list[float]]:
    """`[T][i]` -- 1.0 if the sentence at step `T` retrieves the sentence at `i`.

    Keyed by **sentence index**, not slot index. A slot holds one sentence's
    gestalt, so the mapping is the memory's business and depends on the policy;
    keeping the ground truth in sentence space is what lets every arm be scored
    against the same truth.

    This is the `r_i(t)` the model's attention is *supposed* to produce. Section
    3.2.1's measured `r_i` is an estimate of it, and E0d asks how good an estimate
    -- against LOO, which is truth where the two disagree.
    """
    n = len(doc.sentences)
    out = [[0.0] * n for _ in range(n)]
    for assert_at, query_at in doc.pairs:
        out[query_at][assert_at] = 1.0
    return out


def fraction_of_pairs_beyond(docs: tuple[Document, ...], m: int) -> float:
    """Fraction of assert->query pairs whose gap exceeds `m` (S0-03 item 4).

    Under FIFO with `m` slots and a write on every sentence, the assert of a pair
    is still in memory at its query iff `gap <= m` (it was written `gap` steps
    earlier and the last `m` writes survive). So this is the fraction of answers
    FIFO **cannot** retrieve -- the population on which eviction bites, and the
    reason answer-token loss is bucketed by gap rather than pooled.
    """
    gaps = [g for d in docs for g in d.gaps]
    if not gaps:
        raise ValueError("no assert->query pairs: the fraction is undefined")
    return sum(g > m for g in gaps) / len(gaps)


def discounted_demand(doc: Document, gamma: float) -> list[list[float]]:
    """`[t][i]` -- `sum_{k>=0} gamma^k * true_demand[t+k][i]`, the MC return.

    **This is what `psi_hat` is regressing onto**, computed exactly rather than
    sampled. Finite streams, so the sum terminates; no bootstrap, no shadow buffer,
    no censoring -- which is precisely why the synthetic corpus is diagnostic and
    why E1 on it is a kill gate rather than a result (defect D-9).

    At `gamma = 0.97` the horizon 1/(1-gamma) = 33 is shorter than `max_gap = 40`,
    so the longest gaps enter at ~0.30 weight. **Attenuated, not invisible** --
    state it in the writeup rather than discovering it as a null.
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma={gamma} is outside [0, 1)")
    r = true_demand(doc)
    n = len(r)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        acc = 0.0
        for t in range(n - 1, -1, -1):  # backwards: G(t) = r(t) + gamma * G(t+1)
            acc = r[t][i] + gamma * acc
            out[t][i] = acc
    return out


# --------------------------------------------------------------------------- #
# Canonical bytes
# --------------------------------------------------------------------------- #


def to_bytes(docs: tuple[Document, ...]) -> bytes:
    """The canonical encoding, for the byte-identity check.

    `sort_keys=True` and an explicit separator, so neither dict insertion order nor
    a json default can move a byte. UTF-8 with `ensure_ascii=True`, so the bytes do
    not depend on the locale.
    """
    payload = [asdict(d) for d in docs]
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
