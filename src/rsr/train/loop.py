"""Minimal training entry point.

What was measured in E0c was **forward and backward only** -- no optimizer step, no
optimizer
state, no data path. This is the loop that closes those three gaps, and it is deliberately
the
smallest thing that is honestly a training run:

  synthetic corpus -> run_policy_loop (retained graph across the stream) -> loss ->
  backward
  -> muP-grouped AdamW step -> heartbeat -> periodic atomic checkpoint

Three things it refuses to do, each because the alternative silently invalidates a result:

* **It does not invent constants.** ``beta`` and ``nu`` come from the registry, which
RAISES when
  they have no logged value (§4.5). A training run that supplied its own would be D-1's
  defect --
  a frozen unmeasured constant governing a mechanism -- and the registry exists to stop
  exactly it.
* **It does not build a plain optimizer.** Parameter groups come from
``build_param_groups``, so
  the value head lands in its own muP group with the ``1/d`` multiplier (§4.3). A single
  flat
  ``AdamW(model.parameters())`` would break width transfer and fail silently in week 9.
* **It does not hide a crash.** The heartbeat records the traceback before the exception
  propagates, so a run that dies at 03:00 is distinguishable in the morning from one that
  never
  started.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rsr.baselines.fifo import FIFOPolicy
from rsr.data.synthetic import (
    SyntheticConfig,
    _generate_document,
    answer_symbol,
    generate,
)
from rsr.exit_codes import ArgumentParser, Exit, run_main, status
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.mup.param_groups import build_param_groups
from rsr.retention.rsr import RSRConfig, RSRPolicy
from rsr.train import checkpoint as ck
from rsr.train.heartbeat import Heartbeat

__all__ = [
    "DocumentPool",
    "DocumentStream",
    "WarmStart",
    "WarmStartRefused",
    "answer_targets",
    "build_policy",
    "check_vocabulary_closure",
    "liveness_batch",
    "liveness_exit",
    "main",
    "stream_documents",
    "train",
    "verify_warm_start",
]


def _sha() -> str:
    """Stamped from git INSIDE this tree. Never accepted as an argument -- a caller-typed
    sha is
    a provenance claim, not provenance."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config_hash(payload: dict) -> str:
    """Freeze the config before the run. The heartbeat carries this, so a config changed
    mid-flight
    cannot be reported against the frozen one."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_vocab(docs) -> dict[str, int]:
    """Word-level vocabulary over the synthetic corpus, deterministic by sort order.

    Ids 0-3 are reserved for the special tokens, because **TG reads the gestalt at the
    first
    [EOS] and writes memory only when a sentence has one.** A vocabulary that cannot
    express the
    EOS id produces `has_eos = False` on every sentence, so nothing is ever written,
    cross-attention
    returns exactly zero, and the model still emits a plausible loss. `config.py:190`
    names this
    trap; E0C's first measurement fell into it and so did the first version of this file.
    """
    words = sorted({w for d in docs for sent in d.sentences for w in sent.text.split()})
    return {w: i + 4 for i, w in enumerate(words)}  # 0=pad 1=bos 2=eos 3=eod


def encode(docs, vocab: dict[str, int], *, max_tokens: int, steps: int):
    """-> ids (n_docs, steps, L) with an explicit EOS per sentence, and a mask.

    Layout per sentence: ``[w0 w1 ... wk, EOS, pad ...]``. The EOS is written even when
    the
    sentence is truncated, because losing it silently disables the memory path.
    """
    n = len(docs)
    L = max_tokens
    ids = torch.zeros(n, steps, L, dtype=torch.long)
    mask = torch.zeros(n, steps, L, dtype=torch.bool)
    for di, doc in enumerate(docs):
        for si, sent in enumerate(doc.sentences[:steps]):
            toks = [vocab[w] for w in sent.text.split()][: L - 1]
            toks.append(2)  # EOS -- never optional
            ids[di, si, : len(toks)] = torch.tensor(toks)
            mask[di, si, : len(toks)] = True
    return ids, mask


def answer_targets(docs, vocab: dict[str, int], *, max_tokens: int, steps: int):
    """-> `target_mask` (n_docs, steps, L) bool and `gap` (n_docs, steps) long (S0-03).

    `target_mask` is a **supervision** mask, in the same token frame as `encode()`'s
    `ids` and `mask`: True at exactly one position per query sentence, the answer
    token. Score it the way `mask` is scored -- `target_mask[..., 1:]` against the
    shifted targets of `lm_token_losses`. `encode()`'s mask is a *padding* mask;
    pooled into it, the answer targets are ~7% of real targets on this corpus and
    their loss cannot be read off the total.

    `gap[d, s]` is `query_index - assert_index` for a query sentence and 0
    elsewhere, so answer-token loss can be bucketed by gap. Bucketing is what makes
    this an eviction test rather than a memorization test: under FIFO with `M`
    slots the assert is in memory iff `gap <= M`.

    🔴 **Raises if a query does not carry its answer as its final token.** An empty
    mask would score as "no answer targets" and every answer-token loss downstream
    would be a mean over nothing -- the pre-S0-03 corpus, where the answer lived
    only in `Sentence.answer`, would pass through silently.
    """
    n, L = len(docs), max_tokens
    target_mask = torch.zeros(n, steps, L, dtype=torch.bool)
    gap = torch.zeros(n, steps, dtype=torch.long)
    for di, doc in enumerate(docs):
        assert_of = {q: a for a, q in doc.pairs}
        for si, sent in enumerate(doc.sentences[:steps]):
            if sent.kind != "query":
                continue
            words = sent.text.split()
            sym = answer_symbol(sent.answer) if sent.answer is not None else None
            if sym is None or not words or words[-1] != sym:
                raise ValueError(
                    f"doc {doc.doc_id} sentence {si}: query {sent.text!r} does not "
                    f"end in its answer {sym!r}. The answer is out of band, so no "
                    f"next-token target requires retrieval (S0-03); generate with "
                    f"SyntheticConfig(answer_in_stream=True)."
                )
            pos = len(words) - 1
            if pos >= L - 1:  # encode() truncates to L - 1 words before the EOS
                raise ValueError(
                    f"doc {doc.doc_id} sentence {si}: the answer at word {pos} is "
                    f"truncated away by max_tokens={L}"
                )
            if vocab.get(words[-1]) is None:
                raise ValueError(f"answer token {words[-1]!r} is not in the vocab")
            target_mask[di, si, pos] = True
            gap[di, si] = si - assert_of[si]
    return target_mask, gap


@dataclass(frozen=True)
class DocumentStream:
    """A deterministic, non-repeating document stream (experiments/fresh-stream).

    Step ``t`` trains on documents ``[offset + stride * t, offset + stride * t +
    batch)`` of the seed's generator, each generated **by id** with
    `rsr.data.synthetic._generate_document` -- prefix-stable (per-document seed
    ``seed * 1_000_003 + doc_id``), so a document's bytes do not depend on which
    others were generated, and nothing is held beyond one batch. Documents are
    encoded with the vocabulary built from documents ``[0, vocab_documents)``,
    the fixed corpus's own, so ``V`` and every token id are the default path's.

    It replaces ONLY the with-replacement sampler (``torch.randint`` in
    `train()`), and draws nothing from any RNG. ``stride >= batch`` is required:
    overlapping windows would repeat documents, which is the thing a stream
    exists to exclude.
    """

    offset: int
    stride: int
    vocab_documents: int = 64

    def __post_init__(self) -> None:
        if self.offset < 0 or self.stride < 1 or self.vocab_documents < 1:
            raise ValueError(f"{self}: offset >= 0, stride >= 1, vocab_documents >= 1")

    def window(self, step: int, batch: int) -> range:
        """The document ids trained on at ``step``."""
        if batch > self.stride:
            raise ValueError(
                f"batch {batch} > stride {self.stride}: consecutive windows would "
                f"overlap and repeat documents"
            )
        start = self.offset + self.stride * step
        return range(start, start + batch)


def stream_documents(stream: DocumentStream, step: int, batch: int, cfg):
    """The documents of ``stream`` at ``step``, generated by id under ``cfg``
    (its seed and generator knobs; ``cfg.n_documents`` is not read)."""
    return tuple(_generate_document(i, cfg) for i in stream.window(step, batch))


@dataclass(frozen=True)
class DocumentPool:
    """A fixed training pool of documents ``[offset, offset + length)`` of the
    seed's generator (m-2 of the feat/rsr-slice1 review; run 1's PREREG names "a
    fixed pool of 256 documents arm B never saw").

    Each document is generated **by id** with `_generate_document` (prefix-stable,
    as `DocumentStream`), and encoded with the vocabulary of documents
    ``[0, vocab_documents)`` -- the fixed corpus's, so ``V`` and every token id
    are the ones the warm-start substrate was trained with. The closure (every
    pool word in that vocabulary) is checked before any step.

    The sampler is today's: ``torch.randint`` over the pool, **with replacement**
    (m-3). One epoch (31(b)) is ``length / batch`` optimizer steps. The pool's
    identity is stamped into ``frozen``, so two pools cannot share a config hash.
    """

    offset: int
    length: int
    vocab_documents: int = 64

    def __post_init__(self) -> None:
        if self.offset < 0 or self.length < 1 or self.vocab_documents < 1:
            raise ValueError(f"{self}: offset >= 0, length >= 1, vocab_documents >= 1")

    @classmethod
    def from_range(cls, start: int, stop: int, vocab_documents: int = 64):
        """The explicit id range ``[start, stop)``."""
        return cls(offset=start, length=stop - start, vocab_documents=vocab_documents)

    @property
    def ids(self) -> range:
        return range(self.offset, self.offset + self.length)


def pool_documents(pool: DocumentPool, cfg) -> tuple:
    """The pool's documents, generated by id under ``cfg`` (seed and knobs;
    ``cfg.n_documents`` is not read)."""
    return tuple(_generate_document(i, cfg) for i in pool.ids)


class WarmStartRefused(ValueError):
    """A warm-start checkpoint that is not provably the one the PREREG names."""


@dataclass(frozen=True)
class WarmStart:
    """A **weights-only** warm start (m-1; `R-2026-09-30-rsr-run-1-scope` D2(a):
    run 1 resumes from fresh-stream arm B ``ckpt-003000.pt``, seed 0).

    ``checkpoint`` is read only; ``manifest`` is the T0 backup manifest
    (``shasum -a 256`` lines, paths relative to ``root``); ``root`` is the
    directory those paths resolve against -- the repo root for the originals,
    the backup root for the copies. A checkpoint whose sha256 is not the
    manifest's, or that the manifest does not list, is refused; so is one whose
    sha256 is not ``expected_sha256`` when that is given (the CLI requires it).

    **The source's config** -- its seed and the inputs of its vocabulary -- is
    read from the checkpoint's ``meta["config"]`` (every checkpoint `train()`
    writes from feat/rsr-run1-ready on carries it), or else from
    ``source_config``, a JSON file holding the source run's frozen config (a
    `train()` result, e.g. arm B's ``result.json``). Either way it is accepted
    only if its config hash equals the hash in the checkpoint's own sha-verified
    ``meta["run_id"]`` -- so the config is bound to the bytes, not trusted.

    Two refusals, each lifted only by a **labelled** flag that enters the
    frozen config, the hash and the label (review MAJOR-1 and the seed ruling):

    * the source's token -> id map must EQUAL this run's (compared as a sha256
      of the map, not as a document count); a source with no recorded config
      is refused unless ``allow_unrecorded_vocabulary`` (label
      ``vocab=unchecked``);
    * the source's seed must equal this run's seed; a different or unrecorded
      seed is refused unless ``allow_seed_mismatch`` (label
      ``seed_mismatch=allowed``).

    What it loads: the transformer and ``W_sent`` (the model ``state_dict``),
    nothing else. Not the optimizer (fresh groups), not the RNG (the run's own
    seed), not the policy (``phi`` is built fresh from its own generator, L11),
    not the step: ``k`` counts from 0 at the warm start, so ``T_warm`` (§3.4,
    correction 31) runs at relative steps ``0 .. T_warm - 1``. The bytes loaded
    are the bytes hashed: read once, hashed, loaded from memory (MINOR-5).
    """

    checkpoint: str
    manifest: str
    root: str
    expected_sha256: str | None = None
    source_config: str | None = None
    allow_unrecorded_vocabulary: bool = False
    allow_seed_mismatch: bool = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vocab_sha256(vocab: Mapping[str, int]) -> str:
    """sha256 of a token -> id map, order-free: what "same vocabulary" means."""
    blob = json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _verified_bytes(ws: WarmStart) -> tuple[dict[str, Any], bytes]:
    """Read the checkpoint ONCE; hash that buffer; check it against the manifest
    (and ``expected_sha256``). Returns the provenance record and the bytes."""
    ckpt = Path(ws.checkpoint).expanduser().resolve()
    manifest = Path(ws.manifest).expanduser().resolve()
    root = Path(ws.root).expanduser().resolve()
    if not ckpt.is_file():
        raise WarmStartRefused(f"warm start: no checkpoint at {ckpt}")
    if not manifest.is_file():
        raise WarmStartRefused(f"warm start: no manifest at {manifest}")
    try:
        rel = ckpt.relative_to(root).as_posix()
    except ValueError:
        raise WarmStartRefused(
            f"warm start: {ckpt} is not under the manifest root {root}"
        ) from None
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, path = line.partition("  ")
        listed[path.strip()] = digest.strip()
    want = listed.get(rel)
    if want is None:
        raise WarmStartRefused(
            f"warm start: {rel} is not in the manifest {manifest} "
            f"({len(listed)} entries); only a T0-listed checkpoint may seed a run"
        )
    data = ckpt.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise WarmStartRefused(
            f"warm start: {rel} has sha256 {got}, the manifest {manifest} lists "
            f"{want}. Refused: this is not the checkpoint the manifest froze."
        )
    if ws.expected_sha256 is not None and got != ws.expected_sha256.lower():
        raise WarmStartRefused(
            f"warm start: {rel} has sha256 {got}, the caller expected "
            f"{ws.expected_sha256}. Refused: not the checkpoint the command names."
        )
    rec = {
        "checkpoint": str(ckpt),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "manifest_entry": rel,
        "sha256": got,
        "expected_sha256": ws.expected_sha256,
    }
    return rec, data


def verify_warm_start(ws: WarmStart) -> dict[str, Any]:
    """The checkpoint's provenance, or `WarmStartRefused`. Reads the checkpoint
    and the manifest, writes nothing. The checkpoint must sit under ``root``, be
    listed in the manifest at its path relative to ``root``, and hash to exactly
    the listed sha256 (and to ``expected_sha256`` if given)."""
    return _verified_bytes(ws)[0]


def _config_vocab_documents(config: Mapping[str, Any]) -> int:
    """How many documents a run's vocabulary was built from, from its frozen
    config -- the same precedence `train()` uses (pool, stream, n_documents,
    SyntheticConfig's default)."""
    if "pool" in config:
        return int(config["pool"]["vocab_documents"])
    if "stream" in config:
        return int(config["stream"]["vocab_documents"])
    if "n_documents" in config:
        return int(config["n_documents"])
    return SyntheticConfig().n_documents


def _source_config(payload: Mapping[str, Any], ws: WarmStart) -> tuple[dict, str] | None:
    """The source run's frozen config, bound to the checkpoint by its hash, and
    where it came from; None if neither the checkpoint nor ``source_config``
    records one. A config whose hash is not the run_id's is REFUSED, not
    treated as unrecorded."""
    meta = payload.get("meta") or {}
    run_id = str(meta.get("run_id") or "")
    want = run_id.rsplit("-", 1)[-1] if run_id else None
    found = None
    if isinstance(meta.get("config"), Mapping):
        found = (dict(meta["config"]), "checkpoint meta")
    elif ws.source_config is not None:
        path = Path(ws.source_config).expanduser().resolve()
        doc = json.loads(path.read_text())
        cfg = doc.get("config", doc) if isinstance(doc, dict) else None
        if not isinstance(cfg, dict):
            raise WarmStartRefused(f"warm start: {path} holds no config dict")
        found = (cfg, f"{path} (sha256 {_sha256_file(path)})")
    if found is None:
        return None
    cfg, origin = found
    if want is None or _config_hash(cfg) != want:
        raise WarmStartRefused(
            f"warm start: the source config from {origin} hashes to "
            f"{_config_hash(cfg)}, but the checkpoint's run_id {run_id!r} names "
            f"{want!r}. Refused: that config is not the run that wrote these bytes."
        )
    return cfg, origin


def _load_warm_start(
    model: TGModel, ws: WarmStart, *, vocab_map: Mapping[str, int], seed: int
) -> dict:
    """Verify, then load ONLY the model weights (strict: every tensor, every
    shape) from the very bytes that were hashed. Returns the record stamped into
    ``frozen``. Draws no RNG and restores none."""
    rec, data = _verified_bytes(ws)
    # Read with nothing to restore into; checked; THEN the weights are copied.
    payload = ck.load_bytes(data, source=rec["checkpoint"])
    source_policy = (payload.get("policy") or {}).get("__policy_name__")
    if source_policy != "fifo":
        raise WarmStartRefused(
            f"warm start: the source was written by policy {source_policy!r}; run 1's "
            "substrate is a FIFO checkpoint (R-2026-09-30-rsr-run-1-scope D2(a)), and "
            "any other policy's state would be silently dropped here"
        )
    run_vocab = vocab_sha256(vocab_map)
    found = _source_config(payload, ws)
    src_cfg, origin = found if found is not None else (None, None)
    src_vocab = src_seed = None
    if src_cfg is not None:
        src_seed = int(src_cfg["seed"])
        recorded = (payload.get("meta") or {}).get("vocab_sha256")
        if recorded is not None:
            src_vocab = str(recorded)
        else:
            # Legacy source: rebuild its map from its own (hash-bound) config.
            # Assumes this tree's generator; the checkpoint's sha pins the bytes.
            src_docs = generate(
                SyntheticConfig(
                    sentences_per_document=int(src_cfg["steps_per_stream"]),
                    seed=src_seed,
                    n_documents=_config_vocab_documents(src_cfg),
                )
            )
            src_vocab = vocab_sha256(build_vocab(src_docs))
        if src_vocab != run_vocab:
            raise WarmStartRefused(
                f"warm start: the source's token->id map (sha256 {src_vocab[:12]}) "
                f"is not this run's ({run_vocab[:12]}). Token ids would mean "
                "different words."
            )
    elif not ws.allow_unrecorded_vocabulary:
        raise WarmStartRefused(
            "warm start: the source records no config, so its vocabulary cannot be "
            "compared with this run's. Pass source_config= (a JSON frozen config "
            "whose hash is the checkpoint's run_id), or set the labelled flag "
            "allow_unrecorded_vocabulary=True."
        )
    if src_seed != seed and not ws.allow_seed_mismatch:
        raise WarmStartRefused(
            f"warm start: the source's seed is {src_seed!r} and this run's is {seed}. "
            "'Documents the source never saw' is a statement about the source's "
            "seed. Refused unless the labelled flag allow_seed_mismatch=True."
        )
    model.load_state_dict(payload["model"])  # strict: every tensor, every shape
    rec.update(
        source_step=int(payload.get("step", 0)),
        source_run_id=(payload.get("meta") or {}).get("run_id"),
        source_policy=source_policy,
        source_seed=src_seed,
        source_config_origin=origin,
        source_vocab_sha256=src_vocab,
        run_vocab_sha256=run_vocab,
        vocab_checked=src_vocab is not None,
        seed_matches=src_seed == seed,
        allow_unrecorded_vocabulary=ws.allow_unrecorded_vocabulary,
        allow_seed_mismatch=ws.allow_seed_mismatch,
        loaded="model state_dict only (transformer + W_sent), from the hashed "
        "bytes; optimizer, RNG, policy and step NOT loaded",
        train_step_origin="k = 0 at the warm start (§3.4 T_warm counts from here)",
    )
    return rec


def check_vocabulary_closure(docs, vocab: dict[str, int]) -> None:
    """Raise ``ValueError`` naming the first document and word outside ``vocab``.

    ``encode()`` would raise a bare ``KeyError`` mid-run; the fresh-stream PREREG
    requires the closure to be proved before any step, and this is that proof's
    one definition, used by both `train()` and the experiment."""
    for doc in docs:
        for sent in doc.sentences:
            for w in sent.text.split():
                if w not in vocab:
                    raise ValueError(
                        f"document {doc.doc_id}: word {w!r} is not in the vocabulary"
                    )


def lm_token_losses(logits, ids_t):
    """Per-target next-token cross-entropy, flat, one sentence step.

    The `[:, :-1]` / `[:, 1:]` shift means the population is **targets**, not
    tokens: `batch x (L - 1)` of them.
    """
    lg = logits
    return F.cross_entropy(
        lg[:, :-1].reshape(-1, lg.shape[-1]),
        ids_t[:, 1:].reshape(-1),
        reduction="none",
    )


def lm_loss(logits, ids_t, mask_t):
    """Next-token cross-entropy over **real** targets only (cycle 1, defect 1).

    The version this replaces reduced with `reduction="mean"` over every target,
    including PAD -- while `mask_t` was already being handed to `step_fn` by
    `rsr.model.tg.policy_loop:140` and already going unused. On the committed
    synthetic corpus that is not a correction at the margin: sentences are ~5
    tokens inside a 64-token frame, so the number being minimised, reported and
    turned into a perplexity was overwhelmingly the model's skill at predicting
    zeros. `tests/test_train_loss.py` measures the fraction rather than quoting it.

    Selection is on `mask_t` rather than on `ids_t != pad_id` because the mask is
    what `encode()` built and what the loop already passes; the two agree here and
    a test says so. **A step with no real target returns 0, not `nan`** -- `0 / 0`
    would poison the accumulated stream loss for any document shorter than
    `steps_per_stream`.
    """
    per = lm_token_losses(logits, ids_t)
    valid = mask_t[:, 1:].reshape(-1)
    n = valid.sum()
    return (per * valid.to(per.dtype)).sum() / n.clamp(min=1)


def lm_loss_unmasked(logits, ids_t, mask_t):
    """🔴 The pre-cycle-1 objective, scoring PAD. **Never the default.**

    It survives behind `train(masked_loss=False)` for exactly one reason: the
    masked and unmasked arms have to be comparable *at one sha*, and an arm that
    only exists at the parent commit is an arm whose ledger cites a different tree.
    It is a documented off-switch in the sense CLAUDE.md means, not an option.
    """
    return lm_token_losses(logits, ids_t).mean()


POLICIES = ("fifo", "rsr")
"""The names `--policy` accepts. H2O and LRU (section 10.1's real E7 controls) are
not here because they are not implemented; listing a name that silently fell through
to FIFO is the defect this tuple exists to prevent."""

#: The AdamW `eps` every run uses: **pinned at 1e-8** (P-2 of the feat/rsr-slice1
#: review). B1 measured beta LIVE at eps = 1e-8 and inert only at 1e-12
#: (`experiments/b1-beta-inertness/RESULTS.md`), so eps is part of what beta means:
#: it is pinned to B1's measured regime rather than inherited from whatever torch
#: defaults to, and every run records the value it used beside torch's default.
ADAMW_EPS: float = 1e-8
#: torch's own AdamW default, read from its signature -- recorded, never used.
#: Equal to ADAMW_EPS at torch 2.14.0; if a torch upgrade moves it, the record
#: shows the two disagree and nothing silently follows.
TORCH_DEFAULT_ADAMW_EPS: float = (
    inspect.signature(torch.optim.AdamW).parameters["eps"].default
)
ADAMW_BETAS = (0.9, 0.95)
ADAMW_WEIGHT_DECAY = 0.01
"""Decoupled (AdamW) decay, one value for every group -- `phi`'s `value_head` group
included, which is B1's measured `decoupled_wd` arm (PREREG row "weight decay")."""


def optimizer_record() -> dict:
    """What the optimizer is, as data for the frozen config and the header."""
    return {
        "type": "AdamW",
        "betas": list(ADAMW_BETAS),
        "weight_decay": ADAMW_WEIGHT_DECAY,
        "weight_decay_kind": "decoupled",
        "eps": ADAMW_EPS,
        "eps_source": "pinned 1e-8: B1's measured regime (P-2)",
        "torch_default_eps": TORCH_DEFAULT_ADAMW_EPS,
        "transformer_clip": 1.0,
        "phi_clip": None,
        "phi_in_transformer_clip": False,
    }


def epoch_steps(pool_documents: int | None, batch: int) -> float | None:
    """Optimizer steps per epoch: **one pass over the document pool**.

    `docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md` (31(b)): "one epoch" is one pass
    over the document pool, i.e. `n_docs / batch` optimizer steps -- 256 / 16 = 16
    in the ruling's example. Derived from the pool `train()` actually built and the
    batch it actually runs, never typed; §3.4's `T_warm` (one epoch) follows from
    it through the registry (D-I).

    A never-repeating `DocumentStream` has no pool, so it has no epoch under this
    definition: None, and an RSR arm refuses it (the ruling does not define it).
    """
    if pool_documents is None:
        return None
    if pool_documents <= 0 or batch <= 0:
        raise ValueError(f"pool_documents={pool_documents}, batch={batch}: both > 0")
    return pool_documents / batch


#: Every RSR step's §7.1 diagnostics, one JSON line per optimizer step (MINOR-3).
DIAGNOSTICS_FILE = "diagnostics.jsonl"


def _write_step_diagnostics(path: Path, record: dict) -> None:
    """Append one step's §7.1 diagnostics as a JSON line (MINOR-3)."""
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def phi_generator(seed: int) -> tuple[torch.Generator, int]:
    """ADR-0009 L11: `phi`'s own **CPU** generator, seeded from `(seed, "phi")`.

    Two latent defects it removes (ADR-0009 Context): `train()` built `phi` from
    the data-order generator `gen`, so an RSR arm and a FIFO arm at the same seed
    saw different streams (§3.7's warning); and on MPS `gen` is an MPS generator,
    from which the CPU-built head's `normal_` raises. The head is built on CPU from
    this generator, then moved.
    """
    digest = hashlib.sha256(f"{seed}:phi".encode()).digest()
    phi_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return torch.Generator(device="cpu").manual_seed(phi_seed), phi_seed


_LABEL_NAMES = {"b_enabled": "b", "shadow_enabled": "shadow", "psi_override": "psi"}


def _label_value(v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if v is None:
        return "learned"
    if isinstance(v, float):
        return format(v, "g")
    return str(v)


def arm_label(
    policy_name: str,
    overrides: Mapping[str, Any] | None = None,
    *,
    freeze_transformer: bool = False,
    warm_start: Mapping[str, Any] | None = None,
) -> str:
    """The arm's label, derived from its overrides (ADR-0009 L12).

    `rsr[b=off,beta=0.1,gamma=0.9,nu=0,shadow=off]`. Every override is a visible
    departure, so it is in the name; two arms that differ in one cannot share a
    label, and (stamped into `frozen`) cannot share a config hash either. The
    frozen-transformer option (R-2026-09-30-rsr-run-1-scope D2(b)) is appended,
    then the warm start's source (m-1):
    `+warm[ckpt-003000.pt@0ee3f8a69b50,src_seed=0]` -- the checkpoint's file
    name, the first 12 hex of its sha256 and its seed, plus `vocab=unchecked` /
    `seed_mismatch=allowed` when a labelled allow flag was needed.
    """
    toks = [
        f"{_LABEL_NAMES.get(k, k)}={_label_value(v)}"
        for k, v in sorted((overrides or {}).items())
    ]
    label = f"{policy_name}[{','.join(toks)}]" if toks else policy_name
    label = f"{label}+frozen" if freeze_transformer else label
    if warm_start is not None:
        name = Path(warm_start["checkpoint"]).name
        seed = warm_start.get("source_seed")
        toks = [
            f"{name}@{warm_start['sha256'][:12]}",
            f"src_seed={'unrecorded' if seed is None else seed}",
        ]
        if not warm_start.get("vocab_checked", True):
            toks.append("vocab=unchecked")
        if not warm_start.get("seed_matches", True):
            toks.append("seed_mismatch=allowed")
        label = f"{label}+warm[{','.join(toks)}]"
    return label


def _check_overrides(overrides: Mapping[str, Any], no_warmup_ablation: bool) -> None:
    """ADR-0009 L12's structural rules, before any registry read."""
    known = {f.name for f in fields(RSRConfig)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(
            f"unknown RSR override(s) {unknown}; overridable fields are {sorted(known)}"
        )
    if "t_warm" in overrides and not no_warmup_ablation:
        # L12: otherwise an override decides correction 31(b) by the back door --
        # the owner ruled what an epoch is; T_warm follows from it (§3.4).
        raise ValueError(
            "a t_warm override is refused unless the arm is declared the no-warmup "
            "ablation (ADR-0009 L12): T_warm is one epoch (§3.4), and what an epoch "
            "is was ruled by the owner (R-2026-09-30-rsr-run-1-scope, 31(b))"
        )
    if no_warmup_ablation and overrides.get("t_warm") != 0.0:
        raise ValueError(
            "no_warmup_ablation=True declares the T_warm = 0 arm: pass t_warm=0.0 "
            "explicitly. A declaration with any other value -- or none -- would be "
            "a label that says something the arm does not do."
        )


def _jsonable(v: Any) -> Any:
    return v.value if isinstance(v, Enum) else v


def _grad_norm(params: list[torch.nn.Parameter]) -> float | None:
    """L2 norm over the gradients that exist; None if none does (beta = 0, L13)."""
    grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
    if not grads:
        return None
    return float(torch.cat(grads).norm())


def build_policy(
    name: str,
    *,
    d_model: int,
    steps_per_epoch: float | None,
    scope: str = "synthetic",
    generator: torch.Generator | None = None,
    overrides: Mapping[str, Any] | None = None,
    no_warmup_ablation: bool = False,
):
    """Map `policy_name` to the policy that will actually run (S0-01 defect (b)).

    🔴 **The name is stamped into the frozen config and the `run_id`.** Before this
    existed, `train()` constructed `FIFOPolicy()` unconditionally while
    `policy_name` flowed straight into both -- so `train(policy_name="rsr")`
    produced a run directory, a heartbeat and a `run_id` all reading `rsr-...` over
    a stream in which FIFO evicted every slot. Nothing in the output disagreed with
    anything else, which is what makes it silent rather than merely wrong.

    Routing both through one function is the fix: there is now no path on which the
    stamped name and the constructed policy can differ.

    **`"rsr"` with no overrides is refused on today's tree, and that is correct.**
    `nu`, `beta` and `gamma` are MEASURED and E1 has not run (§4.5), so the
    registry raises `UnmeasuredConstant`. The registry refusing the read is the D-1
    guard; the answer is E1, or a labelled override an arm's PREREG declares
    (ADR-0009 L12) -- never a default supplied here.

    `steps_per_epoch` feeds `T_warm`'s derivation (optimizer steps per epoch).
    `train()` used to pass `iters`, which made §3.4's "one epoch" the whole run and
    the arm FIFO throughout. **Ruled 2026-09-30**
    (`docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md`, 31(b)): one epoch is
    one pass over the document pool, and `train()`
    now passes `epoch_steps(pool, batch)`. A caller with no epoch still gets None
    refused here, naming correction 31.

    **`overrides` is ADR-0009 L12's path** (signed): read lazily by
    `RSRConfig.from_registry`, so a field not overridden still reads the registry
    and still raises naming E1 -- `nu`, `beta` and `gamma` are MEASURED and owed by
    E1, and run 1 supplies them only as labelled overrides declared in its own
    PREREG. Nothing is recorded in the registry. The rules, each refused loudly:
    `b_enabled=False` is required (no `ProtectionBias` exists, §12 item 3);
    `shadow_enabled=False` must be passed explicitly (L6(a)); a `t_warm` override
    needs `no_warmup_ablation=True` and must be 0.
    """
    if name == "fifo":
        if overrides:
            raise ValueError(
                f"overrides {sorted(overrides)} on the FIFO arm: FIFO has no RSR "
                "config, so they would be stamped into its label and change nothing"
            )
        if no_warmup_ablation:
            # m-5: the branch's own rule -- a switch that is set and does nothing.
            raise ValueError(
                "no_warmup_ablation=True on the FIFO arm: FIFO has no warmup, so the "
                "declaration would change nothing (ADR-0009 L12)"
            )
        return FIFOPolicy()
    if name == "rsr":
        if steps_per_epoch is None:
            raise ValueError(
                "policy 'rsr' needs steps_per_epoch for T_warm (spec §3.4), and what "
                "an epoch is on train()'s loop is an open owner decision "
                "(docs/spec-corrections.md correction 31). train() used to pass "
                "iters, which made the warmup the entire run -- FIFO throughout."
            )
        ov = dict(overrides or {})
        _check_overrides(ov, no_warmup_ablation)
        cfg = RSRConfig.from_registry(scope, steps_per_epoch=steps_per_epoch, **ov)
        if ov.get("b_enabled") is not False:
            raise ValueError(
                "ADR-0009 L12: b_enabled=False is required, and must be passed as an "
                "override: ProtectionBias does not exist (RESEARCH-CONTEXT §12 item "
                "3; E0e's tau and gamma_b are unrecorded). The arm is then partially "
                "reduced toward TG (§3.7's b == 0) and its label says b=off."
            )
        if ov.get("shadow_enabled") is not False:
            raise ValueError(
                "ADR-0009 L6(a)/L12: shadow_enabled=False must be passed explicitly "
                "-- the shadow buffer is slice 2, and the dataclass default is True"
            )
        # ADR-0009 L11: `generator` is phi's own CPU stream (see `phi_generator`),
        # never the data-order generator. ADR-0009 L6: the section 7.1
        # diagnostics are on for every RSR arm built here (they never change a
        # decision; `RSRPolicy`'s default keeps them off on E0b's direct path).
        return RSRPolicy(cfg, d_model, generator=generator, diagnostics=True)
    raise ValueError(
        f"unknown policy_name {name!r}; known policies are {POLICIES}. "
        f"Falling back to FIFO here would stamp {name!r} into the run_id of a FIFO "
        f"run (S0-01 defect (b)), so this refuses instead."
    )


def train(
    *,
    d: int = 128,
    steps_per_stream: int = 48,
    batch: int = 16,
    vocab: int | None = None,  # None -> derive from the corpus
    max_tokens: int = 64,
    memory_slots: int = 16,
    iters: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "mps",
    out_dir: str | Path = "runs/dev",
    beat_every: int = 1,
    ckpt_every: int = 25,
    resume: str | Path | None = None,
    policy_name: str = "fifo",
    masked_loss: bool = True,
    srep_norm_reg_weight: float | None = None,  # None -> TGConfig's 0.01
    n_documents: int | None = None,  # None -> SyntheticConfig's 64, today's path
    stream: DocumentStream | None = None,  # None -> the with-replacement sampler
    policy_overrides: Mapping[str, Any] | None = None,  # ADR-0009 L12
    no_warmup_ablation: bool = False,  # ADR-0009 L12: the only t_warm override
    freeze_transformer: bool = False,  # R-2026-09-30-rsr-run-1-scope D2(b)
    pool: DocumentPool | None = None,  # m-2: None -> documents [0, n_documents)
    warm_start: WarmStart | None = None,  # m-1: weights-only, D2(a)
) -> dict:
    """One training run. See the module docstring.

    `pool` (m-2): train on documents ``[pool.offset, pool.offset + pool.length)``
    of the seed, encoded with the vocabulary of ``[0, pool.vocab_documents)``;
    the epoch (31(b)) is ``pool.length / batch``. Stamped into `frozen`.

    `warm_start` (m-1; `docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md` D2(a)):
    load ONLY the transformer and `W_sent` from a T0-manifest-verified FIFO
    checkpoint. `phi` is fresh from its own generator (L11), the optimizer groups
    are fresh, the RNG is this run's own, and the optimizer step `k` counts from
    0 at the warm start, so §3.4's warmup runs at relative steps
    ``0 .. T_warm - 1``. The source's path and sha256 are stamped into `frozen`,
    the hash and the label. The liveness decoy stays "the untrained model at the
    same seed and config" (this model's own init, copied before the load), so a
    frozen warm-started arm measures the substrate's liveness rather than the
    ``ratio == 1.0`` of a decoy that is the trained model.

    `freeze_transformer=True` (run 1's substrate rule, ruled in
    `docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md` D2(b)): the transformer
    and `W_sent` get **no
    update** -- `requires_grad` off, out of the optimizer, `L_NTP` reported but not
    optimised -- so a FIFO and an RSR arm on the same substrate differ only in the
    eviction rule. The model's train/eval MODE is unchanged (dropout stays as in
    any training run); only the update is removed. Stamped into `frozen`, the hash
    and the label. ADR-0009 T4 is vacuous under it, which is why the isolation
    tests also run non-frozen.
    """
    # fresh-stream (experiments/fresh-stream/PREREG.md): `stream=None` is today's
    # path byte for byte -- nothing below runs differently and nothing is stamped.
    # Set, it replaces only the `torch.randint` sampler; the vocabulary is still
    # built from documents [0, stream.vocab_documents), so the model, V and the
    # init (after the same `torch.manual_seed`) are the fixed corpus's.
    if stream is not None and n_documents is not None:
        raise ValueError(
            "stream replaces the fixed corpus; its vocabulary documents are "
            "stream.vocab_documents -- pass n_documents=None"
        )
    if pool is not None and (stream is not None or n_documents is not None):
        raise ValueError(
            "pool replaces the fixed corpus [0, n_documents) and the stream; its "
            "vocabulary documents are pool.vocab_documents -- pass neither"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    # corpus-size-curve: `n_documents=None` builds the corpus with SyntheticConfig's
    # own default, byte for byte the pre-existing call; an int sets the training-set
    # size (experiments/corpus-size-curve/PREREG.md). The generator is prefix-stable
    # (`src/rsr/data/synthetic.py`, per-document RNG), so documents 0..N-1 are the
    # same at every N.
    corpus_kw = {} if n_documents is None else {"n_documents": n_documents}
    if stream is not None:
        corpus_kw = {"n_documents": stream.vocab_documents}
    if pool is not None:
        corpus_kw = {"n_documents": pool.vocab_documents}  # m-2: vocab docs only
    probe = generate(
        SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed, **corpus_kw)
    )
    V = vocab if vocab else 4 + len(build_vocab(probe))
    cfg = TGConfig(
        D=d,
        V=V,
        F=int(d * 2.6875),
        max_sentence_tokens=max_tokens,
        max_sentences_in_short_term=memory_slots,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).to(device)
    # liveness-wiring: the decoy is "the untrained model at the same seed and the
    # same config" (decisive PREREG *Decoy*) -- this model's own init. Copied, not
    # rebuilt: a second `TGModel(cfg)` would draw from the global RNG and shift
    # every draw after it (CLAUDE.md, the E0b note).
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # m-1: weights-only warm start, AFTER the decoy is copied (so the decoy stays
    # the untrained init) and BEFORE phi, the freeze and the optimizer (so all
    # three are this run's own). Reads the checkpoint; draws and restores no RNG.
    ws_record = None
    if warm_start is not None:
        ws_record = _load_warm_start(
            model, warm_start, vocab_map=build_vocab(probe), seed=seed
        )

    # S0-01 defect (c). `StepOutput.srep_norm_penalty` is computed at `model.py:424`
    # and was discarded: `grep -c srep_norm src/rsr/train/loop.py` returned 0.
    # `docs/spec-corrections.md` correction 15 item 4 -- "the hinge penalty is in
    # TG's loss" -- and §3.1 requires the base model be unmodified, so a loop that
    # drops it is not training TG. The weight is the documented off-switch
    # (CLAUDE.md): 0.0 restores the pre-fix objective exactly, and it is stamped
    # into `frozen` so two different objectives cannot share a config hash.
    w_srep = (
        cfg.srep_norm_reg_weight if srep_norm_reg_weight is None else srep_norm_reg_weight
    )

    # Correction 31 (b), ruled 2026-09-30
    # (docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md): one epoch = one pass
    # over the document pool = n_docs / batch optimizer steps, from the pool this
    # run actually built -- not `iters`, which made §3.4's "one epoch" the whole
    # run. A stream has no pool (None); a `DocumentPool` is its own length (m-2).
    pool_docs = None if stream is not None else len(probe)
    if pool is not None:
        pool_docs = pool.length
    steps_per_epoch = epoch_steps(pool_docs, batch)
    if policy_name == "rsr" and stream is not None:
        raise ValueError(
            "an RSR arm on a never-repeating DocumentStream: the 31(b) ruling defines "
            "an epoch as one pass over the document POOL (R-2026-09-30-rsr-run-1-"
            "scope), and a stream has none -- T_warm (§3.4) is undefined here. Train "
            "on a fixed pool (n_documents), or get the stream's epoch ruled."
        )

    # S0-01 defect (b): built FROM `policy_name`, so the name stamped below into the
    # frozen config and the run_id cannot disagree with the policy that ran.
    # ADR-0009 L11: phi is built from its OWN CPU generator, so the data-order
    # generator `gen` is untouched (FIFO and RSR arms at one seed see one stream),
    # and nothing CPU-built is drawn from an MPS generator. Built BEFORE the
    # optimizer (ADR-0009 §3), which with L11 moves no RNG.
    phi_gen, phi_seed = phi_generator(seed)
    policy = build_policy(
        policy_name,
        d_model=d,
        steps_per_epoch=steps_per_epoch,
        generator=phi_gen,
        overrides=policy_overrides,
        no_warmup_ablation=no_warmup_ablation,
    )
    head = getattr(policy, "head", None)
    if head is not None:
        head.to(device)  # built on CPU (L11), then moved -- before the optimizer
    phi_params = [] if head is None else list(head.parameters())
    learns = bool(getattr(policy, "learns", False))  # ADR-0009 L13(a): beta != 0
    eval_ctx = bool(getattr(policy, "needs_eval_context", False))  # L1(a), L2
    if freeze_transformer:
        model.requires_grad_(False)  # D2(b): no L_NTP update to transformer, W_sent

    # muP groups, NOT a flat AdamW: the value head lands in its OWN `value_head`
    # group (§4.3; ADR-0009 T9). One AdamW, decoupled decay, eps pinned at 1e-8
    # (ADAMW_EPS, P-2: B1's measured regime; ADR-0009 L8 provisional answer).
    # A frozen arm with no head has nothing to optimise, and builds no
    # optimizer rather than an empty one.
    opt = None
    if any(p.requires_grad for p in model.parameters()) or phi_params:
        groups = build_param_groups(model, head, base_lr=lr, d_model=d, base_width=128)
        opt = torch.optim.AdamW(
            groups, betas=ADAMW_BETAS, weight_decay=ADAMW_WEIGHT_DECAY, eps=ADAMW_EPS
        )
    label = arm_label(
        policy_name,
        policy_overrides,
        freeze_transformer=freeze_transformer,
        warm_start=ws_record,
    )

    frozen = {
        "tg": asdict(cfg),
        "iters": iters,
        "batch": batch,
        "steps_per_stream": steps_per_stream,
        "lr": lr,
        "seed": seed,
        "policy": policy_name,
        "device": device,
        "masked_loss": masked_loss,
        "srep_norm_reg_weight": w_srep,
    }
    if n_documents is not None:
        # Stamped only when set, so the default path's config_hash is unchanged
        # and two corpus sizes can never share one.
        frozen["n_documents"] = n_documents
    if stream is not None:
        # Stamped only when set, like n_documents: the default path's config_hash
        # is unchanged and a stream run can never share it.
        frozen["stream"] = asdict(stream)
    if pool is not None:
        # m-2: stamped only when set, like n_documents and stream -- two pools
        # cannot share a config hash, and the default path's is unchanged.
        frozen["pool"] = {**asdict(pool), "ids": [pool.offset, pool.offset + pool.length]}
    if ws_record is not None:
        # m-1: the source's path, manifest entry and sha256, in the hash.
        frozen["warm_start"] = ws_record
    if policy_name == "rsr" or freeze_transformer or ws_record is not None:
        # Stamped only off the default path (its hash is S0-03's, pinned by
        # test_corpus_size_curve / test_fresh_stream). ADR-0009 L12: the label and
        # every override are in the hash, so two arms cannot share one.
        frozen["arm_label"] = label
        frozen["freeze_transformer"] = freeze_transformer
        frozen["optimizer"] = optimizer_record()
    if freeze_transformer:
        # m-4: freezing removes the UPDATE only. The model stays in train mode, so
        # dropout is live in the training forward and in the memory writes.
        frozen["frozen_model_mode"] = "train (dropout live)"
    if policy_name == "rsr":
        t_warm = float(policy.config.t_warm)
        frozen["policy_overrides"] = {
            k: _jsonable(v) for k, v in sorted((policy_overrides or {}).items())
        }
        frozen["no_warmup_ablation"] = no_warmup_ablation
        frozen["phi_seed"] = phi_seed
        frozen["observe"] = learns
        frozen["eval_capture"] = eval_ctx
        # 31(b) and §3.4: T_warm in steps, and as a fraction of total epochs.
        frozen["epoch"] = {
            "definition": "one pass over the document pool (R-2026-09-30-rsr-run-1-"
            "scope 31(b)): pool_documents / batch optimizer steps",
            "pool_documents": pool_docs,
            # m-3: the count follows the ruling (pool / batch draws), but the
            # draws are torch.randint over the pool, WITH replacement: one epoch
            # covers about 1 - 1/e of the documents, not each exactly once.
            "sampling": "with replacement (torch.randint over the pool)",
            "steps_per_epoch": steps_per_epoch,
            "t_warm_steps": t_warm,
            "t_warm_epochs": t_warm / steps_per_epoch,
            "total_epochs": iters / steps_per_epoch,
            "t_warm_fraction_of_total_epochs": (t_warm / iters) if iters else None,
        }
    run_id = f"{policy_name}-d{d}-s{steps_per_stream}-b{batch}-{_config_hash(frozen)}"

    start = 0
    if resume:
        state = ck.load(
            resume, model=model, optimizer=opt, policy=policy, restore_rng=True
        )
        start = int(state.get("step", 0))

    docs = generate(
        SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed, **corpus_kw)
    )
    vocab_map = build_vocab(docs)
    if pool is not None:
        # m-2: the pool's documents, by id, in the vocabulary of [0, vocab_docs).
        docs = pool_documents(
            pool, SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed)
        )
        check_vocabulary_closure(docs, vocab_map)
    if pool_docs is not None and len(docs) != pool_docs:
        raise RuntimeError(
            f"the epoch was derived from a pool of {pool_docs} documents but the "
            f"sampler draws from {len(docs)} (31(b): one pass over THIS pool)"
        )
    all_ids, all_mask = encode(
        docs, vocab_map, max_tokens=max_tokens, steps=steps_per_stream
    )
    all_ids, all_mask = all_ids.to(device), all_mask.to(device)
    # S0-03: the answer-token supervision mask, for the tally only. It does NOT
    # enter the objective -- `loss` is the same next-token loss as before; the
    # answer tokens are simply now among its targets.
    all_tmask, _ = answer_targets(
        docs, vocab_map, max_tokens=max_tokens, steps=steps_per_stream
    )
    all_tmask = all_tmask.to(device)
    stream_cfg = (
        None
        if stream is None
        else SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed)
    )
    cur: dict[str, torch.Tensor] = {}
    hb = Heartbeat(
        out / "heartbeat.jsonl",
        run_id=run_id,
        provenance={
            "git_sha": _sha(),
            "device": device,
            "seed": seed,
            "config_hash": _config_hash(frozen),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "documents": len(docs),
        },
    )
    hb.header(
        config=frozen,
        resumed_from=str(resume) if resume else None,
        start_step=start,
        # Every run records its optimizer, eps included (B1: beta's liveness turns
        # on eps). In the header, not `frozen`, on the default path: that path's
        # config hash is S0-03's and is pinned.
        optimizer=optimizer_record() if opt is not None else None,
        arm_label=label,
    )

    # Both numbers are computed every step and only one is optimised. The other is
    # the *common yardstick*: comparing a masked arm's masked loss against an
    # unmasked arm's unmasked loss compares two different objectives and is
    # tautologically different. Real-token NLL is the same quantity in both arms.
    tally: dict[str, float] = {}

    def step_fn(t, o, ids_t, mask_t, row_valid):
        lm = (lm_loss if masked_loss else lm_loss_unmasked)(o.logits, ids_t, mask_t)
        # Defect (c): the squared hinge on the PRE-normalization gestalt norm,
        # holding it in [target +/- margin]. Gradient flows into the transformer and
        # `W_sent` -- correctly: this is TG's own regularizer, not the retention
        # loss, and CLAUDE.md's stop-gradient prohibition is about the latter.
        hinge = o.srep_norm_penalty.mean()
        loss = (lm + w_srep * hinge) if w_srep else lm
        with torch.no_grad():
            valid = mask_t[:, 1:].reshape(-1)
            tally["lm_sum"] = tally.get("lm_sum", 0.0) + float(lm)
            tally["srep_hinge_sum"] = tally.get("srep_hinge_sum", 0.0) + float(hinge)
            tally["real_sum"] = tally.get("real_sum", 0.0) + float(
                lm_loss(o.logits, ids_t, mask_t)
            )
            tally["all_sum"] = tally.get("all_sum", 0.0) + float(
                lm_loss_unmasked(o.logits, ids_t, mask_t)
            )
            tally["n_targets"] = tally.get("n_targets", 0.0) + valid.numel()
            tally["n_pad_targets"] = tally.get("n_pad_targets", 0.0) + float(
                (~valid).sum()
            )
            # S0-03: answer-token NLL, read off the supervision mask.
            ans = cur["tmask"][:, t, 1:].reshape(-1)
            per = lm_token_losses(o.logits, ids_t)
            tally["ans_sum"] = tally.get("ans_sum", 0.0) + float(per[ans].sum())
            tally["n_ans"] = tally.get("n_ans", 0.0) + float(ans.sum())
            tally["n_real"] = tally.get("n_real", 0.0) + float(valid.sum())
        return loss

    last: dict[str, float] = {}
    try:
        for it in range(start, iters):
            t0 = time.time()
            tally.clear()
            if stream is None:
                sel = torch.randint(
                    0, all_ids.shape[0], (batch,), generator=gen, device=device
                )
                ids, mask = all_ids[sel], all_mask[sel]
                cur["tmask"] = all_tmask[sel]
            else:
                # Step `it` reads its own window, so a resumed run continues the
                # stream at the resumed step.
                sdocs = stream_documents(stream, it, batch, stream_cfg)
                check_vocabulary_closure(sdocs, vocab_map)
                ids, mask = encode(
                    sdocs, vocab_map, max_tokens=max_tokens, steps=steps_per_stream
                )
                tm, _ = answer_targets(
                    sdocs, vocab_map, max_tokens=max_tokens, steps=steps_per_stream
                )
                ids, mask, cur["tmask"] = ids.to(device), mask.to(device), tm.to(device)
            lengths = torch.full((batch,), steps_per_stream, device=device)
            # Correction 31: §3.4's warmup counts optimizer steps. `it` counts THIS
            # run's steps, so a resumed run keeps its place, and a warm start (m-1)
            # begins at k = 0: the source checkpoint's step is never carried in, so
            # T_warm cannot be skipped by starting "past" it.
            if hasattr(policy, "set_train_step"):
                policy.set_train_step(it)
            timings: dict[str, float] = {}
            loss = run_policy_loop(
                model,
                ids,
                mask,
                lengths,
                policy,
                step_fn=step_fn,
                observe=learns,
                eval_capture=eval_ctx,
                timings=timings,
            )
            # §3.3, read per correction 2: L = L_NTP + beta * L_MC. ADR-0009 L13(a):
            # beta = 0 means L_MC is NOT computed (`learns` is False), never x0.
            l_mc = policy.retention_loss() if learns else None
            objective = None if freeze_transformer else loss
            if l_mc is not None:
                weighted = policy.config.beta * l_mc
                objective = weighted if objective is None else objective + weighted
            gnorm = phi_gnorm = None
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                if objective is not None:
                    objective.backward()
                if not freeze_transformer:
                    # ADR-0009 L8 (signed): phi is NOT in the transformer's clip. A
                    # joint norm would make the transformer's clip factor depend on
                    # L_MC -- a gradient-free channel from the retention loss into
                    # transformer updates (§2). phi has no clip of its own (L8).
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if phi_params:
                    phi_gnorm = _grad_norm(phi_params)  # logged, never clipped (L8)
                opt.step()
            if device == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0

            # Per sentence step, so the number is a per-token NLL rather than a
            # stream sum. `loss_real_tokens` is the same quantity in both arms and
            # is the only honest masked-vs-unmasked comparison; `loss` is whichever
            # objective this arm actually minimised.
            last = {
                "loss": float(loss.detach()) / steps_per_stream,
                # The objective decomposed. `loss_lm` is the language-model term
                # alone; `loss_srep_hinge` is the (c) hinge UNWEIGHTED, so the term
                # stays auditable independently of `srep_norm_reg_weight`.
                "loss_lm": tally["lm_sum"] / steps_per_stream,
                "loss_srep_hinge": tally["srep_hinge_sum"] / steps_per_stream,
                "srep_norm_reg_weight": w_srep,
                "loss_real_tokens": tally["real_sum"] / steps_per_stream,
                "loss_all_targets": tally["all_sum"] / steps_per_stream,
                "pad_target_fraction": tally["n_pad_targets"] / tally["n_targets"],
                "n_targets_scored": tally["n_targets"],
                "n_pad_targets_scored": tally["n_pad_targets"],
                # S0-03. Mean NLL per answer target (not per sentence step), and
                # what fraction of the real targets the answers are.
                "loss_answer_tokens": tally["ans_sum"] / max(tally["n_ans"], 1.0),
                "n_answer_targets": tally["n_ans"],
                "answer_target_fraction": tally["n_ans"] / max(tally["n_real"], 1.0),
                "step": it,
            }
            rsr_beat: dict[str, Any] = {}
            if policy_name == "rsr":
                # ADR-0009 §4: L_MC tagged with k and warm, so it splits at §3.4's
                # on/off-policy boundary; mean G by t (the truncation profile);
                # corr(psi_hat, age) for §7.1; L1's eval-forward cost, measured.
                stats = dict(getattr(policy, "last_mc_stats", {}) or {})
                rsr_beat = {
                    "arm_label": label,
                    "train_step": it,
                    "loss_mc": None if l_mc is None else float(l_mc.detach()),
                    "mc": stats or None,
                    "phi_grad_norm": phi_gnorm,
                    "eval_forward_s": timings.get("eval_forward_s"),
                    "step_s": dt,
                    # Spec §7.1 / ADR-0009 L6 (m-8): this step's displacement
                    # distribution, victim rank and age histograms, partial
                    # rho(score, age | content). Logged, never gated.
                    "diag": policy.step_diagnostics(),
                }
                last.update(
                    loss_mc=rsr_beat["loss_mc"],
                    phi_grad_norm=phi_gnorm,
                    eval_forward_s=rsr_beat["eval_forward_s"],
                    step_s=dt,
                )
                # Review MINOR-3: the diagnostics are computed EVERY step, so they
                # are written every step -- to their own file -- whatever
                # beat_every is. The beat carries the same dict on beat steps.
                _write_step_diagnostics(
                    out / DIAGNOSTICS_FILE,
                    {
                        "run_id": run_id,
                        "arm_label": label,
                        "step": it,
                        "train_step": it,
                        "diag": rsr_beat["diag"],
                    },
                )
            if it % beat_every == 0:
                attr = policy.attribution() if hasattr(policy, "attribution") else None
                hb.beat(
                    it,
                    loss=last["loss"],
                    loss_sum=float(loss.detach()),
                    loss_lm=last["loss_lm"],
                    loss_srep_hinge=last["loss_srep_hinge"],
                    srep_norm_reg_weight=w_srep,
                    loss_real_tokens=last["loss_real_tokens"],
                    loss_all_targets=last["loss_all_targets"],
                    pad_target_fraction=last["pad_target_fraction"],
                    n_targets_scored=last["n_targets_scored"],
                    loss_answer_tokens=last["loss_answer_tokens"],
                    n_answer_targets=last["n_answer_targets"],
                    answer_target_fraction=last["answer_target_fraction"],
                    masked_loss=masked_loss,
                    # 🔴 `exp(loss / steps)` was a perplexity only while `loss` WAS
                    # the LM loss. With (c)'s hinge in the objective it no longer
                    # is, and leaving it would have shipped a new silent defect
                    # inside the fix for an old one: a regularizer inside a number
                    # reported as a perplexity. At `w_srep = 0` this is identical
                    # to the old expression.
                    ppl=float(torch.exp(torch.tensor(last["loss_lm"]))),
                    gini=None,  # wired when the policy exposes an attention trace
                    attribution=attr,
                    rank_shift=None,  # wired with RSRPolicy; 0 under FIFO by construction
                    grad_norm=None if gnorm is None else float(gnorm),
                    sent_per_s=round(batch * steps_per_stream / dt, 1),
                    mem_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2)
                    if device == "mps"
                    else None,
                    lr=None if opt is None else opt.param_groups[0]["lr"],
                    **rsr_beat,
                )
            if ckpt_every and (it + 1) % ckpt_every == 0:
                ck.save(
                    out / f"ckpt-{it + 1:06d}.pt",
                    step=it + 1,
                    model=model,
                    optimizer=opt,
                    policy=policy,
                    # The frozen config and the vocabulary's sha256 travel with
                    # the bytes, so a later warm start can check the map and the
                    # seed against the checkpoint itself (review MAJOR-1).
                    meta={
                        "run_id": run_id,
                        "config": json.loads(json.dumps(frozen, default=str)),
                        "vocab_sha256": vocab_sha256(vocab_map),
                    },
                    **(
                        {}
                        if stream is None
                        else {"stream": {**asdict(stream), "next_step": it + 1}}
                    ),
                )
    except BaseException as e:
        hb.crash(e)
        raise
    # liveness-wiring: every run measures its own memory's liveness before it
    # reports (R-2026-09-22-inert-exit-5), and an INERT run's checkpoints are
    # quarantined (R-2026-09-22-inert-checkpoint-quarantine).
    liveness = _measure_liveness(model, init_state, cfg, frozen, device=device)
    if liveness["band"] == "inert":
        liveness["quarantine"] = str(
            ck.quarantine(
                out,
                {
                    "run_id": run_id,
                    "ratio": liveness.get("ratio"),
                    "reason": liveness.get("reason"),
                    "ruling": "R-2026-09-22-inert-checkpoint-quarantine",
                },
            )
        )
    liveness["exit_code"] = int(liveness_exit(liveness))
    hb.footer("completed", final_step=iters, liveness=liveness)
    return {
        "run_id": run_id,
        "heartbeat": str(out / "heartbeat.jsonl"),
        "config_hash": _config_hash(frozen),
        # The frozen config itself, not just its hash: a hash proves two runs agree
        # and proves nothing about what either of them ran.
        "config": frozen,
        "policy": type(policy).__name__,
        "arm_label": label,
        "git_sha": _sha(),
        "steps_requested": iters,
        # `steps_done` is the loop's own count, not the request. A resumed or
        # crashed run must not be able to report the number it asked for.
        "steps_done": (last["step"] + 1) if last else start,
        "final": last,
        "liveness": liveness,
        # §7.1 / ADR-0009 L6 (m-8), cumulative over the run: ADR-0006's
        # displacement distribution and the victim rank/age histograms. The
        # per-step partial rho is on every beat's `diag`.
        "diagnostics": (
            policy.rank_shifts.summary() if hasattr(policy, "rank_shifts") else None
        ),
    }


#: decisive PREREG *Condition*: `measure_documents = 8`, the first documents of
#: the seed's training corpus (#17's measurement batch).
LIVENESS_MEASURE_DOCUMENTS = 8

#: `R-2026-09-22-inert-exit-5`'s mapping, band -> exit. Anything else -- no
#: measurement, an unknown band -- is `3`: the run did not demonstrate anything.
LIVENESS_EXIT = {
    "live": Exit.OK,
    "inert": Exit.INERT,
    "inconclusive": Exit.FAIL,
    "invalid": Exit.DID_NOT_RUN,
}


def liveness_exit(liveness: dict | None) -> Exit:
    """The run's exit code from its liveness band. Never `0` without a band."""
    band = (liveness or {}).get("band")
    return LIVENESS_EXIT.get(band, Exit.DID_NOT_RUN)


def liveness_batch(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (ids, mask) on CPU: the first `LIVENESS_MEASURE_DOCUMENTS` documents of
    the corpus `train()` built from `config` (its frozen dict). A stream run's
    are its vocabulary documents -- the fixed corpus it takes `V` from."""
    kw = {}
    if "n_documents" in config:
        kw = {"n_documents": config["n_documents"]}
    if "stream" in config:
        kw = {"n_documents": config["stream"]["vocab_documents"]}
    S = config["steps_per_stream"]
    if "pool" in config:
        # m-2: a pool run's training corpus is the pool; its first documents, in
        # the vocabulary of [0, vocab_documents) that train() encoded them with.
        p = config["pool"]
        vdocs = generate(
            SyntheticConfig(
                sentences_per_document=S,
                seed=config["seed"],
                n_documents=p["vocab_documents"],
            )
        )
        n = min(LIVENESS_MEASURE_DOCUMENTS, p["length"])
        pdocs = pool_documents(
            DocumentPool(p["offset"], n, p["vocab_documents"]),
            SyntheticConfig(sentences_per_document=S, seed=config["seed"]),
        )
        return encode(
            pdocs,
            build_vocab(vdocs),
            max_tokens=config["tg"]["max_sentence_tokens"],
            steps=S,
        )
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=config["seed"], **kw))
    return encode(
        docs[:LIVENESS_MEASURE_DOCUMENTS],
        build_vocab(docs),
        max_tokens=config["tg"]["max_sentence_tokens"],
        steps=S,
    )


def _measure_liveness(
    model: TGModel, init_state: dict, cfg: TGConfig, frozen: dict, *, device: str
) -> dict:
    """`memory_liveness.measure_liveness` on the trained model vs its own init.

    A measurement that raises is `invalid` (exit 3), never `0`, `1` or `5`: the
    traceback is recorded, and the run reports that it did not demonstrate
    anything. Imported here, not at module top: `memory_liveness` imports this
    module.

    Both models are fresh modules loaded from state dicts, as the decisive run
    loads checkpoints: the in-loop model holds non-leaf tensors and cannot be
    deep-copied, and `with_memory_disabled` deep-copies. They are built under
    `fork_rng`, so the global RNG is exactly where training left it."""
    import traceback

    from rsr.metrics import memory_liveness as ml

    try:
        with torch.random.fork_rng(devices=[]):
            trained, decoy = TGModel(cfg), TGModel(cfg)
        trained.load_state_dict(model.state_dict())
        decoy.load_state_dict(init_state)
        trained, decoy = trained.to(device), decoy.to(device)
        ids, mask = liveness_batch(frozen)
        return ml.measure_liveness(
            trained, decoy, ids.to(device), mask.to(device), seed=frozen["seed"]
        )
    except Exception as e:
        return {
            "band": "invalid",
            "reason": f"the measurement raised {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """`["beta=0.1", "b_enabled=false", "psi_override=neg_age"]` -> a dict.

    A value is read as JSON when it parses (numbers, true/false, null) and kept as
    the bare string otherwise. A key given twice is refused, not last-one-wins."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--policy-override {item!r}: expected KEY=VALUE")
        if key in out:
            raise ValueError(f"--policy-override {key!r} given twice")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
        if isinstance(out[key], int) and not isinstance(out[key], bool):
            # RSRConfig's numeric fields are floats except a_max; keep a_max int.
            out[key] = out[key] if key == "a_max" else float(out[key])
    return out


def _cli_pool_and_warm_start(p, a) -> tuple[DocumentPool | None, WarmStart | None]:
    """The CLI's pool and warm start, built as the same objects `train()` takes.
    A partial or inconsistent set of flags is a usage error (exit 3), never a
    silently ignored flag."""
    pool = warm = None
    if (a.pool_offset is None) != (a.pool_length is None):
        p.error("--pool-offset and --pool-length go together")
    if a.pool_offset is not None:
        try:
            pool = DocumentPool(a.pool_offset, a.pool_length, a.pool_vocab_documents)
        except ValueError as e:
            p.error(str(e))
    ws_flags = (
        a.warm_start_sha256,
        a.warm_start_manifest,
        a.warm_start_root,
        a.warm_start_source_config,
    )
    flagged = a.allow_unrecorded_vocabulary or a.allow_seed_mismatch
    if a.warm_start is None:
        if any(f is not None for f in ws_flags) or flagged:
            p.error("--warm-start-* and --allow-* flags need --warm-start")
        return pool, None
    missing = [
        n
        for n, v in (
            ("--warm-start-sha256", a.warm_start_sha256),
            ("--warm-start-manifest", a.warm_start_manifest),
            ("--warm-start-root", a.warm_start_root),
        )
        if v is None
    ]
    if missing:
        p.error(f"--warm-start needs {', '.join(missing)}")
    warm = WarmStart(
        checkpoint=a.warm_start,
        manifest=a.warm_start_manifest,
        root=a.warm_start_root,
        expected_sha256=a.warm_start_sha256,
        source_config=a.warm_start_source_config,
        allow_unrecorded_vocabulary=a.allow_unrecorded_vocabulary,
        allow_seed_mismatch=a.allow_seed_mismatch,
    )
    return pool, warm


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog="rsr-train", description=__doc__)
    for name, typ, dflt in (
        ("--d", int, 128),
        ("--batch", int, 16),
        ("--iters", int, 50),
        ("--steps-per-stream", int, 48),
        ("--memory-slots", int, 16),
        ("--lr", float, 1e-3),
        ("--seed", int, 0),
        ("--beat-every", int, 1),
        ("--ckpt-every", int, 25),
    ):
        p.add_argument(name, type=typ, default=dflt)
    # S0-01 defect (e). This defaulted to 50257 against a corpus of 156 unique
    # words, so `V = vocab if vocab else 4 + len(build_vocab(probe))` never reached
    # its derived branch from the CLI **at the default** -- `--vocab 0` always did,
    # which is why the original "unreachable from the CLI" was the larger claim and
    # this is the true one. `None` is falsy, so the default now derives; an explicit
    # value still wins, which is what the flag is for.
    p.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="vocabulary size; omit to derive it from the corpus (4 specials + "
        "unique words). The synthetic corpus at seed 0 derives V=172 (156 words + "
        "16 answer symbols, S0-03).",
    )
    # S0-01 defect (b): the policy was unreachable from the CLI while its NAME was
    # stamped into the run_id. `choices` refuses an unknown name rather than
    # letting it through to be stamped over a FIFO run.
    p.add_argument("--policy", default="fifo", choices=POLICIES)
    # ADR-0009 L12: labelled overrides, e.g. --policy-override beta=0.1. Values are
    # JSON (0.1, false, null) or a bare string (neg_age). Stamped into the frozen
    # config, the hash and the label -- never a default.
    p.add_argument(
        "--policy-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="an RSRConfig field override (repeatable); b_enabled=false and "
        "shadow_enabled=false are required for --policy rsr in slice 1",
    )
    p.add_argument(
        "--no-warmup-ablation",
        action="store_true",
        help="declare the T_warm = 0 ablation arm; the only way t_warm may be "
        "overridden (ADR-0009 L12)",
    )
    p.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="no update to the transformer or W_sent (R-2026-09-30-rsr-run-1-scope "
        "D2(b)); recorded in the frozen config and the label",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--out-dir", default="runs/dev")
    p.add_argument("--resume", default=None)
    # Review MINOR-7: run 1's RESULTS.md quotes ONE command. The pool and the warm
    # start are reachable from the CLI and go through the same objects train()
    # validates (DocumentPool / WarmStart), never a second parser of their own.
    g = p.add_argument_group("document pool (m-2)")
    g.add_argument("--pool-offset", type=int, default=None)
    g.add_argument("--pool-length", type=int, default=None)
    g.add_argument("--pool-vocab-documents", type=int, default=64)
    w = p.add_argument_group("weights-only warm start (m-1, D2(a))")
    w.add_argument("--warm-start", default=None, metavar="CKPT")
    w.add_argument(
        "--warm-start-sha256",
        default=None,
        help="required with --warm-start: the checkpoint's expected sha256",
    )
    w.add_argument("--warm-start-manifest", default=None, metavar="T0_MANIFEST")
    w.add_argument("--warm-start-root", default=None, metavar="MANIFEST_ROOT")
    w.add_argument("--warm-start-source-config", default=None, metavar="JSON")
    w.add_argument("--allow-unrecorded-vocabulary", action="store_true")
    w.add_argument("--allow-seed-mismatch", action="store_true")
    a = p.parse_args(argv)
    pool, warm = _cli_pool_and_warm_start(p, a)
    r = train(
        d=a.d,
        batch=a.batch,
        iters=a.iters,
        steps_per_stream=a.steps_per_stream,
        vocab=a.vocab,
        memory_slots=a.memory_slots,
        lr=a.lr,
        seed=a.seed,
        device=a.device,
        out_dir=a.out_dir,
        beat_every=a.beat_every,
        ckpt_every=a.ckpt_every,
        resume=a.resume,
        policy_name=a.policy,
        policy_overrides=parse_overrides(a.policy_override) or None,
        no_warmup_ablation=a.no_warmup_ablation,
        freeze_transformer=a.freeze_transformer,
        pool=pool,
        warm_start=warm,
    )
    print(json.dumps(r, indent=2, default=str))
    # liveness-wiring: 0 live, 5 INERT, 1 inconclusive, 3 not a valid measurement.
    return status(liveness_exit(r.get("liveness")))


if __name__ == "__main__":
    run_main(main)
