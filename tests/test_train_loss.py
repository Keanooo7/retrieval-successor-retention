"""Cycle 1 -- `rsr.train.loop`'s language-model loss scores only real targets.

**Why this file exists.** `rsr.train` had no test that imported anything but
`checkpoint`, and the one line nobody was watching was the objective itself:

```python
F.cross_entropy(lg[:, :-1].reshape(-1, V), ids_t[:, 1:].reshape(-1), reduction="mean")
```

No `ignore_index`, no mask -- while `mask_t` was already being passed in by
`policy_loop.py:140` and already unused. On the committed synthetic corpus the
padding is not a rounding error: sentences are ~5 tokens inside a 64-token frame,
so the overwhelming majority of scored targets are PAD and the reported number was
mostly the model's skill at predicting zeros.

The corpus test below measures that fraction rather than quoting it. The `lm_loss`
tests are analytic -- logits are pinned to a known shape so the expected loss is
arithmetic, not a second implementation of the thing under test.

📌 §12.4: measure, don't extrapolate. The 0.50 threshold in the corpus test is the
falsifier's own bound from `docs/lab-notes/dispatch-cycle-01-masked-loss.md`, not a
number chosen after seeing the data.
"""

from __future__ import annotations

import pytest
import torch

from rsr.data.synthetic import SyntheticConfig, generate
from rsr.train.loop import build_vocab, encode, lm_loss

PAD = 0

# The falsifier's first half: "the fraction of scored targets that are PAD is
# below 0.50" kills the reading. Registered here as the bound, not as the result.
FALSIFIER_PAD_FLOOR = 0.50


@pytest.fixture(scope="module")
def corpus():
    """The committed synthetic corpus, encoded exactly as `train()` encodes it."""
    steps, max_tokens = 48, 64
    docs = generate(SyntheticConfig(sentences_per_document=steps, seed=0))
    ids, mask = encode(docs, build_vocab(docs), max_tokens=max_tokens, steps=steps)
    return ids, mask


def test_pad_dominates_the_scored_targets_on_the_committed_corpus(corpus):
    """The population is `ids[:, :, 1:]` -- targets, not tokens, because of the
    `[:, :-1]` / `[:, 1:]` shift. Corpus-shaped: the whole corpus encoded once."""
    ids, _ = corpus
    tgt = ids[:, :, 1:]
    frac = float((tgt == PAD).sum()) / tgt.numel()
    assert frac > FALSIFIER_PAD_FLOOR, (
        f"PAD fraction {frac:.4f} is at or below the falsifier's 0.50 floor; "
        f"the reading that the loss is dominated by padding is dead."
    )


def test_the_mask_and_the_pad_id_identify_the_same_positions(corpus):
    """`lm_loss` selects on `mask_t`, not on `ids != pad_id`. They agree on this
    corpus, and the test says so rather than leaving it assumed -- the vocabulary
    reserves 0-3 (`loop.build_vocab`), so no real token can collide with PAD."""
    ids, mask = corpus
    assert bool(((ids != PAD) == mask).all())


def _pad_certain_logits(batch: int, length: int, vocab: int) -> torch.Tensor:
    """Logits that predict PAD with near-certainty everywhere.

    `logit[PAD] = 20`, every other class `0`, so for a PAD target the loss is
    `~7e-9` and for any real target it is `20 - logsumexp ~= 20`. That makes both
    the masked and the unmasked expectations arithmetic.
    """
    lg = torch.zeros(batch, length, vocab)
    lg[..., PAD] = 20.0
    return lg


def test_lm_loss_scores_only_real_targets():
    """The gate. Unmasked, the easy PAD targets drag the mean down; masked, they
    are absent and the number is the model's real-token loss."""
    vocab, length = 8, 5
    ids = torch.tensor([[5, 6, 2, PAD, PAD], [7, 2, PAD, PAD, PAD]])
    mask = ids != PAD
    lg = _pad_certain_logits(ids.shape[0], length, vocab)

    # targets = ids[:, 1:] -> [[6, 2, 0, 0], [2, 0, 0, 0]] : 3 real of 8.
    n_targets, n_real = 8, 3
    per_real, per_pad = 20.0, 0.0  # up to ~7e-9 for the PAD term
    unmasked_expected = (n_real * per_real + (n_targets - n_real) * per_pad) / n_targets
    masked_expected = per_real

    got = float(lm_loss(lg, ids, mask))
    assert got == pytest.approx(masked_expected, abs=1e-3), (
        f"lm_loss returned {got:.6f}; the mean over the 3 real targets is "
        f"{masked_expected:.6f} and the mean over all 8 is "
        f"{unmasked_expected:.6f}. A value at the latter means the padding is "
        f"still being scored."
    )
    assert abs(got - unmasked_expected) > 1.0


def test_lm_loss_falls_back_to_zero_when_a_step_has_no_real_target():
    """An all-PAD step contributes nothing rather than `nan`. `0 / 0` here would
    poison the whole stream's accumulated loss, and a `nan` that appears only for
    documents shorter than `steps_per_stream` is exactly the kind of failure that
    shows up as a crash three hours in."""
    vocab, length = 8, 5
    ids = torch.zeros(2, length, dtype=torch.long)
    mask = ids != PAD
    got = float(lm_loss(_pad_certain_logits(2, length, vocab), ids, mask))
    assert got == pytest.approx(0.0, abs=1e-6)


def test_lm_loss_keeps_its_gradient():
    """The masked reduction must not detach. `phi` is the only thing forbidden the
    gradient (CLAUDE.md); the language-model loss is the transformer's own."""
    vocab, length = 8, 4
    ids = torch.tensor([[5, 6, 2, PAD]])
    lg = _pad_certain_logits(1, length, vocab).requires_grad_(True)
    lm_loss(lg, ids, ids != PAD).backward()
    assert lg.grad is not None and bool((lg.grad != 0).any())
