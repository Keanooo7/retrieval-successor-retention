"""PG-19 subset assembly (spec section 5.3).

`deepmind/pg19` on HuggingFace, Apache-2.0, ungated. 28,602 train books /
11.45 GB. The 30M-token subset is ~0.25% of train, so assembling it is cheap.

**The subset-selection rule is pre-registered with the E0i threshold** -- picking
books after seeing the gap histogram would make the gate unfalsifiable.

Section 5.3 cites [P8], which introduced the benchmark and whose answer to this
problem is "compress rather than evict"; section 11 must address that directly.
[P8] is **arXiv:1911.05507, 2019** -- see correction 12.

Note correction 13: [P2] trains on **WikiText**, not PG-19. "PG-19 -- where the
claim lives" is RSR's own choice, so TG's behaviour on this corpus at `S = 80` is
uncharacterized on two axes at once.
"""

from __future__ import annotations

__all__ = ["build_subset"]


def build_subset(*args, **kwargs):
    raise NotImplementedError("E0i / kickoff T3. Spec section 5.3.")
