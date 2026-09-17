"""Reintroduction loss vs gap `k` -- the primary metric (spec section 5.5).

Loss on sentences reintroducing an entity last mentioned `k` steps prior,
bucketed. E3's headline: RSR flatter than FIFO, H2O **and RSR(gamma=0)** for
`40 < k <= 64`, with no PPL regression.

**Lookahead should matter *more* at large `k` -- that interaction is the strongest
form of the result**, and it is why E0i's threshold requires the `(40, 64]` window
to split into two independently powered sub-buckets.
"""

from __future__ import annotations

__all__ = ["reintroduction_loss_by_gap"]


def reintroduction_loss_by_gap(*args, **kwargs):
    raise NotImplementedError("Sprint 2 / E3. Spec section 5.5.")
