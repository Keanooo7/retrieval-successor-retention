"""The Monte-Carlo return (spec §3.3; correction 2; ADR-0009 §1).

    G_i(t) = sum_{k=0}^{S-t} gamma^k * r_i(t+k)

read with 0-based `t` over one stream of `T_b` observed steps as

    G_i(t) = sum_{k=0}^{T_b-1-t} gamma^k * r_i(t+k)

**The return is truncated at the stream end, as the spec writes it** (ADR-0009
§1): late samples carry systematically shorter returns. That is spec-fixed, not a
choice, and `RSRPolicy.retention_loss` reports mean `G` by `t` so the truncation
profile is visible.

**Without the shadow buffer (ADR-0009 L6(a), slice 1) the return is also cut at
the policy's own eviction**: `r_i` exists only for slots in memory (§3.4), so an
evicted gestalt contributes 0 to every later term. Under FIFO (every step before
`T_warm`) that cut is at age `M`, which makes the *target* age-shaped even though
`psi_hat` has no age input -- the FIFO-censored-target caveat every slice-1
RESULTS must carry. Nothing here hides it; the zeros are the censoring.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["mc_returns"]


def mc_returns(r: Tensor, gamma: float) -> Tensor:
    """`[..., T]` realized `r` along the last axis -> `[..., T]` returns `G`.

    `r[..., t]` is the realized `r_i(t)`, **0 where the slot was not live** (not
    yet written, or evicted -- slice 1 has no shadow buffer). The sum starts at
    `k = 0`, so `G(t)` includes the present step's already-paid `r(t)` (ADR-0009,
    "Observations for Brendan": spec-fixed), and stops at the last step of the
    axis, which the caller makes the row's last observed step.

    `gamma = 0` is the separating control arm (§2): `G(t) = r(t)` exactly. It is
    implemented as that case rather than left to `0 ** 0` in the recursion
    (ADR-0009 §1).
    """
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma = {gamma} is outside [0, 1) (spec §4.5)")
    if gamma == 0.0:
        return r.clone()
    out = torch.empty_like(r)
    nxt = torch.zeros_like(r[..., 0])
    for t in reversed(range(r.shape[-1])):
        out[..., t] = r[..., t] + gamma * nxt
        nxt = out[..., t]
    return out
