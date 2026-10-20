"""The within-step pairwise ranking loss (PREREG rsr-run3-ranking §2, the one change).

PLAN-2026-10-19 §2 (phase 2), for this test only (ruling draft
R-2026-10-19-l9-ranking-loss): **the only change is the loss**. ADR-0009 L9's squared
error (signed in `docs/owner/rulings/R-2026-09-30-adr0009-signoff.md`) is replaced, when
the switch `RSRConfig.loss == "rank"`, by

    l_g = sum_{(i, j) in P} w_ij * softplus(s_hat_j - s_hat_i) / sum_{(i, j) in P} w_ij

over one **group** g = one `(row, t)`, where

* `P = {(i, j) in L x L : G_i > G_j}` (float compare as stored; ties form no pair);
* `b_j` is the exact bottom-k membership over `L` (phase 1's 1c definition: 1 strictly
  below the k-th smallest value, `(k - #strictly below) / #tied` at it, 0 above; every
  member is 1 when `|L| <= k`); no RNG;
* `w_ij = 1 + (omega - 1) * max(b_i, b_j)`;

and the batch loss is the mean of `l_g` over the groups with `|P| >= 1` (L9's
structure: one term per `(row, t)`, then a mean over `(row, t)`). A group with no pair
contributes nothing and is counted. No temperature, no secondary loss (PREREG §2.2).

Everything here is on padded groups: ``[n_groups, L]`` scores, returns and a ``valid``
mask (the pair set `L` of each group). Padded cells must hold FINITE scores (a NaN there
would reach the head's gradient through ``0 * NaN``); callers fill them with 0.

`G` is data: it is detached here, and the membership `b` is computed from it without a
graph. Only the scores carry gradient, so only phi = {W, u} receives one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "LOSSES",
    "LOSS_MC",
    "LOSS_RANK",
    "RankLoss",
    "argmin_agreement",
    "bottom_k_membership",
    "pair_weights",
    "rank_loss_groups",
]

LOSS_MC = "mc"
"""ADR-0009 L9's squared error (run 1, run 2). The default: the switch off."""

LOSS_RANK = "rank"
"""PREREG rsr-run3-ranking §2: the within-step pairwise logistic loss."""

LOSSES = (LOSS_MC, LOSS_RANK)


@dataclass(frozen=True)
class RankLoss:
    """The batch loss and its bookkeeping. ``loss`` is None when no group has a pair."""

    loss: Tensor | None
    per_group: Tensor  # [n_groups] l_g, NaN where the group has no pair
    has_pair: Tensor  # [n_groups] bool
    n_groups: int
    n_groups_with_pair: int
    n_groups_no_pair: int
    n_pairs: int


def bottom_k_membership(G: Tensor, valid: Tensor, k: int) -> Tensor:
    """``[n_groups, L]`` exact bottom-k membership ``b`` (0 off ``valid``).

    With ``n = |L|`` the valid count of a group: if ``n <= k`` every member is 1;
    otherwise, with ``v`` the k-th smallest valid value, ``b = 1`` strictly below
    ``v``, ``b = (k - #strictly below) / #tied`` at ``v``, 0 above. ``sum(b) ==
    min(n, k)`` per group. No RNG."""
    if int(k) != k or k < 1:
        raise ValueError(f"k={k!r}: a positive integer")
    k = int(k)
    G = G.detach()
    if G.shape != valid.shape or G.dim() != 2:
        raise ValueError(f"G {tuple(G.shape)} and valid {tuple(valid.shape)}: [g, L]")
    n = valid.sum(-1)
    b = valid.to(G.dtype)  # every member is 1 when n <= k
    if G.shape[1] > k:
        inf = torch.full_like(G, float("inf"))
        kth = torch.where(valid, G, inf).sort(-1).values[:, k - 1 : k]  # [g, 1]
        below = valid & (kth > G)
        tied = valid & (kth == G)
        n_below = below.sum(-1, keepdim=True).to(G.dtype)
        n_tied = tied.sum(-1, keepdim=True).to(G.dtype)
        frac = (k - n_below) / n_tied.clamp(min=1)
        bk = below.to(G.dtype) + tied.to(G.dtype) * frac
        b = torch.where((n > k).unsqueeze(-1), bk, b)
    return b


def pair_weights(G: Tensor, valid: Tensor, k: int, omega: float) -> tuple[Tensor, Tensor]:
    """``(pair, w)``, each ``[n_groups, L, L]`` indexed ``[g, i, j]``: ``pair`` is
    ``G_i > G_j`` with both valid; ``w_ij = 1 + (omega - 1) max(b_i, b_j)`` on pairs,
    0 elsewhere."""
    if not (float(omega) == float(omega) and float(omega) >= 1.0):
        raise ValueError(f"omega={omega!r}: a finite number >= 1")
    G = G.detach()
    b = bottom_k_membership(G, valid, k)
    pair = valid.unsqueeze(2) & valid.unsqueeze(1) & (G.unsqueeze(2) > G.unsqueeze(1))
    w = 1.0 + (float(omega) - 1.0) * torch.maximum(b.unsqueeze(2), b.unsqueeze(1))
    return pair, torch.where(pair, w, torch.zeros_like(w))


def rank_loss_groups(
    scores: Tensor, G: Tensor, valid: Tensor, *, k: int, omega: float
) -> RankLoss:
    """PREREG §2.1 over padded groups. ``scores`` ``[n_groups, L]`` (the only tensor
    with a graph), ``G`` ``[n_groups, L]`` (detached here), ``valid`` bool."""
    if scores.shape != G.shape or scores.shape != valid.shape:
        raise ValueError(
            f"scores {tuple(scores.shape)}, G {tuple(G.shape)}, valid "
            f"{tuple(valid.shape)}: one [n_groups, L] shape"
        )
    valid = valid.to(torch.bool)
    pair, w = pair_weights(G, valid, k, omega)
    w = w.to(scores.dtype)
    # sp[g, i, j] = softplus(s_j - s_i): i is the slot with the larger G.
    sp = F.softplus(scores.unsqueeze(1) - scores.unsqueeze(2))
    num = torch.where(pair, w * sp, torch.zeros_like(sp)).sum((-1, -2))
    den = w.sum((-1, -2))
    has = pair.flatten(1).any(-1)
    per = torch.where(has, num / den.clamp(min=1e-30), torch.full_like(num, float("nan")))
    n_with = int(has.sum())
    loss = per[has].mean() if n_with else None
    return RankLoss(
        loss=loss,
        per_group=per,
        has_pair=has,
        n_groups=int(scores.shape[0]),
        n_groups_with_pair=n_with,
        n_groups_no_pair=int(scores.shape[0]) - n_with,
        n_pairs=int(pair.sum()),
    )


def argmin_agreement(scores: Tensor, G: Tensor, valid: Tensor) -> float | None:
    """Fraction of groups with ``|L| >= 2`` whose argmin score is a slot of minimal
    ``G`` (T5's within-group argmin agreement). None if no such group."""
    valid = valid.to(torch.bool)
    keep = valid.sum(-1) >= 2
    if not bool(keep.any()):
        return None
    inf_s = torch.full_like(scores, float("inf"))
    inf_g = torch.full_like(G, float("inf"))
    s = torch.where(valid, scores.detach(), inf_s)[keep]
    g = torch.where(valid, G.detach(), inf_g)[keep]
    pick = s.argmin(-1, keepdim=True)
    ok = g.gather(-1, pick).squeeze(-1) == g.min(-1).values
    return float(ok.double().mean())
