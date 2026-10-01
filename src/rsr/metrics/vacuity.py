"""E2 -- the vacuity gate (spec section 7.1). **The most likely failure mode.**

**Gate on the full eviction score `z(psi_hat) + b - nu * ...`, not on `psi_hat`
alone.** Given the `u_bar` age confound (section 3.5), gating on `psi_hat`
measures the wrong object.

Three tests, all cheap:

1. **Partial correlation**, not raw rho: does content carry information about
   future retrieval *beyond* age? **Gate at rho < 0.7.** v0.1's 0.9 was far too
   permissive -- Spearman 0.85 with age is essentially recency.
2. **Age-only value head** as an explicit arm. If content+age ~= age-only, content
   contributes nothing, in one cheap run.
3. **RSR must beat LRU.** LRU *is* the recency policy.

Report the number regardless of outcome.

## What this module computes, and what it does NOT decide

ADR-0009 L6 (signed, `R-2026-09-30-adr0009-signoff`) makes the section 7.1
diagnostics mandatory in every RESULTS. This module is the **statistic**, logged
at run time by `RSRPolicy.step_diagnostics`; it is not a gate. The 0.7 threshold
is the spec's, and no code here compares against it -- a gate is a PREREG's to
declare, before the run it governs.

**The operational definition used here** (the build brief's reading, item 8 of
run-1 readiness): `partial rho(score, age | content)` -- the Spearman correlation
of the full eviction score with slot age, after both ranks are residualised on the
slots' content by least squares. Content is supplied by the caller; at run time
it is the slot gestalts `s_i` (`d` columns plus an intercept). Two properties a
reader must carry into any RESULTS:

* The residualisation is **linear** in the content columns. `psi_hat` is bilinear
  in `(s_i, c_t)` and `c_t` changes per decision, so a purely content-driven score
  is not fully absorbed by a linear fit on `s_i` alone: some residual age
  correlation can survive for a score that never read age. Read the number as an
  upper bound on the age dependence content does not explain *linearly*.
* Raw Spearman `rho(score, age)` (the v0.1 quantity) is logged beside it, never in
  its place.

Whether this is the statistic run 1's PREREG gates on is the PREREG's decision,
not this module's.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "partial_correlation_with_age",
    "partial_spearman",
    "rankdata",
    "spearman",
]


def rankdata(x: Tensor) -> Tensor:
    """1-based ranks, ties given their **average** rank, as float64 on CPU.

    `rankdata([10, 20, 20, 30]) == [1, 2.5, 2.5, 4]` -- scipy's `method="average"`,
    written out so torch is the only dependency (CLAUDE.md: PyTorch only).
    """
    v = x.detach().reshape(-1).to("cpu", torch.float64)
    n = v.numel()
    order = torch.argsort(v, stable=True)
    _, inverse, counts = torch.unique(v[order], return_inverse=True, return_counts=True)
    first = torch.cumsum(counts, 0) - counts  # 0-based start of each tie block
    avg = first.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = avg[inverse]
    return ranks


def partial_spearman(x: Tensor, y: Tensor, covariates: Tensor | None = None):
    """Spearman `rho(x, y | covariates)`, or None where it is undefined.

    `x` and `y` are rank-transformed (`rankdata`); both rank vectors are
    residualised on `[1, covariates]` by least squares (float64), and the result
    is the Pearson correlation of the two residuals. With `covariates=None` it is
    plain Spearman. For a single covariate whose values are already ranks this is
    the textbook first-order partial
    `(r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))`.

    **None, never a number, when:** fewer than `k + 2` observations for `k`
    regressors (no residual degree of freedom), or either residual is zero --
    e.g. the covariates determine age exactly, so nothing of age is left to
    correlate. A missing measurement must look like one.
    """
    rx, ry = rankdata(x), rankdata(y)
    n = rx.numel()
    if ry.numel() != n:
        raise ValueError(f"x has {n} values, y has {ry.numel()}")
    ones = torch.ones(n, 1, dtype=torch.float64)
    if covariates is None:
        Z = ones
    else:
        z = covariates.detach().to("cpu", torch.float64).reshape(n, -1)
        Z = torch.cat([ones, z], dim=1)
    if n < Z.shape[1] + 2:
        return None
    Y = torch.stack([rx, ry], dim=1)
    beta = torch.linalg.lstsq(Z, Y).solution
    res = Y - Z @ beta
    ex, ey = res[:, 0], res[:, 1]
    sx, sy = float(ex.pow(2).sum()), float(ey.pow(2).sum())
    # Relative to the ranks' own spread: a residual that is round-off is zero.
    tol = 1e-12 * float(n) ** 3
    if sx <= tol or sy <= tol:
        return None
    return float((ex * ey).sum() / (sx * sy) ** 0.5)


def spearman(x: Tensor, y: Tensor):
    """Plain Spearman rho: `partial_spearman` with no covariates."""
    return partial_spearman(x, y, None)


def partial_correlation_with_age(score: Tensor, age: Tensor, content: Tensor):
    """Section 7.1 item 1, as defined in the module docstring:
    `rho(score, age | content)`, `score` the FULL eviction score of each live slot
    at each decision, `age` its age, `content` one row of covariates per slot.
    None where undefined. A statistic, not a gate."""
    return partial_spearman(score, age, content)
