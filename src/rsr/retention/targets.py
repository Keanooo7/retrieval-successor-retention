"""Names for the retention TARGET `psi_hat` regresses on (owner ruling C3).

Kept import-free on purpose: `RSRConfig` needs these names, and the module that
computes the LOO target (`rsr.retention.loo_target`) imports `rsr.metrics.loo`,
which imports `rsr.train.loop`, which imports `rsr.retention.rsr`.
"""

from __future__ import annotations

__all__ = [
    "STATUS_EMPTY",
    "STATUS_NO_DONOR",
    "STATUS_NO_TARGET",
    "STATUS_OK",
    "TARGETS",
    "TARGET_LOO_RESAMPLE",
    "TARGET_R_I",
]

TARGET_R_I = "r_i"
"""Run 1's target: the gated (correction 17), eval-mode (correction 20), share-of-M
rescaled (section 3.2.1) `r_i`. The default, so run 1 reproduces unchanged."""

TARGET_LOO_RESAMPLE = "loo_resample"
"""Owner ruling C3 (R-2026-10-07-c3-loo-target): E0d's resample-knockout `Delta`,
measured on-policy (`rsr.retention.loo_target`)."""

TARGETS = (TARGET_R_I, TARGET_LOO_RESAMPLE)

STATUS_EMPTY = -1
STATUS_OK = 0
STATUS_NO_DONOR = 1
"""No same-kind, same-rank, different-document donor at this step: NaN, imputed 0."""
STATUS_NO_TARGET = 2
"""The sentence has no real next-token target (padding): NaN, imputed 0."""
