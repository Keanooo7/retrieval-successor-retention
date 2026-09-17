"""muP parameter groups (spec section 4.3).

**Three bills, all of which fail silently.**

1. The value head is readout-like -- and "readout-like" is not a prescription for
   a bilinear form (defect C6). The explicit table lives in
   `rsr.retention.value_head.BilinearValueHead`, with the correct scaling
   argument and an explicit warning against the wrong one.
2. **muP transfers across width only.** `M`, `S` and depth are non-width axes and
   must be fixed within each comparison (section 5.1). `rsr.constants` enforces
   the `M`/`S` half of this by refusing scope-free reads.
3. **Recurrence through a retained graph is untested territory for muP.** E0a is a
   check for this configuration at these widths, not a general result. Run it
   twice -- bare TG, then TG with the value head attached.

**Rejected: Muon.** muP and Muon are both width-correct parameterizations;
stacking them double-corrects, and the matrix-method advantage under equally-tuned
baselines is small and shrinking with scale. One correction: muP + AdamW.
"""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["build_param_groups"]


def build_param_groups(
    model: nn.Module,
    value_head: nn.Module | None,
    *,
    base_lr: float,
    d_model: int,
    base_width: int = 128,
) -> list[dict[str, Any]]:
    """AdamW parameter groups under muP.

    The value head lives in **its own group**, separate from the transformer
    (section 4.3). Mis-grouping is the failure that surfaces in week 9 with no
    error message, which is why E0a checks `psi_hat`'s scalar output coordinate
    scale specifically rather than only transformer activations.
    """
    raise NotImplementedError(
        "Sprint 2. Spec section 4.3. The prescription is written out in "
        "rsr.retention.value_head.BilinearValueHead's docstring; this function "
        "applies it to an optimizer."
    )
