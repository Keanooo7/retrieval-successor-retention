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

## The prescription, for AdamW

With width multiplier `m = d_model / base_width`:

| Parameter shape | Group | LR |
|---|---|---|
| matrix-like (`ndim >= 2`), hidden | `transformer.hidden` | `base_lr / m` |
| vector-like (biases, norm gains) | `transformer.vector` | `base_lr` |
| embedding / readout by name | `transformer.vector` | `base_lr` |
| `phi = {W, u}` | **`value_head`** | `base_lr / m` |

Both value-head parameters scale as `1/m`: `W` has `fan_in = d`, `u` has
`fan_in = 2d`, and both are `base_width`-proportional, so the ratio to the base
configuration is `m` either way. That is what the head's own table means by "both
LRs follow their muP rule, ~ 1/d".

🔴 **The value head is in its own group, and that is the bill gauntlet 1.6 names.**
Not because its LR differs from the transformer's hidden group today -- it does not
-- but because the two are governed by different prescriptions and will diverge the
moment either changes. Sharing a group makes that divergence invisible: it surfaces
in week 9 as a transfer failure with no error message. The group is also what E0a
reads when it checks `psi_hat`'s **scalar output** coordinate scale specifically,
rather than only transformer activations.
"""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["VALUE_HEAD_GROUP", "build_param_groups"]

VALUE_HEAD_GROUP = "value_head"

_VECTOR_LIKE_NAMES = ("embed", "wte", "wpe", "readout", "lm_head", "unembed")


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

    Returns groups tagged with `name`, so a run config can be dumped and read
    (gauntlet 3.5) rather than trusted.
    """
    if d_model <= 0 or base_width <= 0:
        raise ValueError(f"d_model={d_model} and base_width={base_width} must be > 0")
    m = d_model / base_width

    head_params = set()
    if value_head is not None:
        head_params = {id(p) for p in value_head.parameters()}

    hidden: list[nn.Parameter] = []
    vector: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in head_params:
            continue
        lowered = name.lower()
        if p.ndim >= 2 and not any(k in lowered for k in _VECTOR_LIKE_NAMES):
            hidden.append(p)
        else:
            vector.append(p)

    groups: list[dict[str, Any]] = []
    if hidden:
        groups.append({"name": "transformer.hidden", "params": hidden, "lr": base_lr / m})
    if vector:
        groups.append({"name": "transformer.vector", "params": vector, "lr": base_lr})

    if value_head is not None:
        params = [p for p in value_head.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "the value head has no trainable parameters; section 3.3 makes "
                "phi = {W, u} the only thing that receives gradient, so an empty "
                "group means the retention loss has nowhere to go."
            )
        groups.append({"name": VALUE_HEAD_GROUP, "params": params, "lr": base_lr / m})

    if not groups:
        raise ValueError("no trainable parameters found")
    return groups
