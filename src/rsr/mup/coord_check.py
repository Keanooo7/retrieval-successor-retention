"""muP coordinate check (E0a).

The check, per [P4]: train for a few optimizer steps at several widths and record the RMS of the
quantities you care about at each step. Under muP those coordinate scales are **width-invariant
during training**. Under standard parameterization they blow up or vanish with width.

⚠️ **The check is on the trained regime, not on init, and for this head that distinction is not
pedantic -- it inverts the result.**

§4.3 reads ``psi-hat = s_i^T W c_t`` as ``h = W c_t`` (Theta(1) coordinates) followed by a dot
product of two d-dimensional Theta(1) vectors. That dot product is:

* **Theta(sqrt(d)) at init**, because ``s`` and ``h`` are uncorrelated; with the ``1/d`` multiplier
  the scalar output is therefore ``Theta(1/sqrt(d))`` -- it **shrinks with width, at init, and that
  is correct**.
* **Theta(d) after training**, once ``s``, ``W`` and ``c`` are correlated; with the ``1/d``
  multiplier the output is ``Theta(1)`` -- width-invariant, which is what muP promises.

So a coordinate check run at step 0 will show this head's output falling as ``1/sqrt(d)`` and look
like a muP violation. It is not. Anyone who "fixes" it by changing the multiplier to ``1/sqrt(d)``
has tuned the head for the init regime and broken it for the trained one -- which is precisely the
failure the spec's §4.3 footnote warns about, met from the other side.

**Therefore: E0a's verdict is read at ``t >= 1``, and step 0 is recorded but never gated on.**
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["CoordRecord", "coord_check", "width_invariance"]


@dataclass
class CoordRecord:
    width: int
    step: int
    name: str
    rms: float


def _rms(t: Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def coord_check(
    build: Callable[[int], tuple[torch.nn.Module, Callable[[torch.nn.Module], dict[str, Tensor]]]],
    widths: Sequence[int],
    *,
    steps: int = 4,
    lr: float = 1e-2,
    seed: int = 0,
    base_d: int = 128,
) -> list[CoordRecord]:
    """Run the check.

    ``build(width)`` returns ``(module, forward)`` where ``forward(module)`` runs one step's
    computation and returns the named tensors to measure. The loss is taken as the sum of squares of
    whichever tensor is named ``"output"``, so the same harness works for a bare head and for a
    head attached to a model.

    ``steps=0`` records init only -- useful for characterising the init regime, never for a verdict.
    """
    out: list[CoordRecord] = []
    for d in widths:
        torch.manual_seed(seed)
        module, forward = build(d)
        groups = (
            module.mup_param_groups(lr)
            if hasattr(module, "mup_param_groups")
            else [{"params": list(module.parameters()), "lr": lr * base_d / d}]
        )
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

        for step in range(steps + 1):
            tensors = forward(module)
            for name, t in tensors.items():
                out.append(CoordRecord(width=d, step=step, name=name, rms=_rms(t)))
            if step == steps:
                break
            loss = tensors["output"].pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return out


def width_invariance(records: Sequence[CoordRecord], name: str, step: int) -> float:
    """``max |rms/rms_at_smallest_width - 1|`` over widths, for one quantity at one step.

    This is the number the ratchet tracks: ``mup_coord_drift``, direction **may fall, never rise**.
    0.0 is perfect invariance. A value of 0.15 means some width's coordinate scale is 15% away from
    the base width's.
    """
    rows = sorted(
        (r for r in records if r.name == name and r.step == step), key=lambda r: r.width
    )
    if len(rows) < 2:
        raise ValueError(f"need >=2 widths for {name!r} at step {step}, got {len(rows)}")
    base = rows[0].rms
    if base == 0.0:
        raise ValueError(f"{name!r} at step {step} has zero RMS at the base width; cannot ratio")
    return max(abs(r.rms / base - 1.0) for r in rows)
