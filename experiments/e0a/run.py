"""E0a -- muP coordinate check.

The spec requires this run **twice**: bare TG, then TG with the value head attached, and it requires
checking ``psi-hat``'s **scalar output** coordinate scale specifically, not only transformer
activations. §4.3: "this is the failure that surfaces in week 9 with no error message."

**This file runs half of E0a.** The bare-TG half is blocked on ADR-0001 (no TG implementation is
available and one is being built from the paper). The value-head half is independent of TG and runs
now, because it is where the novel muP prescription lives -- the transformer's parameterization is
[P4]-standard, the bilinear head's is not.

Probe: fit ``psi-hat`` to a Theta(1) per-slot regression target under MSE. That is the shape of
``L_MC`` (§3.3), so the coordinate scales measured here are the ones training will actually produce.

Usage:  uv run python experiments/e0a/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from rsr.mup.coord_check import coord_check, width_invariance
from rsr.retention.value_head import BilinearValueHead

WIDTHS = (128, 192, 256, 384)
N_LIVE = 40  # M on the corpora
STEPS = 8
SEEDS = (0, 1, 2, 3, 4)
SEED = SEEDS[0]


def build(d: int):
    torch.manual_seed(SEED)
    head = BilinearValueHead(d, base_d=WIDTHS[0])

    # Fixed inputs and targets per width, drawn from the same seed so widths are comparable.
    # Coordinates are Theta(1), matching what a muP-parameterized transformer emits.
    g = torch.randn(N_LIVE, d)
    c = torch.randn(d)
    target = torch.randn(N_LIVE)

    state = {"prev": None}

    def forward(m: torch.nn.Module) -> dict[str, torch.Tensor]:
        out = m(g, c)
        h = m.W @ c
        delta = out - state["prev"] if state["prev"] is not None else torch.zeros_like(out)
        state["prev"] = out.detach().clone()
        return {
            "output": out - target,  # the residual is what the loss sees
            "psi": out,  # the scalar the spec names
            "hidden_Wc": h,  # the Theta(1) intermediate
            "delta_psi": delta,  # per-step change -- the muP quantity
        }

    return head, forward


def main() -> None:
    records = coord_check(build, WIDTHS, steps=STEPS, lr=1e-2, seed=SEED, base_d=WIDTHS[0])
    here = Path(__file__).parent

    rows = [vars(r) for r in records]
    (here / "coord_records.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"E0a (value-head half) -- widths {WIDTHS}, n_live={N_LIVE}, {STEPS} AdamW steps\n")
    for name in ("psi", "hidden_Wc", "delta_psi"):
        print(f"  {name}")
        header = "    step  " + "".join(f"d={d:<10}" for d in WIDTHS) + "  drift"
        print(header)
        for step in range(STEPS + 1):
            cells = []
            for d in WIDTHS:
                r = next(
                    (x for x in records if x.name == name and x.step == step and x.width == d),
                    None,
                )
                cells.append(f"{r.rms:<12.5f}" if r else f"{'-':<12}")
            try:
                drift = width_invariance(records, name, step)
                dcell = f"{drift:6.3f}"
            except ValueError:
                dcell = "     -"
            flag = ""
            if name == "psi" and step == 0:
                flag = "   <- init regime: Theta(1/sqrt(d)) BY CONSTRUCTION, not a violation"
            print(f"    {step:<4}  " + "".join(cells) + dcell + flag)
        print()

    verdict_step = STEPS
    drift = width_invariance(records, "psi", verdict_step)
    init_drift = width_invariance(records, "psi", 0)
    summary = {
        "widths": list(WIDTHS),
        "steps": STEPS,
        "psi_drift_at_init": init_drift,
        "psi_drift_at_verdict_step": drift,
        "verdict_step": verdict_step,
        "note": (
            "Verdict is read at t>=1. Step 0 is recorded, never gated on: with the 1/d multiplier "
            "the bilinear output is Theta(1/sqrt(d)) at init by construction (uncorrelated s, Wc) "
            "and Theta(1) once correlated. See src/rsr/mup/coord_check.py."
        ),
    }
    (here / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  psi drift at init  : {init_drift:.3f}   (recorded, NOT gated)")
    print(f"  psi drift at step {verdict_step}: {drift:.3f}   <- the E0a number")


if __name__ == "__main__":
    main()
