"""E0a -- muP coordinate check (value-head half).

§4.3 requires checking **`psi-hat`'s scalar output** coordinate scale specifically, not only
transformer activations: "this is the failure that surfaces in week 9 with no error message."

🔴 **Read the verdict at `t >= 1`, never at step 0.** With the `1/d` multiplier the bilinear output
is `Theta(1/sqrt(d))` at init by construction (uncorrelated `s`, `Wc`) and `Theta(1)` once
correlated. A check read at init shows a *correct* parameterization shrinking with width.

⚠️ **This measures the Theta(1)-coordinate regime. TG's real gestalts are L2-normalised to unit
norm** (`srep_norm_target = 1.0`), so their coordinates are `O(1/sqrt(d))`. See
docs/spec-corrections.md D-15. This file confirms §4.3's *reasoning*; it does not confirm its
*prescription* for TG's actual inputs.

📌 Two defects this file has already carried, both of which produced confident wrong numbers:
  1. `SEEDS` was declared and then `SEED = SEEDS[0]` was used, with no loop -- so RESULTS.md
     reported a five-seed aggregate no committed code path could produce.
  2. `build()` called `torch.manual_seed(SEED)` on the module constant, clobbering the per-seed
     seeding, so five seeds gave five byte-identical results. **The reported `sd` of 0.0000 is
     what exposed it** -- which is the argument for always reporting spread, not just a mean.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from rsr.mup.coord_check import coord_check, width_invariance
from rsr.retention.value_head import BilinearValueHead

WIDTHS = (128, 192, 256, 384)
N_LIVE = 40          # M on the corpora
STEPS = 8
SEEDS = (0, 1, 2, 3, 4)


def make_build(seed: int):
    """Factory closing over `seed`, so each seed genuinely differs."""

    def build(d: int):
        torch.manual_seed(seed)
        head = BilinearValueHead(d, base_d=WIDTHS[0])
        g = torch.randn(N_LIVE, d)
        c = torch.randn(d)
        target = torch.randn(N_LIVE)

        def forward(m: torch.nn.Module) -> dict[str, torch.Tensor]:
            out = m(g, c)
            return {"output": out - target, "psi": out, "hidden_Wc": m.W @ c}

        return head, forward

    return build


def _agg(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows]
    return {
        "mean": statistics.mean(v),
        "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
        "min": min(v), "max": max(v), "n": len(v),
    }


def main() -> None:
    here = Path(__file__).parent
    per_seed: list[dict] = []

    for sd in SEEDS:
        recs = coord_check(make_build(sd), WIDTHS, steps=STEPS, lr=1e-2,
                           seed=sd, base_d=WIDTHS[0])
        base = next(r.rms for r in recs
                    if r.name == "psi" and r.step == 0 and r.width == WIDTHS[0])
        wide = next(r.rms for r in recs
                    if r.name == "psi" and r.step == 0 and r.width == WIDTHS[-1])
        per_seed.append({
            "seed": sd,
            "drift_at_init": width_invariance(recs, "psi", 0),
            "drift_at_verdict": width_invariance(recs, "psi", STEPS),
            "init_rms_ratio_wide_over_base": wide / base,
        })

    summary = {
        "seeds_actually_run": list(SEEDS),
        "n_seeds": len(SEEDS),
        "widths": list(WIDTHS),
        "steps": STEPS,
        "verdict_step": STEPS,
        "per_seed": per_seed,
        "drift_at_init": _agg(per_seed, "drift_at_init"),
        "drift_at_verdict": _agg(per_seed, "drift_at_verdict"),
        "init_rms_ratio_wide_over_base": _agg(per_seed, "init_rms_ratio_wide_over_base"),
        "predicted_init_ratio": (WIDTHS[0] / WIDTHS[-1]) ** 0.5,
        "note": (
            "Verdict read at t>=1. Step 0 is recorded, never gated on: with the 1/d multiplier the "
            "bilinear output is Theta(1/sqrt(d)) at init by construction. Inputs here are "
            "Theta(1)-coordinate; TG's real gestalts are unit-norm (spec-corrections D-15)."
        ),
    }
    (here / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"E0a value-head half -- widths {WIDTHS}, n_live={N_LIVE}, {STEPS} steps, "
          f"{len(SEEDS)} seeds\n")
    print("  seed   drift@init   drift@t=8   init ratio d384/d128")
    for r in per_seed:
        print(f"  {r['seed']:>4}   {r['drift_at_init']:>10.4f}  {r['drift_at_verdict']:>10.4f}"
              f"   {r['init_rms_ratio_wide_over_base']:>10.4f}")
    a = summary["init_rms_ratio_wide_over_base"]
    v = summary["drift_at_verdict"]
    print(f"\n  init ratio : mean {a['mean']:.4f}  sd {a['sd']:.4f}  n={a['n']}"
          f"   vs {summary['predicted_init_ratio']:.4f} predicted by Theta(1/sqrt(d))")
    print(f"  drift @t=8 : mean {v['mean']:.4f}  sd {v['sd']:.4f}")


if __name__ == "__main__":
    main()
