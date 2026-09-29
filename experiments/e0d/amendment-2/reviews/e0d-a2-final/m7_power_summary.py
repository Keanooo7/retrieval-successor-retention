"""E0D-AMD2-FINAL: single-gate power summary, arithmetic on m6_out_012.json only (no data read)."""
import json
from pathlib import Path
from statistics import NormalDist
N = NormalDist(); HERE = Path(__file__).resolve().parent
o = json.loads((HERE / "m6_out_012.json").read_text()); A = o["A_STAR"]
z1 = N.inv_cdf(0.8) + 1.96                    # one seed, 80% power
z3 = N.inv_cdf(0.8 ** (1 / 3)) + 1.96          # all three seeds, 80% joint, equal truths
out = {"z_one_seed_80": z1, "z_three_seed_80": z3, "per_seed": {}}
for s in (0, 1, 2):
    c = o[f"seed{s}"]["cases"]
    near = {k: v["pct_sd_n1024"] for k, v in c.items() if k.startswith("content a=") and "+" not in k and "," not in k and "rescaled" not in k and 0.75 <= v["AUROC_strat_pct"] <= 0.95}
    sds = [v["pct_sd_n1024"] for v in c.values() if v["pct_sd_n1024"] > 0]
    out["per_seed"][s] = {"sd_near_bar_content": near, "sd_max_any": max(sds),
        "detect80_one_seed_sd_near_max": A + z1 * max(near.values()),
        "detect80_three_seed_sd_near_max": A + z3 * max(near.values()),
        "detect80_three_seed_sd_max_any": A + z3 * max(sds),
        "content_a1.5_P1": c["content a=1.5"]["P(CI_lo>=0.85) one seed"]}
p = 1.0
for s in (0, 1, 2): p *= o[f"seed{s}"]["cases"]["content a=1.5"]["P(CI_lo>=0.85) one seed"]
out["content_a1.5_three_seed"] = p
out["false_pass_one_seed_at_bar"] = 0.025; out["false_pass_three_seeds_indep"] = 0.025 ** 3
print(json.dumps(out, indent=1))
