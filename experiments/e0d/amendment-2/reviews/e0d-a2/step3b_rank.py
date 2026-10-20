"""E0D-A2 step 3b: rho_Q,rank (within write-order rank, Q-cells) for the same synthetic
scores as step 3. Delta = real LOO (step 1); scores synthetic only; no arm-B r_i."""
import runpy, sys, json
from pathlib import Path
import numpy as np
g = runpy.run_path(str(Path(__file__).with_name("step3_candidates.py")).replace(".py", "_lib.py"))
m, c, d, Q, M, W = g["m"], g["c"], g["d"], g["Q"], g["M"], g["W"]
out = {}
for nm, r in g["scores"].items():
    have = Q & ~np.isnan(d) & ~np.isnan(r)
    rk = m.RankStrata(r[have], d[have], c["rank"][have], n_ranks=M)
    dd = c["doc"][have]
    bs = np.array([rk(w[dd])["rho"] for w in W])
    out[nm] = {"pt": rk()["rho"], "sd1024": float(np.std(bs) * g["SCALE"])}
    print(nm.ljust(30), round(out[nm]["pt"], 4), round(out[nm]["sd1024"], 4), flush=True)
Path(__file__).with_name("step3b_out.json").write_text(json.dumps(out, indent=1))
