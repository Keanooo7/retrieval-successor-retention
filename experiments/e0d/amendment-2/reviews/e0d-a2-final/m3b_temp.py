"""AMD2 OI-2b: per-step temperature on a fixed within-step score (within-step order unchanged)."""
import json, numpy as np
HERE="/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2-final/"
src=open(HERE+"m2_power.py").read(); exec(src[:src.index("res={}")])
out={}; rng3=np.random.default_rng(11)
for pname,keep in (("bos_excluded",bos),("all_cells",np.ones(len(d),bool))):
    Pp=build(keep); e=Pp["elig"]; has=(Pp["kt"]>=1)[:,None]; crit=np.nan_to_num(Pp["y"].astype(float))
    eps=rng3.standard_normal(e.shape)
    ones=np.ones(64); Wb=rng3.multinomial(1024,np.ones(64)/64,size=200).astype(float)
    for a in (0.5,1.0):
        base=a*crit+eps
        for nm,T in (("T=1 everywhere",np.ones_like(has,float)),
                     ("flat(T=.2) on critical steps, peaky(T=3) on k0 steps",np.where(has,0.2,3.0)),
                     ("flat(T=.05) on critical steps, peaky(T=6) on k0 steps",np.where(has,0.05,6.0)),
                     ("peaky(T=3) on critical steps, flat(T=.2) on k0 steps",np.where(has,3.0,0.2))):
            rg=np.where(e,softmax_steps(T*base,e),np.nan); pt=stats(Pp,rg,ones); bs=[stats(Pp,rg,w) for w in Wb]
            ent={k:round(float(pt[k]),4) for k in ("AUROC_strat","companion","H","R_H")}
            ent["AUROC_strat_sd1024"]=round(float(np.nanstd([b["AUROC_strat"] for b in bs])),4)
            out[f"{pname}.a={a}.{nm}"]=ent
print(json.dumps(out,indent=1))
