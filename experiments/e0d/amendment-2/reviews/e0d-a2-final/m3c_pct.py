"""AMD2 OI-2c: AUROC_strat on within-step percentiles (scale-free option). Synthetic scores only."""
import json, numpy as np
HERE="/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2-final/"
src=open(HERE+"m2_power.py").read(); exec(src[:src.index("res={}")])
def pctgrid(rg,e):
    out=np.full(rg.shape,np.nan)
    for j in range(rg.shape[0]):
        ok=e[j]&~np.isnan(rg[j]); v=rg[j,ok]
        uv,ix,cnt=np.unique(v,return_inverse=True,return_counts=True); cs=np.cumsum(cnt)-cnt
        out[j,ok]=((cs+(cnt+1)/2)[ix]-1)/(ok.sum()-1)
    return out
out={}; rng3=np.random.default_rng(11)
for pname,keep in (("bos_excluded",bos),):
    Pp=build(keep); e=Pp["elig"]; has=(Pp["kt"]>=1)[:,None]; crit=np.nan_to_num(Pp["y"].astype(float))
    eps=rng3.standard_normal(e.shape); ones=np.ones(64)
    Wb=rng3.multinomial(1024,np.ones(64)/64,size=200).astype(float)
    A_grid=np.zeros(e.shape); A_grid[Pp["inv"],c["rank"][Pp["sel"]]]=Acell[Pp["sel"]]
    rep=np.full(e.shape,np.nan); rep[Pp["inv"],c["rank"][Pp["sel"]]]=rep1[Pp["sel"]]
    cases={"PC":A_grid+0.0,"ORACLE":np.nan_to_num(rep),"pure step-level (SC1b)":softmax_steps(np.where(has,0.0,50.0)*eps,e)}
    for a in (0.5,1.0,1.5):
        cases[f"a={a} T=1"]=softmax_steps(a*crit+eps,e)
        cases[f"a={a} flat-crit/peaky-k0"]=softmax_steps(np.where(has,0.05,6.0)*(a*crit+eps),e)
    for nm,rg in cases.items():
        rg=np.where(e,rg,np.nan); pg=pctgrid(rg,e)
        a1=stats(Pp,rg,ones)["AUROC_strat"]; a2=stats(Pp,pg,ones)["AUROC_strat"]
        sd2=float(np.nanstd([stats(Pp,pg,w)["AUROC_strat"] for w in Wb[:100]]))
        out[f"{pname}.{nm}"]={"AUROC_strat":round(float(a1),4),"AUROC_strat_pct":round(float(a2),4),"pct_sd1024":round(sd2,4)}
print(json.dumps(out,indent=1))
