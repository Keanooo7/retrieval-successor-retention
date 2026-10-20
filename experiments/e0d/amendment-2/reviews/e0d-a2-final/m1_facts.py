"""AMD2 fact check on set E [64,128) seed 0 cells (step1_cells.npz). No r_i, no model."""
import hashlib, json, numpy as np
P="/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2/step1_cells.npz"
sha=hashlib.sha256(open(P,"rb").read()).hexdigest()
c=dict(np.load(P)); M=16
out={"sha256":sha,"keys":sorted(c)}
full=c["tfull"].astype(bool); q=c["q"].astype(bool); a=c["a"].astype(bool)
Q=full&q; A=Q&a
age=c["t"]-c["sent"]
out["age_range_full"]=[int(age[full].min()),int(age[full].max())]
out["rank_eq_16_minus_age_all_full"]=bool(np.all(c["rank"][full]==M-age[full]))
key=c["doc"]*49+c["t"]
# multiplicity: per full step, count of each age
fk=key[full]; fa=age[full]
u,inv=np.unique(fk,return_inverse=True)
cnt=np.zeros((len(u),M+1),int); np.add.at(cnt,(inv,fa),1)
out["full_steps"]=int(len(u)); out["max_slots_per_age_per_step"]=int(cnt.max())
out["steps_with_exactly_one_per_age_1_16"]=int(np.all(cnt[:,1:]==1,axis=1).sum())
# tau
d=c["dres"]; off=Q&~A&~np.isnan(d)
ad=np.abs(d[off]); out["n_tau_pop"]=int(off.sum())
out["tau"]={str(qq):float(np.quantile(ad,qq)) for qq in (0.99,0.995,0.999)}
# doc-bootstrap CI of tau (2000, seed 20260927)
rng=np.random.default_rng(20260927); docs=np.unique(c["doc"]); dd=c["doc"][off]
bydoc=[ad[dd==x] for x in docs]
bt={str(qq):[] for qq in (0.99,0.995,0.999)}
for _ in range(2000):
    s=np.concatenate([bydoc[i] for i in rng.integers(0,len(docs),len(docs))])
    for qq in (0.99,0.995,0.999): bt[str(qq)].append(np.quantile(s,qq))
out["tau_boot95"]={k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in bt.items()}
# labels + harm baselines per tau, primary and bos-excluded
def summarize(tau,keep):
    sel=Q&keep
    y=(d>tau)
    k=key[sel]; uu,ii=np.unique(k,return_inverse=True)
    nanstep=np.zeros(len(uu),bool); np.logical_or.at(nanstep,ii,np.isnan(d[sel]))
    ncell=np.bincount(ii,minlength=len(uu))
    kt=np.bincount(ii,weights=y[sel].astype(float),minlength=len(uu))
    crit=sel&y&~np.isnan(d)
    r={"n_Q_steps":int(len(uu)),"cells_per_step_set":sorted(set(ncell.tolist())),
       "n_Q_cells_with_value":int((sel&~np.isnan(d)).sum()),
       "n_crit":int(crit.sum()),"crit_share_of_Qcells":float(crit.sum()/(sel&~np.isnan(d)).sum()),
       "crit_that_are_A":int((crit&A).sum()),"A_with_value":int((sel&A&~np.isnan(d)).sum()),
       "A_crit":int((sel&A&y&~np.isnan(d)).sum()),
       "steps_with_any_nan":int(nanstep.sum())}
    qc=kt>=1
    r["n_Qcrit_steps"]=int(qc.sum()); r["n_Qcrit_complete"]=int((qc&~nanstep).sum())
    r["k_hist"]={str(x):int((kt==x).sum()) for x in range(0,6)}
    r["H_random_mean_k_over_M"]=float((kt[qc]/M).mean())
    r["H_random_mean_k_over_n_cells"]=float((kt[qc]/ncell[qc]).mean())
    r["crit_per_age"]={str(x):int((crit&(age==x)).sum()) for x in range(1,M+1)}
    r["cells_per_age"]={str(x):int((sel&~np.isnan(d)&(age==x)).sum()) for x in range(1,M+1)}
    r["FIFO_harm_age16"]=float(np.mean([0]) if False else (crit&(age==16)).sum()/qc.sum())
    return r
keepall=np.ones(len(d),bool)
bos=(c["rank"]!=M-1)&~(q&(c["gap"]==1))
out["by_tau"]={}
for qq,tau in out["tau"].items():
    out["by_tau"][qq]={"primary":summarize(tau,keepall),"bos_excluded":summarize(tau,bos)}
out["A_gap_hist"]={str(g):int((A&(c["gap"]==g)).sum()) for g in range(1,17)}
print(json.dumps(out,indent=1))
