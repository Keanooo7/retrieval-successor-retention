"""AMD2 open issues 1-2, set E [64,128) seed 0. Structure from the generator (set E, already
inspected; reserved range untouched). Delta real; scores SYNTHETIC. No r_i."""
import json, sys, numpy as np
sys.path.insert(0,"/Users/keanooo7/retrieval-successor-retention/.worktrees/i5-flake/src")
from rsr.data.synthetic import SyntheticConfig, _generate_document
HERE="/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2-final/"
src=open(HERE+"m2_power.py").read(); exec(src[:src.index("res={}")])   # build, wauc, stats, softmax_steps, c, d, ...
LO,HI=64,128; assert HI<=262144
docs=[_generate_document(i,SyntheticConfig(sentences_per_document=48,seed=0,answer_in_stream=True)) for i in range(LO,HI)]
out={}
# ---- OI-1: is a critical cell's sentence needed after t? (generator truth, one query per fact)
qof={}
for b,doc in enumerate(docs):
    for a_,q_ in doc.pairs: qof[(b,a_)]=q_
assert all(len(set(q for (bb,a_),q in qof.items() if bb==b))==len(docs[b].pairs) for b in range(len(docs)))
for pname,keep in (("bos_excluded",bos),("all_cells",np.ones(len(d),bool))):
    sel=full&q&keep&~np.isnan(d)&(d>TAU)
    fut=np.array([qof.get((int(b),int(s)),-1)>int(t) for b,s,t in zip(c["doc"][sel],c["sent"][sel],c["t"][sel])])
    isA=Acell[sel]
    kinds=np.array([docs[int(b)].sentences[int(s)].kind for b,s in zip(c["doc"][sel],c["sent"][sel])])
    out[f"OI1.{pname}"]={"n_critical":int(sel.sum()),"A_cells":int(isA.sum()),
       "needed_after_t(pending_assert)":int(fut.sum()),"frac_needed_after_t":float(fut.mean()),
       "critical_by_kind":{k:int((kinds==k).sum()) for k in np.unique(kinds)}}
    # how many A-cells (consumed at t) vs pending asserts among ALL Q-step cells
    allsel=full&q&keep
    futall=np.array([qof.get((int(b),int(s)),-1)>int(t) for b,s,t in zip(c["doc"][allsel],c["sent"][allsel],c["t"][allsel])])
    out[f"OI1.{pname}"]["pending_assert_cells_in_Q"]=int(futall.sum())
    out[f"OI1.{pname}"]["pending_assert_cells_critical"]=int((futall&(d[allsel]>TAU)).sum())
# ---- OI-2: step-level concentration with NO within-step content
rng2=np.random.default_rng(7)
for pname,keep in (("bos_excluded",bos),("all_cells",np.ones(len(d),bool))):
    Pp=build(keep); e=Pp["elig"]; kt=Pp["kt"]; crit=np.nan_to_num(Pp["y"].astype(float))
    ag=np.nan_to_num(Pp["ag"]).astype(int)
    eps=rng2.standard_normal(e.shape)
    has=(kt>=1)[:,None]
    cases={
     "SC0_null_iid":softmax_steps(eps,e),
     "SC1_flat_if_critical_step_else_peaky":softmax_steps(np.where(has,0.05,4.0)*eps,e),
     "SC1b_extreme(flat vs one-hot)":softmax_steps(np.where(has,0.0,50.0)*eps,e),
     "CONTENT_a1.0_only":softmax_steps(1.0*crit+eps,e),
     "SC1+CONTENT_a1.0":softmax_steps(np.where(has,0.3,3.0)*eps+1.0*crit,e),
     "SC1+CONTENT_a0.5":softmax_steps(np.where(has,0.3,3.0)*eps+0.5*crit,e),
    }
    ones=np.ones(64); Wb=rng2.multinomial(1024,np.ones(64)/64,size=200).astype(float)
    # analytic cap for pure step-level signal: per bin, share of negatives on k=0 steps (pos-weighted)
    cm=Pp["cm"]; yy=Pp["y"]; num=den=0.0
    for bb in range(1,17):
        m=cm&(Pp["ag"]==bb)
        npos=(m&yy).sum(); nneg=(m&~yy).sum()
        if npos==0 or nneg==0: continue
        f=(m&~yy&~has).sum()/nneg
        num+=npos*(0.5+0.5*f); den+=npos
    out[f"OI2.{pname}.pure_step_level_AUROC_cap(0.5+0.5*f_neg_on_k0_steps)"]=num/den
    for nm,rg in cases.items():
        rg=np.where(e,rg,np.nan); pt=stats(Pp,rg,ones); bs=[stats(Pp,rg,w) for w in Wb]
        ent={k:float(pt[k]) for k in ("AUROC_strat","AUROC_unstrat","companion","H","H_age_random","R_H")}
        for k in ("AUROC_strat","companion","R_H"):
            v=np.array([b[k] for b in bs]); ent[k+"_sd1024"]=float(np.nanstd(v))
        out[f"OI2.{pname}.{nm}"]=ent
print(json.dumps(out,indent=1,default=float))
