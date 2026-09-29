"""AMD2: AUROC_strat / harm-ratio controls and power, set E [64,128) seed 0.
Delta side: real LOO (step1_cells.npz, sha 1338de4b...). Score side: SYNTHETIC ONLY.
No arm-B r_i is computed or read."""
import json, numpy as np
from statistics import NormalDist
R="/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2/"
c=dict(np.load(R+"step1_cells.npz")); reps=dict(np.load(R+"step1b_reps.npz"))
M=16; TAU=0.4157434984576128; A_STAR=0.85; H_STAR=0.5
Phi=NormalDist().cdf
full=c["tfull"].astype(bool); q=c["q"].astype(bool); Acell=q&full&c["a"].astype(bool)
age=(c["t"]-c["sent"]).astype(int); d=c["dres"]
rep1=reps["dres_donor1"][c["doc"],c["t"],c["rank"]]
bos=(c["rank"]!=M-1)&~(q&(c["gap"]==1))
key=c["doc"]*49+c["t"]; ndoc=64
rng=np.random.default_rng(20260927)

def build(pop_keep):
    sel=full&q&pop_keep
    k=key[sel]; u,inv=np.unique(k,return_inverse=True)
    G=lambda col,fill=np.nan: (lambda g:(g.__setitem__((inv,c["rank"][sel]),col[sel]),g)[1])(np.full((len(u),M),fill,float))
    elig=G(np.ones(len(d)),0.0)>0
    dg=G(d); ag=G(age.astype(float))
    sdoc=np.zeros(len(u),int); sdoc[inv]=c["doc"][sel]
    y=(dg>TAU)
    complete=~np.any(elig&np.isnan(dg),axis=1)
    kt=np.where(elig,y,False).sum(1); nt=elig.sum(1)
    qc=complete&(kt>=1)
    # cell arrays for AUROC: eligible & has d
    cm=elig&~np.isnan(dg)
    return dict(sel=sel,inv=inv,elig=elig,dg=dg,ag=ag,sdoc=sdoc,y=y,qc=qc,kt=kt,nt=nt,cm=cm)

def wauc(r,y,w):
    pos=y; Wp=w[pos].sum(); Wn=w[~pos].sum()
    if Wp<=0 or Wn<=0: return np.nan,0.0
    uv,ix=np.unique(r,return_inverse=True)
    wn=np.bincount(ix,weights=w*(~pos),minlength=len(uv))
    below=np.cumsum(wn)-wn
    s=(w[pos]*(below[ix[pos]]+0.5*wn[ix[pos]])).sum()
    return s/(Wp*Wn),Wp

def stats(P,rg,dw):
    """rg: score grid [steps,M]; dw: per-doc weight [64]"""
    sw=dw[P["sdoc"]]  # step weight
    cm=P["cm"]; W=np.broadcast_to(sw[:,None],cm.shape)[cm]
    r=rg[cm]; y=P["y"][cm]; a=P["ag"][cm]
    num=den=0.0; nb=0
    for b in range(1,M+1):
        m=a==b
        if not m.any(): continue
        au,wp=wauc(r[m],y[m],W[m])
        if np.isnan(au): continue
        num+=wp*au; den+=wp; nb+=1
    A_s=num/den if den>0 else np.nan
    A_u,_=wauc(r,y,W)
    # harm on complete Q_crit steps
    qc=P["qc"]; e=P["elig"][qc]; yy=P["y"][qc]; rr=np.where(e,rg[qc],np.inf); aa=P["ag"][qc]; ww=sw[qc]
    mn=rr.min(1,keepdims=True); tie=(rr==mn)&e; nt_=tie.sum(1)
    harm=(tie&yy).sum(1)/nt_
    Wt=ww.sum()
    H=(ww*harm).sum()/Wt
    Hr=(ww*(np.where(e,yy,False).sum(1)/e.sum(1))).sum()/Wt
    pi=np.zeros(M+1)
    for b in range(1,M+1): pi[b]=(ww*((tie&(aa==b)).sum(1)/nt_)).sum()/Wt
    ya=np.zeros(len(ww))
    for b in range(1,M+1): ya+=pi[b]*((aa==b)&yy&e).sum(1)
    Ha=(ww*ya).sum()/Wt
    # within-step companion: percentile of critical slots, age-adjusted (all complete Q steps w/ elig)
    Pc=P["cm"]; rgm=np.where(Pc,rg,np.nan)
    pct=np.full(rg.shape,np.nan)
    for j in range(rg.shape[0]):
        ok=Pc[j]
        if ok.sum()<2: continue
        v=rgm[j,ok]; order=v.argsort(); rk=np.empty(len(v)); 
        # midranks
        uv,ix,cnt=np.unique(v,return_inverse=True,return_counts=True); cs=np.cumsum(cnt)-cnt; mid=cs+(cnt+1)/2
        pct[j,ok]=(mid[ix]-1)/(ok.sum()-1)
    mbin=np.zeros(M+1)
    for b in range(1,M+1):
        m=Pc&(P["ag"]==b)
        if m.any(): mbin[b]=(np.broadcast_to(sw[:,None],m.shape)[m]*pct[m]).sum()/np.broadcast_to(sw[:,None],m.shape)[m].sum()
    adj=np.where(Pc&P["y"],pct-mbin[np.nan_to_num(P["ag"]).astype(int)],np.nan)
    stepv=np.nanmean(adj,axis=1) if True else None
    okst=~np.isnan(stepv)
    comp=(sw[okst]*stepv[okst]).sum()/sw[okst].sum() if okst.any() else np.nan
    return dict(AUROC_strat=A_s,AUROC_unstrat=A_u,H=H,H_random=Hr,H_age_random=Ha,
                R_H=H/Ha if Ha>0 else np.nan,R_H_uniform=H/Hr if Hr>0 else np.nan,
                companion=comp,n_bins=nb,n_Qcrit=int(qc.sum()))

def softmax_steps(s,elig):
    s=np.where(elig,s,-np.inf); s=s-s.max(1,keepdims=True); e=np.exp(s); return e/e.sum(1,keepdims=True)

def scores(P):
    elig=P["elig"]; n=elig.shape[0]
    crit=np.nan_to_num(P["y"].astype(float))
    ag=np.nan_to_num(P["ag"]).astype(int)
    # age channel: logit of per-age critical rate on this sample (structure-like prior)
    rate=np.array([ (P["y"]&(P["ag"]==b)&elig).sum()/max(1,((P["ag"]==b)&elig).sum()) for b in range(M+1)])
    lr=np.log(np.clip(rate,1e-3,1))
    lr=(lr-lr[1:].mean())/lr[1:].std()
    A_grid=np.zeros(elig.shape); A_grid[P["inv"],c["rank"][P["sel"]]]=Acell[P["sel"]]
    rep=np.full(elig.shape,np.nan); rep[P["inv"],c["rank"][P["sel"]]]=rep1[P["sel"]]
    out={"PC_binary_true_demand":A_grid+0.0,
         "ORACLE_loo_replicate":np.nan_to_num(rep,nan=0.0),
         "AGE_only":lr[ag]+0.0,
         "RECENCY":-ag.astype(float)}
    base=rng.standard_normal(elig.shape)
    out["NULL_shuffled_within_step"]=np.array([rng.permutation(row) for row in softmax_steps(base+lr[ag],elig)])
    for a in (0.5,1.0,1.5,2.0,2.5,3.0,4.0):
        out[f"SYN_a{a}_ageb1"]=softmax_steps(a*crit+1.0*lr[ag]+base,elig)
    return out

res={}
for pname,keep in (("bos_excluded(operative)",bos),("all_cells",np.ones(len(d),bool))):
    P=build(keep); S_=scores(P); res[pname]={}
    ones=np.ones(ndoc)
    Wb=rng.multinomial(1024,np.ones(ndoc)/ndoc,size=300).astype(float)
    Wn=np.array([np.bincount(rng.integers(0,ndoc,ndoc),minlength=ndoc) for _ in range(300)],float)
    for nm,rg in S_.items():
        rg=np.where(P["elig"],rg,np.nan)
        pt=stats(P,rg,ones)
        bs=[stats(P,rg,w) for w in Wb]
        sd={k:float(np.nanstd([b[k] for b in bs])) for k in ("AUROC_strat","R_H","AUROC_unstrat","H","H_age_random","companion")}
        nanfrac=float(np.mean([np.isnan(b["R_H"]) for b in bs]))
        ent={k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in pt.items()}
        ent["sd_at_1024"]=sd; ent["R_H_nan_frac_1024"]=nanfrac
        if nm.startswith("SYN") or nm.startswith("PC") or nm.startswith("ORACLE"):
            pa=Phi((pt["AUROC_strat"]-A_STAR)/sd["AUROC_strat"]-1.96) if sd["AUROC_strat"]>0 else float(pt["AUROC_strat"]>=A_STAR)
            ph=Phi((H_STAR-pt["R_H"])/sd["R_H"]-1.96) if sd["R_H"]>0 else float(pt["R_H"]<=H_STAR)
            ent["power_one_seed_AUROC"]=pa; ent["power_one_seed_harm"]=ph
        res[pname][nm]=ent
    # generic MDE at the bars
    res[pname]["_MDE_note"]="theta_80 = bar + 2.80*sd (AUROC), bar - 2.80*sd (R_H); normal approx"
    P0=P
out={"tau":TAU,"populations":res}
print(json.dumps(out,indent=1,default=float))
