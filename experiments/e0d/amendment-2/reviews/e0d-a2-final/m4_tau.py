import json, numpy as np
c=dict(np.load("/Users/keanooo7/Documents/RSR-2026-09-27-plan/reviews/e0d-a2/step1_cells.npz"))
full=c["tfull"].astype(bool); q=c["q"].astype(bool); A=full&q&c["a"].astype(bool); Q=full&q
o={}
for kn in ("dres","dzero"):
    d=c[kn]; off=Q&~A&~np.isnan(d); ad=np.abs(d[off])
    o[kn]={"n":int(off.sum()),**{f"q{qq}":float(np.quantile(ad,qq)) for qq in (0.99,0.995,0.999)}}
d=c["dres"]; v=Q&~np.isnan(d)
for tau in (0.3110,0.3779,0.4157,0.4716,0.7130):
    o[f"crit_count_resample_tau={tau}"]={"all":int((v&(d>tau)).sum()),"A":int((v&A&(d>tau)).sum()),"nonA":int((v&~A&(d>tau)).sum())}
# numpy quantile method disclosed
o["numpy_version"]=np.__version__; o["quantile_method"]="numpy default 'linear' (type 7)"
print(json.dumps(o,indent=1))
