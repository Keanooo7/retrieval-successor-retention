import time, torch, torch.nn.functional as F, gc, statistics
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.baselines.fifo import FIFOPolicy
DEV="mps"
def mg(): return torch.mps.driver_allocated_memory()/1e9
def step_fn(t,out,ids,mask,rv):
    lg=out.logits
    return F.cross_entropy(lg[:,:-1].reshape(-1,lg.shape[-1]), ids[:,1:].reshape(-1), reduction="mean")
def trial(d,S,batch,V=50257,L=64):
    gc.collect(); torch.mps.empty_cache(); base=mg()
    cfg=TGConfig(D=d,V=V,F=int(d*2.6875),max_sentence_tokens=L,max_sentences_in_short_term=40)
    m=TGModel(cfg).to(DEV)
    ids=torch.randint(0,V,(batch,S,L),device=DEV); mask=torch.ones(batch,S,L,dtype=torch.bool,device=DEV)
    lens=torch.full((batch,),S,device=DEV)
    t0=time.time(); acc=run_policy_loop(m,ids,mask,lens,FIFOPolicy(),step_fn=step_fn)
    acc.backward(); torch.mps.synchronize(); dt=time.time()-t0
    p=mg()-base; del m,ids,mask,acc; gc.collect(); torch.mps.empty_cache()
    return p, batch*S/dt

STEPS_30M = 1.2e6   # spec §4.1: a 30M-token corpus is ~1.2M sentence steps
print(f"{'config':<34}{'peak GB':>9}{'sent/s':>9}{'  hrs/run (1.2M steps)':>24}")
print("-"*78)
for d,S,b,note in ((128,80,16,"E3 shape, small width"),
                   (128,48,16,"synthetic (E1/E2)"),
                   (384,80,8,"E3 shape, widest")):
    peaks=[];thrs=[]
    for _ in range(3):
        p,t=trial(d,S,b); peaks.append(p); thrs.append(t)
    mp=statistics.mean(peaks); mt=statistics.mean(thrs)
    sd=statistics.stdev(thrs) if len(thrs)>1 else 0
    print(f"d={d} S={S} batch={b}  {note:<12}{mp:>9.1f}{mt:>7.0f}±{sd:<3.0f}{STEPS_30M/mt/3600:>16.1f} h")
print()
print(f"  spec §4.1 assumed <=7 sent/sec  ->  {STEPS_30M/7/3600:>6.1f} h per run  (48h quoted)")
