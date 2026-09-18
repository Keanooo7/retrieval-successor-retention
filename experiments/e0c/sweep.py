import time, torch, torch.nn.functional as F, gc
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.baselines.fifo import FIFOPolicy

DEV = "mps"

def mem_gb():
    try: return torch.mps.driver_allocated_memory()/1e9
    except Exception: return float("nan")

def step_fn(t, out, ids_t, mask_t, row_valid):
    lg = out.logits
    return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                           ids_t[:, 1:].reshape(-1), reduction="mean")

def trial(d, S, batch, V, L=64):
    gc.collect(); torch.mps.empty_cache()
    base = mem_gb()
    cfg = TGConfig(D=d, V=V, F=int(d*2.6875), max_sentence_tokens=L,
                   max_sentences_in_short_term=40)
    m = TGModel(cfg).to(DEV)
    ids = torch.randint(0, V, (batch, S, L), device=DEV)
    mask = torch.ones(batch, S, L, dtype=torch.bool, device=DEV)
    lens = torch.full((batch,), S, device=DEV)
    t0 = time.time()
    acc = run_policy_loop(m, ids, mask, lens, FIFOPolicy(), step_fn=step_fn)
    acc.backward()
    torch.mps.synchronize()
    dt = time.time() - t0
    peak = mem_gb() - base
    del m, ids, mask, acc; gc.collect(); torch.mps.empty_cache()
    return peak, dt, batch*S/dt

print(f"{'V':>6} {'d':>5} {'S':>4} {'batch':>6} {'peak GB':>9} {'sec':>7} {'sent/s':>8}")
print("-"*52)
for V, tag in ((512,"test cfg"), (50257,"GPT-2 real")):
    for d, S in ((128,48), (128,80), (256,80), (384,80)):
        for batch in (8, 16, 32):
            try:
                p, dt, thr = trial(d, S, batch, V)
                print(f"{V:>6} {d:>5} {S:>4} {batch:>6} {p:>9.2f} {dt:>7.1f} {thr:>8.1f}")
            except RuntimeError as e:
                print(f"{V:>6} {d:>5} {S:>4} {batch:>6} {'OOM/ERR':>9}  {str(e)[:40]}")
                break
    print()
