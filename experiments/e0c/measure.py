"""E0C throughput and memory, with the memory path actually exercised.

**Why this exists alongside `sweep.py` / `confirm.py`.** Those drew tokens with
`randint(0, V)` and left `TGConfig`'s default special-token ids in place. At
`V = 50257` the default `eos_id` is **50259** -- outside the range the tokens are
drawn from -- so `has_eos` was never true:

```
sentences processed              64
sentences that WROTE to memory    0
times the policy was consulted    0
```

Memory is never written, so `row_has_mem` is false on every row and
`TgCrossAttn` returns **exactly zero** (`out * row_has_mem`). The configuration
being timed is a stack of self-attention blocks with a dead cross-attention branch
and no retained gestalt graph across the stream -- which is the dominant memory
bill (§4.2) and the whole mechanism of TG.

So those numbers are neither a floor nor an estimate: they are for a different
model. This script fixes the token layout and **asserts** the memory fills.

## What is measured

Both policies, because the difference is the open question:

* **FIFO** -- `select_eviction` is an `argmin` over a `[M]` tensor.
* **RSR** -- runs the bilinear value head per row per step through a Python loop
  in `run_policy_loop`. That is the number that governs the schedule; FIFO's is a
  floor on it.

## Caveats that travel into RESULTS.md

* **Random tokens, not real text.** Real batching uses uniform token-budget
  bucketing (20,000 supervised tokens/step in the reference), so sentence lengths
  vary and the effective batch does too. These are fixed-shape, fully-packed
  sentences: the easy case.
* **No optimizer step.** Forward + backward only. AdamW over 22M parameters adds
  two more state tensors and a step, which is small beside the activation graph
  but not zero.
* **One data shape per row of the table.**
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsr.baselines.fifo import FIFOPolicy
from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.retention.rsr import RSRConfig, RSRPolicy

STEPS_30M = 1.2e6  # §4.1: a 30M-token corpus is ~1.2M sentence steps


def _mem_gb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1e9 if sys.platform == "darwin" else usage / 1e6


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _empty(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()


def build_batch(cfg: TGConfig, batch: int, s: int, device: str, seed: int = 0):
    """Sentences that are actually sentences: `[BOS] ... [EOS]`.

    🔴 Every sentence must end in `[EOS]` or it never reaches memory, which is the
    defect this script exists to correct.
    """
    g = torch.Generator().manual_seed(seed)
    content = cfg.L - 2
    ids = torch.randint(0, cfg.V - 4, (batch, s, cfg.L), generator=g)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    assert content > 0
    mask = torch.ones(batch, s, cfg.L, dtype=torch.long)
    lengths = torch.full((batch,), s, dtype=torch.long)
    return ids.to(device), mask.to(device), lengths.to(device)


def make_config(d: int, vocab: int, m: int, tokens: int) -> TGConfig:
    """Special ids INSIDE the vocabulary, and `F` from the reference formula."""
    return TGConfig(
        D=d,
        H=max(1, d // 64),
        V=vocab,
        max_sentence_tokens=tokens,
        max_sentences_in_short_term=m,
        pad_id=vocab - 4,
        bos_id=vocab - 3,
        eos_id=vocab - 2,
        eod_id=vocab - 1,
    )


def _step_fn(t, out, ids, mask, row_valid):
    logits = out.logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="mean",
    )


def trial(d, s, batch, *, policy_name, vocab, m, tokens, device):
    gc.collect()
    _empty(device)
    base = _mem_gb(device)
    cfg = make_config(d, vocab, m, tokens)
    model = TGModel(cfg).to(device)
    ids, mask, lengths = build_batch(cfg, batch, s, device)

    if policy_name == "fifo":
        policy = FIFOPolicy()
    elif policy_name == "rsr_neg_age":
        # §3.7's reduction. Measures POLICY DISPATCH only -- psi_hat == -a_i runs
        # no value head at all, so this is a floor on RSR, not RSR.
        policy = RSRPolicy(
            RSRConfig.reduction_to_tg(capacity=cfg.M),
            d_model=cfg.D,
            generator=torch.Generator().manual_seed(0),
        )
    elif policy_name == "rsr":
        # 🔴 The arm E3 actually runs: the LEARNED bilinear head, evaluated per
        # row per step through the Python loop in `run_policy_loop`. Measuring
        # `reduction_to_tg` instead reports dispatch overhead and calls it RSR.
        policy = RSRPolicy(
            RSRConfig(
                nu=0.0,
                beta=0.0,
                gamma=0.0,
                t_warm=0.0,
                a_max=cfg.M,
                psi_override=None,
                b_enabled=False,
                shadow_enabled=False,
            ),
            d_model=cfg.D,
            generator=torch.Generator().manual_seed(0),
        )
    else:
        raise ValueError(f"unknown policy {policy_name!r}")

    writes = {"n": 0, "evictions": 0}

    # Count evictions in the harness, not from `policy.records` -- FIFO has no
    # `records` attribute, so reading it reported 0 for every FIFO row.
    inner_select = policy.select_eviction

    def counted_select(slots, context, step):
        writes["evictions"] += 1
        return inner_select(slots, context, step)

    policy.select_eviction = counted_select  # type: ignore[method-assign]

    def counting_step(t, out, i, mk, rv):
        writes["n"] += int((rv & out.has_eos).sum())
        return _step_fn(t, out, i, mk, rv)

    start = time.time()
    loss = run_policy_loop(model, ids, mask, lengths, policy, step_fn=counting_step)
    loss.backward()
    _sync(device)
    seconds = time.time() - start
    peak = _mem_gb(device) - base

    # 🔴 The assertions the original harness lacked.
    evictions = writes["evictions"]
    if evictions == 0:
        raise RuntimeError(
            f"the policy was consulted 0 times in {batch * s} sentences: memory "
            f"never filled, so the eviction rule -- the entire contribution -- was "
            f"never exercised."
        )
    if writes["n"] != batch * s:
        raise RuntimeError(
            f"only {writes['n']} of {batch * s} sentences reached memory. The "
            f"memory path is not being exercised and the number is meaningless."
        )
    params = sum(p.numel() for p in model.parameters())
    del model, ids, mask, loss, policy
    gc.collect()
    _empty(device)
    return {
        "peak_gb": round(peak, 2),
        "seconds": round(seconds, 2),
        "sent_per_s": batch * s / seconds,
        "writes": writes["n"],
        "evictions": evictions,
        "params": params,
    }


def main() -> Exit:
    ap = ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--memory", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--policies", default="fifo,rsr_neg_age,rsr")
    ap.add_argument("--configs", default="128:80:16,128:48:16,384:80:8")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        return did_not_run("MPS unavailable (ADR-0001 D3: best-effort, not required)")

    rows = []
    print(
        f"{'policy':<13}{'config':<24}{'peak GB':>9}{'sent/s':>11}"
        f"{'  hrs/1.2M steps':>18}{'  writes':>9}{'  evict':>8}"
    )
    print("-" * 86)
    for spec in args.configs.split(","):
        d, s, batch = (int(x) for x in spec.split(":"))
        for policy_name in args.policies.split(","):
            peaks, rates, last = [], [], None
            error = None
            for _ in range(args.repeats):
                try:
                    r = trial(
                        d,
                        s,
                        batch,
                        policy_name=policy_name,
                        vocab=args.vocab,
                        m=args.memory,
                        tokens=args.tokens,
                        device=args.device,
                    )
                except (RuntimeError, MemoryError) as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:80]}"
                    break
                peaks.append(r["peak_gb"])
                rates.append(r["sent_per_s"])
                last = r
            label = f"d={d} S={s} batch={batch}"
            if error:
                print(f"{policy_name:<13}{label:<24}{'FAIL':>9}  {error}")
                rows.append(
                    {
                        "policy": policy_name,
                        "d": d,
                        "S": s,
                        "batch": batch,
                        "error": error,
                    }
                )
                continue
            mp, mt = statistics.mean(peaks), statistics.mean(rates)
            sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            hours = STEPS_30M / mt / 3600
            print(
                f"{policy_name:<13}{label:<24}{mp:>9.1f}{mt:>8.0f}±{sd:<3.0f}"
                f"{hours:>16.1f} h{last['writes']:>9}{last['evictions'] or 0:>8}"
            )
            rows.append(
                {
                    "policy": policy_name,
                    "d": d,
                    "S": s,
                    "batch": batch,
                    "peak_gb": round(mp, 2),
                    "sent_per_s": round(mt, 1),
                    "sent_per_s_sd": round(sd, 1),
                    "hours_per_1_2M_steps": round(hours, 2),
                    "repeats": args.repeats,
                    "writes": last["writes"],
                    "evictions": last["evictions"],
                    "params": last["params"],
                }
            )

    payload = {
        "device": args.device,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
        "vocab": args.vocab,
        "tokens_per_sentence": args.tokens,
        "memory_slots": args.memory,
        "repeats": args.repeats,
        "steps_30M": STEPS_30M,
        "results": rows,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
