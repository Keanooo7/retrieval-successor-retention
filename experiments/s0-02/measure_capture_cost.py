"""S0-02 bar item 2: what the §3.2.1 capture bridge costs when it is on.

## The question, stated so it cannot drift

**One harness, one sitting, one device, three arms, `N` repeats each, interleaved.**

* `capture_off` -- `run_policy_loop(...)`, the default path. The control.
* `capture_on`  -- `run_policy_loop(..., observe=True)`: the bridge builds an
  `AttentionTrace` per live row per step and hands it to `policy.observe`.
* `capture_on_ri` -- the same, and the policy actually computes `r_i` from every
  trace via `rsr.retention.reward.retrieval_demand`. The bridge's cost and the
  reward's cost are different numbers and reporting one as the other would be the
  same mistake `310 sent/s` made.

## 🔴 Why `310 ± 2 sent/s` is not quoted here as a baseline

The brief's bar item 2 offers `experiments/e0c/RESULTS.md:58,67`'s **310 ± 2
sent/s** as the yardstick. It was really run, but:

* **it has no ledger row** -- no ledger in this project holds a throughput key at
  all (`sent_per_s`, `sent/s`, `throughput`: no file matches), so it is a
  RESULTS.md number, and citing it as a ledger row would be a provenance claim
  nobody can check;
* **`docs/RESEARCH-CONTEXT.md:736-744` disqualifies it as a *training* rate**:
  random tokens, one data shape per row, **no optimizer step**, MPS only --
  *"a measurement of the model's throughput, not of a training loop's."*

So it is used for exactly one thing: as a **prediction under test**, that
`capture_off` measured here lands near it. A delta reported as a percentage of a
training throughput nobody measured would be a number about a number.

## Interleaving, and why it is not optional here

The arms are timed **round-robin within a repeat**, not arm-by-arm. On this
machine thermal state drifts over minutes and an arm-major loop gives the last arm
the hot chassis; that is a systematic bias exactly the size of the effect being
measured. The `sd` reported is across repeats of the same arm, so it carries that
drift rather than hiding it.

Caveats inherited from `experiments/e0c/measure.py`, which this reuses
deliberately so the two are comparable: random tokens rather than real text, fixed
fully-packed sentence shapes, forward + backward with **no optimizer step**.
"""

from __future__ import annotations

import argparse
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
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.retention.policy import AttentionTrace, MemoryState
from rsr.retention.reward import retrieval_demand

ARMS = ("capture_off", "capture_on", "capture_on_ri")


class CountingFIFO(FIFOPolicy):
    """FIFO, plus a count of the traces it was handed.

    🔴 The count is the assertion. A capture arm that silently observed **zero**
    steps would time identically to the control and be reported as "free", which
    is the e0c failure (`policy consulted 0 times`) one level up.
    """

    name = "fifo_counting"

    def __init__(self) -> None:
        self.n_observed = 0
        self.n_ri = 0

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        self.n_observed += 1


class RewardFIFO(CountingFIFO):
    """...and computes `r_i` from every trace, which is what `observe` is *for*."""

    name = "fifo_reward"

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        self.n_observed += 1
        if slots.n_live:
            retrieval_demand(
                attn, n_live=slots.n_live, capacity=slots.capacity, gated=True
            )
            self.n_ri += 1


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


def make_config(d: int, vocab: int, m: int, tokens: int) -> TGConfig:
    """`experiments/e0c/measure.py`'s config verbatim, so the arms are comparable.

    Special ids **inside** the vocabulary: at `V = 50257` the default `eos_id` is
    50259, outside the range tokens are drawn from, so `has_eos` was never true and
    e0c's first three scripts timed a model whose memory path was dead.
    """
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


def build_batch(cfg: TGConfig, batch: int, s: int, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.V - 4, (batch, s, cfg.L), generator=g)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    mask = torch.ones(batch, s, cfg.L, dtype=torch.long)
    lengths = torch.full((batch,), s, dtype=torch.long)
    return ids.to(device), mask.to(device), lengths.to(device)


def _step_fn(t, out, ids, mask, row_valid):
    logits = out.logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1),
        reduction="mean",
    )


def trial(d, s, batch, *, arm, vocab, m, tokens, device, seed=0):
    gc.collect()
    _empty(device)
    base = _mem_gb(device)
    cfg = make_config(d, vocab, m, tokens)
    torch.manual_seed(seed)
    model = TGModel(cfg).to(device)
    # 🔴 eval() is not a speed trick: correction 20 / D-F. `attn_dropout = 0.2`
    # stochastically zeroes alpha, so an on-policy `r_i` collected in train mode
    # is noisy in a policy-relevant direction and `reward` refuses the trace. The
    # control runs in eval too, or the arms differ in two things.
    model.eval()
    ids, mask, lengths = build_batch(cfg, batch, s, device)

    policy = RewardFIFO() if arm == "capture_on_ri" else CountingFIFO()
    observe = arm != "capture_off"

    start = time.time()
    loss = run_policy_loop(
        model, ids, mask, lengths, policy, step_fn=_step_fn, observe=observe
    )
    loss.backward()
    _sync(device)
    seconds = time.time() - start
    peak = _mem_gb(device) - base

    expected = batch * s
    if observe and policy.n_observed != expected:
        raise RuntimeError(
            f"{arm}: observe() ran {policy.n_observed} times, expected {expected}. "
            f"A capture arm that captured nothing times like the control and gets "
            f"reported as free."
        )
    if not observe and policy.n_observed:
        raise RuntimeError(
            f"{arm}: observe() ran {policy.n_observed} times with "
            f"observe=False; 'off by default' is false"
        )
    if arm == "capture_on_ri" and policy.n_ri == 0:
        raise RuntimeError("capture_on_ri computed r_i zero times")

    out = {
        "peak_gb": round(peak, 2),
        "seconds": round(seconds, 3),
        "sent_per_s": batch * s / seconds,
        "observed": policy.n_observed,
        "r_i_computed": policy.n_ri,
    }
    del model, ids, mask, loss, policy
    gc.collect()
    _empty(device)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--memory", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--S", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable (ADR-0001 D3: best-effort, not required)")
        return 3

    # One warm-up per arm, discarded: the first MPS graph compile is not the cost
    # of capture and would land entirely on whichever arm ran first.
    for arm in ARMS:
        trial(
            args.d,
            args.S,
            args.batch,
            arm=arm,
            vocab=args.vocab,
            m=args.memory,
            tokens=args.tokens,
            device=args.device,
            seed=99,
        )

    samples: dict[str, list[float]] = {a: [] for a in ARMS}
    peaks: dict[str, list[float]] = {a: [] for a in ARMS}
    counts: dict[str, dict] = {}
    # Interleaved, deliberately -- see the module docstring.
    for rep in range(args.repeats):
        for arm in ARMS:
            r = trial(
                args.d,
                args.S,
                args.batch,
                arm=arm,
                vocab=args.vocab,
                m=args.memory,
                tokens=args.tokens,
                device=args.device,
                seed=rep,
            )
            samples[arm].append(r["sent_per_s"])
            peaks[arm].append(r["peak_gb"])
            counts[arm] = {"observed": r["observed"], "r_i_computed": r["r_i_computed"]}
            print(
                f"rep {rep}  {arm:15s} {r['sent_per_s']:7.1f} sent/s  "
                f"{r['peak_gb']:5.1f} GB  observed={r['observed']}"
            )

    def sd(xs):
        return statistics.stdev(xs) if len(xs) > 1 else None

    rows = []
    off = statistics.mean(samples["capture_off"])
    print()
    for arm in ARMS:
        mean = statistics.mean(samples[arm])
        rows.append(
            {
                "arm": arm,
                "sent_per_s_samples": [round(x, 2) for x in samples[arm]],
                "sent_per_s_mean": round(mean, 2),
                "sent_per_s_sd": None
                if sd(samples[arm]) is None
                else round(sd(samples[arm]), 2),
                "peak_gb_samples": peaks[arm],
                "delta_vs_off_sent_per_s": round(mean - off, 2),
                "delta_vs_off_pct": round(100.0 * (mean - off) / off, 2),
                "observe_calls": counts[arm]["observed"],
                "r_i_computed": counts[arm]["r_i_computed"],
            }
        )
        s = sd(samples[arm])
        print(
            f"{arm:15s} {mean:7.1f} +/- {('n/a' if s is None else f'{s:.1f}'):>5} sent/s"
            f"   delta {mean - off:+7.1f} ({100.0 * (mean - off) / off:+.1f}%)"
        )

    payload = {
        "device": args.device,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
        "config": {
            "d": args.d,
            "S": args.S,
            "batch": args.batch,
            "vocab": args.vocab,
            "tokens_per_sentence": args.tokens,
            "memory_slots": args.memory,
            "repeats": args.repeats,
            "mode": "eval",
            "optimizer_step": False,
        },
        "arms": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
