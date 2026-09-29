"""Expire-Span wiring smoke -- ONE forward + backward + optimizer step. Not a result.

The 2026-09-29 build brief asks for "a single short synthetic forward+train step with
Expire-Span eviction on the M=16 synthetic corpus, to show it runs", with its
wall-clock. This is that, and nothing more: no comparison, no metric, no claim about
Expire-Span vs RSR (that is the referendum, which needs a PREREG and B-3's protocol).

The corpus, vocabulary, TGConfig and optimizer mirror `rsr.train.loop.train()`'s
defaults (d=128, S=48, batch=16, M=16, seed 0). **Every ExpireSpanConfig value below
is SMOKE-ONLY** -- chosen so the mask has slots on its ramp and the gradient path is
exercised, not proposed for any run. ADR-0010 q2 is where the real values are decided.
The predictor's optimizer group is likewise a placeholder (ADR-0010 q9).

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det \\
        --slots 1 -- .venv/bin/python scripts/expire_span_smoke.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.baselines.expire_span import ExpireSpanConfig, ExpireSpanPolicy  # noqa: E402
from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.mup.param_groups import build_param_groups  # noqa: E402
from rsr.train.loop import build_vocab, encode, lm_loss  # noqa: E402

SMOKE_ONLY = dict(
    max_span=16.0, ramp=4.0, loss_coef=1e-3, dropout=0.1, init_span_fraction=0.5
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grad-path", default="predictor")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    d, steps, batch, m, seed = 128, 48, 16, 16, 0
    torch.manual_seed(seed)
    docs = generate(SyntheticConfig(sentences_per_document=steps, seed=seed))
    vocab = build_vocab(docs)
    ids, mask = encode(docs, vocab, max_tokens=64, steps=steps)
    cfg = TGConfig(
        D=d,
        V=4 + len(vocab),
        F=int(d * 2.6875),
        max_sentence_tokens=64,
        max_sentences_in_short_term=m,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).to(a.device).train()
    policy = ExpireSpanPolicy(
        ExpireSpanConfig(grad_path=a.grad_path, seed=seed, **SMOKE_ONLY), d_model=d
    ).to(a.device)
    groups = build_param_groups(model, None, base_lr=1e-3, d_model=d, base_width=128)
    groups.append({"params": list(policy.parameters()), "lr": 1e-3})  # q9 placeholder
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)

    x, mk = ids[:batch].to(a.device), mask[:batch].to(a.device)
    lengths = torch.full((batch,), steps, device=a.device)

    def step_fn(t, o, ids_t, mask_t, row_valid):
        return lm_loss(o.logits, ids_t, mask_t)

    def sync():
        if a.device == "mps":
            torch.mps.synchronize()
        elif a.device.startswith("cuda"):
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    loss = run_policy_loop(model, x, mk, lengths, policy, step_fn=step_fn)
    sync()
    t_fwd = time.perf_counter() - t0
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    w_grad = None if policy.weight.grad is None else float(policy.weight.grad.norm())
    opt.step()
    sync()
    t_total = time.perf_counter() - t0

    with torch.no_grad():
        spans = policy.spans(x.new_zeros(1, d, dtype=torch.float32))
    sha = subprocess.run(
        ["/opt/homebrew/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    rec = {
        "kind": "SMOKE -- wiring check, not a result",
        "git_sha": sha,
        "device": a.device,
        "torch": torch.__version__,
        "grad_path": a.grad_path,
        "smoke_only_config": SMOKE_ONLY,
        "d": d,
        "S": steps,
        "batch": batch,
        "M": m,
        "seed": seed,
        "documents": len(docs),
        "loss_sum_over_steps": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "evictions_in_stream_all_rows": len(policy.victims),
        "predictor_weight_grad_norm": w_grad,
        "span_at_zero_input_after_step": float(spans.reshape(-1)[0]),
        "wall_s_forward": round(t_fwd, 3),
        "wall_s_forward_backward_step": round(t_total, 3),
    }
    print(json.dumps(rec, indent=2))
    if a.out:
        a.out.write_text(json.dumps(rec, indent=2) + "\n")
    return 0 if rec["loss_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
