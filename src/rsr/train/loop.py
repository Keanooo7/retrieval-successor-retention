"""Minimal training entry point.

What was measured in E0c was **forward and backward only** -- no optimizer step, no
optimizer
state, no data path. This is the loop that closes those three gaps, and it is deliberately
the
smallest thing that is honestly a training run:

  synthetic corpus -> run_policy_loop (retained graph across the stream) -> loss ->
  backward
  -> muP-grouped AdamW step -> heartbeat -> periodic atomic checkpoint

Three things it refuses to do, each because the alternative silently invalidates a result:

* **It does not invent constants.** ``beta`` and ``nu`` come from the registry, which
RAISES when
  they have no logged value (§4.5). A training run that supplied its own would be D-1's
  defect --
  a frozen unmeasured constant governing a mechanism -- and the registry exists to stop
  exactly it.
* **It does not build a plain optimizer.** Parameter groups come from
``build_param_groups``, so
  the value head lands in its own muP group with the ``1/d`` multiplier (§4.3). A single
  flat
  ``AdamW(model.parameters())`` would break width transfer and fail silently in week 9.
* **It does not hide a crash.** The heartbeat records the traceback before the exception
  propagates, so a run that dies at 03:00 is distinguishable in the morning from one that
  never
  started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from rsr.baselines.fifo import FIFOPolicy
from rsr.data.synthetic import SyntheticConfig, generate
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.mup.param_groups import build_param_groups
from rsr.train import checkpoint as ck
from rsr.train.heartbeat import Heartbeat

__all__ = ["main", "train"]


def _sha() -> str:
    """Stamped from git INSIDE this tree. Never accepted as an argument -- a caller-typed
    sha is
    a provenance claim, not provenance."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config_hash(payload: dict) -> str:
    """Freeze the config before the run. The heartbeat carries this, so a config changed
    mid-flight
    cannot be reported against the frozen one."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_vocab(docs) -> dict[str, int]:
    """Word-level vocabulary over the synthetic corpus, deterministic by sort order.

    Ids 0-3 are reserved for the special tokens, because **TG reads the gestalt at the
    first
    [EOS] and writes memory only when a sentence has one.** A vocabulary that cannot
    express the
    EOS id produces `has_eos = False` on every sentence, so nothing is ever written,
    cross-attention
    returns exactly zero, and the model still emits a plausible loss. `config.py:190`
    names this
    trap; E0C's first measurement fell into it and so did the first version of this file.
    """
    words = sorted({w for d in docs for sent in d.sentences for w in sent.text.split()})
    return {w: i + 4 for i, w in enumerate(words)}  # 0=pad 1=bos 2=eos 3=eod


def encode(docs, vocab: dict[str, int], *, max_tokens: int, steps: int):
    """-> ids (n_docs, steps, L) with an explicit EOS per sentence, and a mask.

    Layout per sentence: ``[w0 w1 ... wk, EOS, pad ...]``. The EOS is written even when
    the
    sentence is truncated, because losing it silently disables the memory path.
    """
    n = len(docs)
    L = max_tokens
    ids = torch.zeros(n, steps, L, dtype=torch.long)
    mask = torch.zeros(n, steps, L, dtype=torch.bool)
    for di, doc in enumerate(docs):
        for si, sent in enumerate(doc.sentences[:steps]):
            toks = [vocab[w] for w in sent.text.split()][: L - 1]
            toks.append(2)  # EOS -- never optional
            ids[di, si, : len(toks)] = torch.tensor(toks)
            mask[di, si, : len(toks)] = True
    return ids, mask


def lm_token_losses(logits, ids_t):
    """Per-target next-token cross-entropy, flat, one sentence step.

    The `[:, :-1]` / `[:, 1:]` shift means the population is **targets**, not
    tokens: `batch x (L - 1)` of them.
    """
    lg = logits
    return F.cross_entropy(
        lg[:, :-1].reshape(-1, lg.shape[-1]),
        ids_t[:, 1:].reshape(-1),
        reduction="none",
    )


def lm_loss(logits, ids_t, mask_t):
    """Next-token cross-entropy over **real** targets only (cycle 1, defect 1).

    The version this replaces reduced with `reduction="mean"` over every target,
    including PAD -- while `mask_t` was already being handed to `step_fn` by
    `rsr.model.tg.policy_loop:140` and already going unused. On the committed
    synthetic corpus that is not a correction at the margin: sentences are ~5
    tokens inside a 64-token frame, so the number being minimised, reported and
    turned into a perplexity was overwhelmingly the model's skill at predicting
    zeros. `tests/test_train_loss.py` measures the fraction rather than quoting it.

    Selection is on `mask_t` rather than on `ids_t != pad_id` because the mask is
    what `encode()` built and what the loop already passes; the two agree here and
    a test says so. **A step with no real target returns 0, not `nan`** -- `0 / 0`
    would poison the accumulated stream loss for any document shorter than
    `steps_per_stream`.
    """
    per = lm_token_losses(logits, ids_t)
    valid = mask_t[:, 1:].reshape(-1)
    n = valid.sum()
    return (per * valid.to(per.dtype)).sum() / n.clamp(min=1)


def lm_loss_unmasked(logits, ids_t, mask_t):
    """🔴 The pre-cycle-1 objective, scoring PAD. **Never the default.**

    It survives behind `train(masked_loss=False)` for exactly one reason: the
    masked and unmasked arms have to be comparable *at one sha*, and an arm that
    only exists at the parent commit is an arm whose ledger cites a different tree.
    It is a documented off-switch in the sense CLAUDE.md means, not an option.
    """
    return lm_token_losses(logits, ids_t).mean()


def train(
    *,
    d: int = 128,
    steps_per_stream: int = 48,
    batch: int = 16,
    vocab: int | None = None,  # None -> derive from the corpus
    max_tokens: int = 64,
    memory_slots: int = 16,
    iters: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "mps",
    out_dir: str | Path = "runs/dev",
    beat_every: int = 1,
    ckpt_every: int = 25,
    resume: str | Path | None = None,
    policy_name: str = "fifo",
    masked_loss: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    probe = generate(SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed))
    V = vocab if vocab else 4 + len(build_vocab(probe))
    cfg = TGConfig(
        D=d,
        V=V,
        F=int(d * 2.6875),
        max_sentence_tokens=max_tokens,
        max_sentences_in_short_term=memory_slots,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).to(device)

    # muP groups, NOT a flat AdamW. The value head is None until RSRPolicy is wired; when
    # it is,
    # it must be passed here or it will not get its own group and width transfer breaks.
    groups = build_param_groups(model, None, base_lr=lr, d_model=d, base_width=128)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)

    policy = (
        FIFOPolicy()
    )  # RSRPolicy requires beta/nu from the registry; see module docstring.

    frozen = {
        "tg": asdict(cfg),
        "iters": iters,
        "batch": batch,
        "steps_per_stream": steps_per_stream,
        "lr": lr,
        "seed": seed,
        "policy": policy_name,
        "device": device,
        "masked_loss": masked_loss,
    }
    run_id = f"{policy_name}-d{d}-s{steps_per_stream}-b{batch}-{_config_hash(frozen)}"

    start = 0
    if resume:
        state = ck.load(
            resume, model=model, optimizer=opt, policy=policy, restore_rng=True
        )
        start = int(state.get("step", 0))

    docs = generate(SyntheticConfig(sentences_per_document=steps_per_stream, seed=seed))
    vocab_map = build_vocab(docs)
    all_ids, all_mask = encode(
        docs, vocab_map, max_tokens=max_tokens, steps=steps_per_stream
    )
    all_ids, all_mask = all_ids.to(device), all_mask.to(device)
    hb = Heartbeat(
        out / "heartbeat.jsonl",
        run_id=run_id,
        provenance={
            "git_sha": _sha(),
            "device": device,
            "seed": seed,
            "config_hash": _config_hash(frozen),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "documents": len(docs),
        },
    )
    hb.header(
        config=frozen, resumed_from=str(resume) if resume else None, start_step=start
    )

    # Both numbers are computed every step and only one is optimised. The other is
    # the *common yardstick*: comparing a masked arm's masked loss against an
    # unmasked arm's unmasked loss compares two different objectives and is
    # tautologically different. Real-token NLL is the same quantity in both arms.
    tally: dict[str, float] = {}

    def step_fn(t, o, ids_t, mask_t, row_valid):
        loss = (lm_loss if masked_loss else lm_loss_unmasked)(o.logits, ids_t, mask_t)
        with torch.no_grad():
            valid = mask_t[:, 1:].reshape(-1)
            tally["real_sum"] = tally.get("real_sum", 0.0) + float(
                lm_loss(o.logits, ids_t, mask_t)
            )
            tally["all_sum"] = tally.get("all_sum", 0.0) + float(
                lm_loss_unmasked(o.logits, ids_t, mask_t)
            )
            tally["n_targets"] = tally.get("n_targets", 0.0) + valid.numel()
            tally["n_pad_targets"] = tally.get("n_pad_targets", 0.0) + float(
                (~valid).sum()
            )
        return loss

    last: dict[str, float] = {}
    try:
        for it in range(start, iters):
            t0 = time.time()
            tally.clear()
            sel = torch.randint(
                0, all_ids.shape[0], (batch,), generator=gen, device=device
            )
            ids, mask = all_ids[sel], all_mask[sel]
            lengths = torch.full((batch,), steps_per_stream, device=device)
            loss = run_policy_loop(model, ids, mask, lengths, policy, step_fn=step_fn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if device == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0

            # Per sentence step, so the number is a per-token NLL rather than a
            # stream sum. `loss_real_tokens` is the same quantity in both arms and
            # is the only honest masked-vs-unmasked comparison; `loss` is whichever
            # objective this arm actually minimised.
            last = {
                "loss": float(loss.detach()) / steps_per_stream,
                "loss_real_tokens": tally["real_sum"] / steps_per_stream,
                "loss_all_targets": tally["all_sum"] / steps_per_stream,
                "pad_target_fraction": tally["n_pad_targets"] / tally["n_targets"],
                "n_targets_scored": tally["n_targets"],
                "n_pad_targets_scored": tally["n_pad_targets"],
                "step": it,
            }
            if it % beat_every == 0:
                attr = policy.attribution() if hasattr(policy, "attribution") else None
                hb.beat(
                    it,
                    loss=last["loss"],
                    loss_sum=float(loss.detach()),
                    loss_real_tokens=last["loss_real_tokens"],
                    loss_all_targets=last["loss_all_targets"],
                    pad_target_fraction=last["pad_target_fraction"],
                    n_targets_scored=last["n_targets_scored"],
                    masked_loss=masked_loss,
                    ppl=float(torch.exp(loss.detach() / steps_per_stream)),
                    gini=None,  # wired when the policy exposes an attention trace
                    attribution=attr,
                    rank_shift=None,  # wired with RSRPolicy; 0 under FIFO by construction
                    grad_norm=float(gnorm),
                    sent_per_s=round(batch * steps_per_stream / dt, 1),
                    mem_gb=round(torch.mps.driver_allocated_memory() / 1e9, 2)
                    if device == "mps"
                    else None,
                    lr=opt.param_groups[0]["lr"],
                )
            if ckpt_every and (it + 1) % ckpt_every == 0:
                ck.save(
                    out / f"ckpt-{it + 1:06d}.pt",
                    step=it + 1,
                    model=model,
                    optimizer=opt,
                    policy=policy,
                    meta={"run_id": run_id},
                )
    except BaseException as e:
        hb.crash(e)
        raise
    hb.footer("completed", final_step=iters)
    return {
        "run_id": run_id,
        "heartbeat": str(out / "heartbeat.jsonl"),
        "config_hash": _config_hash(frozen),
        "git_sha": _sha(),
        "steps_requested": iters,
        # `steps_done` is the loop's own count, not the request. A resumed or
        # crashed run must not be able to report the number it asked for.
        "steps_done": (last["step"] + 1) if last else start,
        "final": last,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsr-train", description=__doc__)
    for name, typ, dflt in (
        ("--d", int, 128),
        ("--batch", int, 16),
        ("--iters", int, 50),
        ("--steps-per-stream", int, 48),
        ("--vocab", int, 50257),
        ("--memory-slots", int, 16),
        ("--lr", float, 1e-3),
        ("--seed", int, 0),
        ("--beat-every", int, 1),
        ("--ckpt-every", int, 25),
    ):
        p.add_argument(name, type=typ, default=dflt)
    p.add_argument("--device", default="mps")
    p.add_argument("--out-dir", default="runs/dev")
    p.add_argument("--resume", default=None)
    a = p.parse_args(argv)
    r = train(
        d=a.d,
        batch=a.batch,
        iters=a.iters,
        steps_per_stream=a.steps_per_stream,
        vocab=a.vocab,
        memory_slots=a.memory_slots,
        lr=a.lr,
        seed=a.seed,
        device=a.device,
        out_dir=a.out_dir,
        beat_every=a.beat_every,
        ckpt_every=a.ckpt_every,
        resume=a.resume,
    )
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
