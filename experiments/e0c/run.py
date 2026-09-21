"""E0C -- the `(S, d, batch)` ceiling ON THIS MACHINE (gauntlet 2.7, ADR-0007).

**Rewritten 2026-09-17.** The earlier version sized against a rented 48 GB card.
There is no rental (ADR-0007), so the machine that sizes the run is the machine
that runs it -- which is the condition §7.6's fallback ladder exists to protect and
never had.

§4.2's rule is unchanged and now applies to hardware in hand:

> **If `S = 80` does not fit, `S` wins and `d` is cut.**

Gaps longer than `S` do not exist in the training data, so `S` silently caps the
headline experiment; width does not. That was the largest hole in v0.1.

## What is measured, and what is not

The bill is **the retained computation graph across the stream**, not the weights.
TG appends each gestalt to memory *without detaching*, and backward depth is
bounded by `S` rather than by `M` -- evicting a slot does not free its graph
(§3.6, [P2] App. A). So peak memory scales with `S x batch x L x d x depth`, and a
forward-only measurement would understate it by roughly the depth of the graph.

**Each configuration therefore runs a full forward AND backward.** A configuration
that forwards and cannot backward has not fitted.

§12.4: measure, don't extrapolate. **Do not inherit the spec's `21 sent/sec`** --
that was measured at `d_model = 768` / 85.6M parameters, roughly 4x wider than
anything here.

## Device

CPU by default, MPS with `--device mps`. MPS is best-effort, not required
(ADR-0001 D3), and unified memory means the two are drawing on the same 64 GB.
**`iogpu.wired_limit_mb` stays at 0, the system default** -- do not raise it.

Reports resident set size rather than a framework allocator figure: on unified
memory the process's RSS is the number that competes with everything else.

🔴 **Each configuration runs in its own subprocess.** `ru_maxrss` is a process
high-water mark and never decreases, so measuring a grid in one process makes every
row inherit the peak of the row before it -- the first run of this script reported
`d=256, batch=8` at 11.34 GB, which was `d=128, batch=64`'s peak, not its own. One
process per point is the only way the numbers mean what the table says.
"""

from __future__ import annotations

import gc
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main
from rsr.model.tg import TGConfig, TGModel, run_sentence_loop


@dataclass
class Point:
    d: int
    S: int
    batch: int
    M: int
    fitted: bool
    device: str = "cpu"
    peak_rss_gb: float | None = None
    mps_driver_gb: float | None = None
    """MPS only. 🔴 **RSS does not measure MPS allocations** -- the first run of
    this script reported 0.44 GB for a configuration that needs ~26 GB on CPU,
    because Metal memory does not appear in the process's resident size. On unified
    memory it is the same 64 GB either way, so the driver figure is the one that
    competes with everything else."""

    delta_rss_gb: float | None = None
    seconds: float | None = None
    sentences_per_sec: float | None = None
    params: int | None = None
    error: str | None = None


def _rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return usage / 1e9 if sys.platform == "darwin" else usage / 1e6


def _loss(t, out, ids, mask, row_valid):
    logp = torch.log_softmax(out.logits[:, :-1].to(torch.float32), dim=-1)
    picked = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    valid = (mask[:, 1:] == 1) & row_valid.unsqueeze(-1)
    return -(picked * valid).sum() / valid.sum().clamp(min=1)


def measure(
    d: int, s: int, batch: int, m: int, device: str, vocab: int, l_tok: int
) -> Point:
    gc.collect()
    before = _rss_gb()
    point = Point(d=d, S=s, batch=batch, M=m, fitted=False, device=device)
    try:
        cfg = TGConfig(
            D=d,
            H=max(1, d // 64),  # head_dim fixed at 64; head COUNT is the width axis
            V=vocab,
            max_sentence_tokens=l_tok,
            max_sentences_in_short_term=m,
            pad_id=vocab - 4,
            bos_id=vocab - 3,
            eos_id=vocab - 2,
            eod_id=vocab - 1,
        )
        model = TGModel(cfg).to(device)
        point.params = sum(p.numel() for p in model.parameters())
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab - 4, (batch, s, cfg.L), generator=g).to(device)
        ids[:, :, 0] = cfg.bos_id
        ids[:, :, -1] = cfg.eos_id
        mask = torch.ones_like(ids)
        lengths = torch.full((batch,), s, dtype=torch.long, device=device)

        start = time.perf_counter()
        loss = run_sentence_loop(model, ids, mask, lengths, step_fn=_loss)
        # 🔴 The backward is the measurement. The graph is retained across the
        # whole stream, so a forward-only number understates peak by the depth of
        # that graph -- which is exactly the bill §4.2 is about.
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
            point.mps_driver_gb = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        point.seconds = time.perf_counter() - start
        point.sentences_per_sec = (batch * s) / point.seconds
        point.fitted = True
    except (RuntimeError, MemoryError) as exc:
        point.error = f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        for name in ("model", "ids", "mask", "loss"):
            if name in dir():
                pass
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        point.peak_rss_gb = round(_rss_gb(), 3)
        point.delta_rss_gb = round(_rss_gb() - before, 3)
    return point


def measure_isolated(
    d: int, s: int, batch: int, m: int, device: str, vocab: int, l_tok: int
) -> Point:
    """Run `measure` in a fresh interpreter so `ru_maxrss` is this point's own."""
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--point",
            f"{d},{s},{batch},{m}",
            "--device",
            device,
            "--vocab",
            str(vocab),
            "--tokens",
            str(l_tok),
        ],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("POINT "):
            return Point(**json.loads(line[6:]))
    return Point(
        d=d,
        S=s,
        batch=batch,
        M=m,
        fitted=False,
        error=f"child exited {proc.returncode}: {(proc.stderr or proc.stdout)[-160:]}",
    )


def main() -> Exit:
    ap = ArgumentParser()
    ap.add_argument("--point", default=None, help="internal: d,S,batch,M for one child")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=24, help="content tokens/sentence")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--grid",
        default="synthetic,e3",
        help="comma-separated: synthetic (M=16,S=48), e3 (M=40,S=80)",
    )
    args = ap.parse_args()

    if args.point:
        d, s, batch, m = (int(x) for x in args.point.split(","))
        point = measure(d, s, batch, m, args.device, args.vocab, args.tokens)
        print("POINT " + json.dumps(asdict(point)))
        return Exit.OK

    if args.device == "mps" and not torch.backends.mps.is_available():
        # did not run -- never report as 0. The template rsr.exit_codes generalises.
        return did_not_run(
            "MPS unavailable; ADR-0001 D3 makes it best-effort, not required."
        )

    plans: list[tuple[int, int, int]] = []  # (M, S, label index)
    scopes = {"synthetic": (16, 48), "e3": (40, 80)}
    for name in args.grid.split(","):
        m, s = scopes[name.strip()]
        plans.append((m, s))

    results: list[Point] = []
    for m, s in plans:
        for d in (128, 256, 384):
            for batch in (8, 16, 32, 64):
                point = measure_isolated(
                    d, s, batch, m, args.device, args.vocab, args.tokens
                )
                results.append(point)
                status = "fit " if point.fitted else "FAIL"
                rate = (
                    f"{point.sentences_per_sec:8.1f} sent/s" if point.fitted else " " * 16
                )
                peak = (
                    point.peak_rss_gb if point.peak_rss_gb is not None else float("nan")
                )
                print(
                    f"  {status} d={d:3d} S={s:2d} M={m:2d} batch={batch:3d}  "
                    f"{'driver' if args.device == 'mps' else 'peak'}={peak:6.2f} GB  "
                    f"{rate}" + (f"  {point.error}" if point.error else "")
                )
                if not point.fitted:
                    break  # larger batches at this width will not fit either

    payload = {
        "device": args.device,
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
        "ram_bytes": int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout.strip()
            or 0
        ),
        "iogpu_wired_limit_mb": subprocess.run(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], capture_output=True, text=True
        ).stdout.strip(),
        "vocab": args.vocab,
        "tokens_per_sentence": args.tokens,
        "results": [asdict(p) for p in results],
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
