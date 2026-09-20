"""Cycle 1 -- does masking the training loss to real tokens change anything?

The falsifier (`docs/lab-notes/dispatch-cycle-01-masked-loss.md`):

> The reading dies if, on the committed synthetic corpus at `seed=0,
> steps_per_stream=48, batch=8`, **either**: the fraction of scored targets that
> are PAD is **below 0.50**; **or** masking the loss to real tokens moves the
> final training loss by **no more than the run-to-run floor**.

Five phases, in this order, because the floor has to exist before a delta can be
judged against it:

1. **PAD census.** Corpus-shaped (the whole committed corpus encoded once) *and*
   run-shaped (one training iteration's batch), separately, because they count
   different populations and will not agree to the last digit. Split by source:
   trailing PAD inside a real sentence vs. every position of an entirely empty
   sentence slot.
2. **The run-to-run floor**, per arm, from two runs differing in **nothing**:
   same seed, same config, same code, separate processes. The statistic is the
   one that gets claimed -- `|final-beat difference|` -- reported alongside the
   trajectory sup-norm so the two are never confused.
3. **The arms**, masked and unmasked, at three seeds each, so every headline
   number carries a spread. An sd of exactly `0.0000` across seeds is a broken
   result, not a clean one.
4. **The shuffle control.** Required of any training run in this project: a row is
   handed another document's entire memory and the real-token loss is re-read. A
   run whose memory contributes ~0 is a run about a different model than the one
   under study, and the number has to be on the page either way.
5. **The ledger.** Every number in the report comes from `runs/<id>/ledger.json`.

🔴 **The device is pinned and it decides whether phase 2 has content.** On CPU the
run-to-run floor is `0.0` (`runs/cycle-arm-identity/RESULTS.md`), which makes the
falsifier's second half *"the delta is more than zero"* -- satisfied by one float
ulp. That is a vacuous discriminator. MPS is the default here for that reason, and
the CPU floor is measured too, and reported as the vacuity it is.

Usage:

    uv run python experiments/cycle-01-masked-loss/run.py              # everything
    uv run python experiments/cycle-01-masked-loss/run.py --single ... # one arm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.model import init_memory  # noqa: E402
from rsr.model.tg.policy_loop import write_at  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import build_vocab, encode, lm_loss, train  # noqa: E402

PAD = 0

#: Everything the falsifier names, plus the two fields it does not. `iters` is
#: `train()`'s own committed default -- the brief does not specify it and the
#: manager's `iters=8` reconstruction was retracted, so the honest choice is the
#: value already in the tree rather than one invented for this cycle.
CONFIG = {
    "d": 128,
    "steps_per_stream": 48,
    "batch": 8,
    "max_tokens": 64,
    "memory_slots": 16,
    "iters": 50,
    "lr": 1e-3,
    "policy": "fifo",
    "corpus": "synthetic, SyntheticConfig(sentences_per_document=48, seed=<seed>)",
    "vocab": "derived from the corpus (NOT --vocab's 50257 default; that is cycle 5)",
}
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# phase 1 -- the PAD census
# --------------------------------------------------------------------------- #


def pad_census(seed: int = 0) -> dict:
    S, L = CONFIG["steps_per_stream"], CONFIG["max_tokens"]
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    ids, mask = encode(docs, build_vocab(docs), max_tokens=L, steps=S)

    tgt, tmask = ids[:, :, 1:], mask[:, :, 1:]
    total = tgt.numel()
    is_pad = tgt == PAD
    # Source A: a PAD target inside a sentence slot that holds at least one real
    # token. Source B: a PAD target in a slot that is entirely empty, i.e. a
    # document with fewer than `steps` sentences. Counted separately because a
    # single fraction hides which one dominates.
    slot_is_real = mask.any(dim=2)
    a = int((is_pad & slot_is_real.unsqueeze(-1)).sum())
    b = int((is_pad & (~slot_is_real).unsqueeze(-1)).sum())
    return {
        "shape": "corpus",
        "n_documents": int(ids.shape[0]),
        "steps_per_document": S,
        "max_tokens": L,
        "denominator_formula": f"{ids.shape[0]} x {S} x ({L} - 1)",
        "n_targets": total,
        "n_pad_targets": a + b,
        "pad_fraction": (a + b) / total,
        "pad_source_A_trailing_in_a_real_sentence": a,
        "pad_source_B_entirely_empty_sentence_slot": b,
        "empty_sentence_slots": int((~slot_is_real).sum()),
        "mask_equals_ids_ne_pad": bool(((tgt != PAD) == tmask).all()),
    }


# --------------------------------------------------------------------------- #
# phase 4 -- the shuffle control
# --------------------------------------------------------------------------- #


def shuffle_control(ckpt: Path, *, seed: int, device: str) -> dict:
    """Real-token loss with the memory honest, vs. with every row handed another
    document's entire memory.

    Two passes over one batch with one model. Pass A runs the loop honestly and
    snapshots the memory it built at every step; pass B replays those exact
    snapshots **row-permuted by a derangement** and reads the loss again. The
    memory contents are therefore identical between passes -- only *whose* they
    are changes.

    `bos_ctx` is left honest on purpose. `bos_replacement_mode == "copy"` puts the
    previous sentence's gestalt at token position 0 *outside* the `use_memory`
    guard, so disturbing it would measure that path instead of cross-attention.
    """
    S, L = CONFIG["steps_per_stream"], CONFIG["max_tokens"]
    batch = CONFIG["batch"]
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    vmap = build_vocab(docs)
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    cfg = TGConfig(
        D=CONFIG["d"],
        V=4 + len(vmap),
        F=int(CONFIG["d"] * 2.6875),
        max_sentence_tokens=L,
        max_sentences_in_short_term=CONFIG["memory_slots"],
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    model = TGModel(cfg).to(device)
    ck.load(ckpt, model=model, restore_rng=False)
    model.eval()  # correction 20: no dropout in a measurement

    ids, mask = ids[:batch].to(device), mask[:batch].to(device)
    lengths = torch.full((batch,), S, device=device)
    perm = torch.roll(torch.arange(batch, device=device), 1)  # a derangement

    snaps, bos_snaps, honest = [], [], []
    with torch.no_grad():
        mem = init_memory(batch, cfg, device=device, dtype=model.embed.weight.dtype)
        bos_ctx = torch.zeros(batch, cfg.D, device=device, dtype=model.embed.weight.dtype)
        bos_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            row_valid = torch.full((batch,), t, device=device) < lengths
            snaps.append((mem.kv.clone(), mem.valid.clone()))
            bos_snaps.append((bos_ctx.clone(), bos_valid.clone()))
            out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
            honest.append(float(lm_loss(out.logits, ids_t, mask_t)))
            write = row_valid & out.has_eos
            victim = torch.zeros(batch, dtype=torch.long, device=device)  # FIFO
            if cfg.use_memory:
                mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (torch.full((batch,), t + 1, device=device) < lengths)

        shuffled = []
        for t in range(S):
            kv, valid = snaps[t]
            bc, bv = bos_snaps[t]
            out = model(ids[:, t], mask[:, t], kv[perm], valid[perm], bc, bv)
            shuffled.append(float(lm_loss(out.logits, ids[:, t], mask[:, t])))

    h = sum(honest) / len(honest)
    s = sum(shuffled) / len(shuffled)
    deltas = [abs(x - y) for x, y in zip(shuffled, honest, strict=True)]
    return {
        "loss_real_tokens_honest_memory": h,
        "loss_real_tokens_shuffled_memory": s,
        "shuffle_delta_nats_per_token": s - h,
        "shuffle_delta_relative": (s - h) / h if h else None,
        "max_per_step_abs_delta": max(deltas),
        "n_steps": len(honest),
        "n_slots_permuted_per_row": CONFIG["memory_slots"],
        "permutation": "torch.roll(arange(batch), 1) -- a derangement, every row "
        "receives another document's entire memory",
    }


# --------------------------------------------------------------------------- #
# one arm, in its own process
# --------------------------------------------------------------------------- #


def single(arm: str, seed: int, replicate: str, device: str, out_dir: Path) -> dict:
    r = train(
        d=CONFIG["d"],
        steps_per_stream=CONFIG["steps_per_stream"],
        batch=CONFIG["batch"],
        vocab=None,
        max_tokens=CONFIG["max_tokens"],
        memory_slots=CONFIG["memory_slots"],
        iters=CONFIG["iters"],
        lr=CONFIG["lr"],
        seed=seed,
        device=device,
        out_dir=out_dir,
        beat_every=1,
        ckpt_every=CONFIG["iters"],
        masked_loss=(arm == "masked"),
    )
    beats = [
        json.loads(ln)
        for ln in (out_dir / "heartbeat.jsonl").read_text().splitlines()
        if json.loads(ln).get("kind") == "beat"
    ]
    r["arm"], r["seed"], r["replicate"], r["device"] = arm, seed, replicate, device
    r["beats_loss"] = [b["loss"] for b in beats]
    r["beats_loss_real_tokens"] = [b["loss_real_tokens"] for b in beats]
    r["n_beats"] = len(beats)
    r["loss_first_beat"] = beats[0]["loss"] if beats else None
    r["loss_real_tokens_first_beat"] = beats[0]["loss_real_tokens"] if beats else None
    r["checkpoint"] = str(out_dir / f"ckpt-{CONFIG['iters']:06d}.pt")
    return r


def _spawn(arm: str, seed: int, replicate: str, device: str, root: Path) -> dict:
    """A separate process per run. Two "identical" runs inside one interpreter
    share more state than their config says they do, and the floor is exactly the
    statistic that would be flattered by it."""
    out_dir = root / f"{arm}-seed{seed}-{replicate}"
    res = out_dir / "result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--replicate",
            replicate,
            "--device",
            device,
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not res.exists():
        raise SystemExit(
            f"{arm}/seed{seed}/{replicate} exited {proc.returncode}\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
        )
    d = json.loads(res.read_text())
    d["exit_code"] = proc.returncode
    return d


# --------------------------------------------------------------------------- #


def _floor(a: dict, b: dict, key: str) -> dict:
    """Two runs differing in nothing. The final-beat difference is the statistic
    this cycle claims moved; the trajectory sup-norm is reported beside it and is
    **not** the bar -- `max_t |x[t] - y[t]| >= |x[-1] - y[-1]|` by construction, so
    comparing a final-beat delta against a sup-norm floor looks rigorous and means
    nothing."""
    xs, ys = a[f"beats_{key}"], b[f"beats_{key}"]
    return {
        "final_beat_abs_diff": abs(xs[-1] - ys[-1]),
        "trajectory_sup_norm": max(abs(x - y) for x, y in zip(xs, ys, strict=True)),
        "bit_identical": xs == ys,
        "n_beats": len(xs),
        "a": xs[-1],
        "b": ys[-1],
    }


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var**0.5


#: 🔒 **Pre-registered, in the commit that precedes the run.** The falsifier's two
#: halves, and the one escape the second half needs: a floor of exactly `0.0` makes
#: *"the delta exceeds the floor"* true of one float ulp, so the discriminator is
#: vacuous and the honest verdict on that half is `inconclusive`, not `survived`.
def decide(pad_fraction: float, delta: float, floor: float) -> tuple[str, str]:
    if pad_fraction <= 0.50:
        return "falsified", f"PAD fraction {pad_fraction:.4f} is at or below 0.50"
    if floor == 0.0:
        return (
            "inconclusive",
            f"half 1 survives (PAD fraction {pad_fraction:.4f} > 0.50), but the "
            f"run-to-run floor measured exactly 0.0, so half 2's test -- 'the "
            f"delta exceeds the floor' -- is satisfied by one float ulp and "
            f"discriminates nothing on this device",
        )
    if delta <= floor:
        return (
            "falsified",
            f"masking moves the loss by {delta:.3e}, within the run-to-run floor "
            f"{floor:.3e}: the defect is real and inert",
        )
    return (
        "survived",
        f"PAD fraction {pad_fraction:.4f} > 0.50 and masking moves the loss by "
        f"{delta:.3e}, {delta / floor:.1f}x the run-to-run floor {floor:.3e}",
    )


def write_ledger(out: dict, led) -> Path:
    """Every number the report cites, each with the artefact it was read out of."""
    c = out["pad_census_corpus"]
    led.note(
        "pad_fraction_corpus_shaped",
        c["pad_fraction"],
        how=(
            f"experiments/cycle-01-masked-loss/run.py::pad_census -- "
            f"(ids[:,:,1:] == 0).sum() / numel over {c['denominator_formula']} "
            f"= {c['n_pad_targets']}/{c['n_targets']}"
        ),
    )
    for key in (
        "n_targets",
        "n_pad_targets",
        "pad_source_A_trailing_in_a_real_sentence",
        "pad_source_B_entirely_empty_sentence_slot",
        "empty_sentence_slots",
    ):
        led.note(f"corpus.{key}", c[key], how="run.py::pad_census, corpus-shaped")
    r = out["pad_census_run"]
    led.note(
        "pad_fraction_run_shaped",
        r["pad_fraction"],
        how=(
            f"heartbeat final beat of masked-seed0-a, field pad_target_fraction; "
            f"denominator {r['denominator_formula']} = {r['n_targets']} per iter"
        ),
    )

    for arm in ("masked", "unmasked"):
        for key in ("loss", "loss_real_tokens"):
            f = out["floor"][arm][key]
            led.note(
                f"floor.{arm}.{key}.final_beat_abs_diff",
                f["final_beat_abs_diff"],
                how=(
                    "two runs at seed 0, identical config and code, separate "
                    "processes; |final-beat difference| -- the statistic claimed"
                ),
            )
            led.note(
                f"floor.{arm}.{key}.trajectory_sup_norm",
                f["trajectory_sup_norm"],
                how=(
                    f"same pair, max_t |x[t]-y[t]| over {f['n_beats']} beats -- "
                    "reported for continuity with runs/cycle-arm-identity, NOT "
                    "the bar this cycle claims against"
                ),
            )
            led.note(
                f"floor.{arm}.{key}.bit_identical",
                f["bit_identical"],
                how="whole beat trajectories compared element-wise",
            )

    stats: dict[str, dict] = {}
    for arm in ("masked", "unmasked"):
        for key in ("loss", "loss_real_tokens"):
            samples = [
                out["runs"][f"{arm}-seed{s}-a"]["final"][key]
                if key == "loss"
                else out["runs"][f"{arm}-seed{s}-a"]["final"]["loss_real_tokens"]
                for s in SEEDS
            ]
            stats[f"{arm}.{key}"] = led.stat(
                f"{arm}.{key}_final",
                samples,
                how=(f"heartbeat final beat, field {key}, seeds {SEEDS}, replicate a"),
            )
        led.stat(
            f"{arm}.loss_first_beat",
            [out["runs"][f"{arm}-seed{s}-a"]["loss_first_beat"] for s in SEEDS],
            how="heartbeat beat 0 -- before any learning, the reduction change alone",
        )

    for arm in ("masked", "unmasked"):
        s = out["shuffle_control"][arm]
        led.note(
            f"shuffle_control.{arm}.delta_nats_per_token",
            s["shuffle_delta_nats_per_token"],
            how=(
                "run.py::shuffle_control -- every row handed another document's "
                "entire 16-slot memory (roll-by-1 derangement), memory contents "
                "replayed from the honest pass so only ownership changes; "
                "real-token NLL, model.eval()"
            ),
        )
        led.note(
            f"shuffle_control.{arm}.max_per_step_abs_delta",
            s["max_per_step_abs_delta"],
            how="same, max over the 48 sentence steps",
        )
        led.note(
            f"shuffle_control.{arm}.loss_real_tokens_honest",
            s["loss_real_tokens_honest_memory"],
            how="same, honest pass",
        )

    delta_loss = abs(stats["masked.loss"]["mean"] - stats["unmasked.loss"]["mean"])
    delta_real = abs(
        stats["masked.loss_real_tokens"]["mean"]
        - stats["unmasked.loss_real_tokens"]["mean"]
    )
    floor = max(
        out["floor"]["masked"]["loss"]["final_beat_abs_diff"],
        out["floor"]["unmasked"]["loss"]["final_beat_abs_diff"],
    )
    floor_real = max(
        out["floor"]["masked"]["loss_real_tokens"]["final_beat_abs_diff"],
        out["floor"]["unmasked"]["loss_real_tokens"]["final_beat_abs_diff"],
    )
    led.note(
        "delta.reported_loss_masked_vs_unmasked",
        delta_loss,
        how="|mean(masked.loss_final) - mean(unmasked.loss_final)| over 3 seeds",
    )
    led.note(
        "delta.real_token_nll_masked_vs_unmasked",
        delta_real,
        how=(
            "|mean(masked.loss_real_tokens_final) - "
            "mean(unmasked.loss_real_tokens_final)| -- the COMMON yardstick; the "
            "row above compares two different objectives and is tautologically "
            "nonzero"
        ),
    )
    led.note("floor.used_for_the_verdict", floor, how="max of the two arms' floors")
    led.note(
        "floor.used_for_the_yardstick", floor_real, how="max of the two arms' floors"
    )

    outcome, detail = decide(c["pad_fraction"], delta_loss, floor)
    out["verdict"] = {"outcome": outcome, "detail": detail}
    led.verdict(
        falsifier=(
            "cycle 1: PAD fraction < 0.50, OR masking moves the final training "
            "loss by no more than the run-to-run floor"
        ),
        outcome=outcome,
        detail=detail,
    )
    return led.write()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--arm", default="masked")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replicate", default="a")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default="cycle-01-masked-loss")
    a = ap.parse_args(argv)

    if a.single:
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.arm, a.seed, a.replicate, a.device, out_dir)
        (out_dir / "result.json").write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k != "beats_loss"}, indent=2))
        return 0

    from ledger import Ledger

    root = ROOT / "runs" / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    device = a.device

    led = Ledger(
        a.run_id,
        cycle=1,
        question=(
            "Does masking the training loss to real tokens move the final "
            "training loss by more than the run-to-run floor, and is the PAD "
            "fraction of scored targets above 0.50?"
        ),
    )
    # 🔒 Frozen BEFORE the run, not after it. `manifest_written_utc` is in the
    # ledger so a reviewer can check the order rather than take the claim.
    manifest = led.manifest(
        {
            **CONFIG,
            "device": device,
            "seeds": SEEDS,
            "replicates_at_seed_0": ["a", "b"],
            "expectation": (
                "Masking RAISES the reported per-token loss, because PAD is "
                "trivially predictable and currently dominates the mean. A masked "
                "loss that is LOWER is a finding, not a bug to tune away."
            ),
            "falsifier": (
                "The reading dies if the PAD fraction of scored targets is below "
                "0.50, or if masking moves the final training loss by no more "
                "than the run-to-run floor at identical seed and config."
            ),
            "floor_statistic": (
                "|final-beat difference| between two runs differing in nothing. "
                "NOT the trajectory sup-norm of runs/cycle-arm-identity, which is "
                ">= it by construction."
            ),
            "device_note": (
                "MPS. On CPU the floor is 0.0 and half 2 of the falsifier is "
                "vacuous; that is measured here too and reported as vacuity."
            ),
        }
    )
    print(f"manifest frozen: {manifest}", flush=True)

    gates = root / "gates.json"
    if gates.exists():
        for g in json.loads(gates.read_text()):
            led.command(g["argv"], exit_code=g["exit_code"], note=g.get("note", ""))

    out: dict = {"config": {**CONFIG, "device": device, "seeds": SEEDS}}
    out["pad_census_corpus"] = pad_census(seed=0)

    runs = {}
    for arm in ("masked", "unmasked"):
        for seed in SEEDS:
            reps = ("a", "b") if seed == 0 else ("a",)
            for rep in reps:
                runs[(arm, seed, rep)] = _spawn(arm, seed, rep, device, root)
                print(f"  ran {arm} seed={seed} rep={rep}", flush=True)

    out["runs"] = {f"{k[0]}-seed{k[1]}-{k[2]}": v for k, v in runs.items()}
    out["floor"] = {
        arm: {
            "loss": _floor(runs[(arm, 0, "a")], runs[(arm, 0, "b")], "loss"),
            "loss_real_tokens": _floor(
                runs[(arm, 0, "a")], runs[(arm, 0, "b")], "loss_real_tokens"
            ),
        }
        for arm in ("masked", "unmasked")
    }
    out["pad_census_run"] = {
        "shape": "run (one training iteration's batch)",
        "n_targets": runs[("masked", 0, "a")]["final"]["n_targets_scored"],
        "n_pad_targets": runs[("masked", 0, "a")]["final"]["n_pad_targets_scored"],
        "pad_fraction": runs[("masked", 0, "a")]["final"]["pad_target_fraction"],
        "denominator_formula": (
            f"{CONFIG['batch']} x {CONFIG['steps_per_stream']} x "
            f"({CONFIG['max_tokens']} - 1)"
        ),
    }
    out["shuffle_control"] = {
        arm: shuffle_control(
            Path(runs[(arm, 0, "a")]["checkpoint"]), seed=0, device=device
        )
        for arm in ("masked", "unmasked")
    }

    steps_done = [v["steps_done"] for v in out["runs"].values()]
    led.run_meta(
        device=device,
        seeds_actually_run=sorted({v["seed"] for v in out["runs"].values()}),
        steps_requested=CONFIG["iters"],
        steps_done=max(steps_done) if len(set(steps_done)) == 1 else min(steps_done),
    )
    led.note(
        "steps_done_per_run",
        {k: v["steps_done"] for k, v in out["runs"].items()},
        how="train() return value, counted by the loop, not the request",
    )
    led.note(
        "runs_actually_executed",
        sorted(out["runs"]),
        how="one subprocess each; every exit code read from the process, not asserted",
    )
    led.note(
        "run_exit_codes",
        {k: v["exit_code"] for k, v in out["runs"].items()},
        how="subprocess.CompletedProcess.returncode",
    )
    for k, v in out["runs"].items():
        led.command(
            f"uv run python experiments/cycle-01-masked-loss/run.py --single "
            f"--arm {v['arm']} --seed {v['seed']} --replicate {v['replicate']} "
            f"--device {device} --out-dir runs/{a.run_id}/{k}",
            exit_code=v["exit_code"],
            note=f"{v['n_beats']} beats, config_hash {v['config_hash']}",
        )
    led.status("ok")
    (root / "raw.json").write_text(json.dumps(out, indent=2) + "\n")
    p = write_ledger(out, led)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    print(f"ledger: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
