"""Shuffle control -- re-derive §10.3's "the trained memory is inert" from committed code.

Pre-registration: `experiments/shuffle-control/PREREG.md`, committed ahead of this
file. It pins the condition (the 2026-09-18 audit's model), the three controls and
the decision rule. `decide()` below is that rule, transcribed.

Per seed, in its own process: train the audit's model with the committed
`rsr.train.loop.train`. Then, in this process, on the first 8 documents x 48
sentences of that seed's corpus:

1. **the reading** -- `shuffle_control` on the trained checkpoint;
2. **control: memory disabled** -- the same checkpoint, `memory_gate` zeroed, must
   read exactly 0.0;
3. **control: own memory replayed** -- identity permutation, must read exactly 0.0;
4. **control: live decoy** -- the untrained model at that seed, must read non-zero.

Every number in `RESULTS.md` comes from `runs/<run-id>/ledger.json`.

Usage:

    uv run python experiments/shuffle-control/run.py --device cpu
    uv run python experiments/shuffle-control/run.py --device cpu \\
        --iters 2 --run-id shuffle-control-smoke            # plumbing only
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main  # noqa: E402
from rsr.metrics.memory_liveness import (  # noqa: E402
    shuffle_control,
    with_memory_disabled,
)
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import build_vocab, encode, train  # noqa: E402

#: PREREG.md "Condition". `iters` is overridable only so the plumbing can be
#: smoke-tested; the ledger records what was actually run.
CONFIG = {
    "d": 128,
    "steps_per_stream": 48,
    "batch": 16,
    "max_tokens": 64,
    "memory_slots": 16,
    "iters": 300,
    "lr": 1e-3,
    "policy": "fifo",
    "masked_loss": False,
    "srep_norm_reg_weight": 0.0,
    "corpus": "synthetic, SyntheticConfig(sentences_per_document=48, seed=<seed>)",
    "vocab": "derived from the corpus",
    "measure_documents": 8,
}
SEEDS = [0, 1, 2]
EXPERIMENT = "experiments/shuffle-control/run.py"

#: PREREG.md "Amendment 1", which supersedes its original decision rule: the
#: signed mean passes a live memory (it cancels), so the statistic is mean
#: |per-token delta| as a ratio to the live decoy's at the same seed.
INERT_MAX_RATIO = 0.01
LIVE_MIN_RATIO = 0.1


def ratio(m: dict) -> float | None:
    decoy = m["control_live_decoy"]["mean_abs_token_delta"]
    return m["reading"]["mean_abs_token_delta"] / decoy if decoy else None


def _cfg(V: int) -> TGConfig:
    d = CONFIG["d"]
    return TGConfig(
        D=d,
        V=V,
        F=int(d * 2.6875),
        max_sentence_tokens=CONFIG["max_tokens"],
        max_sentences_in_short_term=CONFIG["memory_slots"],
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )


def _measure_batch(seed: int):
    S, L = CONFIG["steps_per_stream"], CONFIG["max_tokens"]
    docs = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    vmap = build_vocab(docs)
    ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
    n = CONFIG["measure_documents"]
    return ids[:n], mask[:n], 4 + len(vmap)


# --------------------------------------------------------------------------- #
# one training run, in its own process
# --------------------------------------------------------------------------- #


def single(seed: int, iters: int, device: str, out_dir: Path) -> dict:
    r = train(
        d=CONFIG["d"],
        steps_per_stream=CONFIG["steps_per_stream"],
        batch=CONFIG["batch"],
        vocab=None,
        max_tokens=CONFIG["max_tokens"],
        memory_slots=CONFIG["memory_slots"],
        iters=iters,
        lr=CONFIG["lr"],
        seed=seed,
        device=device,
        out_dir=out_dir,
        beat_every=1,
        ckpt_every=iters,
        policy_name=CONFIG["policy"],
        masked_loss=CONFIG["masked_loss"],
        srep_norm_reg_weight=CONFIG["srep_norm_reg_weight"],
    )
    r["seed"] = seed
    r["checkpoint"] = str(out_dir / f"ckpt-{iters:06d}.pt")
    return r


def _spawn(seed: int, iters: int, device: str, root: Path) -> tuple[dict, list[str]]:
    out_dir = root / f"seed{seed}"
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--seed",
        str(seed),
        "--iters",
        str(iters),
        "--device",
        device,
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    res = out_dir / "result.json"
    if proc.returncode != 0 or not res.exists():
        # A real failure (1), chosen explicitly rather than defaulted by a bare
        # `raise SystemExit(msg)`: the arm's child crashed or wrote no result on
        # a committed, fixed config. S0-05 records this as a decided ambiguity.
        refuse(
            Exit.FAIL,
            f"seed{seed} exited {proc.returncode}\n"
            f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}",
        )
    d = json.loads(res.read_text())
    d["exit_code"] = proc.returncode
    return d, argv


# --------------------------------------------------------------------------- #
# the measurement and its controls
# --------------------------------------------------------------------------- #


def measure(ckpt: Path, seed: int, device: str) -> dict:
    ids, mask, V = _measure_batch(seed)
    ids, mask = ids.to(device), mask.to(device)

    trained = TGModel(_cfg(V)).to(device)
    ck.load(ckpt, model=trained, restore_rng=False)

    torch.manual_seed(seed)
    untrained = TGModel(_cfg(V)).to(device)

    n = ids.shape[0]
    return {
        "reading": shuffle_control(trained, ids, mask),
        "control_memory_disabled": shuffle_control(
            with_memory_disabled(trained), ids, mask
        ),
        "control_own_memory": shuffle_control(
            trained, ids, mask, perm=torch.arange(n, device=device)
        ),
        "control_live_decoy": shuffle_control(untrained, ids, mask),
    }


def controls_pass(m: dict) -> tuple[bool, list[str]]:
    why = []
    if not m["control_memory_disabled"]["delta_exactly_zero"]:
        why.append("memory disabled did not read exactly 0.0")
    if not m["control_own_memory"]["delta_exactly_zero"]:
        why.append("own memory replayed did not read exactly 0.0")
    if m["control_live_decoy"]["mean_abs_token_delta"] == 0.0:
        why.append("the live decoy read zero -- the instrument cannot register memory")
    return (not why), why


#: 🔒 Transcribed from PREREG.md Amendment 1, committed ahead of this change.
def decide(per_seed: dict[int, dict]) -> tuple[str, str]:
    broken = {
        s: controls_pass(m)[1] for s, m in per_seed.items() if not controls_pass(m)[0]
    }
    if broken:
        return (
            "inconclusive",
            f"a control failed, so the instrument is not shown to work: {broken}",
        )
    r = {s: ratio(m) for s, m in per_seed.items()}
    if any(x >= LIVE_MIN_RATIO for x in r.values()):
        return (
            "falsified",
            f"on some seed the trained model's tokens move at >= {LIVE_MIN_RATIO} "
            f"of a live memory's rate (mean |token delta| ratio to decoy): {r}",
        )
    if all(x <= INERT_MAX_RATIO for x in r.values()):
        return (
            "survived",
            f"every seed's mean |token delta| is <= {INERT_MAX_RATIO} of the live "
            f"decoy's ({r}), and all three controls read as required on every seed",
        )
    return "inconclusive", f"between the two thresholds: {r}"


# --------------------------------------------------------------------------- #


def write_ledger(out: dict, led) -> Path:
    per = out["per_seed"]
    seeds = sorted(per)
    how_reading = (
        f"{EXPERIMENT}::measure -> memory_liveness.shuffle_control on the trained "
        f"checkpoint; every row handed another document's 16-slot memory "
        f"(roll-by-1 derangement), contents replayed from the honest pass; "
        f"real-token NLL over {CONFIG['measure_documents']} docs x 48 sentences, "
        f"model.eval()"
    )
    led.stat(
        "ratio_to_live_decoy",
        [ratio(per[s]) for s in seeds],
        how=(
            "mean_abs_token_delta(trained) / mean_abs_token_delta(untrained decoy, "
            "same seed) -- PREREG.md Amendment 1's primary statistic"
        ),
    )
    for key in (
        "mean_abs_token_delta",
        "delta_nats_per_token",
        "delta_relative",
        "max_token_delta",
        "min_token_delta",
        "loss_real_tokens_honest_memory",
    ):
        led.stat(
            f"reading.{key}",
            [per[s]["reading"][key] for s in seeds],
            how=f"{how_reading}; field {key}; seeds {seeds}",
        )
    for s in seeds:
        r = per[s]["reading"]
        for key in (
            "delta_exactly_zero",
            "n_tokens_moved",
            "n_real_tokens",
            "n_sentences",
            "n_sentences_with_eos",
            "final_slots_filled_per_row",
            "memory_gates",
        ):
            led.note(f"seed{s}.reading.{key}", r[key], how=how_reading)
        for ctl in (
            "control_memory_disabled",
            "control_own_memory",
            "control_live_decoy",
        ):
            c = per[s][ctl]
            for key in (
                "mean_abs_token_delta",
                "delta_nats_per_token",
                "delta_exactly_zero",
                "n_tokens_moved",
            ):
                led.note(
                    f"seed{s}.{ctl}.{key}",
                    c[key],
                    how=f"{EXPERIMENT}::measure, {ctl} (PREREG.md 'Controls')",
                )
    led.note(
        "audit_2026_09_18_for_comparison",
        {
            "delta": "exactly 0.0",
            "max_token_delta": 1.2e-4,
            "min_token_delta": -1.9e-4,
            "n_sentences": 384,
            "memory_gate_range": [0.949, 0.989],
        },
        how=(
            "docs/RESEARCH-CONTEXT.md §10.3 -- PROSE, from an uncommitted script on "
            "uncommitted checkpoints. Reported beside the reading, NOT the bar."
        ),
    )
    outcome, detail = decide(per)
    out["verdict"] = {"outcome": outcome, "detail": detail}
    led.verdict(
        falsifier=(
            "PREREG.md Amendment 1: any seed whose mean |token delta| is >= 0.1 of "
            "the live decoy's falsifies 'the memory is inert'; any failed control "
            "makes the run inconclusive"
        ),
        outcome=outcome,
        detail=detail,
    )
    return led.write()


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=CONFIG["iters"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default="shuffle-control")
    a = ap.parse_args(argv)

    if a.single:
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.seed, a.iters, a.device, out_dir)
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
        return Exit.OK

    from ledger import Ledger

    root = ROOT / "runs" / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(
        a.run_id,
        question=(
            "Does handing a row another document's entire memory move the "
            "real-token loss of the audit's model (300 iters, unmasked, hinge off), "
            "with an instrument shown to read both ways?"
        ),
    )
    manifest = led.manifest(
        {
            **CONFIG,
            "iters": a.iters,
            "device": a.device,
            "seeds": SEEDS,
            "prereg": "experiments/shuffle-control/PREREG.md",
            "thresholds": {
                "inert_max_ratio": INERT_MAX_RATIO,
                "live_min_ratio": LIVE_MIN_RATIO,
                "statistic": "mean |token delta|, trained / live decoy (Amendment 1)",
            },
        }
    )
    print(f"manifest frozen: {manifest}", flush=True)

    out: dict = {"config": {**CONFIG, "iters": a.iters, "device": a.device}, "runs": {}}
    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        r, argv_ = _spawn(seed, a.iters, a.device, root)
        out["runs"][f"seed{seed}"] = r
        rel_argv = " ".join(
            [
                "uv run python",
                EXPERIMENT,
                *[
                    str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                    for x in argv_[2:]
                ],
            ]
        )
        led.command(
            rel_argv, exit_code=r["exit_code"], note=f"config_hash {r['config_hash']}"
        )
        per_seed[seed] = measure(Path(r["checkpoint"]), seed, a.device)
        print(f"  seed {seed}: {json.dumps(per_seed[seed]['reading'])[:300]}", flush=True)

    out["per_seed"] = per_seed
    steps_done = [r["steps_done"] for r in out["runs"].values()]
    led.run_meta(
        device=a.device,
        seeds_actually_run=SEEDS,
        steps_requested=a.iters,
        steps_done=min(steps_done),
    )
    led.status("ok")
    (root / "raw.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    p = write_ledger(out, led)
    print(json.dumps(out["verdict"], indent=2))
    print(f"ledger: {p}")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
