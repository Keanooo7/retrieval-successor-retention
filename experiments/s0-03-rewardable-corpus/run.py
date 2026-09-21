"""S0-03 -- is the rewardable synthetic corpus actually testing retrieval?

Pre-registration: `experiments/s0-03-rewardable-corpus/PREREG.md`, committed ahead
of any number this script produces. `decide()` below is its rule, transcribed.

Per seed (0, 1, 2), in its own process: train `experiments/shuffle-control/run.py`'s
`CONFIG` with `masked_loss=True` and `srep_norm_reg_weight=0.0` on the S0-03 corpus
(the committed `rsr.train.loop.train`, FIFO, CPU). Then, in this process, read the
**answer-token** NLL of the trained checkpoint under FIFO, bucketed by gap, in four
memory conditions:

* `live` -- the honest FIFO loop;
* `slots_zeroed` -- every memory slot's content replaced by zeros (validity as in
  the honest loop). The brief's literal "all `M` slots zeroed";
* `gate_zeroed` -- every `memory_gate` zeroed, so cross-attention contributes
  nothing (`memory_liveness.with_memory_disabled`);
* `gate_zeroed_bos_off` -- as `gate_zeroed`, and the bos-copy path (the previous
  sentence's gestalt at token 0, which sits OUTSIDE the memory) switched off too.
  Diagnostic only: it is what separates the gap-1 bucket from the memory.

on two document sets: `heldout` (documents 64..127 of the seed's generator, never
trained on -- PRIMARY) and `train` (documents 0..63, the ones trained on).

🔴 **No shuffle / liveness readout is run here.** That is the decisive run's
measurement (`experiments/decisive-shuffle/PREREG.md`), pre-registered separately.

Usage:

    uv run python experiments/s0-03-rewardable-corpus/run.py --device cpu
    uv run python experiments/s0-03-rewardable-corpus/run.py --device cpu \\
        --iters 2 --run-id s0-03-smoke                      # plumbing only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import (  # noqa: E402
    ANSWER_SYMBOLS,
    SyntheticConfig,
    fraction_of_pairs_beyond,
    generate,
)
from rsr.metrics.memory_liveness import with_memory_disabled  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.model import init_memory  # noqa: E402
from rsr.model.tg.policy_loop import write_at  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import (  # noqa: E402
    answer_targets,
    build_vocab,
    encode,
    lm_token_losses,
    train,
)


def _shuffle_control_config() -> dict:
    """The brief: "use `experiments/shuffle-control/run.py`'s `CONFIG`". Loaded from
    that file rather than retyped, so the two cannot drift."""
    path = ROOT / "experiments" / "shuffle-control" / "run.py"
    spec = importlib.util.spec_from_file_location("_shuffle_control_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.CONFIG)


_SC = _shuffle_control_config()
CONFIG = {
    **{k: v for k, v in _SC.items() if k not in ("corpus", "measure_documents")},
    "masked_loss": True,  # the brief, overriding shuffle-control's False
    "srep_norm_reg_weight": 0.0,  # the brief (equal to shuffle-control's)
    "corpus": (
        "synthetic S0-03, SyntheticConfig(sentences_per_document=48, seed=<seed>), "
        "answer_in_stream=True (the default)"
    ),
    "measure_heldout": "generate(SyntheticConfig(n_documents=128, seed=<seed>))[64:]",
    "measure_train": "generate(SyntheticConfig(seed=<seed>))  # docs 0..63",
}
SEEDS = [0, 1, 2]
EXPERIMENT = "experiments/s0-03-rewardable-corpus/run.py"
M = CONFIG["memory_slots"]
CHANCE = math.log(len(ANSWER_SYMBOLS))

#: 🔒 PREREG.md. The one tolerance the brief did not give; chosen before any
#: number existed and used identically in all three rules.
MARGIN = 0.10

CONDITIONS = ("live", "slots_zeroed", "gate_zeroed", "gate_zeroed_bos_off")
BUCKETS = {
    "all": lambda g: g >= 1,
    "gap_eq_1": lambda g: g == 1,
    "gap_ge_2": lambda g: g >= 2,
    "gap_2_to_M": lambda g: (g >= 2) & (g <= M),
    "gap_lt_M": lambda g: (g >= 1) & (g < M),
    "gap_eq_M": lambda g: g == M,
    "gap_le_M": lambda g: (g >= 1) & (g <= M),
    "gap_gt_M": lambda g: g > M,
}


def _cfg(V: int) -> TGConfig:
    d = CONFIG["d"]
    return TGConfig(
        D=d,
        V=V,
        F=int(d * 2.6875),
        max_sentence_tokens=CONFIG["max_tokens"],
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )


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
        memory_slots=M,
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


def _argv(seed: int, iters: int, device: str, out_dir: Path) -> list[str]:
    return [
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


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #


def _sets(seed: int):
    """-> {name: docs}, the vocab (from the TRAINING corpus, as `train()` builds
    it) and V. Raises if a held-out word is missing from that vocab."""
    S = CONFIG["steps_per_stream"]
    tr = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    ho = generate(SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=128))
    ho = ho[len(tr) :]
    vmap = build_vocab(tr)
    missing = set(build_vocab(ho)) - set(vmap)
    if missing:
        raise SystemExit(f"seed {seed}: held-out words not in the vocab: {missing}")
    return {"heldout": ho, "train": tr}, vmap, 4 + len(vmap)


@torch.no_grad()
def answer_readout(model: TGModel, ids, mask, tmask, gap, sym_ids, *, cond: str):
    """Per answer target: NLL, argmax-correct, NLL renormalised over the 16 answer
    symbols, and its gap. FIFO, eval mode, one pass over the stream."""
    cfg = model.cfg
    was = model.training
    model.eval()
    B, S, L = ids.shape
    dtype = model.embed.weight.dtype
    out_nll, out_ok, out_r16, out_gap = [], [], [], []
    real_sum, real_n = 0.0, 0
    try:
        mem = init_memory(B, cfg, device=ids.device, dtype=dtype)
        bos_ctx = torch.zeros(B, cfg.D, device=ids.device, dtype=dtype)
        bos_valid = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            kv = torch.zeros_like(mem.kv) if cond == "slots_zeroed" else mem.kv
            bv = torch.zeros_like(bos_valid) if cond.endswith("bos_off") else bos_valid
            out = model(ids_t, mask_t, kv, mem.valid, bos_ctx, bv)
            per = lm_token_losses(out.logits, ids_t).view(B, L - 1)
            real = mask_t[:, 1:]
            real_sum += float(per[real].double().sum())
            real_n += int(real.sum())
            am = tmask[:, t, 1:]
            if am.any():
                lg = out.logits[:, :-1][am]  # [n, V]
                tgt = ids_t[:, 1:][am]
                out_nll.append(per[am])
                out_ok.append(lg.argmax(-1) == tgt)
                lp16 = torch.log_softmax(lg[:, sym_ids], dim=-1)
                pos = (sym_ids.unsqueeze(0) == tgt.unsqueeze(1)).float().argmax(-1)
                out_r16.append(-lp16.gather(1, pos.unsqueeze(1)).squeeze(1))
                out_gap.append(gap[:, t].unsqueeze(1).expand(B, L - 1)[am])
            write = out.has_eos  # every row runs the full stream, as in train()
            victim = torch.zeros(B, dtype=torch.long, device=ids.device)  # FIFO
            if cfg.use_memory:
                mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (t + 1 < S)
    finally:
        model.train(was)
    return {
        "nll": torch.cat(out_nll).double(),
        "ok": torch.cat(out_ok),
        "nll16": torch.cat(out_r16).double(),
        "gap": torch.cat(out_gap),
        "real_token_nll": real_sum / max(real_n, 1),
    }


def _summarise(r: dict) -> dict:
    out = {"real_token_nll": r["real_token_nll"]}
    for b, sel in BUCKETS.items():
        m = sel(r["gap"])
        n = int(m.sum())
        out[b] = {
            "n": n,
            "answer_nll": float(r["nll"][m].mean()) if n else None,
            "answer_acc": float(r["ok"][m].float().mean()) if n else None,
            "answer_nll_over_16": float(r["nll16"][m].mean()) if n else None,
        }
    return out


def measure(ckpt: Path, seed: int, device: str) -> dict:
    sets, vmap, V = _sets(seed)
    trained = TGModel(_cfg(V)).to(device)
    ck.load(ckpt, model=trained, restore_rng=False)
    gated = with_memory_disabled(trained)
    sym_ids = torch.tensor([vmap[s] for s in ANSWER_SYMBOLS], device=device)
    res: dict = {}
    for name, docs in sets.items():
        S, L = CONFIG["steps_per_stream"], CONFIG["max_tokens"]
        ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
        tmask, gap = answer_targets(docs, vmap, max_tokens=L, steps=S)
        ids, mask, tmask, gap = (x.to(device) for x in (ids, mask, tmask, gap))
        res[name] = {
            "n_documents": len(docs),
            "fraction_of_pairs_gap_gt_M": fraction_of_pairs_beyond(docs, M),
        }
        for cond in CONDITIONS:
            model = gated if cond.startswith("gate_zeroed") else trained
            r = answer_readout(model, ids, mask, tmask, gap, sym_ids, cond=cond)
            res[name][cond] = _summarise(r)
    res["memory_gates"] = [
        float(b.memory_gate.detach()) for b in trained.blocks if b.block_type == "C"
    ]
    return res


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #


def _v(per, s, ds, cond, bucket, key="answer_nll"):
    return per[s][ds][cond][bucket][key]


#: 🔒 Transcribed from PREREG.md, committed ahead of this run.
def decide(per: dict[int, dict]) -> dict:
    seeds = sorted(per)
    ds = "heldout"
    # Bar 1: with memory zeroed, answer NLL is not materially BELOW chance.
    zero = {
        c: [_v(per, s, ds, c, "gap_ge_2") for s in seeds]
        for c in ("slots_zeroed", "gate_zeroed")
    }
    bar1 = all(x >= CHANCE - MARGIN for xs in zero.values() for x in xs)
    # Bar 2: under FIFO, live answer NLL at gap > M is worse than at gap < M.
    contrast = [
        _v(per, s, ds, "live", "gap_gt_M") - _v(per, s, ds, "live", "gap_lt_M")
        for s in seeds
    ]
    bar2 = all(x > 0 for x in contrast)
    # H4: retrieval shown iff live beats slots-zeroed by MARGIN at 2 <= gap <= M.
    lz = [
        _v(per, s, ds, "slots_zeroed", "gap_2_to_M")
        - _v(per, s, ds, "live", "gap_2_to_M")
        for s in seeds
    ]
    live_2m = [_v(per, s, ds, "live", "gap_2_to_M") for s in seeds]
    zero_2m = [_v(per, s, ds, "slots_zeroed", "gap_2_to_M") for s in seeds]
    shown = all(x >= MARGIN for x in lz)
    both_chance = all(abs(x - CHANCE) < MARGIN for x in live_2m + zero_2m)
    if not bar1:
        outcome, why = "falsified", "zeroed memory reads the answer below chance"
    elif shown and bar2:
        outcome, why = "survived", "retrieval shown and eviction bites"
    elif shown:
        outcome, why = "inconclusive", "retrieval shown, but eviction does not bite"
    elif both_chance:
        outcome, why = "inconclusive", "corpus built, retrieval not shown"
    else:
        outcome, why = "inconclusive", "retrieval not shown (not both at chance)"
    return {
        "bar1_zeroed_not_below_chance": bar1,
        "bar2_fifo_gt_M_worse_than_lt_M": bar2,
        "h4_retrieval_shown": shown,
        "h4_live_and_zeroed_both_at_chance": both_chance,
        "outcome": outcome,
        "detail": why,
    }


# --------------------------------------------------------------------------- #


def write_ledger(out: dict, led) -> Path:
    per = out["per_seed"]
    seeds = sorted(per)
    how = (
        f"{EXPERIMENT}::measure -> answer_readout; trained ckpt, eval(), FIFO, "
        f"answer targets from rsr.train.loop.answer_targets, NLL over the full "
        f"softmax (nats per answer token), mean within seed; seeds {seeds}"
    )
    for ds in ("heldout", "train"):
        for cond in CONDITIONS:
            for b in BUCKETS:
                for key in ("answer_nll", "answer_acc", "answer_nll_over_16"):
                    xs = [_v(per, s, ds, cond, b, key) for s in seeds]
                    k = f"{ds}.{cond}.{b}.{key}"
                    h = f"{how}; set {ds}; condition {cond}; bucket {b}"
                    if any(x is None for x in xs):  # an empty bucket on some seed
                        led.note(k, xs, how=f"{h}; EMPTY bucket on a seed, no stat")
                    else:
                        led.stat(k, xs, how=h)
            led.stat(
                f"{ds}.{cond}.real_token_nll",
                [per[s][ds][cond]["real_token_nll"] for s in seeds],
                how=f"{how}; set {ds}; condition {cond}; all real targets",
            )
        for z in ("slots_zeroed", "gate_zeroed"):
            for b in ("all", "gap_eq_1", "gap_ge_2", "gap_2_to_M", "gap_gt_M"):
                led.stat(
                    f"{ds}.{z}_minus_live.{b}.answer_nll",
                    [_v(per, s, ds, z, b) - _v(per, s, ds, "live", b) for s in seeds],
                    how=f"per seed: {ds}.{z}.{b}.answer_nll - {ds}.live.{b}.answer_nll",
                )
        for cond in ("live", "slots_zeroed"):
            for lo in ("gap_lt_M", "gap_le_M", "gap_2_to_M"):
                led.stat(
                    f"{ds}.{cond}.gap_gt_M_minus_{lo}.answer_nll",
                    [
                        _v(per, s, ds, cond, "gap_gt_M") - _v(per, s, ds, cond, lo)
                        for s in seeds
                    ],
                    how=(
                        f"per seed: {ds}.{cond}.gap_gt_M.answer_nll - "
                        f"{ds}.{cond}.{lo}.answer_nll (FIFO)"
                    ),
                )
        led.stat(
            f"{ds}.fraction_of_pairs_gap_gt_M",
            [per[s][ds]["fraction_of_pairs_gap_gt_M"] for s in seeds],
            how=f"rsr.data.synthetic.fraction_of_pairs_beyond(docs, {M}); set {ds}",
        )
        for s in seeds:
            led.note(
                f"seed{s}.{ds}.n_answer_targets_by_bucket",
                {b: per[s][ds]["live"][b]["n"] for b in BUCKETS},
                how=f"{how}; counts of answer targets per bucket",
            )
    for s in seeds:
        led.note(f"seed{s}.memory_gates", per[s]["memory_gates"], how="trained ckpt")
        f = out["runs"][f"seed{s}"]["final"]
        for k in ("loss_answer_tokens", "answer_target_fraction", "loss_real_tokens"):
            led.note(
                f"seed{s}.train_final.{k}",
                f[k],
                how="rsr.train.loop.train return['final'] -- last training batch",
            )
    led.note("chance_ln16", CHANCE, how="math.log(len(ANSWER_SYMBOLS))")
    led.note("margin", MARGIN, how="PREREG.md, fixed before the run")
    v = decide(per)
    out["verdict"] = v
    led.note("decision", v, how=f"{EXPERIMENT}::decide (PREREG.md rule)")
    led.verdict(
        falsifier=(
            "PREREG.md: the S0-03 corpus tests retrieval only if (bar 1) answer NLL "
            "with memory zeroed is not below ln16 - 0.10 at gap >= 2 on held-out "
            "documents, every seed. H4: retrieval is shown only if live beats "
            "slots-zeroed by >= 0.10 at 2 <= gap <= M on every seed"
        ),
        outcome=v["outcome"],
        detail=v["detail"],
    )
    return led.write()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=CONFIG["iters"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default="s0-03-rewardable-corpus")
    ap.add_argument("--threads-per-seed", type=int, default=5)
    a = ap.parse_args(argv)

    if a.single:
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.seed, a.iters, a.device, out_dir)
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
        return 0

    from ledger import Ledger

    root = ROOT / "runs" / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(
        a.run_id,
        question=(
            "On the S0-03 corpus, is answer-token NLL at chance with memory zeroed, "
            "does live memory beat zeroed at 2 <= gap <= M, and is gap > M worse "
            "than gap < M under FIFO?"
        ),
    )
    manifest = led.manifest(
        {
            **CONFIG,
            "iters": a.iters,
            "device": a.device,
            "seeds": SEEDS,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "threads_per_seed": a.threads_per_seed,
            "seeds_trained_in_parallel": True,
            "chance_ln16": CHANCE,
            "margin": MARGIN,
            "conditions": list(CONDITIONS),
            "buckets": list(BUCKETS),
            "prereg": "experiments/s0-03-rewardable-corpus/PREREG.md",
            "expected": (
                "Bar 1 passes (zeroed ~ chance or above at gap >= 2 held-out). "
                "Gap-1 answers fall below chance live via the bos-copy path. At "
                "2 <= gap <= M, live is at most modestly below zeroed after 300 "
                "iters; gap > M ~ chance. Most likely verdict: corpus built, "
                "retrieval not shown beyond gap 1."
            ),
            "falsifier": (
                "'the S0-03 corpus's answer tokens cannot be predicted without the "
                "memory' (bar 1) and 'FIFO eviction raises answer loss at gap > M' "
                "(bar 2)"
            ),
        }
    )
    print(f"manifest frozen: {manifest}", flush=True)

    out: dict = {"config": {**CONFIG, "iters": a.iters, "device": a.device}, "runs": {}}
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(a.threads_per_seed),
        "MKL_NUM_THREADS": str(a.threads_per_seed),
    }
    procs = {}
    for seed in SEEDS:
        od = root / f"seed{seed}"
        od.mkdir(parents=True, exist_ok=True)
        argv_ = _argv(seed, a.iters, a.device, od)
        log = open(od / "train.log", "w")  # noqa: SIM115
        procs[seed] = (
            subprocess.Popen(argv_, cwd=ROOT, stdout=log, stderr=log, env=env),
            argv_,
            log,
        )
    per_seed: dict[int, dict] = {}
    failed = []
    for seed, (p, argv_, log) in procs.items():
        rc = p.wait()
        log.close()
        res = root / f"seed{seed}" / "result.json"
        rel = " ".join(
            [
                "uv run python",
                EXPERIMENT,
                *[
                    str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                    for x in argv_[2:]
                ],
            ]
        )
        if rc != 0 or not res.exists():
            led.command(rel, exit_code=rc, note="training child failed")
            failed.append(seed)
            continue
        r = json.loads(res.read_text())
        r["exit_code"] = rc
        out["runs"][f"seed{seed}"] = r
        led.command(rel, exit_code=rc, note=f"config_hash {r['config_hash']}")
    if failed:
        led.run_meta(
            device=a.device,
            seeds_actually_run=sorted(int(k[4:]) for k in out["runs"]),
            steps_requested=a.iters,
        )
        led.status("crashed")
        (root / "raw.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"training failed for seeds {failed}; see runs/{a.run_id}/seed*/train.log")
        led.write()
        return 3

    for seed in SEEDS:
        r = out["runs"][f"seed{seed}"]
        per_seed[seed] = measure(Path(r["checkpoint"]), seed, a.device)
        h = per_seed[seed]["heldout"]
        print(
            f"  seed {seed}: heldout live {h['live']['gap_2_to_M']['answer_nll']:.4f} "
            f"zeroed {h['slots_zeroed']['gap_2_to_M']['answer_nll']:.4f} (2..M)",
            flush=True,
        )

    out["per_seed"] = per_seed
    led.run_meta(
        device=a.device,
        seeds_actually_run=SEEDS,
        steps_requested=a.iters,
        steps_done=min(r["steps_done"] for r in out["runs"].values()),
    )
    led.status("ok")
    (root / "raw.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    p = write_ledger(out, led)
    print(json.dumps(out["verdict"], indent=2))
    print(f"ledger: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
