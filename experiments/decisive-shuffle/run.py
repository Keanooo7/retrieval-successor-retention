"""Decisive run -- is the TG working memory live on the rewardable corpus? (§10.3)

Pre-registration: `experiments/decisive-shuffle/PREREG.md`, committed ahead of this
file, with its Amendment 1. It pins the condition, the three arms, the decision
rule (#17's Amendment 1, per arm) and the secondary readouts. `decide_arm()` and
`overall()` below are that rule, transcribed.

This follows `experiments/shuffle-control/run.py` and changes only (a) the arm
settings, (b) the corpus -- `train()` now generates S0-03's rewardable corpus,
`SyntheticConfig(answer_in_stream=True)` being the default -- and (c) the added
readouts. The instrument is `rsr.metrics.memory_liveness.shuffle_control`, reused.

Per (arm, seed), in its own process with a fixed `torch.set_num_threads`: train
with the committed `rsr.train.loop.train`. Every (arm, seed) result is then
verified against that arm's **frozen manifest entry** (`verify_arm`) -- a config
that does not match refuses the run. Then, in this process, on two batches:

* `train` -- #17's measurement batch, the first 8 training documents (primary);
* `heldout` -- documents 64..71 of the seed's generator stream (PREREG Amendment
  1; descriptive rows and the answer-token readout).

Every number in `RESULTS.md` comes from `runs/<run-id>/ledger.json`.

Usage:

    uv run python experiments/decisive-shuffle/run.py
    uv run python experiments/decisive-shuffle/run.py --iters 2 \\
        --run-id decisive-shuffle-smoke                     # plumbing only
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.metrics.memory_liveness import (  # noqa: E402
    cross_row_cosine,
    random_replacement,
    shuffle_control,
    with_memory_disabled,
)
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import (  # noqa: E402
    _config_hash,
    answer_targets,
    build_vocab,
    encode,
    train,
)

#: PREREG.md "Condition" -- #17's CONFIG with the corpus replaced. `iters` is
#: overridable only for a plumbing smoke; the ledger records what was run.
CONFIG = {
    "d": 128,
    "steps_per_stream": 48,
    "batch": 16,
    "max_tokens": 64,
    "memory_slots": 16,
    "iters": 300,
    "lr": 1e-3,
    "policy": "fifo",
    "corpus": (
        "synthetic, S0-03 rewardable: SyntheticConfig(sentences_per_document=48, "
        "seed=<seed>), answer_in_stream=True (the default)"
    ),
    "vocab": "derived from the training corpus",
    "measure_documents": 8,
    "measure_train": "generate(SyntheticConfig(sentences_per_document=48, "
    "seed=<seed>))[:8]",
    "measure_heldout": "generate(SyntheticConfig(sentences_per_document=48, "
    "seed=<seed>, n_documents=128))[64:72]",
}

#: 🔒 PREREG.md "Arms", transcribed. `None` -> TGConfig's default hinge weight.
ARMS: dict[str, dict] = {
    "A": {"masked_loss": False, "srep_norm_reg_weight": 0.0},
    "B": {"masked_loss": True, "srep_norm_reg_weight": 0.0},
    "C": {"masked_loss": True, "srep_norm_reg_weight": None},
}
SEEDS = [0, 1, 2]
EXPERIMENT = "experiments/decisive-shuffle/run.py"

#: Fixed per training process and recorded in the manifest (Bar 6). Measured at
#: 1/2/4 threads before the run: ~4.6-4.9 s/iter either way, so one thread each.
THREADS_PER_PROCESS = 1
#: The matched-norm random replacement's generator: seeded independently of the
#: model (PREREG Secondary 2). Seed = RANDOM_SEED_BASE + model seed.
RANDOM_SEED_BASE = 10_000
#: Brief: stop starting new trainings at 06:30 local.
NO_NEW_TRAININGS_AFTER = (6, 30)

#: PREREG "Primary readout": #17's Amendment 1 thresholds, fixed in ba71e86.
INERT_MAX_RATIO = 0.01
LIVE_MIN_RATIO = 0.1


def hinge_default() -> float:
    return next(
        f.default
        for f in dataclasses.fields(TGConfig)
        if f.name == "srep_norm_reg_weight"
    )


def resolved_arm(arm: str) -> dict:
    a = dict(ARMS[arm])
    if a["srep_norm_reg_weight"] is None:
        a["srep_norm_reg_weight"] = hinge_default()
    return a


def arm_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def manifest_arms() -> dict:
    """The per-arm entries frozen into the manifest before the first seed."""
    out = {}
    for arm in ARMS:
        r = resolved_arm(arm)
        out[arm] = {"as_registered": ARMS[arm], "resolved": r, "arm_hash": arm_hash(r)}
    return out


class ArmMismatch(SystemExit):
    pass


def verify_arm(frozen_arms: dict, arm: str, result: dict) -> None:
    """Refuse unless the checkpoint's config is the frozen arm's (PREREG
    *Mutation bar*, arm swap). Reads the config `train()` itself returned and the
    hash it stamped, never the script's ARMS, so a swap in the spawn path is
    caught against the manifest."""
    cfg = result["config"]
    got = {
        "masked_loss": cfg["masked_loss"],
        "srep_norm_reg_weight": cfg["srep_norm_reg_weight"],
    }
    want = frozen_arms[arm]
    why = []
    if arm_hash(got) != want["arm_hash"]:
        why.append(f"arm hash {arm_hash(got)} != frozen {want['arm_hash']} ({got})")
    if _config_hash(cfg) != result["config_hash"]:
        why.append("the result's config does not hash to its own config_hash")
    for k in ("iters", "batch", "steps_per_stream", "lr", "policy"):
        exp = CONFIG[k] if k != "iters" else result["steps_requested"]
        if cfg[k] != exp:
            why.append(f"{k}={cfg[k]!r} != {exp!r}")
    if (
        cfg["tg"]["D"] != CONFIG["d"]
        or cfg["tg"]["max_sentences_in_short_term"] != CONFIG["memory_slots"]
    ):
        why.append("D or M differs from CONFIG")
    if why:
        raise ArmMismatch(f"arm {arm}: checkpoint config refused: {why}")


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


def batches(seed: int) -> tuple[dict, int]:
    """-> {"train": (ids, mask, tmask, gap), "heldout": (...)}, V. The vocab is the
    TRAINING corpus's, as `train()` builds it; a held-out word outside it raises."""
    S, L, n = (
        CONFIG["steps_per_stream"],
        CONFIG["max_tokens"],
        CONFIG["measure_documents"],
    )
    tr = generate(SyntheticConfig(sentences_per_document=S, seed=seed))
    ho = generate(SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=128))
    ho = ho[len(tr) :]
    vmap = build_vocab(tr)
    missing = set(build_vocab(ho)) - set(vmap)
    if missing:
        raise SystemExit(f"seed {seed}: held-out words not in the vocab: {missing}")
    out = {}
    for name, docs in (("train", tr[:n]), ("heldout", ho[:n])):
        ids, mask = encode(docs, vmap, max_tokens=L, steps=S)
        tmask, gap = answer_targets(docs, vmap, max_tokens=L, steps=S)
        out[name] = (ids, mask, tmask, gap)
    return out, 4 + len(vmap)


def gap_masks(tmask: torch.Tensor, gap: torch.Tensor) -> dict[str, torch.Tensor]:
    """PREREG Amendment 1's buckets: gap = 1, 2 <= gap <= M, gap > M."""
    M = CONFIG["memory_slots"]
    g = gap.unsqueeze(-1)
    return {
        "answer_all": tmask,
        "gap_1": tmask & (g == 1),
        "gap_2_to_M": tmask & (g >= 2) & (g <= M),
        "gap_gt_M": tmask & (g > M),
    }


# --------------------------------------------------------------------------- #
# one training run, in its own process
# --------------------------------------------------------------------------- #


def single(arm: str, seed: int, iters: int, out_dir: Path) -> dict:
    torch.set_num_threads(THREADS_PER_PROCESS)
    a = ARMS[arm]
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
        device="cpu",
        out_dir=out_dir,
        beat_every=1,
        ckpt_every=iters,
        policy_name=CONFIG["policy"],
        masked_loss=a["masked_loss"],
        srep_norm_reg_weight=a["srep_norm_reg_weight"],
    )
    r.update(
        arm=arm,
        seed=seed,
        checkpoint=str(out_dir / f"ckpt-{iters:06d}.pt"),
        torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(),
    )
    return r


def _argv(arm: str, seed: int, iters: int, root: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--iters",
        str(iters),
        "--out-dir",
        str(root / f"arm{arm}" / f"seed{seed}"),
    ]


def _past_deadline(now: _dt.datetime | None = None) -> bool:
    now = now or _dt.datetime.now()
    h, m = NO_NEW_TRAININGS_AFTER
    return (now.hour, now.minute) >= (h, m) and now.hour < 12


def run_trainings(jobs: list[tuple[str, int]], iters: int, root: Path) -> dict:
    """All jobs in parallel subprocesses. A job not started because of the
    deadline is recorded as not run; it is never silently dropped."""
    procs, results = {}, {}
    for arm, seed in jobs:
        if _past_deadline():
            results[(arm, seed)] = {"status": "not_started_deadline"}
            continue
        argv = _argv(arm, seed, iters, root)
        log = root / f"arm{arm}" / f"seed{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w")  # noqa: SIM115 -- closed below
        procs[(arm, seed)] = (
            subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=fh),
            fh,
            argv,
        )
    for key, (p, fh, argv) in procs.items():
        rc = p.wait()
        fh.close()
        res = Path(argv[-1]) / "result.json"
        if rc != 0 or not res.exists():
            results[key] = {"status": "crashed", "exit_code": rc, "argv": argv}
            continue
        d = json.loads(res.read_text())
        d.update(status="completed", exit_code=rc, argv=argv)
        results[key] = d
    return results


# --------------------------------------------------------------------------- #
# the measurement and its controls
# --------------------------------------------------------------------------- #


def measure(ckpt: Path, seed: int, *, decoy_ckpt: Path | None = None) -> dict:
    """Every readout for one (arm, seed). `decoy_ckpt` exists for the aliasing
    mutation only: pointing it at `ckpt` must give ratio == 1.0 exactly."""
    bs, V = batches(seed)
    trained = TGModel(_cfg(V))
    ck.load(ckpt, model=trained, restore_rng=False)
    torch.manual_seed(seed)
    decoy = TGModel(_cfg(V))
    if decoy_ckpt is not None:
        ck.load(decoy_ckpt, model=decoy, restore_rng=False)

    out: dict = {}
    for name, (ids, mask, tmask, gap) in bs.items():
        n = ids.shape[0]
        tms = gap_masks(tmask, gap)
        g = torch.Generator().manual_seed(RANDOM_SEED_BASE + seed)
        gd = torch.Generator().manual_seed(RANDOM_SEED_BASE + seed)
        out[name] = {
            "reading": shuffle_control(trained, ids, mask, token_masks=tms),
            "control_memory_disabled": shuffle_control(
                with_memory_disabled(trained), ids, mask
            ),
            "control_own_memory": shuffle_control(
                trained, ids, mask, perm=torch.arange(n)
            ),
            "control_live_decoy": shuffle_control(decoy, ids, mask, token_masks=tms),
            "control_random_self": random_replacement(
                trained, ids, mask, generator=g, self_replace=True
            ),
            "random_replacement": random_replacement(
                trained, ids, mask, generator=g, token_masks=tms
            ),
            "random_replacement_decoy": random_replacement(
                decoy, ids, mask, generator=gd
            ),
            "cosine_trained": cross_row_cosine(trained, ids, mask),
            "cosine_decoy": cross_row_cosine(decoy, ids, mask),
        }
    return out


def ratio(m: dict) -> float | None:
    decoy = m["control_live_decoy"]["mean_abs_token_delta"]
    return m["reading"]["mean_abs_token_delta"] / decoy if decoy else None


def random_ratio(m: dict) -> float | None:
    decoy = m["control_live_decoy"]["mean_abs_token_delta"]
    return m["random_replacement"]["mean_abs_token_delta"] / decoy if decoy else None


def bucket_ratio(m: dict, bucket: str) -> float | None:
    t = m["reading"]["by_mask"][bucket]["mean_abs_token_delta"]
    dcy = m["control_live_decoy"]["by_mask"][bucket]["mean_abs_token_delta"]
    return t / dcy if (t is not None and dcy) else None


def controls_pass(m: dict) -> tuple[bool, list[str]]:
    """PREREG decision rule's three controls. The random-self control gates only
    the random-replacement readout (`random_valid`); it is not in the arm rule."""
    why = []
    if not m["control_memory_disabled"]["delta_exactly_zero"]:
        why.append("memory disabled did not read exactly 0.0")
    if not m["control_own_memory"]["delta_exactly_zero"]:
        why.append("own memory replayed did not read exactly 0.0")
    if m["control_live_decoy"]["mean_abs_token_delta"] == 0.0:
        why.append("the live decoy read zero -- the instrument cannot register memory")
    return (not why), why


def random_valid(m: dict) -> bool:
    return bool(m["control_random_self"]["delta_exactly_zero"])


def band(x: float | None) -> str:
    """Amendment 1's bands, applied DESCRIPTIVELY to secondary readouts."""
    if x is None:
        return "undefined"
    if x <= INERT_MAX_RATIO:
        return "<=0.01"
    if x >= LIVE_MIN_RATIO:
        return ">=0.1"
    return "between"


#: 🔒 Transcribed from PREREG.md "Primary readout and decision rule", per arm.
def decide_arm(per_seed: dict[int, dict], *, n_required: int = 3) -> tuple[str, str]:
    """`per_seed[s]` is the `train` batch's readings for one arm."""
    broken = {
        s: controls_pass(m)[1] for s, m in per_seed.items() if not controls_pass(m)[0]
    }
    r = {s: ratio(m) for s, m in per_seed.items()}
    if broken:
        return "inconclusive", f"a control failed: {broken}"
    if any(x == 1.0 for x in r.values()):
        return (
            "inconclusive",
            f"ratio == 1.0 exactly on some seed -- the decoy is the trained model "
            f"(aliasing control): {r}",
        )
    if len(per_seed) < n_required:
        return "inconclusive (partial)", f"only {len(per_seed)} seed(s) ran: {r}"
    if any(x >= LIVE_MIN_RATIO for x in r.values()):
        return "live", f"some seed has ratio >= {LIVE_MIN_RATIO}: {r}"
    if all(x <= INERT_MAX_RATIO for x in r.values()):
        return "inert", f"every seed has ratio <= {INERT_MAX_RATIO}: {r}"
    return "inconclusive", f"between the two thresholds: {r}"


def overall(verdicts: dict[str, str]) -> str:
    """PREREG *Which §10.3 outcome occurred*, read off B and C (A qualifies it)."""
    if any(v == "live" for v in verdicts.values()):
        a = verdicts.get("A")
        return (
            "live on at least one arm: the cause was the objective or the corpus; "
            f"arm A (corpus alone) is {a!r}"
        )
    if verdicts.get("B") == "inert" and verdicts.get("C") == "inert":
        return "inert on B and C: the memory path itself is broken -- escalate"
    return f"neither outcome: reported as is, not rounded ({verdicts})"


# --------------------------------------------------------------------------- #


def _stat(led, key: str, xs: list, *, how: str) -> None:
    """`Ledger.stat`, except a list containing `None` (an empty bucket, a zero
    decoy) is written as a note -- it is not a statistic -- and one seed (a
    partial run) is allowed with sd=None rather than refused."""
    if any(x is None for x in xs):
        led.note(key, xs, how=how + " [contains None, so not a statistic]")
        return
    led.stat(key, xs, how=how, allow_single=len(xs) < 2)


def write_ledger(out: dict, led) -> Path:
    per = out["per_arm_seed"]  # {arm: {seed: measure()}}
    how_r = (
        f"{EXPERIMENT}::measure -> memory_liveness.shuffle_control, roll-by-1 "
        f"derangement of whole 16-slot memory, contents replayed from the honest "
        f"pass; real-token NLL over 8 docs x 48 sentences, model.eval(), CPU"
    )
    verdicts = {}
    for arm in sorted(per):
        seeds = sorted(per[arm])
        tr = {s: per[arm][s]["train"] for s in seeds}
        ho = {s: per[arm][s]["heldout"] for s in seeds}
        _stat(
            led,
            f"arm{arm}.ratio",
            [ratio(tr[s]) for s in seeds],
            how=(
                "PRIMARY. mean_abs_token_delta(trained) / mean_abs_token_delta("
                "untrained decoy, same seed, same config) on #17's measurement batch "
                f"(first 8 training docs) -- PREREG decision rule; seeds {seeds}"
            ),
        )
        _stat(
            led,
            f"arm{arm}.heldout.ratio",
            [ratio(ho[s]) for s in seeds],
            how="DESCRIPTIVE, no verdict (Amendment 1): the same on held-out docs 64..71",
        )
        for bn, b in (("train", tr), ("heldout", ho)):
            _stat(
                led,
                f"arm{arm}.{bn}.random_ratio",
                [random_ratio(b[s]) for s in seeds],
                how=(
                    "DESCRIPTIVE. mean_abs_token_delta(trained, matched-norm Gaussian "
                    "replacement, generator seed 10000+seed) / mean_abs_token_delta("
                    "decoy shuffle) -- PREREG Secondary 2"
                ),
            )
            for who in ("trained", "decoy"):
                _stat(
                    led,
                    f"arm{arm}.{bn}.cosine_{who}",
                    [b[s][f"cosine_{who}"]["mean_offdiag_cosine"] for s in seeds],
                    how=(
                        "DESCRIPTIVE. memory_liveness.cross_row_cosine: mean over "
                        "(step, slot valid in every row) of the mean off-diagonal "
                        "cosine between rows' memory vectors"
                    ),
                )
            for key in (
                "mean_abs_token_delta",
                "delta_nats_per_token",
                "loss_real_tokens_honest_memory",
                "n_tokens_moved",
            ):
                _stat(
                    led,
                    f"arm{arm}.{bn}.reading.{key}",
                    [b[s]["reading"][key] for s in seeds],
                    how=f"{how_r}; batch {bn}; field {key}",
                )
            _stat(
                led,
                f"arm{arm}.{bn}.decoy.mean_abs_token_delta",
                [b[s]["control_live_decoy"]["mean_abs_token_delta"] for s in seeds],
                how=f"{how_r}; untrained decoy; batch {bn}",
            )
            _stat(
                led,
                f"arm{arm}.{bn}.random_decoy_ratio",
                [
                    b[s]["random_replacement_decoy"]["mean_abs_token_delta"]
                    / b[s]["control_live_decoy"]["mean_abs_token_delta"]
                    for s in seeds
                ],
                how="DESCRIPTIVE. decoy's random-replacement A / decoy's shuffle A",
            )
        for bucket in ("answer_all", "gap_1", "gap_2_to_M", "gap_gt_M"):
            _stat(
                led,
                f"arm{arm}.heldout.answer.{bucket}.ratio",
                [bucket_ratio(ho[s], bucket) for s in seeds],
                how=(
                    "SECONDARY 1, descriptive. The shuffle ratio restricted to the "
                    "answer-token supervision mask (loop.answer_targets) on held-out "
                    f"docs, bucket {bucket} (Amendment 1)"
                ),
            )
            _stat(
                led,
                f"arm{arm}.heldout.answer.{bucket}.n_targets",
                [ho[s]["reading"]["by_mask"][bucket]["n_targets"] for s in seeds],
                how="count of answer targets in the bucket, held-out 8 docs",
            )
            for which, k in (("honest_nll", "loss_honest_memory"),):
                _stat(
                    led,
                    f"arm{arm}.heldout.answer.{bucket}.{which}",
                    [ho[s]["reading"]["by_mask"][bucket][k] for s in seeds],
                    how="trained model's answer-token NLL with its own memory, held-out",
                )
        for s in seeds:
            for bn in ("train", "heldout"):
                m = per[arm][s][bn]
                for ctl in (
                    "control_memory_disabled",
                    "control_own_memory",
                    "control_live_decoy",
                    "control_random_self",
                ):
                    for key in ("mean_abs_token_delta", "delta_exactly_zero"):
                        led.note(
                            f"arm{arm}.seed{s}.{bn}.{ctl}.{key}",
                            m[ctl][key],
                            how=f"{EXPERIMENT}::measure, {ctl}",
                        )
                led.note(
                    f"arm{arm}.seed{s}.{bn}.random_band",
                    band(random_ratio(m)) if random_valid(m) else "invalid: self != 0",
                    how="Amendment 1 bands applied descriptively to random_ratio",
                )
                for key in (
                    "memory_gates",
                    "n_sentences",
                    "n_sentences_with_eos",
                    "final_slots_filled_per_row",
                    "max_token_delta",
                    "min_token_delta",
                ):
                    led.note(
                        f"arm{arm}.seed{s}.{bn}.reading.{key}",
                        m["reading"][key],
                        how=how_r,
                    )
        outcome, detail = decide_arm(tr)
        verdicts[arm] = outcome
        led.note(
            f"arm{arm}.verdict",
            {"outcome": outcome, "detail": detail},
            how=("decide_arm: PREREG decision rule per arm on the train batch"),
        )
    ov = overall(verdicts)
    out["verdicts"] = verdicts
    out["overall"] = ov
    led.verdict(
        falsifier=(
            "PREREG: 'the TG working memory is inert on a corpus whose objective "
            "requires retrieval'; any arm with a seed ratio >= 0.1 falsifies it"
        ),
        outcome=(
            "falsified"
            if any(v == "live" for v in verdicts.values())
            else "survived"
            if verdicts.get("B") == "inert" and verdicts.get("C") == "inert"
            else "inconclusive"
        ),
        detail=f"per arm {verdicts}; {ov}",
    )
    return led.write()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=CONFIG["iters"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-id", default="decisive-shuffle")
    a = ap.parse_args(argv)

    if a.single:
        out_dir = Path(a.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        r = single(a.arm, a.seed, a.iters, out_dir)
        (out_dir / "result.json").write_text(json.dumps(r, indent=2, default=str) + "\n")
        return 0

    from ledger import Ledger

    torch.set_num_threads(THREADS_PER_PROCESS)
    root = ROOT / "runs" / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    led = Ledger(
        a.run_id,
        question=(
            "On S0-03's rewardable corpus, does handing a row another document's "
            "entire memory move the real-token loss, per arm (A corpus only, B "
            "masked loss, C masked + hinge), in units of a live decoy?"
        ),
    )
    frozen_arms = manifest_arms()
    manifest = led.manifest(
        {
            **CONFIG,
            "iters": a.iters,
            "device": "cpu",
            "seeds": SEEDS,
            "arms": frozen_arms,
            "torch_version": torch.__version__,
            "torch_num_threads_per_training_process": THREADS_PER_PROCESS,
            "torch_num_threads_measure_process": THREADS_PER_PROCESS,
            "parallel": "all 9 (arm, seed) trainings as concurrent subprocesses",
            "random_replacement_generator_seed": f"{RANDOM_SEED_BASE} + seed",
            "no_new_trainings_after_local": "06:30",
            "prereg": "experiments/decisive-shuffle/PREREG.md (+ Amendment 1)",
            "thresholds": {
                "inert_max_ratio": INERT_MAX_RATIO,
                "live_min_ratio": LIVE_MIN_RATIO,
                "statistic": "mean |token delta|, trained / live decoy (#17 Amendment 1)",
            },
        }
    )
    print(f"manifest frozen: {manifest}", flush=True)

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    results = run_trainings(jobs, a.iters, root)
    out: dict = {"config": {**CONFIG, "iters": a.iters}, "runs": {}, "per_arm_seed": {}}
    seeds_run: dict[str, list[int]] = {arm: [] for arm in ARMS}
    steps_done = []
    for (arm, seed), r in results.items():
        out["runs"][f"arm{arm}.seed{seed}"] = r
        if r["status"] != "completed":
            print(f"  arm {arm} seed {seed}: {r['status']}", flush=True)
            continue
        verify_arm(frozen_arms, arm, r)  # refuses the whole run on a mismatch
        rel = " ".join(
            ["uv run python", EXPERIMENT]
            + [
                str(Path(x).relative_to(ROOT)) if x.startswith(str(ROOT)) else x
                for x in r["argv"][2:]
            ]
        )
        led.command(rel, exit_code=r["exit_code"], note=f"config_hash {r['config_hash']}")
        steps_done.append(r["steps_done"])
        seeds_run[arm].append(seed)
        out["per_arm_seed"].setdefault(arm, {})[seed] = measure(
            Path(r["checkpoint"]), seed
        )
        print(
            f"  arm {arm} seed {seed}: ratio "
            f"{ratio(out['per_arm_seed'][arm][seed]['train'])}",
            flush=True,
        )

    all_run = sorted({s for v in seeds_run.values() for s in v})
    led.run_meta(
        device="cpu",
        seeds_actually_run=all_run,
        steps_requested=a.iters,
        steps_done=min(steps_done) if steps_done else 0,
    )
    led.note("seeds_actually_run_per_arm", seeds_run, how="run_trainings + verify_arm")
    led.note(
        "training_threads_per_process",
        {k: r.get("torch_num_threads") for k, r in out["runs"].items()},
        how="torch.get_num_threads() inside each training subprocess",
    )
    complete = all(len(v) == len(SEEDS) for v in seeds_run.values())
    led.status("ok" if complete else "partial")
    (root / "raw.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    p = write_ledger(out, led)
    print(json.dumps({"verdicts": out["verdicts"], "overall": out["overall"]}, indent=2))
    print(f"ledger: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
