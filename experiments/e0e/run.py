"""E0e -- the `u_bar` distribution and `E[lifetime]` on a FIFO run (§3.5).

Pre-registration: `experiments/e0e/PREREG.md`, committed alone ahead of this file.
It fixes the model and protocol, every definition below, the `tau` rule, the
consistency checks and the exit codes. This file implements them; it chooses none.

    uv run python experiments/e0e/run.py            # the run: runs/e0e/
    uv run python experiments/e0e/run.py --help

Eval-only FIFO over held-out streams of fresh-stream arm B's three `ckpt-003000`s
(read only), with liveness as a precondition. `r_i` is
`rsr.retention.reward.retrieval_demand(..., gated=True)`, §3.2.1's target, collected
in eval mode (correction 20).

🔴 **It never calls `rsr.constants.record()` and never writes
`measurements/ledger.json`.** Which substrate E1's constants come from is the owner's
open decision D2. The `record()` calls it would make are written into this run's
own ledger as a note, verbatim, and printed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import rsr.constants as C  # noqa: E402
from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main  # noqa: E402
from rsr.metrics.memory_liveness import measure_liveness  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.retention.policy import AttentionTrace, MemoryState  # noqa: E402
from rsr.retention.reward import retrieval_demand  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import (  # noqa: E402
    build_vocab,
    check_vocabulary_closure,
    encode,
    liveness_batch,
)

EXPERIMENT = "experiments/e0e/run.py"
RUN_ID = "e0e"
SEEDS = (0, 1, 2)
SCOPE = "synthetic"

#: PREREG *Model and protocol*: fresh-stream arm B, ckpt 3000, read only.
DEFAULT_CKPT_ROOT = Path(
    "/Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/"
    "runs/fresh-stream/B"
)
CKPT_NAME = "ckpt-003000.pt"
#: PREREG, taken before the PREREG was committed. A mismatch is exit 3.
CKPT_SHA256 = {
    0: "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60",
    1: "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8",
    2: "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8",
}
#: PREREG: held-out documents 64..127 of each seed's generator.
HELDOUT = (64, 128)

#: 🔒 PREREG *Definitions* 4: tau = this quantile of |M u_bar - 1|, so that b
#: updates fire on 1 - TAU_QUANTILE of observations -- the 0.25 of §3.5 item 4.
TAU_QUANTILE = 0.75
#: PREREG *Definitions* 2: half-life = E_lifetime / HALF_LIFE_DIVISOR (§3.5 item 2).
HALF_LIFE_DIVISOR = 4.0
#: PREREG *Consistency checks*: the full-memory sum of r_i.
SUM_TOL = 1e-5
#: The v0.4 band §3.5 item 3 worries about, reported descriptively.
V04_TAU = 0.25

EXPECTED = (
    "E_lifetime = 632/48 = 13.1667 exactly on every seed (every sentence writes; "
    "FIFO lifetimes fixed by S=48, M=16), half-life 3.2917, gamma_b = 0.3038; u_bar "
    "heavy-tailed: fraction inside +/-25% < 0.5 and tau > 0.25 (low confidence); "
    "all three seeds live."
)


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_e0e.py drives each on hand-built inputs)
# --------------------------------------------------------------------------- #


def r_of(trace: AttentionTrace, n_live: int, M: int) -> torch.Tensor:
    """`[M]` §3.2.1's target: norm-weighted, gated (correction 17), fill-rescaled."""
    return retrieval_demand(trace, n_live=n_live, capacity=M, gated=True)


def half_life(e_lifetime: float) -> float:
    """§3.5 item 2: `E[lifetime] / 4` steps."""
    return e_lifetime / HALF_LIFE_DIVISOR


def ema_alpha(h: float) -> float:
    """Per-step EMA weight whose half-life is `h` steps: `(1 - a)^h = 1/2`."""
    return 1.0 - 2.0 ** (-1.0 / h)


def ema_series(rs: list[float], alpha: float, init: float | None) -> list[float]:
    """Post-update `u_bar` after each observation. `init=None` starts at the first
    observation (the descriptive variant); the primary starts at `1/M`."""
    out: list[float] = []
    u = init
    for r in rs:
        u = r if u is None else (1.0 - alpha) * u + alpha * r
        out.append(u)
    return out


def tau_from(ubars: list[float], M: int, q: float = TAU_QUANTILE) -> float:
    """PREREG *Definitions* 4: the `q` quantile of `|M u_bar - 1|` (linear
    interpolation, `torch.quantile`)."""
    d = (torch.tensor(ubars, dtype=torch.float64) * M - 1.0).abs()
    return float(torch.quantile(d, q))


def fraction_inside(ubars: list[float], M: int, tau: float) -> float:
    """Fraction of observations with `(1-tau)/M <= u_bar <= (1+tau)/M`."""
    d = (torch.tensor(ubars, dtype=torch.float64) * M - 1.0).abs()
    return float((d <= tau).double().mean())


def gamma_b(e_lifetime: float, b_max: float) -> float:
    """§3.5 item 4, the registry's `_derive_gamma_b`, on the would-be value."""
    return b_max / (0.25 * e_lifetime)


def live_seeds(bands: dict[int, str]) -> list[int]:
    """PREREG: only seeds whose Amendment 1 band is `live` contribute."""
    return sorted(s for s, b in bands.items() if b == "live")


@dataclass
class Sentence:
    """One written sentence of one stream, keyed by its write step."""

    written: int
    rs: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    full: list[bool] = field(default_factory=list)
    evicted: bool = False

    @property
    def lifetime(self) -> int:
        """PREREG *Definitions* 1: steps at which it is live in the memory a
        forward pass reads -- its number of observations."""
        return len(self.rs)


def lifetime_stats(sentences: list[Sentence], S: int) -> dict:
    """PREREG *Definitions* 1 over a list of sentences (any number of streams)."""
    lt = [s.lifetime for s in sentences]
    ev = [s.lifetime for s in sentences if s.evicted]
    wb = [s.lifetime if s.evicted else S - s.written for s in sentences]
    return {
        "E_lifetime": sum(lt) / len(lt),
        "n_written": len(lt),
        "evicted_only_mean": sum(ev) / len(ev) if ev else None,
        "n_evicted": len(ev),
        "write_to_boundary_mean": sum(wb) / len(wb),
    }


def consistency_problems(sentences: list[Sentence], step_sums: list[tuple]) -> list[str]:
    """PREREG *Consistency checks*. `step_sums`: `(stream, step, n_live, M, sum_r)`."""
    out = []
    for stream, t, n_live, M, total in step_sums:
        if n_live == M and abs(total - 1.0) > SUM_TOL:
            out.append(f"stream {stream} step {t}: full-memory sum r = {total!r}")
    for s in sentences:
        if s.steps and s.steps[0] <= s.written:
            out.append(f"sentence written at {s.written} observed at {s.steps[0]}")
        if s.steps != list(range(s.written + 1, s.written + 1 + len(s.steps))):
            out.append(f"sentence written at {s.written}: non-contiguous {s.steps}")
    return out


# --------------------------------------------------------------------------- #
# the FIFO recorder
# --------------------------------------------------------------------------- #


class E0eRecorder(FIFOPolicy):
    """`FIFOPolicy`, eviction unchanged, recording `r_i` per written sentence.

    Batch 1 per stream, so the row is never ambiguous; a sentence's identity is
    its `written_at` step (slots shift under `write_at`: index is not identity).
    """

    def __init__(self, M: int) -> None:
        self.M = M
        self.by_write: dict[int, Sentence] = {}
        self.step_sums: list[tuple] = []
        self.final_live: set[int] = set()

    def reset(self) -> None:
        self.by_write, self.step_sums, self.final_live = {}, [], set()

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        n_live = slots.n_live
        if n_live == 0:
            return None
        r = r_of(attn, n_live, self.M)
        total = 0.0
        for j in torch.nonzero(slots.live).flatten().tolist():
            w = int(slots.written_at[j])
            s = self.by_write.setdefault(w, Sentence(written=w))
            s.rs.append(float(r[j]))
            s.steps.append(step)
            s.full.append(n_live == self.M)
            total += float(r[j])
        self.step_sums.append((None, step, n_live, self.M, total))
        return None

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        self.by_write.setdefault(step, Sentence(written=step))
        self.final_live = {
            int(slots.written_at[j]) for j in torch.nonzero(slots.live).flatten()
        }
        return None

    def sentences(self) -> list[Sentence]:
        for w, s in self.by_write.items():
            s.evicted = w not in self.final_live
        return [self.by_write[w] for w in sorted(self.by_write)]


def fifo_stream(model: TGModel, ids: torch.Tensor, mask: torch.Tensor, M: int):
    """One stream (`ids`, `mask`: `[S, L]`) under FIFO, eval + no_grad."""
    rec = E0eRecorder(M)
    S = ids.shape[0]
    model.eval()  # correction 20: r_i in eval mode; reward.py refuses otherwise
    with torch.no_grad():
        run_policy_loop(
            model,
            ids.unsqueeze(0),
            mask.unsqueeze(0),
            torch.full((1,), S),
            rec,
            step_fn=lambda *a: torch.zeros(()),
            observe=True,
        )
    return rec.sentences(), rec.step_sums


# --------------------------------------------------------------------------- #
# per seed
# --------------------------------------------------------------------------- #


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_config(seed_dir: Path) -> dict:
    """The checkpoint's own frozen config: its heartbeat header."""
    first = (seed_dir / "heartbeat.jsonl").read_text().splitlines()[0]
    return json.loads(first)["config"]


def heldout(config: dict):
    """PREREG *Streams*: documents 64..127, encoded with documents 0..63's vocab."""
    S, seed = config["steps_per_stream"], config["seed"]
    vocab_docs = config.get("stream", {}).get("vocab_documents", 64)
    vdocs = generate(
        SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=vocab_docs)
    )
    docs = generate(
        SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=HELDOUT[1])
    )[HELDOUT[0] :]
    vocab = build_vocab(vdocs)
    check_vocabulary_closure(docs, vocab)
    return encode(docs, vocab, max_tokens=config["tg"]["max_sentence_tokens"], steps=S)


def measure_seed(seed: int, ckpt_root: Path) -> dict:
    seed_dir = ckpt_root / f"seed{seed}"
    path = seed_dir / CKPT_NAME
    got = sha256(path)
    if got != CKPT_SHA256[seed]:
        raise RuntimeError(f"seed {seed}: {path} sha256 {got} != PREREG's")
    config = header_config(seed_dir)
    cfg = TGConfig(**config["tg"])
    model = TGModel(cfg)
    ck.load(path, model=model, restore_rng=False)  # refuses a quarantined ckpt
    model.eval()
    torch.manual_seed(seed)
    decoy = TGModel(cfg)
    lids, lmask = liveness_batch(config)
    liveness = measure_liveness(model, decoy, lids, lmask, seed=seed)

    ids, mask = heldout(config)
    M = cfg.max_sentences_in_short_term
    sentences: list[Sentence] = []
    sums: list[tuple] = []
    per_stream_ok = []
    for i in range(ids.shape[0]):
        ss, st = fifo_stream(model, ids[i], mask[i], M)
        n_obs = sum(1 for x in st for _ in range(x[2]))
        per_stream_ok.append(n_obs == sum(s.lifetime for s in ss))
        sentences += ss
        sums += [(i, *x[1:]) for x in st]
    return {
        "seed": seed,
        "ckpt": str(path),
        "ckpt_sha256": got,
        "config": config,
        "M": M,
        "S": config["steps_per_stream"],
        "liveness": liveness,
        "sentences": sentences,
        "step_sums": sums,
        "n_streams": int(ids.shape[0]),
        "obs_count_ok": all(per_stream_ok),
    }


def ubar_distribution(
    sentences: list[Sentence], M: int, alpha: float, *, init: float | None
):
    """Every post-update u_bar, with its full-memory flag."""
    vals, full = [], []
    for s in sentences:
        vals += ema_series(s.rs, alpha, init)
        full += s.full
    return vals, full


def summarise(per_seed: dict[int, dict], b_max: float) -> dict:
    """PREREG *Definitions* over the live seeds, pooled, plus per seed."""
    M = next(iter(per_seed.values()))["M"]
    S = next(iter(per_seed.values()))["S"]
    pooled = [s for r in per_seed.values() for s in r["sentences"]]
    lt = lifetime_stats(pooled, S)
    e_lt = lt["E_lifetime"]
    h = half_life(e_lt)
    a = ema_alpha(h)
    ub, full = ubar_distribution(pooled, M, a, init=1.0 / M)
    ub_first, _ = ubar_distribution(pooled, M, a, init=None)
    ub_full = [u for u, f in zip(ub, full, strict=True) if f]
    tau = tau_from(ub, M)
    seeds = {}
    for s, r in per_seed.items():
        lts = lifetime_stats(r["sentences"], S)
        us, _ = ubar_distribution(
            r["sentences"], M, ema_alpha(half_life(lts["E_lifetime"])), init=1.0 / M
        )
        seeds[s] = {**lts, "tau": tau_from(us, M), "n_obs": len(us)}
    return {
        "M": M,
        "S": S,
        **lt,
        "half_life": h,
        "ema_alpha": a,
        "tau": tau,
        "tau_quantile": TAU_QUANTILE,
        "n_obs": len(ub),
        "fraction_inside_tau": fraction_inside(ub, M, tau),
        "fraction_inside_v04_025": fraction_inside(ub, M, V04_TAU),
        "tau_at_50pct_firing": tau_from(ub, M, 0.5),
        "tau_init_first_obs": tau_from(ub_first, M),
        "tau_full_memory_only": tau_from(ub_full, M),
        "n_obs_full_memory": len(ub_full),
        "ubar_mean": sum(ub) / len(ub),
        "ubar_min": min(ub),
        "ubar_max": max(ub),
        "M_ubar_quantiles": {
            q: float(torch.quantile(torch.tensor(ub, dtype=torch.float64) * M, q))
            for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        },
        "b_max": b_max,
        "gamma_b": gamma_b(e_lt, b_max),
        "per_seed": seeds,
    }


def would_be_record_calls(summary: dict, run_id: str) -> list[str]:
    """The `record()` calls this run does NOT make (D2 is the owner's)."""
    return [
        f'rsr.constants.record("E_lifetime", {summary["E_lifetime"]!r}, '
        f'experiment="E0e", run_id="{run_id}", scope="{SCOPE}", note="FIFO, '
        f'fresh-stream arm B ckpt 3000, held-out docs 64..127, live seeds")',
        f'rsr.constants.record("tau", {summary["tau"]!r}, experiment="E0e", '
        f'run_id="{run_id}", scope="{SCOPE}", note="q{TAU_QUANTILE} of |M u_bar - 1|, '
        f'half-life E_lifetime/4, u_bar init 1/M (PREREG)")',
    ]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog="e0e", description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--run-id", default=RUN_ID)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    a = p.parse_args(argv)

    import ledger as ledger_mod

    led = ledger_mod.Ledger(
        a.run_id,
        question="E0e (§3.5): the u_bar distribution and E[lifetime] on a FIFO run "
        "-> tau, E_lifetime, the EMA half-life and gamma_b (not recorded: D2)",
    )
    led.manifest(
        {
            "experiment": EXPERIMENT,
            "prereg": "experiments/e0e/PREREG.md",
            "run_id": a.run_id,
            "seeds": a.seeds,
            "ckpt_root": str(a.ckpt_root),
            "ckpt_name": CKPT_NAME,
            "ckpt_sha256": {str(k): v for k, v in CKPT_SHA256.items()},
            "heldout_documents": list(HELDOUT),
            "tau_quantile": TAU_QUANTILE,
            "half_life_divisor": HALF_LIFE_DIVISOR,
            "sum_tol": SUM_TOL,
            "scope": SCOPE,
            "device": "cpu",
            "policy": "fifo",
            "r_i": "rsr.retention.reward.retrieval_demand(gated=True), eval mode",
            "ubar_init": "1/M",
            "expected": EXPECTED,
            "falsifier": "NONE (§2 sense): a Sprint-2 instrument measurement",
        }
    )
    led.run_meta(device="cpu")
    argv_s = " ".join(["uv run python", EXPERIMENT, *(argv or sys.argv[1:])])
    torch.set_num_threads(1)

    per_seed: dict[int, dict] = {}
    try:
        for s in a.seeds:
            print(f"seed {s}: measuring", flush=True)
            per_seed[s] = measure_seed(s, a.ckpt_root)
    except (ck.CheckpointError, OSError, RuntimeError, ValueError) as e:
        led.note("error", f"{type(e).__name__}: {e}", how="measure_seed raised")
        led.status("did_not_run")
        led.command(argv_s, exit_code=int(Exit.DID_NOT_RUN))
        led.write()
        return did_not_run(f"{type(e).__name__}: {e}")

    led.run_meta(seeds_actually_run=sorted(per_seed))
    bands = {s: r["liveness"]["band"] for s, r in per_seed.items()}
    for s, r in per_seed.items():
        lv = r["liveness"]
        led.note(
            f"seed{s}.liveness",
            {k: lv.get(k) for k in ("band", "ratio", "A_trained", "A_decoy", "reason")},
            how="rsr.metrics.memory_liveness.measure_liveness vs untrained decoy",
        )
    live = live_seeds(bands)
    if not live:
        led.status("did_not_run")
        led.command(argv_s, exit_code=int(Exit.DID_NOT_RUN))
        led.write()
        return did_not_run(f"no seed is live: {bands}")

    problems = []
    for s, r in per_seed.items():
        problems += [
            f"seed {s}: {x}" for x in consistency_problems(r["sentences"], r["step_sums"])
        ]
        if not r["obs_count_ok"]:
            problems.append(f"seed {s}: observations != sum of lifetimes")

    b_max = float(C.get("b_max"))
    summary = summarise({s: per_seed[s] for s in live}, b_max)
    calls = would_be_record_calls(summary, a.run_id)

    for k in (
        "E_lifetime", "n_written", "evicted_only_mean", "n_evicted",
        "write_to_boundary_mean", "half_life", "ema_alpha", "tau", "n_obs",
        "fraction_inside_tau", "fraction_inside_v04_025", "tau_at_50pct_firing",
        "tau_init_first_obs", "tau_full_memory_only", "n_obs_full_memory",
        "ubar_mean", "ubar_min", "ubar_max", "M_ubar_quantiles", "b_max", "gamma_b",
    ):  # fmt: skip
        led.note(k, summary[k], how=f"pooled over live seeds {live} (PREREG)")
    for key in ("E_lifetime", "tau", "evicted_only_mean"):
        led.stat(
            f"per_seed.{key}",
            [summary["per_seed"][s][key] for s in live],
            how=f"per live seed {live}; each seed's own half-life for tau",
            allow_single=len(live) < 2,
        )
    led.note("live_seeds", live, how="Amendment 1 band == live")
    led.note("consistency_problems", problems, how="PREREG consistency checks")
    led.note(
        "would_be_record_calls_NOT_MADE",
        calls,
        how="D2 is the owner's: printed and ledgered, never executed",
    )
    # One code, one status: a consistency failure is exit 1 and "failed", a NaN
    # tau is exit 3 and "did_not_run" (it wins, as it always did). Never "ok"
    # beside a non-zero exit.
    tau_nan = math.isnan(summary["tau"])
    if tau_nan:
        code, status = Exit.DID_NOT_RUN, "did_not_run"
    elif problems:
        code, status = Exit.FAIL, "failed"
    else:
        code, status = Exit.OK, "ok"
    led.status(status)
    led.command(argv_s, exit_code=int(code))
    out = led.write()
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=1))
    print("per_seed:", json.dumps(summary["per_seed"], indent=1))
    print("record() calls NOT made (D2):")
    for c in calls:
        print("  " + c)
    print(f"ledger: {out}; problems: {problems}")
    if tau_nan:
        return did_not_run("tau is NaN")
    return code


if __name__ == "__main__":
    run_main(main)
