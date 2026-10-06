"""The Expire-Span referendum on frozen arm B -- the harness (PREREG
`experiments/expire-span-referendum/PREREG.md`).

Falsifier 6's referendum on this substrate (spec §5.4, §7; ADR-0010; ruling draft
R-2026-10-06-expire-span-q1-q2-q9): does a tuned Expire-Span [P7] (`grad_path =
predictor`) match or beat RSR run 1's psiC and FIFO on held-out answer accuracy?

The TRAINING launch is not here: it is `launch_es.zsh` (27 jobs, 3 seeds x 9 configs,
`rsr.train.loop --policy expire_span`, run 1's substrate, pool and settings). This file:

* ``grid`` -- the 9 configs, read from the PREREG's machine-read tables (§4.1, §4.2),
  one line each, for the launcher. Nothing here types a grid value: the values are
  CONDITIONAL on the PREREG (ADR-0010 q10 is open, so none enters `rsr.constants`).
* ``preflight`` -- E0: VAL disjoint from the pool, EVAL and every range run 1 checked;
  below `1_000_003`; closed under the vocabulary; the map hash. Before any model loads.
* ``validate --seed s`` -- the controls E1-E5 on each config's training, then the VAL
  pass ([994000, 994512), eval mode, ``no_grad``) for all 9 configs.
* ``select`` -- §6.3: one config, by mean VAL all-query accuracy over the 3 seeds,
  exact (`Fraction`) arithmetic, ties to the lowest grid row.
* ``evaluate --seed s`` -- §6.4: EVAL, ONCE, for the selected config only. Refused while
  any §7.3 delta cell is OPEN, before selection, or if this seed's EVAL marker exists.
* ``finalize`` -- §7: ES - psiC and ES - fifo per seed, paired per-document bootstrap
  with run 1's own resample weights (`boot_weights(n, seed, 2000)`), and the §7.2
  classification.

🔴 Run 1's harness is IMPORTED, never copied (`_load`): `run_batch`, the bootstrap, the
controls' helpers. Its `Recorder` does not forward `memory_weight`, so an Expire-Span
pass through it would run WITHOUT the mask -- ``es_recorder()`` swaps in a subclass
that forwards it, for the duration of a pass, and counts the calls (E7).

Never: trains anything; reads EVAL outside ``evaluate``; calls `rsr.constants.record()`;
fills §15; makes a scaling claim.

Exit codes (`rsr.exit_codes`): 0 ran and every control passed · 3 did not run, or a
control failed (numbers written, marked not read).
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.expire_span import ExpireSpanPolicy  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train import loop  # noqa: E402

EXPERIMENT = "experiments/expire-span-referendum/run.py"
PREREG = ROOT / "experiments/expire-span-referendum/PREREG.md"
LAUNCHER = ROOT / "experiments/expire-span-referendum/launch_es.zsh"
RUN_ID = "expire-span-referendum"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def R1():
    """Run 1's harness (`experiments/rsr-run1/run.py`), unchanged."""
    return _load("_es_ref_rsr_run1", "experiments/rsr-run1/run.py")


# --------------------------------------------------------------------------- #
# Registered values (PREREG §2, §5, §6)
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)  # §2: run 1's (A1.4)
ITERS = 3000  # §2: run 1's (R-1, A4.1)
VAL = (994000, 994512)  # §5
#: §7.1: run 1's measured per-document records are read from here, read-only.
RUN1_OUT = Path(
    "/Users/keanooo7/retrieval-successor-retention/.worktrees/rsr-run1-run/runs/rsr-run1"
)
RUN1_SHA = "ed6302aae37a2831c9b8567b9f80a093b4c00f70"  # §2: run 1's launch head
#: Run 1's launcher's sha256 (§2). This referendum never edits it; a test pins it.
RUN1_LAUNCHER_SHA256 = "cb0bef7f7d9efdd8e3b0a0286d385116aec19a19e8353f5e84d1b6ed4f823f0f"
DEFAULT_OUT = Path("runs/expire-span-referendum")

GRID_HEADING = "### 4.1 The grid (machine-read)"
FIXED_HEADING = "### 4.2 Fixed fields (machine-read)"
DELTA_HEADING = "### 7.3 δ per seed (machine-read; PM-OPEN)"
GRID_HEADER = (
    "| config | L (max_span) | R (ramp) | κ | \u03b1 (loss_coef) |",  # alpha, as §4.1
    "|---|---|---|---|---|",
)
FIXED_HEADER = ("| field | value |", "|---|---|")
DELTA_HEADER = ("| seed | δ |", "|---|---|")
FIXED_FIELDS = ("grad_path", "structured_dropout", "init_span_fraction", "ramp_fraction")
CLASSES = ("SUPERSEDES", "LOSES", "TIES", "MIXED")  # §7.2, in order of precedence
HEADLINE = "acc.all.es_minus_psiC"


class ControlFailure(RuntimeError):
    """A control failed: exit 3, numbers kept, marked not read."""


class Refused(RuntimeError):
    """A precondition refused the step (exit 3, nothing read)."""


# --------------------------------------------------------------------------- #
# The PREREG's machine-read tables
# --------------------------------------------------------------------------- #


def _table(text: str, heading: str, header: tuple[str, str]) -> list[list[str]]:
    """The rows of the first table after ``heading``: its two header lines must be
    exactly ``header``. Cells stripped. Raises `Refused` on any other shape."""
    lines = text.splitlines()
    try:
        i = lines.index(heading)
    except ValueError:
        raise Refused(f"PREREG has no heading {heading!r}") from None
    j = i + 1
    while j < len(lines) and not lines[j].startswith("|"):
        if lines[j].startswith("#"):
            raise Refused(f"no table under {heading!r}")
        j += 1
    if tuple(lines[j : j + 2]) != header:
        raise Refused(f"the table under {heading!r} has header {lines[j : j + 2]!r}")
    rows = []
    for line in lines[j + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip().strip("`") for c in line.strip().strip("|").split("|")])
    return rows


def _num(s: str) -> float:
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)?(e-?[0-9]+)?", s):
        raise Refused(f"{s!r} is not a plain decimal literal")
    return float(s)


def fixed_fields(text: str | None = None) -> dict[str, Any]:
    """§4.2: the fields every config shares, with the ramp's ratio to L."""
    text = PREREG.read_text() if text is None else text
    rows = _table(text, FIXED_HEADING, FIXED_HEADER)
    got = {r[0]: r[1] for r in rows}
    if tuple(got) != FIXED_FIELDS:
        raise Refused(f"§4.2 fields are {tuple(got)}, not {FIXED_FIELDS}")
    if got["grad_path"] != "predictor":  # q1 as ruled (R-2026-10-06)
        raise Refused(f"grad_path {got['grad_path']!r}: q1 was ruled 'predictor'")
    if got["structured_dropout"] not in ("true", "false"):
        raise Refused(f"structured_dropout {got['structured_dropout']!r}")
    return {
        "grad_path": got["grad_path"],
        "structured_dropout": got["structured_dropout"],
        "init_span_fraction": got["init_span_fraction"],
        "ramp_fraction": got["ramp_fraction"],
    }


def grid(text: str | None = None) -> list[dict[str, Any]]:
    """§4.1: nine configs, the full 3 L x 3 kappa cross product, in table order.

    Each row is checked against its own definition (§4.1): `R = ramp_fraction * L`
    and `alpha = kappa / (L * R)` to three significant figures. The literal strings
    are kept for the CLI, so what runs is the PREREG's text, not a re-print of it."""
    text = PREREG.read_text() if text is None else text
    fx = fixed_fields(text)
    rf = _num(fx["ramp_fraction"])
    _num(fx["init_span_fraction"])
    rows = _table(text, GRID_HEADING, GRID_HEADER)
    out = []
    for cid, Ls, Rs, ks, als in rows:
        L, R, k, a = (_num(x) for x in (Ls, Rs, ks, als))
        if abs(R - rf * L) > 1e-9:
            raise Refused(f"{cid}: R = {R} is not {rf} x L = {rf * L}")
        if float(f"{k / (L * R):.2e}") != a:
            raise Refused(f"{cid}: alpha = {a} is not kappa / (L R) = {k / (L * R):.3g}")
        if not re.fullmatch(r"L[0-9]+-k[0-9e.-]+", cid):
            raise Refused(f"config id {cid!r}")
        out.append(
            {
                "config": cid,
                "L": L,
                "R": R,
                "kappa": k,
                "alpha": a,
                "overrides": {
                    "max_span": Ls,
                    "ramp": Rs,
                    "loss_coef": als,
                    "structured_dropout": fx["structured_dropout"],
                    "grad_path": fx["grad_path"],
                    "init_span_fraction": fx["init_span_fraction"],
                },
            }
        )
    ids = [c["config"] for c in out]
    Ls = sorted({c["L"] for c in out})
    ks = sorted({c["kappa"] for c in out})
    cross = {(c["L"], c["kappa"]) for c in out}
    if len(out) != 9 or len(set(ids)) != 9 or len(Ls) != 3 or len(ks) != 3:
        raise Refused(f"§4.1 must be 3 L x 3 kappa = 9 distinct configs, got {ids}")
    if cross != {(L, k) for L in Ls for k in ks}:
        raise Refused("§4.1 is not the full L x kappa cross product")
    return out


def config_by_id(cid: str, text: str | None = None) -> dict[str, Any]:
    for c in grid(text):
        if c["config"] == cid:
            return c
    raise Refused(f"no config {cid!r} in §4.1")


def parsed_overrides(c: dict[str, Any]) -> dict[str, Any]:
    """The overrides exactly as `rsr.train.loop.parse_overrides` reads the CLI's."""
    return loop.parse_overrides([f"{k}={v}" for k, v in c["overrides"].items()])


def delta_per_seed(text: str | None = None) -> dict[int, float] | None:
    """§7.3: δ per seed, or None while any cell reads OPEN (then `evaluate` refuses)."""
    text = PREREG.read_text() if text is None else text
    rows = _table(text, DELTA_HEADING, DELTA_HEADER)
    seeds = [int(r[0]) for r in rows]
    if tuple(seeds) != SEEDS:
        raise Refused(f"§7.3 rows are seeds {seeds}, not {SEEDS}")
    if any(r[1] == "OPEN" for r in rows):
        return None
    return {int(r[0]): _num(r[1]) for r in rows}


# --------------------------------------------------------------------------- #
# E0: the VAL split
# --------------------------------------------------------------------------- #


def check_val(seeds=SEEDS) -> dict:
    """E0 (§5): VAL disjoint from run 1's checked ranges, its pool and its EVAL; below
    K; closed under each seed's vocabulary, whose map hash is run 1's. Generates VAL
    documents only -- never EVAL's."""
    r1 = R1()

    def ov(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])

    out: dict[str, Any] = {
        "val": list(VAL),
        "overlaps": [k for k, r in r1.USED_RANGES.items() if ov(VAL, tuple(r))],
        "pool_disjoint": not ov(VAL, r1.POOL),
        "eval_disjoint": not ov(VAL, r1.EVAL),
        "below_K": VAL[1] <= r1.GEN_SEED_K,
        "seeds": {},
    }
    for s in seeds:
        cfg = r1.prereg_config(s, ITERS)
        vmap = r1.vocab_map(cfg)
        rec = {"vocab_sha256": loop.vocab_sha256(vmap)}
        try:
            loop.check_vocabulary_closure(r1.docs_by_id(cfg, range(*VAL)), vmap)
            rec["closure"] = True
        except ValueError as e:
            rec["closure"], rec["error"] = False, str(e)
        out["seeds"][s] = rec
    bad = []
    if out["overlaps"]:
        bad.append("overlaps")
    bad += [k for k in ("pool_disjoint", "eval_disjoint", "below_K") if not out[k]]
    for s, rec in out["seeds"].items():
        if rec["closure"] is not True:
            bad.append(f"closure[{s}]")
        if rec["vocab_sha256"] != r1.EXPECTED_VOCAB_SHA256:
            bad.append(f"vocab_sha256[{s}]")
    out["ok"], out["failed"] = not bad, bad
    if bad:
        raise ControlFailure(f"E0 failed: {bad}")
    return out


# --------------------------------------------------------------------------- #
# The Expire-Span held-out pass
# --------------------------------------------------------------------------- #


def _es_recorder_cls():
    r1 = R1()

    class ESRecorder(r1.Recorder):
        """Run 1's `Recorder`, plus the one hook it does not forward: Expire-Span's
        `memory_weight`. Without it `run_policy_loop` finds no mask and the pass is
        FIFO-shaped attention under Expire-Span's evictions. Counts its calls (E7)."""

        def memory_weight(self, *args, **kwargs):
            inner = self.inner
            inner.weight_calls = getattr(inner, "weight_calls", 0) + 1
            return inner.memory_weight(*args, **kwargs)

    return ESRecorder


@contextlib.contextmanager
def es_recorder():
    """ADAPTER: for the duration of one pass, run 1's `run_batch` wraps the policy in
    `ESRecorder` (it looks `Recorder` up in its module at call time)."""
    r1 = R1()
    orig = r1.Recorder
    r1.Recorder = _es_recorder_cls()
    try:
        yield
    finally:
        r1.Recorder = orig


def es_policy_from_payload(payload: dict) -> ExpireSpanPolicy:
    """The trained Expire-Span at a checkpoint: `build_policy` from the checkpoint's
    own frozen overrides, then its `w, b` and dropout generator (`load_policy_state`
    refuses another arm's checkpoint)."""
    conf = payload["meta"]["config"]
    if conf.get("policy") != "expire_span":
        raise ControlFailure(f"checkpoint is policy {conf.get('policy')!r}")
    pol = loop.build_policy(
        "expire_span",
        d_model=int(conf["tg"]["D"]),
        steps_per_epoch=None,
        overrides=dict(conf["policy_overrides"]),
        seed=int(conf["seed"]),
        n_cross_layers=int(conf["expire_span_n_layers"]),
    )
    ck.load_policy_state(pol, dict(payload["policy"]))
    return pol


def es_pass(model, pol: ExpireSpanPolicy, docs: list, vmap, r1cfg) -> list[dict]:
    """One Expire-Span pass over ``docs`` in run 1's held-out batches (eval mode,
    ``no_grad``): run 1's per-document records. E7: the mask was read at every
    sentence step of every batch; E8 (C7): residency equals the model-free replay."""
    r1 = R1()
    pol.weight_calls = 0
    recs: list[dict] = []
    n_batches = 0
    with es_recorder():
        for i in range(0, len(docs), r1cfg.eval_batch):
            chunk = docs[i : i + r1cfg.eval_batch]
            recs += r1.run_batch(model, pol, chunk, vmap, r1cfg)
            n_batches += 1
    want = n_batches * r1cfg.steps_per_stream
    if pol.weight_calls != want:
        raise ControlFailure(
            f"E7: the mask was read {pol.weight_calls} times, not {want}"
        )
    bad = [
        r["doc"]
        for r, d in zip(recs, docs, strict=True)
        if not r1.residency_ok(d, r, r1cfg.memory_slots)
    ]
    if bad:
        raise ControlFailure(f"E8: residency mismatch on documents {bad[:10]}")
    return recs


def all_query(records: list[dict]) -> tuple[int, int]:
    """(answers, correct) pooled over ``records`` (S0-03's all-query accuracy)."""
    r1 = R1()
    x = r1.answer_counts(records, r1.buckets_for(r1.M_SLOTS)["all"])
    return int(x[:, 0].sum()), int(x[:, 1].sum())


def _write_gz(path: Path, recs: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# E1-E5: the training each config ran
# --------------------------------------------------------------------------- #


def job_dir(out: Path, cid: str, seed: int) -> Path:
    return Path(out) / cid / f"seed{seed}"


def job_key(cid: str, seed: int) -> str:
    return f"{cid}-s{seed}"


def check_training(out: Path, c: dict, seed: int, substrate_model: dict) -> dict:
    """E1-E5 for one (config, seed); returns the final checkpoint's payload too.

    E1 the launcher's rc is 0 and `steps_done == ITERS`; E2 the frozen config is the
    §4.1 row (overrides, grad path) on run 1's substrate pin, pool, frozen eval mode and
    `srep_norm_reg_weight = 0.0`; E3 the model at step 3000 is bit-equal to the
    substrate's; E4 every step read the same pool ids as run 1's FIFO arm at that seed;
    E5 every logged loss is finite."""
    r1 = R1()
    d = job_dir(out, c["config"], seed)
    k = job_key(c["config"], seed)
    rc_path = Path(out) / "logs" / f"{k}.rc"
    rc = int(rc_path.read_text().strip()) if rc_path.exists() else None
    res = r1.read_cli_result(Path(out) / "logs" / f"{k}.log")
    hb = r1.read_heartbeat(d)
    rec: dict[str, Any] = {
        "rc": rc,
        "steps_done": None if res is None else res["steps_done"],
    }
    rec["E1"] = rc == 0 and res is not None and res["steps_done"] == ITERS
    conf = (res or {}).get("config") or {}
    want_ov = parsed_overrides(c)
    sub = r1.SUBSTRATE[seed]
    rec["E2"] = (
        conf.get("policy") == "expire_span"
        and conf.get("policy_overrides") == dict(sorted(want_ov.items()))
        and (conf.get("expire_span") or {}).get("grad_path") == "predictor"
        and (conf.get("warm_start") or {}).get("sha256") == sub["sha256"]
        and conf.get("freeze_transformer") is True
        and conf.get("frozen_model_mode") == loop.FROZEN_MODEL_MODE_RECORD["eval"]
        and conf.get("srep_norm_reg_weight") == 0.0
        and (conf.get("pool") or {}).get("ids") == list(r1.POOL)
        and conf.get("seed") == seed
        and conf.get("iters") == ITERS
    )
    payload = None
    ckpt = d / f"ckpt-{ITERS:06d}.pt"
    if ckpt.exists():
        payload = ck.load(ckpt, restore_rng=False)
        rec["E3"] = r1.c3_frozen({"es": payload}, substrate_model)["ok"]
    else:
        rec["E3"] = False
    mine = r1.step_pool_ids(r1.read_step_records(d))
    theirs = r1.step_pool_ids(r1.read_step_records(RUN1_OUT / "fifo" / f"seed{seed}"))
    rec["E4"] = (
        len(mine) == ITERS
        and len(theirs) >= ITERS
        and all(mine.get(t) == theirs.get(t) for t in range(ITERS))
    )
    losses = [b.get(x) for b in hb["beats"] for x in ("loss", "loss_lm")]
    rec["E5"] = bool(losses) and all(
        isinstance(v, (int, float)) and torch.isfinite(torch.tensor(float(v)))
        for v in losses
    )
    rec["ok"] = all(rec[e] for e in ("E1", "E2", "E3", "E4", "E5"))
    return {"controls": rec, "payload": payload}


# --------------------------------------------------------------------------- #
# validate, select, evaluate, finalize
# --------------------------------------------------------------------------- #


def validate(seed: int, out: Path) -> dict:
    """§6.2: E0, then for each config E1-E5 and the VAL pass. The summary keeps every
    config's (answers, correct); nothing is selected here."""
    r1 = R1()
    check_val(seeds=(seed,))
    r1cfg = r1.prereg_config(seed, ITERS)
    vmap = r1.vocab_map(r1cfg)
    docs = r1.docs_by_id(r1cfg, range(*VAL))
    substrate_model, _ = r1.load_substrate_model(r1cfg)
    summary: dict[str, Any] = {"seed": seed, "val": list(VAL), "configs": {}}
    vdir = Path(out) / "val" / f"seed{seed}"
    for c in grid():
        tr = check_training(out, c, seed, substrate_model)
        row: dict[str, Any] = {"controls": tr["controls"]}
        if tr["payload"] is not None and tr["controls"]["ok"]:
            model = r1.model_from_payload(tr["payload"]).to(r1cfg.device)
            pol = es_policy_from_payload(tr["payload"])
            t0 = time.perf_counter()
            recs = es_pass(model, pol, docs, vmap, r1cfg)
            row["seconds"] = time.perf_counter() - t0
            row["records_sha256"] = _write_gz(vdir / f"{c['config']}.jsonl.gz", recs)
            row["answers"], row["correct"] = all_query(recs)
        summary["configs"][c["config"]] = row
    summary["ok"] = all(r["controls"]["ok"] for r in summary["configs"].values())
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if not summary["ok"]:
        raise ControlFailure(f"seed {seed}: a training control failed (see summary.json)")
    return summary


def selection(summaries: dict[int, dict], configs: list[dict]) -> dict:
    """§6.3: the config with the highest mean over seeds of VAL all-query accuracy,
    each seed's accuracy pooled over its own VAL queries; exact `Fraction`
    arithmetic, so a tie is an exact tie; ties go to the lowest §4.1 row."""
    if tuple(sorted(summaries)) != SEEDS:
        raise Refused(f"selection needs seeds {SEEDS}, got {sorted(summaries)}")
    table = []
    for i, c in enumerate(configs):
        accs = []
        for s in SEEDS:
            row = summaries[s]["configs"].get(c["config"]) or {}
            if not (row.get("controls") or {}).get("ok") or not row.get("answers"):
                raise Refused(f"config {c['config']} seed {s} has no clean VAL reading")
            accs.append(Fraction(int(row["correct"]), int(row["answers"])))
        mean = sum(accs, Fraction(0)) / len(accs)
        table.append({"row": i, "config": c["config"], "mean": mean, "per_seed": accs})
    best = max(table, key=lambda r: (r["mean"], -r["row"]))
    tied = [r["config"] for r in table if r["mean"] == best["mean"]]
    return {
        "selected": best["config"],
        "tied_at_max": tied,
        "rule": (
            "max mean VAL all-query accuracy over seeds 0-2; exact; "
            "ties -> lowest §4.1 row"
        ),
        "table": [
            {
                "config": r["config"],
                "mean": float(r["mean"]),
                "mean_exact": f"{r['mean'].numerator}/{r['mean'].denominator}",
                "per_seed": [float(a) for a in r["per_seed"]],
            }
            for r in table
        ],
    }


def select(out: Path) -> dict:
    summaries = {}
    for s in SEEDS:
        p = Path(out) / "val" / f"seed{s}" / "summary.json"
        if not p.exists():
            raise Refused(f"no VAL summary for seed {s}: run validate first")
        summaries[s] = {**json.loads(p.read_text()), "seed": s}
    sel = selection(summaries, grid())
    path = Path(out) / "selection.json"
    if path.exists() and json.loads(path.read_text())["selected"] != sel["selected"]:
        raise ControlFailure(f"{path} already selects a different config")
    path.write_text(json.dumps(sel, indent=2))
    return sel


def eval_marker(out: Path, seed: int) -> Path:
    return Path(out) / "eval" / f"seed{seed}" / "EVAL_READ"


def evaluate(seed: int, out: Path, *, delta_text: str | None = None) -> dict:
    """§6.4: the selected config on EVAL, once. The marker is written BEFORE the first
    EVAL document is generated, so a crash mid-pass cannot be retried silently."""
    if delta_per_seed(delta_text) is None:
        raise Refused(
            "§7.3: δ is OPEN for at least one seed; TIES is undefined until an amendment "
            "registers it, and EVAL is never read before the classification is complete"
        )
    sel_path = Path(out) / "selection.json"
    if not sel_path.exists():
        raise Refused("no selection.json: run select first")
    marker = eval_marker(out, seed)
    if marker.exists():
        raise Refused(f"{marker} exists: EVAL is read once (§6.4)")
    sel = json.loads(sel_path.read_text())
    c = config_by_id(sel["selected"])
    r1 = R1()
    r1cfg = r1.prereg_config(seed, ITERS)
    substrate_model, _ = r1.load_substrate_model(r1cfg)
    tr = check_training(out, c, seed, substrate_model)
    if not tr["controls"]["ok"]:
        raise ControlFailure(f"seed {seed}: training controls {tr['controls']}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"config": c["config"], "utc": time.time()}))
    vmap = r1.vocab_map(r1cfg)
    docs = r1.docs_by_id(r1cfg, range(*r1.EVAL))
    loop.check_vocabulary_closure(docs, vmap)
    model = r1.model_from_payload(tr["payload"]).to(r1cfg.device)
    pol = es_policy_from_payload(tr["payload"])
    t0 = time.perf_counter()
    recs = es_pass(model, pol, docs, vmap, r1cfg)
    sha = _write_gz(marker.parent / "es.jsonl.gz", recs)
    rec = {"seed": seed, "config": c["config"], "seconds": time.perf_counter() - t0}
    rec["records_sha256"] = sha
    (marker.parent / "eval.json").write_text(json.dumps(rec, indent=2))
    return rec


def classify(ci: dict[int, dict], delta: dict[int, float] | None) -> str:
    """§7.2, applied in the order SUPERSEDES, LOSES, TIES, MIXED (first match).

    ``ci[s]`` holds ES - psiC's `lo`, `hi` at seed s. TIES reads "|ES - psiC| CI within
    ±δ" as the seed's whole CI inside [-δ_s, δ_s], on all three seeds (§7.2's reading
    notes; flagged for review)."""
    if delta is None:
        raise Refused("§7.3 δ is OPEN: TIES is undefined, so no class is assigned")
    if tuple(sorted(ci)) != SEEDS:
        raise Refused(f"classification needs seeds {SEEDS}, got {sorted(ci)}")
    lo = {s: ci[s]["lo"] for s in SEEDS}
    hi = {s: ci[s]["hi"] for s in SEEDS}
    if any(v is None for v in (*lo.values(), *hi.values())):
        raise ControlFailure("a seed's ES - psiC CI is undefined")
    if all(lo[s] > 0 for s in SEEDS):
        return "SUPERSEDES"
    if all(hi[s] < 0 for s in SEEDS):
        return "LOSES"
    if all(-delta[s] <= lo[s] and hi[s] <= delta[s] for s in SEEDS):
        return "TIES"
    return "MIXED"


def paired_seed(seed: int, out: Path, run1_out: Path = RUN1_OUT) -> dict:
    """§7.1 for one seed: ES (this run) against run 1's psiC and fifo passes on the
    same EVAL documents, with run 1's resample weights."""
    r1 = R1()
    es_path = eval_marker(out, seed).parent / "es.jsonl.gz"
    hd = Path(run1_out) / "heldout" / f"seed{seed}"
    recs = {"es": r1.read_pass(es_path.parent, "es")}
    shas = {"es": _sha(es_path)}
    for arm in ("psiC", "fifo"):
        recs[arm] = r1.read_pass(hd, arm)
        shas[arm] = _sha(hd / f"{arm}.jsonl.gz")
    order = [r["doc"] for r in recs["fifo"]]
    if order != list(range(*r1.EVAL)):
        raise ControlFailure(f"seed {seed}: run 1's fifo pass is not EVAL in order")
    for k, rs in recs.items():
        if [r["doc"] for r in rs] != order:
            raise ControlFailure(f"seed {seed}: the {k} pass is not in EVAL order (E6)")
    W = r1.boot_weights(len(order), seed, r1.N_BOOT)
    buckets = r1.buckets_for(r1.M_SLOTS)
    res: dict[str, Any] = {
        "n_documents": len(order),
        "sha256": shas,
        "acc": {},
        "con": {},
    }
    for b in r1.BUCKET_NAMES:
        per = {k: r1.answer_counts(v, buckets[b]) for k, v in recs.items()}
        res["acc"][b] = {k: r1.level(per[k], W) for k in per}
        res["con"][f"acc.{b}.es_minus_psiC"] = r1.contrast(per, "es", "psiC", W)
        res["con"][f"acc.{b}.es_minus_fifo"] = r1.contrast(per, "es", "fifo", W)
    for kind in ("real", "answer"):
        per = {k: r1.loss_counts(v, kind) for k, v in recs.items()}
        res["con"][f"nll.{kind}.es_minus_psiC"] = r1.contrast(per, "es", "psiC", W)
        res["con"][f"nll.{kind}.es_minus_fifo"] = r1.contrast(per, "es", "fifo", W)
    return res


def finalize(out: Path, run1_out: Path = RUN1_OUT) -> dict:
    delta = delta_per_seed()
    per = {s: paired_seed(s, out, run1_out) for s in SEEDS}
    cls = classify({s: per[s]["con"][HEADLINE] for s in SEEDS}, delta)
    ledger = {
        "run_id": RUN_ID,
        "class": cls,
        "falsifier_6": cls == "SUPERSEDES",
        "delta": delta,
        "selection": json.loads((Path(out) / "selection.json").read_text()),
        "seeds": per,
    }
    (Path(out) / "ledger.json").write_text(json.dumps(ledger, indent=2, default=str))
    return ledger


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(prog="expire-span-referendum", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grid")
    sub.add_parser("preflight")
    for name in ("validate", "evaluate"):
        q = sub.add_parser(name)
        q.add_argument("--seed", type=int, required=True, choices=SEEDS)
        q.add_argument("--out", type=Path, default=DEFAULT_OUT)
    for name in ("select", "finalize"):
        sub.add_parser(name).add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args(argv)
    try:
        if a.cmd == "grid":
            for c in grid():
                kv = " ".join(f"{k}={v}" for k, v in c["overrides"].items())
                print(f"{c['config']} {kv}")
            return status(Exit.OK)
        if a.cmd == "preflight":
            v = check_val()
            print(
                json.dumps(
                    {
                        "E0": v,
                        "grid": [c["config"] for c in grid()],
                        "delta": delta_per_seed(),
                    },
                    indent=2,
                    default=str,
                )
            )
            return status(Exit.OK)
        if a.cmd == "validate":
            print(json.dumps(validate(a.seed, a.out), indent=2, default=str))
        elif a.cmd == "select":
            print(json.dumps(select(a.out), indent=2))
        elif a.cmd == "evaluate":
            print(json.dumps(evaluate(a.seed, a.out), indent=2))
        elif a.cmd == "finalize":
            led = finalize(a.out)
            print(json.dumps({"class": led["class"], "falsifier_6": led["falsifier_6"]}))
        return status(Exit.OK)
    except (ControlFailure, Refused) as e:
        print(f"DID NOT RUN / CONTROL FAILED: {e}", file=sys.stderr)
        return status(Exit.DID_NOT_RUN)


if __name__ == "__main__":
    run_main(main)
