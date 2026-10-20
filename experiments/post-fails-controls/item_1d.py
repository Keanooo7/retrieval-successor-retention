"""Item 1d (post-fails-controls PREREG §6): model-free residency of FIFO, the oracle
and FactFiller over a grid of question-spacing mixes.

Pre-registration: `experiments/post-fails-controls/PREREG.md`, committed alone at
9f22d87. `main(argv)` measures and writes `runs/post-fails-controls-1d/{manifest.json,
ledger.json}`; `finalize(argv)` renders `RESULTS.md` and `claims.json`.

Model-free (`rsr.metrics.headroom.simulate`, ``M = 16``): no checkpoint, no forward.
Documents ``[995000, 997048)`` per seed (EVAL, already used), generated under each grid
config through the G-C1 guard (`rsr.data.synthetic._generate_document`, read from the
module at call time). Reading rule: descriptive, gates nothing.

Exit codes (§0): 0 ran, every control passed; 1 a defect after the controls passed; 3
did not run or a control failed.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.baselines.fifo import FIFOPolicy  # noqa: E402
from rsr.baselines.oracle import OraclePolicy  # noqa: E402
from rsr.data import synthetic as syn  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics.headroom import simulate  # noqa: E402

EXPERIMENT = "experiments/post-fails-controls/item_1d.py"
ITEM = "1d"
RUN_ID = "post-fails-controls-1d"
PREREG = "experiments/post-fails-controls/PREREG.md"
PREREG_COMMIT = "9f22d87"
OUT_DEFAULT = ROOT / "runs" / RUN_ID
RUN_PY = "experiments/post-fails-controls/run.py"


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


IC = _load("_pfc_item_1c", "experiments/post-fails-controls/item_1c.py")
R1, LR, OP = IC.R1, IC.LR, IC.OP

# --------------------------------------------------------------------------- #
# 🔒 PREREG §6, transcribed.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
M = 16
S = 48
GEOMETRIC_P = (0.10, 0.20, 0.30, 0.50, 0.80)
HEAVY_TAIL_W = (0.0, 0.10, 0.20, 0.35, 0.50, 0.70)
BASE = (0.30, 0.35)
EVAL = R1.EVAL  # (995000, 997048), already used
ORACLE_GAMMA = R1.ORACLE_GAMMA  # run 1's gamma_o (0.9)
POLICIES = ("fifo", "oracle", "factfiller")
BUCKETS = {
    "all": lambda g: True,
    "gap_2_to_M": lambda g: 2 <= g <= M,
    "gap_gt_M": lambda g: g > M,
}
N_BOOT = 2000
RUN2_OUT = Path(OP.REFS.run2_out)
RUN2_LEDGER = RUN2_OUT / "ledger.json"
RUN2_FF_KEY = "seed{s}.run2.heldout.hit_rate.all.hit"

QUESTION = (
    "Is FIFO's model-free strength a quirk of the question-spacing mix (PREREG §6)?"
)
FALSIFIER = "NONE: descriptive sensitivity table (PREREG §6); it gates nothing."
EXPECTED = (
    "PREREG §10: FIFO's hit rate falls as heavy_tail_weight rises; FactFiller's gain "
    "over FIFO rises."
)


def grid() -> list[tuple[float, float]]:
    return [(p, w) for p in GEOMETRIC_P for w in HEAVY_TAIL_W]


def gkey(p: float, w: float) -> str:
    return f"p{p:g}_w{w:g}"


def config(seed: int, p: float, w: float) -> syn.SyntheticConfig:
    """Run 1's corpus config with the two spacing knobs changed (§6). Every other field
    is `SyntheticConfig`'s default, as `R1.syn_cfg` builds it."""
    return syn.SyntheticConfig(
        sentences_per_document=S, seed=seed, geometric_p=p, heavy_tail_weight=w
    )


def documents(cfg: syn.SyntheticConfig, ids) -> list:
    """Through the guard: the module attribute is looked up at call time (G-C1)."""
    return [syn._generate_document(i, cfg) for i in ids]


def policies(doc, seed: int) -> dict:
    return {
        "fifo": FIFOPolicy(),
        "oracle": OraclePolicy(syn.discounted_demand(doc, ORACLE_GAMMA)),
        "factfiller": LR.FactFiller(doc, random.Random(f"ff:{seed}:{doc.doc_id}")),
    }


def doc_hits(doc, seed: int) -> dict[str, list[list]]:
    """Per policy: ``[[gap, hit], ...]`` (run 1's record ``hits`` format)."""
    return {
        k: [[q["gap"], bool(q["hit"])] for q in simulate(doc, pol, M)["queries"]]
        for k, pol in policies(doc, seed).items()
    }


def per_doc_counts(hits: list[list[list]], pred) -> torch.Tensor:
    """``[D, 2]`` = (queries in the bucket, hits) per document (run 1's convention)."""
    return torch.tensor(
        [
            [sum(1 for g, _ in h if pred(g)), sum(1 for g, x in h if pred(g) and x)]
            for h in hits
        ],
        dtype=torch.float64,
    ).reshape(-1, 2)


def pooled(c: torch.Tensor) -> float | None:
    n = float(c[:, 0].sum())
    return None if n == 0 else float(c[:, 1].sum()) / n


def measure_point(seed: int, p: float, w: float, ids, W: torch.Tensor) -> dict:
    cfg = config(seed, p, w)
    docs = documents(cfg, ids)
    H = {k: [] for k in POLICIES}
    for d in docs:
        for k, v in doc_hits(d, seed).items():
            H[k].append(v)
    out: dict = {"hit": {}, "contrast": {}, "n_queries": {}}
    for b, pred in BUCKETS.items():
        per = {k: per_doc_counts(H[k], pred) for k in POLICIES}
        out["hit"][b] = {k: pooled(per[k]) for k in POLICIES}
        out["n_queries"][b] = int(per["fifo"][:, 0].sum())
        out["contrast"][b] = {
            f"{a}_minus_fifo": R1.contrast(per, a, "fifo", W)
            for a in ("factfiller", "oracle")
        }
    out["fraction_of_pairs_beyond_M"] = syn.fraction_of_pairs_beyond(tuple(docs), M)
    out["n_documents"] = len(docs)
    return {"summary": out, "hits": H}


def run2_records(seed: int, name: str) -> tuple[list[dict], str]:
    p = RUN2_OUT / "heldout" / f"seed{seed}" / f"{name}.jsonl.gz"
    sha = IC.sha256_file(p)
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f], sha


def d_c1(seed: int, hits: dict, ids: list[int]) -> dict:
    """D-C1: at the base point, FIFO's and the oracle's per-document hits equal run 2's
    records, and FactFiller's pooled hit rate equals run 2's ledger, ``==``."""
    out: dict = {}
    for name in ("fifo", "oracle"):
        recs, sha = run2_records(seed, name)
        docs_ok = [r["doc"] for r in recs] == list(ids)
        bad = [
            r["doc"]
            for r, mine in zip(recs, hits[name], strict=False)
            if [[int(g), bool(x)] for g, x in r["hits"]] != mine
        ]
        out[name] = {
            "sha256": sha,
            "n": len(recs),
            "doc_order_ok": docs_ok,
            "n_mismatch": len(bad),
            "mismatch": bad[:10],
            "ok": docs_ok and not bad and len(recs) == len(ids),
        }
    led = IC.ledger_rows(RUN2_LEDGER)
    want = led[RUN2_FF_KEY.format(s=seed)]["value"]["factfiller"]["point"]
    got = pooled(per_doc_counts(hits["factfiller"], BUCKETS["all"]))
    out["factfiller"] = {"mine": got, "ledger": want, "ok": got == want}
    out["run2_ledger_sha256"] = IC.sha256_file(RUN2_LEDGER)
    out["ok"] = all(out[k]["ok"] for k in ("fifo", "oracle", "factfiller"))
    return out


def d_c2() -> dict:
    """D-C2: every grid config passes `SyntheticConfig.__post_init__`."""
    bad = []
    for p, w in grid():
        try:
            config(0, p, w)
        except ValueError as e:
            bad.append([p, w, str(e)])
    return {"n": len(grid()), "bad": bad, "ok": not bad}


def measure_seed(seed: int, *, limit: int | None = None, n_boot: int = N_BOOT) -> dict:
    t0 = time.perf_counter()
    ids = list(range(*EVAL))
    if limit is not None:
        ids = ids[:limit]
    W = R1.boot_weights(len(ids), seed, n_boot)
    out: dict = {"points": {}}
    c1 = None
    for p, w in grid():
        r = measure_point(seed, p, w, ids, W)
        out["points"][gkey(p, w)] = r["summary"]
        if (p, w) == BASE:
            c1 = (
                d_c1(seed, r["hits"], ids)
                if limit is None
                else {"ok": False, "smoke": True}
            )
    out["controls"] = {"D-C1": c1 or {"ok": False, "why": "base point not in grid"}}
    out["seconds"] = time.perf_counter() - t0
    out["n_documents"] = len(ids)
    return out


# --------------------------------------------------------------------------- #
# ledger, execute, main
# --------------------------------------------------------------------------- #


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def write_rows(led, per: dict[int, dict]) -> None:
    how = "item_1d.py::measure_seed; seed order 0,1,2"
    sd = [per[s] for s in SEEDS]

    def stat(key, vals, h=how):
        if all(_finite(v) for v in vals):
            led.stat(key, vals, how=h)
        else:
            led.note(key, vals, how=h + " (not finite on every seed)")

    for p, w in grid():
        g = gkey(p, w)
        pts = [x["points"][g] for x in sd]
        stat(
            f"{g}.fraction_of_pairs_beyond_M",
            [x["fraction_of_pairs_beyond_M"] for x in pts],
        )
        for b in BUCKETS:
            stat(f"{g}.n_queries.{b}", [x["n_queries"][b] for x in pts])
            for k in POLICIES:
                stat(f"{g}.hit.{k}.{b}", [x["hit"][b][k] for x in pts])
            for c in ("factfiller_minus_fifo", "oracle_minus_fifo"):
                for f in ("point", "lo", "hi", "sd"):
                    stat(
                        f"{g}.contrast.{c}.{b}.{f}",
                        [x["contrast"][b][c][f] for x in pts],
                        how + " -> R1.contrast (per document, 2000, paired)",
                    )
    stat("seconds_per_seed", [x["seconds"] for x in sd], "wall clock")
    stat("n_documents", [x["n_documents"] for x in sd])


def manifest_config(args: dict) -> dict:
    return {
        "run_id": RUN_ID,
        "item": ITEM,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": list(SEEDS),
        "grid": {
            "geometric_p": list(GEOMETRIC_P),
            "heavy_tail_weight": list(HEAVY_TAIL_W),
        },
        "base": list(BASE),
        "documents": list(EVAL),
        "already_used": True,
        "M": M,
        "oracle_gamma": ORACLE_GAMMA,
        "factfiller_rng": 'random.Random(f"ff:{seed}:{doc_id}")',
        "bootstrap": {
            "n": N_BOOT,
            "fn": "R1.boot_weights",
            "seed_base": R1.BOOT_SEED_BASE,
        },
        "run2_out": str(RUN2_OUT),
        "device": "cpu",
        "args": args,
    }


def execute(led, a) -> Exit:
    g = IC.Globals(led, reads_substrate=False, args=a)
    failed = g.start()
    c2 = d_c2()
    led.note("D-C2", c2, how="item_1d.py::d_c2")
    if not c2["ok"]:
        failed.append("D-C2")
    per: dict[int, dict] = {}
    if not failed or a.limit is not None:
        for s in a.seeds:
            per[s] = measure_seed(s, limit=a.limit, n_boot=a.n_boot)
            print(f"1d seed {s} done {per[s]['seconds']:.0f}s", flush=True)
    g.end(sorted(per))
    for s, r in per.items():
        led.note(f"seed{s}.controls", r["controls"], how="item_1d.py::measure_seed")
        if not r["controls"]["D-C1"]["ok"]:
            failed.append(f"D-C1 seed{s}")
    if a.limit is not None:
        failed.append(f"smoke: --limit {a.limit} (not the registered set)")
    led.note("controls_failed", failed, how="item_1d.py::execute")
    if sorted(per) == list(SEEDS) and a.limit is None:
        write_rows(led, per)
    if failed:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail="descriptive sensitivity table; every control passed",
    )
    return Exit.OK


def _run(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--limit", type=int, default=None, help="smoke only: first N ids")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--run-id", default=RUN_ID)
    IC.add_global_args(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    from ledger import Ledger

    led = Ledger(a.run_id, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config({k: str(v) for k, v in vars(a).items()}))
    try:
        rc = execute(led, a)
    except Exception as e:
        import traceback

        led.note("exception", traceback.format_exc(), how="item_1d.py::main")
        led.status("crashed")
        sealed = type(e).__name__ in ("SealedRange", "ControlFailure")
        rc = Exit.DID_NOT_RUN if sealed else Exit.FAIL
    led.command(
        [sys.executable, RUN_PY, ITEM] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the item's",
    )
    led.write()
    return status(rc)


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #


def _row(rows: dict, key: str):
    r = rows.get(key)
    return None if r is None else r.get("samples", r.get("value"))


def _fmt(x, nd: int = 4) -> str:
    return "n/a" if not _finite(x) else f"{x:.{nd}f}"


def render(out_root: Path = OUT_DEFAULT) -> Path:
    led = json.loads((Path(out_root) / "ledger.json").read_text())
    rows = {r["key"]: r for r in led["rows"]}
    failed = rows["controls_failed"]["value"]
    cmd = led["commands"][-1] if led["commands"] else {}
    L = [
        "# post-fails-controls 1d: model-free residency over question-spacing mixes",
        "",
        f"Generated by `{EXPERIMENT}::finalize` from `{Path(out_root) / 'ledger.json'}`. "
        "No number below is typed. Descriptive; gates nothing (PREREG §6).",
        "",
        f"- PREREG: `{PREREG}` @ {PREREG_COMMIT} (§6)",
        f"- git sha: `{led['provenance'].get('git_sha')}` · seeds actually run: "
        f"{led['seeds_actually_run']} · status: {led['status']}",
        f"- command: `{cmd.get('argv')}` · exit code: {cmd.get('exit_code')}",
        f"- config hash: `{led['config_hash']}`",
        f"- controls failed: {failed if failed else 'none'}"
        + ("" if not failed else " (NOT READ)"),
        f"- documents: ids {list(EVAL)} per seed (EVAL, already used); at non-base "
        "points they are different documents with the same ids.",
        "",
    ]
    for b in BUCKETS:
        L += [
            f"## Bucket {b}: hit rate FIFO / oracle / FactFiller, and FactFiller - FIFO "
            "point [95% CI], per seed",
            "",
            "| geometric_p | heavy_tail_w | seed | FIFO | oracle | FactFiller | "
            "FF - FIFO | oracle - FIFO | frac beyond M |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for p, w in grid():
            g = gkey(p, w)
            for i, s in enumerate(SEEDS):
                hs = [(_row(rows, f"{g}.hit.{k}.{b}") or [None] * 3)[i] for k in POLICIES]
                cs = []
                for c in ("factfiller_minus_fifo", "oracle_minus_fifo"):
                    v = [
                        (_row(rows, f"{g}.contrast.{c}.{b}.{f}") or [None] * 3)[i]
                        for f in ("point", "lo", "hi")
                    ]
                    cs.append(f"{_fmt(v[0])} [{_fmt(v[1])}, {_fmt(v[2])}]")
                fr = (_row(rows, f"{g}.fraction_of_pairs_beyond_M") or [None] * 3)[i]
                base = " (base)" if (p, w) == BASE else ""
                L.append(
                    f"| {p:g}{base} | {w:g} | {s} | "
                    + " | ".join(_fmt(h) for h in hs)
                    + f" | {cs[0]} | {cs[1]} | {_fmt(fr)} |"
                )
        L.append("")
    L += ["## Controls", ""]
    for s in SEEDS:
        c = (rows.get(f"seed{s}.controls") or {}).get("value") or {}
        L.append(
            f"- seed {s}: D-C1 {'pass' if (c.get('D-C1') or {}).get('ok') else 'FAIL'}"
        )
    L.append(f"- D-C2: {'pass' if rows['D-C2']['value']['ok'] else 'FAIL'}")
    L += [
        "",
        "## What 1d does NOT show (PREREG §6)",
        "",
        "Residency, not accuracy: the reader was trained on the base mix only. Not a "
        "claim about natural text.",
        "",
    ]
    pth = Path(out_root) / "RESULTS.md"
    pth.write_text("\n".join(L) + "\n")
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        f"--slots 1 --wait 3600 -- .venv/bin/python {RUN_PY} {ITEM} recompute"
    )
    claims = []
    for p, w in (BASE, (0.10, 0.70), (0.80, 0.0)):
        g = gkey(p, w)
        for i, s in enumerate(SEEDS):
            for key in (
                f"{g}.hit.fifo.all",
                f"{g}.contrast.factfiller_minus_fifo.all.point",
            ):
                v = _row(rows, key)
                if v is not None:
                    claims.append(
                        {
                            "claim": f"seed {s} {key} = {v[i]!r}",
                            "command": f"{base} --seed {s} --what {key}",
                            "expected": v[i],
                        }
                    )
    (Path(out_root) / "claims.json").write_text(json.dumps(claims, indent=1) + "\n")
    return pth


def recompute(what: str, seed: int) -> str:
    g, rest = what.split(".", 1)
    p, w = (float(x[1:]) for x in g.split("_"))
    ids = list(range(*EVAL))
    r = measure_point(seed, p, w, ids, R1.boot_weights(len(ids), seed, N_BOOT))["summary"]
    parts = rest.split(".")
    if parts[0] == "hit":
        return repr(r["hit"][parts[2]][parts[1]])
    if parts[0] == "contrast":
        return repr(r["contrast"][parts[2]][parts[1]][parts[3]])
    raise ValueError(what)


def finalize(argv: list[str] | None = None) -> int:
    """``run.py finalize 1d [--out DIR]``: RESULTS.md and claims.json, from the ledger
    in DIR (default ``runs/{RUN_ID}``). No number is typed."""
    ap = ArgumentParser(description="finalize 1d")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(list(argv or []))
    print(render(a.out))
    return int(Exit.OK)


def main(argv: list[str] | None = None) -> int:
    """``run.py 1d [measure] [options]`` measures and writes the ledger;
    ``run.py 1d finalize`` / ``run.py finalize 1d`` renders; ``run.py 1d recompute
    --what KEY [--seed S]`` re-derives one value (the verifier's entry)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        return finalize(argv[1:])
    if argv[:1] == ["recompute"]:
        ap = ArgumentParser(description="recompute one 1d value")
        ap.add_argument("--what", required=True)
        ap.add_argument("--seed", type=int, required=True)
        a = ap.parse_args(argv[1:])
        IC.common().install_seal()
        print(recompute(a.what, a.seed))
        return 0
    if argv[:1] == ["measure"]:
        argv = argv[1:]
    return int(_run(argv))


if __name__ == "__main__":
    run_main(main)
