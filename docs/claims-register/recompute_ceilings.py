"""Recompute B0 / gamma-ceiling / CF-ceiling ckpt3000 numbers for one seed from the
archived raw records, through each run's own scoring functions, and print one JSON
line ``{"<ledger key>": value-for-this-seed}`` using the ledger's own key names.

    recompute_ceilings.py b0    --seed S   # B0.measure_seed on lookahead-room-r2 D.pt
    recompute_ceilings.py gamma --seed S   # gamma-ceiling's measure_seed on the same D.pt
    recompute_ceilings.py cf    --seed S   # CF's score_arm on the archived Delta (delta.pt)
    recompute_ceilings.py sample <ledger.json> <key> <index>   # one seed's sample

Raw inputs, read only from the archive (never written):
``~/rsr-runs-archive/lookahead-room/runs/lookahead-room-r2/raw/B-ckpt3000-seed<s>.D.pt``
(sha256 checked against the substrate manifest the runs used) and
``~/rsr-runs-archive/cf-ceiling/runs/cf-ceiling/raw/B-ckpt3000-seed<s>.delta.pt``
(sha256 checked against the CF ledger's ``B.ckpt3000.seed<s>.raw.sha256``).
Documents are the U range the runs used (``lookahead-room``'s ``documents(seed)``).
Exit 3 on a sha mismatch or missing input (did not run, not "found nothing").
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

sys.path.insert(0, str(C.ROOT / "src"))
sys.path.insert(0, str(C.ROOT / "scripts"))

import torch

LR2 = C.ARCHIVE / "lookahead-room" / "runs" / "lookahead-room-r2" / "raw"
CF_RAW = C.ARCHIVE / "cf-ceiling" / "runs" / "cf-ceiling" / "raw"
CKPT = 3000


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, C.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class DidNotRun(Exception):
    pass


def load_D(B0, s: int) -> dict:
    p = LR2 / f"B-ckpt{CKPT}-seed{s}.D.pt"
    want = B0.read_manifest(B0.MANIFEST).get(B0.d_rel(CKPT, s))
    got = sha(p)
    if want is None or got != want:
        raise DidNotRun(f"D.pt sha256 {got} != manifest {want} for {p}")
    return torch.load(p, weights_only=True)


def b0(s: int) -> dict:
    B0 = _load("_cr_b0", "experiments/b0-ceilings/run.py")
    r = B0.measure_seed(s, {CKPT: load_D(B0, s)})
    out: dict = {}
    for ref in B0.REFS:
        for b in B0.BUCKETS:
            out[f"U.hit.{ref}.{b}"] = r["refs"][ref][b]
    for f in ("point", "lo", "hi"):
        out[f"U.cap.oracle.{f}"] = r["oracle"]["cap"][f]
    co = r["ckpt"][CKPT]
    pre = f"B.ckpt{CKPT}"
    for h in B0.HINDSIGHT:
        for f in ("point", "lo", "hi"):
            out[f"{pre}.cap.{h}.{f}"] = co["hindsight"][h]["cap"][f]
    for g in B0.GAMMAS:
        for rule in B0.RULES:
            x = co["gammas"][g]["rules"][rule]
            for b in B0.BUCKETS:
                out[f"{pre}.{g}.hit.{rule}.{b}"] = x["by_bucket"][b]
            for f in ("point", "lo", "hi"):
                out[f"{pre}.{g}.cap.{rule}.{f}"] = x["cap"][f]
                out[f"{pre}.{g}.diff_fifo.{rule}.{f}"] = x["diff_fifo"][f]
        for k, v in co["gammas"][g]["E_G_class"].items():
            out[f"{pre}.{g}.E_G.class.{k}"] = v
    return out


def gamma(s: int) -> dict:
    GC = _load("_cr_gc", "experiments/gamma-ceiling/run.py")
    r = GC.measure_seed(s, {CKPT: load_D(GC.B0, s)})
    out: dict = {}
    for g, x in r["ckpt"][CKPT].items():
        for f in ("point", "lo", "hi"):
            out[f"B.ckpt{CKPT}.{g}.cap.{f}"] = x["cap"][f]
            out[f"B.ckpt{CKPT}.{g}.diff_fifo.{f}"] = x["diff_fifo"][f]
    return out


def cf(s: int) -> dict:
    CF = _load("_cr_cf", "experiments/cf-ceiling/run.py")
    B0, LR = CF.B0, CF.LR
    p = CF_RAW / f"B-ckpt{CKPT}-seed{s}.delta.pt"
    led = {
        r["key"]: r
        for r in C.jload(C.ROOT / "runs" / "cf-ceiling" / "ledger.json")["rows"]
        if isinstance(r.get("key"), str)
    }
    want = led[f"B.ckpt{CKPT}.seed{s}.raw"]["value"]["sha256"]
    got = sha(p)
    if got != want:
        raise DidNotRun(f"delta.pt sha256 {got} != ledger {want}")
    saved = torch.load(p, weights_only=False)
    docs, *_ = LR.documents(s)
    docs = list(docs)
    W = B0.boot_weights(len(docs), s)
    fifo_pd, ff_pd = [], []
    for doc in docs:
        fifo_pd.append(B0.pairs_of(doc, CF.FIFOPolicy()))
        ff_pd.append(
            B0.pairs_of(doc, LR.FactFiller(doc, CF.random.Random(f"ff:{s}:{doc.doc_id}")))
        )
    fifo_all = [B0.rate(x) for x in fifo_pd]
    ff_all = [B0.rate(x) for x in ff_pd]
    out: dict = {}
    for g, gm in CF.GAMMAS.items():
        for arm, src in (("cf", "Delta"), ("cfshift", "shift")):
            T = {d: LR.discounted_returns(x, gm) for d, x in saved[src].items()}
            x = CF.score_arm(docs, T, W, fifo_all, ff_all)
            for f in ("point", "lo", "hi"):
                out[f"B.ckpt{CKPT}.{arm}.{g}.cap.{f}"] = x["cap"][f]
                out[f"B.ckpt{CKPT}.{arm}.{g}.diff_fifo.{f}"] = x["diff_fifo"][f]
    return out


def sample(path: str, key: str, idx: int) -> str:
    for r in C.jload(path)["rows"]:
        if r.get("key") == key:
            v = r["samples"][idx] if "samples" in r else C.dig(r.get("value"), str(idx))
            return C.fmt_exact(v)
    raise DidNotRun(f"key {key!r} not in {path}")


def labels(which: str) -> str:
    """Re-apply the run's own PREREG reading rule (``label`` + class function) to the
    committed ledger's per-seed ckpt3000 cap points and CI lowers. Re-derives the
    labels and class from the ledger's caps (the caps themselves are raw-checked by the
    ``cf`` / ``gamma`` subcommands); prints ``{"labels": {...}, "class": ...}``."""
    if which == "cf":
        mod, run, pre, cls_fn = (
            "experiments/cf-ceiling/run.py",
            "cf-ceiling",
            "cf.",
            "cf_class",
        )
    else:
        mod, run, pre, cls_fn = (
            "experiments/gamma-ceiling/run.py",
            "gamma-ceiling",
            "",
            "sweep_class",
        )
    M = _load(f"_cr_lab_{which}", mod)
    rows = {
        r["key"]: r
        for r in C.jload(C.ROOT / "runs" / run / "ledger.json")["rows"]
        if isinstance(r.get("key"), str)
    }
    lab = {}
    for g in M.GAMMAS:
        k = f"B.ckpt{CKPT}.{pre}{g}.cap"
        lab[g] = M.label(rows[f"{k}.point"]["samples"], rows[f"{k}.lo"]["samples"])
    return json.dumps({"labels": lab, "class": getattr(M, cls_fn)(lab)}, sort_keys=True)


def main(argv: list[str]) -> int:
    try:
        if argv and argv[0] == "sample":
            print(sample(argv[1], argv[2], int(argv[3])))
            return 0
        if argv and argv[0] == "labels":
            print(labels(argv[1]))
            return 0
        what, s = argv[0], int(argv[argv.index("--seed") + 1])
        out = {"b0": b0, "gamma": gamma, "cf": cf}[what](s)
    except DidNotRun as e:
        print(f"DID NOT RUN: {e}", file=sys.stderr)
        return 3
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
