"""Item 1e (post-fails-controls PREREG §7): group knockout against the sum of single
leave-one-out values, on a registered U subset.

Pre-registration: `experiments/post-fails-controls/PREREG.md`, committed alone at
9f22d87. `main(argv)` measures and writes `runs/post-fails-controls-1e/{manifest.json,
ledger.json}`; `finalize(argv)` renders `RESULTS.md` and `claims.json`.

Subset: U positions ``[0, 64)`` (ids ``[64, 128)``), B0-CF's batches 0 and 1 (32
documents each, so the donors drawn are B0-CF's). FIFO memory, full-memory steps ``t in
[16, 48)``, eval mode, ``no_grad``; the frozen arm-B ckpt-003000 substrate.

* Singles: `rsr.metrics.loo.loo_delta_loss(mode="resample", seed)` -- imported, the
  instrument B0-CF measured with (E-C1 checks it reproduces B0-CF's Delta).
* Groups per (document, t), resident slots only: G1 every pair of filler slots with
  identical text; G2 every (assert, its query) pair, both resident; G3 8 random pairs;
  G4 2 random sets of 4, then 1 of 8 (G3 then G4 from ``random.Random(f"grp:{seed}:
  {doc_id}:{t}")``). A group's knockout replaces every member with its own single
  donor (`knockout_kv`, ``mode="replace"``), so ``I = D_G - sum_i D_i`` carries no
  donor noise. A group with a member lacking a donor is skipped and counted.

Reading rule: descriptive (the interaction share with CIs); gates nothing.
Exit codes (§0): 0 ran, every control passed; 1 a defect after; 3 did not run or a
control failed.
"""

from __future__ import annotations

import importlib.util
import itertools
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

from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.metrics import loo  # noqa: E402
from rsr.model.tg.model import init_memory  # noqa: E402
from rsr.model.tg.policy_loop import write_at  # noqa: E402
from rsr.train.loop import encode  # noqa: E402

EXPERIMENT = "experiments/post-fails-controls/item_1e.py"
ITEM = "1e"
RUN_ID = "post-fails-controls-1e"
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
#: B0-CF's sentence loss (loo_delta_loss's own `sent_loss`), imported.
CFD = _load("_pfc_cfdelta", "experiments/cf-ceiling/cfdelta.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG §7, transcribed.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
M = 16
S = 48
U_POSITIONS = (0, 64)  # ids [64, 128): B0-CF's batches 0 and 1
BATCH = 32  # B0-CF's BATCH (experiments/cf-ceiling/run.py:86)
T_LO, T_HI = M, S
N_G3 = 8
G4_SIZES = (4, 4, 8)
TYPES = ("G1_dup_fillers", "G2_fact_pairs", "G3_random_pairs", "G4_random_sets")
STEPSETS = ("Q", "other")
N_BOOT = 2000

QUESTION = (
    "Does single leave-one-out mis-value slots whose value is shared: the interaction "
    "share of group knockout against the sum of singles (PREREG §7)?"
)
FALSIFIER = (
    "NONE: descriptive (PREREG §7): the interaction share with CIs; gates nothing."
)
EXPECTED = (
    "PREREG §10: interaction share small for G3/G4; G2 not distinguishable from G3."
)


# --------------------------------------------------------------------------- #
# groups (pure)
# --------------------------------------------------------------------------- #


def groups_for(doc, t: int, seed: int, m: int = M) -> list[tuple[str, tuple[int, ...]]]:
    """Every registered group at step ``t`` as ``(type, sorted ranks)``; rank ``j``
    holds sentence ``t - m + j`` (FIFO, slot 0 the oldest)."""
    sent = [t - m + j for j in range(m)]
    kinds = [doc.sentences[i].kind for i in sent]
    texts = [doc.sentences[i].text for i in sent]
    out: list[tuple[str, tuple[int, ...]]] = []
    for a, b in itertools.combinations(range(m), 2):
        if kinds[a] == "filler" and kinds[b] == "filler" and texts[a] == texts[b]:
            out.append(("G1_dup_fillers", (a, b)))
    rank = {i: j for j, i in enumerate(sent)}
    for a, q in doc.pairs:
        if a in rank and q in rank and q < t:
            out.append(("G2_fact_pairs", tuple(sorted((rank[a], rank[q])))))
    rng = random.Random(f"grp:{seed}:{doc.doc_id}:{t}")
    pairs = list(itertools.combinations(range(m), 2))
    for pr in rng.sample(pairs, N_G3):
        out.append(("G3_random_pairs", tuple(pr)))
    for k in G4_SIZES:
        out.append(("G4_random_sets", tuple(sorted(rng.sample(range(m), k)))))
    return out


def group_kv(kv: torch.Tensor, members: list[list[int]], reps: list[list[torch.Tensor]]):
    """Knock out, per row ``b``, every rank in ``members[b]`` (``[]`` leaves the row),
    replacing rank ``j`` with ``reps[b][idx]`` -- one `knockout_kv` per member index.
    -> (new kv, untouched-slot pass-through ok)."""
    B, Mm, D = kv.shape
    out = kv
    width = max((len(x) for x in members), default=0)
    for k in range(width):
        slot = torch.full((B,), -1, dtype=torch.long)
        rep = torch.zeros(B, D, dtype=kv.dtype)
        for b in range(B):
            if k < len(members[b]):
                slot[b] = members[b][k]
                rep[b] = reps[b][k]
        out = loo.knockout_kv(out, slot, mode="replace", replacement=rep)
    touched = torch.zeros(B, Mm, dtype=torch.bool)
    for b in range(B):
        for j in members[b]:
            touched[b, j] = True
    keep = ~touched
    ok = bool(torch.equal(out[keep], kv[keep]))
    return out, ok


# --------------------------------------------------------------------------- #
# one batch
# --------------------------------------------------------------------------- #


@torch.no_grad()
def measure_batch(model, chunk: list, vmap: dict, seed: int, blob: dict) -> dict:
    ids, mask = encode(chunk, vmap, max_tokens=LR.L, steps=S)
    B = len(chunk)
    t0 = time.perf_counter()
    sing = loo.loo_delta_loss(
        model, chunk, ids, mask, mode="resample", seed=seed, steps=range(T_LO, T_HI)
    )
    t_single = time.perf_counter() - t0
    # E-C1: singles == B0-CF's Delta on the resident cells (missing donors: B0-CF
    # imputed 0 and flagged them; here they are NaN -- the flags must agree)
    e1 = {"n_cells": 0, "n_equal": 0, "n_missing_agree": 0, "bad": []}
    for b, doc in enumerate(chunk):
        Dd = blob["Delta"][doc.doc_id]
        miss = blob["missing"][doc.doc_id]
        for t in range(T_LO, T_HI):
            for j in range(M):
                sidx = int(sing["slot_sentence"][b, t, j])
                if sidx != t - M + j:
                    e1["bad"].append(f"doc {doc.doc_id} t {t} rank {j}: holds {sidx}")
                    continue
                v = float(sing["delta"][b, t, j])
                e1["n_cells"] += 1
                if bool(miss[t, sidx]):
                    if v != v:
                        e1["n_missing_agree"] += 1
                    else:
                        e1["bad"].append(f"doc {doc.doc_id} t {t} i {sidx}: missing flag")
                elif v == float(Dd[t, sidx]):
                    e1["n_equal"] += 1
                else:
                    e1["bad"].append(
                        f"doc {doc.doc_id} t {t} i {sidx}: {v} != {Dd[t, sidx]}"
                    )
    e1["ok"] = not e1["bad"] and e1["n_cells"] == e1["n_equal"] + e1["n_missing_agree"]
    e1["bad"] = e1["bad"][:10]
    # the group pass: loo_delta_loss's own rollout, re-run to hold the memory
    cfg = model.cfg
    dt = model.embed.weight.dtype
    mem = init_memory(B, cfg, device=ids.device, dtype=dt)
    bos_ctx = torch.zeros(B, cfg.D, dtype=dt)
    bos_valid = torch.zeros(B, dtype=torch.bool)
    recs: list[dict] = []
    e2 = {"n": 0, "n_equal": 0, "passthrough_ok": True, "base_equal": True, "bad": []}
    n_fwd = 0
    t1 = time.perf_counter()
    for t in range(S):
        ids_t, mask_t = ids[:, t], mask[:, t]
        out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
        base = CFD._sent_loss(out.logits, ids_t, mask_t)
        if not torch.equal(
            base.nan_to_num(-1.0), sing["live_loss"][:, t].nan_to_num(-1.0)
        ):
            e2["base_equal"] = False
        if T_LO <= t < T_HI:
            drow = sing["donor_row"][:, t]  # [B, M]

            def reps_of(b, mem_kv, ranks, drow=drow):
                return [mem_kv[int(drow[b, j]), j] for j in ranks]

            # E-C2: a group of one through this path equals its single
            for j in range(M):
                mem_l = [[j] if int(drow[b, j]) >= 0 else [] for b in range(B)]
                reps = [reps_of(b, mem.kv, ml) for b, ml in enumerate(mem_l)]
                kv2, ok = group_kv(mem.kv, mem_l, reps)
                e2["passthrough_ok"] &= ok
                o = model(ids_t, mask_t, kv2, mem.valid, bos_ctx, bos_valid)
                n_fwd += 1
                d1 = CFD._sent_loss(o.logits, ids_t, mask_t) - base
                for b in range(B):
                    if mem_l[b]:
                        e2["n"] += 1
                        if float(d1[b]) == float(sing["delta"][b, t, j]):
                            e2["n_equal"] += 1
                        elif len(e2["bad"]) < 10:
                            e2["bad"].append([chunk[b].doc_id, t, j])
            # the registered groups
            per_row = [groups_for(doc, t, seed) for doc in chunk]
            usable: list[list[tuple[str, tuple[int, ...]]]] = []
            for b, gl in enumerate(per_row):
                ok_l = []
                for typ, mem_r in gl:
                    if all(int(drow[b, j]) >= 0 for j in mem_r):
                        ok_l.append((typ, mem_r))
                    else:
                        recs.append({"b": b, "t": t, "type": typ, "skipped": True})
                usable.append(ok_l)
            width = max(len(x) for x in usable)
            for g in range(width):
                mem_l = [
                    list(usable[b][g][1]) if g < len(usable[b]) else [] for b in range(B)
                ]
                reps = [reps_of(b, mem.kv, ml) for b, ml in enumerate(mem_l)]
                kv2, ok = group_kv(mem.kv, mem_l, reps)
                e2["passthrough_ok"] &= ok
                o = model(ids_t, mask_t, kv2, mem.valid, bos_ctx, bos_valid)
                n_fwd += 1
                dG = CFD._sent_loss(o.logits, ids_t, mask_t) - base
                for b in range(B):
                    if g >= len(usable[b]):
                        continue
                    typ, mem_r = usable[b][g]
                    singles = [float(sing["delta"][b, t, j]) for j in mem_r]
                    recs.append(
                        {
                            "b": b,
                            "t": t,
                            "type": typ,
                            "members": list(mem_r),
                            "dG": float(dG[b]),
                            "sum_single": sum(singles),
                            "singles": singles,
                            "skipped": False,
                        }
                    )
        write = out.has_eos
        victim = torch.zeros(B, dtype=torch.long)
        mem = write_at(mem, out.srep, write, victim, t)
        if cfg.bos_replacement_mode == "copy":
            bos_ctx = out.srep
            bos_valid = write & (t + 1 < S)
    e2["ok"] = e2["passthrough_ok"] and e2["base_equal"] and e2["n"] == e2["n_equal"] > 0
    return {
        "records": recs,
        "doc_ids": [d.doc_id for d in chunk],
        "E-C1": e1,
        "E-C2": e2,
        "n_group_forwards": n_fwd,
        "seconds_single": t_single,
        "seconds_groups": time.perf_counter() - t1,
    }


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

FIELDS = ("n", "dG", "sum_single", "I", "absI", "absdG")


def per_doc_sums(recs: list[dict], docs: list, typ: str, qsel: str) -> torch.Tensor:
    """``[n_docs, 6]`` per document: (groups, sum D_G, sum sum D_i, sum I, sum |I|,
    sum |D_G|) over the groups of ``typ`` at the step set ``qsel``."""
    out = torch.zeros(len(docs), len(FIELDS), dtype=torch.float64)
    for r in recs:
        if r["skipped"] or r["type"] != typ:
            continue
        doc = docs[r["doc_pos"]]
        isq = doc.sentences[r["t"]].kind == "query"
        if (qsel == "Q") != isq:
            continue
        I_ = r["dG"] - r["sum_single"]
        row = out[r["doc_pos"]]
        row += torch.tensor(
            [1.0, r["dG"], r["sum_single"], I_, abs(I_), abs(r["dG"])],
            dtype=torch.float64,
        )
    return out


def stats_of(X: torch.Tensor, W: torch.Tensor) -> dict:
    """Means (pooled ratio per replicate) of D_G, sum D_i, I; the interaction share
    ``sum |I| / sum |D_G|`` (ratio of pooled per-document sums); run 1's `_ci`."""
    R = W @ X  # [B, 6]
    P = X.sum(0)
    out = {"n_groups": int(P[0])}
    for name, num, den in (
        ("mean_dG", 1, 0),
        ("mean_sum_single", 2, 0),
        ("mean_I", 3, 0),
        ("interaction_share", 4, 5),
    ):
        point = float(P[num] / P[den]) if float(P[den]) != 0 else None
        ci = R1._ci(R[:, num] / R[:, den])
        out[name] = {
            "point": point,
            "lo": ci["lo"],
            "hi": ci["hi"],
            "sd": ci["sd"],
            "dropped": ci["dropped"],
        }
    return out


def summarize(recs: list[dict], docs: list, seed: int, n_boot: int = N_BOOT) -> dict:
    W = R1.boot_weights(len(docs), seed, n_boot)
    out: dict = {}
    for typ in TYPES:
        for qs in STEPSETS:
            X = per_doc_sums(recs, docs, typ, qs)
            out.setdefault(typ, {})[qs] = stats_of(X, W)
        out[typ]["skipped"] = sum(1 for r in recs if r["skipped"] and r["type"] == typ)
    return out


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #


def measure_seed(
    seed: int, *, n_batches: int | None = None, n_boot: int = N_BOOT
) -> dict:
    t0 = time.perf_counter()
    docs_all, vmap, V, closure = LR.documents(seed)
    docs = list(docs_all[U_POSITIONS[0] : U_POSITIONS[1]])
    blob, dsha = OP.load_delta(seed)
    sub = R1.SUBSTRATE[seed]
    model = LR.load_model(Path(sub["checkpoint"]), V)
    model.requires_grad_(False)
    model.eval()
    chunks = [docs[i : i + BATCH] for i in range(0, len(docs), BATCH)]
    if n_batches is not None:
        chunks = chunks[:n_batches]
    batches, recs = [], []
    for ci, ch in enumerate(chunks):
        mb = measure_batch(model, ch, vmap, seed, blob)
        for r in mb["records"]:
            r["doc_pos"] = ci * BATCH + r.pop("b")
        recs += mb["records"]
        batches.append({k: v for k, v in mb.items() if k != "records"})
        print(f"1e seed {seed} batch {ci} {time.perf_counter() - t0:.0f}s", flush=True)
    used = docs[: sum(len(c) for c in chunks)]
    fin = all(
        math.isfinite(r["dG"]) and all(math.isfinite(x) for x in r["singles"])
        for r in recs
        if not r["skipped"]
    )
    controls = {
        "E-C1": {
            "batches": [b["E-C1"] for b in batches],
            "delta_sha256": dsha,
            "delta_pin": OP.DELTA_SHA[seed],
            "ok": all(b["E-C1"]["ok"] for b in batches) and dsha == OP.DELTA_SHA[seed],
        },
        "E-C2": {
            "batches": [b["E-C2"] for b in batches],
            "ok": all(b["E-C2"]["ok"] for b in batches),
        },
        "E-C3": {"finite": fin, "ok": fin},
        "closure": closure,
        "subset_ids": [used[0].doc_id, used[-1].doc_id, len(used)],
    }
    return {
        "seed": seed,
        "n_documents": len(used),
        "stats": summarize(recs, used, seed, n_boot),
        "batches": batches,
        "n_records": len(recs),
        "controls": controls,
        "model_grads": IC.common().no_grads(model),
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# ledger, execute, main
# --------------------------------------------------------------------------- #


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def write_rows(led, per: dict[int, dict]) -> None:
    how = "item_1e.py::measure_seed -> stats_of (per document, 2000, run 1's _ci)"
    sd = [per[s] for s in SEEDS]

    def stat(key, vals, h=how):
        if all(_finite(v) for v in vals):
            led.stat(key, vals, how=h)
        else:
            led.note(key, vals, how=h + " (not finite on every seed)")

    for typ in TYPES:
        stat(f"{typ}.skipped", [x["stats"][typ]["skipped"] for x in sd])
        for qs in STEPSETS:
            st = [x["stats"][typ][qs] for x in sd]
            stat(f"{typ}.{qs}.n_groups", [y["n_groups"] for y in st])
            for f in ("mean_dG", "mean_sum_single", "mean_I", "interaction_share"):
                for g in ("point", "lo", "hi", "sd", "dropped"):
                    stat(f"{typ}.{qs}.{f}.{g}", [y[f][g] for y in st])
    stat("n_documents", [x["n_documents"] for x in sd])
    stat("seconds_per_seed", [x["seconds"] for x in sd], "wall clock")


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
        "subset": {"U_positions": list(U_POSITIONS), "batch": BATCH},
        "steps": [T_LO, T_HI],
        "groups": {
            "G1": "every pair of filler slots with identical text",
            "G2": "every (assert, its query) pair, both resident, q < t",
            "G3": f"{N_G3} pairs, rng.sample(all unordered pairs, {N_G3}) (distinct)",
            "G4": f"sizes {list(G4_SIZES)}, each rng.sample(range(16), k), after G3",
            "rng": 'random.Random(f"grp:{seed}:{doc_id}:{t}")',
        },
        "donors": "loo_delta_loss(mode='resample', seed) donor_row (_ROLE_E0D + rank)",
        "bootstrap": {"n": N_BOOT, "fn": "R1.boot_weights(64, seed, 2000)"},
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "args": args,
    }


def execute(led, a) -> Exit:
    g = IC.Globals(led, reads_substrate=True, args=a)
    failed = g.start()
    per: dict[int, dict] = {}
    if not failed or a.n_batches is not None:
        for s in a.seeds:
            g.pin_substrate(s)
            per[s] = measure_seed(s, n_batches=a.n_batches, n_boot=a.n_boot)
            print(f"1e seed {s} done {per[s]['seconds']:.0f}s", flush=True)
    g.end(sorted(per))
    for s, r in per.items():
        led.note(f"seed{s}.controls", r["controls"], how="item_1e.py::measure_seed")
        led.note(f"seed{s}.batches", r["batches"], how="item_1e.py::measure_batch")
        led.note(f"seed{s}.G-C4.no_grads", r["model_grads"], how="common.no_grads")
        for name in ("E-C1", "E-C2", "E-C3"):
            if not r["controls"][name]["ok"]:
                failed.append(f"{name} seed{s}")
        if not r["controls"]["closure"]["ok"]:
            failed.append(f"closure seed{s}")
        if not r["model_grads"]["ok"]:
            failed.append(f"G-C4 seed{s}")
    if a.n_batches is not None:
        failed.append(f"smoke: --n-batches {a.n_batches} (not the registered subset)")
    led.note("controls_failed", failed, how="item_1e.py::execute")
    if sorted(per) == list(SEEDS) and a.n_batches is None:
        write_rows(led, per)
    if failed:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail="descriptive; every control passed",
    )
    return Exit.OK


def _run(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--n-batches", type=int, default=None, help="smoke only")
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

        led.note("exception", traceback.format_exc(), how="item_1e.py::main")
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
        "# post-fails-controls 1e: group knockout against the sum of single LOO",
        "",
        f"Generated by `{EXPERIMENT}::finalize` from `{Path(out_root) / 'ledger.json'}`. "
        "No number below is typed. Descriptive; gates nothing (PREREG §7).",
        "",
        f"- PREREG: `{PREREG}` @ {PREREG_COMMIT} (§7)",
        f"- git sha: `{led['provenance'].get('git_sha')}` · seeds actually run: "
        f"{led['seeds_actually_run']} · status: {led['status']}",
        f"- command: `{cmd.get('argv')}` · exit code: {cmd.get('exit_code')}",
        f"- config hash: `{led['config_hash']}`",
        f"- controls failed: {failed if failed else 'none'}"
        + ("" if not failed else " (NOT READ)"),
        "- subset: U positions [0, 64) (ids [64, 128)), B0-CF's batches 0 and 1; FIFO; "
        "t in [16, 48)",
        "",
        "## Interaction share sum|I| / sum|D_G|, point [95% CI], per seed",
        "",
        "| group | steps | seed 0 | seed 1 | seed 2 | groups (s0/s1/s2) |",
        "|---|---|---|---|---|---|",
    ]
    for typ in TYPES:
        for qs in STEPSETS:
            base = f"{typ}.{qs}.interaction_share"
            p, lo, hi = (
                _row(rows, f"{base}.{f}") or [None] * 3 for f in ("point", "lo", "hi")
            )
            n = _row(rows, f"{typ}.{qs}.n_groups") or [None] * 3
            L.append(
                f"| {typ} | {qs} | "
                + " | ".join(
                    f"{_fmt(p[i])} [{_fmt(lo[i])}, {_fmt(hi[i])}]" for i in range(3)
                )
                + " | "
                + "/".join(str(int(x)) if _finite(x) else "n/a" for x in n)
                + " |"
            )
    L += [
        "",
        "## Means: D_G, sum of singles, I = D_G - sum D_i, point [95% CI]",
        "",
        "| group | steps | quantity | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|---|---|",
    ]
    for typ in TYPES:
        for qs in STEPSETS:
            for f in ("mean_dG", "mean_sum_single", "mean_I"):
                base = f"{typ}.{qs}.{f}"
                p, lo, hi = (
                    _row(rows, f"{base}.{g}") or [None] * 3 for g in ("point", "lo", "hi")
                )
                L.append(
                    f"| {typ} | {qs} | {f} | "
                    + " | ".join(
                        f"{_fmt(p[i])} [{_fmt(lo[i])}, {_fmt(hi[i])}]" for i in range(3)
                    )
                    + " |"
                )
    L += [
        "",
        "Skipped groups (a member had no donor): "
        + ", ".join(f"{typ} {_row(rows, f'{typ}.skipped')}" for typ in TYPES),
        "",
    ]
    L += ["## Controls", ""]
    for s in SEEDS:
        c = (rows.get(f"seed{s}.controls") or {}).get("value") or {}
        L.append(
            f"- seed {s}: "
            + ", ".join(
                f"{k} {'pass' if (c.get(k) or {}).get('ok') else 'FAIL'}"
                for k in ("E-C1", "E-C2", "E-C3")
            )
        )
    L += [
        "",
        "## What 1e does NOT show (PREREG §7)",
        "",
        "Delta is single-step: the effect on sentence t only, not on the trajectory. G2 "
        "pairs carry no future demand. Not a claim about natural text.",
        "",
    ]
    pth = Path(out_root) / "RESULTS.md"
    pth.write_text("\n".join(L) + "\n")
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        f"--slots 1 --wait 3600 -- .venv/bin/python {RUN_PY} {ITEM} recompute"
    )
    claims = []
    for typ in TYPES:
        for qs in STEPSETS:
            key = f"{typ}.{qs}.interaction_share.point"
            v = _row(rows, key)
            if v is None:
                continue
            for i, s in enumerate(SEEDS):
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
    r = measure_seed(seed)
    typ, qs, f, g = what.split(".")
    return repr(r["stats"][typ][qs][f][g])


def finalize(argv: list[str] | None = None) -> int:
    """``run.py finalize 1e [--out DIR]``: RESULTS.md and claims.json from the ledger."""
    ap = ArgumentParser(description="finalize 1e")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(list(argv or []))
    print(render(a.out))
    return int(Exit.OK)


def main(argv: list[str] | None = None) -> int:
    """``run.py 1e [measure] [options]`` | ``finalize`` | ``recompute --what K --seed
    S``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        return finalize(argv[1:])
    if argv[:1] == ["recompute"]:
        ap = ArgumentParser(description="recompute one 1e value")
        ap.add_argument("--what", required=True)
        ap.add_argument("--seed", type=int, required=True)
        a = ap.parse_args(argv[1:])
        torch.set_num_threads(1)
        IC.common().install_seal()
        print(recompute(a.what, a.seed))
        return 0
    if argv[:1] == ["measure"]:
        argv = argv[1:]
    return int(_run(argv))


if __name__ == "__main__":
    run_main(main)
