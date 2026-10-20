"""Item 1a: FactFiller and the kind-oracle through the frozen reader on EVAL.

PREREG ``experiments/post-fails-controls/PREREG.md`` §3 (committed alone at 9f22d87),
with §0 (conventions) and §2 (global controls G-C1 to G-C5).

**Run 2's harness is reused by import, never copied.** Run 1's ``run.py`` is loaded
as a private module instance ``R1`` and exactly two of its names are rebound:

* ``R1.reference_policy`` -- the same function, plus two names it did not know:
  ``factfiller`` (lookahead-room's ``FactFiller``, ``random.Random(f"ff:{seed}:{id}")``,
  one per document, a ``PerRow`` per block exactly as the oracle's) and
  ``kindoracle`` (B0's rule (i) ``ko`` verbatim: B0's ``ScorePolicy`` over B0's
  ``scorers(...)["ko"]``, ``tie="random"``). Every other name is delegated unchanged,
  so the ``oracle`` reproduction pass runs run 2's own code path.
* ``R1.model_from_payload`` -- the same function, wrapped only to keep a reference to
  each model it builds, for G-C4's gradient check.

``R1.run_heldout`` (blocks of 16, ``run_batch``, ``eval_forward``, residency, the
records) runs the four passes. Run 2's ``run.py`` supplies ``read_delta`` (Cdelta),
``seed_label`` / ``label_contrast``, ``factfiller_hits``, ``hit_counts``,
``age1_share`` and ``_tg``. B0's ``run.py`` supplies ``ScorePolicy`` and ``scorers``.

Order, per seed (A-C2: no factfiller or kindoracle statistic before A-C2 passes):
G-C2 start -> pins -> A-C1, A-C7 -> reader frozen -> passes ``fifo``, ``oracle`` ->
A-C2, A-C3 -> passes ``factfiller``, ``kindoracle`` -> A-C4, A-C5, A-C6 -> statistics
(only if A-C2 passed) -> G-C4 -> A-C8 -> G-C2 end -> G-C1 census.

Subcommands (via ``run.py 1a ...``): ``launch-checks`` (G-C1(e), G-C3, the frozen
manifest), ``measure --seed s``, ``recompute`` (claims.json); ``run.py finalize 1a``.
Exit codes (§0): 0 ran, every control passed; 1 a defect after the controls passed;
3 did not run, or a control failed (numbers written, labels NOT READ).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _common():
    """``common.py`` under its one shared module name, ``pfc_common`` (the seal is
    process-wide state; two instances would split it)."""
    name = "pfc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


C = _common()

ITEM = "1a"
RUN_ID = f"{C.RUN_ID_PREFIX}{ITEM}"
EXPERIMENT = "experiments/post-fails-controls/run.py"
NEW_ARMS = ("factfiller", "kindoracle")
REPRO = ("fifo", "oracle")
PASS_ORDER = (*REPRO, *NEW_ARMS)
KINDS = ("assert", "query", "filler")
BUCKETS = ("all", "gap_2_to_M", "gap_gt_M")
#: A-C2 / A-C3: the registered record fields (PREREG §3).
REG_FIELDS = ("doc", "answers", "hits", "victims", "evictions")
QUESTION = (
    "PREREG post-fails-controls §3 (1a): can a legal content-only fixed rule "
    "(FactFiller; the kind-oracle) beat FIFO through this frozen reader on run 2's "
    "EVAL (already used)?"
)
FALSIFIER = (
    "PLAN §2 / Gate P2 input: FactFiller LOSS on any seed would place the failure "
    "reader-side (D2); WIN on all seeds learning-side. Gate P2 is applied by the PM."
)
READING = {
    "WIN_ALL_SEEDS": "Win on all seeds: the failure is in learning.",
    "LOSS_ANY_SEED": "Any LOSS: reader-side (D2), an owner item.",
    "INDETERMINATE": "INDETERMINATE: neither learning-side nor reader-side is supported.",
}


@dataclasses.dataclass(frozen=True)
class Refs:
    """Where 1a reads what it did not produce (PREREG §3 Inputs). Tests repoint them."""

    run2_out: str = (
        "/Users/keanooo7/retrieval-successor-retention/.worktrees/"
        "rsr-run2-run/runs/rsr-run2"
    )
    run2_ledger: str = str(ROOT / "runs" / "rsr-run2" / "ledger.json")
    b0_ledger: str = str(ROOT / "runs" / "b0-ceilings" / "ledger.json")
    b0_key: str = "B.ckpt3000.g09.E_G.kind.{k}"
    acc_key: str = "seed{s}.heldout.accuracy.all.{arm}"
    hit_key: str = "seed{s}.run2.heldout.hit_rate.all.hit"
    ckpt_step: int = 3000


REFS = Refs()

# --------------------------------------------------------------------------- #
# The harness modules, imported (never copied)
# --------------------------------------------------------------------------- #

_H: dict[str, Any] = {}
_MODELS: list = []
_KIND_MEANS: dict[int, dict[str, float]] = {}


def harness() -> dict[str, Any]:
    """``{"R1", "R2", "B0", "LR"}``, loaded once, after the seal is installed (so
    every ``from rsr.data.synthetic import ...`` they do binds the guard)."""
    if _H:
        return _H
    C.install_seal()
    R1 = C.load_module("_pfc1a_r1", "experiments/rsr-run1/run.py")
    # Run 2's module registers its private run-1 instance as ``_rsr_run2_r1`` in
    # sys.modules and wraps that instance's ``cli_argv`` at import. A second run-2
    # instance would reuse a first one's R1 and wrap ``cli_argv`` twice (launch
    # attempt 1: tests/test_rsr_run2_harness.py red). So: load ours against a fresh R1
    # and restore whatever sys.modules held before.
    key = "_rsr_run2_r1"
    saved = sys.modules.pop(key, None)
    try:
        R2 = C.load_module("_pfc1a_r2", "experiments/rsr-run2/run.py")
    finally:
        if saved is not None:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)
    B0 = C.load_module("_pfc1a_b0", "experiments/b0-ceilings/run.py")
    LR = R2._lr()  # run 2's own lookahead-room instance: its FactFiller
    orig_ref, orig_model = R1.reference_policy, R1.model_from_payload

    def reference_policy(name: str, docs: list, seed: int):
        if name == "factfiller":
            pols = [
                LR.FactFiller(d, random.Random(f"ff:{seed}:{d.doc_id}")) for d in docs
            ]
            return R1.PerRow(pols, "factfiller")
        if name == "kindoracle":
            if seed not in _KIND_MEANS:
                raise C.ControlFailure(f"kindoracle: no m_kind read for seed {seed}")
            mu = {"kind": dict(_KIND_MEANS[seed])}
            return R1.PerRow([kind_oracle(B0, d, mu, seed) for d in docs], "kindoracle")
        return orig_ref(name, docs, seed)

    def model_from_payload(payload: dict):
        m = orig_model(payload)
        _MODELS.append(m)
        return m

    R1.reference_policy = reference_policy
    R1.model_from_payload = model_from_payload
    C.install_seal()  # rescan: nothing loaded above may hold an unguarded generator
    _H.update(R1=R1, R2=R2, B0=B0, LR=LR)
    return _H


def kind_oracle(B0, doc, mu: dict, seed: int):
    """B0's rule (i) ``ko``, verbatim: score ``m_kind(kind(i))``, argmin, ties
    ``random.Random(f"ko:{seed}:{doc_id}:{t}").choice(sorted(tied))``. No age."""
    return B0.ScorePolicy(
        B0.scorers(doc, mu)["ko"], seed=seed, doc_id=doc.doc_id, tie="random"
    )


def set_kind_means(seed: int, m_kind: dict[str, float]) -> None:
    _KIND_MEANS[int(seed)] = {k: float(m_kind[k]) for k in KINDS}


# --------------------------------------------------------------------------- #
# Controls (pure functions on records)
# --------------------------------------------------------------------------- #


def records_equal(new: list[dict], ref: list[dict], *, prefix: bool = False) -> dict:
    """A-C2 / A-C3: record for record on the registered fields; the length is asserted
    (``prefix=True`` only in a smoke run, and says so). Full-record equality (every
    field, incl. ``real_sum``, ``n_live_before``) is reported beside it."""
    ref_c = ref[: len(new)] if prefix else ref
    out: dict[str, Any] = {
        "n_new": len(new),
        "n_ref": len(ref),
        "prefix_only": prefix,
        "length_equal": len(new) == len(ref_c),
        "fields": list(REG_FIELDS),
    }
    bad = []
    full_bad = 0
    for i, (a, b) in enumerate(zip(new, ref_c, strict=False)):
        diff = [f for f in REG_FIELDS if a.get(f) != b.get(f)]
        if diff:
            bad.append({"index": i, "doc": a.get("doc"), "fields": diff})
        if a != b:
            full_bad += 1
    out.update(
        n_mismatched=len(bad),
        first_mismatches=bad[:10],
        n_full_record_mismatched=full_bad,
        ok=out["length_equal"] and not bad and len(new) > 0,
    )
    return out


def pairing(R2, new: list[dict], fifo_ref: list[dict], *, prefix: bool = False) -> dict:
    """A-C4: ``(doc, [(t, gap)])`` equals run 2's fifo pass; length asserted."""
    ref = fifo_ref[: len(new)] if prefix else fifo_ref
    a, b = R2._tg(new), R2._tg(ref)
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    return {
        "n_new": len(a),
        "n_ref": len(b),
        "prefix_only": prefix,
        "first_mismatch_index": first,
        "ok": len(a) == len(b) and a == b and len(a) > 0,
    }


def residency(ho: dict, name: str, n: int) -> dict:
    """A-C5 from ``R1.run_heldout``'s own check (``R1.residency_ok``: in-loop residency
    equals the ``headroom.simulate`` replay of the pass's own victims)."""
    r = (ho.get("residency") or {}).get(name)
    if r is None:
        return {"ok": False, "error": f"no residency record for {name}"}
    return {**r, "ok": r["n_documents"] == n and r["n_mismatched"] == 0}


def point_equal(value, ledger_value) -> dict:
    return {"value": value, "ledger": ledger_value, "ok": value == ledger_value}


def acc_point(R1, recs: list[dict], m: int, bucket: str = "all"):
    return R1._point(R1.answer_counts(recs, R1.buckets_for(m)[bucket]))


def hit_point(R1, R2, hits: list[list], m: int, bucket: str = "all"):
    return R1._point(R2.hit_counts(hits, R1.buckets_for(m)[bucket]))


def _norm_hits(hits) -> list[list]:
    return [[int(g), bool(h)] for g, h in hits]


def factfiller_check(R1, R2, recs: list[dict], docs: list, seed: int, m: int) -> dict:
    """A-C6 (first half): in-loop factfiller hits equal run 2's model-free
    ``factfiller_hits`` per document."""
    mf = R2.factfiller_hits(docs, seed, m)
    loop_h = [_norm_hits(r["hits"]) for r in recs]
    mf_h = [_norm_hits(h) for h in mf]
    bad = [
        recs[i]["doc"]
        for i, (a, b) in enumerate(zip(loop_h, mf_h, strict=False))
        if a != b
    ]
    return {
        "n_docs": len(loop_h),
        "n_model_free": len(mf_h),
        "mismatched_docs": bad[:20],
        "n_mismatched": len(bad),
        "pooled_hit_in_loop": hit_point(R1, R2, loop_h, m),
        "pooled_hit_model_free": hit_point(R1, R2, mf_h, m),
        "ok": len(loop_h) == len(mf_h) and not bad and len(loop_h) > 0,
    }


# --------------------------------------------------------------------------- #
# Statistics (A-C2 gates them)
# --------------------------------------------------------------------------- #


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def seed_stats(
    H: dict,
    recs: dict[str, list[dict]],
    seed: int,
    delta,
    *,
    a_c2: dict,
    n_boot: int,
    m: int,
) -> dict:
    """PREREG §3 Metric, Headline and Secondary for one seed. ``recs["fifo"]`` is run
    2's FIFO records. Refuses to compute anything unless A-C2 passed for this seed."""
    if not (a_c2 or {}).get("ok"):
        raise C.ControlFailure(
            "A-C2 has not passed for this seed: no statistic is computed"
        )
    R1, R2 = H["R1"], H["R2"]
    order = [r["doc"] for r in recs["fifo"]]
    for k, rs in recs.items():
        if [r["doc"] for r in rs] != order:
            raise C.ControlFailure(f"{k} is not in the fifo pass's document order")
    W = R1.boot_weights(len(order), seed, n_boot)
    bk = R1.buckets_for(m)
    out: dict[str, Any] = {
        "n_documents": len(order),
        "n_boot": n_boot,
        "boot_seed": R1.BOOT_SEED_BASE + seed,
        "delta": delta,
        "documents": "EVAL, already used",
        "accuracy": {},
        "contrast": {},
        "hit_rate": {},
        "hit_contrast": {},
    }
    for b in BUCKETS:
        per = {k: R1.answer_counts(v, bk[b]) for k, v in recs.items()}
        out["accuracy"][b] = {k: R1.level(p, W) for k, p in per.items()}
        out["contrast"][b] = {a: R1.contrast(per, a, "fifo", W) for a in NEW_ARMS}
        hp = {k: R2.hit_counts([r["hits"] for r in v], bk[b]) for k, v in recs.items()}
        out["hit_rate"][b] = {k: R1.level(p, W) for k, p in hp.items()}
        out["hit_contrast"][b] = {a: R1.contrast(hp, a, "fifo", W) for a in NEW_ARMS}
    out["headline"] = {
        a: R2.label_contrast(out["contrast"]["all"][a], delta) for a in NEW_ARMS
    }
    out["age1_share"] = {k: R2.age1_share(v) for k, v in recs.items()}
    return out


def stats_defects(st: dict | None) -> list[str]:
    """Exit 1: a NaN (or a missing number) in a reported statistic."""
    if not st:
        return ["no statistics"]
    bad = []
    for a in NEW_ARMS:
        h = st["headline"].get(a) or {}
        for k in ("point", "lo", "hi"):
            if not _finite(h.get(k)):
                bad.append(f"headline.{a}.{k} = {h.get(k)}")
    for b in BUCKETS:
        for k, v in st["accuracy"][b].items():
            if not _finite(v.get("point")):
                bad.append(f"accuracy.{b}.{k}.point = {v.get('point')}")
    return bad


def reading(labels: dict[int, str | None], seeds) -> dict:
    """PREREG §3's registered reading rule, tightened: WIN on all seeds; any LOSS;
    otherwise INDETERMINATE."""
    got = [labels.get(s) for s in seeds]
    if any(x == "LOSS" for x in got):
        key = "LOSS_ANY_SEED"
    elif got and all(x == "WIN" for x in got):
        key = "WIN_ALL_SEEDS"
    else:
        key = "INDETERMINATE"
    return {
        "pattern": key,
        "text": READING[key],
        "labels": {str(s): labels.get(s) for s in seeds},
    }


# --------------------------------------------------------------------------- #
# measure: one seed
# --------------------------------------------------------------------------- #


def read_kind_means(seed: int, refs: Refs = REFS) -> dict:
    """A-C7: ``m_kind`` from B0's ledger (sample ``seed``) and the values the policy
    holds compared ``==``; the order per seed is ledgered."""
    rows = C.ledger_rows(refs.b0_ledger)
    vals = {k: rows[refs.b0_key.format(k=k)]["samples"][seed] for k in KINDS}
    set_kind_means(seed, vals)
    held = _KIND_MEANS[seed]
    return {
        "ledger": refs.b0_ledger,
        "keys": {k: refs.b0_key.format(k=k) for k in KINDS},
        "m_kind": vals,
        "policy_m_kind": dict(held),
        "order_ascending": sorted(KINDS, key=lambda k: vals[k]),
        "ok": all(held[k] == vals[k] for k in KINDS)
        and all(_finite(v) for v in vals.values()),
    }


def measure_seed(
    seed: int,
    out_root: Path,
    *,
    limit: int | None = None,
    refs: Refs = REFS,
    threads: int = 1,
) -> dict:
    import torch

    torch.set_num_threads(threads)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    smoke = limit is not None
    doc: dict[str, Any] = {
        "run_id": RUN_ID,
        "item": ITEM,
        "seed": seed,
        "smoke": smoke,
        "limit": limit,
        "refs": dataclasses.asdict(refs),
        "git_head": C.git_head(),
        "git_dirty": C.git_dirty_paths(),
        "argv": sys.argv,
        "torch_threads": torch.get_num_threads(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "controls": {},
        "passes": {},
        "stats": None,
        "notes": [],
        "phase": "setup",
    }
    ctl = doc["controls"]
    pins = C.Pins()

    def finish(code: int, status: str) -> dict:
        doc.update(exit=int(code), status=status)
        doc["controls_failed"] = sorted(
            k for k, v in ctl.items() if v.get("ok") is not True
        )
        doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f = out_root / f"measure-seed{seed}.json"
        f.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        return doc

    try:
        ctl["G-C1.seal"] = C.install_seal()
        ctl["G-C1.seal"]["ok"] = True
        ctl["G-C4.record_refused"] = C.forbid_constants_record()
        H = harness()
        R1, R2 = H["R1"], H["R2"]
        cfg = R1.prereg_config(seed, refs.ckpt_step)
        m = cfg.memory_slots
        sub = cfg.substrate
        # G-C2, start
        ctl["G-C2.manifest_start"] = C.manifest_check(
            sub.manifest, R1.MANIFEST_SHA256, sub.root
        )
        ctl["G-C2.substrate_pin_start"] = C.checkpoint_pin(sub.checkpoint, sub.sha256)
        # A-C8 pins: every run-2 file read, and every other input
        r2 = Path(refs.run2_out)
        hd2 = r2 / "heldout" / f"seed{seed}"
        ckpt = R1.checkpoint_path(R1.arm_dir(r2, "fifo", seed), refs.ckpt_step)
        for p in (
            hd2 / "fifo.jsonl.gz",
            hd2 / "oracle.jsonl.gz",
            ckpt,
            refs.run2_ledger,
            R2.REFS.b2_ledger,
            R2.REFS.prereg,
            refs.b0_ledger,
        ):
            pins.pin(p)
        doc["pins_first"] = dict(pins.first)
        # A-C1: delta (Cdelta, run 2's read_delta)
        dl = R2.read_delta(seed)
        ctl["A-C1"] = dl
        delta = dl.get("delta")
        # A-C7: m_kind
        ctl["A-C7"] = read_kind_means(seed, refs)
        # the reader: run 2's fifo arm checkpoint, frozen = the substrate's tensors
        payload = R1.read_checkpoint(ckpt)
        sub_model, sub_sha = R1.load_substrate_model(cfg)
        ctl["G-C2.reader_equals_substrate"] = {
            **R1.c3_frozen({"fifo": payload}, sub_model),
            "substrate_sha256": sub_sha,
            "reader": str(ckpt),
        }
        payloads = {"fifo": payload}
        r2rows = C.ledger_rows(refs.run2_ledger)
        run2 = {a: R1.read_pass(hd2, a) for a in REPRO}
        # passes 1-2: the reproductions
        doc["phase"] = "passes"
        hd = out_root / "heldout" / f"seed{seed}"
        ho1 = R1.run_heldout(
            cfg, payloads, hd, passes=[(a, "fifo", False) for a in REPRO], limit=limit
        )
        new = {a: R1.read_pass(hd, a) for a in REPRO}
        n = len(new["fifo"])
        for a, cid in (("fifo", "A-C2"), ("oracle", "A-C3")):
            eq = records_equal(new[a], run2[a], prefix=smoke)
            mine = acc_point(R1, new[a], m)
            if smoke:
                ref_pt = acc_point(R1, run2[a][:n], m)
                acc = {**point_equal(mine, ref_pt), "against": "run 2's records, prefix"}
            else:
                lp = r2rows[refs.acc_key.format(s=seed, arm=a)]["value"]["point"]
                acc = {
                    **point_equal(mine, lp),
                    "against": refs.acc_key.format(s=seed, arm=a),
                }
            ctl[cid] = {"records": eq, "accuracy_all": acc, "ok": eq["ok"] and acc["ok"]}
        # passes 3-4: the new arms
        ho2 = R1.run_heldout(
            cfg, payloads, hd, passes=[(a, "fifo", False) for a in NEW_ARMS], limit=limit
        )
        for a in NEW_ARMS:
            new[a] = R1.read_pass(hd, a)
        doc["passes"] = {
            "timing": {**ho1["timing"], **ho2["timing"]},
            "n_documents": {a: len(new[a]) for a in PASS_ORDER},
            "doc_ids": {
                "first": ho1["doc_ids"][0],
                "last": ho1["doc_ids"][-1],
                "n": len(ho1["doc_ids"]),
            },
            "records": {a: str(hd / f"{a}.jsonl.gz") for a in PASS_ORDER},
            "sha256": {a: C.sha256_file(hd / f"{a}.jsonl.gz") for a in PASS_ORDER},
        }
        ctl["A-C4"] = {
            a: pairing(R2, new[a], run2["fifo"], prefix=smoke) for a in PASS_ORDER
        }
        ctl["A-C4"]["ok"] = all(v["ok"] for v in ctl["A-C4"].values())
        res = {**ho1, "residency": {**ho1["residency"], **ho2["residency"]}}
        ctl["A-C5"] = {a: residency(res, a, n) for a in PASS_ORDER}
        ctl["A-C5"]["ok"] = all(v["ok"] for v in ctl["A-C5"].values())
        docs = R1.docs_by_id(cfg, [r["doc"] for r in new["factfiller"]])
        ffc = factfiller_check(R1, R2, new["factfiller"], docs, seed, m)
        if smoke:
            ff_pool = {
                "value": ffc["pooled_hit_model_free"],
                "ledger": None,
                "ok": True,
                "note": "smoke: the ledger point is over 2048 documents",
            }
        else:
            lv = r2rows[refs.hit_key.format(s=seed)]["value"]["factfiller"]["point"]
            ff_pool = point_equal(ffc["pooled_hit_in_loop"], lv)
            ff_pool["against"] = f"{refs.hit_key.format(s=seed)} (factfiller point)"
        ctl["A-C6"] = {
            "per_document": ffc,
            "pooled": ff_pool,
            "ok": ffc["ok"] and ff_pool["ok"],
        }
        # statistics: only once A-C2 has passed for this seed
        doc["phase"] = "statistics"
        if ctl["A-C2"]["ok"]:
            recs = {"fifo": run2["fifo"][:n] if smoke else run2["fifo"]}
            recs.update({a: new[a] for a in NEW_ARMS})
            n_boot = R1.N_BOOT
            if not smoke and n != R1.EVAL[1] - R1.EVAL[0]:
                raise C.ControlFailure(f"{n} documents, not EVAL's")
            doc["stats"] = seed_stats(
                H, recs, seed, delta, a_c2=ctl["A-C2"], n_boot=n_boot, m=m
            )
        else:
            doc["notes"].append(
                "A-C2 failed: no factfiller or kindoracle statistic computed"
            )
        doc["phase"] = "end-controls"
        # G-C4
        g = [C.no_grads(x) for x in _MODELS]
        frozen_after = R1.c3_frozen(
            {f"model{i}": {"model": x.state_dict()} for i, x in enumerate(_MODELS)},
            sub_model,
        )
        ctl["G-C4.frozen"] = {
            "n_models": len(_MODELS),
            "no_grads": g,
            "state_equals_substrate_after": frozen_after["ok"],
            "optimizers": 0,
            "ok": bool(_MODELS) and all(x["ok"] for x in g) and frozen_after["ok"],
        }
        # A-C8, G-C2 end, G-C1 census
        ctl["A-C8"] = pins.recheck()
        ctl["G-C2.substrate_pin_end"] = C.checkpoint_pin(sub.checkpoint, sub.sha256)
        ctl["G-C2.manifest_end"] = C.manifest_check(
            sub.manifest, R1.MANIFEST_SHA256, sub.root
        )
        cen = C.seal_census({"EVAL": tuple(R1.EVAL), "vocab": (0, cfg.vocab_documents)})
        ctl["G-C1.census"] = cen
        doc["seeds_actually_run"] = [seed] if n > 0 else []
        doc["documents_done"] = n
    except Exception as e:
        H_ = _H
        ctl_types = (C.ControlFailure, C.SealedRange)
        if H_:
            ctl_types = (*ctl_types, H_["R1"].ControlFailure)
        doc["error"] = f"{type(e).__name__}: {e}"
        if doc["phase"] == "statistics" and not isinstance(e, ctl_types):
            doc["notes"].append(f"DEFECT in the computation: {doc['error']}")
            return finish(1, "failed")
        doc["notes"].append(f"DID NOT RUN / CONTROL FAILED: {doc['error']}")
        return finish(3, "did_not_run")
    failed = sorted(k for k, v in ctl.items() if v.get("ok") is not True)
    if failed:
        doc["notes"].append(f"CONTROL_FAILED: {failed}")
        return finish(3, "did_not_run")
    bad = stats_defects(doc["stats"])
    if bad:
        doc["notes"].append(f"DEFECT (exit 1): {bad}")
        return finish(1, "failed")
    return finish(0, "ok")


# --------------------------------------------------------------------------- #
# launch-checks: G-C1(e), G-C3, the frozen manifest
# --------------------------------------------------------------------------- #


def prereg_passage(start: str, stop: str) -> str:
    """A PREREG passage, read from the file (quoted, never retyped)."""
    text = C.PREREG_PATH.read_text()
    i = text.index(start)
    j = text.index(stop, i + len(start))
    return text[i:j].strip()


def expectation_text() -> str:
    return prereg_passage("- 1a:", "- 1b:")


def manifest_config(refs: Refs, seeds) -> dict:
    H = harness()
    R1 = H["R1"]
    return {
        "run_id": RUN_ID,
        "item": ITEM,
        "prereg": C.PREREG_REL,
        "prereg_commit": C.PREREG_COMMIT,
        "seeds": list(seeds),
        "passes": list(PASS_ORDER),
        "eval_range": list(R1.EVAL),
        "eval_batch": R1.EVAL_BATCH,
        "memory_slots": R1.M_SLOTS,
        "steps_per_stream": R1.S_STEPS,
        "device": R1.DEVICE,
        "threads_per_process": 1,
        "n_boot": R1.N_BOOT,
        "boot_seed_base": R1.BOOT_SEED_BASE,
        "decision_context": "eval_forward",
        "reader": f"run 2 fifo arm ckpt-{refs.ckpt_step:06d}.pt, payload key fifo",
        "refs": dataclasses.asdict(refs),
        "substrate": {str(s): v for s, v in R1.SUBSTRATE.items()},
        "manifest_sha256": R1.MANIFEST_SHA256,
        "sealed": {k: list(v) for k, v in C.sealed_ranges().items()},
        "expected": expectation_text(),
        "falsifier": FALSIFIER,
    }


def launch_checks(
    out_root: Path,
    *,
    suite_log: Path,
    suite_rc: Path,
    suite_head: Path,
    refs: Refs = REFS,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    C.install_seal()
    head = C.git_head()
    rep: dict[str, Any] = {
        "run_id": RUN_ID,
        "head": head,
        "dirty": C.git_dirty_paths(),
        "launched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # G-C3: the suite, run by the launcher at this head (its rc read directly there)
    try:
        rc_txt = Path(suite_rc).read_text().strip()
        rc = int(rc_txt) if rc_txt.lstrip("-").isdigit() else None
        cen = C.suite_census(Path(suite_log).read_text(), rc)
        sh = Path(suite_head).read_text().strip()
        cen.update(head=sh, head_is_launch_head=sh == head, log=str(suite_log))
        cen["ok"] = cen["ok"] and cen["head_is_launch_head"]
    except OSError as e:
        cen = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    rep["G-C3"] = cen
    # G-C1(e): the §1.1 search, every local branch
    rep["G-C1.search"] = C.branch_search()
    seeds = C.registered_seeds()
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    if led.path.parent != out_root:
        raise C.ControlFailure(f"out_root {out_root} is not runs/{RUN_ID}")
    led.manifest(manifest_config(refs, seeds))
    rep["manifest"] = {
        "path": str(out_root / "manifest.json"),
        "config_hash": led.doc["config_hash"],
        "manifest_written_utc": led.doc["manifest_written_utc"],
    }
    rep["clean_tree"] = {"dirty": rep["dirty"], "ok": rep["dirty"] == []}
    rep["ok"] = bool(
        cen.get("ok") and rep["G-C1.search"]["ok"] and rep["clean_tree"]["ok"]
    )
    rep["exit"] = 0 if rep["ok"] else 3
    (out_root / "launch.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    return rep


# --------------------------------------------------------------------------- #
# recompute (claims.json): from the records on disk, not from the ledger
# --------------------------------------------------------------------------- #


def recompute(
    what: str, out_root: Path, seed: int, arm: str | None = None, refs: Refs = REFS
) -> str:
    H = harness()
    R1, R2 = H["R1"], H["R2"]
    m = R1.M_SLOTS
    hd = Path(out_root) / "heldout" / f"seed{seed}"
    hd2 = Path(refs.run2_out) / "heldout" / f"seed{seed}"
    if what == "headline":
        recs = {"fifo": R1.read_pass(hd2, "fifo"), arm: R1.read_pass(hd, arm)}
        W = R1.boot_weights(len(recs["fifo"]), seed, R1.N_BOOT)
        per = {k: R1.answer_counts(v, R1.buckets_for(m)["all"]) for k, v in recs.items()}
        c = R2.label_contrast(
            R1.contrast(per, arm, "fifo", W), R2.read_delta(seed)["delta"]
        )
        return f"{c['point']:.6f} {c['lo']:.6f} {c['hi']:.6f} {c['label']}"
    if what == "accuracy":
        return f"{acc_point(R1, R1.read_pass(hd, arm), m):.6f}"
    if what == "ac2":
        eq = records_equal(R1.read_pass(hd, "fifo"), R1.read_pass(hd2, "fifo"))
        return f"{eq['ok']} {eq['n_new']}"
    if what == "ac3":
        eq = records_equal(R1.read_pass(hd, "oracle"), R1.read_pass(hd2, "oracle"))
        return f"{eq['ok']} {eq['n_new']}"
    if what == "hit":
        hits = [r["hits"] for r in R1.read_pass(hd, arm)]
        return f"{hit_point(R1, R2, hits, m):.6f}"
    if what == "age1":
        a = R2.age1_share(R1.read_pass(hd, arm))
        return f"{a['n_age1']} {a['n_evictions']}"
    if what == "delta":
        return f"{R2.read_delta(seed)['delta']!r}"
    if what == "kind":
        r = read_kind_means(seed, refs)
        return " ".join(f"{k}={r['m_kind'][k]!r}" for k in KINDS)
    raise ValueError(what)


# --------------------------------------------------------------------------- #
# finalize: ledger, claims.json, RESULTS.md
# --------------------------------------------------------------------------- #


def _rc(out_root: Path, name: str) -> int | None:
    p = Path(out_root) / "logs" / f"{name}.rc"
    if not p.exists():
        return None
    t = p.read_text().strip()
    return int(t) if t.lstrip("-").isdigit() else None


def summarize(docs: dict[int, dict], launch: dict | None, seeds, out_root: Path) -> dict:
    exits = [int(d.get("exit", 3)) for d in docs.values()]
    missing = [s for s in seeds if s not in docs]
    launch_exit = 3 if launch is None else int(launch.get("exit", 3))
    heads = {d.get("git_head") for d in docs.values()}
    same_head = launch is not None and heads == {launch.get("head")}
    ran = sorted(s for s, d in docs.items() if d.get("seeds_actually_run") == [s])
    g_c5 = {
        "seeds_actually_run": ran,
        "registered": list(seeds),
        "ok": ran == list(seeds),
    }
    smoke = any(d.get("smoke") for d in docs.values())
    rcs = {str(s): _rc(out_root, f"1a-s{s}") for s in seeds}
    rc_ok = all(rcs[str(s)] == int(docs[s]["exit"]) for s in docs) if docs else False
    run_controls = {
        "launch": {"exit": launch_exit, "ok": launch_exit == 0},
        "same_head": {
            "heads": sorted(str(h) for h in heads),
            "launch_head": (launch or {}).get("head"),
            "ok": same_head,
        },
        "G-C5": g_c5,
        "all_seeds_present": {"missing": missing, "ok": not missing},
        "process_rc_equals_recorded_exit": {"rc": rcs, "ok": rc_ok},
        "not_smoke": {"ok": not smoke},
    }
    failed = [k for k, v in run_controls.items() if not v["ok"]]
    code = C.worst_exit([*exits, launch_exit, *([3] if failed else [])])
    labels = {a: {} for a in NEW_ARMS}
    for s, d in docs.items():
        for a in NEW_ARMS:
            labels[a][s] = (
                ((d.get("stats") or {}).get("headline") or {}).get(a) or {}
            ).get("label")
    readings = {a: reading(labels[a], seeds) for a in NEW_ARMS}
    read = code == 0
    for a in NEW_ARMS:
        readings[a]["read"] = read
        if not read:
            readings[a]["text"] = f"NOT READ (item exit {code})"
            readings[a]["pattern"] = "NOT_READ"
    readings["factfiller"]["role"] = "registered reading (PLAN §2); Gate P2's input"
    readings["kindoracle"]["role"] = "reported and read the same way; never gates"
    return {
        "exit": code,
        "status": C.LEDGER_STATUS[code],
        "run_controls": run_controls,
        "run_controls_failed": failed,
        "seed_exits": {str(s): int(d.get("exit", 3)) for s, d in docs.items()},
        "labels": {a: {str(s): v for s, v in labels[a].items()} for a in NEW_ARMS},
        "readings": readings,
        "documents": "EVAL, already used",
    }


def _f(x, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if _finite(x) else "n/a"


def _ci(c: dict | None) -> str:
    c = c or {}
    return f"{_f(c.get('point'))} [{_f(c.get('lo'))}, {_f(c.get('hi'))}]"


def render_results(
    sm: dict, docs: dict[int, dict], launch: dict | None, led_doc: dict, seeds
) -> str:
    """RESULTS.md, every number read from the measure documents the ledger was built
    from (no number typed)."""
    w: list[str] = []
    p = w.append
    prov = led_doc.get("provenance") or {}
    p(f"# Item 1a: FactFiller and the kind-oracle through the frozen reader ({RUN_ID})\n")
    p(
        f"PREREG `{C.PREREG_REL}` §3 (committed alone at {C.PREREG_COMMIT}). "
        f"Generated by `{EXPERIMENT} finalize 1a` from `runs/{RUN_ID}/ledger.json`; "
        "no number is typed.\n"
    )
    p(
        "**Documents: EVAL, already used** (run 1, run 2, Expire-Span, offline-psi, "
        "run2-vs-random). **The outcome is largely SEEN by proxy** (PREREG §8).\n"
    )
    p("## Provenance\n")
    p(
        f"- git sha (ledger): `{prov.get('git_sha')}`; launch head: "
        f"`{(launch or {}).get('head')}`; platform `{prov.get('platform')}`; device cpu, "
        f"1 thread per process"
    )
    p(
        f"- manifest: `runs/{RUN_ID}/manifest.json`, "
        f"config hash `{led_doc.get('config_hash')}`"
    )
    p(f"- seeds actually run: {led_doc.get('seeds_actually_run')}")
    for c in led_doc.get("commands", []):
        p(f"- `{c['argv']}` exit {c['exit_code']} ({c['note']})")
    p("")
    p(f"## Exit {sm['exit']} ({sm['status']})\n")
    p(
        f"- per-seed exits: {sm['seed_exits']}; run-level controls failed: "
        f"{sm['run_controls_failed'] or 'none'}"
    )
    if launch:
        g3 = launch.get("G-C3") or {}
        p(
            f"- G-C3 suite at the launch head: rc {g3.get('rc')}, "
            f"passed {g3.get('passed')}, "
            f"failed {g3.get('failed')}, skipped {g3.get('skipped')}, errors "
            f"{g3.get('errors')}; skipped tests: {g3.get('skipped_tests') or 'none'}"
        )
        s1 = launch.get("G-C1.search") or {}
        p(
            f"- G-C1(e) branch search: {s1.get('n_refs')} refs, {s1.get('n_hits')} hits, "
            f"{s1.get('n_new_hits')} new hits, ok {s1.get('ok')}"
        )
    p("")
    p("## Labels (run 2's `seed_label`, δ_seed from B2's ledger)\n")
    p("| arm | " + " | ".join(f"seed {s}" for s in seeds) + " | reading |")
    p("|---|" + "---|" * len(seeds) + "---|")
    for a in NEW_ARMS:
        r = sm["readings"][a]
        cells = []
        for s in seeds:
            h = (((docs.get(s) or {}).get("stats") or {}).get("headline") or {}).get(
                a
            ) or {}
            cells.append(f"{h.get('label')} {_ci(h)} δ={_f(h.get('delta'))}")
        p(f"| {a} | " + " | ".join(cells) + f" | {r['pattern']}: {r['text']} |")
    p("")
    p(
        "FactFiller's row is the registered reading (PLAN §2); the kind-oracle's is "
        "read the "
        "same way and never gates. Gate P2 is applied by the PM after verification, not "
        "here.\n"
    )
    p("## Per seed\n")
    for s in seeds:
        d = docs.get(s) or {}
        st = d.get("stats") or {}
        p(
            f"### seed {s}: exit {d.get('exit')}, controls failed "
            f"{d.get('controls_failed') or 'none'}\n"
        )
        if not st:
            p(f"No statistics: {d.get('notes')}\n")
            continue
        p(
            "| bucket | acc fifo (run 2) | acc factfiller | acc kindoracle | "
            "ff - fifo | ko - fifo |"
        )
        p("|---|---|---|---|---|---|")
        for b in BUCKETS:
            ac, co = st["accuracy"][b], st["contrast"][b]
            p(
                f"| {b} | {_ci(ac['fifo'])} | {_ci(ac['factfiller'])} | "
                f"{_ci(ac['kindoracle'])} | {_ci(co['factfiller'])} | "
                f"{_ci(co['kindoracle'])} |"
            )
        p("")
        p(
            "| bucket | hit fifo | hit factfiller | hit kindoracle | "
            "ff - fifo | ko - fifo |"
        )
        p("|---|---|---|---|---|---|")
        for b in BUCKETS:
            hr, hc = st["hit_rate"][b], st["hit_contrast"][b]
            p(
                f"| {b} | {_ci(hr['fifo'])} | {_ci(hr['factfiller'])} | "
                f"{_ci(hr['kindoracle'])} | {_ci(hc['factfiller'])} | "
                f"{_ci(hc['kindoracle'])} |"
            )
        p("")
        ag = st["age1_share"]
        p(
            "- age-1 eviction share: "
            + "; ".join(
                f"{k} {_f(v.get('share'))} ({v.get('n_age1')}/{v.get('n_evictions')})"
                for k, v in ag.items()
            )
        )
        ko = (d.get("controls") or {}).get("A-C7") or {}
        p(
            f"- m_kind (B0, SEEN): {ko.get('m_kind')}; order ascending "
            f"{ko.get('order_ascending')}"
        )
        t = (d.get("passes") or {}).get("timing") or {}
        p(
            "- pass wall time (s): "
            + "; ".join(f"{k} {_f(v.get('total_s'), 1)}" for k, v in t.items())
        )
        p("")
        p("| control | ok |")
        p("|---|---|")
        for k, v in sorted((d.get("controls") or {}).items()):
            p(f"| {k} | {v.get('ok')} |")
        p("")
    p("## What 1a does NOT show (PREREG §3, quoted)\n")
    p(prereg_passage("**What 1a does NOT show.**", "## 4. Item 1b"))
    p("")
    p("## Expectation (PREREG §10, quoted, written before any number)\n")
    p(expectation_text())
    p("")
    p("## Notes\n")
    p(
        "- Test files: `tests/test_post_fails_controls_common.py` and "
        "`tests/test_post_fails_controls_1a.py` (a naming-only split of PREREG §0's "
        "single "
        "`tests/test_post_fails_controls.py`)."
    )
    p(
        "- Every statistic is per seed; nothing is averaged across seeds into a claim. "
        "The ledger's cross-seed rows carry `sd_exactly_zero`."
    )
    return "\n".join(w) + "\n"


def write_claims(docs: dict[int, dict], out_root: Path, seeds) -> Path:
    slot = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        "--slots 1 --wait 3600 -- .venv/bin/python"
    )
    base = f"{slot} {EXPERIMENT} 1a recompute --out-root {Path(out_root).resolve()}"
    claims = []
    for s in seeds:
        if not (docs.get(s) or {}).get("stats"):
            continue
        for a in NEW_ARMS:
            claims.append(
                {
                    "claim": f"seed {s} {a} minus FIFO all-query accuracy, CI, label",
                    "command": f"{base} --what headline --seed {s} --arm {a}",
                    "expected": recompute("headline", out_root, s, a),
                }
            )
            claims.append(
                {
                    "claim": f"seed {s} {a} age-1 evictions / evictions",
                    "command": f"{base} --what age1 --seed {s} --arm {a}",
                    "expected": recompute("age1", out_root, s, a),
                }
            )
        for what in ("ac2", "ac3", "delta", "kind"):
            claims.append(
                {
                    "claim": f"seed {s} {what}",
                    "command": f"{base} --what {what} --seed {s}",
                    "expected": recompute(what, out_root, s),
                }
            )
        claims.append(
            {
                "claim": f"seed {s} factfiller pooled in-loop hit rate (A-C6)",
                "command": f"{base} --what hit --seed {s} --arm factfiller",
                "expected": recompute("hit", out_root, s, "factfiller"),
            }
        )
    p = Path(out_root) / "claims.json"
    p.write_text(json.dumps(claims, indent=2) + "\n")
    return p


def finalize_item(out_root: Path, *, argv_self: list[str]) -> int:
    out_root = Path(out_root)
    C.install_seal()
    seeds = C.registered_seeds()
    lp = out_root / "launch.json"
    launch = json.loads(lp.read_text()) if lp.exists() else None
    docs = {}
    for s in seeds:
        f = out_root / f"measure-seed{s}.json"
        if f.exists():
            docs[s] = json.loads(f.read_text())
    sm = summarize(docs, launch, seeds, out_root)
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    if launch and launch.get("manifest"):
        led.doc["config_hash"] = launch["manifest"]["config_hash"]
        led.doc["manifest_written_utc"] = launch["manifest"]["manifest_written_utc"]
    py = ".venv/bin/python"
    led.command(
        [py, EXPERIMENT, "1a", "launch-checks"],
        exit_code=_rc(out_root, "launch-checks"),
        note="G-C1(e), G-C3, manifest",
    )
    for s in seeds:
        led.command(
            [py, EXPERIMENT, "1a", "measure", "--seed", str(s)],
            exit_code=_rc(out_root, f"1a-s{s}"),
            note=f"seed {s}",
        )
    led.command([py, EXPERIMENT, *argv_self], exit_code=sm["exit"], note="this finalize")
    n_req = None
    done = [d.get("documents_done") for d in docs.values()]
    if launch:
        rng = json.loads((out_root / "manifest.json").read_text())["eval_range"]
        n_req = rng[1] - rng[0]
    led.run_meta(
        device="cpu",
        seeds_actually_run=sm["run_controls"]["G-C5"]["seeds_actually_run"],
        steps_requested=n_req,
        steps_done=min(done) if done and None not in done else None,
    )
    led.note(
        "units", "steps_* count EVAL documents per seed (no training)", how="item_1a"
    )
    led.note("summary", sm, how=f"{EXPERIMENT} finalize 1a")
    led.note("prereg", {"path": C.PREREG_REL, "commit": C.PREREG_COMMIT}, how="item_1a")
    if launch:
        for k, v in C.flatten("launch", launch, max_depth=2):
            led.note(k, v, how=f"runs/{RUN_ID}/launch.json")
    for s, d in sorted(docs.items()):
        for k, v in C.flatten(f"seed{s}", d, max_depth=4):
            led.note(k, v, how=f"runs/{RUN_ID}/measure-seed{s}.json")
    if len(docs) >= 2 and all(d.get("stats") for d in docs.values()):
        for a in NEW_ARMS:
            for f in ("point", "lo", "hi"):
                xs = [docs[s]["stats"]["headline"][a][f] for s in sorted(docs)]
                if all(_finite(x) for x in xs):
                    led.stat(
                        f"headline.{a}.{f}",
                        xs,
                        how="per seed, seed order 0,1,2; never averaged into a claim",
                    )
    led.status(sm["status"])
    led.write()
    (out_root / "RESULTS.md").write_text(render_results(sm, docs, launch, led.doc, seeds))
    if sm["exit"] in (0, 1):
        write_claims(docs, out_root, seeds)
    return sm["exit"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def default_out_root() -> Path:
    return ROOT / "runs" / RUN_ID


def main(argv: list[str]) -> int:
    from rsr.exit_codes import ArgumentParser

    p = ArgumentParser(prog="run.py 1a", description="PREREG §3, item 1a")
    sub = p.add_subparsers(dest="cmd", required=True)
    pm = sub.add_parser("measure")
    pm.add_argument("--seed", type=int, required=True)
    pm.add_argument("--out-root", default=str(default_out_root()))
    pm.add_argument("--limit", type=int, default=None, help="smoke only (not a result)")
    pl = sub.add_parser("launch-checks")
    pl.add_argument("--out-root", default=str(default_out_root()))
    pl.add_argument("--suite-log", default=None)
    pl.add_argument("--suite-rc", default=None)
    pl.add_argument("--suite-head", default=None)
    pr = sub.add_parser("recompute")
    pr.add_argument("--out-root", required=True)
    pr.add_argument("--what", required=True)
    pr.add_argument("--seed", type=int, required=True)
    pr.add_argument("--arm", default=None)
    a = p.parse_args(argv)
    if a.cmd == "measure":
        if a.seed not in C.registered_seeds():
            print(f"DID NOT RUN: seed {a.seed} is not registered", file=sys.stderr)
            return 3
        d = measure_seed(a.seed, Path(a.out_root), limit=a.limit)
        print(
            json.dumps(
                {
                    "seed": a.seed,
                    "exit": d["exit"],
                    "status": d["status"],
                    "controls_failed": d["controls_failed"],
                    "notes": d["notes"],
                },
                indent=2,
            )
        )
        return int(d["exit"])
    if a.cmd == "launch-checks":
        out = Path(a.out_root)
        logs = out / "logs"
        r = launch_checks(
            out,
            suite_log=Path(a.suite_log or logs / "suite.log"),
            suite_rc=Path(a.suite_rc or logs / "suite.rc"),
            suite_head=Path(a.suite_head or logs / "suite.head"),
        )
        print(json.dumps({k: r[k] for k in ("head", "ok", "exit")}, indent=2))
        print(
            json.dumps(
                {
                    "G-C3": r["G-C3"],
                    "search_new_hits": r["G-C1.search"].get("new_hits"),
                    "search_ok": r["G-C1.search"]["ok"],
                },
                indent=2,
                default=str,
            )
        )
        return int(r["exit"])
    if a.cmd == "recompute":
        print(recompute(a.what, Path(a.out_root), a.seed, a.arm))
        return 0
    return 3


def finalize(argv: list[str]) -> int:
    from rsr.exit_codes import ArgumentParser

    p = ArgumentParser(prog="run.py finalize 1a")
    p.add_argument("--out-root", default=str(default_out_root()))
    a = p.parse_args(argv)
    code = finalize_item(Path(a.out_root), argv_self=["finalize", "1a", *argv])
    print(Path(a.out_root) / "RESULTS.md")
    return code
