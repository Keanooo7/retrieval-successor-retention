"""RSR run 1's measurement harness (`experiments/rsr-run1/run.py`; PREREG T3, A1, A2).

🔴 **Nothing here reads the real substrate or `~/rsr-substrate`.** The substrate is a
tiny FIFO checkpoint this file trains itself (the real architecture -- d = 128, M = 16,
S = 48 -- so fresh-stream's C4 instrument, which builds S0-03's model, accepts it; batch
2, 40 steps), rewritten as a legacy checkpoint the way arm B's is (its config only in a
hash-bound `result.json`), and listed in a T0-style manifest of its own. The three arms
run through the real `rsr.train.loop` CLI (the launcher's command line, without the slot)
into launcher v2's layout, through the held-out harness, the analysis, `RESULTS.md` and
the ledger. The fixture runs with ``prereg_strict=False``: this tree has no C-2 pool ids,
no slot job records, and may be dirty.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import shlex
import sys
from pathlib import Path

import pytest
import torch

import rsr.train.loop as loop
from rsr.exit_codes import Exit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "rsr-run1" / "run.py"
SUB_ITERS = 40  # the fixture substrate: measured live at this length
ARM_ITERS = 6  # pool 4 at batch 2 -> T_warm = 2 steps; CKPT = 2 -> snapshots 2, 4, 6
POOL = (993000, 993004)
EVAL = (995000, 995008)


@pytest.fixture(scope="module")
def r1():
    spec = importlib.util.spec_from_file_location("rsr_run1_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rsr_run1_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def substrate(tmp_path_factory, r1) -> dict:
    """A miniature arm B: a FIFO checkpoint, legacy (no `meta.config`, no map hash),
    its config in `result.json` bound by the run_id hash, a T0-style manifest, and a
    C4 reference ledger holding the fixture's OWN `R_quantity`, measured with
    fresh-stream's instrument -- so C4 asks what it asks of arm B: does the frozen FIFO
    arm reproduce its substrate's R?"""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("substrate")
    run = root / "train"
    loop.train(
        policy_name="fifo",
        d=128,
        batch=2,
        iters=SUB_ITERS,
        steps_per_stream=48,
        memory_slots=16,
        device="cpu",
        out_dir=run,
        ckpt_every=SUB_ITERS,
        beat_every=1,
    )
    (found,) = sorted(run.rglob(f"ckpt-{SUB_ITERS:06d}.pt"))
    payload = torch.load(found, map_location="cpu", weights_only=False)
    cfg = payload["meta"].pop("config")
    payload["meta"].pop("vocab_sha256")
    sdir = root / "B" / "seed0"
    sdir.mkdir(parents=True)
    ckpt = sdir / f"ckpt-{SUB_ITERS:06d}.pt"
    torch.save(payload, ckpt)
    result = sdir / "result.json"
    result.write_text(json.dumps({"config": cfg}))
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(f"{_sha(ckpt)}  {ckpt.relative_to(root).as_posix()}\n")
    fs = r1.FS()
    ref = float(fs.r_quantity(fs.measure_checkpoint(sdir, 0, SUB_ITERS, 64)["s003"]))
    ledger = root / "ref-ledger.json"
    rows = [{"key": r1.C4_KEY, "samples": [ref]}]
    ledger.write_text(json.dumps({"seeds_actually_run": [0], "rows": rows}))
    return {
        "root": root,
        "ckpt": ckpt,
        "sha": _sha(ckpt),
        "manifest": manifest,
        "result": result,
        "ledger": ledger,
        "ref": ref,
        "model": payload["model"],
    }


def tiny_cfg(r1, sub: dict, **kw):
    s = r1.Substrate(
        checkpoint=str(sub["ckpt"]),
        sha256=kw.pop("sha256", sub["sha"]),
        manifest=str(sub["manifest"]),
        root=str(sub["root"]),
        source_config=str(sub["result"]),
    )
    base = dict(
        seed=0,
        substrate=s,
        iters=ARM_ITERS,
        batch=2,
        pool=POOL,
        eval_range=EVAL,
        eval_batch=4,
        c4_ledger=str(sub["ledger"]),
        n_boot=200,
        prereg_strict=False,
    )
    base.update(kw)
    return r1.Run1Config(**base)


def owner(r1, **kw):
    base = dict(gamma=0.9, phi_weight_decay=0.01, iters=ARM_ITERS, o5="train", seeds=(0,))
    base.update(kw)
    return r1.OwnerValues(**base)


def _ledger_seams(mp, r1) -> None:
    """Test seams on scripts/ledger.py: the worktree under test may hold uncommitted
    edits, and the real gate (`Ledger.write` refusing dirty source) is the run's."""
    L = r1.ledger_mod()
    mp.setattr(L, "_dirty_source_paths", lambda: [])
    mp.setattr(L, "_exists_at_sha", lambda path, sha: True)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, r1, substrate) -> dict:
    """launch_direct -> measure_seed -> finalize, once, on the fixture."""
    out_root = tmp_path_factory.mktemp("e2e") / "runs" / r1.RUN_ID
    cfg = tiny_cfg(r1, substrate)
    ow = owner(r1)
    with pytest.MonkeyPatch.context() as mp:
        _ledger_seams(mp, r1)
        rcs = r1.launch_direct([cfg], ow, out_root, threads=1, allow_dirty=True)
        doc = r1.measure_seed(cfg, ow, out_root, threads=1)
        results = out_root / "RESULTS.md"
        code = r1.finalize([doc], ow, out_root, results_path=results)
    return {
        "out_root": out_root,
        "cfg": cfg,
        "owner": ow,
        "rcs": rcs,
        "doc": doc,
        "code": code,
        "results": results,
    }


def _payload(r1, e2e, arm, step=ARM_ITERS):
    return r1.read_checkpoint(
        r1.checkpoint_path(r1.arm_dir(e2e["out_root"], arm, 0), step)
    )


def _results(r1, e2e) -> dict:
    return {
        a: r1.read_cli_result(e2e["out_root"] / "logs" / f"{a}-s0.log")
        for a in ("psiC", "fifo", "age")
    }


def _hbs(r1, e2e) -> dict:
    return {a: r1.read_heartbeat(r1.arm_dir(e2e["out_root"], a, 0)) for a in r1.ARMS}


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_e2e_every_arm_ran_live_with_its_own_rc(e2e):
    assert e2e["rcs"] == {"psiC-s0": 0, "fifo-s0": 0, "age-s0": 0}
    for a in ("psiC", "fifo", "age"):
        assert (e2e["out_root"] / "logs" / f"{a}-s0.rc").read_text() == "0\n"
        rec = e2e["doc"]["arms"][a]
        assert rec["rc_meaning"] == "ran, live" and rec["run_id"] and rec["config_hash"]


def test_e2e_every_control_passes_and_the_run_exits_0(e2e):
    doc = e2e["doc"]
    assert doc["controls_failed"] == [], json.dumps(doc["controls"], default=str)[:4000]
    assert doc["status"] == "ok" and e2e["code"] == Exit.OK
    assert doc["scoped_controls"]["C9_C9b"]["ok"] and doc["oracle_numbers_read"]


def test_e2e_c4_reproduces_the_substrate_exactly(e2e, substrate):
    c4 = e2e["doc"]["controls"]["C4"]
    assert c4["reference"] == substrate["ref"]
    assert c4["abs_diff"] == 0.0 and c4["ok"] is True


def test_e2e_launch_record_precedes_every_arm(e2e):
    lg = e2e["out_root"] / "launch-git.json"
    body = json.loads(lg.read_text())
    assert (
        body["o5"] == "train" and body["ckpt"] == "2" and body["steps"] == str(ARM_ITERS)
    )
    for a in ("psiC", "fifo", "age"):
        hb = e2e["out_root"] / a / "seed0" / "heartbeat.jsonl"
        assert lg.stat().st_mtime <= hb.stat().st_birthtime


def test_e2e_per_document_answer_vectors_are_written(e2e, r1):
    hd = e2e["out_root"] / "heldout" / "seed0"
    for name, _src, _z in r1.heldout_passes([2, 4]):
        recs = r1.read_pass(hd, name)
        assert [r["doc"] for r in recs] == list(range(*EVAL))
        for r in recs:
            assert all(len(a) == 4 for a in r["answers"])
            assert r["real_n"] > 0


def test_e2e_fifo_evicts_age_m_and_shifts_nothing_held_out(e2e):
    ev = e2e["doc"]["heldout"]["heldout_evictions"]["fifo"]
    assert set(ev["victim_age_distribution"]) == {16}
    assert ev["displacement_distribution"] == {0: ev["n_evictions"]}


def test_e2e_snapshots_are_measured_as_diagnostics(e2e):
    snaps = e2e["doc"]["heldout"]["snapshots"]
    assert set(snaps) == {"psiC@2", "psiC@4", "age@2", "age@4"}
    assert "DIAGNOSTIC snapshot `psiC@2`" in e2e["results"].read_text()


def test_e2e_results_carry_headline_scope_caveat_and_no_forbidden_word(e2e, r1):
    text = e2e["results"].read_text()
    assert "## 1. HEADLINE" in text and "(HEADLINE)" in text
    assert r1.scope_sentence() in text and r1.MULTIPLICITY in text
    assert "The three per-seed values show the **combined** variation" in text
    assert r1.caveat_text() in text and r1.expectation_text() in text
    assert r1.confound_text() in text  # O5 = train in the fixture (A2.1)
    assert "TRAIN/EVAL MEMORY SHIFT (psiC only)" in text
    assert r1.LRU_CAVEAT in text and r1.ATTRIBUTION_NOTE in text
    assert "TEST FIXTURE" in text
    assert ("un" + "biased") not in text.lower()
    assert "not implemented" not in text.lower()  # A1.1: LRU runs
    assert "met or missed" in text and "**met**" not in text


def test_e2e_ledger_is_written_with_the_run_meta(e2e, r1):
    led = json.loads((e2e["out_root"] / "ledger.json").read_text())
    assert led["seeds_actually_run"] == [0] and led["status"] == "ok"
    assert led["steps_requested"] == ARM_ITERS and led["steps_done"] == ARM_ITERS
    lg = (e2e["out_root"] / "launch-git.json").read_bytes()
    assert led["config_hash"] == hashlib.sha256(lg).hexdigest()
    keys = {r["key"] for r in led["rows"]}
    assert "headline" in keys and any(k.startswith("seed0.controls.C10") for k in keys)


def test_e2e_headline_is_per_seed_and_across_seeds_with_a_sign_count(e2e):
    run = json.loads((e2e["out_root"] / "run.json").read_text())
    h = run["headline"]
    assert h["contrast"] == "acc.all.psiC_minus_fifo"
    assert h["per_seed"]["0"]["ci_scope"] == "within-seed only"
    a = h["across_seeds"]
    assert a["n_seeds"] == 1 and a["sd"] is None
    assert sum(a["sign_count"].values()) == 1


def test_the_headline_across_three_seeds(r1):
    def doc(seed, point):
        con = {"acc.all.psiC_minus_fifo": {"point": point, "lo": -1, "hi": 1}}
        return {"seed": seed, "heldout": {"contrasts": con}}

    h = r1.headline([doc(0, 0.01), doc(1, -0.02), doc(2, 0.0)])
    a = h["across_seeds"]
    assert a["points"] == [0.01, -0.02, 0.0] and a["mean"] == pytest.approx(-0.01 / 3)
    assert a["sd"] == pytest.approx(
        math.sqrt(
            ((0.01 + 0.01 / 3) ** 2 + (-0.02 + 0.01 / 3) ** 2 + (0.01 / 3) ** 2) / 2
        )
    )
    assert a["sign_count"] == {"above_0": 1, "below_0": 1, "exactly_0": 1}


def test_the_decision_context_choice_does_not_change_held_out_decisions(e2e, r1):
    """In eval mode the loop's own srep IS an eval-mode srep, so the extra eval forward
    (training's L2 path) buys nothing held out: same victims and answers (an option to
    cut the RSR arms' cost -- not taken silently)."""
    cfg = e2e["cfg"]
    payload = _payload(r1, e2e, "psiC")
    model = r1.model_from_payload(payload)
    docs = r1.docs_by_id(cfg, range(*EVAL))[:4]
    vmap = r1.vocab_map(cfg)
    runs = [
        r1.run_batch(
            model,
            r1.eval_policy("psiC", payload, cfg),
            docs,
            vmap,
            cfg,
            decision_context=dc,
        )
        for dc in ("eval_forward", "step_forward")
    ]
    assert [r["victims"] for r in runs[0]] == [r["victims"] for r in runs[1]]
    assert [r["answers"] for r in runs[0]] == [r["answers"] for r in runs[1]]


def test_the_held_out_policy_runs_the_scored_rule_with_the_trained_head(e2e, r1):
    """§7.1: phi from ``__head__``, ``set_train_step`` at the checkpoint's step -- so the
    scored rule decides (attribution "psi" / "age_only"), never the FIFO warmup."""
    cfg = e2e["cfg"]
    for arm, attribution in (("psiC", "psi"), ("age", "age_only")):
        for step in (2, ARM_ITERS):
            payload = _payload(r1, e2e, arm, step)
            pol = r1.eval_policy(arm, payload, cfg)
            head = r1.head_state(payload)
            assert all(torch.equal(v, head[k]) for k, v in pol.head.state_dict().items())
            model = r1.model_from_payload(payload)
            docs = r1.docs_by_id(cfg, range(*EVAL))[:2]
            r1.run_batch(model, pol, docs, r1.vocab_map(cfg), cfg)
            assert pol.records and {r.attribution for r in pol.records} == {attribution}
            assert {r.warm for r in pol.records} == {False}
            assert {r.train_step for r in pol.records} == {step}


def test_in_loop_residency_matches_the_replay_for_every_reference(r1, substrate):
    """C7's two records are independent: in-loop residency is read off the memory the
    loop holds, the replay off the victims. Every reference policy, held out."""
    cfg = tiny_cfg(r1, substrate)
    payload = r1.read_checkpoint(substrate["ckpt"])
    payload["meta"]["config"] = json.loads(substrate["result"].read_text())["config"]
    model = r1.model_from_payload(payload)
    docs = r1.docs_by_id(cfg, range(*EVAL))[:4]
    vmap = r1.vocab_map(cfg)
    for name in ("oracle", "lru", "random0"):
        pol = r1.reference_policy(name, docs, 0)
        recs = r1.run_batch(model, pol, docs, vmap, cfg, observe=name == "lru")
        assert all(
            r1.residency_ok(d, r, cfg.memory_slots)
            for d, r in zip(docs, recs, strict=True)
        ), name
    recs = r1.run_batch(model, r1.FIFOPolicy(), docs, vmap, cfg)
    assert all(r1.residency_ok(d, r, 16) for d, r in zip(docs, recs, strict=True))


def test_lru_observes_held_out_and_so_departs_from_fifo(r1, substrate):
    """T11: LRU reads `attn.alpha` in `observe`; without observe it would be FIFO."""
    cfg = tiny_cfg(r1, substrate)
    payload = r1.read_checkpoint(substrate["ckpt"])
    payload["meta"]["config"] = json.loads(substrate["result"].read_text())["config"]
    model = r1.model_from_payload(payload)
    docs = r1.docs_by_id(cfg, range(*EVAL))[:4]
    vmap = r1.vocab_map(cfg)
    fifo = r1.run_batch(model, r1.FIFOPolicy(), docs, vmap, cfg)
    blind = r1.run_batch(model, r1.LRUPolicy(), docs, vmap, cfg)
    seeing = r1.run_batch(model, r1.LRUPolicy(), docs, vmap, cfg, observe=True)
    assert [r["victims"] for r in blind] == [r["victims"] for r in fifo]
    assert [r["victims"] for r in seeing] != [r["victims"] for r in fifo]


# --------------------------------------------------------------------------- #
# C9 / C9b (scoped to the oracle numbers)
# --------------------------------------------------------------------------- #


def _fifo_model(r1, e2e):
    return r1.model_from_payload(_payload(r1, e2e, "fifo"))


def test_c9_and_c9b_are_green_on_the_row_correct_oracle(r1, e2e):
    cfg = e2e["cfg"]
    c = r1.c9_oracle(_fifo_model(r1, e2e), cfg, r1.vocab_map(cfg))
    assert c["ok"] and c["documents"] == [995000, 995008]


def test_c9b_sees_a_batched_oracle_with_another_rows_demand(r1, e2e):
    """review 7fcd3f91ca8f MINOR-A2-2: a wrong-document demand is identical at both
    gammas (C9 passes) but differs from each document's own model-free replay."""
    cfg = e2e["cfg"]

    def rotated(chunk, g):
        return r1.oracle_policy(chunk[1:] + chunk[:1], g)

    c = r1.c9_oracle(_fifo_model(r1, e2e), cfg, r1.vocab_map(cfg), policy_fn=rotated)
    assert c["C9"]["ok"] and not c["C9b"]["ok"] and not c["ok"]


def test_c9_sees_a_demand_whose_order_depends_on_gamma(r1, e2e):
    cfg = e2e["cfg"]

    def warped(chunk, g):
        pols = []
        for d in chunk:
            rng = random.Random(f"{d.doc_id}:{g}")
            dem = [[v + rng.random() for v in row] for row in r1.discounted_demand(d, g)]
            pols.append(r1.OraclePolicy(dem))
        return r1.PerRow(pols, "oracle")

    c = r1.c9_oracle(_fifo_model(r1, e2e), cfg, r1.vocab_map(cfg), policy_fn=warped)
    assert not c["C9"]["ok"] and not c["ok"]


def test_a_c9_failure_marks_only_the_oracle_numbers_not_read(r1, e2e):
    doc = copy.deepcopy(e2e["doc"])
    doc["oracle_numbers_read"] = False
    doc["scoped_controls"]["C9_C9b"]["ok"] = False
    run = r1.headline([doc])
    text = r1.render_results(
        {
            "status": "ok",
            "exit": 0,
            "owner_values": dict(doc["owner_values"]),
            "headline": run,
            "tree": {},
            "prereg_strict": False,
        },
        [doc],
    )
    for line in text.splitlines():
        if line.startswith(("- `oracle_regret.", "- headroom share")):
            assert "NOT READ (C9/C9b failed)" in line, line
        if line.startswith("- `acc.all.psiC_minus_fifo`"):
            assert "NOT READ" not in line
    assert doc["exit"] == 0  # C9 is not in `controls`: the run's exit is unchanged


# --------------------------------------------------------------------------- #
# Controls go red when their invariant is broken
# --------------------------------------------------------------------------- #


def _beats(e2e, r1) -> dict:
    return {a: h["beats"] for a, h in _hbs(r1, e2e).items()}


def _sampling(r1, e2e):
    return _results(r1, e2e)["psiC"]["config"]["epoch"]["sampling"]


def test_c2_is_green_on_the_real_arms(e2e, r1):
    c2 = r1.c2_pairing(_beats(e2e, r1), {}, e2e["cfg"], _sampling(r1, e2e))
    assert c2["ok"] and c2["steps_compared"] == ARM_ITERS
    assert c2["a_pool_ids"]["present"] is False  # 211d6da logs no pool ids (C-2)


def test_c2_requires_the_logged_pool_ids_when_strict(e2e, r1):
    cfg = r1.dataclasses.replace(e2e["cfg"], prereg_strict=True)
    c2 = r1.c2_pairing(_beats(e2e, r1), {}, cfg, _sampling(r1, e2e))
    assert not c2["ok"] and c2["a_pool_ids"]["required"]


def test_c2_sees_a_batch_mismatch_in_one_arm(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    beats["fifo"][4]["n_answer_targets"] += 1  # a scored step: loss_lm cannot see it
    c2 = r1.c2_pairing(beats, {}, e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["b_fingerprint"]["cross_arm_mismatch_steps"] == [4]


def test_c2_sees_a_batch_every_arm_shares_but_the_sampler_did_not_draw(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    for a in beats:
        beats[a][5]["pad_target_fraction"] = 0.5
    c2 = r1.c2_pairing(beats, {}, e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["b_fingerprint"]["replay_mismatch_steps"] == [5]
    assert c2["b_fingerprint"]["cross_arm_mismatch_steps"] == []


def test_c2_sees_a_warm_step_loss_that_is_not_bitwise_equal(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    v = beats["age"][1]["loss_lm"]
    beats["age"][1]["loss_lm"] = math.nextafter(v, math.inf)
    c2 = r1.c2_pairing(beats, {}, e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["c_loss_lm_warm_mismatch_steps"] == [1]


def _permutation_arms(r1, cfg, steps):
    """Beats and pool ids for three arms that drew launcher-v2 permutation batches."""
    perm = r1.replay_batches(cfg, steps, "per-epoch permutation (each document once)")
    fp = r1.pool_fingerprints(cfg)
    n_t = cfg.batch * cfg.steps_per_stream * (r1.CLI_MAX_TOKENS - 1)
    beats, ids = {}, {}
    for a in r1.ARMS:
        beats[a] = [
            {
                "step": k,
                "loss_lm": 1.0,
                "n_targets_scored": float(n_t),
                "n_answer_targets": float(sum(fp[i][0] for i in b)),
                "pad_target_fraction": sum(fp[i][1] for i in b) / n_t,
            }
            for k, b in enumerate(perm)
        ]
        ids[a] = {k: [cfg.pool[0] + i for i in b] for k, b in enumerate(perm)}
    return beats, ids


def test_c2a_pool_ids_green_and_red(r1, substrate):
    cfg = tiny_cfg(r1, substrate, prereg_strict=True)
    s = "per-epoch permutation (each document once per epoch)"
    beats, ids = _permutation_arms(r1, cfg, 6)
    assert r1.c2_pairing(beats, ids, cfg, s)["ok"]
    bad = copy.deepcopy(ids)
    bad["age"][3] = list(reversed(bad["age"][3]))
    c2 = r1.c2_pairing(beats, bad, cfg, s)
    assert not c2["ok"] and c2["a_pool_ids"]["cross_arm_mismatch_steps"] == [3]
    dup = copy.deepcopy(ids)
    for a in dup:
        dup[a][1] = list(dup[a][0])  # epoch 0 sees two documents twice
    c2 = r1.c2_pairing(beats, dup, cfg, s)
    assert not c2["ok"] and 0 in c2["a_pool_ids"]["epochs_not_exactly_the_pool"]


def test_the_permutation_replay_visits_every_pool_document_once_per_epoch(r1, substrate):
    cfg = tiny_cfg(r1, substrate)
    b = r1.replay_batches(cfg, 6, "per-epoch permutation")
    spe = int(cfg.steps_per_epoch)
    for e in range(3):
        assert sorted(i for k in range(e * spe, (e + 1) * spe) for i in b[k]) == [
            0,
            1,
            2,
            3,
        ]
    assert b == r1.replay_batches(cfg, 6, "per-epoch permutation")


def test_batch_fingerprint_does_not_read_n_targets_scored(r1):
    """A1.8 m-4: it is `batch x S x (L-1)` every step, so it cannot see a batch."""
    a = {"n_targets_scored": 6048.0, "n_answer_targets": 20.0, "pad_target_fraction": 0.9}
    for other in (
        dict(a, n_answer_targets=21.0),
        dict(a, pad_target_fraction=5400 / 6048),
    ):
        assert other["n_targets_scored"] == a["n_targets_scored"]
        assert r1.batch_fingerprint(a) != r1.batch_fingerprint(other)


def test_c3_sees_one_moved_weight(e2e, r1, substrate):
    payloads = {a: _payload(r1, e2e, a) for a in ("psiC", "fifo", "age")}
    assert r1.c3_frozen(payloads, substrate["model"])["ok"]
    k = next(iter(payloads["age"]["model"]))
    payloads["age"]["model"][k] = payloads["age"]["model"][k] + 1e-7
    c3 = r1.c3_frozen(payloads, substrate["model"])
    assert not c3["ok"] and c3["per_checkpoint"]["age"]["differ"] == [k]


def test_c7_sees_in_loop_residency_that_disagrees_with_the_replay(e2e, r1):
    cfg = e2e["cfg"]
    recs = r1.read_pass(e2e["out_root"] / "heldout" / "seed0", "psiC")
    docs = r1.docs_by_id(cfg, [r["doc"] for r in recs])
    assert all(
        r1.residency_ok(d, r, cfg.memory_slots) for d, r in zip(docs, recs, strict=True)
    )
    bad = copy.deepcopy(recs[0])
    g, hit = bad["hits"][0]
    bad["hits"][0] = [g, not hit]
    assert not r1.residency_ok(docs[0], bad, cfg.memory_slots)


def test_c0_sees_an_overlapping_or_aliasing_range(r1, substrate):
    with pytest.raises(r1.ControlFailure, match="pool_overlaps"):
        r1.check_c0(tiny_cfg(r1, substrate, pool=(4100, 4104)))
    with pytest.raises(r1.ControlFailure, match="eval_below_K"):
        r1.check_c0(tiny_cfg(r1, substrate, eval_range=(1_000_000, 1_000_008)))
    with pytest.raises(r1.ControlFailure, match="pool_eval_disjoint"):
        r1.check_c0(tiny_cfg(r1, substrate, eval_range=(993002, 993010)))
    with pytest.raises(r1.ControlFailure, match="vocab_sha256"):
        r1.check_c0(tiny_cfg(r1, substrate, expected_vocab_sha256="0" * 64))


def test_c1_sees_a_labelled_allow_flag_or_another_checkpoint(e2e, r1, substrate):
    res = _results(r1, e2e)
    assert r1.c1_warm_start(res, e2e["cfg"])["ok"]
    bad = copy.deepcopy(res)
    bad["fifo"]["config"]["warm_start"]["allow_seed_mismatch"] = True
    assert not r1.c1_warm_start(bad, e2e["cfg"])["ok"]
    assert not r1.c1_warm_start(res, tiny_cfg(r1, substrate, sha256="f" * 64))["ok"]


def test_c4_goes_red_on_a_reference_it_does_not_reproduce(
    r1, substrate, monkeypatch, tmp_path
):
    led = tmp_path / "ref.json"
    rows = [{"key": r1.C4_KEY, "samples": [0.5]}]
    led.write_text(json.dumps({"seeds_actually_run": [0], "rows": rows}))

    class Fake:
        @staticmethod
        def measure_checkpoint(seed_dir, seed, label, n):
            assert (label, n) == (ARM_ITERS, 64)
            return {"s003": "m"}

        @staticmethod
        def r_quantity(m):
            return 0.5 + 2e-6

    monkeypatch.setattr(r1, "FS", lambda: Fake)
    cfg = tiny_cfg(r1, substrate, c4_ledger=str(led))
    assert r1.c4_measure(Path("x"), cfg)["ok"] is False
    monkeypatch.setattr(Fake, "r_quantity", staticmethod(lambda m: 0.5 + 5e-7))
    assert r1.c4_measure(Path("x"), cfg)["ok"] is True


def test_c4_refuses_a_ledger_row_that_is_not_the_prereg_value(r1, substrate):
    with pytest.raises(r1.ControlFailure, match="not the PREREG"):
        r1.c4_reference(tiny_cfg(r1, substrate, c4_prereg_value=0.123))


def test_c5_c6_c8_go_red(r1, e2e):
    res = _results(r1, e2e)
    hbs = _hbs(r1, e2e)
    assert r1.c5_liveness(res)["ok"] and r1.c6_steps(res, hbs, e2e["cfg"])["ok"]
    bad = copy.deepcopy(res)
    bad["age"]["liveness"]["band"] = "inert"
    c5 = r1.c5_liveness(bad)
    assert not c5["ok"] and c5["inert"] == ["age"]
    bad["psiC"]["steps_done"] = ARM_ITERS - 1
    assert not r1.c6_steps(bad, hbs, e2e["cfg"])["ok"]
    beats = copy.deepcopy({a: h["beats"] for a, h in hbs.items()})
    assert r1.c8_finite(beats)["ok"]
    beats["psiC"][3]["loss_mc"] = float("nan")
    beats["age"][2]["phi_grad_norm"] = None
    c8 = r1.c8_finite(beats)
    assert c8["per_arm"]["psiC"]["nonfinite_steps"] == [3]
    assert c8["per_arm"]["age"]["nonfinite_steps"] == [2]


def _c10(
    r1, e2e, *, cfg=None, owner_=None, launch=None, results=None, hbs=None, jobs=None
):
    lg = json.loads((e2e["out_root"] / "launch-git.json").read_text())
    return r1.c10_launch_values(
        cfg or e2e["cfg"],
        owner_ or e2e["owner"],
        launch if launch is not None else lg,
        results or _results(r1, e2e),
        hbs or _hbs(r1, e2e),
        jobs if jobs is not None else dict.fromkeys(r1.ARMS),
        e2e["out_root"],
    )


def test_c10_is_green_on_the_fixture(r1, e2e):
    c = _c10(r1, e2e)
    assert c["ok"], c


def test_c10_sees_a_value_the_launcher_recorded_but_the_job_did_not_run(r1, e2e):
    """review 7fcd3f91ca8f MINOR-A2-1: values, not flag presence."""
    lg = json.loads((e2e["out_root"] / "launch-git.json").read_text())
    for key, val in (
        ("wd", "0.0"),
        ("gamma", "0.5"),
        ("ckpt", "3"),
        ("o5", "eval"),
        ("steps", "9"),
    ):
        assert not _c10(r1, e2e, launch={**lg, key: val})["ok"], key
    res = _results(r1, e2e)
    for arm, path, val in (
        ("fifo", ("frozen_model_mode",), "eval (dropout off)"),
        ("age", ("optimizer", "weight_decay"), 0.0),
        ("psiC", ("policy_overrides", "gamma"), 0.5),
        ("psiC", ("arm_label",), "rsr[gamma=0.5]"),
    ):
        bad = copy.deepcopy(res)
        node = bad[arm]["config"]
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = val
        assert not _c10(r1, e2e, results=bad)["ok"], (arm, path)


def test_c10_sees_a_cadence_that_is_not_the_derived_one(r1, e2e):
    """The arms wrote checkpoints at 2, 4, 6; a ruled R-1 of 9 derives CKPT 3."""
    ow = owner(r1, iters=9)
    cfg = r1.dataclasses.replace(e2e["cfg"], iters=9)
    c = _c10(r1, e2e, cfg=cfg, owner_=ow)
    assert not c["ok"]
    assert any("cadence" in p for v in c["per_arm"].values() for p in v["problems"])


def test_c10_sees_disagreeing_provenance(r1, e2e):
    hbs = copy.deepcopy(_hbs(r1, e2e))
    hbs["age"]["header"]["provenance"]["git_sha"] = "0" * 40
    assert not _c10(r1, e2e, hbs=hbs)["ok"]


def test_c10_strict_wants_each_job_record_pinned_to_the_run_sha(r1, e2e):
    cfg = r1.dataclasses.replace(e2e["cfg"], prereg_strict=True)
    lg = json.loads((e2e["out_root"] / "launch-git.json").read_text())
    clean = {**lg, "dirty": False}
    job = {"pinned_sha": lg["run_sha"], "cmd": ["x", "--ckpt-every", "2"]}
    assert not _c10(r1, e2e, cfg=cfg, launch=clean)["ok"]  # no job records
    assert _c10(r1, e2e, cfg=cfg, launch=clean, jobs=dict.fromkeys(r1.ARMS, job))["ok"]
    wrong = {**job, "pinned_sha": "f" * 40}
    assert not _c10(
        r1, e2e, cfg=cfg, launch=clean, jobs={**dict.fromkeys(r1.ARMS, job), "age": wrong}
    )["ok"]
    every3 = {**job, "cmd": ["x", "--ckpt-every", "3"]}
    assert not _c10(r1, e2e, cfg=cfg, launch=clean, jobs=dict.fromkeys(r1.ARMS, every3))[
        "ok"
    ]
    assert not _c10(
        r1,
        e2e,
        cfg=cfg,
        launch={**clean, "dirty": True},
        jobs=dict.fromkeys(r1.ARMS, job),
    )["ok"]


def test_the_training_analysis_reads_loss_lm_never_loss(e2e, r1):
    hbs = {a: h for a, h in _hbs(r1, e2e).items() if a in ("psiC", "fifo")}
    recs = r1.read_step_records(r1.arm_dir(e2e["out_root"], "psiC", 0))
    diags = {"psiC": r1.step_diagnostics(recs)}
    a = r1.analyse_training(hbs, diags, e2e["cfg"])
    poisoned = copy.deepcopy(hbs)
    for h in poisoned.values():
        for b in h["beats"]:
            b["loss"] = 1e9
            b["loss_sum"] = 1e9
    assert r1.analyse_training(poisoned, diags, e2e["cfg"]) == a
    for h in poisoned.values():
        h["beats"][3]["loss_lm"] += 1.0
    assert r1.analyse_training(poisoned, diags, e2e["cfg"]) != a


def test_the_diagnostics_split_warm_from_scored_and_report_the_first_block(e2e, r1):
    dg = e2e["doc"]["training"]["diagnostics"]["psiC"]
    tw = int(e2e["cfg"].t_warm_steps)
    assert dg["warm"]["n_steps"] == tw and dg["scored"]["n_steps"] == ARM_ITERS - tw
    assert dg["warm"]["warm_flag_matches_step"] and dg["scored"]["warm_flag_matches_step"]
    assert set(dg["warm"]["victim_age_distribution"]) == {16}  # warmup is FIFO
    fb = dg["first_scored_block"]
    assert fb["n_evictions"] == dg["scored"]["n_evictions"]  # 4 scored steps < 500
    assert sum(fb["uniform_random_expected"].values()) == pytest.approx(fb["n_evictions"])


# --------------------------------------------------------------------------- #
# Launcher, preflight, refusals
# --------------------------------------------------------------------------- #


def test_the_committed_launcher_is_byte_identical_to_a2_7(r1):
    assert (ROOT / r1.LAUNCHER).read_text() == r1.launcher_text_from_prereg()


def _launcher_command(
    r1, arm: str, seed: int, o5: str, gamma: str, out: str
) -> list[str]:
    """Launcher v2's command for (arm, seed), variables filled by text substitution."""
    text = (ROOT / r1.LAUNCHER).read_text()
    i = text.index(".venv/bin/python -m rsr.train.loop")
    j = text.index('"${P[@]}" --out-dir $OUT/$a/seed$s') + len(
        '"${P[@]}" --out-dir $OUT/$a/seed$s'
    )
    cmd = text[i:j].replace("\\\n", " ")
    rsr = (
        "--policy rsr --policy-override b_enabled=false --policy-override "
        "shadow_enabled=false --policy-override nu=0 --policy-override beta=1 "
        f"--policy-override gamma={gamma}"
    )
    p = {
        "psiC": rsr,
        "fifo": "--policy fifo",
        "age": rsr + " --policy-override psi_override=age_only",
    }[arm]
    s = r1.SUBSTRATE[seed]
    b = str(Path(s["checkpoint"]).parent)
    for k, v in (
        ('"${C1[@]}"', " ".join(r1.C1_FLAGS[o5])),
        ('"${C3[@]}"', ""),
        ('"${C2[@]}"', " ".join(r1.C2_FLAGS)),
        ('"${P[@]}"', p),
        ("${SHA[$s]}", s["sha256"]),
        ("$STEPS", "3000"),
        ("$CKPT", "1000"),
        ("$OUT/$a/seed$s", out),
        ("$B", b),
        ("$R", r1._R),
        ("$s", str(seed)),
    ):
        cmd = cmd.replace(k, v)
    return shlex.split(cmd)


@pytest.mark.parametrize("o5", ["train", "eval"])
def test_cli_argv_is_the_launchers_command_token_for_token(r1, o5):
    ow = owner(r1, iters=3000, o5="train")
    ow = r1.dataclasses.replace(ow, o5=o5)  # eval's flag is TO-CONFIRM T9
    for seed in (0, 1, 2):
        cfg = r1.prereg_config(seed, 3000)
        for arm in r1.ARMS:
            out = f"runs/rsr-run1/{arm}/seed{seed}"
            got = r1.cli_argv(cfg, arm, ow, Path(out))
            assert got == _launcher_command(r1, arm, seed, o5, "0.9", out), (arm, seed)


def test_derive_ckpt(r1):
    assert r1.derive_ckpt(3000) == 1000 and r1.derive_ckpt(6) == 2
    assert r1.derive_ckpt(3001) == 3001 and r1.derive_ckpt(1) == 1


def test_env_problems_refuse_a_foreign_orch_root_and_dry_run_hooks(r1):
    assert (
        r1.env_problems(
            {"RSR_ORCH_ROOT": None, "OUT": None, "JOB_SUFFIX": None}, dry_run=False
        )
        == []
    )
    assert r1.env_problems({"RSR_ORCH_ROOT": r1.SHARED_ORCH_ROOT}, dry_run=False) == []
    assert r1.env_problems({"RSR_ORCH_ROOT": "/tmp/iso"}, dry_run=False)
    assert r1.env_problems({"RSR_ORCH_ROOT": "/tmp/iso"}, dry_run=True)
    assert r1.env_problems({"OUT": "x", "JOB_SUFFIX": "-d"}, dry_run=False)
    assert r1.env_problems({"OUT": "x", "JOB_SUFFIX": "-d"}, dry_run=True) == []


def test_measure_refuses_a_foreign_orch_root_when_strict(r1, substrate, tmp_path):
    cfg = tiny_cfg(r1, substrate, prereg_strict=True)
    doc = r1.measure_seed(
        cfg, owner(r1), tmp_path, threads=1, environ={"RSR_ORCH_ROOT": "/tmp/iso"}
    )
    assert doc["status"] == "did_not_run" and doc["exit"] == int(Exit.DID_NOT_RUN)


@pytest.mark.parametrize("drop", ["--gamma", "--phi-weight-decay", "--iters", "--o5"])
def test_the_cli_refuses_without_each_owner_value(r1, drop):
    args = {
        "--gamma": "0.9",
        "--phi-weight-decay": "0.01",
        "--iters": "3000",
        "--o5": "train",
    }
    argv = ["finalize"]
    for k, v in args.items():
        if k != drop:
            argv += [k, v]
    with pytest.raises(SystemExit) as e:
        r1.main(argv)
    assert e.value.code == 3


def test_owner_values_the_code_would_not_run_are_refused(r1):
    r1.validate_owner_values(owner(r1))
    for bad in (
        dict(phi_weight_decay=0.0),
        dict(gamma=1.0),
        dict(gamma=float("nan")),
        dict(seeds=()),
        dict(seeds=(0, 0)),
        dict(iters=0),
        dict(o5="maybe"),
    ):
        with pytest.raises(r1.Refused):
            r1.validate_owner_values(owner(r1, **bad))
    if not r1.frozen_mode_supported("eval"):  # no C-1 on this tree
        with pytest.raises(r1.Refused, match="C-1"):
            r1.validate_owner_values(owner(r1, o5="eval"))
    assert (
        r1.main(
            [
                "finalize",
                "--gamma",
                "0.9",
                "--phi-weight-decay",
                "0.01",
                "--iters",
                "3000",
                "--o5",
                "maybe",
            ]
        )
        == Exit.DID_NOT_RUN
    )


def test_the_registered_shas_are_the_t0_lines_the_prereg_quotes(r1):
    assert r1.SUBSTRATE[0]["sha256"].startswith("0ee3f8a6")
    assert r1.SUBSTRATE[1]["sha256"].startswith("dadd1e08")
    assert r1.SUBSTRATE[1]["sha256"].endswith("3da8")
    assert r1.SUBSTRATE[2]["sha256"].startswith("b507ebc5")
    assert r1.SUBSTRATE[2]["sha256"].endswith("63b8")
    launcher = (ROOT / r1.LAUNCHER).read_text()
    for s in (0, 1, 2):
        assert f"SHA[{s}]={r1.SUBSTRATE[s]['sha256']}" in launcher
    with pytest.raises(r1.Refused):
        r1.prereg_config(3, 3000)


def test_a_started_arm_is_refused_not_resumed(r1, substrate, tmp_path, monkeypatch):
    out_root = tmp_path / "runs" / r1.RUN_ID
    started = r1.arm_dir(out_root, "fifo", 0)
    started.mkdir(parents=True)
    (started / "heartbeat.jsonl").write_text("{}\n")
    monkeypatch.setattr(r1, "tree_state", lambda: pytest.fail("got past the refusal"))
    with pytest.raises(r1.Refused, match="never resumed"):
        r1.launch_direct([tiny_cfg(r1, substrate)], owner(r1), out_root, threads=1)
    assert not (out_root / "launch-git.json").exists()


def test_a_launched_run_root_is_refused(r1, substrate, tmp_path, monkeypatch):
    out_root = tmp_path / "runs" / r1.RUN_ID
    out_root.mkdir(parents=True)
    (out_root / "launch-git.json").write_text("{}\n")
    monkeypatch.setattr(r1, "tree_state", lambda: pytest.fail("got past the refusal"))
    with pytest.raises(r1.Refused, match="launched already"):
        r1.launch_direct([tiny_cfg(r1, substrate)], owner(r1), out_root, threads=1)


def test_a_dirty_tree_is_refused(r1, substrate, tmp_path, monkeypatch):
    monkeypatch.setattr(
        r1, "tree_state", lambda: {"sha": "abc", "dirty": True, "porcelain": " M x"}
    )
    monkeypatch.setattr(r1, "check_c0", lambda c: pytest.fail("got past the refusal"))
    with pytest.raises(r1.Refused, match="clean committed head"):
        r1.launch_direct([tiny_cfg(r1, substrate)], owner(r1), tmp_path / "r", threads=1)


def test_a_wrong_substrate_sha_is_rc_1_with_no_heartbeat_and_measure_says_so(
    r1, substrate, tmp_path
):
    """A1.2 E10, measured: `WarmStartRefused` escapes `main()` -> rc 1, no heartbeat."""
    out_root = tmp_path / "runs" / r1.RUN_ID
    cfg = tiny_cfg(r1, substrate, sha256="0" * 64)
    rcs = r1.launch_direct([cfg], owner(r1), out_root, threads=1, allow_dirty=True)
    assert rcs == {"psiC-s0": 1, "fifo-s0": 1, "age-s0": 1}
    assert "WarmStartRefused" in (out_root / "logs" / "fifo-s0.log").read_text()
    doc = r1.measure_seed(cfg, owner(r1), out_root, threads=1)
    assert doc["status"] == "crashed" and doc["exit"] == int(Exit.DID_NOT_RUN)
    assert doc["arms"]["fifo"]["rc_meaning"].startswith("did not run: exception before")
    with pytest.raises(r1.ControlFailure):
        r1.load_substrate_model(cfg)


def test_time_batch_never_runs_an_eval_document(r1, substrate):
    with pytest.raises(r1.Refused, match="EVAL"):
        r1.time_one_batch(tiny_cfg(r1, substrate), [EVAL[0]])


def test_classify_rc(r1):
    no_hb = {"exists": False, "footer": None, "crash": None}
    crashed = {"exists": True, "footer": None, "crash": {"status": "crashed"}}
    done = {"exists": True, "footer": {"status": "completed"}, "crash": None}
    res = {"liveness": {"band": "live"}}
    assert r1.classify_rc(0, None, done, res) == "ran, live"
    assert r1.classify_rc(1, None, no_hb, None).startswith("did not run: exception")
    assert r1.classify_rc(3, None, no_hb, None).startswith("slot refused")
    assert r1.classify_rc(3, {"rc": 3}, no_hb, None) == "CLI usage error"
    assert r1.classify_rc(1, None, crashed, None).startswith("crashed mid-loop")
    assert r1.classify_rc(1, None, done, res).startswith("liveness inconclusive")
    assert r1.classify_rc(5, None, done, res) == "INERT"
    assert r1.classify_rc(143, {}, crashed, None) == "killed by signal 15"
    assert r1.classify_rc(None, None, no_hb, None).startswith("no rc file")


def test_read_cli_result_takes_the_printed_result_through_noise(r1, tmp_path):
    log = tmp_path / "a.log"
    r = {"run_id": "x", "liveness": {"band": "live"}, "config": {"a": [1, 2]}}
    log.write_text(
        "warning: {\nnot json\n" + json.dumps(r, indent=2) + "\nUserWarning: z\n"
    )
    assert r1.read_cli_result(log) == r
    log.write_text("Traceback\nWarmStartRefused: no\n")
    assert r1.read_cli_result(log) is None


# --------------------------------------------------------------------------- #
# The bootstrap
# --------------------------------------------------------------------------- #


def test_the_bootstrap_is_b2s_paired_bootstrap_exactly(r1):
    g = torch.Generator().manual_seed(7)
    D = 37
    n = torch.randint(1, 9, (D,), generator=g).double()
    per = {}
    for a in ("x", "y"):
        k = torch.minimum(n, torch.randint(0, 9, (D,), generator=g).double())
        per[a] = torch.stack([n, k], 1)
    want = r1.B2().paired_bootstrap(per, {"c": ("x", "y")}, seed=3, n_boot=500)["c"]
    got = r1.contrast(per, "x", "y", r1.boot_weights(D, 3, 500))
    for k in ("point", "lo", "hi", "sd"):
        assert got[k] == want[k], k


def test_the_bootstrap_on_a_hand_case(r1):
    """Three documents, five replicates, every number by hand in plain Python."""
    per = {
        "a": torch.tensor([[2.0, 2.0], [1.0, 0.0], [3.0, 1.0]]),
        "b": torch.tensor([[2.0, 1.0], [1.0, 1.0], [3.0, 0.0]]),
    }
    D, B, seed = 3, 5, 11
    g = torch.Generator().manual_seed(r1.BOOT_SEED_BASE + seed)
    idx = torch.randint(0, D, (B, D), generator=g).tolist()

    def ratio(x, row):
        return sum(x[i][1].item() for i in row) / sum(x[i][0].item() for i in row)

    reps = [ratio(per["a"], row) - ratio(per["b"], row) for row in idx]

    def q(xs, p):  # torch.quantile's default: linear interpolation
        xs = sorted(xs)
        pos = p * (len(xs) - 1)
        lo = math.floor(pos)
        return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)

    c = r1.contrast(per, "a", "b", r1.boot_weights(D, seed, B))
    assert c["point"] == pytest.approx(3 / 6 - 2 / 6, abs=0, rel=1e-15)
    assert c["lo"] == pytest.approx(q(reps, 0.025), rel=1e-12)
    assert c["hi"] == pytest.approx(q(reps, 0.975), rel=1e-12)


def test_ci_reading_has_three_values_and_nothing_else(r1):
    assert r1.ci_reading(0.01, 0.2) == "CI above 0"
    assert r1.ci_reading(-0.2, -0.01) == "CI below 0"
    assert r1.ci_reading(-0.1, 0.1) == "CI includes 0"
    assert r1.ci_reading(0.0, 0.1) == "CI includes 0"
    assert r1.ci_reading(None, 0.1) is None


def test_a_zero_spread_contrast_is_flagged_not_clean(r1):
    x = torch.tensor([[1.0, 1.0]] * 6)
    c = r1.contrast({"a": x, "b": x}, "a", "b", r1.boot_weights(6, 0, 50))
    assert c["sd"] == 0.0 and c["sd_exactly_zero"] is True


def test_the_buckets_are_s0_03s(r1):
    s003 = r1.FS().S003.BUCKETS
    g = torch.arange(0, 49)
    for name, sel in r1.buckets_for(16).items():
        assert torch.equal(sel(g), s003[name](g)), name


# --------------------------------------------------------------------------- #
# T7 (build-time probe)
# --------------------------------------------------------------------------- #


def test_t7_the_oracle_is_one_rule_across_the_probed_gammas(r1, substrate):
    cfg = tiny_cfg(r1, substrate)
    docs = r1.docs_by_id(cfg, range(993000, 993032))
    r = r1.probe_oracle_invariance(docs, 16)
    assert r["ok"] and r["n_documents"] == 32


def test_t7_probe_sees_a_rule_that_changes_with_gamma(r1, substrate, monkeypatch):
    """The probe is not vacuous: a demand whose order depends on gamma goes red."""
    cfg = tiny_cfg(r1, substrate)
    docs = r1.docs_by_id(cfg, range(993000, 993004))
    real = r1.discounted_demand

    def warped(doc, gamma):
        d = real(doc, gamma)
        rng = random.Random(f"{doc.doc_id}:{gamma}")
        return [[v + rng.random() for v in row] for row in d]

    monkeypatch.setattr(r1, "discounted_demand", warped)
    assert not r1.probe_oracle_invariance(docs, 16)["ok"]
