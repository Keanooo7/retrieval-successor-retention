"""RSR run 1's measurement harness (`experiments/rsr-run1/run.py`, PREREG T3-T8).

🔴 **Nothing here reads the real substrate or `~/rsr-substrate`.** The substrate is a
tiny FIFO checkpoint this file trains itself (the real architecture -- d = 128, M = 16,
S = 48 -- so fresh-stream's C4 instrument, which builds S0-03's model, accepts it; batch
2, 40 steps), rewritten as a legacy checkpoint the way arm B's is (its config only in a
hash-bound `result.json`), and listed in a T0-style manifest of its own. The three arms
run through the real `rsr.train.loop` CLI as subprocesses, a few steps each, through
the held-out harness, the analysis, `RESULTS.md` and the ledger.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch

import rsr.train.loop as loop
from rsr.exit_codes import Exit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "rsr-run1" / "run.py"
SUB_ITERS = 40  # the fixture substrate: measured live at this length
ARM_ITERS = 6  # pool 4 at batch 2 -> T_warm = 2 steps, 4 scored steps
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
    fresh-stream's instrument -- so C4 asks exactly what it asks of arm B: does the
    frozen FIFO arm reproduce its substrate's R?"""
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
    ledger.write_text(
        json.dumps(
            {"seeds_actually_run": [0], "rows": [{"key": r1.C4_KEY, "samples": [ref]}]}
        )
    )
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
    )
    base.update(kw)
    return r1.Run1Config(**base)


def owner(r1, **kw):
    base = dict(gamma=0.9, seeds=(0,), phi_weight_decay=0.01, iters=ARM_ITERS)
    base.update(kw)
    return r1.OwnerValues(**base)


def _ledger_seams(mp, r1) -> None:
    """Test seams on scripts/ledger.py: the worktree under test may hold uncommitted
    edits, and the real gate (`Ledger.write` refusing dirty source) is the run's, not
    this test's."""
    L = r1.ledger_mod()
    mp.setattr(L, "_dirty_source_paths", lambda: [])
    mp.setattr(L, "_exists_at_sha", lambda path, sha: True)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, r1, substrate) -> dict:
    """launch -> measure -> finalize, once, on the fixture; tests read its outputs."""
    out_root = tmp_path_factory.mktemp("e2e") / "runs" / r1.RUN_ID
    cfg = tiny_cfg(r1, substrate)
    ow = owner(r1)
    with pytest.MonkeyPatch.context() as mp:
        _ledger_seams(mp, r1)
        recs = r1.launch(
            [cfg],
            ow,
            out_root,
            launcher="direct",
            threads=1,
            parallel=False,
            allow_dirty=True,
        )
        doc = r1.measure_seed(cfg, ow, out_root, threads=1)
        results = out_root / "RESULTS.md"
        code = r1.finalize([doc], ow, out_root, results_path=results)
    return {
        "out_root": out_root,
        "cfg": cfg,
        "owner": ow,
        "recs": recs,
        "doc": doc,
        "code": code,
        "results": results,
    }


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_e2e_every_arm_ran_live_with_its_own_rc(e2e):
    for a in ("psiC", "fifo", "age"):
        rec = e2e["recs"][f"{a}-seed0"]
        assert rec["rc"] == 0, (
            a,
            rec["rc_meaning"],
            Path(rec["log"]).read_text()[-2000:],
        )
        assert rec["rc_meaning"] == "completed, live"
        on_disk = json.loads((e2e["out_root"] / f"arm-{a}-seed0.json").read_text())
        assert on_disk["rc"] == 0 and on_disk["git_dirty"] is not None
        assert on_disk["run_id"] and on_disk["config_hash"]


def test_e2e_every_control_passes_and_the_run_exits_0(e2e):
    doc = e2e["doc"]
    assert doc["controls_failed"] == [], json.dumps(doc["controls"], default=str)[:4000]
    assert doc["status"] == "ok" and e2e["code"] == Exit.OK


def test_e2e_c4_reproduces_the_substrate_exactly(e2e, substrate):
    c4 = e2e["doc"]["controls"]["C4"]
    assert c4["reference"] == substrate["ref"]
    assert c4["abs_diff"] == 0.0 and c4["ok"] is True


def test_e2e_the_manifest_is_frozen_before_any_arm_starts(e2e):
    man = e2e["out_root"] / "manifest.json"
    first = min(
        json.loads((e2e["out_root"] / f"launch-{a}-seed0.json").read_text())[
            "started_epoch"
        ]
        for a in ("psiC", "fifo", "age")
    )
    assert man.stat().st_mtime <= first
    body = json.loads(man.read_text())
    assert (
        body["owner_values"]["gamma"] == 0.9
        and body["headline"] == "acc.all.psiC_minus_fifo"
    )


def test_e2e_per_document_answer_vectors_are_written(e2e, r1):
    hd = e2e["out_root"] / "heldout" / "seed0"
    for name, _src, _z in r1.heldout_passes():
        recs = r1.read_pass(hd, name)
        assert [r["doc"] for r in recs] == list(range(*EVAL))
        for r in recs:
            assert all(len(a) == 4 for a in r["answers"])
            assert r["real_n"] > 0


def test_e2e_fifo_evicts_age_m_and_shifts_nothing_held_out(e2e):
    ev = e2e["doc"]["heldout"]["heldout_evictions"]["fifo"]
    assert set(ev["victim_age_distribution"]) == {16}
    assert ev["displacement_distribution"] == {0: ev["n_evictions"]}


def test_e2e_results_carry_headline_scope_caveat_and_no_forbidden_word(e2e, r1):
    text = e2e["results"].read_text()
    assert "## 1. Headline" in text and "(HEADLINE)" in text
    assert r1.SCOPE_SENTENCE in text and r1.MULTIPLICITY_SENTENCE in text
    assert (
        r1.prereg_section("  > The shadow buffer is off", "- Every number is reported")
        in text
    )
    assert r1.prereg_section("### 7.6 Author's expectation", "## 8. Controls") in text
    assert ("un" + "biased") not in text.lower()
    assert "not implemented" not in text.lower()  # review MAJOR-1: LRU runs
    assert "partial rho (a diagnostic, never a gate)" in text


def test_e2e_ledger_is_written_with_the_run_meta(e2e, r1):
    led = json.loads((e2e["out_root"] / "ledger.json").read_text())
    assert led["seeds_actually_run"] == [0] and led["status"] == "ok"
    assert led["steps_requested"] == ARM_ITERS and led["steps_done"] == ARM_ITERS
    man = (e2e["out_root"] / "manifest.json").read_bytes()
    assert led["config_hash"] == hashlib.sha256(man).hexdigest()
    keys = {r["key"] for r in led["rows"]}
    assert "headline" in keys and any(k.startswith("seed0.controls.C4") for k in keys)


def test_e2e_headline_is_per_seed_and_across_seeds(e2e):
    run = json.loads((e2e["out_root"] / "run.json").read_text())
    h = run["headline"]
    assert h["contrast"] == "acc.all.psiC_minus_fifo"
    assert h["per_seed"]["0"]["ci_scope"] == "within-seed only"
    assert h["across_seeds"]["n_seeds"] == 1 and h["across_seeds"]["sd"] is None


def test_the_decision_context_choice_does_not_change_held_out_decisions(e2e, r1):
    """In eval mode the loop's own srep IS an eval-mode srep, so the extra eval forward
    (training's L2 path) buys nothing held out: same victims, bitwise (an option to
    halve the RSR arms' cost -- not taken silently; see the hand-back)."""
    cfg = e2e["cfg"]
    out = e2e["out_root"]
    payload = r1.read_checkpoint(
        r1.final_checkpoint(r1.arm_dir(out, "psiC", 0), cfg.iters)
    )
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
    """PREREG §7.1: phi from ``__head__``, ``set_train_step(iters)`` -- so the scored
    rule decides (attribution "psi"), never the FIFO warmup."""
    cfg = e2e["cfg"]
    for arm, attribution in (("psiC", "psi"), ("age", "age_only")):
        payload = r1.read_checkpoint(
            r1.final_checkpoint(r1.arm_dir(e2e["out_root"], arm, 0), cfg.iters)
        )
        pol = r1.eval_policy(arm, payload, cfg)
        head = r1.head_state(payload)
        assert all(torch.equal(v, head[k]) for k, v in pol.head.state_dict().items())
        model = r1.model_from_payload(payload)
        docs = r1.docs_by_id(cfg, range(*EVAL))[:2]
        r1.run_batch(model, pol, docs, r1.vocab_map(cfg), cfg)
        assert pol.records and {r.attribution for r in pol.records} == {attribution}
        assert {r.warm for r in pol.records} == {False}


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


# --------------------------------------------------------------------------- #
# Controls go red when their invariant is broken
# --------------------------------------------------------------------------- #


def _beats(e2e, r1) -> dict:
    return {
        a: r1.read_heartbeat(r1.arm_dir(e2e["out_root"], a, 0))["beats"]
        for a in ("psiC", "fifo", "age")
    }


def test_c2_is_green_on_the_real_arms(e2e, r1):
    c2 = r1.c2_pairing(_beats(e2e, r1), e2e["cfg"])
    assert c2["ok"] and c2["replay_checked"] and c2["steps_compared"] == ARM_ITERS


def test_c2_sees_a_batch_mismatch_in_one_arm(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    beats["fifo"][4]["n_answer_targets"] += 1  # a scored step: loss_lm cannot see it
    c2 = r1.c2_pairing(beats, e2e["cfg"])
    assert not c2["ok"] and c2["cross_arm_mismatch_steps"] == [4]


def test_c2_sees_a_batch_every_arm_shares_but_the_sampler_did_not_draw(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    for a in beats:
        beats[a][5]["pad_target_fraction"] = 0.5
    c2 = r1.c2_pairing(beats, e2e["cfg"])
    assert not c2["ok"] and c2["replay_mismatch_steps"] == [5]
    assert c2["cross_arm_mismatch_steps"] == []


def test_c2_sees_a_warm_step_loss_that_is_not_bitwise_equal(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    v = beats["age"][1]["loss_lm"]
    beats["age"][1]["loss_lm"] = math.nextafter(v, math.inf)
    c2 = r1.c2_pairing(beats, e2e["cfg"])
    assert not c2["ok"] and c2["loss_lm_warm_mismatch_steps"] == [1]


def test_c2_does_not_read_n_targets_scored(r1):
    """review MINOR-4: it is `batch x S x (L-1)` every step, so it cannot see a batch."""
    a = {"n_targets_scored": 6048.0, "n_answer_targets": 20.0, "pad_target_fraction": 0.9}
    # Same n_targets_scored (as every step has), different documents: told apart.
    for other in (
        dict(a, n_answer_targets=21.0),
        dict(a, pad_target_fraction=5400 / 6048),
    ):
        assert other["n_targets_scored"] == a["n_targets_scored"]
        assert r1.batch_fingerprint(a, None) != r1.batch_fingerprint(other, None)
    # A logged batch hash, when the fixed head writes one, wins over the counts.
    h = dict(a, batch_ids_sha256="ab")
    assert r1.batch_fingerprint(h, None) == ("batch_ids_sha256", '"ab"')


def test_c3_sees_one_moved_weight(e2e, r1, substrate):
    out, cfg = e2e["out_root"], e2e["cfg"]
    payloads = {
        a: r1.read_checkpoint(r1.final_checkpoint(r1.arm_dir(out, a, 0), cfg.iters))
        for a in ("psiC", "fifo", "age")
    }
    assert r1.c3_frozen(payloads, substrate["model"])["ok"]
    k = next(iter(payloads["age"]["model"]))
    payloads["age"]["model"][k] = payloads["age"]["model"][k] + 1e-7
    c3 = r1.c3_frozen(payloads, substrate["model"])
    assert not c3["ok"] and c3["per_arm"]["age"]["differ"] == [k]


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


def test_c1_sees_a_labelled_allow_flag_or_another_checkpoint(e2e, r1):
    res = {
        a: json.loads((e2e["out_root"] / f"arm-{a}-seed0.json").read_text())["result"]
        for a in ("psiC", "fifo")
    }
    assert r1.c1_warm_start(res, e2e["cfg"])["ok"]
    bad = copy.deepcopy(res)
    bad["fifo"]["config"]["warm_start"]["allow_seed_mismatch"] = True
    assert not r1.c1_warm_start(bad, e2e["cfg"])["ok"]
    other = tiny_cfg(r1, _sub_like(e2e), sha256="f" * 64)
    assert not r1.c1_warm_start(res, other)["ok"]


def _sub_like(e2e) -> dict:
    s = e2e["cfg"].substrate
    return {
        "ckpt": s.checkpoint,
        "sha": s.sha256,
        "manifest": s.manifest,
        "root": s.root,
        "result": s.source_config,
        "ledger": e2e["cfg"].c4_ledger,
    }


def test_c4_goes_red_on_a_reference_it_does_not_reproduce(r1, e2e, monkeypatch, tmp_path):
    led = tmp_path / "ref.json"
    led.write_text(
        json.dumps(
            {"seeds_actually_run": [0], "rows": [{"key": r1.C4_KEY, "samples": [0.5]}]}
        )
    )

    class Fake:
        @staticmethod
        def measure_checkpoint(seed_dir, seed, label, n):
            assert (label, n) == (ARM_ITERS, 64)
            return {"s003": "m"}

        @staticmethod
        def r_quantity(m):
            return 0.5 + 2e-6

    monkeypatch.setattr(r1, "FS", lambda: Fake)
    cfg = tiny_cfg(r1, _sub_like(e2e), c4_ledger=str(led))
    assert r1.c4_measure(Path("x"), cfg)["ok"] is False
    monkeypatch.setattr(Fake, "r_quantity", staticmethod(lambda m: 0.5 + 5e-7))
    assert r1.c4_measure(Path("x"), cfg)["ok"] is True


def test_c4_refuses_a_ledger_row_that_is_not_the_prereg_value(r1, e2e):
    cfg = tiny_cfg(r1, _sub_like(e2e), c4_prereg_value=0.123)
    with pytest.raises(r1.ControlFailure, match="not the PREREG"):
        r1.c4_reference(cfg)


def test_c5_c6_c8_go_red(r1, e2e):
    res = {
        a: json.loads((e2e["out_root"] / f"arm-{a}-seed0.json").read_text())["result"]
        for a in ("psiC", "fifo", "age")
    }
    hbs = {a: r1.read_heartbeat(r1.arm_dir(e2e["out_root"], a, 0)) for a in res}
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


def test_the_training_analysis_reads_loss_lm_never_loss(e2e, r1):
    hbs = {
        a: r1.read_heartbeat(r1.arm_dir(e2e["out_root"], a, 0)) for a in ("psiC", "fifo")
    }
    diags = {"psiC": r1.read_step_diagnostics(r1.arm_dir(e2e["out_root"], "psiC", 0))}
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


def test_the_diagnostics_split_warm_from_scored(e2e, r1):
    dg = e2e["doc"]["training"]["diagnostics"]["psiC"]
    tw = int(e2e["cfg"].t_warm_steps)
    assert dg["warm"]["n_steps"] == tw and dg["scored"]["n_steps"] == ARM_ITERS - tw
    assert dg["warm"]["warm_flag_matches_step"] and dg["scored"]["warm_flag_matches_step"]
    assert set(dg["warm"]["victim_age_distribution"]) == {16}  # warmup is FIFO


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("drop", ["--gamma", "--seeds", "--phi-weight-decay", "--iters"])
def test_the_cli_refuses_without_each_owner_value(r1, drop):
    args = {
        "--gamma": ["0.9"],
        "--seeds": ["0"],
        "--phi-weight-decay": ["0.01"],
        "--iters": ["3000"],
    }
    argv = ["launch", "--threads", "1", "--slot-wait", "60"]
    for k, v in args.items():
        if k != drop:
            argv += [k, *v]
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
    ):
        with pytest.raises(r1.Refused):
            r1.validate_owner_values(owner(r1, **bad))


def test_a_seed_with_no_registered_substrate_is_refused(r1):
    assert {0, 1, 2} <= set(r1.SUBSTRATE)
    with pytest.raises(r1.Refused):
        r1.prereg_config(3, 3000)


def test_the_registered_shas_are_the_t0_manifest_lines(r1):
    """Seeds 1 and 2's sha256 match the PREREG's truncated T0 quotes (no file read)."""
    assert r1.SUBSTRATE[0]["sha256"].startswith("0ee3f8a6")
    assert r1.SUBSTRATE[1]["sha256"].startswith("dadd1e08")
    assert r1.SUBSTRATE[1]["sha256"].endswith("3da8")
    assert r1.SUBSTRATE[2]["sha256"].startswith("b507ebc5")
    assert r1.SUBSTRATE[2]["sha256"].endswith("63b8")


def test_a_started_arm_is_refused_not_resumed(r1, substrate, tmp_path, monkeypatch):
    out_root = tmp_path / "runs" / r1.RUN_ID
    started = r1.arm_dir(out_root, "fifo", 0)
    started.mkdir(parents=True)
    (started / "heartbeat.jsonl").write_text("{}\n")
    monkeypatch.setattr(r1, "launch_arm", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(r1.Refused, match="never resumed"):
        r1.launch(
            [tiny_cfg(r1, substrate)],
            owner(r1),
            out_root,
            launcher="direct",
            threads=1,
            parallel=False,
            allow_dirty=True,
        )
    assert not (out_root / "manifest.json").exists()


def test_a_dirty_tree_is_refused(r1, substrate, tmp_path, monkeypatch):
    monkeypatch.setattr(
        r1, "tree_state", lambda: {"sha": "abc", "dirty": True, "porcelain": " M x"}
    )
    monkeypatch.setattr(r1, "launch_arm", lambda *a, **k: pytest.fail("launched"))
    with pytest.raises(r1.Refused, match="clean committed head"):
        r1.launch(
            [tiny_cfg(r1, substrate)],
            owner(r1),
            tmp_path / "runs" / r1.RUN_ID,
            launcher="slot",
            threads=1,
            parallel=False,
        )


def test_the_slot_launch_pins_the_sha_and_waits(r1, substrate, tmp_path):
    pre = r1.slot_prefix(1, pinned_sha="abc123", wait_s=600, job_id="j")
    assert pre[pre.index("--pinned-sha") + 1] == "abc123"
    assert float(pre[pre.index("--wait") + 1]) == 600.0
    assert pre[pre.index("--lane") + 1] == "cpu-det" and pre[-1] == "--"


def test_a_wrong_substrate_sha_is_rc_1_with_no_heartbeat_and_measure_says_so(
    r1, substrate, tmp_path
):
    """review MAJOR-2, measured: `WarmStartRefused` escapes `main()` -> rc 1, and the
    loop never opened a heartbeat. The record says what rc 1 meant here."""
    out_root = tmp_path / "runs" / r1.RUN_ID
    cfg = tiny_cfg(r1, substrate, sha256="0" * 64)
    with pytest.MonkeyPatch.context() as mp:
        _ledger_seams(mp, r1)
        recs = r1.launch(
            [cfg],
            owner(r1),
            out_root,
            arms=("fifo",),
            launcher="direct",
            threads=1,
            parallel=False,
            allow_dirty=True,
        )
    rec = recs["fifo-seed0"]
    assert rec["rc"] == 1 and rec["heartbeat_exists"] is False
    assert rec["rc_meaning"].startswith("refused before the loop")
    assert "WarmStartRefused" in Path(rec["log"]).read_text()
    doc = r1.measure_seed(cfg, owner(r1), out_root, threads=1)
    assert doc["status"] == "crashed" and doc["exit"] == int(Exit.DID_NOT_RUN)
    with pytest.raises(r1.ControlFailure):
        r1.load_substrate_model(cfg)


def test_time_batch_never_runs_an_eval_document(r1, substrate):
    with pytest.raises(r1.Refused, match="EVAL"):
        r1.time_one_batch(tiny_cfg(r1, substrate), [EVAL[0]])


def test_classify_rc(r1):
    no_hb = {"exists": False, "footer": None, "crash": None}
    crashed = {"exists": True, "footer": None, "crash": {"status": "crashed"}}
    done = {"exists": True, "footer": {"status": "completed"}, "crash": None}
    res = {"liveness": {"band": "inconclusive"}}
    assert r1.classify_rc(1, None, no_hb).startswith("refused before the loop")
    assert r1.classify_rc(3, None, no_hb).startswith("slot refused")
    assert r1.classify_rc(1, None, crashed).startswith("crashed after start")
    assert r1.classify_rc(1, res, done) == "completed, liveness inconclusive"
    assert r1.classify_rc(5, res, done) == "completed, INERT"


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
    per = {
        a: torch.stack(
            [n, torch.minimum(n, torch.randint(0, 9, (D,), generator=g).double())], 1
        )
        for a in ("x", "y")
    }
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
    reps = []
    for row in idx:
        ra = sum(per["a"][i][1].item() for i in row) / sum(
            per["a"][i][0].item() for i in row
        )
        rb = sum(per["b"][i][1].item() for i in row) / sum(
            per["b"][i][0].item() for i in row
        )
        reps.append(ra - rb)

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
# T7
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
