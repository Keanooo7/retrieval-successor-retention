"""RSR run 1's measurement harness (`experiments/rsr-run1/run.py`; PREREG T3, A1-A3c).

🔴 **Nothing here reads the real substrate or `~/rsr-substrate`.** The substrate is a
tiny FIFO checkpoint this file trains itself (the real architecture -- d = 128, M = 16,
S = 48 -- so fresh-stream's C4 instrument, which builds S0-03's model, accepts it),
rewritten as a legacy checkpoint the way arm B's is (its config only in a hash-bound
`result.json`), and listed in a T0-style manifest of its own. The three arms run
through the real `rsr.train.loop` CLI (launcher v5's command line, without the slot,
O5 = eval, C-2 permutation epochs) into the launcher's layout, then through the held-out
harness, the analysis, `RESULTS.md` and the ledger. ``prereg_strict=False`` waives only
the slot job records and the clean-tree launch (there are none in a test), and points
C10's owner-table read at a synthetic PREREG holding the fixture's values.

🔴 **No EVAL document goes through any model here** (review 6267fd645698 MINOR-7): the
fixture's held-out range is 998000-998001, outside the run's EVAL (995000-997047) and
every registered range (C0 checks it).

**Cost** (the battery runs this file once per mutation). Everything heavy is built once
per session and shared; every size is the smallest that still exercises each control:
a 3-step substrate, 3-step arms on a 2-document pool at batch 2 (T_warm = 1, CKPT = 1,
snapshots at 1 and 2), 2 held-out documents. Two costs are floors, both inside the code
under test: each `train()` measures its own liveness (~6.8 s, R-2026-09-22), and
fresh-stream's C4 instrument (`measure_checkpoint`, ~23 s) is run once -- see
`_c4_memo`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import shlex
import shutil
import sys
from pathlib import Path

import pytest
import torch

import rsr.train.loop as loop
from rsr.exit_codes import Exit

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "rsr-run1" / "run.py"
SUB_ITERS = 3  # the fixture substrate, at SUB_LR: its frozen arms measure `live`
SUB_LR = 1e-2
ARM_ITERS = 3  # pool 2 at batch 2 -> T_warm = 1 step; CKPT = 1 -> snapshots 1, 2, 3
POOL = (993000, 993002)
EVAL = (998000, 998002)  # NOT the run's EVAL: the fixture's held-out range (MINOR-7)
O5 = "eval"  # exercises C-1 end to end; the train branch is rendered separately
DIAG = 2  # A4.2's DIAGNOSTIC step, shrunk with the run (160 of 3000 in the run)


@pytest.fixture(scope="session")
def r1(tmp_path_factory):
    """run.py, loaded once, with its **shared orchestrator root injected**: an empty,
    session-private root, never the live checkout's `.orchestrator/jobs`.

    Root cause of 7 failures (2026-10-07, PREREG rsr-run2 §10 item 7): C10 reads each
    arm's slot job record `rsr-run1-<arm>-s<seed>.json` from `SHARED_ORCH_ROOT`, which is
    the real checkout. Once run 1 had launched, the e2e fixture -- launched directly,
    with no slot and so no job record of its own -- read run 1's REAL records
    (iters 3000, pinned ed6302a) and C10 failed on them. A test must never depend on
    what has been launched on this machine. Injecting the root here leaves run.py, and
    so every real run's output, unchanged; `tiny_cfg` passes the same root."""
    spec = importlib.util.spec_from_file_location("rsr_run1_run", RUN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rsr_run1_run"] = mod
    spec.loader.exec_module(mod)
    iso = tmp_path_factory.mktemp("orch-root")
    (iso / ".orchestrator" / "jobs").mkdir(parents=True)
    mod.SHARED_ORCH_ROOT = str(iso)
    return mod


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _c4_memo(fs):
    """In-process, CONTENT-keyed memo of fresh-stream's `measure_checkpoint` (a test
    seam; never on disk, never across processes). The key is the sha256 of the
    checkpoint's model tensors plus (seed, n), and the instrument's own guard -- stored
    step == label -- is checked on every call with the instrument's own functions. So a
    checkpoint whose weights differ by one bit is measured afresh; only a byte-identical
    model (the frozen FIFO arm vs its substrate, which C3 also asserts) reuses the one
    real measurement. Both calls would run the same (possibly mutated) instrument code
    in this process anyway."""
    real = fs.measure_checkpoint
    cache: dict = {}

    def memo(seed_dir, seed, label, n):
        ckpt = fs.RC.ckpt_path(Path(seed_dir), label)
        step = fs.RC.stored_step(ckpt)
        if step != label:
            return real(seed_dir, seed, label, n)  # let the instrument raise
        model = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        h = hashlib.sha256()
        for k in sorted(model):
            h.update(k.encode())
            h.update(model[k].numpy().tobytes())
        key = (h.hexdigest(), seed, n)
        if key not in cache:
            cache[key] = real(seed_dir, seed, label, n)
        return cache[key]

    return memo


@pytest.fixture(scope="session")
def substrate(tmp_path_factory, r1) -> dict:
    """A miniature arm B: a FIFO checkpoint, legacy (no `meta.config`, no map hash),
    its config in `result.json` bound by the run_id hash, a T0-style manifest, and a
    C4 reference ledger holding the fixture's OWN `R_quantity`, measured with
    fresh-stream's instrument -- so C4 asks what it asks of arm B: does the frozen FIFO
    arm reproduce its substrate's R? The substrate's own liveness is not measured here
    (the arms measure theirs, which is the one C5 reads)."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("substrate")
    run = root / "train"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, "_measure_liveness", lambda *a, **k: {"band": "live"})
        loop.train(
            policy_name="fifo",
            d=128,
            batch=2,
            iters=SUB_ITERS,
            lr=SUB_LR,
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
    mp = pytest.MonkeyPatch()
    mp.setattr(fs, "measure_checkpoint", _c4_memo(fs))
    ref = float(fs.r_quantity(fs.measure_checkpoint(sdir, 0, SUB_ITERS, 64)["s003"]))
    ledger = root / "ref-ledger.json"
    rows = [{"key": r1.C4_KEY, "samples": [ref]}]
    ledger.write_text(json.dumps({"seeds_actually_run": [0], "rows": rows}))
    owner_prereg = root / "PREREG-owner.md"
    owner_prereg.write_text(_owner_table_text(r1, "0.9", "0.01", str(ARM_ITERS), O5))
    yield {
        "root": root,
        "ckpt": ckpt,
        "sha": _sha(ckpt),
        "manifest": manifest,
        "result": result,
        "ledger": ledger,
        "ref": ref,
        "model": payload["model"],
        "owner_prereg": owner_prereg,
    }
    mp.undo()


def _owner_table_text(r1, o1: str, o4: str, r_1: str, o5: str, heading=None) -> str:
    """A synthetic PREREG holding the owner table in the one format C10 accepts."""
    return "\n".join(
        [
            "# a synthetic PREREG",
            "",
            heading or r1.OWNER_TABLE_HEADING,
            "",
            "| slot | value |",
            "|---|---|",
            f"| **O1** gamma, both RSR arms only | `{o1}` |",
            f"| **O4** phi weight decay (decoupled) | `{o4}` |",
            f"| **R-1** total optimizer steps per arm | `{r_1}` |",
            f"| **O5** substrate mode under freeze | `{o5}` |",
            "",
            "- prose after the table",
            "",
        ]
    )


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
        eval_batch=2,
        c4_ledger=str(sub["ledger"]),
        n_boot=200,
        owner_prereg=str(sub["owner_prereg"]),
        diag_step=DIAG,
        prereg_strict=False,
        orch_root=r1.SHARED_ORCH_ROOT,  # the injected, isolated root (see `r1`)
    )
    base.update(kw)
    return r1.Run1Config(**base)


def owner(r1, **kw):
    base = dict(gamma=0.9, phi_weight_decay=0.01, iters=ARM_ITERS, o5=O5, seeds=(0,))
    base.update(kw)
    return r1.OwnerValues(**base)


def _ledger_seams(mp, r1) -> None:
    """Test seams on scripts/ledger.py: the worktree under test may hold uncommitted
    edits, and the real gate (`Ledger.write` refusing dirty source) is the run's."""
    L = r1.ledger_mod()
    mp.setattr(L, "_dirty_source_paths", lambda: [])
    mp.setattr(L, "_exists_at_sha", lambda path, sha: True)


@pytest.fixture(scope="session")
def e2e(tmp_path_factory, r1, substrate) -> dict:
    """launch_direct -> measure_seed -> finalize, once per session, on the fixture."""
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


def test_e2e_launch_record_precedes_every_arm(e2e, r1):
    lg = e2e["out_root"] / "launch-git.json"
    body = json.loads(lg.read_text())
    assert body["o5"] == O5 and body["ckpt"] == "1" and body["steps"] == str(ARM_ITERS)
    assert body["ckpt_rule"] == r1.CKPT_RULE
    assert [body[k] for k in ("rsr_orch_root", "out", "job_suffix", "dry_run")] == [
        ""
    ] * 4
    for a in ("psiC", "fifo", "age"):  # MINOR-8: no st_birthtime (absent on Linux)
        hb = e2e["out_root"] / a / "seed0" / "heartbeat.jsonl"
        last = json.loads(hb.read_text().splitlines()[-1])
        header_written = hb.stat().st_mtime - float(last["elapsed_s"])
        assert lg.stat().st_mtime <= header_written + 0.05


def test_e2e_per_document_answer_vectors_are_written(e2e, r1):
    hd = e2e["out_root"] / "heldout" / "seed0"
    for name, _src, _z in r1.heldout_passes([1, 2]):
        recs = r1.read_pass(hd, name)
        assert [r["doc"] for r in recs] == list(range(*EVAL))
        for r in recs:
            assert all(len(a) == 4 for a in r["answers"])
            assert r["real_n"] > 0


def test_e2e_fifo_evicts_age_m_and_shifts_nothing_held_out(e2e):
    ev = e2e["doc"]["heldout"]["heldout_evictions"]["fifo"]
    assert set(ev["victim_age_distribution"]) == {16}
    assert ev["displacement_distribution"] == {0: ev["n_evictions"]}


def test_e2e_lru_held_out_observes_and_departs_from_fifo(e2e):
    """T11: the held-out LRU pass runs with observe on, so it is not FIFO."""
    ev = e2e["doc"]["heldout"]["heldout_evictions"]
    assert ev["fifo"]["fraction_oldest"] == 1.0
    assert ev["lru"]["fraction_oldest"] < 1.0


def test_e2e_snapshots_are_measured_as_diagnostics(e2e):
    snaps = e2e["doc"]["heldout"]["snapshots"]
    assert set(snaps) == {"psiC@1", "psiC@2", "age@1", "age@2"}
    assert "DIAGNOSTIC snapshot `psiC@1`" in e2e["results"].read_text()


def test_e2e_results_carry_headline_scope_caveat_and_no_forbidden_word(e2e, r1):
    text = e2e["results"].read_text()
    assert "## 1. HEADLINE" in text and "(HEADLINE)" in text
    assert r1.scope_sentence() in text and r1.MULTIPLICITY in text
    assert "The three per-seed values show the **combined** variation" in text
    assert r1.caveat_text() in text and r1.expectation_text() in text
    assert r1.confound_text() not in text  # O5 = eval in the fixture (A2.1)
    assert "TRAIN/EVAL MEMORY SHIFT" not in text
    assert "In-loop `loss_lm` values are eval-mode values" in text
    assert r1.LRU_CAVEAT in text and r1.ATTRIBUTION_NOTE in text
    assert "TEST FIXTURE" in text
    assert ("un" + "biased") not in text.lower()
    assert "not implemented" not in text.lower()  # A1.1: LRU runs
    assert "met or missed" in text and "**met**" not in text


def _render(r1, doc, o5):
    ow = {**doc["owner_values"], "o5": o5}
    run = {
        "status": doc["status"],
        "exit": doc["exit"],
        "owner_values": ow,
        "headline": r1.headline([doc]),
        "tree": {},
        "prereg_strict": False,
    }
    return r1.render_results(run, [doc])


def test_o5_train_names_the_confound_beside_every_psic_number(r1, e2e):
    """A2.1: under O5 = train, the named confound is printed verbatim, and beside the
    headline and every psiC contrast."""
    text = _render(r1, e2e["doc"], "train")
    assert r1.confound_text() in text
    n = 0
    for line in text.splitlines():
        parts = line.split("`")
        name = parts[1] if line.startswith("- ") and len(parts) > 2 else ""
        if "psiC" in name and all(ch.isalnum() or ch in "._@-" for ch in name):
            assert "TRAIN/EVAL MEMORY SHIFT (psiC only)" in line, line
            n += 1
        elif line.startswith("- `") and "TRAIN/EVAL MEMORY SHIFT" in line:
            assert "psiC" in line, line  # never beside a number psiC is not in
    assert n > 10
    assert "In-loop `loss_lm` values are eval-mode values" not in text


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
    docs = r1.docs_by_id(cfg, range(*EVAL))
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
        for step in (1, ARM_ITERS):
            payload = _payload(r1, e2e, arm, step)
            pol = r1.eval_policy(arm, payload, cfg)
            head = r1.head_state(payload)
            assert all(torch.equal(v, head[k]) for k, v in pol.head.state_dict().items())
            model = r1.model_from_payload(payload)
            docs = r1.docs_by_id(cfg, range(*EVAL))
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
    docs = r1.docs_by_id(cfg, range(*EVAL))
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
    docs = r1.docs_by_id(cfg, range(*EVAL))
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
    assert c["ok"] and c["documents"] == [998000, 998002]


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
    text = _render(r1, doc, O5)
    tagged = 0
    for line in text.splitlines():
        if line.startswith(("- `oracle_regret.", "- headroom share")):
            assert "NOT READ (C9/C9b failed)" in line, line
            tagged += 1
        if line.startswith("| all |"):  # A3.6: the oracle's own accuracy row too
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert "NOT READ" in cells[6] and all("NOT READ" not in c for c in cells[:6])
            tagged += 1
        if line.startswith("- `acc.all.psiC_minus_fifo`"):
            assert "NOT READ" not in line
    assert tagged > 1
    assert doc["exit"] == 0  # C9 is not in `controls`: the run's exit is unchanged


# --------------------------------------------------------------------------- #
# Controls go red when their invariant is broken
# --------------------------------------------------------------------------- #


def _beats(e2e, r1) -> dict:
    return {a: h["beats"] for a, h in _hbs(r1, e2e).items()}


def _sampling(r1, e2e):
    return _results(r1, e2e)["psiC"]["config"]["epoch"]["sampling"]


def _ids(r1, e2e) -> dict:
    return {
        a: r1.step_pool_ids(r1.read_step_records(r1.arm_dir(e2e["out_root"], a, 0)))
        for a in r1.ARMS
    }


def test_c2_is_green_on_the_real_arms(e2e, r1):
    c2 = r1.c2_pairing(_beats(e2e, r1), _ids(r1, e2e), e2e["cfg"], _sampling(r1, e2e))
    assert c2["ok"] and c2["steps_compared"] == ARM_ITERS
    assert c2["a_pool_ids"]["present"] is True  # C-2 (c02b16d) logs pool_doc_ids
    assert _sampling(r1, e2e) == loop.SAMPLING_PERMUTATION


def test_every_arm_records_its_sampler_and_c2_requires_them_to_agree(e2e, r1):
    """262b5c5 (review c02b16d MINOR-1): `frozen["sampling"]` in EVERY pool arm, FIFO
    included; C2 requires the three to agree, and an RSR arm's two records to agree."""
    res = _results(r1, e2e)
    got = {a: r1.arm_sampling(r) for a, r in res.items()}
    assert got == dict.fromkeys(r1.ARMS, loop.SAMPLING_PERMUTATION)
    assert e2e["doc"]["controls"]["C2"]["b_fingerprint"]["sampling_agrees"] is True
    args = (_beats(e2e, r1), _ids(r1, e2e), e2e["cfg"], _sampling(r1, e2e))
    assert r1.c2_pairing(*args, samplings=got)["ok"]
    for bad in ({**got, "fifo": None}, {**got, "age": loop.SAMPLING_WITH_REPLACEMENT}):
        assert not r1.c2_pairing(*args, samplings=bad)["ok"]
    split = copy.deepcopy(res["psiC"])
    split["config"]["epoch"]["sampling"] = loop.SAMPLING_WITH_REPLACEMENT
    assert r1.arm_sampling(split).startswith("DISAGREE")


def test_c2_requires_the_logged_pool_ids(e2e, r1):
    c2 = r1.c2_pairing(_beats(e2e, r1), {}, e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["a_pool_ids"]["required"]


def test_c2_sees_a_batch_mismatch_in_one_arm(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    beats["fifo"][2]["n_answer_targets"] += 1  # a scored step: loss_lm cannot see it
    c2 = r1.c2_pairing(beats, _ids(r1, e2e), e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["b_fingerprint"]["cross_arm_mismatch_steps"] == [2]


def test_c2_sees_a_batch_every_arm_shares_but_the_sampler_did_not_draw(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    for a in beats:
        beats[a][2]["pad_target_fraction"] = 0.5
    c2 = r1.c2_pairing(beats, _ids(r1, e2e), e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["b_fingerprint"]["replay_mismatch_steps"] == [2]
    assert c2["b_fingerprint"]["cross_arm_mismatch_steps"] == []


def test_c2_sees_a_warm_step_loss_that_is_not_bitwise_equal(e2e, r1):
    beats = copy.deepcopy(_beats(e2e, r1))
    v = beats["age"][0]["loss_lm"]
    beats["age"][0]["loss_lm"] = math.nextafter(v, math.inf)
    c2 = r1.c2_pairing(beats, _ids(r1, e2e), e2e["cfg"], _sampling(r1, e2e))
    assert not c2["ok"] and c2["c_loss_lm_warm_mismatch_steps"] == [0]


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
        dup[a][0] = [dup[a][0][0]] * 2  # epoch 0 reads one document twice
    c2 = r1.c2_pairing(beats, dup, cfg, s)
    assert not c2["ok"] and 0 in c2["a_pool_ids"]["epochs_not_exactly_the_pool"]


def test_the_permutation_replay_visits_every_pool_document_once_per_epoch(r1, substrate):
    cfg = tiny_cfg(r1, substrate)
    b = r1.replay_batches(cfg, 6, "per-epoch permutation")
    spe = int(cfg.steps_per_epoch)
    for e in range(3):
        assert sorted(i for k in range(e * spe, (e + 1) * spe) for i in b[k]) == [0, 1]
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
        r1.check_c0(tiny_cfg(r1, substrate, pool=(4100, 4102)))
    with pytest.raises(r1.ControlFailure, match="eval_below_K"):
        r1.check_c0(tiny_cfg(r1, substrate, eval_range=(1_000_000, 1_000_008)))
    with pytest.raises(r1.ControlFailure, match="pool_eval_disjoint"):
        r1.check_c0(tiny_cfg(r1, substrate, eval_range=(993001, 993003)))
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
    beats["psiC"][2]["loss_mc"] = float("nan")
    beats["age"][1]["phi_grad_norm"] = None
    c8 = r1.c8_finite(beats)
    assert c8["per_arm"]["psiC"]["nonfinite_steps"] == [2]
    assert c8["per_arm"]["age"]["nonfinite_steps"] == [1]


def _c10(
    r1,
    e2e,
    *,
    cfg=None,
    owner_=None,
    launch=None,
    results=None,
    hbs=None,
    jobs=None,
    rcs=None,
    registered=None,
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
        rcs=rcs if rcs is not None else dict.fromkeys(r1.ARMS, 0),
        registered=registered,
    )


def _problems(c: dict) -> list[str]:
    return c["launch_problems"] + [
        p for v in c["per_arm"].values() for p in v["problems"]
    ]


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
        ("o5", "train"),
        ("steps", "9"),
        ("ckpt_rule", "STEPS"),
        ("rsr_orch_root", "/tmp/iso"),
        ("out", "/tmp/dry"),
        ("job_suffix", "-dry"),
        ("dry_run", "1"),
    ):
        assert not _c10(r1, e2e, launch={**lg, key: val})["ok"], key
    res = _results(r1, e2e)
    for arm, path, val in (
        ("fifo", ("frozen_model_mode",), "train (dropout live)"),
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
    """The arms wrote checkpoints at 1, 2, 3; a ruled R-1 of 9 derives CKPT 3."""
    ow = owner(r1, iters=9)
    cfg = r1.dataclasses.replace(e2e["cfg"], iters=9)
    c = _c10(r1, e2e, cfg=cfg, owner_=ow)
    assert not c["ok"]
    assert any("cadence" in p for v in c["per_arm"].values() for p in v["problems"])


def test_c10_sees_disagreeing_provenance(r1, e2e):
    hbs = copy.deepcopy(_hbs(r1, e2e))
    hbs["age"]["header"]["provenance"]["git_sha"] = "0" * 40
    assert not _c10(r1, e2e, hbs=hbs)["ok"]


def _strict(r1, e2e):
    """The fixture read as the run reads it: strict, owner values from the PREREG (its
    table handed in filled, as Amendment 4 will fill it), slot job records whose `cmd`
    is the command that ran, one thread, the launcher's rc."""
    cfg = r1.dataclasses.replace(
        e2e["cfg"], prereg_strict=True, owner_prereg=r1.PREREG, diag_step=r1.DIAG_STEP
    )
    lg = json.loads((e2e["out_root"] / "launch-git.json").read_text())
    clean = {**lg, "dirty": False, "diag_step": str(r1.DIAG_STEP)}  # as v6 records
    jobs = {
        a: {
            "pinned_sha": lg["run_sha"],
            "threads": 1,
            "rc": 0,
            "status": "done",
            "cmd": r1.cli_argv(
                e2e["cfg"], a, e2e["owner"], r1.arm_dir(e2e["out_root"], a, 0)
            ),
        }
        for a in r1.ARMS
    }
    reg = r1.parse_owner_table(_owner_table_text(r1, "0.9", "0.01", str(ARM_ITERS), O5))
    return cfg, clean, jobs, reg


def test_c10_strict_wants_each_job_record_pinned_to_the_run_sha(r1, e2e):
    cfg, clean, jobs, reg = _strict(r1, e2e)

    def c10(**kw):
        return _c10(r1, e2e, **{"cfg": cfg, "launch": clean, "registered": reg, **kw})

    assert not c10()["ok"]  # no job records
    assert c10(jobs=jobs)["ok"], _problems(c10(jobs=jobs))
    for arm, key, val, why in (
        ("age", "pinned_sha", "f" * 40, "pinned_sha"),
        ("fifo", "threads", 8, "threads"),  # MINOR-5
        ("psiC", "threads", None, "threads"),
        ("psiC", "rc", 1, "job rc"),  # MINOR-5: the record's rc is the launcher's
        ("age", "rc", None, "job rc"),
    ):
        bad = {**jobs, arm: {**jobs[arm], key: val}}
        c = c10(jobs=bad)
        assert not c["ok"] and any(why in p for p in c["per_arm"][arm]["problems"]), key
    c = c10(jobs=jobs, rcs={**dict.fromkeys(r1.ARMS, 0), "fifo": 143})
    assert not c["ok"] and any("job rc" in p for p in c["per_arm"]["fifo"]["problems"])
    assert not c10(jobs=jobs, launch={**clean, "dirty": True})["ok"]
    small = r1.dataclasses.replace(cfg, diag_step=DIAG)  # the run reads A4.2's 160 only
    c = c10(jobs=jobs, cfg=small, launch={**clean, "diag_step": str(DIAG)})
    assert not c["ok"] and any("A4.2's 160" in p for p in c["launch_problems"])
    wrong = r1.dataclasses.replace(cfg, owner_prereg=e2e["cfg"].owner_prereg)
    assert not _c10(r1, e2e, cfg=wrong, launch=clean, registered=reg, jobs=jobs)["ok"]


def _cmd_set(cmd: list[str], flag: str, val: str | None) -> list[str]:
    cmd = list(cmd)
    i = cmd.index(flag)
    return (
        [*cmd[:i], *cmd[i + 2 :]] if val is None else [*cmd[: i + 1], val, *cmd[i + 2 :]]
    )


def test_c10_reads_every_value_from_the_jobs_command(r1, e2e):
    """MAJOR-2 / MINOR-5 / A3c.5: the command the job ran carries the owner values, the
    seed and srep 0.0; a tampered token reddens C10 even where the config agrees."""
    cfg, clean, jobs, reg = _strict(r1, e2e)
    cases = [
        ("psiC", "--seed", "1", "--seed"),
        ("fifo", "--iters", "9", "--iters"),
        ("age", "--ckpt-every", "3", "--ckpt-every"),
        ("psiC", "--srep-norm-reg-weight", "0.01", "srep"),
        ("fifo", "--srep-norm-reg-weight", None, "srep"),  # the token is missing
        ("age", "--frozen-model-mode", "train", "frozen-model-mode"),
    ]
    for arm, flag, val, why in cases:
        bad = {**jobs, arm: {**jobs[arm], "cmd": _cmd_set(jobs[arm]["cmd"], flag, val)}}
        c = _c10(r1, e2e, cfg=cfg, launch=clean, registered=reg, jobs=bad)
        assert not c["ok"] and any(why in p for p in c["per_arm"][arm]["problems"]), (
            flag,
            val,
        )
    g = [x.replace("gamma=0.9", "gamma=0.5") for x in jobs["psiC"]["cmd"]]
    c = _c10(
        r1,
        e2e,
        cfg=cfg,
        launch=clean,
        registered=reg,
        jobs={**jobs, "psiC": {**jobs["psiC"], "cmd": g}},
    )
    assert not c["ok"] and any("gamma" in p for p in c["per_arm"]["psiC"]["problems"])
    extra = jobs["fifo"]["cmd"] + ["--policy-override", "gamma=0.9"]
    c = _c10(
        r1,
        e2e,
        cfg=cfg,
        launch=clean,
        registered=reg,
        jobs={**jobs, "fifo": {**jobs["fifo"], "cmd": extra}},
    )
    assert not c["ok"]  # fifo carries no gamma


def test_c10_requires_srep_0_in_every_recorded_config(r1, e2e):
    """A3c.5: the recorded `srep_norm_reg_weight` is 0.0 in every arm (the e2e ran
    launcher v5's command, so it records 0.0, not TGConfig's 0.01)."""
    res = _results(r1, e2e)
    assert {
        a: r["config"]["srep_norm_reg_weight"] for a, r in res.items()
    } == dict.fromkeys(r1.ARMS, 0.0)
    for val in (0.01, None, "0.0", True):
        bad = copy.deepcopy(res)
        bad["age"]["config"]["srep_norm_reg_weight"] = val
        c = _c10(r1, e2e, results=bad)
        assert not c["ok"] and any(
            "srep" in p for p in c["per_arm"]["age"]["problems"]
        ), val


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
        h["beats"][2]["loss_lm"] += 1.0
    assert r1.analyse_training(poisoned, diags, e2e["cfg"]) != a


def test_run_level_histograms_are_cross_checked_against_the_per_step_sums(r1, e2e):
    tr = e2e["doc"]["training"]
    assert tr["run_level_histograms_agree"] == {"psiC": True, "age": True}
    per = {
        p: {"displacement_distribution": {int(k): v for k, v in d.items()}}
        for p, d in (
            ("warm", tr["diagnostics"]["psiC"]["warm"]["displacement_distribution"]),
            ("scored", tr["diagnostics"]["psiC"]["scored"]["displacement_distribution"]),
        )
    }
    good = {
        "fifo_warmup": {"shift_distribution": per["warm"]["displacement_distribution"]},
        "scored": {"shift_distribution": per["scored"]["displacement_distribution"]},
    }
    assert r1.run_level_agrees(good, per) is True
    bad = copy.deepcopy(good)
    bad["scored"]["shift_distribution"] = {0: 1}
    assert r1.run_level_agrees(bad, per) is False
    assert r1.run_level_agrees(None, per) is None


def test_the_diagnostics_split_warm_from_scored_and_report_the_first_block(e2e, r1):
    dg = e2e["doc"]["training"]["diagnostics"]["psiC"]
    tw = int(e2e["cfg"].t_warm_steps)
    assert dg["warm"]["n_steps"] == tw and dg["scored"]["n_steps"] == ARM_ITERS - tw
    assert dg["warm"]["warm_flag_matches_step"] and dg["scored"]["warm_flag_matches_step"]
    assert set(dg["warm"]["victim_age_distribution"]) == {16}  # warmup is FIFO
    fb = dg["first_scored_block"]
    assert fb["n_evictions"] == dg["scored"]["n_evictions"]  # 2 scored steps < 500
    assert sum(fb["uniform_random_expected"].values()) == pytest.approx(fb["n_evictions"])


# --------------------------------------------------------------------------- #
# Launcher, preflight, refusals
# --------------------------------------------------------------------------- #


def test_the_committed_launcher_is_byte_identical_to_the_prereg(r1):
    """T14 / T19: launcher v6, extracted from "### A4.3 Launcher, version 6"."""
    assert r1.LAUNCHER_HEADING == "### A4.3 Launcher, version 6" and r1.LAUNCHER_V6
    text = (ROOT / r1.LAUNCHER).read_text()
    assert text == r1.launcher_text_from_prereg()
    assert "        --srep-norm-reg-weight 0.0 \\\n" in text  # A3c.1
    assert "--ckpt-every $EVERY --device cpu" in text and "DIAG_STEP=160\n" in text
    assert f'ckpt_every_rule "{r1.EVERY_RULE}"' in text  # recorded verbatim (A4.3)
    want = "cb0bef7f7d9efdd8e3b0a0286d385116aec19a19e8353f5e84d1b6ed4f823f0f"
    assert hashlib.sha256(text.encode()).hexdigest() == want


def _launcher_command(
    r1, arm: str, seed: int, o5: str, gamma: str, out: str
) -> list[str]:
    """Launcher v5's command for (arm, seed), variables filled by text substitution."""
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
        ("$EVERY", "40"),  # launcher v6 (A4.2): gcd(160, 1000)
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
    ow = r1.dataclasses.replace(ow, o5=o5)
    for seed in (0, 1, 2):
        cfg = r1.prereg_config(seed, 3000)
        for arm in r1.ARMS:
            out = f"runs/rsr-run1/{arm}/seed{seed}"
            got = r1.cli_argv(cfg, arm, ow, Path(out))
            assert got == _launcher_command(r1, arm, seed, o5, "0.9", out), (arm, seed)


def test_derive_ckpt(r1):
    assert r1.derive_ckpt(3000) == 1000 and r1.derive_ckpt(6) == 2
    assert r1.derive_ckpt(3001) == 3001 and r1.derive_ckpt(1) == 1
    assert r1.derive_ckpt(0) == 0  # launcher v3: STEPS > 0 and STEPS % 3 == 0


def test_env_problems_refuse_a_foreign_orch_root_and_dry_run_hooks(r1):
    assert (
        r1.env_problems(
            {"RSR_ORCH_ROOT": None, "OUT": None, "JOB_SUFFIX": None}, dry_run=False
        )
        == []
    )
    assert r1.env_problems({"RSR_ORCH_ROOT": r1.SHARED_ORCH_ROOT}, dry_run=False) == []
    assert r1.env_problems({"RSR_ORCH_ROOT": "/tmp/iso"}, dry_run=False)
    # MINOR-4: a dry run (STEPS 0) may use its isolated root; C10 fails it afterwards
    assert r1.env_problems({"RSR_ORCH_ROOT": "/tmp/iso"}, dry_run=True) == []
    assert r1.env_problems({"OUT": "x", "JOB_SUFFIX": "-d"}, dry_run=False)
    assert r1.env_problems({"OUT": "x", "JOB_SUFFIX": "-d"}, dry_run=True) == []


def test_measure_refuses_a_foreign_orch_root_when_strict(r1, substrate, tmp_path):
    cfg = tiny_cfg(r1, substrate, prereg_strict=True)
    doc = r1.measure_seed(
        cfg, owner(r1), tmp_path, threads=1, environ={"RSR_ORCH_ROOT": "/tmp/iso"}
    )
    assert doc["status"] == "did_not_run" and doc["exit"] == int(Exit.DID_NOT_RUN)
    assert any("environment" in n for n in doc["notes"])
    dry = r1.measure_seed(
        cfg, owner(r1), tmp_path / "d", threads=1, environ={"OUT": "x"}, dry_run=True
    )  # A3.12: --dry-run lets OUT through, and then nothing was launched there
    assert dry["status"] == "did_not_run" and not any(
        "environment" in n for n in dry["notes"]
    )


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
    # MINOR-4: STEPS 0 is admitted for a dry run, and only STEPS 0 (A3c.1)
    r1.validate_owner_values(owner(r1, iters=0), dry_run=True)
    with pytest.raises(r1.Refused, match="never trains"):
        r1.validate_owner_values(owner(r1), dry_run=True)
    with pytest.raises(r1.Refused, match="O5 must be one of"):
        r1.validate_owner_values(owner(r1, o5="maybe"))  # whatever C-1 says
    assert not r1.frozen_mode_supported("maybe")
    assert r1.frozen_mode_supported("eval")  # C-1 is in the code (c02b16d)
    assert r1.FROZEN_MODE_RECORDED == {
        "train": "train (dropout live)",
        "eval": "eval (dropout off)",
    }  # A3.7a's two strings
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(r1, "ARMS", ("fifo",))  # one refused process proves it
        rcs = r1.launch_direct([cfg], owner(r1), out_root, threads=1, allow_dirty=True)
        assert rcs == {"fifo-s0": 1}
        assert "WarmStartRefused" in (out_root / "logs" / "fifo-s0.log").read_text()
        doc = r1.measure_seed(cfg, owner(r1), out_root, threads=1)
    assert doc["status"] == "did_not_run" and doc["exit"] == int(Exit.DID_NOT_RUN)
    assert any(n.startswith("ARM_DID_NOT_RUN") for n in doc["notes"])  # A3b.3
    assert doc["arms"]["fifo"]["rc_meaning"].startswith("did not run: exception before")
    with pytest.raises(r1.ControlFailure):
        r1.load_substrate_model(cfg)


def _copy_run(e2e, dst: Path) -> Path:
    out = dst / "runs" / "rsr-run1"
    shutil.copytree(e2e["out_root"], out)
    return out


def test_ledger_statuses_follow_a3b_3(r1, e2e, tmp_path):
    """T15, A3b.3: inert -> partial / 5 / INERT; inconclusive liveness -> failed / 1 /
    LIVENESS_INCONCLUSIVE; an arm that crashed mid-loop -> crashed / 3 / ARM_CRASHED; a
    control that fails after training -> did_not_run / 3 / CONTROL_FAILED, with
    steps_done and seeds_actually_run kept; a refusal -> did_not_run / 3."""

    def run(name, mutate):
        out = _copy_run(e2e, tmp_path / name)
        mutate(out)
        return r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)

    doc = run("inert", lambda o: (o / "logs" / "age-s0.rc").write_text("5\n"))
    assert (doc["status"], doc["exit"]) == ("partial", int(Exit.INERT))
    assert "INERT: age" in doc["notes"]
    # rc 1 with a completed footer: the band was inconclusive
    doc = run("unsure", lambda o: (o / "logs" / "fifo-s0.rc").write_text("1\n"))
    assert (doc["status"], doc["exit"]) == ("failed", int(Exit.FAIL))
    assert "LIVENESS_INCONCLUSIVE: fifo" in doc["notes"]

    def two(o):  # the worst arm wins (A3b.3 + A3b.5), not the first non-OK one
        (o / "logs" / "fifo-s0.rc").write_text("1\n")
        (o / "logs" / "age-s0.rc").write_text("5\n")

    doc = run("two", two)
    assert (doc["status"], doc["exit"]) == ("partial", int(Exit.INERT))
    assert {"LIVENESS_INCONCLUSIVE: fifo", "INERT: age"} <= set(doc["notes"])

    def crash(o):
        hb = o / "psiC" / "seed0" / "heartbeat.jsonl"
        rows = [x for x in hb.read_text().splitlines() if '"kind": "footer"' not in x]
        hb.write_text("\n".join([*rows, json.dumps({"kind": "crash"})]) + "\n")
        (o / "logs" / "psiC-s0.rc").write_text("1\n")

    doc = run("crash", crash)
    assert (doc["status"], doc["exit"]) == ("crashed", int(Exit.DID_NOT_RUN))
    assert "ARM_CRASHED: psiC" in doc["notes"]

    def late(o):
        lg = json.loads((o / "launch-git.json").read_text())
        (o / "launch-git.json").write_text(json.dumps({**lg, "wd": "0.0"}))

    doc = run("late", late)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert doc["controls_failed"] == ["C10"] and "CONTROL_FAILED: ['C10']" in doc["notes"]
    assert doc["steps_done"] == dict.fromkeys(r1.ARMS, ARM_ITERS)
    assert doc["seeds_actually_run"] == [0]
    none = r1.measure_seed(e2e["cfg"], e2e["owner"], tmp_path / "none", threads=1)
    assert (none["status"], none["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert none["steps_done"] == {} and none["seeds_actually_run"] == []

    def invalid(o):  # MINOR-2: rc 3 after a completed footer is liveness `invalid`
        (o / "logs" / "age-s0.rc").write_text("3\n")

    doc = run("invalid", invalid)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert doc["notes"] == ["ARM_DID_NOT_RUN (liveness invalid): age"]
    assert doc["arms"]["age"]["rc_meaning"].startswith("liveness invalid")
    assert doc["steps_done"]["age"] == ARM_ITERS  # it trained; nothing is cleared


def test_finalize_takes_status_and_exit_from_one_worst_seed(r1, e2e, tmp_path):
    def seed(s, code, status):
        d = copy.deepcopy(e2e["doc"])
        d.update(seed=s, exit=int(code), status=status)
        return d

    ow = r1.dataclasses.replace(e2e["owner"], seeds=(0, 1, 2))
    docs = [seed(0, Exit.OK, "ok"), seed(1, Exit.DID_NOT_RUN, "crashed"),
            seed(2, Exit.INERT, "partial")]  # fmt: skip
    res = tmp_path / "R.md"
    code = r1.finalize(docs, ow, tmp_path, results_path=res, write_ledger=False)
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == Exit.INERT and (run["exit"], run["status"]) == (5, "partial")
    docs[2].update(exit=int(Exit.OK), status="ok")
    code = r1.finalize(docs, ow, tmp_path, results_path=res, write_ledger=False)
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == Exit.DID_NOT_RUN and (run["exit"], run["status"]) == (3, "crashed")
    # A3b.5: 3 outranks 1; a tie goes to the lowest seed
    docs[0].update(exit=int(Exit.FAIL), status="failed")
    docs[2].update(exit=int(Exit.DID_NOT_RUN), status="did_not_run")
    r1.finalize(docs, ow, tmp_path, results_path=res, write_ledger=False)
    run = json.loads((tmp_path / "run.json").read_text())
    assert (run["exit"], run["status"]) == (3, "crashed")
    docs[1].update(exit=int(Exit.OK), status="ok")
    r1.finalize(docs, ow, tmp_path, results_path=res, write_ledger=False)
    run = json.loads((tmp_path / "run.json").read_text())
    assert (run["exit"], run["status"]) == (3, "did_not_run")
    docs[2].update(exit=int(Exit.OK), status="ok")
    code = r1.finalize(docs, ow, tmp_path, results_path=res, write_ledger=False)
    run = json.loads((tmp_path / "run.json").read_text())
    assert code == Exit.FAIL and (run["exit"], run["status"]) == (1, "failed")


def test_a_c9_failure_tags_the_oracle_ledger_rows(r1, e2e, tmp_path):
    doc = copy.deepcopy(e2e["doc"])
    doc["oracle_numbers_read"] = False
    run = json.loads((e2e["out_root"] / "run.json").read_text())
    out = tmp_path / "runs" / r1.RUN_ID
    out.mkdir(parents=True)
    with pytest.MonkeyPatch.context() as mp:
        _ledger_seams(mp, r1)
        led = json.loads(r1.write_run_ledger(run, [doc], out).read_text())
    rows = {r["key"]: r["how"] for r in led["rows"]}
    tagged = {k for k, h in rows.items() if h.startswith("NOT READ (C9/C9b failed")}
    assert any(".oracle_regret." in k for k in tagged)
    assert any("headroom_share" in k for k in tagged)
    assert "seed0.heldout.accuracy.all.oracle" in tagged
    assert not any("psiC_minus_fifo" in k for k in tagged)
    # MINOR-3: exactly A3b.4's three families -- never a control row, never an NLL row
    assert not any(k.startswith("seed0.controls.") for k in tagged)
    assert not any(".nll." in k for k in tagged)
    assert "seed0.controls.C7.per_pass.oracle" in rows
    assert "seed0.heldout.nll.real.oracle" in rows
    for k in tagged:
        assert k.startswith(
            (
                "seed0.heldout.accuracy.",
                "seed0.heldout.contrasts.oracle_regret.",
                "seed0.heldout.headroom_share",
            )
        ), k
    assert {k for k in rows if k.startswith("seed0.heldout.accuracy.") and k.endswith(
        ".oracle")} <= tagged  # fmt: skip
    by_key = {r["key"]: r for r in led["rows"]}
    for k in tagged:  # A3b.4: the fields, and the value kept
        assert by_key[k]["not_read"] is True
        assert by_key[k]["not_read_reason"] == "C9/C9b failed"
        assert "value" in by_key[k]  # kept as recorded, never deleted
    assert all("not_read" not in r for k, r in by_key.items() if k not in tagged)


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


# --------------------------------------------------------------------------- #
# Review 6267fd645698: MAJOR-1..3, MINOR-1/4/8, A3c
# --------------------------------------------------------------------------- #


def test_the_owner_table_heading_is_amendment_4s(r1):
    """Amendment 3d pre-declares A4's table under this heading (append-only: A2.8's
    blanks are never filled in place)."""
    assert r1.OWNER_TABLE_HEADING == "### A4.1 Owner values (fills A2.8's slots)"


def test_the_real_a4_table_is_filled_with_the_owners_values(r1, e2e, tmp_path):
    """A4.4 item 7: C10 reads A4.1 through the unchanged parser -- `filled` with 0.9 /
    0.01 / 3000 / eval. A fixture run (3 steps) read against the real PREREG is a
    mismatch with the registered R-1, so C10 fails and the seed exits 3."""
    reg = r1.owner_table(r1.prereg_config(0, 3000))
    assert reg["heading"] == r1.OWNER_TABLE_HEADING and reg["problems"] == []
    assert reg["status"] == "filled"
    assert reg["values"] == {
        "gamma": 0.9,
        "phi_weight_decay": 0.01,
        "iters": 3000,
        "o5": "eval",
    }
    out = _copy_run(e2e, tmp_path / "real")
    cfg = r1.dataclasses.replace(e2e["cfg"], owner_prereg=r1.PREREG)
    doc = r1.measure_seed(cfg, e2e["owner"], out, threads=1)
    c10 = doc["controls"]["C10"]
    assert c10["ok"] is False and c10["registered"]["status"] == "filled"
    assert any("registered R-1 3000" in p for p in c10["launch_problems"])
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert doc["controls_failed"] == ["C10"]


def test_a_filled_a4_table_under_the_heading_passes_c10(r1, e2e, substrate):
    """The fixture's synthetic PREREG carries the table under A4.1's heading, filled
    with the fixture's values: C10 reads it `filled`, and the e2e seed exits 0."""
    text = substrate["owner_prereg"].read_text()
    assert r1.OWNER_TABLE_HEADING in text.splitlines()
    reg = e2e["doc"]["controls"]["C10"]["registered"]
    assert reg["status"] == "filled" and reg["heading"] == r1.OWNER_TABLE_HEADING
    assert reg["values"] == {
        "gamma": 0.9,
        "phi_weight_decay": 0.01,
        "iters": ARM_ITERS,
        "o5": O5,
    }
    assert e2e["doc"]["controls"]["C10"]["ok"] and e2e["doc"]["exit"] == 0


def test_the_owner_table_parser_takes_the_one_format_and_nothing_looser(r1):
    ok = r1.parse_owner_table(_owner_table_text(r1, "0.9", "0.01", "3000", "train"))
    assert ok["status"] == "filled", ok["problems"]
    assert ok["values"] == {
        "gamma": 0.9,
        "phi_weight_decay": 0.01,
        "iters": 3000,
        "o5": "train",
    }
    part = r1.parse_owner_table(_owner_table_text(r1, "0.9", "______", "3000", "eval"))
    assert part["status"] == "blank" and "O4" in part["problems"][0]
    for args in (
        ("0.9 (ruled)", "0.01", "3000", "eval"),  # text beside the value
        ("9e-1", "0.01", "3000", "eval"),  # an exponent
        ("-0.9", "0.01", "3000", "eval"),
        ("0.9", "0.01", "3,000", "eval"),  # a separator
        ("0.9", "0.01", "0", "eval"),
        ("0.9", "0.01", "3000", "Eval"),
        ("", "0.01", "3000", "eval"),
    ):
        got = r1.parse_owner_table(_owner_table_text(r1, *args))
        assert got["status"] == "malformed", (args, got)
    good = _owner_table_text(r1, "0.9", "0.01", "3000", "eval")
    for bad in (
        good.replace("`0.9`", "0.9"),  # no backticks
        good.replace("| **O4**", "| O4"),  # the slot not bolded
        good.replace("|---|---|", "|--|--|"),
        good.replace("| **O1** gamma, both RSR arms only | `0.9` |\n", ""),  # three rows
        good.replace("`eval` |\n", "`eval` |\n| **O9** extra | `1` |\n"),  # five rows
        good + "\n" + r1.OWNER_TABLE_HEADING + "\n",  # the heading twice
    ):
        assert r1.parse_owner_table(bad)["status"] == "malformed"
    swapped = good.replace("| **O1** gamma", "| **OX**").replace("| **O4**", "| **O1**")
    assert r1.parse_owner_table(swapped)["status"] == "malformed"  # order is fixed
    assert r1.parse_owner_table("no table here")["status"] == "absent"


def test_c10_compares_the_owner_values_with_the_registered_ones(r1, e2e):
    """MAJOR-2: a value mistyped identically at launch and at measure-seed still
    reddens, because it is not the owner's registered value."""
    assert _c10(r1, e2e)["ok"]  # the fixture's synthetic PREREG holds its values
    blank = r1.parse_owner_table(_owner_table_text(r1, "______", "0.01", "3", O5))
    c = _c10(r1, e2e, registered=blank)
    assert not c["ok"] and any("blank" in p for p in c["launch_problems"])
    for args in (
        ("0.5", "0.01", str(ARM_ITERS), O5),
        ("0.9", "0.001", str(ARM_ITERS), O5),
        ("0.9", "0.01", "9", O5),
        ("0.9", "0.01", str(ARM_ITERS), "train"),
    ):
        reg = r1.parse_owner_table(_owner_table_text(r1, *args))
        c = _c10(r1, e2e, registered=reg)
        assert not c["ok"] and any("registered" in p for p in c["launch_problems"]), args


def _rewrite_beats(out, arm, n_beats, *, tail=None):
    """An arm's heartbeat as a failed process leaves it: the header, `n_beats` beats
    (steps 0 .. n-1, copied from the real ones), no footer, then `tail` rows."""
    hb = out / arm / "seed0" / "heartbeat.jsonl"
    rows = [json.loads(x) for x in hb.read_text().splitlines() if x.strip()]
    header = next(r for r in rows if r["kind"] == "header")
    beats = [r for r in rows if r["kind"] == "beat"]
    made = [{**beats[min(k, len(beats) - 1)], "step": k} for k in range(n_beats)]
    hb.write_text("\n".join(json.dumps(r) for r in [header, *made, *(tail or [])]) + "\n")
    log = out / "logs" / f"{arm}-s0.log"
    log.write_text("Traceback (most recent call last):\n  ...\nRuntimeError: probe\n")


def test_the_ledger_records_what_ran_not_what_was_asked(r1, e2e, tmp_path):
    """MAJOR-1, the reviewer's three probes as record shapes: (1) seed 1's launcher
    never ran; (2) seed 0's psiC crashed after 2 steps (crash line, rc 1); (3) an age
    arm killed by SIGTERM through zsh's `wait` after 35 beats (rc 143, no footer).
    `seeds_actually_run` and `steps_done` come from the records, per arm and seed."""
    ow = r1.dataclasses.replace(e2e["owner"], seeds=(0, 1))
    seed1 = r1.measure_seed(
        r1.dataclasses.replace(e2e["cfg"], seed=1), ow, tmp_path / "nothing", threads=1
    )
    assert seed1["status"] == "did_not_run" and seed1["steps_done"] == {}
    assert seed1["seeds_actually_run"] == []

    def ledger(name, docs):
        out = tmp_path / name / "runs" / r1.RUN_ID
        out.mkdir(parents=True)
        shutil.copy(e2e["out_root"] / "launch-git.json", out / "launch-git.json")
        with pytest.MonkeyPatch.context() as mp:
            _ledger_seams(mp, r1)
            r1.finalize(docs, ow, out, results_path=out / "R.md")
        return json.loads((out / "ledger.json").read_text())

    led = ledger("p1", [e2e["doc"], seed1])
    assert led["status"] == "did_not_run" and led["seeds_actually_run"] == [0]
    assert led["steps_done"] == ARM_ITERS
    rows = {r["key"]: r["value"] for r in led["rows"]}
    assert rows["steps_done_by_job"] == {f"{a}-s0": ARM_ITERS for a in r1.ARMS}

    out = _copy_run(e2e, tmp_path / "p2")
    _rewrite_beats(out, "psiC", 2, tail=[{"kind": "crash", "exc_type": "RuntimeError"}])
    (out / "logs" / "psiC-s0.rc").write_text("1\n")
    crashed = r1.measure_seed(e2e["cfg"], ow, out, threads=1)
    assert (crashed["status"], crashed["exit"]) == ("crashed", int(Exit.DID_NOT_RUN))
    assert crashed["steps_done"] == {"psiC": 2, "fifo": ARM_ITERS, "age": ARM_ITERS}
    assert crashed["seeds_actually_run"] == [0]
    ok1 = {**copy.deepcopy(e2e["doc"]), "seed": 1, "seeds_actually_run": [1]}
    led = ledger("p2led", [crashed, ok1])
    assert led["status"] == "crashed" and led["seeds_actually_run"] == [0, 1]
    assert led["steps_done"] == 2  # the least-trained job, as recorded
    rows = {r["key"]: r["value"] for r in led["rows"]}
    assert rows["steps_done_by_job"]["psiC-s0"] == 2

    out = _copy_run(e2e, tmp_path / "p3")
    _rewrite_beats(out, "age", 35)
    (out / "logs" / "age-s0.rc").write_text("143\n")
    killed = r1.measure_seed(e2e["cfg"], ow, out, threads=1)
    assert (killed["status"], killed["exit"]) == ("crashed", int(Exit.DID_NOT_RUN))
    assert killed["notes"] == ["ARM_CRASHED: age"]
    assert killed["arms"]["age"]["rc_meaning"] == "killed by signal 15"
    assert killed["steps_done"]["age"] == 35 and killed["seeds_actually_run"] == [0]


def test_two_exit_3_arms_tie_to_the_first_in_psic_fifo_age(r1, e2e, tmp_path):
    """A3c.2 (review 6267fd645698 MINOR-1): psiC crashed + fifo did-not-run -> the seed
    is `crashed`; swapped -> `did_not_run`. The notes list every non-OK arm."""

    def refused(out, arm):
        shutil.rmtree(out / arm / "seed0")  # a copy: nothing of the e2e run is touched
        (out / "logs" / f"{arm}-s0.log").write_text("WarmStartRefused: probe\n")
        (out / "logs" / f"{arm}-s0.rc").write_text("1\n")

    def crashed(out, arm):
        _rewrite_beats(out, arm, 1, tail=[{"kind": "crash"}])
        (out / "logs" / f"{arm}-s0.rc").write_text("1\n")

    out = _copy_run(e2e, tmp_path / "a")
    crashed(out, "psiC")
    refused(out, "fifo")
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert doc["status"] == "crashed" and doc["exit"] == int(Exit.DID_NOT_RUN)
    assert doc["notes"] == ["ARM_CRASHED: psiC", "ARM_DID_NOT_RUN: fifo"]
    assert doc["steps_done"] == {"psiC": 1, "fifo": 0, "age": ARM_ITERS}
    out = _copy_run(e2e, tmp_path / "b")
    refused(out, "psiC")
    crashed(out, "fifo")
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert doc["status"] == "did_not_run" and doc["exit"] == int(Exit.DID_NOT_RUN)
    assert doc["notes"] == ["ARM_DID_NOT_RUN: psiC", "ARM_CRASHED: fifo"]
    assert r1.worst_outcome([(0, "ok", "ARM_OK")] * 3) is None


def test_a_failed_control_outranks_inconclusive_liveness_within_a_seed(r1, e2e, tmp_path):
    """Review de8f813 MINOR-A3c-1: C5 inconclusive (1) and a C10 failure (3) in one
    seed -> 3, whichever check runs first; every finding is a note."""
    out = _copy_run(e2e, tmp_path / "both")
    log = out / "logs" / "fifo-s0.log"
    log.write_text(log.read_text().replace('"band": "live"', '"band": "inconclusive"'))
    lg = json.loads((out / "launch-git.json").read_text())
    (out / "launch-git.json").write_text(json.dumps({**lg, "wd": "0.0"}))
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert "CONTROL_FAILED: ['C10']" in doc["notes"]
    assert "LIVENESS_INCONCLUSIVE: ['fifo']" in doc["notes"]
    out = _copy_run(e2e, tmp_path / "unsure")  # alone, inconclusive is 1 (A3c.4)
    log = out / "logs" / "fifo-s0.log"
    log.write_text(log.read_text().replace('"band": "live"', '"band": "inconclusive"'))
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("failed", int(Exit.FAIL))


def test_measure_seed_dry_run_reads_a_steps_0_launch_and_c10_still_fails_it(
    r1, e2e, tmp_path
):
    """MINOR-4: launcher v5's only dry run (STEPS 0, CKPT 0, no checkpoint, an isolated
    RSR_ORCH_ROOT, a JOB_SUFFIX) is read by `measure-seed --dry-run`: the job records
    are found under the isolated root and suffix, nothing is measured held out, and C10
    fails on the dry-run fields by design."""
    out = _copy_run(e2e, tmp_path / "dry")
    iso, suffix = tmp_path / "iso", "-dry"
    for a in r1.ARMS:
        d = out / a / "seed0"
        for f in d.glob("ckpt-*.pt"):
            f.unlink()  # a copy's files; a STEPS-0 run writes no checkpoint
        rows = [json.loads(x) for x in (d / "heartbeat.jsonl").read_text().splitlines()]
        keep = [r for r in rows if r["kind"] in ("header", "footer")]
        (d / "heartbeat.jsonl").write_text("\n".join(json.dumps(r) for r in keep) + "\n")
        log = out / "logs" / f"{a}-s0.log"
        res = r1.read_cli_result(log)
        res.update(steps_done=0, steps_requested=0)
        res["config"]["iters"] = 0
        log.write_text(json.dumps(res, indent=2) + "\n")
    ow0 = r1.dataclasses.replace(e2e["owner"], iters=0)
    cfg0 = r1.dataclasses.replace(e2e["cfg"], iters=0, prereg_strict=True)
    assert cfg0.snapshots == [] and cfg0.ckpt_every == 0
    (iso / ".orchestrator" / "jobs").mkdir(parents=True)
    for a in r1.ARMS:
        job = {"pinned_sha": "x", "threads": 1, "rc": 0, "cmd": r1.cli_argv(
            cfg0, a, ow0, r1.arm_dir(out, a, 0))}  # fmt: skip
        f = iso / ".orchestrator" / "jobs" / f"rsr-run1-{a}-s0{suffix}.json"
        f.write_text(json.dumps(job))
    lg = json.loads((out / "launch-git.json").read_text())
    lg.update(steps="0", ckpt="0", dry_run="1", rsr_orch_root=str(iso), out=str(out),
              job_suffix=suffix)  # fmt: skip
    (out / "launch-git.json").write_text(json.dumps(lg))
    shutil.rmtree(out / "heldout")
    env = {"RSR_ORCH_ROOT": str(iso), "OUT": str(out), "JOB_SUFFIX": suffix}
    doc = r1.measure_seed(cfg0, ow0, out, threads=1, environ=env, dry_run=True)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert doc["config"]["orch_root"] == str(iso)
    assert doc["config"]["job_suffix"] == suffix
    assert all(doc["arms"][a]["job_record"] is not None for a in r1.ARMS)
    assert "DRY_RUN: STEPS 0, no step trained; no held-out pass" in doc["notes"]
    c10 = doc["controls"]["C10"]
    assert not c10["ok"]
    for what in ("dry_run", "rsr_orch_root", "job_suffix", "job records read from"):
        assert any(what in p for p in c10["launch_problems"]), what
    assert not any("no job record" in p for v in c10["per_arm"].values()
                   for p in v["problems"])  # fmt: skip
    assert not (out / "heldout").exists() and "heldout" not in doc
    assert doc["steps_done"] == dict.fromkeys(r1.ARMS, 0)
    assert doc["seeds_actually_run"] == []  # nothing trained
    # The same records without --dry-run are refused before anything is read
    real = r1.measure_seed(cfg0, ow0, out, threads=1, environ=env)
    assert real["status"] == "did_not_run" and "arms" not in real


def test_the_git_resolver_is_portable(r1, monkeypatch, tmp_path):
    """MINOR-8: Homebrew's git if present, else the PATH's as an absolute path, else a
    clear refusal (tree_state then reports the error, and a launch refuses)."""
    exe = r1.git_exe()
    assert Path(exe).is_absolute()
    monkeypatch.setattr(r1, "HOMEBREW_GIT", (str(tmp_path / "none"),))
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "git").write_text("#!/bin/sh\nexit 0\n")
    (fake / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake))
    assert r1.git_exe() == str(fake / "git")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(r1.Refused, match="no git"):
        r1.git_exe()
    st = r1.tree_state()
    assert st["sha"] is None and "no git" in st["error"]


class _Net(torch.nn.Module):
    """A test-side wrapper, written independently of the harness's `ZeroSlots`.
    ``zero``: the memory every forward reads is zeros. ``boost``: the true next token is
    made the argmax wherever its id is even, so answer correctness is a mix of right
    and wrong -- a target read one position off cannot pass unseen."""

    def __init__(self, model, *, zero=False, boost=False):
        super().__init__()
        self.inner, self.cfg, self.zero, self.boost = model, model.cfg, zero, boost

    @property
    def embed(self):
        return self.inner.embed

    @property
    def blocks(self):
        return self.inner.blocks

    def forward(self, ids_t, mask_t, kv, valid, bos_ctx, bos_valid, **kw):
        kv = kv * 0 if self.zero else kv
        out = self.inner(ids_t, mask_t, kv, valid, bos_ctx, bos_valid, **kw)
        if not self.boost:
            return out
        nxt = ids_t[:, 1:]
        boost = torch.zeros_like(out.logits)
        hit = (nxt % 2 == 0).to(out.logits.dtype) * 1e3
        boost[:, :-1].scatter_(-1, nxt.unsqueeze(-1), hit.unsqueeze(-1))
        return dataclasses.replace(out, logits=out.logits + boost)


def _independent(r1, net, docs, vmap, cfg):
    """`run_batch`'s per-document numbers, recomputed from scratch (MAJOR-3; the
    reviewer's hand_cases.py A2): float64 `log_softmax`, explicit (j-1 -> j) indices, the
    real-token mask read at the TARGET position, through the same policy loop."""
    from rsr.model.tg.policy_loop import run_policy_loop

    S, L = cfg.steps_per_stream, int(net.cfg.max_sentence_tokens)
    ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=S)
    tm, gap = loop.answer_targets(docs, vmap, max_tokens=L, steps=S)
    B = len(docs)
    mine = [{"real_sum": 0.0, "real_n": 0, "answers": []} for _ in range(B)]

    def step_fn(t, out, ids_t, mask_t, row_valid):
        lp = torch.log_softmax(out.logits.double(), -1)
        for b in range(B):
            for j in range(1, L):
                if mask_t[b, j]:
                    nll = -float(lp[b, j - 1, ids_t[b, j]])
                    mine[b]["real_sum"] += nll
                    mine[b]["real_n"] += 1
                    if tm[b, t, j]:
                        ok = int(lp[b, j - 1].argmax()) == int(ids_t[b, j])
                        mine[b]["answers"].append((t, int(gap[b, t]), ok, nll))
        return torch.zeros(())

    net.eval()
    with torch.no_grad():
        lengths = torch.full((B,), S)
        run_policy_loop(net, ids, mask, lengths, r1.FIFOPolicy(), step_fn=step_fn)
    return mine


def test_run_batch_matches_an_independent_recomputation(r1, substrate):
    """MAJOR-3: answer correctness (the headline quantity), the real-token NLL mask
    (A3.4) and the slots-zeroed floor (§7.2), each recomputed independently, with and
    without memory and with correctness exercised both ways; FIFO's residency is a hit
    iff gap <= M; the floor's answers differ from the live ones; and every held-out
    forward is eval mode and no_grad, with the caller's mode restored."""
    cfg = tiny_cfg(r1, substrate)
    payload = r1.read_checkpoint(substrate["ckpt"])
    payload["meta"]["config"] = json.loads(substrate["result"].read_text())["config"]
    model = r1.model_from_payload(payload)
    docs = r1.docs_by_id(cfg, range(*EVAL))
    vmap = r1.vocab_map(cfg)
    seen = []
    hook = model.register_forward_pre_hook(
        lambda m, a: seen.append((m.training, torch.is_grad_enabled()))
    )
    model.train()
    live = r1.run_batch(model, r1.FIFOPolicy(), docs, vmap, cfg)
    hook.remove()
    assert seen and set(seen) == {(False, False)} and model.training is True
    floor = r1.run_batch(model, r1.FIFOPolicy(), docs, vmap, cfg, zero_slots=True)
    assert [r["answers"] for r in floor] != [r["answers"] for r in live]
    assert all(hit == (g <= cfg.memory_slots) for r in live for g, hit in r["hits"])
    flags = []
    for zero in (False, True):
        for boost in (False, True):
            under = _Net(model, boost=boost)  # what run_batch is handed
            recs = r1.run_batch(under, r1.FIFOPolicy(), docs, vmap, cfg, zero_slots=zero)
            mine = _independent(r1, _Net(model, zero=zero, boost=boost), docs, vmap, cfg)
            for m, r in zip(mine, recs, strict=True):
                assert m["real_n"] == r["real_n"], (zero, boost)
                assert abs(m["real_sum"] - r["real_sum"]) / m["real_n"] < 1e-6
                got = [tuple(a[:3]) for a in r["answers"]]
                assert [a[:3] for a in m["answers"]] == got, (zero, boost)
                pairs = zip(m["answers"], r["answers"], strict=True)
                assert all(abs(x[3] - y[3]) < 1e-4 for x, y in pairs)
            if boost:
                flags += [a[2] for r in recs for a in r["answers"]]
    assert any(flags) and not all(flags)  # correctness is exercised both ways


def test_a3d2_c6_crashed_outranks_a_control_failure_at_exit_3(r1, e2e, tmp_path):
    """A3d.2: at exit 3 within a seed, C6 (`crashed`) sets the status over a C1 / C10
    failure (`did_not_run`); both are in the notes."""
    out = _copy_run(e2e, tmp_path / "c6c10")
    log = out / "logs" / "psiC-s0.log"
    res = r1.read_cli_result(log)
    res["steps_done"] = ARM_ITERS - 1  # the arm is OK by rc; C6 reads the short count
    log.write_text(json.dumps(res, indent=2) + "\n")
    lg = json.loads((out / "launch-git.json").read_text())
    (out / "launch-git.json").write_text(json.dumps({**lg, "wd": "0.0"}))
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("crashed", int(Exit.DID_NOT_RUN))
    assert any(n.startswith("ARM_CRASHED: C6") for n in doc["notes"])
    assert "CONTROL_FAILED: ['C10']" in doc["notes"]


def test_a3d2_an_arms_exit_1_does_not_hide_a_control_failure(r1, e2e, tmp_path):
    """A3d.2: the order spans arms and controls. An arm's own exit 1 (rc 1 after a
    completed footer) with a C10 failure gives 3, and both are in the notes."""
    out = _copy_run(e2e, tmp_path / "arm1c10")
    (out / "logs" / "fifo-s0.rc").write_text("1\n")
    lg = json.loads((out / "launch-git.json").read_text())
    (out / "launch-git.json").write_text(json.dumps({**lg, "wd": "0.0"}))
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert "LIVENESS_INCONCLUSIVE: fifo" in doc["notes"]
    assert "CONTROL_FAILED: ['C10']" in doc["notes"]


def _a4(r1, gamma="0.9") -> str:
    return _owner_table_text(r1, gamma, "0.01", "3000", "eval")


def test_the_a4_heading_counts_once_in_the_whole_file_outside_every_fence(r1):
    """A3d.1 rule 1 (review d86af31 MINOR-N3): the heading line occurs exactly once in
    the WHOLE file, fenced copies counted, and outside every fence -- else `malformed`
    (`absent` at 0). Fences are CommonMark's (MINOR-N2). Inline-backtick mentions are
    not heading lines."""
    h = r1.OWNER_TABLE_HEADING

    def status(text):
        return r1.parse_owner_table(text)["status"]

    for fence in ("```", "~~~", "```markdown", "   ```", "````"):
        alone = (
            f"# prose\n\n{fence}\n"
            + _a4(r1, "0.5")
            + f"{fence.strip().rstrip('markdown')}\n"
        )
        got = r1.parse_owner_table(alone)  # one copy, but inside a fence: rule 1 says so
        assert got["status"] == "malformed", fence
        assert "in a fence" in got["problems"][0] and "rule 1" in got["problems"][0]
    both = "```\n" + _a4(r1, "0.5") + "```\n\n" + _a4(r1)
    assert status(both) == "malformed"  # two copies (a811714's reading, restored)
    # (a) a 4-backtick fence containing a ``` line: the inner ``` does not close it
    inner = "````\n```\n" + _a4(r1, "0.5") + "```\n````\n"
    assert status(inner) in ("absent", "malformed") and status(inner) == "malformed"
    assert status("~~~\n```\n" + _a4(r1, "0.5") + "```\n~~~\n") == "malformed"
    # a closer must be the same character, at least as long, with no info string
    for opener, closer in (("````", "```"), ("```", "~~~"), ("```", "``` x")):
        text = f"{opener}\nfenced\n{closer}\n\n" + _a4(r1)
        assert status(text) == "malformed", (opener, closer)  # still inside the fence
    closes = "```\nfenced\n````\n\n" + _a4(r1)
    assert status(closes) == "filled"  # a longer closer of the same character closes
    assert status("```a`b\n\n" + _a4(r1)) == "filled"  # a backtick info string: no fence
    assert status("    ```\n\n" + _a4(r1)) == "filled"  # 4-space indent: not a fence
    # (c) inline-backtick mentions of the heading do not count
    inline = f"A3d quotes `{h}` inline.\n   `{h}`\n\n" + _a4(r1)
    got = r1.parse_owner_table(inline)
    assert got["status"] == "filled" and got["values"]["gamma"] == 0.9
    assert status(f"A3d quotes `{h}` inline.\n") == "absent"
    unclosed = _a4(r1).replace("| slot | value |", "```\n| slot | value |")
    assert status(unclosed) == "malformed"


def test_a_non_newline_line_break_makes_the_a4_table_malformed(r1):
    """Review fe0e01f MINOR-A3d-1(b): the parser splits on "\\n" only; a line break of
    any other kind on the heading's line, a table line, or a would-be blank line is
    `malformed`, never a fresh line. CRLF endings still parse."""
    good = _a4(r1)
    h = r1.OWNER_TABLE_HEADING
    for ch in (
        chr(0x2028),
        chr(0x2029),
        "\x85",
        "\x0b",
        "\x0c",
        "\x1c",
        "\x1d",
        "\x1e",
        "\r",
    ):
        before = good.replace(h + "\n", "prose" + ch + h + "\n")
        assert r1.parse_owner_table(before)["status"] == "malformed", repr(ch)
        row = good.replace("| **O4** phi", "| **O4** phi" + ch)
        assert r1.parse_owner_table(row)["status"] == "malformed", repr(ch)
        if ch != "\r":  # "\r\n" alone is a CRLF blank line, which is blank
            gap = good.replace(h + "\n\n", h + "\n" + ch + "\n")
            assert r1.parse_owner_table(gap)["status"] == "malformed", repr(ch)
    crlf = good.replace("\n", "\r\n")
    got = r1.parse_owner_table(crlf)
    assert got["status"] == "filled" and got["values"]["iters"] == 3000


def test_the_prereg_layers_cite_every_amendment_and_review(r1):
    """Review d86af31 MINOR-N4: every `prereg(rsr-run1): Amendment <k>` commit in this
    branch's history is in PREREG_COMMITS under `A<k>`, and every PREREG review commit
    is in REVIEW_COMMITS -- so Amendment 4 cannot land uncited."""
    import subprocess

    def log(path):
        r = subprocess.run(
            [r1.git_exe(), "-C", str(ROOT), "log", "--format=%h%x09%s", "--abbrev=7",
             "HEAD", "--", path],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        return [x.split("\t", 1) for x in r.stdout.splitlines()]

    prereg = log(r1.PREREG)
    if r1.PREREG_COMMITS["base"] not in {sha for sha, _ in prereg}:
        pytest.skip("shallow history: the PREREG's base commit is not in this clone")
    amendments = {
        "A" + s.split("Amendment ", 1)[1].split()[0]: sha
        for sha, s in prereg
        if s.startswith("prereg(rsr-run1): Amendment")
    }
    assert amendments, "no amendment commit found"
    for key, sha in amendments.items():
        assert r1.PREREG_COMMITS.get(key) == sha, (key, sha)
    reviews = {
        sha
        for sha, s in log("docs/reviews/prereg-rsr-run1/")
        if s.startswith("review(prereg-rsr-run1):")
    }
    assert reviews and reviews <= set(r1.REVIEW_COMMITS), reviews - set(r1.REVIEW_COMMITS)


def _set_rc(out, arms, rc):
    for a in arms:
        (out / "logs" / f"{a}-s0.rc").write_text(f"{rc}\n")


def _inconclusive(out, arms):
    """rc 1 after a completed footer, and the printed band `inconclusive` (train()'s
    own mapping), in each of ``arms``."""
    _set_rc(out, arms, 1)
    for a in arms:
        log = out / "logs" / f"{a}-s0.log"
        log.write_text(
            log.read_text().replace('"band": "live"', '"band": "inconclusive"')
        )


def test_an_exit_1_stop_still_reads_c2_c3_c8_and_a_failure_gives_3(r1, e2e, tmp_path):
    """Review d86af31 MINOR-N1 (A3d.2): every arm inconclusive, and fifo's steps drew
    OTHER pool documents -> C2 fails -> 3, not 1. C4 and C7 are listed as not
    evaluated. Without the pairing break the seed is `failed` / 1 with C2, C3 and C8
    read and green. Liveness is noted once per arm (review NIT), never twice."""
    out = _copy_run(e2e, tmp_path / "ok1")
    _inconclusive(out, r1.ARMS)
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("failed", int(Exit.FAIL))
    assert all(doc["controls"][k]["ok"] for k in ("C2", "C3", "C8"))
    assert "NOT_EVALUATED: ['C4', 'C7'] (the seed stopped; A3d.2)" in doc["notes"]
    liveness = [n for n in doc["notes"] if n.startswith("LIVENESS_INCONCLUSIVE")]
    assert liveness == [f"LIVENESS_INCONCLUSIVE: {a}" for a in r1.ARMS]
    assert "heldout" not in doc

    out = _copy_run(e2e, tmp_path / "other")
    _inconclusive(out, r1.ARMS)
    f = out / "fifo" / "seed0" / "diagnostics.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    for r in rows:
        r["pool_doc_ids"] = [i + 2 for i in r["pool_doc_ids"]]  # outside the pool
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert doc["controls"]["C2"]["ok"] is False
    assert "CONTROL_FAILED: ['C2']" in doc["notes"]
    assert "NOT_EVALUATED: ['C4', 'C7'] (the seed stopped; A3d.2)" in doc["notes"]
    assert [n for n in doc["notes"] if n.startswith("LIVENESS")] == liveness


def test_measure_seed_passes_every_arms_sampler_into_c2(r1, e2e, tmp_path):
    """Review d86af31 NIT (Y4): the wiring, not only the unit. The e2e's C2 carries the
    three arms' samplers, and a sampler that differs in one arm's recorded config
    reddens C2 through `measure_seed`."""
    c2 = e2e["doc"]["controls"]["C2"]["b_fingerprint"]
    assert c2["sampling_per_arm"] == dict.fromkeys(r1.ARMS, loop.SAMPLING_PERMUTATION)
    assert c2["sampling_agrees"] is True
    out = _copy_run(e2e, tmp_path / "sampler")
    log = out / "logs" / "fifo-s0.log"
    res = r1.read_cli_result(log)
    res["config"]["sampling"] = loop.SAMPLING_WITH_REPLACEMENT
    log.write_text(json.dumps(res, indent=2) + "\n")
    doc = r1.measure_seed(e2e["cfg"], e2e["owner"], out, threads=1)
    assert doc["controls"]["C2"]["b_fingerprint"]["sampling_agrees"] is False
    assert (doc["status"], doc["exit"]) == ("did_not_run", int(Exit.DID_NOT_RUN))
    assert "C2" in doc["controls_failed"]


# --------------------------------------------------------------------------- #
# A4.2: the file cadence EVERY and the DIAGNOSTIC step-160 reading
# --------------------------------------------------------------------------- #


def test_every_is_gcd_of_the_diagnostic_step_and_ckpt(r1, monkeypatch):
    """A4.2 / launcher v6: EVERY = gcd(160, CKPT) for STEPS >= 160 and CKPT > 0, else
    CKPT. At 3000 steps: CKPT 1000, EVERY 40, 75 files; snapshots 160, 1000, 2000 and
    the final 3000. Under v5 (`LAUNCHER_V6` False) the files stay at CKPT and there is
    no step-160 checkpoint, so no diagnostic reading."""
    assert r1.DIAG_STEP == 160
    assert r1.derive_every(3000, 1000, 160) == 40
    assert r1.derive_every(150, 50, 160) == 50  # STEPS < 160
    assert r1.derive_every(0, 0, 160) == 0  # the dry run
    assert r1.derive_every(3001, 3001, 160) == math.gcd(160, 3001)
    cfg = r1.prereg_config(0, 3000)
    monkeypatch.setattr(r1, "LAUNCHER_V6", False)
    assert cfg.ckpt_every == 1000 and cfg.diag is None
    assert cfg.snapshots == [1000, 2000, 3000]
    monkeypatch.setattr(r1, "LAUNCHER_V6", True)
    assert cfg.ckpt_every == 40 and len(cfg.checkpoint_files) == 75
    assert cfg.checkpoint_files == list(range(40, 3001, 40)) and cfg.ckpt == 1000
    assert cfg.diag == 160 and cfg.snapshots == [160, 1000, 2000, 3000]
    ow = owner(r1, iters=3000)
    argv = r1.cli_argv(cfg, "psiC", ow, Path("x"))
    assert argv[argv.index("--ckpt-every") + 1] == "40"
    names = [n for n, _src, _z in r1.heldout_passes([160, 1000, 2000], cfg.diag)]
    for n in ("psiC@160", "age@160", "sz_psiC@160", "sz_age@160", "psiC@1000",
              "age@2000"):  # fmt: skip
        assert names.count(n) == 1, n
    assert not any(n.startswith(("fifo@", "lru@", "oracle@", "sz_fifo@")) for n in names)
    assert "sz_psiC@1000" not in names  # the floors are re-run at 160 only


def test_the_step_diag_reading_is_diagnostic_and_never_the_headline(r1, e2e):
    """A4.2: psiC@d, age@d, sz_psiC@d, sz_age@d on the same EVAL documents; fifo, LRU
    and sz_fifo reused from the final reading; the §7.1 contrast set labelled
    DIAGNOSTIC; the headline is the final reading's alone."""
    cfg, doc = e2e["cfg"], e2e["doc"]
    assert cfg.diag == DIAG
    hd = e2e["out_root"] / "heldout" / "seed0"
    for n in (f"psiC@{DIAG}", f"age@{DIAG}", f"sz_psiC@{DIAG}", f"sz_age@{DIAG}"):
        assert [r["doc"] for r in r1.read_pass(hd, n)] == list(range(*EVAL)), n
    ho = doc["heldout"]
    dg = ho["diagnostic_step"]
    assert dg["step"] == DIAG and "DIAGNOSTIC" in dg["label"] and "never" in dg["label"]
    acc, con = ho["accuracy"]["all"], dg["contrasts"]
    want = acc[f"psiC@{DIAG}"]["point"] - acc["fifo"]["point"]
    assert con["acc.all.psiC_minus_fifo"]["point"] == pytest.approx(want)
    want = acc[f"psiC@{DIAG}"]["point"] - acc[f"age@{DIAG}"]["point"]
    assert con["acc.all.psiC_minus_age"]["point"] == pytest.approx(want)
    names = ("psiC_minus_fifo", "age_minus_fifo", "psiC_minus_age")  # A4.2's set
    want_keys = {f"acc.{b}.{c}" for b in r1.BUCKET_NAMES for c in names}
    want_keys |= {f"nll_{k}.{c}" for k in ("real", "answer") for c in names}
    want_keys |= {f"{r}.{a}" for r in ("R", "R_own_floor") for a in r1.RSR_ARMS}
    want_keys |= {f"floor_difference.sz_{a}_minus_sz_fifo" for a in r1.RSR_ARMS}
    assert set(con) == want_keys
    rd = doc["training"]["diagnostic_step"]  # §7.4 recency over steps < d
    assert rd["step"] == DIAG and rd["label"] == r1.DIAG_LABEL
    for a in r1.RSR_ARMS:
        ph = rd["diagnostics"][a]
        assert ph["warm"]["n_steps"] + ph["scored"]["n_steps"] == DIAG
    assert f"### Step-{DIAG} recency statistics (§7.4) -- {r1.DIAG_LABEL}" in (
        e2e["results"].read_text()
    )
    led = json.loads((e2e["out_root"] / "ledger.json").read_text())
    rows = {r["key"]: r for r in led["rows"]}
    tagged = {k for k, r in rows.items() if r.get("diagnostic")}
    assert "seed0.heldout.diagnostic_step.contrasts.acc.all.psiC_minus_fifo" in tagged
    assert f"seed0.heldout.accuracy.all.psiC@{DIAG}" in tagged
    assert f"seed0.heldout.accuracy.all.sz_age@{DIAG}" in tagged
    assert any(k.startswith("seed0.training.diagnostic_step") for k in tagged)
    assert all(rows[k]["how"].startswith(r1.DIAG_LABEL) for k in tagged)
    assert "seed0.heldout.contrasts.acc.all.psiC_minus_fifo" not in tagged
    assert "seed0.heldout.accuracy.all.psiC@1" not in tagged and "headline" not in tagged
    g = ho["accuracy"]["gap_2_to_M"]
    want = g[f"age@{DIAG}"]["point"] - g[f"sz_age@{DIAG}"]["point"]
    assert con["R_own_floor.age"]["point"] == pytest.approx(want)
    assert set(dg["accuracy"]["all"]) == {
        f"psiC@{DIAG}",
        f"age@{DIAG}",
        f"sz_psiC@{DIAG}",
        f"sz_age@{DIAG}",
    }
    assert not any("oracle" in k for k in con)  # no oracle-derived number here
    head = r1.headline([doc])["per_seed"][0]
    assert head["point"] == ho["contrasts"][r1.HEADLINE]["point"]  # the final reading
    text = e2e["results"].read_text()
    assert f"### Step-{DIAG} phi reading -- {r1.DIAG_LABEL}" in text
    assert f"- DIAGNOSTIC step {DIAG} `acc.all.psiC_minus_fifo`" in text
    assert r1.DIAG_COST_NOTE in text and "EXTRAPOLATED" in r1.DIAG_COST_NOTE
    assert "1,090 s per seed" in r1.DIAG_COST_NOTE


def test_c10_under_v6_checks_every_and_the_new_launch_fields(r1, e2e, monkeypatch):
    """A4.2 C10: launch-git's `ckpt_every` (EVERY, and the v6 rule on steps),
    `ckpt_every_rule` and `diag_step`; each job's `--ckpt-every` == EVERY; the CKPT
    derivation is still checked."""
    monkeypatch.setattr(r1, "LAUNCHER_V6", True)
    lg = json.loads((e2e["out_root"] / "launch-git.json").read_text())
    assert (lg["ckpt_every"], lg["diag_step"]) == ("1", str(DIAG))
    assert _c10(r1, e2e)["ok"], _problems(_c10(r1, e2e))
    for key, val, why in (
        ("ckpt_every", "2", "ckpt_every"),
        ("ckpt_every_rule", "CKPT", "ckpt_every_rule"),
        ("diag_step", "160", "diag_step"),
        ("ckpt", "3", "ckpt"),
    ):
        c = _c10(r1, e2e, launch={**lg, key: val})
        assert not c["ok"] and any(why in p for p in c["launch_problems"]), key
    cfg = r1.dataclasses.replace(e2e["cfg"], iters=3000, diag_step=160)
    ran = {  # the commands that actually ran: --ckpt-every 1
        a: {"cmd": r1.cli_argv(e2e["cfg"], a, e2e["owner"], Path("x"))} for a in r1.ARMS
    }
    c = _c10(r1, e2e, cfg=cfg, owner_=owner(r1, iters=3000), jobs=ran)
    assert c["derived_ckpt"] == 1000 and c["derived_every"] == 40
    probs = c["per_arm"]["psiC"]["problems"]
    assert any("--ckpt-every '1' != 40" in p for p in probs)
    assert any("derived cadence [40, 80," in p for p in probs)
    monkeypatch.setattr(r1, "LAUNCHER_V6", False)  # v5: the fields are not v5's
    assert _c10(r1, e2e, launch={**lg, "diag_step": "160"})["ok"]
