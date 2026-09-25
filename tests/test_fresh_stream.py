"""Fresh stream's own gates (`experiments/fresh-stream/PREREG.md`, 83128ee).

The training change is exercised on tiny configurations (a few iterations of a
d=32 model, or `train()` up to its first step); the parent loop is driven by fake
children over checkpoint files touched on disk and a fake clock; the rule is fed
synthetic S0-03 tables. Mutation-battery gates (`scripts/mutation_battery.py`,
``# --- fresh-stream ---``):

* ``test_default_path_config_is_unchanged`` -- the stream stamped on the default path;
* ``test_initial_parameters_identical_with_and_without_the_stream`` -- a draw from
  the global RNG before model init in stream mode;
* ``test_resume_continues_the_stream_at_the_resumed_step`` -- a resumed run restarts
  the stream at its first window;
* ``test_classify_table`` -- the classification table miswired;
* ``test_control_1_fails_at_2e_6`` -- the control-1 tolerance widened;
* ``test_vocabulary_closure_raises_on_an_out_of_vocab_word`` -- the closure disabled;
* ``test_disjointness_catches_overlap`` -- the held-out overlap dropped from control 3;
* ``test_C_needs_every_seed`` -- "every seed" weakened to "any seed" for C.
"""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import rsr.train.loop as loop  # noqa: E402
from rsr.data.synthetic import SyntheticConfig, generate, to_bytes  # noqa: E402
from rsr.train.loop import DocumentStream  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "fresh_stream_run", ROOT / "experiments" / "fresh-stream" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]
YES, NO = 0.0625, 0.015625  # binary-exact, either side of DELTA
C_YES, C_NO = 0.875, 0.9375  # either side of C_THRESHOLD = 0.8875

TINY = dict(
    d=32,
    steps_per_stream=48,
    batch=2,
    max_tokens=16,
    memory_slots=4,
    device="cpu",
    beat_every=1,
)
TINY_STREAM = DocumentStream(offset=200, stride=2)


def _prereg_text() -> str:
    return (ROOT / "experiments" / "fresh-stream" / "PREREG.md").read_text()


def _prereg() -> dict:
    return yaml.safe_load(_prereg_text().split("---")[1])


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed; the instrument, imported
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    p = _prereg()
    t = p["thresholds"]
    assert run.SEEDS == p["seeds"] == SEEDS
    assert list(run.ARMS) == list(p["arms"]) == ["A", "B"]
    assert {a: list(c) for a, c in run.CHECKPOINTS.items()} == p["checkpoints_measured"]
    assert run.FINAL == run.ITERS == 3000 and run.RESUME_STEP == 1000
    assert t["DELTA"].startswith("0.03 ") and run.DELTA == 0.03 == run.ST.DELTA
    assert t["BRIER_MARGIN"].startswith(f"{run.BRIER_MARGIN} ")
    assert "0.9375 - 0.05 = 0.8875" in t["BRIER_MARGIN"]
    assert pytest.approx(0.8875) == run.C_THRESHOLD
    assert t["reproduction_tolerance"].startswith("1e-6 ") and run.REPRO_TOL == 1e-6
    assert t["deadline"].startswith("2026-09-26 06:00 America/Los_Angeles")
    la = dt.datetime(2026, 9, 26, 6, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert run.deadline_epoch() == la.timestamp()
    assert "[4160 + 16 t, 4160 + 16 t + 16)" in p["stream"]
    assert (run.STREAM.offset, run.STREAM.stride, run.BATCH) == (4160, 16, 16)
    assert run.STREAM.vocab_documents == 64 and run.VOCAB_DOCUMENTS == (0, 64)
    assert p["heldout"].startswith("documents [4096, 4160)")
    assert p["probe"].startswith("documents [0, 64)")
    assert run.HELDOUT == (4096, 4160) and run.PROBE == (0, 64)
    assert run.CONTROL_CHECKPOINT == 100 and run.CKPT_EVERY == 100
    assert run.THREADS_PER_SEED == 5 and run.DEVICE == "cpu"
    assert " ".join(p["decision_rule"].split()) == run.DECISION_RULE
    assert " ".join(p["question"].split()) == run.QUESTION
    assert pytest.approx(2.7726 - 0.10, abs=1e-4) == run.STREAM_LOSS_THRESHOLD


def test_expected_is_the_preregs_verbatim():
    sec = _prereg_text().split("## Author's expectation")[1].split("\n\n", 1)[1]
    sec = sec.split("\n\n")[0]
    assert " ".join(sec.split()) == run.EXPECTED


def test_measurement_is_the_corpus_size_function():
    """Identity, not a copy: the object the parent calls is the corpus-size curve's
    ``measure_checkpoint``, the same object scaffold-timing measured with."""
    assert run.measure_checkpoint is run.CSC.measure_checkpoint
    assert run.measure_checkpoint is run.ST.measure_checkpoint
    assert run.measure_checkpoint.__code__.co_filename.endswith(
        "experiments/corpus-size-curve/run.py"
    )
    src = (ROOT / "experiments" / "fresh-stream" / "run.py").read_text()
    assert "def measure(" not in src and "def measure_checkpoint(" not in src
    assert "def doc_sets(" not in src


# --------------------------------------------------------------------------- #
# the stream: indexing, generation, disjointness
# --------------------------------------------------------------------------- #


def test_stream_window_indexing():
    s = DocumentStream(offset=4160, stride=16)
    assert s.window(0, 16) == range(4160, 4176)
    assert s.window(1, 16) == range(4176, 4192)
    assert s.window(1000, 16) == range(20160, 20176)
    assert s.window(2999, 16) == range(52144, 52160)
    assert DocumentStream(offset=10, stride=4).window(3, 2) == range(22, 24)
    with pytest.raises(ValueError, match="overlap"):
        s.window(0, 17)
    with pytest.raises(ValueError):
        DocumentStream(offset=-1, stride=16)


@pytest.mark.parametrize("seed", SEEDS)
def test_stream_documents_are_the_generators_by_id(seed):
    """Per id, prefix-stable: the stream's documents at step t are exactly documents
    [offset + stride t, ... + batch) of the seed's corpus."""
    cfg = SyntheticConfig(sentences_per_document=48, seed=seed)
    full = generate(
        SyntheticConfig(sentences_per_document=48, seed=seed, n_documents=260)
    )
    s = DocumentStream(offset=200, stride=16)
    for t in (0, 2):
        got = loop.stream_documents(s, t, 16, cfg)
        assert [d.doc_id for d in got] == list(s.window(t, 16))
        assert to_bytes(got) == to_bytes(full[200 + 16 * t : 216 + 16 * t])


def test_stream_is_disjoint_and_never_repeats():
    d = run.stream_disjointness()
    assert d["ok"] and d["repeats"] == 0 and d["n_ids"] == 16 * 3000
    assert (d["first_id"], d["last_id"]) == (4160, 4160 + 16 * 3000 - 1)
    assert not d["ids_in_probe"] and not d["ids_in_heldout"]
    # A and B read the same window function: at every t in [1000, 3000) they agree
    assert d["first_window_of_B"] == list(range(20160, 20176))


def test_disjointness_catches_overlap(monkeypatch):
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=4000, stride=16))
    d = run.stream_disjointness(iters=20)
    assert not d["ok"] and d["ids_in_heldout"][0] == 4096 and not d["ids_in_probe"]
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=32, stride=16))
    d = run.stream_disjointness(iters=4)
    assert not d["ok"] and d["ids_in_probe"][0] == 32 and not d["ids_in_heldout"]
    monkeypatch.setattr(run, "STREAM", DocumentStream(offset=64, stride=16))
    assert run.stream_disjointness(iters=4)["ok"]


def test_vocabulary_closure_raises_on_an_out_of_vocab_word():
    docs = generate(SyntheticConfig(sentences_per_document=48, seed=0, n_documents=4))
    vocab = loop.build_vocab(docs)
    loop.check_vocabulary_closure(docs, vocab)  # closed: no raise
    word = docs[2].sentences[0].text.split()[0]
    del vocab[word]
    with pytest.raises(ValueError, match="not in the vocabulary"):
        loop.check_vocabulary_closure(docs, vocab)


def test_run_vocabulary_closure_over_a_short_stream():
    c = run.vocabulary_closure(0, iters=3)
    assert c["ok"] and c["n_stream_documents"] == 48
    assert c["n_measurement_documents"] == 128


def test_preflight_reports_a_closure_failure(monkeypatch):
    real = loop.build_vocab

    def short(docs):
        v = real(docs)
        v.pop("the_harbour", None)
        return v

    monkeypatch.setattr(loop, "build_vocab", short)
    pf = run.preflight(iters=2)
    assert not pf["ok"] and "the_harbour" in pf["error"]


# --------------------------------------------------------------------------- #
# the training change: default path unchanged, same init, resume continues
# --------------------------------------------------------------------------- #


class _Stop(Exception):
    pass


def _first_step(tmp_path, monkeypatch, name: str, **kw) -> dict:
    """train() with S0-03's CONFIG up to its first step: the header, the model's
    parameters, the global RNG state and the first batch's ids at that moment."""
    seen: dict = {}

    def stop(model, ids, *a, **k):
        seen["params"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        seen["rng"] = torch.get_rng_state().clone()
        seen["ids"] = ids.clone()
        raise _Stop

    monkeypatch.setattr(loop, "run_policy_loop", stop)
    C = run.S003.CONFIG
    out = tmp_path / name
    with pytest.raises(_Stop):
        loop.train(
            d=C["d"],
            steps_per_stream=C["steps_per_stream"],
            batch=C["batch"],
            vocab=None,
            max_tokens=C["max_tokens"],
            memory_slots=C["memory_slots"],
            iters=300,
            lr=C["lr"],
            seed=kw.pop("seed", 0),
            device="cpu",
            out_dir=out,
            beat_every=1,
            ckpt_every=100,
            policy_name=C["policy"],
            masked_loss=C["masked_loss"],
            srep_norm_reg_weight=C["srep_norm_reg_weight"],
            **kw,
        )
    seen["header"] = json.loads((out / "heartbeat.jsonl").read_text().splitlines()[0])
    return seen


def _s003_config_hash(seed: int) -> str:
    doc = json.loads((ROOT / "runs/s0-03-rewardable-corpus/ledger.json").read_text())
    for c in doc["commands"]:
        if f"--single --seed {seed} " in c["argv"] + " ":
            return c["note"].removeprefix("config_hash ")
    raise AssertionError(f"no S0-03 command for seed {seed}")


def test_default_path_config_is_unchanged(tmp_path, monkeypatch):
    """``stream=None`` is the argument omitted: the same frozen config and hash (S0-03's
    own), nothing stamped; a stream is stamped and never shares the hash."""
    omitted = _first_step(tmp_path, monkeypatch, "omitted")["header"]
    none = _first_step(tmp_path, monkeypatch, "none", stream=None)["header"]
    assert none["config"] == omitted["config"] and "stream" not in none["config"]
    assert none["provenance"] == omitted["provenance"]
    assert none["provenance"]["config_hash"] == _s003_config_hash(0)
    st = _first_step(tmp_path, monkeypatch, "stream", stream=run.STREAM)["header"]
    assert st["config"]["stream"] == {"offset": 4160, "stride": 16, "vocab_documents": 64}
    assert st["provenance"]["config_hash"] != none["provenance"]["config_hash"]
    assert {k: v for k, v in st["config"].items() if k != "stream"} == none["config"]


@pytest.mark.parametrize("seed", [0, 2])
def test_initial_parameters_identical_with_and_without_the_stream(
    tmp_path, monkeypatch, seed
):
    """Arm A's initialisation is the N = 64 arm's for the seed: every parameter, and
    the global RNG state, at the first step. Only the batch differs."""
    a = _first_step(tmp_path, monkeypatch, f"default{seed}", seed=seed)
    b = _first_step(tmp_path, monkeypatch, f"stream{seed}", seed=seed, stream=run.STREAM)
    assert a["params"].keys() == b["params"].keys()
    for k in a["params"]:
        assert torch.equal(a["params"][k], b["params"][k]), k
    assert torch.equal(a["rng"], b["rng"])
    assert a["header"]["config"]["tg"] == b["header"]["config"]["tg"]
    assert not torch.equal(a["ids"], b["ids"])
    # and the stream batch is documents [4160, 4176), encoded with [0, 64)'s map
    cfg = SyntheticConfig(sentences_per_document=48, seed=seed)
    vocab = loop.build_vocab(generate(cfg))
    docs = loop.stream_documents(run.STREAM, 0, 16, cfg)
    ids, _ = loop.encode(docs, vocab, max_tokens=64, steps=48)
    assert torch.equal(b["ids"], ids)


def test_train_default_path_is_bit_identical_with_explicit_none(tmp_path):
    a = loop.train(out_dir=tmp_path / "a", iters=2, ckpt_every=0, seed=1, **TINY)
    b = loop.train(
        out_dir=tmp_path / "b", iters=2, ckpt_every=0, seed=1, stream=None, **TINY
    )
    assert a["final"] == b["final"] and a["config_hash"] == b["config_hash"]


def test_stream_and_n_documents_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="n_documents"):
        loop.train(out_dir=tmp_path, iters=1, n_documents=128, stream=TINY_STREAM, **TINY)


def _record_ids(monkeypatch) -> list:
    got: list = []
    real = loop.run_policy_loop

    def rec(model, ids, *a, **k):
        got.append(ids.clone())
        return real(model, ids, *a, **k)

    monkeypatch.setattr(loop, "run_policy_loop", rec)
    return got


def test_resume_continues_the_stream_at_the_resumed_step(tmp_path, monkeypatch):
    """Resumed at step 2, the run trains on window 2 (not window 0), and ends
    bit-identical to the uninterrupted run."""
    got = _record_ids(monkeypatch)
    full = tmp_path / "full"
    loop.train(out_dir=full, iters=3, ckpt_every=1, seed=0, stream=TINY_STREAM, **TINY)
    straight = list(got)
    got.clear()
    part = tmp_path / "part"
    r = loop.train(
        out_dir=part,
        iters=3,
        ckpt_every=1,
        seed=0,
        stream=TINY_STREAM,
        resume=full / "ckpt-000002.pt",
        **TINY,
    )
    assert len(straight) == 3 and len(got) == 1 and r["steps_done"] == 3
    assert torch.equal(got[0], straight[2])
    cfg = SyntheticConfig(sentences_per_document=48, seed=0)
    vocab = loop.build_vocab(generate(cfg))
    docs = loop.stream_documents(TINY_STREAM, 2, 2, cfg)
    assert [d.doc_id for d in docs] == [204, 205]
    assert torch.equal(got[0], loop.encode(docs, vocab, max_tokens=16, steps=48)[0])
    a = torch.load(full / "ckpt-000003.pt", weights_only=False)
    b = torch.load(part / "ckpt-000003.pt", weights_only=False)
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert a["stream"] == b["stream"] == {**vars(TINY_STREAM), "next_step": 3}


# --------------------------------------------------------------------------- #
# control 2 and the arm calls
# --------------------------------------------------------------------------- #


def test_resume_check_passes_on_a_real_resume(tmp_path):
    loop.train(out_dir=tmp_path / "a", iters=1, ckpt_every=1, seed=0, **TINY)
    ckpt = tmp_path / "a" / "ckpt-000001.pt"
    check = run.ResumeCheck(ckpt, tmp_path / "resume_check.json")
    with check.installed():
        loop.train(
            out_dir=tmp_path / "b",
            iters=2,
            ckpt_every=0,
            seed=0,
            stream=TINY_STREAM,
            resume=ckpt,
            **TINY,
        )
    r = json.loads((tmp_path / "resume_check.json").read_text())
    assert r["ok"] and r["stored_step"] == 1
    assert r["model"]["n_tensors"] > 0 and r["optimizer"]["n_tensors"] > 0
    import rsr.train.checkpoint as ck

    assert loop.run_policy_loop.__name__ == "run_policy_loop"  # uninstalled
    assert ck.load.__name__ == "load"


def test_resume_check_is_exact(tmp_path):
    """A changed parameter or optimizer moment, however small, fails control 2."""
    loop.train(out_dir=tmp_path, iters=1, ckpt_every=1, seed=0, **TINY)
    ckpt = tmp_path / "ckpt-000001.pt"
    payload = torch.load(ckpt, weights_only=False)
    from rsr.model.tg import TGConfig, TGModel

    model = TGModel(TGConfig(**payload_tg(tmp_path)))
    model.load_state_dict(payload["model"])
    from rsr.mup.param_groups import build_param_groups

    groups = build_param_groups(model, None, base_lr=1e-3, d_model=32, base_width=128)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.01)
    opt.load_state_dict(payload["optimizer"])
    assert run.compare_to_checkpoint(ckpt, model, opt)["ok"]
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] += 1e-7
    r = run.compare_to_checkpoint(ckpt, model, opt)
    assert not r["ok"] and not r["model"]["equal"] and r["optimizer"]["equal"]
    model.load_state_dict(payload["model"])
    st = opt.state_dict()
    first = next(iter(st["state"].values()))
    first["exp_avg"].view(-1)[0] += 1e-9
    r = run.compare_to_checkpoint(ckpt, model, opt)
    assert not r["ok"] and r["model"]["equal"] and not r["optimizer"]["equal"]


def payload_tg(out: Path) -> dict:
    h = json.loads((out / "heartbeat.jsonl").read_text().splitlines()[0])
    tg = dict(h["config"]["tg"])
    tg["block_config"] = tuple(tg["block_config"])
    return tg


def test_resume_check_refuses_when_nothing_was_loaded(tmp_path):
    check = run.ResumeCheck(tmp_path / "x.pt", tmp_path / "rc.json")
    with pytest.raises(run.ResumeMismatch):
        check.check()
    assert json.loads((tmp_path / "rc.json").read_text())["ok"] is False


def test_arms_are_the_n64_call_plus_the_stream(tmp_path, monkeypatch):
    """Arm A is the corpus-size N = 64 arm's own ``train()`` call with only the stream
    added; arm B adds the resume from that arm's ckpt 1000."""
    seen = []
    monkeypatch.setattr(loop, "train", lambda **kw: seen.append(kw) or {"ok": 1})
    run.CSC.single(1, 3000, 64, tmp_path)
    run.single("A", 1, tmp_path)
    assert loop.train.__name__ == "<lambda>"  # restored after the call
    ref, a = seen
    assert "n_documents" not in ref and "stream" not in ref
    assert a == {**ref, "stream": run.STREAM}

    class NoCheck:
        def __init__(self, *a):
            self.result = {"ok": True}

        def installed(self):
            import contextlib

            return contextlib.nullcontext()

    monkeypatch.setattr(run, "ResumeCheck", NoCheck)
    run.single("B", 1, tmp_path, source_root=tmp_path / "src")
    b = seen[2]
    want = tmp_path / "src" / "n64" / "seed1" / "ckpt-001000.pt"
    assert b == {**ref, "stream": run.STREAM, "resume": str(want)}


def test_child_argv_and_control_child(tmp_path, monkeypatch):
    argv = run.child_argv("B", 2, tmp_path / "o", tmp_path / "s")
    assert argv[1].endswith("experiments/fresh-stream/run.py")
    assert argv[2:] == [
        "--single",
        "--arm",
        "B",
        "--seed",
        "2",
        "--out-dir",
        str(tmp_path / "o"),
        "--source-root",
        str(tmp_path / "s"),
    ]
    got = []
    monkeypatch.setattr(run.CSC, "Child", lambda *a, **k: got.append((a, k)))
    run.control_child(1, tmp_path)
    assert got == [((1, 64, tmp_path), {"iters": 100})]


# --------------------------------------------------------------------------- #
# synthetic S0-03 tables and the rule
# --------------------------------------------------------------------------- #


def table(r: float = YES, brier: float = C_YES, *, base: float = 0.125) -> dict:
    """An S0-03 measure() result: at gap_2_to_M held-out live = base + r, held-out
    slots_zeroed = base, held-out live Brier16 = ``brier``."""
    ds = {"n_documents": 64, "fraction_of_pairs_gap_gt_M": 0.1}
    for cond in run.S003.CONDITIONS:
        ds[cond] = {"real_token_nll": 1.5}
        for b in run.S003.BUCKETS:
            ds[cond][b] = {
                "n": 10,
                "answer_nll": 2.5,
                "answer_acc": base,
                "answer_nll_over_16": 2.25,
                "answer_brier_over_16": 0.9375,
            }
    ho = copy.deepcopy(ds)
    ho["live"]["gap_2_to_M"]["answer_acc"] = base + r
    ho["live"]["gap_2_to_M"]["answer_brier_over_16"] = brier
    tr = copy.deepcopy(ds)
    tr["live"]["gap_2_to_M"]["answer_acc"] = 0.75
    tr["slots_zeroed"]["gap_2_to_M"]["answer_acc"] = 0.5
    tr["gate_zeroed"]["gap_2_to_M"]["answer_acc"] = 0.5
    tr["gate_zeroed_bos_off"]["gap_2_to_M"]["answer_acc"] = 0.375
    return {"heldout": ho, "train": tr, "memory_gates": [0.25]}


def arm_per(arm: str, r: float = YES, brier: float = C_YES, cells=None) -> dict:
    cells = cells or {}
    return {
        s: {
            c: {"stored_step": c, "s003": table(*cells.get((s, c), (r, brier)))}
            for c in run.CHECKPOINTS[arm]
        }
        for s in SEEDS
    }


def ok_control() -> dict:
    return run.ST.reproduction_control({"k": [1.0, 2.0, 3.0]}, {"k": [1.0, 2.0, 3.0]})


def res_for(per_a: dict, per_b: dict, **over) -> dict:
    res = {
        "preflight": {"ok": True, "error": None},
        "control_1": ok_control(),
        "arms": {"A": {"per": per_a}, "B": {"per": per_b}},
        "resume_check": {s: {"ok": True} for s in SEEDS},
        "error": None,
    }
    res.update(over)
    return res


@pytest.mark.parametrize(
    ("r_a", "r_b", "want"),
    [
        (True, True, "DATA_SUFFICES"),
        (False, True, "SCAFFOLD"),
        (True, False, "TRAP"),
        (False, False, "NEITHER"),
    ],
)
def test_classify_table(r_a, r_b, want):
    assert run.classify(r_a, r_b) == want


@pytest.mark.parametrize(
    ("ra", "rb", "want"),
    [
        (YES, YES, "DATA_SUFFICES"),
        (NO, YES, "SCAFFOLD"),
        (YES, NO, "TRAP"),
        (NO, NO, "NEITHER"),
    ],
)
def test_verdict_all_four_cells_from_tables(ra, rb, want):
    v = run.verdict(res_for(arm_per("A", ra), arm_per("B", rb)))
    assert (v["classification"], v["exit"]) == (want, Exit.OK)
    assert v["R3000"] == {"A": ra == YES, "B": rb == YES}


def test_verdict_reads_ckpt3000_only():
    """R at every earlier checkpoint is irrelevant; one seed failing at 3000 fails."""
    a = arm_per("A", NO, cells={(s, 3000): (YES, C_YES) for s in SEEDS})
    b = arm_per("B", YES, cells={(1, 3000): (NO, C_YES)})
    v = run.verdict(res_for(a, b))
    assert v["classification"] == "TRAP" and v["R3000"] == {"A": True, "B": False}


def test_usable_line_needs_R_and_C():
    a = arm_per("A", YES, C_YES)
    b = arm_per("B", YES, C_NO)
    v = run.verdict(res_for(a, b))
    assert v["classification"] == "DATA_SUFFICES"
    assert v["usable"] == {"A": True, "B": False}
    assert v["C3000"] == {"A": True, "B": False}


def test_R_and_C_thresholds():
    assert run.r_holds([run.DELTA] * 3) and not run.r_holds([run.DELTA - 1e-9] * 3)
    assert not run.r_holds([YES, YES])
    assert run.c_holds([run.C_THRESHOLD] * 3)
    assert not run.c_holds([run.C_THRESHOLD + 1e-9] * 3)
    assert not run.c_holds([C_YES, C_YES])


def test_C_needs_every_seed():
    assert not run.c_holds([C_YES, C_NO, C_YES])
    b = arm_per("B", YES, cells={(2, 3000): (YES, C_NO)})
    v = run.verdict(res_for(arm_per("A"), b))
    assert v["C3000"]["B"] is False and v["usable"]["B"] is False


@pytest.mark.parametrize(
    "case",
    [
        "error",
        "preflight",
        "control_1_failed",
        "control_1_absent",
        "control_2_failed",
        "control_2_missing",
        "A_missing_3000",
        "B_missing_3000",
        "B_never_ran",
    ],
)
def test_inconclusive_rows_exit_3(case):
    a, b = arm_per("A"), arm_per("B")
    res = res_for(a, b)
    if case == "error":
        res["error"] = "boom"
    elif case == "preflight":
        res["preflight"] = {"ok": False, "error": "word 'x' not in the vocabulary"}
    elif case == "control_1_failed":
        res["control_1"] = run.ST.reproduction_control({"k": [1.0] * 3}, {"k": [2.0] * 3})
    elif case == "control_1_absent":
        res["control_1"] = None
    elif case == "control_2_failed":
        res["resume_check"][1] = {"ok": False}
    elif case == "control_2_missing":
        res["resume_check"][2] = None
    elif case == "A_missing_3000":
        del a[0][3000]
    elif case == "B_missing_3000":
        del b[2][3000]
    elif case == "B_never_ran":
        del res["arms"]["B"]
        res["resume_check"] = {s: None for s in SEEDS}
    v = run.verdict(res)
    assert (v["classification"], v["exit"]) == ("inconclusive", Exit.DID_NOT_RUN), case


def test_quantities():
    t = table(YES, C_YES)
    assert run.r_quantity(t) == YES and run.c_quantity(t) == C_YES
    assert run.probe_r_quantity(t) == 0.25


# --------------------------------------------------------------------------- #
# control 1
# --------------------------------------------------------------------------- #


def _control_per() -> dict:
    return {s: {100: {"stored_step": 100, "s003": table(0.01 * s, 1.25)}} for s in SEEDS}


def test_control_1_reference_keys_are_the_scaffold_ledgers():
    """Scaffold-timing's own arm_rows, fed the re-trained ckpt-100 tables, produces
    exactly the statistic keys its committed ledger holds under n64.ckpt100."""
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    ref = run.reference_rows(doc)
    got = run.remeasured_rows(_control_per())
    assert sorted(got) == sorted(ref) and len(ref) > 200
    assert all(k.startswith("n64.ckpt100.") for k in ref)
    assert "n64.ckpt100.R_quantity" in ref


def test_control_1_exact_passes_and_inside_tolerance_passes():
    ref = run.remeasured_rows(_control_per())
    assert run.control_1(ref, _control_per())["ok"]
    near = copy.deepcopy(ref)
    k = sorted(near)[5]
    near[k][1] += 5e-7
    assert run.control_1(near, _control_per())["ok"]


def test_control_1_fails_at_2e_6():
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[7]
    ref[k][2] += 2e-6
    c = run.control_1(ref, _control_per())
    assert not c["ok"] and c["tolerance"] == 1e-6
    assert c["per_seed"][2]["ok"] is False and c["per_seed"][0]["ok"]
    assert c["per_seed"][2]["failures"][0]["key"] == k


# --------------------------------------------------------------------------- #
# the parent loop
# --------------------------------------------------------------------------- #


class FakeChild:
    """Writes the listed checkpoints up to ``upto`` at once (and arm B's
    resume_check.json), then exits ``rc`` -- or stays alive if ``alive``."""

    def __init__(self, d: Path, labels, upto=3000, rc=0, alive=False, resume=None):
        d.mkdir(parents=True, exist_ok=True)
        for c in labels:
            if c <= upto:
                run.RC.ckpt_path(d, c).write_bytes(b"x")
        if resume is not None:
            (d / "resume_check.json").write_text(json.dumps({"ok": resume}))
        self.argv = ["py", "run.py", "--single"]
        self.rc, self.alive = rc, alive

    def poll(self):
        return None if self.alive else self.rc

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return self.rc


def fake_measure(r_a=YES, r_b=NO):
    calls = []

    def m(seed_dir, seed, label, n):
        arm = seed_dir.parent.name
        calls.append((arm, seed, label, n))
        if arm == "control":
            return _control_per()[seed][label]
        r = r_a if arm == "A" else r_b
        return {"stored_step": label, "s003": table(r, C_YES)}

    return m, calls


def _kw(measure_fn, *, reference=None, upto=None, resume=True, clock=None, **extra):
    upto = upto or {}
    spawned = []

    def spawn(arm, s, d):
        spawned.append((arm, s))
        return FakeChild(
            d,
            run.CHECKPOINTS[arm],
            upto=upto.get((arm, s), 3000),
            alive=(arm, s) in upto,
            resume=resume if arm == "B" else None,
        )

    kw = {
        "preflight_fn": lambda: {"ok": True, "error": None, "closure": {}},
        "spawn_control": lambda s, d: FakeChild(d, (100,)),
        "spawn": spawn,
        "measure_fn": measure_fn,
        "reference": reference
        if reference is not None
        else run.remeasured_rows(_control_per()),
        "deadline": 1e9,
        "clock": clock or (lambda: 0.0),
        "sleep": lambda s: None,
        "log": lambda s: None,
        **extra,
    }
    return kw, spawned


def test_run_all_happy_path_order_and_scaffold(tmp_path):
    m, calls = fake_measure(r_a=NO, r_b=YES)
    kw, spawned = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert spawned == [("A", s) for s in SEEDS] + [("B", s) for s in SEEDS]
    assert [c for c in calls if c[0] == "control"] == [
        ("control", s, 100, 64) for s in SEEDS
    ]
    first_a = next(i for i, c in enumerate(calls) if c[0] == "A")
    assert all(c[0] == "control" for c in calls[:first_a])
    assert {c[3] for c in calls} == {64}  # every measurement with n = 64
    assert res["control_1"]["ok"] and res["resume_check"][0] == {"ok": True}
    v = run.verdict(res)
    assert (v["classification"], v["exit"]) == ("SCAFFOLD", Exit.OK)


def test_control_1_failure_stops_before_any_arm(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[0]
    ref[k] = [x + 2e-6 for x in ref[k]]
    m, calls = fake_measure()
    kw, spawned = _kw(m, reference=ref)
    res = run.run_all(tmp_path, **kw)
    assert spawned == [] and all(c[0] == "control" for c in calls)
    assert res["not_run"] == {"A": res["stopped"], "B": res["stopped"]}
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


def test_preflight_failure_trains_nothing(tmp_path):
    m, calls = fake_measure()
    kw, spawned = _kw(m)
    trained = []
    kw["spawn_control"] = lambda s, d: trained.append(s)
    kw["preflight_fn"] = lambda: {"ok": False, "error": "word 'x' not in the vocabulary"}
    res = run.run_all(tmp_path, **kw)
    assert trained == [] and spawned == [] and calls == []
    v = run.verdict(res)
    assert v["exit"] == Exit.DID_NOT_RUN and "vocabulary" in v["detail"]


def test_deadline_stops_and_missing_ckpt3000_is_inconclusive(tmp_path):
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]

    m, _ = fake_measure()
    kw, _ = _kw(m, upto={("A", 1): 2000}, clock=clock)
    kw["deadline"] = 30.0
    res = run.run_all(tmp_path, **kw)
    assert res["arms"]["A"]["stopped"] == "deadline"
    assert sorted(res["arms"]["A"]["per"][1]) == [300, 1000, 1500, 2000]
    assert res["not_run"].get("B") == "deadline" and "B" not in res["arms"]
    v = run.verdict(res)
    assert v["classification"] == "inconclusive" and "A.seed1" in v["detail"]


def test_failed_resume_check_is_inconclusive(tmp_path):
    m, _ = fake_measure(r_a=YES, r_b=YES)
    kw, _ = _kw(m, resume=False)
    v = run.verdict(run.run_all(tmp_path, **kw))
    assert v["classification"] == "inconclusive" and "control_2" in v["detail"]


def test_measurement_raised_stops_later_arms(tmp_path):
    def m(seed_dir, seed, label, n):
        if seed_dir.parent.name == "A" and label == 1500:
            raise RuntimeError("bad table")
        return fake_measure()[0](seed_dir, seed, label, n)

    kw, _ = _kw(m)
    res = run.run_all(tmp_path, **kw)
    assert "bad table" in res["error"] and "B" not in res["arms"]
    assert run.verdict(res)["exit"] == Exit.DID_NOT_RUN


# --------------------------------------------------------------------------- #
# the stream loss from the heartbeats
# --------------------------------------------------------------------------- #


def test_stream_loss_windows_and_first_window_below(tmp_path):
    hb = tmp_path / "heartbeat.jsonl"
    lines = [json.dumps({"kind": "header", "start_step": 1000})]
    for s in range(1000, 1250):
        loss = 2.78 if s < 1100 else 2.5
        lines.append(json.dumps({"kind": "beat", "step": s, "loss_answer_tokens": loss}))
    lines.append("{truncated")
    hb.write_text("\n".join(lines) + "\n")
    w = run.stream_loss_windows(hb)
    assert sorted(w) == [1000, 1100]  # 1200..1249 is incomplete
    assert w[1000] == pytest.approx(2.78) and w[1100] == pytest.approx(2.5)
    assert run.first_window_below(w) == 1100
    assert run.first_window_below({0: 2.7}) is None
    assert run.stream_loss_windows(tmp_path / "absent.jsonl") == {}


# --------------------------------------------------------------------------- #
# the ledger, RESULTS.md, the manifest, the wrapper
# --------------------------------------------------------------------------- #

START = {
    "R_quantity": [0.0625, 0.125, 0.0625],
    "train.live.gap_2_to_M.answer_acc": [1.0, 1.0, 1.0],
    "train.slots_zeroed.gap_2_to_M.answer_acc": [0.5, 0.5, 0.5],
}


def _ledger_doc(tmp_path, res: dict) -> dict:
    import ledger as ledger_mod

    led = ledger_mod.Ledger(run.RUN_ID, question=run.QUESTION, runs_root=tmp_path)
    led.run_meta(device="cpu", seeds_actually_run=SEEDS)
    led.status("ok")
    run.write_rows(led, res, run.verdict(res), START)
    return led.doc


def _audit(tmp_path, doc, page) -> list:
    import render_scoreboard as rs

    (tmp_path / run.RUN_ID).mkdir(exist_ok=True)
    (tmp_path / run.RUN_ID / "ledger.json").write_text(json.dumps(doc))
    return rs.audit_prose(tmp_path, page)


def test_scaffold_start_reads_the_committed_ledger():
    doc = json.loads((ROOT / run.REF_LEDGER).read_text())
    s = run.scaffold_start(doc)
    assert s["R_quantity"] == pytest.approx([0.0609065, 0.1121884, 0.0720339], abs=1e-6)
    assert len(s["train.live.gap_2_to_M.answer_acc"]) == 3


def test_render_results_passes_the_audit(tmp_path):
    m, _ = fake_measure(r_a=NO, r_b=YES)
    kw, _ = _kw(m)
    res = run.run_all(tmp_path, **kw)
    res["root"] = str(tmp_path)
    res["preflight"] = {
        "ok": True,
        "error": None,
        "disjointness": run.stream_disjointness(iters=4),
        "closure": {
            s: {
                "ok": True,
                "n_stream_documents": 48000,
                "n_measurement_documents": 128,
                "vocab_words": 172,
            }
            for s in SEEDS
        },
    }
    hb = tmp_path / "A" / "seed0" / "heartbeat.jsonl"
    hb.write_text(
        "\n".join(
            json.dumps({"kind": "beat", "step": s, "loss_answer_tokens": 2.5})
            for s in range(300, 400)
        )
        + "\n"
    )
    doc = _ledger_doc(tmp_path, res)
    rows = {r["key"]: r for r in doc["rows"]}
    assert rows["classification"]["value"] == "SCAFFOLD"
    assert rows["A.R3000"]["value"] is False and rows["B.usable"]["value"] is True
    assert rows["A.stream_answer_loss_first_window_below"]["value"]["seed0"] == 300
    assert rows["B.growth_R_quantity"]["samples"] == pytest.approx(
        [YES - x for x in START["R_quantity"]]
    )
    assert rows["control_2.ok"]["value"] == [True] * 3
    page = run.render_results(doc, "none")
    assert "`classification` **SCAFFOLD**" in page and "not measured" not in page
    assert _audit(tmp_path, doc, page) == []


def test_render_results_after_a_failed_control_passes_the_audit(tmp_path):
    ref = run.remeasured_rows(_control_per())
    k = sorted(ref)[3]
    ref[k] = [x + 2e-6 for x in ref[k]]
    m, _ = fake_measure()
    kw, _ = _kw(m, reference=ref)
    res = run.run_all(tmp_path, **kw)
    doc = _ledger_doc(tmp_path, res)
    page = run.render_results(doc, None)
    assert "**inconclusive**" in page and k in page and "not measured" in page
    assert _audit(tmp_path, doc, page) == []


def test_manifest_records_the_start_checkpoints():
    m = run.manifest(
        Path("/src"), 12, "abc", {"seed0": "s0", "seed1": "s1", "seed2": "s2"}
    )
    assert m["arm_B_start_checkpoint_sha256"] == {
        "seed0": "s0",
        "seed1": "s1",
        "seed2": "s2",
    }
    assert m["stream"] == {"offset": 4160, "stride": 16, "vocab_documents": 64}
    assert m["stream_documents"] == [4160, 52160] and m["expected"] == run.EXPECTED
    assert m["prereg_commit"] == "83128ee" and m["deadline"] == run.DEADLINE
    assert json.dumps(m, default=str)


def test_dry_run_trains_and_measures_nothing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ran")

    monkeypatch.setattr(run.CSC, "measure_checkpoint", boom)
    monkeypatch.setattr(loop, "train", boom)
    text = run.dry_run(tmp_path, now=0.0)
    assert "NOTHING TRAINED OR MEASURED" in text and text.count("exists=False") == 3
    assert "repeats 0, disjoint True" in text


def test_run_outputs_are_gitignored_and_the_wrapper_writes_run_rc():
    for rel in (
        "runs/fresh-stream/A/seed0/ckpt-003000.pt",
        "runs/fresh-stream/B/seed2/heartbeat.jsonl",
        "runs/fresh-stream/run.rc",
    ):
        r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=ROOT)
        assert r.returncode == 0, rel
    r = subprocess.run(
        ["git", "check-ignore", "-q", "runs/fresh-stream/ledger.json"], cwd=ROOT
    )
    assert r.returncode == 1
    sh = ROOT / "experiments" / "fresh-stream" / "run.sh"
    assert sh.stat().st_mode & 0o111
    text = sh.read_text()
    assert "rc=$?" in text and "runs/fresh-stream/run.rc" in text
