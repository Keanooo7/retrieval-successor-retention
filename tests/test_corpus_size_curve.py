"""The corpus-size curve's own gates (`experiments/corpus-size-curve/PREREG.md`, brief
`docs/lab-notes/dispatch-corpus-size-curve.md` *Bar*).

Nothing here trains past a header. The parent loop is driven by fake children over
checkpoint files touched on disk and a fake clock; the rule is fed synthetic S0-03
tables. Mutation-battery gates (`scripts/mutation_battery.py`,
``# --- corpus-size-curve ---``):

* ``test_default_path_config_hash_is_s003s`` -- n_documents stamped on the default path;
* ``test_heldout_is_disjoint_from_every_arm`` -- held-out drawn from training docs;
* ``test_brier16_on_a_known_distribution`` -- the Brier readout miscomputed;
* ``test_verdict_reads_the_n4096_arm_only`` -- the verdict read off another arm;
* ``test_control_reads_s003s_own_heldout`` -- the control read on the common set;
* ``test_control_arm_omits_n_documents`` -- the control arm passes n_documents;
* ``test_reproduction_control_tolerance`` -- the tolerance widened.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "corpus_size_curve_run", ROOT / "experiments" / "corpus-size-curve" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]


def _prereg() -> dict:
    text = (ROOT / "experiments" / "corpus-size-curve" / "PREREG.md").read_text()
    return yaml.safe_load(text.split("---")[1])


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    p = _prereg()
    t = p["thresholds"]
    assert list(run.ARMS) == p["arms_n_documents"] == [64, 4096, 512]
    assert run.SEEDS == p["seeds"] == run.S003.SEEDS
    assert list(run.CHECKPOINTS) == t["checkpoints"] and t["iters_per_arm"] == run.ITERS
    assert t["BRIER_MARGIN"].startswith(f"{run.BRIER_MARGIN} ")
    assert t["reproduction_tolerance"].startswith("1e-6 ") and run.REPRO_TOL == 1e-6
    assert t["deadline"].startswith("2026-09-25 07:30 America/Los_Angeles")
    assert run.DEADLINE == "2026-09-25T07:30:00-07:00"  # PDT on that date
    assert run.MARGIN == run.S003.MARGIN == 0.10 and run.CHANCE == run.S003.CHANCE
    assert run.HELDOUT == (4096, 4160) and run.PROBE == (0, 64)
    assert p["common_heldout"].startswith("documents [4096, 4160)")
    assert run.PRIMARY_ARM == 4096 and run.CONTROL_ARM == 64
    assert run.CONTROL_CHECKPOINT == 300 and run.CKPT_EVERY == 100
    assert all(c % run.CKPT_EVERY == 0 for c in run.CHECKPOINTS)
    assert run.THREADS_PER_SEED == 5 and run.DEVICE == "cpu"
    assert pytest.approx(0.9375) == run.BRIER16_UNIFORM


def test_decide_and_measure_are_imported_not_copied():
    src = (ROOT / "experiments" / "corpus-size-curve" / "run.py").read_text()
    assert "def decide(" not in src and "def measure(" not in src
    assert "def answer_readout(" not in src
    assert run.S003.decide.__code__.co_filename.endswith("s0-03-rewardable-corpus/run.py")


# --------------------------------------------------------------------------- #
# the one factor: n_documents, default path unchanged
# --------------------------------------------------------------------------- #


class _Stop(Exception):
    pass


def _header(tmp_path, monkeypatch, seed: int, **kw) -> dict:
    """train() up to its heartbeat header (written before the first iteration)."""
    import rsr.train.loop as loop

    def boom(*a, **k):
        raise _Stop

    monkeypatch.setattr(loop, "run_policy_loop", boom)
    C = run.S003.CONFIG
    out = tmp_path / f"s{seed}-{kw.get('n_documents')}"
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
            seed=seed,
            device="cpu",
            out_dir=out,
            beat_every=1,
            ckpt_every=300,
            policy_name=C["policy"],
            masked_loss=C["masked_loss"],
            srep_norm_reg_weight=C["srep_norm_reg_weight"],
            **kw,
        )
    return json.loads((out / "heartbeat.jsonl").read_text().splitlines()[0])


def test_default_path_config_hash_is_s003s(tmp_path, monkeypatch):
    """train() without n_documents hashes to exactly S0-03's per-seed config_hash
    (its ledger's command notes), and a set n_documents never shares it."""
    doc = json.loads((ROOT / "runs/s0-03-rewardable-corpus/ledger.json").read_text())
    ref = {}
    for c in doc["commands"]:
        if "--single --seed" in c["argv"]:
            seed = int(c["argv"].split("--seed ")[1].split()[0])
            ref[seed] = c["note"].removeprefix("config_hash ")
    assert sorted(ref) == SEEDS
    for s in SEEDS:
        h = _header(tmp_path, monkeypatch, s)
        assert h["provenance"]["config_hash"] == ref[s]
        assert h["provenance"]["documents"] == 64
        assert "n_documents" not in h["config"]
    big = _header(tmp_path, monkeypatch, 0, n_documents=4096)
    assert big["provenance"]["documents"] == 4096
    assert big["config"]["n_documents"] == 4096
    assert big["provenance"]["config_hash"] != ref[0]


def test_default_corpus_call_is_unchanged(tmp_path, monkeypatch):
    """Both of train()'s generate() calls get SyntheticConfig's own default."""
    import rsr.train.loop as loop
    from rsr.data.synthetic import SyntheticConfig

    seen = []
    real = loop.generate
    monkeypatch.setattr(loop, "generate", lambda cfg=None: seen.append(cfg) or real(cfg))
    _header(tmp_path, monkeypatch, 1)
    assert seen == [SyntheticConfig(sentences_per_document=48, seed=1)] * 2
    seen.clear()
    _header(tmp_path, monkeypatch, 1, n_documents=512)
    assert [c.n_documents for c in seen] == [512, 512]


def test_control_arm_omits_n_documents(tmp_path, monkeypatch):
    import rsr.train.loop as loop

    seen = []
    monkeypatch.setattr(loop, "train", lambda **kw: seen.append(kw) or {"ok": 1})
    for n in run.ARMS:
        run.single(2, run.ITERS, n, tmp_path)
    assert "n_documents" not in seen[0]  # N = 64: the default path
    assert [kw.get("n_documents") for kw in seen[1:]] == [4096, 512]
    C = run.S003.CONFIG
    for kw in seen:
        assert kw["iters"] == 1000 and kw["ckpt_every"] == 100 and kw["seed"] == 2
        for k in ("d", "steps_per_stream", "batch", "max_tokens", "memory_slots", "lr"):
            assert kw[k] == C[k], k
        assert kw["masked_loss"] is True and kw["srep_norm_reg_weight"] == 0.0
        assert kw["policy_name"] == "fifo" and kw["device"] == "cpu"


def test_child_argv_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    env = run.RC.child_env()
    assert env["OMP_NUM_THREADS"] == "5" and env["MKL_NUM_THREADS"] == "5"
    argv = run.child_argv(1, 1000, 4096, tmp_path)
    assert argv[1].endswith("experiments/corpus-size-curve/run.py")
    assert argv[2:] == [
        "--single",
        "--seed",
        "1",
        "--iters",
        "1000",
        "--n-documents",
        "4096",
        "--out-dir",
        str(tmp_path),
    ]


# --------------------------------------------------------------------------- #
# document sets
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", SEEDS)
def test_generate_is_prefix_stable(seed):
    from rsr.data.synthetic import to_bytes

    full = run._generate(seed, run.HELDOUT[1])
    for n in run.ARMS:
        assert to_bytes(full[:n]) == to_bytes(run._generate(seed, n)), n


@pytest.mark.parametrize("seed", SEEDS)
def test_heldout_is_disjoint_from_every_arm(seed):
    """Held-out is documents [4096, 4160) in every arm: disjoint from each arm's
    training documents [0, N) by index AND by content, identical across arms."""
    from rsr.data.synthetic import to_bytes

    sets = {n: run.doc_sets(seed, n)[0] for n in run.ARMS}
    ids = {n: {d.doc_id for d in s["heldout"]} for n, s in sets.items()}
    for n in run.ARMS:
        assert ids[n] == set(range(*run.HELDOUT))
        assert ids[n].isdisjoint(range(n))
        train_bodies = {
            to_bytes((d,)).split(b'"pairs"')[1] for d in run._generate(seed, n)
        }
        assert all(
            to_bytes((d,)).split(b'"pairs"')[1] not in train_bodies
            for d in sets[n]["heldout"]
        )
    assert len({to_bytes(s["heldout"]) for s in sets.values()}) == 1
    assert len({to_bytes(s["train"]) for s in sets.values()}) == 1
    assert [d.doc_id for d in sets[4096]["train"]] == list(range(64))
    with pytest.raises(ValueError):
        run.doc_sets(seed, 4097)


@pytest.mark.parametrize("seed", SEEDS)
def test_vocab_is_identical_across_arms(seed):
    """So V and every token id are shared: only the documents differ."""
    ref = run.S003._sets(seed)
    for n in run.ARMS:
        _, vmap, V = run.doc_sets(seed, n)
        assert vmap == ref[1] and ref[2] == V


def test_n64_probe_is_s003s_train_set():
    ref = run.S003._sets(0)[0]["train"]
    assert run.doc_sets(0, 64)[0]["train"] == ref


# --------------------------------------------------------------------------- #
# Brier16
# --------------------------------------------------------------------------- #


class _Stub(torch.nn.Module):
    """A model whose logits at every position are ``row``: the answer_readout
    loop runs with memory and bos-copy off."""

    def __init__(self, row: torch.Tensor):
        super().__init__()
        self.embed = torch.nn.Embedding(1, 4)
        self.cfg = SimpleNamespace(
            M=2, D=4, use_memory=False, bos_replacement_mode="none"
        )
        self.row = row

    def forward(self, ids_t, mask_t, kv, valid, bos_ctx, bv):
        B, L = ids_t.shape
        return SimpleNamespace(
            logits=self.row.expand(B, L, -1).clone(),
            srep=torch.zeros(B, 4),
            has_eos=torch.zeros(B, dtype=torch.bool),
        )


@pytest.mark.parametrize(
    ("target", "want"),
    [(2, 0.3**2 + 3 * 0.1**2), (3, 0.7**2 + 0.9**2 + 2 * 0.1**2)],
)
def test_brier16_on_a_known_distribution(target, want):
    """Symbols 2..5 carry probability .7/.1/.1/.1 after renormalisation over the
    symbols (the non-symbol logits 0, 1 must not enter)."""
    sym = torch.tensor([2, 3, 4, 5])
    row = torch.tensor([9.0, -3.0, *torch.log(torch.tensor([0.7, 0.1, 0.1, 0.1]))])
    ids = torch.tensor([[[0, 0, target]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    tmask = torch.zeros(1, 1, 3, dtype=torch.bool)
    tmask[0, 0, 2] = True
    gap = torch.tensor([[3]])
    r = run.S003.answer_readout(_Stub(row), ids, mask, tmask, gap, sym, cond="live")
    assert r["brier16"].shape == (1,)
    assert float(r["brier16"][0]) == pytest.approx(want, abs=1e-6)
    s = run.S003._summarise(r)
    assert s["gap_2_to_M"]["answer_brier_over_16"] == pytest.approx(want, abs=1e-6)


def test_brier16_uniform_is_the_reference():
    sym = torch.arange(4, 20)
    row = torch.zeros(20)
    ids = torch.tensor([[[0, 0, 7]]])
    tmask = torch.zeros(1, 1, 3, dtype=torch.bool)
    tmask[0, 0, 2] = True
    r = run.S003.answer_readout(
        _Stub(row),
        ids,
        torch.ones(1, 1, 3, dtype=torch.bool),
        tmask,
        torch.tensor([[2]]),
        sym,
        cond="live",
    )
    assert float(r["brier16"][0]) == pytest.approx(run.BRIER16_UNIFORM)


def test_measure_default_sets_are_s003s(monkeypatch):
    """measure() without ``sets`` reads S0-03's own _sets(seed): the control path."""
    calls = []

    def fake_sets(seed):
        calls.append(seed)
        raise _Stop

    monkeypatch.setattr(run.S003, "_sets", fake_sets)
    with pytest.raises(_Stop):
        run.S003.measure(Path("x.pt"), 2, "cpu")
    assert calls == [2]


# --------------------------------------------------------------------------- #
# synthetic tables for the rule
# --------------------------------------------------------------------------- #


def s003_table(bgap: float, *, seed: int = 0, bar1: bool = True) -> dict:
    """An S0-03 measure() result with Brier16. ``bgap`` = slots_zeroed - live
    Brier16 at gap_2_to_M on held-out. train is memorised (a huge gap)."""
    base_nll = run.CHANCE + 0.01 * (seed + 1)
    zero_nll = base_nll if bar1 else run.CHANCE - 0.5
    ds = {"n_documents": 64, "fraction_of_pairs_gap_gt_M": 0.1}
    for cond in run.S003.CONDITIONS:
        ds[cond] = {"real_token_nll": 1.5}
        for b in run.S003.BUCKETS:
            nll = base_nll
            if cond in ("slots_zeroed", "gate_zeroed") and b == "gap_ge_2":
                nll = zero_nll
            brier = 0.9
            if cond == "slots_zeroed" and b == "gap_2_to_M":
                brier = 0.9 + bgap
            if cond == "live" and b == "gap_gt_M":
                nll = base_nll + 0.05
            ds[cond][b] = {
                "n": 10,
                "answer_nll": nll,
                "answer_acc": 0.06,
                "answer_nll_over_16": nll - 0.2,
                "answer_brier_over_16": brier,
            }
    tr = copy.deepcopy(ds)
    tr["live"]["gap_2_to_M"]["answer_brier_over_16"] = 0.0
    tr["slots_zeroed"]["gap_2_to_M"]["answer_brier_over_16"] = 1.9
    return {"heldout": ds, "train": tr, "memory_gates": [0.1]}


def arm_per(bgap: dict[int, float], labels=run.CHECKPOINTS, bar1=None, seedgap=None):
    bar1 = bar1 or {}
    seedgap = seedgap or {}
    return {
        s: {
            c: {
                "stored_step": c,
                "s003": s003_table(
                    seedgap.get((s, c), bgap[c]), seed=s, bar1=bar1.get(c, True)
                ),
            }
            for c in labels
        }
        for s in SEEDS
    }


def arm(bgap, **kw) -> dict:
    return {"per": arm_per(bgap, **kw), "stopped": None}


def ok_control() -> dict:
    return {
        s: {"seed": s, "measured": 2.8, "reference": 2.8, "abs_diff": 0.0, "ok": True}
        for s in SEEDS
    }


YES, NO = 0.10, 0.01


# --------------------------------------------------------------------------- #
# the decision table, row by row
# --------------------------------------------------------------------------- #


def test_row_measurement_raised():
    v = run.verdict({4096: arm({300: YES, 1000: YES})}, ok_control(), "boom")
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


def test_row_control_failed_overrides_everything():
    c = ok_control()
    c[1] = {**c[1], "ok": False, "abs_diff": 1e-3}
    v = run.verdict({4096: arm({300: YES, 1000: YES})}, c)
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN
    assert "seed1" in v["detail"]


def test_row_control_incomplete():
    c = ok_control()
    del c[2]
    v = run.verdict({4096: arm({300: YES, 1000: YES})}, c)
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


@pytest.mark.parametrize(
    ("at", "gaps"), [(1000, {300: NO, 1000: YES}), (300, {300: YES, 1000: NO})]
)
def test_row_falsified_at_any_checkpoint(at, gaps):
    v = run.verdict({4096: arm(gaps)}, ok_control())
    assert v["outcome"] == "falsified" and v["falsified_at"] == [at]
    assert v["exit"] == Exit.OK


def test_B_needs_every_seed():
    per = arm({300: NO, 1000: YES}, seedgap={(2, 1000): 0.049})
    v = run.verdict({4096: per}, ok_control())
    assert v["outcome"] == "survived"
    per = arm({300: NO, 1000: YES}, seedgap={(2, 1000): 0.05})
    assert run.verdict({4096: per}, ok_control())["outcome"] == "falsified"


def test_row_survived():
    v = run.verdict({4096: arm({300: NO, 1000: NO})}, ok_control())
    assert v["outcome"] == "survived" and v["exit"] == Exit.OK


def test_row_bar1_fails_where_B_would_be_read():
    v = run.verdict(
        {4096: arm({300: YES, 1000: YES}, bar1={1000: False, 300: False})}, ok_control()
    )
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.OK


def test_row_stopped_before_ckpt1000():
    a = {"per": arm_per({300: NO}, labels=(300,)), "stopped": "deadline"}
    v = run.verdict({4096: a}, ok_control())
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN
    assert "deadline" in v["detail"]
    v = run.verdict({}, ok_control())
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


def test_verdict_reads_the_n4096_arm_only():
    """N = 64 and N = 512 show B; N = 4096 does not: survived."""
    arms = {
        64: arm({300: YES, 1000: YES}),
        4096: arm({300: NO, 1000: NO}),
        512: arm({300: YES, 1000: YES}),
    }
    assert run.verdict(arms, ok_control())["outcome"] == "survived"
    arms[4096] = arm({300: NO, 1000: YES})
    arms[64] = arm({300: NO, 1000: NO})
    assert run.verdict(arms, ok_control())["outcome"] == "falsified"


def test_rule_is_fed_heldout_not_train():
    """Train is memorised (Brier gap 1.9 there); held-out has none: survived."""
    rule = run.arm_rule(arm_per({300: NO, 1000: NO}))
    assert not any(r["B"] for r in rule.values())
    assert all(g == pytest.approx(NO) for r in rule.values() for g in r["brier_gap"])


# --------------------------------------------------------------------------- #
# the reproduction control and the parent loop
# --------------------------------------------------------------------------- #


def test_reproduction_control_tolerance():
    ref = {0: 2.8}
    at = {"heldout": {"live": {"gap_2_to_M": {"answer_nll": 2.8 + 9e-7}}}}
    off = {"heldout": {"live": {"gap_2_to_M": {"answer_nll": 2.8 + 2e-6}}}}
    assert run.RC.reproduction_control(0, at, ref)["ok"]
    assert run.REPRO_TOL == 1e-6 == run.RC.REPRO_TOL
    r = run.RC.reproduction_control(0, off, ref)
    assert not r["ok"] and r["tolerance"] == run.REPRO_TOL


class FakeChild:
    """Writes the checkpoints up to ``upto`` at once, then exits 0."""

    def __init__(self, seed, n, d, upto=1000):
        d.mkdir(parents=True, exist_ok=True)
        for c in run.CHECKPOINTS:
            if c <= upto:
                run.RC.ckpt_path(d, c).write_bytes(b"x")
        self.argv = ["py", "run.py", "--single"]
        self.rc = 0

    def poll(self):
        return self.rc

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return self.rc


def fake_measure(nll_default: float = 2.8, bgap: float = NO, common_nll=None):
    """``common_nll`` is the COMMON set's live gap_2_to_M NLL; by default equal to
    the reference, so only the test that sets it can tell the two populations
    apart."""
    calls = []

    def m(seed_dir, seed, label, n):
        calls.append((n, seed, label))
        rec = {"stored_step": label, "s003": s003_table(bgap, seed=seed)}
        live = rec["s003"]["heldout"]["live"]["gap_2_to_M"]
        live["answer_nll"] = nll_default if common_nll is None else common_nll
        if n == run.CONTROL_ARM and label == run.CONTROL_CHECKPOINT:
            t = s003_table(bgap, seed=seed)
            t["heldout"]["live"]["gap_2_to_M"]["answer_nll"] = nll_default
            rec["s003_default"] = t
        return rec

    return m, calls


def _kw(measure_fn, clock=lambda: 0.0, deadline=1e9):
    return {
        "spawn": lambda s, n, d: FakeChild(s, n, d),
        "measure_fn": measure_fn,
        "reference": {s: 2.8 for s in SEEDS},
        "deadline": deadline,
        "clock": clock,
        "sleep": lambda _: None,
        "log": lambda _: None,
    }


def test_run_all_runs_arms_in_order_and_reaches_a_verdict(tmp_path):
    m, calls = fake_measure()
    res = run.run_all(tmp_path, **_kw(m))
    assert list(res["arms"]) == [64, 4096, 512] and res["not_run"] == {}
    assert [c[0] for c in calls] == [64] * 6 + [4096] * 6 + [512] * 6
    assert sorted(res["control"]) == SEEDS and all(
        c["ok"] for c in res["control"].values()
    )
    assert run.verdict(res["arms"], res["control"], res["error"])["outcome"] == "survived"


def test_control_reads_s003s_own_heldout(tmp_path):
    """The common set's live NLL differs from S0-03's; the control must read the
    ``s003_default`` measurement (no override), which matches."""
    m, _ = fake_measure(nll_default=2.8, common_nll=3.9)
    res = run.run_all(tmp_path, **_kw(m))
    assert all(c["ok"] for c in res["control"].values())
    assert all(c["measured"] == 2.8 for c in res["control"].values())


def test_control_failure_stops_every_later_arm(tmp_path):
    m, calls = fake_measure(nll_default=2.8 + 1e-3)
    res = run.run_all(tmp_path, **_kw(m))
    assert list(res["arms"]) == [64]
    assert set(res["not_run"]) == {4096, 512}
    assert all(lbl == 300 for _, _, lbl in calls)  # nothing past the control
    v = run.verdict(res["arms"], res["control"], res["error"])
    assert v["outcome"] == "inconclusive" and v["exit"] == Exit.DID_NOT_RUN


def test_deadline_skips_arms_not_started(tmp_path):
    t = {"now": 0.0}

    def clock():
        return t["now"]

    m0, _ = fake_measure()

    def m(seed_dir, seed, label, n):
        if n == 4096 and label == 1000:
            t["now"] = 10.0  # the deadline passes during N = 4096
        return m0(seed_dir, seed, label, n)

    res = run.run_all(tmp_path, **_kw(m, clock=clock, deadline=5.0))
    assert list(res["arms"]) == [64, 4096] and res["not_run"] == {512: "deadline"}


def test_measure_checkpoint_checks_the_stored_step(tmp_path, monkeypatch):
    monkeypatch.setattr(run.RC, "stored_step", lambda p: 1000)
    with pytest.raises(run.StepMismatch):
        run.measure_checkpoint(tmp_path, 0, 300, 4096)


def test_measure_checkpoint_default_measure_only_for_the_control(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(run.RC, "stored_step", lambda p: int(p.name[5:11]))
    monkeypatch.setattr(run, "doc_sets", lambda s, n: ("sets", n))
    monkeypatch.setattr(
        run.S003, "measure", lambda ck, s, dev, sets=None: seen.append(sets) or {}
    )
    run.measure_checkpoint(tmp_path, 0, 300, 64)
    assert seen == [("sets", 64), None]
    seen.clear()
    run.measure_checkpoint(tmp_path, 0, 1000, 64)
    run.measure_checkpoint(tmp_path, 0, 300, 4096)
    assert seen == [("sets", 64), ("sets", 4096)]


def test_render_results_backs_every_number_with_a_key(tmp_path):
    """A rendered page from a real Ledger over the fake run: every row key it cites
    exists."""
    sys.path.insert(0, str(ROOT / "scripts"))
    m, _ = fake_measure()
    res = run.run_all(tmp_path, **_kw(m))

    class L:
        def __init__(self):
            self.rows = []

        def note(self, key, value, *, how):
            self.rows.append({"key": key, "value": value})

        def stat(self, key, xs, *, how):
            self.rows.append({"key": key, "samples": xs})

        def verdict(self, **kw):
            self.v = kw

    led = L()
    v = run.write_rows(led, res)
    doc = {"rows": led.rows, "verdict": led.v, "run_id": "t", "status": "ok"}
    page = run.render_results(doc, "none")
    assert v["outcome"] == "survived" and "**survived**" in page
    keys = {r["key"] for r in led.rows}
    assert (
        "n4096.ckpt1000.heldout.slots_zeroed_minus_live.gap_2_to_M.answer_brier_over_16"
        in keys
    )
    assert "n64.ckpt300.train_minus_heldout.live.gap_2_to_M.answer_acc" in keys
    assert "not measured" not in page
