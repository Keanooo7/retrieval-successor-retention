"""The retrieval curve's own gates (`experiments/retrieval-curve/PREREG.md`, brief
`docs/lab-notes/dispatch-retrieval-curve.md` *Bar*).

Nothing here trains. The parent loop (`run_curve`) is driven by fake children and a
fake clock over checkpoint files touched on disk; the measurement is either a spy
around a real `rsr.train.checkpoint` file (the step check) or a synthetic S0-03
table. Mutation-battery gates (`scripts/mutation_battery.py`,
``# --- retrieval-curve ---``):

* ``test_decide_is_fed_heldout_not_train`` -- decide() fed the train population;
* ``test_reproduction_control_tolerance`` -- the control's tolerance widened;
* ``test_each_checkpoint_label_`` -- every label measured from the final checkpoint.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ledger as ledger_mod  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "retrieval_curve_run", ROOT / "experiments" / "retrieval-curve" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
Exit = run.Exit
SEEDS = [0, 1, 2]


def _prereg() -> dict:
    """The PREREG's front matter, read independently of run.py's transcription."""
    text = (ROOT / "experiments" / "retrieval-curve" / "PREREG.md").read_text()
    return yaml.safe_load(text.split("---")[1])


# --------------------------------------------------------------------------- #
# synthetic S0-03 tables
# --------------------------------------------------------------------------- #


def s003_table(zml: float, *, seed: int = 0, bar1: bool = True) -> dict:
    """An S0-03 ``measure()`` result. ``zml`` is slots_zeroed - live answer NLL at
    gap_2_to_M; train is an exact copy of heldout (only the population gate test
    makes them differ)."""
    base = run.CHANCE + 0.01 * (seed + 1)
    zero = base if bar1 else run.CHANCE - 0.5
    ds = {"n_documents": 64, "fraction_of_pairs_gap_gt_M": 0.1}
    for cond in run.S003.CONDITIONS:
        ds[cond] = {"real_token_nll": 1.5 + 0.01 * seed}
        for b in run.S003.BUCKETS:
            nll = base
            if cond in ("slots_zeroed", "gate_zeroed") and b == "gap_ge_2":
                nll = zero
            if cond == "slots_zeroed" and b == "gap_2_to_M":
                nll = base + zml
            if cond == "live" and b == "gap_gt_M":
                nll = base + 0.05  # S0-03's bar 2 holds: eviction bites
            ds[cond][b] = {
                "n": 10,
                "answer_nll": nll,
                "answer_acc": 0.06 + 0.001 * seed,
                "answer_nll_over_16": nll - 0.2,
            }
    return {
        "heldout": ds,
        "train": copy.deepcopy(ds),
        "memory_gates": [0.1, 0.2],
    }


def decisive_summary(seed: int) -> dict:
    x = {
        "ratio": 5.0 + seed,
        "random_ratio": 2.0 + seed,
        "cosine_trained": 0.3 + 0.01 * seed,
        "cosine_decoy": 0.2 + 0.01 * seed,
        "controls_pass": True,
        "controls_failed": [],
        "random_valid": True,
    }
    return {"train": dict(x), "heldout": dict(x)}


def record(seed: int, label: int, zml: float, *, bar1: bool = True) -> dict:
    return {
        "checkpoint": f"seed{seed}/ckpt-{label:06d}.pt",
        "stored_step": label,
        "s003": s003_table(zml, seed=seed, bar1=bar1),
        "decisive": decisive_summary(seed),
    }


def per_table(zml: dict[int, float], labels=run.CHECKPOINTS, bar1=None) -> dict:
    bar1 = bar1 or {}
    return {
        s: {c: record(s, c, zml[c], bar1=bar1.get(c, True)) for c in labels}
        for s in SEEDS
    }


def passing_control() -> dict:
    return {
        s: {"seed": s, "measured": 2.8, "reference": 2.8, "abs_diff": 0.0, "ok": True}
        for s in SEEDS
    }


# --------------------------------------------------------------------------- #
# the pre-registration, transcribed
# --------------------------------------------------------------------------- #


def test_thresholds_are_the_preregs():
    t = _prereg()["thresholds"]
    assert list(run.CHECKPOINTS) == t["checkpoints"]
    assert t["wall_deadline_hours"] == run.DEADLINE_HOURS
    assert t["threads"].startswith(f"{run.THREADS_PER_SEED} per seed")
    assert run.MARGIN == run.S003.MARGIN == 0.10  # imported, not re-chosen
    assert run.CHANCE == run.S003.CHANCE
    assert run.SEEDS == _prereg()["seeds"] == run.S003.SEEDS
    assert run.CONTROL_CHECKPOINT == 300 and run.ITERS == 3000
    assert run.CKPT_EVERY == 100 and all(c % run.CKPT_EVERY == 0 for c in run.CHECKPOINTS)
    assert run.DEVICE == "cpu"
    # the brief names S0-03's manifest as THREADS_PER_SEED's source
    s003 = json.loads((ROOT / "runs/s0-03-rewardable-corpus/manifest.json").read_text())
    assert s003["threads_per_seed"] == run.THREADS_PER_SEED


def test_decide_and_measure_are_imported_not_copied():
    src = (ROOT / "experiments" / "retrieval-curve" / "run.py").read_text()
    assert "def decide(" not in src and "def measure(" not in src
    assert run.S003.decide.__code__.co_filename.endswith("s0-03-rewardable-corpus/run.py")


def test_single_trains_s003_config_with_only_iters_and_ckpt_every_changed(
    tmp_path, monkeypatch
):
    import rsr.train.loop as loop

    seen = {}
    monkeypatch.setattr(loop, "train", lambda **kw: seen.update(kw) or {"ok": 1})
    run.single(1, run.ITERS, tmp_path)
    C = run.S003.CONFIG
    assert seen["ckpt_every"] == 100 and seen["iters"] == 3000
    assert seen["device"] == "cpu" and seen["seed"] == 1
    for k in ("d", "steps_per_stream", "batch", "max_tokens", "memory_slots", "lr"):
        assert seen[k] == C[k], k
    assert seen["masked_loss"] is True and seen["srep_norm_reg_weight"] == 0.0
    assert seen["policy_name"] == C["policy"] == "fifo"
    assert seen["vocab"] is None and seen["beat_every"] == 1


def test_child_env_and_argv_are_s003s(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    env = run.child_env()
    assert env["OMP_NUM_THREADS"] == "5" and env["MKL_NUM_THREADS"] == "5"
    argv = run.child_argv(2, 3000, tmp_path)
    assert argv[2:] == [
        "--single",
        "--seed",
        "2",
        "--iters",
        "3000",
        "--out-dir",
        str(tmp_path),
    ]


def test_ckpt_path_is_the_loops_filename(tmp_path):
    assert run.ckpt_path(tmp_path, 300).name == "ckpt-000300.pt"
    assert run.ckpt_path(tmp_path, 3000).name == "ckpt-003000.pt"


# --------------------------------------------------------------------------- #
# (1) decide() is fed the held-out population
# --------------------------------------------------------------------------- #


def test_decide_is_fed_heldout_not_train():
    """Held-out at chance (no retrieval), train memorised (live far below zeroed):
    the verdict must read held-out. Fed train, it would read 'falsified'."""
    per = per_table({c: 0.0 for c in run.CHECKPOINTS})
    for s in SEEDS:
        for c in run.CHECKPOINTS:
            t = per[s][c]["s003"]["train"]
            t["slots_zeroed"]["gap_2_to_M"]["answer_nll"] += 0.5
    d = run.decisions(per)
    assert set(d) == set(run.CHECKPOINTS)
    assert not any(x["h4_retrieval_shown"] for x in d.values())
    v = run.verdict(per, passing_control(), None)
    assert v["outcome"] == "survived", v


# --------------------------------------------------------------------------- #
# (2) the reproduction control
# --------------------------------------------------------------------------- #


def test_reproduction_control_tolerance_is_1e_6_and_a_2e_6_miss_fails():
    raw = _prereg()["thresholds"]["reproduction_tolerance"]
    assert raw.startswith("1e-6 absolute")
    assert float(raw.split()[0]) == run.REPRO_TOL
    ref = {0: 2.8, 1: 2.9, 2: 3.0}
    m = s003_table(0.0)
    m["heldout"]["live"]["gap_2_to_M"]["answer_nll"] = 2.9 + 2e-6
    c = run.reproduction_control(1, m, ref)
    assert c["ok"] is False and c["abs_diff"] == pytest.approx(2e-6)
    m["heldout"]["live"]["gap_2_to_M"]["answer_nll"] = 2.9 + 5e-7
    assert run.reproduction_control(1, m, ref)["ok"] is True


def test_reproduction_control_reads_the_s003_ledger_by_key_in_seed_order(tmp_path):
    assert run.S003_LEDGER == "runs/s0-03-rewardable-corpus/ledger.json"
    assert run.REPRO_KEY == "heldout.live.gap_2_to_M.answer_nll"
    doc = json.loads((ROOT / run.S003_LEDGER).read_text())
    assert doc["device"] == "cpu"
    (row,) = [r for r in doc["rows"] if r["key"] == run.REPRO_KEY]
    assert run.s003_reference() == dict(
        zip(doc["seeds_actually_run"], row["samples"], strict=True)
    )
    # looked up by key, not position: a decoy row first changes nothing
    fake = {
        "seeds_actually_run": [0, 1, 2],
        "rows": [
            {"key": "heldout.live.gap_ge_2.answer_nll", "samples": [9.0, 9.0, 9.0]},
            {"key": run.REPRO_KEY, "samples": [1.0, 2.0, 3.0]},
        ],
    }
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps(fake))
    assert run.s003_reference(p) == {0: 1.0, 1: 2.0, 2: 3.0}


# --------------------------------------------------------------------------- #
# (3) each checkpoint label measured from its own checkpoint
# --------------------------------------------------------------------------- #


@pytest.fixture
def spies(monkeypatch):
    seen = []
    monkeypatch.setattr(
        run.S003,
        "measure",
        lambda ckpt, seed, device: (
            seen.append(("s003", Path(ckpt).name, seed, device)) or {"heldout": {}}
        ),
    )

    def dec(ckpt, seed, *, decoy_ckpt="unset"):
        seen.append(("decisive", Path(ckpt).name, seed, decoy_ckpt))
        return {}

    monkeypatch.setattr(run.DECISIVE, "measure", dec)
    return seen


def _save(path: Path, step: int) -> None:
    from rsr.train import checkpoint as ck

    ck.save(path, step=step, model=torch.nn.Linear(1, 1))


def test_each_checkpoint_label_is_measured_from_its_own_checkpoint(tmp_path, spies):
    for c in run.CHECKPOINTS:
        _save(run.ckpt_path(tmp_path, c), c)
    for c in run.CHECKPOINTS:
        r = run.measure_checkpoint(tmp_path, 2, c)
        assert r["stored_step"] == c
        assert r["checkpoint"].endswith(f"ckpt-{c:06d}.pt")
    names = [f"ckpt-{c:06d}.pt" for c in run.CHECKPOINTS]
    assert spies == [
        x for n in names for x in (("s003", n, 2, "cpu"), ("decisive", n, 2, None))
    ]


def test_each_checkpoint_label_refuses_a_stored_step_that_differs(tmp_path, spies):
    _save(run.ckpt_path(tmp_path, 300), 1000)  # named 300, holds step 1000
    with pytest.raises(run.StepMismatch):
        run.measure_checkpoint(tmp_path, 0, 300)
    assert spies == []  # refused before anything was measured


# --------------------------------------------------------------------------- #
# the verdict table
# --------------------------------------------------------------------------- #


def _no(c):
    return {k: 0.0 for k in run.CHECKPOINTS} | ({c: 0.3} if c else {})


@pytest.mark.parametrize(
    ("zml", "labels", "bar1", "control_ok", "stopped", "outcome", "code", "at"),
    [
        (_no(1000), run.CHECKPOINTS, {}, True, None, "falsified", 0, [1000]),
        (_no(None), run.CHECKPOINTS, {}, True, None, "survived", 0, []),
        (_no(None), run.CHECKPOINTS, {1000: False}, True, None, "inconclusive", 0, []),
        # bar 1 failing where H4 holds is not retrieval shown
        (_no(3000), run.CHECKPOINTS, {3000: False}, True, None, "inconclusive", 0, []),
        # the deadline cut the run before 3000, no H4: partial
        (_no(None), (300, 1000), {}, True, "deadline", "inconclusive", 3, []),
        # the deadline cut the run, but H4 already held at 1000
        (_no(1000), (300, 1000), {}, True, "deadline", "falsified", 0, [1000]),
        # the control overrides every row, H4 included
        (_no(300), (300,), {}, False, "control", "inconclusive", 3, []),
    ],
    ids=[
        "falsified-at-1000",
        "survived",
        "bar1-fails",
        "h4-without-bar1",
        "deadline-partial",
        "deadline-after-falsified",
        "control-fails-overrides-h4",
    ],
)
def test_verdict_mapping(zml, labels, bar1, control_ok, stopped, outcome, code, at):
    per = per_table(zml, labels, bar1)
    control = passing_control()
    if not control_ok:
        control[1]["ok"] = False
    v = run.verdict(per, control, stopped)
    assert (v["outcome"], int(v["exit"]), v["falsified_at"]) == (outcome, code, at), v


def test_s003_label_is_not_this_verdict():
    """S0-03's 'survived' (H4 and bar 2) is this PREREG's 'falsified'."""
    per = per_table(_no(1000))
    d = run.decisions(per)[1000]
    assert d["h4_retrieval_shown"] and d["outcome"] == "survived"
    assert run.verdict(per, passing_control(), None)["outcome"] == "falsified"


# --------------------------------------------------------------------------- #
# the parent loop, with fake children and a fake clock
# --------------------------------------------------------------------------- #


class FakeChild:
    def __init__(self, seed, out_dir, world):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.seed, self.dir, self.world = seed, out_dir, world
        self.argv = run.child_argv(seed, run.ITERS, out_dir)
        self.rc = None
        self.terminated = False

    def poll(self):
        if self.rc is None and run.ckpt_path(self.dir, run.ITERS).exists():
            self.rc = 0
        return self.rc

    def terminate(self):
        self.terminated = True
        self.rc = -15

    def kill(self):
        self.rc = -9

    def wait(self, timeout=None):
        rc = self.poll()
        assert rc is not None, "waited on a child that was never stopped"
        return rc


class World:
    """Checkpoints appear one wave per sleep; the clock advances per call."""

    def __init__(self, root, waves, *, tick=60.0):
        self.root, self.waves, self.tick, self.t = root, list(waves), tick, 0.0
        self.children = {}

    def spawn(self, seed, d):
        self.children[seed] = FakeChild(seed, d, self)
        self._touch_next()
        return self.children[seed]

    def _touch_next(self):
        if len(self.children) == len(SEEDS) and self.waves:
            for c in self.waves.pop(0):
                for s in SEEDS:
                    run.ckpt_path(self.root / f"seed{s}", c).touch()

    def clock(self):
        return self.t

    def sleep(self, _):
        self.t += self.tick
        self._touch_next()


def fake_measure(zml, *, off=None):
    """``measure_fn`` returning a synthetic record; ``off`` = {seed: delta} moves
    that seed's held-out live gap_2_to_M NLL away from the control reference."""
    calls = []

    def m(seed_dir, seed, label):
        calls.append((seed, label))
        r = record(seed, label, zml[label])
        if off and label == run.CONTROL_CHECKPOINT and seed in off:
            r["s003"]["heldout"]["live"]["gap_2_to_M"]["answer_nll"] += off[seed]
        return r

    m.calls = calls
    return m


def reference():
    return {
        s: record(s, 300, 0.0)["s003"]["heldout"]["live"]["gap_2_to_M"]["answer_nll"]
        for s in SEEDS
    }


@pytest.fixture
def led(tmp_path, monkeypatch):
    """A real Ledger under tmp_path; the tree gates stubbed (a battery run dirties
    the tree)."""
    monkeypatch.setattr(ledger_mod, "_dirty_source_paths", lambda: [])
    monkeypatch.setattr(ledger_mod, "_exists_at_sha", lambda path, sha: True)
    return ledger_mod.Ledger(run.RUN_ID, question="test", runs_root=tmp_path / "runs")


def _execute(led, tmp_path, world, measure, *, deadline_s=1e9):
    return run.execute(
        led,
        world.root,
        spawn=world.spawn,
        measure_fn=measure,
        reference=reference(),
        deadline_s=deadline_s,
        results_path=tmp_path / "RESULTS.md",
        clock=world.clock,
        sleep=world.sleep,
    )


def _rows(led):
    return {r["key"]: r for r in json.loads(led.path.read_text())["rows"]}


def test_control_failure_terminates_every_child_and_exits_3(led, tmp_path):
    """PREREG: "Stop, exit 3, report both sets of samples". Seed 1 fails; seed 2's
    ckpt-300, already on disk, is still measured so every control sample is
    reported. Nothing past ckpt-300 is read."""
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300, 1000)])  # 1000 is already on disk, and never read
    measure = fake_measure(_no(1000), off={1: 0.05})
    code = _execute(led, tmp_path, world, measure)
    assert code == Exit.DID_NOT_RUN
    assert all(ch.terminated for ch in world.children.values())
    assert sorted(measure.calls) == [(s, 300) for s in SEEDS]
    doc = json.loads(led.path.read_text())
    assert doc["verdict"]["outcome"] == "inconclusive"
    assert doc["status"] == "partial"
    rows = _rows(led)
    assert rows["reproduction_control.ok"]["value"] == [True, False, True]
    assert None not in rows["reproduction_control.measured"]["value"]
    assert None not in rows["reproduction_control.reference"]["value"]
    assert rows["stopped_because"]["value"] == "reproduction control failed"


def test_deadline_terminates_and_measures_what_is_on_disk(led, tmp_path):
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300,), (1000,), (), (), ()], tick=3600.0)
    measure = fake_measure(_no(None))
    code = _execute(led, tmp_path, world, measure, deadline_s=2 * 3600.0)
    assert code == Exit.DID_NOT_RUN
    assert all(ch.terminated for ch in world.children.values())
    rows = _rows(led)
    assert rows["checkpoints_measured_on_every_seed"]["value"] == [300, 1000]
    assert rows["stopped_because"]["value"] == "deadline"
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "partial" and doc["verdict"]["outcome"] == "inconclusive"


def test_a_child_that_dies_early_is_crashed_not_waited_for(led, tmp_path):
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300,), (), ()])
    measure = fake_measure(_no(None))
    orig = world.sleep

    def sleep(x):
        orig(x)
        world.children[2].rc = 1  # seed 2 crashes after ckpt-300

    world.sleep = sleep
    assert _execute(led, tmp_path, world, measure) == Exit.DID_NOT_RUN
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "crashed"
    assert all(world.children[s].terminated for s in (0, 1))


def test_full_run_writes_every_key_and_renders_audited_results(
    led, tmp_path, monkeypatch
):
    import render_scoreboard

    # Isolate from the repository's hand-written BRIEF-ERRORS.md: it quotes the real
    # ledger, which this synthetic ledger cannot back. Absent, as before the run.
    monkeypatch.setattr(run, "BRIEF_ERRORS", str(tmp_path / "BRIEF-ERRORS.md"))
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300,), (1000,), (3000,)])
    measure = fake_measure(_no(1000))
    led.note("corpus_sha256", {f"seed{s}": "ab" * 32 for s in SEEDS}, how="t")
    assert _execute(led, tmp_path, world, measure) == Exit.OK
    assert sorted(measure.calls) == sorted((s, c) for s in SEEDS for c in run.CHECKPOINTS)
    doc = json.loads(led.path.read_text())
    assert doc["verdict"]["outcome"] == "falsified" and doc["status"] == "ok"
    rows = _rows(led)
    for c in run.CHECKPOINTS:
        p = f"ckpt{c}"
        for ds in ("heldout", "train"):
            for cond in ("live", "slots_zeroed"):
                for b in run.S003.BUCKETS:
                    assert rows[f"{p}.{ds}.{cond}.{b}.answer_nll"]["samples"]
        for batch in ("train", "heldout"):
            for k in ("ratio", "random_ratio", "cosine_trained", "cosine_decoy"):
                assert len(rows[f"{p}.decisive.{batch}.{k}"]["samples"]) == 3
        assert rows[f"{p}.s003_outcome"]["value"] in ("survived", "inconclusive")
    assert rows["ckpt1000.s003_outcome"]["value"] == "survived"  # S0-03's label
    assert rows["falsified_at_checkpoints"]["value"] == [1000]
    text = (tmp_path / "RESULTS.md").read_text()
    assert text == run.render_results(doc)
    assert "**falsified**" in text and "cannot move here" in text
    assert render_scoreboard.audit_prose(tmp_path / "runs", text) == []


@pytest.mark.parametrize(
    "exc",
    [run.StepMismatch("stored 1100"), SystemExit("S0-03 _sets refused")],
    ids=["step-mismatch", "s003-systemexit"],
)
def test_a_measurement_that_raises_stops_every_child_and_writes_a_ledger(
    led, tmp_path, exc
):
    """A raise inside the measurement never escapes past live training children:
    they are stopped, the ledger is written, the run is inconclusive, exit 3."""
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300,), (1000,), (3000,)])
    inner = fake_measure(_no(None))

    def measure(seed_dir, seed, label):
        if label == 1000:
            raise exc
        return inner(seed_dir, seed, label)

    assert _execute(led, tmp_path, world, measure) == Exit.DID_NOT_RUN
    assert all(ch.terminated for ch in world.children.values())
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "crashed"
    assert doc["verdict"]["outcome"] == "inconclusive"
    rows = _rows(led)
    assert type(exc).__name__ in rows["measurement_error"]["value"]
    assert rows["stopped_because"]["value"] == "measurement raised"
    assert (root / "raw.json").exists()


def test_a_child_exiting_nonzero_after_every_checkpoint_is_crashed(led, tmp_path):
    root = tmp_path / "runs" / run.RUN_ID
    world = World(root, [(300,), (1000,), (3000,)])
    orig = world.sleep

    def sleep(x):
        orig(x)
        if run.ckpt_path(root / "seed0", run.ITERS).exists():
            world.children[0].rc = 1  # wrote ckpt-3000, then failed

    world.sleep = sleep
    _execute(led, tmp_path, world, fake_measure(_no(None)))
    assert json.loads(led.path.read_text())["status"] == "crashed"


def test_results_carry_the_hardware_and_the_hand_written_brief_errors(tmp_path):
    doc = {
        "run_id": run.RUN_ID,
        "provenance": {"git_sha": "abc", "machine": "arm64", "platform": "macOS-x"},
        "rows": [],
    }
    text = run.render_results(doc)
    assert "`provenance.machine` `arm64`" in text
    assert "`provenance.platform` `macOS-x`" in text
    assert run.BRIEF_ERRORS in text.split("## BRIEF ERRORS")[1]
    p = tmp_path / "BRIEF-ERRORS.md"
    assert run.read_brief_errors(p) is None
    p.write_text("none\n")
    errs = run.read_brief_errors(p)
    assert run.render_results(doc, errs).split("## BRIEF ERRORS")[1].strip() == "none"


def test_corpus_sha256_is_per_seed_and_deterministic():
    a = run.corpus_sha256(0)
    assert len(a) == 64 and int(a, 16) >= 0
    assert a == run.corpus_sha256(0)
    assert a != run.corpus_sha256(1)
