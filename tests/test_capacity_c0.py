"""C0's own gates (`experiments/capacity-c0/PREREG.md`, brief
`docs/lab-notes/dispatch-capacity-c0.md` *Bar*).

Every capacity rule is a pure function in `experiments/capacity-c0/run.py`; each is
fed a synthetic sweep table here. The machine-touching parts (process list,
signals, server restart, jobs, the suite) are injected, so **no test here ever
lists, signals or starts a real foreign process**. The one real computation is
`test_output_hash_*`: two 1-iteration core-workload jobs, concurrently, because
the bar asks for two identical solo runs whose checkpoint FILES differ.

Mutation-battery gates (`scripts/mutation_battery.py`, ``# --- capacity-c0 ---``):
``test_det_verdict`` (the comparison skipped), ``test_cpu_det_slots_slower_rep``
(the faster repetition used), ``test_output_hash`` (the checkpoint file hashed),
``test_job_out_dirs`` (two jobs sharing an out_dir) and
``test_servers_restarted`` (the restart moved out of the ``finally``).
"""

from __future__ import annotations

import importlib.util
import json
import math
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ledger as ledger_mod  # noqa: E402
from orchestrator import lanes  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "capacity_c0_run", ROOT / "experiments" / "capacity-c0" / "run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through it
    spec.loader.exec_module(mod)
    return mod


run = _load()
GIB = 2**30
MEMSIZE = 64 * GIB
CAPACITY_KEYS = set(lanes.C0_LEDGER_KEYS.values())


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeMachine:
    """A process table the run can list and signal; nothing real is touched."""

    OWN = 1000

    def __init__(self, procs):
        me = run.Proc(self.OWN, 1, 100 * 2**20, 0.0, "python run.py")
        self.procs = [*procs, me]
        self.killed: list[tuple[int, int]] = []
        self.spawned: list[dict] = []
        self.t = 0.0

    def list(self):
        return list(self.procs)

    def kill(self, pid, sig):
        self.killed.append((pid, sig))
        self.procs = [p for p in self.procs if p.pid != pid]

    def spawn(self, record, log_dir):
        self.spawned.append(record)
        return 90000 + len(self.spawned)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def system(self):
        return run.System(
            procs=self.list,
            kill=self.kill,
            spawn_server=self.spawn,
            cwd_of=lambda pid: "/Users/owner/servers",
            sleep=self.sleep,
            now=self.now,
            own_pid=self.OWN,
            memsize=lambda: MEMSIZE,
            free_bytes=lambda: 40 * GIB,
        )


def _server(pid, gib, cmd):
    return run.Proc(pid, 1, int(gib * GIB), 1.0, cmd)


MLX = "/opt/homebrew/bin/python3.12 -m mlx_lm.server --model qwen --port 8080"
LLAMA = "/opt/homebrew/bin/llama-server -m /models/x.gguf --port 8081"


class FakeSampler:
    def __init__(self, agent_gib=0.8, cores=(2.1, 2.4, 3.2)):
        self.agent_rss = [int(agent_gib * GIB)] * 10
        self.mps_cores = list(cores)
        self.watch: list[int] = []

    def sample_once(self):
        self.agent_rss.append(self.agent_rss[-1])


def _core_time(k: int) -> float:
    return 60.0 * (1 + 0.02 * k)


def fake_measure(*, mismatch_at=None, raise_at=None, sampler=None):
    """A `Measure` whose waves return synthetic results: training jobs 6.8 GiB,
    core jobs 2.5 GiB, core job time 60 * (1 + 0.02 k) s, one hash for all."""

    def make(deadline_at):
        def wave(specs):
            s0 = specs[0]
            if raise_at is not None and s0.tag.startswith(raise_at):
                raise RuntimeError(f"boom at {s0.tag}")
            out = []
            for s in specs:
                h = "0.5:" + "a" * 64
                if mismatch_at is not None and s.tag == mismatch_at:
                    h = "0.5:" + "b" * 64
                rss = 2.5 if s.workload == "core" else 6.8
                r = run.JobResult(
                    s,
                    f"/fake/{s.tag}",
                    ["py", "run.py", "--job"],
                    0,
                    None,
                    int(rss * GIB),
                )
                r.output_hash = h
                r.job_time_s = _core_time(s.k) if s.workload == "core" else 50.0
                r.n_beats = s.iters
                r.mem_gb_max = 9.5 if s.device == "mps" else None
                out.append(r)
            return out

        def suite(n):
            return {"n": n, "wall_s": {1: 100.0, 2: 70.0, 4: 62.0, 8: 60.0}[n], "rc": 0}

        return run.Measure(wave=wave, suite=suite, sampler=sampler or FakeSampler())

    return make


@pytest.fixture
def led(tmp_path, monkeypatch):
    """A real `Ledger` under tmp_path. The source-clean and entry-point-at-sha gates
    are stubbed: they check the tree, which a mutation-battery run has dirtied."""
    monkeypatch.setattr(ledger_mod, "_dirty_source_paths", lambda: [])
    monkeypatch.setattr(ledger_mod, "_exists_at_sha", lambda path, sha: True)
    return ledger_mod.Ledger("capacity-c0", question="test", runs_root=tmp_path / "runs")


YIELD = run.ServersRuling("R-2026-09-22-mlx-servers", "yield", "... yield to it ...")
STAY = run.ServersRuling("R-2099-01-01-mlx-servers", "stay_up", "the servers STAY UP")
NONE = run.ServersRuling(None, "none", "")


def _rows(led):
    return {r["key"]: r for r in led.doc["rows"]}


def _execute(led, machine, make, ruling=YIELD, tmp=None):
    return run.execute(
        led, machine.system(), make, ruling=ruling, log_dir=Path(tmp or "/nonexistent")
    )


# --------------------------------------------------------------------------- #
# quiet machine and the servers
# --------------------------------------------------------------------------- #


def test_quiet_refuses_a_foreign_process_over_2_gib(led):
    m = FakeMachine([_server(10, 3.0, "/usr/bin/python3 train_something.py")])
    with pytest.raises(run.Refusal, match="machine not quiet"):
        _execute(led, m, fake_measure())
    assert m.killed == [], "a foreign process is never signalled"
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "did_not_run"
    assert not CAPACITY_KEYS & {r["key"] for r in doc["rows"]}


def test_quiet_under_2_gib_is_quiet(led):
    m = FakeMachine([_server(10, 1.9, "/Applications/Big.app/Contents/MacOS/Big")])
    assert _execute(led, m, fake_measure()) == run.Exit.OK
    assert m.killed == []


def test_quiet_yield_stops_only_the_servers_and_records_them(led, tmp_path):
    other = _server(12, 3.0, "/usr/bin/python3 -m mlx_lm.generate --prompt hi")
    m = FakeMachine([_server(10, 20.0, MLX), _server(11, 5.0, LLAMA), other])
    with pytest.raises(run.Refusal, match="machine not quiet"):
        _execute(led, m, fake_measure(), tmp=tmp_path)
    assert sorted(p for p, _ in m.killed) == [10, 11]
    assert all(sig == signal.SIGTERM for _, sig in m.killed)
    rows = _rows(led)
    rec = rows["c0.stopped_servers"]["value"]
    assert [r["pid"] for r in rec] == [10, 11]
    assert rec[0]["command"] == MLX and rec[0]["cwd"] == "/Users/owner/servers"
    assert rows["c0.stopped_servers_rss_gib"]["value"] == pytest.approx(25.0)
    # the not-quiet refusal still restarts what it stopped
    assert [r["pid"] for r in m.spawned] == [10, 11]


def test_quiet_yield_then_measures_a_quiet_machine(led, tmp_path):
    m = FakeMachine([_server(10, 20.0, MLX)])
    assert _execute(led, m, fake_measure(), tmp=tmp_path) == run.Exit.OK
    assert [r["pid"] for r in m.spawned] == [10]
    assert _rows(led)["c0.restarted_servers"]["value"][0]["restarted"] is True


def test_quiet_no_ruling_servers_are_foreign(led):
    m = FakeMachine([_server(10, 20.0, MLX)])
    with pytest.raises(run.Refusal, match="machine not quiet"):
        _execute(led, m, fake_measure(), ruling=NONE)
    assert m.killed == [] and m.spawned == []


def test_quiet_stay_up_ruling_exempts_servers_and_budgets_them(led):
    m = FakeMachine([_server(10, 20.0, MLX)])
    assert _execute(led, m, fake_measure(), ruling=STAY) == run.Exit.OK
    assert m.killed == []
    rows = _rows(led)
    assert rows["c0.foreign_rss_gb"]["value"] == pytest.approx(20.0)
    assert rows["c0.servers_ruling"]["value"]["text"] == "the servers STAY UP"
    # 2 sessions x 0.8 GiB + 4.0 + 20.0 = 25.6 -> 26.0
    assert rows["c0.reserve_gb"]["value"] == 26.0


def test_quiet_respawned_server_refuses(led):
    m = FakeMachine([_server(10, 20.0, MLX)])
    sysm = m.system()
    real_kill = m.kill

    def kill_and_respawn(pid, sig):
        real_kill(pid, sig)
        m.procs.append(_server(77, 20.0, MLX))  # launchd KeepAlive

    sysm.kill = kill_and_respawn
    with pytest.raises(run.Refusal, match="respawned"):
        run.execute(led, sysm, fake_measure(), ruling=YIELD, log_dir=Path("/x"))
    assert all(pid in (10, 77) for pid, _ in m.killed)


def test_the_signed_servers_ruling_reads_as_yield():
    r = run.read_servers_ruling(ROOT / "docs" / "owner" / "rulings")
    assert r.id == "R-2026-09-22-mlx-servers"
    assert r.mode == "yield"


@pytest.mark.parametrize(
    "cmd,hit",
    [
        (MLX, True),
        (LLAMA, True),
        ("/Users/b/.venv/bin/mlx_lm.server --port 1", True),
        ("llama-server", True),
        ("/usr/bin/python3 -m mlx_lm.generate", False),
        ("/usr/bin/grep llama-server-notes.txt", False),
        ("/usr/bin/vim notes-about-mlx_lm.server.md", False),
    ],
)
def test_is_server_by_command_line(cmd, hit):
    assert run.is_server(cmd) is hit


def test_parse_ps_and_trees():
    procs = run.parse_ps(
        "  1     0  1000  0.0 /sbin/launchd\n"
        " 50     1  2048 12.5 claude\n"
        " 60    50  1024 50.0 /bin/zsh -c run\n"
        " 70    60  4096 100.0 python run.py\n"
        " 80    70  8192 300.0 python run.py --job\n"
    )
    assert procs[1] == run.Proc(50, 1, 2048 * 1024, 12.5, "claude")
    assert run.find_agent_pid(procs, 70) == 50
    assert run.descendants(procs, 70) == {70, 80}
    assert run.tree_rss_bytes(procs, 50, exclude={70, 80}) == (2048 + 1024) * 1024
    assert run.tree_cores(procs, 70) == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# the rules, on synthetic tables
# --------------------------------------------------------------------------- #


def test_maxrss_units_bytes_on_macos_kib_on_linux():
    assert run.maxrss_bytes(7 * GIB, "darwin") == 7 * GIB
    assert run.maxrss_bytes(7 * 2**20, "linux") == 7 * GIB


def test_k_mem_and_sweep_levels():
    assert run.k_mem(MEMSIZE, int(6.8 * GIB)) == 8  # (64 - 8) / 6.8 = 8.2
    assert run.k_mem(MEMSIZE, 7 * GIB) == 8
    assert run.k_mem(MEMSIZE, int(7.1 * GIB)) == 7  # k=8 dropped: the rule working
    assert run.sweep_levels(run.DET_T1_KS, 7) == [1, 2, 4]
    assert run.sweep_levels(run.CORE_SWEEP_K, 22) == list(run.CORE_SWEEP_K)


def test_cpu_det_slots_first_step_always_passes_minimum_1():
    # k=2 takes 3x as long: T(2) = 2/180 < T(1) = 1/60, g < 0 -> stop at 1
    slots, table = run.cpu_det_slots({1: 60.0, 2: 180.0, 4: 60.0})
    assert slots == 1
    assert table[0]["step_passes"] is True and table[0]["g"] == pytest.approx(1 / 60)
    # a single level is still 1
    assert run.cpu_det_slots({1: 999.0})[0] == 1


def test_cpu_det_slots_k15_fails():
    spans = {k: 60.0 for k in (1, 2, 4, 8, 10, 12, 14)}
    spans[15] = 60.0 * 15 / 14.2  # T(15) = 14.2/60: g = 0.2 T(1) < 0.5 T(1)
    spans[16] = 60.0  # recovers, but the rule is a prefix: never counted
    slots, table = run.cpu_det_slots(spans)
    assert slots == 14
    assert [r["k"] for r in table if not r["step_passes"]] == [15]
    assert table[-1]["step_passes"] is True and table[-1]["prefix_passes"] is False


def test_cpu_det_slots_threshold_is_half_of_t1():
    # g(2) = (2/80 - 1/60) = 0.00833 = exactly 0.5 * T(1): passes (>=)
    slots, _ = run.cpu_det_slots({1: 60.0, 2: 80.0})
    assert slots == 2
    slots, _ = run.cpu_det_slots({1: 60.0, 2: 80.5})
    assert slots == 1


def test_cpu_det_slots_slower_rep_is_used():
    # rep0 fast (60 s) passes at k=4; rep1 slow (200 s) fails. The slower governs.
    reps = {
        1: [[60.0], [60.0]],
        2: [[60.0, 61.0], [60.0, 60.5]],
        4: [[60.0] * 4, [200.0] * 4],
    }
    spans = {k: run.makespan(r) for k, r in reps.items()}
    assert spans == {1: 60.0, 2: 61.0, 4: 200.0}
    assert run.cpu_det_slots(spans)[0] == 2


def test_battery_cpu_slots_raw_smallest_within_15_percent():
    W = {1: 100.0, 2: 70.0, 4: 62.0, 8: 60.0}
    assert run.battery_cpu_slots_raw(W) == 4
    assert run.battery_cpu_slots_raw({1: 60.0, 2: 55.0, 4: 80.0, 8: 90.0}) == 1
    assert run.clamp_to(4, 2) == 2 and run.clamp_to(4, 8) == 4


def test_p95_nearest_rank_and_mps_reserve():
    xs = list(range(1, 21))  # 20 samples: ceil(19) -> the 19th smallest
    assert run.p95(xs) == 19
    assert run.mps_reserves_cpu_slots_raw([0.1, 0.2, 0.3]) == 1  # minimum 1
    assert run.mps_reserves_cpu_slots_raw([2.1] * 19 + [9.0]) == 3  # ceil(2.1)
    assert run.clamp_to(run.mps_reserves_cpu_slots_raw([5.5] * 20), 4) == 4


def test_agent_sessions_policy_and_feasibility():
    assert run.agent_sessions(0.8, 64.0) == 2
    assert run.agent_sessions(3.2, 64.0) == 2  # 6.4 > 6.4? no
    assert run.agent_sessions(3.3, 64.0) == 1


def test_reserve_gb_rounds_up_to_half():
    assert run.reserve_gb(2, 0.8) == 6.0  # 5.6 -> 6.0
    assert run.reserve_gb(2, 0.75) == 5.5  # exactly 5.5
    assert run.reserve_gb(1, 0.1, 20.0) == 24.5


def test_conflict_lines():
    c = run.conflicts(
        cpu_det=12, k_mem_training=8, reserve=6.0, mps_job_peak_gb=9.5, physical_gib=64.0
    )
    assert c["a"]["blocks_retrieval_curve"] is True
    assert c["a"]["cpu_det_slots_ge_retrieval_curve"] is False
    assert c["b"]["conflict"] is True
    assert c["c"]["conflict"] is False
    c = run.conflicts(
        cpu_det=16,
        k_mem_training=22,
        reserve=30.0,
        mps_job_peak_gb=40.0,
        physical_gib=64.0,
    )
    assert (c["a"]["blocks_retrieval_curve"], c["b"]["conflict"], c["c"]["conflict"]) == (
        False,
        False,
        True,
    )


# --------------------------------------------------------------------------- #
# the determinism verdict and what it writes
# --------------------------------------------------------------------------- #


def _hj(t, k, rep, j, h):
    return run.HashedJob(t, k, rep, j, h)


def test_det_verdict_all_match_survives():
    solo = {1: "x", 5: "y"}
    jobs = [_hj(1, 2, 0, j, "x") for j in range(2)] + [
        _hj(5, 3, 0, j, "y") for j in range(3)
    ]
    outcome, levels, mism = run.determinism_verdict(solo, jobs)
    assert outcome == "survived" and mism == []
    assert levels == {
        "t1.k2.rep0": {"n_jobs": 2, "n_match": 2},
        "t5.k3.rep0": {"n_jobs": 3, "n_match": 3},
    }


def test_det_verdict_one_mismatch_falsifies():
    solo = {1: "x", 5: "y"}
    jobs = [_hj(1, 4, 1, j, "x") for j in range(4)] + [_hj(5, 3, 0, 2, "x")]
    outcome, levels, mism = run.determinism_verdict(solo, jobs)
    assert outcome == "falsified"
    assert levels["t5.k3.rep0"] == {"n_jobs": 1, "n_match": 0}
    assert mism == [
        {"threads": 5, "k": 3, "rep": 0, "j": 2, "output_hash": "x", "solo_hash": "y"}
    ]


def test_det_verdict_falsified_writes_no_capacity_keys(led):
    m = FakeMachine([])
    code = _execute(led, m, fake_measure(mismatch_at="det/t1/k4/rep1/job3"))
    assert code == run.Exit.FAIL
    doc = json.loads(led.path.read_text())
    keys = {r["key"] for r in doc["rows"]}
    assert not CAPACITY_KEYS & keys
    assert doc["verdict"]["outcome"] == "falsified"
    assert _rows(led)["c0.det.levels"]["value"]["t1.k4.rep1"] == {
        "n_jobs": 4,
        "n_match": 3,
    }
    with pytest.raises(lanes.Refused, match="no row for ledger key"):
        lanes.generate(led.path)


def test_survived_writes_the_five_keys_from_the_rules(led):
    m = FakeMachine([])
    assert _execute(led, m, fake_measure()) == run.Exit.OK
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "ok" and doc["verdict"]["outcome"] == "survived"
    rows = _rows(led)
    assert set(rows) >= CAPACITY_KEYS
    # recomputed here from the same synthetic inputs, by the rules
    spans = {k: _core_time(k) for k in run.CORE_SWEEP_K}
    slots = run.cpu_det_slots(spans)[0]
    assert rows["c0.cpu_det_slots"]["value"] == slots
    assert rows["c0.k_mem_training"]["value"] == 8
    assert rows["c0.k_mem_core"]["value"] == 22
    assert rows["c0.battery_cpu_slots"]["value"] == min(4, slots)
    assert rows["c0.mps_reserves_cpu_slots"]["value"] == min(4, slots)
    assert rows["c0.agent_sessions"]["value"] == 2
    assert rows["c0.reserve_gb"]["value"] == 6.0
    assert rows["c0.mps_job_peak_gb"]["value"] == 9.5
    assert rows["c0.det.n_jobs"]["value"] == 15 * 2 + 4
    doc_l = lanes.generate(led.path)
    assert doc_l["cpu_det_slots"] == slots


def test_partial_on_deadline(led, tmp_path):
    m = FakeMachine([_server(10, 20.0, MLX)])

    def make(deadline_at):
        w = run.Wave(tmp_path / "c0", deadline_at, now=lambda: deadline_at + 1.0)
        return run.Measure(wave=w, suite=lambda n: {}, sampler=FakeSampler())

    with pytest.raises(run.DeadlineExceeded):
        _execute(led, m, make, tmp=tmp_path)
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "partial"
    assert not (tmp_path / "c0" / "sweep").exists(), "nothing was spawned"
    assert [r["pid"] for r in m.spawned] == [10], "servers restarted after the deadline"
    with pytest.raises(lanes.Refused, match="partial"):
        lanes.generate(led.path)


def test_servers_restarted_when_measurement_raises(led, tmp_path):
    m = FakeMachine([_server(10, 20.0, MLX), _server(11, 5.0, LLAMA)])
    with pytest.raises(RuntimeError, match="boom"):
        _execute(led, m, fake_measure(raise_at="core/t1/k8"), tmp=tmp_path)
    assert [r["pid"] for r in m.spawned] == [10, 11]
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "partial"
    assert [r["restarted"] for r in _rows(led)["c0.restarted_servers"]["value"]] == [
        True,
        True,
    ]


def test_restart_skips_a_server_already_running(tmp_path):
    m = FakeMachine([_server(10, 20.0, MLX)])
    out = run.restart_servers(
        m.system(), [{"pid": 3, "command": MLX, "stopped": True}], tmp_path
    )
    assert out == [{"pid_was": 3, "restarted": False, "why": "already running"}]
    assert m.spawned == []


# --------------------------------------------------------------------------- #
# job layout, output hash, heartbeat
# --------------------------------------------------------------------------- #


def test_job_out_dirs_are_unique_and_follow_the_layout():
    root = Path("/r")
    specs = [s for w in run.full_plan(16, 16) for s in w]
    dirs = [run.job_out_dir(root, s) for s in specs]
    assert len(set(dirs)) == len(dirs)
    assert run.job_out_dir(root, run.det_waves(8)[-1][2]) == Path(
        "/r/sweep/det/t5/k3/rep0/job2"
    )
    assert sum(len(w) for w in run.core_waves(16)) + 1 == 2 * sum(run.CORE_SWEEP_K)


def test_det_plan_is_the_preregistered_sweep():
    waves = [run.solo_training_wave(), *run.det_waves(8)]
    t1 = [(w[0].k, w[0].rep, len(w)) for w in waves if w[0].threads == 1]
    assert t1 == [(k, r, k) for k in (1, 2, 4, 8) for r in (0, 1)]
    t5 = [len(w) for w in waves if w[0].threads == 5]
    assert t5 == [1, 3]
    assert all(s.iters == 10 and s.device == "cpu" for w in waves for s in w)
    # k_mem_training 7 drops k=8; 2 drops the 3 concurrent 5-thread jobs
    assert max(w[0].k for w in run.det_waves(7) if w[0].threads == 1) == 4
    assert [len(w) for w in run.det_waves(2) if w[0].threads == 5] == [1]


def test_state_dict_sha256_feeds_name_dtype_shape_bytes():
    import torch

    a = {"w": torch.zeros(2, 3), "b": torch.ones(3)}
    h = run.state_dict_sha256(a)
    assert h == run.state_dict_sha256({"b": torch.ones(3), "w": torch.zeros(2, 3)})
    assert h != run.state_dict_sha256({"w": torch.zeros(3, 2), "b": torch.ones(3)})
    assert h != run.state_dict_sha256(
        {"w": torch.zeros(2, 3, dtype=torch.float64), "b": torch.ones(3)}
    )
    assert h != run.state_dict_sha256({"v": torch.zeros(2, 3), "b": torch.ones(3)})
    assert run.output_hash(0.25, h) == f"0.25:{h}"


def test_output_hash_identical_runs_equal_while_checkpoint_files_differ(tmp_path):
    """The bar: two identical solo runs (iters=1, the batch-4 core workload so the
    suite stays small) hash equal, and their checkpoint FILES differ -- which is
    why the file is never hashed. Real subprocesses through the real `Wave`."""
    specs = [run.JobSpec("core", "core", 1, 2, 0, j, "cpu", 1) for j in range(2)]
    rs = run.Wave(tmp_path, deadline_at=math.inf)(specs)
    assert [r.rc for r in rs] == [0, 0]
    assert rs[0].out_dir != rs[1].out_dir
    assert rs[0].output_hash == rs[1].output_hash
    ck = [Path(r.result["checkpoint"]).read_bytes() for r in rs]
    assert ck[0] != ck[1], "identical runs, different checkpoint bytes (RngState)"
    assert all(r.maxrss_bytes > 100 * 2**20 for r in rs)
    assert all(r.result["torch_num_threads"] == 1 for r in rs)


def test_heartbeat_stats_first_to_last_beat(tmp_path):
    hb = tmp_path / "heartbeat.jsonl"
    hb.write_text(
        "\n".join(
            json.dumps(x)
            for x in (
                {"kind": "header", "elapsed_s": 0.0},
                {"kind": "beat", "elapsed_s": 5.0, "mem_gb": 3.0},
                {"kind": "beat", "elapsed_s": 9.0, "mem_gb": 4.5},
                {"kind": "beat", "elapsed_s": 12.5, "mem_gb": None},
                {"kind": "footer", "elapsed_s": 20.0},
            )
        )
    )
    assert run.heartbeat_stats(hb) == {"n_beats": 3, "job_time_s": 7.5, "mem_gb_max": 4.5}


def test_job_env_caps_or_uncaps_threads(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    assert run.job_env(5)["RSR_TORCH_THREADS"] == "5"
    assert run.job_env(5)["OMP_NUM_THREADS"] == "5"
    assert not set(lanes.THREAD_ENV_VARS) & set(run.job_env(None))


# --------------------------------------------------------------------------- #
# RESULTS.md
# --------------------------------------------------------------------------- #


def test_rendered_results_pass_the_prose_audit(led, tmp_path):
    import render_scoreboard

    led.note(
        "c0.machine",
        {
            "summary": "sysctl hw.model Mac16,9, Apple M4 Max, hw.perflevel0.physicalcpu "
            "12, hw.perflevel1.physicalcpu 4, hw.memsize 68719476736"
        },
        how="t",
    )
    led.note("c0.corpus_sha256", "ab" * 32, how="t")
    led.run_meta(seeds_actually_run=[0])
    assert _execute(led, FakeMachine([]), fake_measure()) == run.Exit.OK
    text = run.render_results(json.loads(led.path.read_text()))
    assert "`c0.cpu_det_slots`" in text and "(c)" in text
    assert render_scoreboard.audit_prose(tmp_path / "runs", text) == []
