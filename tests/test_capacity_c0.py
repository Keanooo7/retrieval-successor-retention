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

``# --- capacity-c0: server safety (2026-09-22) ---``:
``test_servers_stop_marks_each_kill`` (S1), ``test_servers_signal_restarts`` and
``test_servers_restart_failure_still_writes_ledger`` (S2),
``test_servers_restart_alive_check`` (S3), ``test_servers_measured_layout`` and
``test_servers_disagreement_refuses`` (S4), ``test_barrier_`` (S5). The measured
process layout of the Studio (2026-09-22) is the fixture ``MEASURED_ARGV``.
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


class FakeHandle:
    """What `spawn_server` returns: a pid and `poll()` (None = still running). The
    clock time of each poll is recorded, for the alive-after-10-s check."""

    def __init__(self, pid, machine, rc=None):
        self.pid, self.machine, self.rc = pid, machine, rc
        self.polled_at: list[float] = []

    def poll(self):
        self.polled_at.append(self.machine.t)
        return self.rc


class FakeMachine:
    """A process table the run can list and signal; nothing real is touched.

    Listening sockets default to: every name-matched process listens on the
    ``--port`` its command line declares. ``listeners=`` overrides that with an
    explicit {pid: {ports}} (the measured layout, where the caffeinate helpers
    carry ``--port`` in their argv and listen on nothing). ``argv=`` gives exact
    argvs by pid; the default is the ps line split on whitespace."""

    OWN = 1000

    def __init__(self, procs, *, listeners=None, argv=None, reap_children=False):
        me = run.Proc(self.OWN, 1, 100 * 2**20, 0.0, "python run.py")
        self.procs = [*procs, me]
        self.killed: list[tuple[int, int]] = []
        self.spawned: list[dict] = []
        self.handles: list[FakeHandle] = []
        self.spawn_rc = None
        self.t = 0.0
        self._listeners = listeners
        self._argv = argv or {}
        self.reap_children = reap_children

    def list(self):
        return list(self.procs)

    def listeners(self):
        alive = {p.pid for p in self.procs}
        if self._listeners is not None:
            return {pid: set(v) for pid, v in self._listeners.items() if pid in alive}
        out = {}
        for p in self.procs:
            port = run.declared_port(p.command)
            if run.is_server(p.command) and port is not None:
                out[p.pid] = {port}
        return out

    def argv_of(self, pid):
        if pid in self._argv:
            a = self._argv[pid]
            return None if a is None else list(a)  # None: unreadable
        return next(p.command.split() for p in self.procs if p.pid == pid)

    def kill(self, pid, sig):
        self.killed.append((pid, sig))
        self.procs = [
            p
            for p in self.procs
            if p.pid != pid and not (self.reap_children and p.ppid == pid)
        ]

    def spawn(self, record, log_dir):
        self.spawned.append(record)
        h = FakeHandle(90000 + len(self.spawned), self, self.spawn_rc)
        self.handles.append(h)
        return h

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
            listeners=self.listeners,
            argv_of=self.argv_of,
            executable=lambda path: True,
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
        m.system(),
        [{"pid": 3, "role": "server", "command": MLX, "stopped": True}],
        tmp_path,
    )
    assert out == [
        {"pid_was": 3, "role": "server", "restarted": False, "why": "already running"}
    ]
    assert m.spawned == []


# --------------------------------------------------------------------------- #
# server safety (2026-09-22): S1-S4, against the layout measured on the Studio
# --------------------------------------------------------------------------- #

#: Measured 2026-09-22 on the Mac Studio, read-only (`ps -wwo pid,ppid,rss,command`,
#: `lsof -nP -iTCP -sTCP:LISTEN -Fpn`, `sysctl kern.procargs2`). RSS ~0.01 GiB
#: each at the time (the models paged out). The caffeinate helpers are CHILDREN of
#: their server, carry the server's argv (so `--port`), and listen on nothing.
_PY = (
    "/opt/homebrew/Cellar/python@3.12/3.12.13_4/Frameworks/Python.framework/"
    "Versions/3.12/Resources/Python.app/Contents/MacOS/Python"
)
_VANTA = "/Users/keanooo7/dev/vanta/.venv/bin/mlx_lm.server"
_LLAMA_BIN = (
    "/Users/keanooo7/.lmstudio/extensions/backends/"
    "llama.cpp-mac-arm64-apple-metal-advsimd-2.39.0/llama-server"
)
_TAIL = ["--chat-template-args", '{"enable_thinking": false}', "--log-level", "INFO"]
_A2051 = [
    _PY,
    _VANTA,
    "--model",
    "mlx-community/Qwen3.8-27B-4bit",
    "--host",
    "127.0.0.1",
    "--port",
    "8090",
    *_TAIL,
]
_A2059 = [
    _PY,
    _VANTA,
    "--model",
    "/Users/keanooo7/dev/vanta/models/fast-nomtp",
    "--host",
    "127.0.0.1",
    "--port",
    "8091",
    *_TAIL,
]
_A2058 = ["/usr/bin/caffeinate", "-s", *_A2051[1:]]
_A2065 = ["/usr/bin/caffeinate", "-s", *_A2059[1:]]
_A2067 = [
    _LLAMA_BIN,
    "--model",
    "/Users/keanooo7/second-brain/models/Qwen3-Embedding-4B-GGUF/"
    "Qwen3-Embedding-4B-Q8_0.gguf",
    "--embedding",
    "--pooling",
    "last",
    "-ub",
    "8192",
    "-c",
    "8192",
    "--host",
    "127.0.0.1",
    "--port",
    "8092",
]
MEASURED_ARGV = {2051: _A2051, 2058: _A2058, 2059: _A2059, 2065: _A2065, 2067: _A2067}
MEASURED_PPID = {2051: 1, 2058: 2051, 2059: 1, 2065: 2059, 2067: 1}
MEASURED_LISTEN = {2051: {8090}, 2059: {8091}, 2067: {8092}}


def measured_machine(**kw):
    procs = [
        run.Proc(pid, MEASURED_PPID[pid], int(0.01 * GIB), 0.0, " ".join(argv))
        for pid, argv in MEASURED_ARGV.items()
    ]
    return FakeMachine(
        procs,
        listeners=MEASURED_LISTEN,
        argv=MEASURED_ARGV,
        reap_children=True,  # caffeinate goes when its server goes
        **kw,
    )


def test_identify_the_measured_layout():
    m = measured_machine()
    servers, helpers, problems = run.identify_servers(
        m.list(), {FakeMachine.OWN}, m.listeners()
    )
    assert [p.pid for p in servers] == [2051, 2059, 2067]
    assert [p.pid for p in helpers] == [2058, 2065]
    assert problems == []


def test_servers_measured_layout_stops_servers_restarts_servers_never_helpers(
    led, tmp_path
):
    m = measured_machine()
    assert _execute(led, m, fake_measure(), tmp=tmp_path) == run.Exit.OK
    # only the three listening servers are signalled; the helpers never are
    assert m.killed == [
        (2051, signal.SIGTERM),
        (2059, signal.SIGTERM),
        (2067, signal.SIGTERM),
    ]
    rec = {r["pid"]: r for r in _rows(led)["c0.stopped_servers"]["value"]}
    assert {p: r["role"] for p, r in rec.items()} == {
        2051: "server",
        2059: "server",
        2067: "server",
        2058: "helper",
        2065: "helper",
    }
    assert rec[2051]["ports"] == [8090] and rec[2067]["ports"] == [8092]
    assert all(r["stopped"] and r["gone"] for r in rec.values())
    # restarted: the three servers, from the venv SCRIPT, argv intact
    assert [r["pid"] for r in m.spawned] == [2051, 2059, 2067]
    a = m.spawned[0]["restart_argv"]
    assert a == _A2051[1:] and a[0] == _VANTA
    assert '{"enable_thinking": false}' in a, "one argument, space and quotes intact"
    assert m.spawned[2]["restart_argv"] == _A2067
    out = {r["pid_was"]: r for r in _rows(led)["c0.restarted_servers"]["value"]}
    assert [p for p, r in out.items() if r["restarted"]] == [2051, 2059, 2067]
    assert out[2058]["restarted"] is False and out[2058]["why"].startswith("helper")
    assert out[2065]["restarted"] is False


@pytest.mark.parametrize(
    "extra,listen,match",
    [
        # a name-matched process that neither listens nor has a server parent
        (
            [run.Proc(3000, 1, 1, 0.0, f"/usr/bin/caffeinate -s {_VANTA} --port 8093")],
            MEASURED_LISTEN,
            "owns no LISTEN",
        ),
        # the port 2051 declares (8090) held by a process that is not a server
        (
            [run.Proc(4000, 1, 1, 0.0, "/usr/bin/python3 -m http.server 8090")],
            {**MEASURED_LISTEN, 2051: {8095}, 4000: {8090}},
            "declares --port 8090",
        ),
    ],
    ids=["name-match-not-server-nor-helper", "declared-port-foreign-owner"],
)
def test_servers_disagreement_refuses_before_stopping_anything(
    led, tmp_path, extra, listen, match
):
    m = measured_machine()
    m.procs = [*extra, *m.procs]
    m._listeners = listen
    with pytest.raises(run.Refusal, match=match):
        _execute(led, m, fake_measure(), tmp=tmp_path)
    assert m.killed == [] and m.spawned == []
    assert json.loads(led.path.read_text())["status"] == "did_not_run"


def test_servers_unrestartable_refuses_before_stopping_anything(led, tmp_path):
    m = measured_machine()
    m._argv = {**MEASURED_ARGV, 2067: None}  # argv unreadable: cannot restart it
    with pytest.raises(run.Refusal, match="cannot establish how to restart"):
        _execute(led, m, fake_measure(), tmp=tmp_path)
    assert m.killed == [] and m.spawned == []


def test_servers_stop_marks_each_kill_so_a_later_failure_still_restarts(led, tmp_path):
    """S1: 10 stops; 11 dies on its own mid-stop (ProcessLookupError); 12's kill
    raises. Both 10 and 11 were stopped, so both are restarted."""
    m = FakeMachine(
        [
            _server(10, 20.0, MLX),
            _server(11, 5.0, LLAMA),
            _server(12, 3.0, MLX.replace("8080", "8082")),
        ]
    )
    real_kill = m.kill

    def kill(pid, sig):
        if pid == 11:
            m.procs = [p for p in m.procs if p.pid != 11]
            raise ProcessLookupError(pid)
        if pid == 12:
            raise PermissionError(pid)
        real_kill(pid, sig)

    sysm = m.system()
    sysm.kill = kill
    with pytest.raises(PermissionError):
        run.execute(led, sysm, fake_measure(), ruling=YIELD, log_dir=tmp_path)
    assert [r["pid"] for r in m.spawned] == [10, 11]
    out = {r["pid_was"]: r for r in _rows(led)["c0.restarted_servers"]["value"]}
    assert out[10]["restarted"] and out[11]["restarted"]
    assert out[12] == {
        "pid_was": 12,
        "role": "server",
        "restarted": False,
        "why": "never stopped",
    }


def _signalling_measure(sig):
    """A measurement that delivers `sig` by calling the INSTALLED handler (no real
    signal is ever sent) during the core sweep."""

    def make(deadline_at):
        inner = fake_measure()(deadline_at)

        def wave(specs):
            if specs[0].tag.startswith("core/t1/k8"):
                signal.getsignal(sig)(sig, None)
            return inner.wave(specs)

        return run.Measure(wave=wave, suite=inner.suite, sampler=inner.sampler)

    return make


@pytest.mark.parametrize(
    "sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"]
)
def test_servers_signal_restarts_and_writes_partial(led, tmp_path, sig):
    before = signal.getsignal(sig)
    m = FakeMachine([_server(10, 20.0, MLX)])
    with pytest.raises(KeyboardInterrupt, match=signal.Signals(sig).name):
        _execute(led, m, _signalling_measure(sig), tmp=tmp_path)
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "partial"
    assert [r["pid"] for r in m.spawned] == [10]
    assert signal.getsignal(sig) is before, "the previous handler is restored"


def test_servers_restart_failure_still_writes_ledger(led, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("restart boom")

    monkeypatch.setattr(run, "restart_servers", boom)
    m = FakeMachine([_server(10, 20.0, MLX)])
    assert _execute(led, m, fake_measure(), tmp=tmp_path) == run.Exit.OK
    doc = json.loads(led.path.read_text())
    assert doc["status"] == "ok"
    (row,) = _rows(led)["c0.restarted_servers"]["value"]
    assert row["restart_failed"] is True and "restart boom" in row["why"]
    assert row["pids_was"] == [10]


def test_servers_restart_alive_check(tmp_path):
    """S3: `restarted` only if the new process is alive RESTART_ALIVE_S later."""
    rec = {
        "pid": 3,
        "role": "server",
        "command": MLX,
        "restart_argv": MLX.split(),
        "stopped": True,
    }
    m = FakeMachine([])
    m.spawn_rc = 1  # dies at once
    (out,) = run.restart_servers(m.system(), [rec], tmp_path)
    assert out["restarted"] is False and out["restart_failed"] is True
    assert "exited rc=1" in out["why"]
    m = FakeMachine([])
    (out,) = run.restart_servers(m.system(), [rec], tmp_path)
    assert out["restarted"] is True and out["new_pid"] == 90001
    assert m.handles[0].polled_at == [pytest.approx(run.RESTART_ALIVE_S)]


def test_restart_argv_runs_the_venv_script_not_the_resolved_interpreter():
    assert run.restart_argv(_A2051) == _A2051[1:]
    assert run.restart_argv(_A2067) == _A2067
    m = ["/opt/homebrew/bin/python3.12", "-m", "mlx_lm.server", "--port", "1"]
    assert run.restart_argv(m) == m  # -m: the interpreter IS the environment
    assert run.restart_argv(["llama-server", "--port", "1"]) is None  # not absolute
    assert run.restart_argv(["/usr/bin/python3", "x.py"]) is None


def test_restart_env_strips_this_runs_venv(tmp_path):
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    base = {
        "PATH": f"{venv_bin}:/opt/homebrew/bin:/usr/bin",
        "VIRTUAL_ENV": str(venv_bin.parent),
        "PYTHONPATH": "/x/src",
        "__PYVENV_LAUNCHER__": "/x/.venv/bin/python",
        "OMP_NUM_THREADS": "1",
        "RSR_TORCH_THREADS": "1",
        "HOME": "/Users/o",
        "HF_HOME": "/Users/o/.hf",
    }
    env = run.restart_env(base, [str(venv_bin)])
    assert env["PATH"] == "/opt/homebrew/bin:/usr/bin"
    assert env["HOME"] == "/Users/o" and env["HF_HOME"] == "/Users/o/.hf"
    for k in ("VIRTUAL_ENV", "PYTHONPATH", "__PYVENV_LAUNCHER__", "OMP_NUM_THREADS"):
        assert k not in env
    assert "RSR_TORCH_THREADS" not in env


def test_parse_lsof_and_procargs2():
    text = "p2051\nf4\nn127.0.0.1:8090\np2067\nf5\nn*:8092\nf6\nn[::1]:8092\n"
    assert run.parse_lsof_listeners(text) == {2051: {8090}, 2067: {8092}}
    argv = ["/bin/x", "--a", '{"k": false}']
    raw = (
        len(argv).to_bytes(4, "little")
        + b"/bin/x\0\0\0\0"
        + b"\0".join(a.encode() for a in argv)
        + b"\0HOME=/Users/o\0"
    )
    assert run.parse_procargs2(raw, "little") == argv
    assert run.declared_port("x --port 8090 --y") == 8090
    assert run.declared_port("x --port=8091") == 8091
    assert run.declared_port("x --portal 1") is None


# --------------------------------------------------------------------------- #
# S5 start barrier and S6 thread count
# --------------------------------------------------------------------------- #


def test_barrier_releases_only_when_every_job_is_ready(tmp_path):
    ready = [tmp_path / f"job{j}" / "ready" for j in range(3)]
    for r in ready:
        r.parent.mkdir()
    go = tmp_path / "go"
    t = [0.0]
    seen = []

    def sleep(s):
        assert not go.exists(), "released before every job was ready"
        n = len(seen)
        if n < len(ready):
            ready[n].write_text("x")
        seen.append(n)
        t[0] += s

    run.barrier_release(
        ready, go, exited=lambda: [], now=lambda: t[0], sleep=sleep, deadline_at=1e9
    )
    assert go.exists() and all(r.exists() for r in ready)
    assert len(seen) == 3


def test_barrier_a_job_never_ready_fails_the_wave(tmp_path):
    ready = [tmp_path / "a", tmp_path / "b"]
    ready[0].write_text("x")
    go = tmp_path / "go"
    t = [0.0]

    def sleep(s):
        t[0] += 1.0

    with pytest.raises(run.JobFailed, match="not ready after"):
        run.barrier_release(
            ready,
            go,
            exited=lambda: [],
            now=lambda: t[0],
            sleep=sleep,
            deadline_at=1e9,
            timeout_s=5.0,
        )
    assert not go.exists()
    with pytest.raises(run.JobFailed, match="exited before becoming ready"):
        run.barrier_release(
            ready,
            go,
            exited=lambda: ["det/t1/k2"],
            now=lambda: 0.0,
            sleep=sleep,
            deadline_at=1e9,
        )
    assert not go.exists()


def test_barrier_child_waits_for_go(tmp_path):
    ready, go = tmp_path / "ready", tmp_path / "go"
    calls = []

    def sleep(s):
        assert ready.exists(), "ready is announced before waiting"
        calls.append(s)
        if len(calls) == 4:
            go.write_text("x")

    info = run.wait_at_barrier(ready, go, sleep=sleep)
    assert len(calls) == 4 and info["go_seen_wall"] >= info["ready_wall"]
    go.unlink()
    t = [0.0]
    with pytest.raises(TimeoutError):
        run.wait_at_barrier(
            tmp_path / "r2",
            go,
            timeout_s=1.0,
            clock=lambda: t[0],
            sleep=lambda s: t.__setitem__(0, t[0] + 0.5),
        )


def test_job_threads_asserted():
    class Torch:
        def __init__(self, obey):
            self.n, self.obey = 16, obey

        def set_num_threads(self, n):
            if self.obey:
                self.n = n

        def get_num_threads(self):
            return self.n

    assert run.apply_threads(5, Torch(True)) == 5
    with pytest.raises(RuntimeError, match="spec says 5"):
        run.apply_threads(5, Torch(False))
    assert run.apply_threads(None, Torch(False)) is None  # uncapped MPS job
    argv = run.job_argv(
        run.JobSpec("det", "training", 5, 3, 0, 1, "cpu", 10), Path("/r/j")
    )
    assert argv[argv.index("--threads") + 1] == "5"
    assert argv[argv.index("--go-file") + 1] == "/r/go"
    mps = run.job_argv(run.mps_wave()[0], Path("/r/m"))
    assert "--threads" not in mps


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
    assert all(r.result["spec_threads"] == 1 for r in rs)
    # S5: no job passed the start barrier before every job of the wave was ready
    b = [r.result["barrier"] for r in rs]
    assert max(x["ready_wall"] for x in b) <= min(x["go_seen_wall"] for x in b)


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
