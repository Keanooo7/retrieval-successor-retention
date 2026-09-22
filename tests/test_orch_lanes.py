"""The lane scheduler's semaphores (`scripts/orchestrator/lanes.py`).

What each test here guards, and the mutation in `scripts/mutation_battery.py`
(block ``# --- orchestrator: lanes ---``) that shows it red:

* the flock is real -- no more than N holders across processes (mutation: the
  flock becomes a no-op);
* a dead holder releases its slots with no cleanup step;
* ``ops/lanes.json`` absent -> refuse, naming C0 (mutation: absent -> defaults);
* the mps lane refuses when ``free - peak < reserve_gb``.

Capacities come from a *fixture* C0 ledger through ``lanes.generate`` -- never a
hand-typed lanes file, which ``load_config`` refuses.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import lanes  # noqa: E402

PY = sys.executable


def c0_ledger(tmp: Path, **values) -> Path:
    """A fixture ledger in `scripts/ledger.py`'s shape, carrying C0's keys."""
    cpu = values.get("cpu_det_slots", 3)
    vals = {
        "cpu_det_slots": 3,
        "battery_cpu_slots": min(2, cpu),
        "agent_sessions": 2,
        "reserve_gb": 8.0,
        "mps_reserves_cpu_slots": 1,
        **values,
    }
    doc = {
        "run_id": "capacity-c0-fixture",
        "status": "ok",
        "provenance": {"git_sha": "f" * 40, "dirty": False},
        "rows": [
            {
                "key": lanes.C0_LEDGER_KEYS[k],
                "kind": "observation",
                "value": v,
                "how": "fixture",
            }
            for k, v in vals.items()
        ],
    }
    path = tmp / "ledger.json"
    path.write_text(json.dumps(doc))
    return path


def make_root(tmp: Path, **values) -> Path:
    root = tmp / "root"
    (root / "ops").mkdir(parents=True)
    doc = lanes.generate(c0_ledger(tmp, **values))
    (root / lanes.LANES_FILE).write_text(json.dumps(doc))
    return root


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "RSR_ORCH_ROOT": str(root), "PYTHONPATH": str(_REPO / "scripts")}
    return subprocess.run(
        [PY, "-m", "orchestrator.lanes", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=root,
    )


HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {scripts!r})
    from orchestrator import lanes
    root = Path({root!r})
    cfg = lanes.load_config(root)
    try:
        lease = lanes.acquire(root, cfg, {lane!r}, {k}, wait=0)
    except lanes.Refused:
        print("REFUSED", flush=True); sys.exit(3)
    print("HELD", flush=True)
    time.sleep(60)
    """
)


def spawn_holder(root: Path, lane: str = "cpu-det", k: int = 1):
    code = HOLDER.format(scripts=str(_REPO / "scripts"), root=str(root), lane=lane, k=k)
    p = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, text=True)
    first = p.stdout.readline().strip()
    return p, first


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGKILL)
        p.wait()


# ------------------------------------------------------------------ the semaphore


def test_processes_cannot_hold_more_than_N_slots(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=3)
    procs = []
    try:
        results = []
        for _ in range(5):
            p, first = spawn_holder(root)
            procs.append(p)
            results.append(first)
        assert results.count("HELD") == 3, results
        assert results.count("REFUSED") == 2, results
    finally:
        _kill(procs)


def test_a_multi_slot_admission_is_all_or_nothing(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=3)
    procs = []
    try:
        p, first = spawn_holder(root, k=2)
        procs.append(p)
        assert first == "HELD"
        cfg = lanes.load_config(root)
        with pytest.raises(lanes.Refused):
            lanes.acquire(root, cfg, "cpu-det", 2, wait=0)
        # the refused attempt left its partial holding behind? then 1 is not free.
        lease = lanes.acquire(root, cfg, "cpu-det", 1, wait=0)
        lease.release()
    finally:
        _kill(procs)


def test_a_killed_holder_releases_its_slots(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=2)
    p, first = spawn_holder(root, k=2)
    try:
        assert first == "HELD"
        cfg = lanes.load_config(root)
        with pytest.raises(lanes.Refused):
            lanes.acquire(root, cfg, "cpu-det", 1, wait=0)
        p.send_signal(signal.SIGKILL)
        p.wait()
        lease = lanes.acquire(root, cfg, "cpu-det", 2, wait=0)
        assert sorted(lease.slots()["cpu-det"]) == [0, 1]
        lease.release()
    finally:
        _kill([p])


def test_wait_admits_once_a_slot_frees(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=1)
    p, first = spawn_holder(root)
    try:
        assert first == "HELD"
        cfg = lanes.load_config(root)
        killer = subprocess.Popen(
            [
                PY,
                "-c",
                f"import os,signal,time; time.sleep(0.5); "
                f"os.kill({p.pid}, signal.SIGKILL)",
            ]
        )
        t0 = time.monotonic()
        lease = lanes.acquire(root, cfg, "cpu-det", 1, wait=30)
        assert time.monotonic() - t0 < 30
        lease.release()
        killer.wait()
    finally:
        _kill([p])


def test_battery_reserves_cpu_det_slots(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=3, battery_cpu_slots=2)
    cfg = lanes.load_config(root)
    bat = lanes.acquire(root, cfg, "battery", 1)
    try:
        assert len(bat.slots()["cpu-det"]) == 2
        one = lanes.acquire(root, cfg, "cpu-det", 1)
        with pytest.raises(lanes.Refused):
            lanes.acquire(root, cfg, "cpu-det", 1)
        one.release()
    finally:
        bat.release()


def test_asking_for_more_slots_than_the_lane_has_is_refused(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=2)
    cfg = lanes.load_config(root)
    with pytest.raises(lanes.Refused, match="2 slot"):
        lanes.acquire(root, cfg, "cpu-det", 3)


# --------------------------------------------------------------------- lanes.json


def test_missing_lanes_file_is_refused_naming_C0(tmp_path):
    with pytest.raises(lanes.Refused, match="C0"):
        lanes.load_config(tmp_path)


def test_status_cli_without_lanes_file_exits_3(tmp_path):
    r = _cli(tmp_path, "status", "--json")
    assert r.returncode == 3, r
    assert "C0" in r.stderr


def test_a_hand_typed_lanes_file_is_refused(tmp_path):
    (tmp_path / "ops").mkdir()
    (tmp_path / lanes.LANES_FILE).write_text(
        json.dumps(
            {
                "cpu_det_slots": 9,
                "battery_cpu_slots": 2,
                "agent_sessions": 2,
                "reserve_gb": 8,
                "mps_reserves_cpu_slots": 1,
                "generated_from": {"run_id": "typed"},
            }
        )
    )
    with pytest.raises(lanes.Refused, match="hand-typed"):
        lanes.load_config(tmp_path)


def test_generate_names_every_ledger_key(tmp_path):
    doc = lanes.generate(c0_ledger(tmp_path))
    assert doc["generated_from"]["ledger_keys"] == lanes.C0_LEDGER_KEYS
    assert doc["generated_from"]["run_id"] == "capacity-c0-fixture"
    assert doc["generated_from"]["ledger_git_sha"] == "f" * 40
    assert set(lanes.C0_LEDGER_KEYS) <= set(doc)


def test_generate_refuses_a_ledger_missing_a_key(tmp_path):
    path = c0_ledger(tmp_path)
    doc = json.loads(path.read_text())
    doc["rows"] = [r for r in doc["rows"] if r["key"] != "c0.reserve_gb"]
    path.write_text(json.dumps(doc))
    with pytest.raises(lanes.Refused, match=r"c0\.reserve_gb"):
        lanes.generate(path)


def test_generate_refuses_a_ledger_that_is_not_ok(tmp_path):
    path = c0_ledger(tmp_path)
    doc = json.loads(path.read_text())
    doc["status"] = "partial"
    path.write_text(json.dumps(doc))
    with pytest.raises(lanes.Refused, match="partial"):
        lanes.generate(path)


def test_generate_cli_writes_a_loadable_file(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    r = _cli(root, "generate", "--ledger", str(c0_ledger(tmp_path)))
    assert r.returncode == 0, r.stderr
    assert lanes.load_config(root).cpu_det_slots == 3


def test_status_json_names_the_holder(tmp_path):
    root = make_root(tmp_path, cpu_det_slots=2)
    cfg = lanes.load_config(root)
    lease = lanes.acquire(root, cfg, "cpu-det", 1, info={"job_id": "J1"})
    try:
        r = _cli(root, "status", "--json")
        assert r.returncode == 0, r.stderr
        rep = json.loads(r.stdout)["lanes"]["cpu-det"]
        assert rep["free"] == 1
        assert rep["held"][0]["holder"]["job_id"] == "J1"
    finally:
        lease.release()


# --------------------------------------------------------------------- mps memory

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               {free}.
Pages active:                            1000000.
Pages inactive:                           {inactive}.
Pages speculative:                        {spec}.
Pages throttled:                                0.
"""


def _vm(gb_free: float) -> str:
    pages = int(gb_free * 1024**3 / 16384)
    return VM_STAT.format(free=pages, inactive=0, spec=0)


def test_free_memory_sums_free_inactive_speculative():
    text = VM_STAT.format(free=10, inactive=20, spec=30)
    assert lanes.free_memory_bytes(text) == 60 * 16384


def test_mps_admission_refuses_when_free_memory_is_insufficient(tmp_path):
    root = make_root(tmp_path, reserve_gb=8.0)
    cfg = lanes.load_config(root)
    with pytest.raises(lanes.Refused, match="insufficient memory"):
        lanes.acquire(
            root,
            cfg,
            "mps",
            1,
            peak_gb=20.0,
            free_bytes=lambda: lanes.free_memory_bytes(_vm(27.0)),
        )
    # the refusal released the mps slot and its cpu-det reservation
    assert lanes.status(root, cfg)["lanes"]["mps"]["free"] == 1
    assert lanes.status(root, cfg)["lanes"]["cpu-det"]["free"] == 3


def test_mps_admission_admits_when_free_memory_suffices(tmp_path):
    root = make_root(tmp_path, reserve_gb=8.0, mps_reserves_cpu_slots=1)
    cfg = lanes.load_config(root)
    lease = lanes.acquire(
        root,
        cfg,
        "mps",
        1,
        peak_gb=20.0,
        free_bytes=lambda: lanes.free_memory_bytes(_vm(29.0)),
    )
    try:
        assert lease.slots() == {"mps": [0], "cpu-det": [0]}
    finally:
        lease.release()


def test_mps_admission_requires_a_declared_peak(tmp_path):
    root = make_root(tmp_path)
    cfg = lanes.load_config(root)
    with pytest.raises(lanes.Refused, match="peak-gb"):
        lanes.acquire(root, cfg, "mps", 1)
