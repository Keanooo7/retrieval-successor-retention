"""The mutation battery's thread cap (`scripts/mutation_battery.py::suite_threads`).

The battery shares the Mac Studio's cores with the lane scheduler's jobs, so its
pytest runs with a capped thread pool: ``RSR_BATTERY_THREADS``, else the
``battery_cpu_slots`` of ``ops/lanes.json``, else unchanged.

⚠️ Every test deletes ``RSR_BATTERY_THREADS`` and the thread variables first: the
battery may itself be running this file with them set, and a test that reads the
harness's own environment proves nothing (the same trap
`test_evidence_machinery.py`'s census test documents).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

import mutation_battery  # noqa: E402
from orchestrator import lanes  # noqa: E402

from test_orch_lanes import make_root  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RSR_BATTERY_THREADS", raising=False)
    for v in lanes.THREAD_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(mutation_battery, "ROOT", tmp_path / "empty")


def test_the_env_override_caps_every_thread_pool(monkeypatch):
    monkeypatch.setenv("RSR_BATTERY_THREADS", "4")
    env = mutation_battery._suite_env()
    assert {v: env.get(v) for v in lanes.THREAD_ENV_VARS} == dict.fromkeys(
        lanes.THREAD_ENV_VARS, "4"
    )
    assert mutation_battery.suite_threads() == (4, "RSR_BATTERY_THREADS")


def test_lanes_file_supplies_battery_cpu_slots(monkeypatch, tmp_path):
    root = make_root(tmp_path, cpu_det_slots=6, battery_cpu_slots=3)
    monkeypatch.setattr(mutation_battery, "ROOT", root)
    env = mutation_battery._suite_env()
    assert {v: env.get(v) for v in lanes.THREAD_ENV_VARS} == dict.fromkeys(
        lanes.THREAD_ENV_VARS, "3"
    )
    assert "battery_cpu_slots" in mutation_battery.suite_threads()[1]


def test_without_either_the_environment_is_unchanged():
    env = mutation_battery._suite_env()
    assert not any(v in env for v in lanes.THREAD_ENV_VARS)
    threads, source = mutation_battery.suite_threads()
    assert threads is None and "absent" in source


@pytest.mark.parametrize("bad", ["0", "-2", "four", ""])
def test_a_bad_override_did_not_run(monkeypatch, bad):
    monkeypatch.setenv("RSR_BATTERY_THREADS", bad)
    with pytest.raises(SystemExit) as e:
        mutation_battery.suite_threads()
    assert e.value.code == 3


def test_an_invalid_lanes_file_did_not_run(monkeypatch, tmp_path):
    root = tmp_path / "bad"
    (root / "ops").mkdir(parents=True)
    (root / lanes.LANES_FILE).write_text(json.dumps({"cpu_det_slots": 4}))
    monkeypatch.setattr(mutation_battery, "ROOT", root)
    with pytest.raises(SystemExit) as e:
        mutation_battery.suite_threads()
    assert e.value.code == 3
