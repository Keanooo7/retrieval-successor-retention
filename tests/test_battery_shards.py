"""I2 (PLAN-v4 §4; red-team M-13): N isolated battery shards, one merged verdict.

`scripts/battery_shards.py` runs N battery processes concurrently, each on a
disjoint slice of ``MUTATIONS`` in its own I1 pool, and merges their verdict JSONs
into the JSON a serial ``--json`` run writes. What this file exists to catch:

* **a mutation that no shard ran reading as a pass.** A shard that was killed,
  refused, exited outside the protocol, or returned a rows file that is short,
  stale or about other mutations makes each of its mutations ``DID_NOT_RUN`` in the
  merged JSON and the driver exit 3 -- never 0;
* **a partition that is not a partition.** Every mutation in exactly one slice;
* **shards sharing a pool**, which I1's ``flock`` would turn into N-1 refusals.

The unit tests fake ``mutation_battery.main``; three end-to-end tests run the real
driver on the I1 stub repository (``tests/_battery_shards_stub.py``). No test here
runs a real suite, and none measures throughput: that is I2's later measurement.

Every test here is the gate of one mutation in `scripts/mutation_battery.py`
(``# --- I2: the sharded driver ---``), except the three end-to-end tests that are
reddened as DECLARED couplings of those mutations rather than as a gate; the table
says which.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tests"))

import battery_isolation as iso  # noqa: E402
import battery_shards as bs  # noqa: E402
import mutation_battery as mb  # noqa: E402

from _battery_stub import build_stub, clean_env  # noqa: E402

STUB = _REPO / "tests" / "_battery_stub.py"
SHARDS_STUB = _REPO / "tests" / "_battery_shards_stub.py"
SHA = "a" * 40
RUN = "run-1"


# --------------------------------------------------------------------- helpers


def _mutation(i: int) -> mb.Mutation:
    return mb.Mutation(f"m{i}", f"test_gate_{i}", "src/a.py", f"x = {i}", "y", f"why {i}")


def _proven(m: mb.Mutation) -> dict:
    return mb._row(m, {f"tests/t.py::{m.gate}": "boom"}, None)


def _serial(rows: list[dict]) -> str:
    """What `mutation_battery._report` writes for ``--json``, byte for byte."""
    return json.dumps(rows, indent=2) + "\n"


@pytest.fixture
def table(tmp_path, monkeypatch):
    """Five fake mutations on a fake tree at a fake pinned SHA."""
    ms = tuple(_mutation(i) for i in range(5))
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setattr(mb, "MUTATIONS", ms)
    monkeypatch.setattr(mb, "ROOT", tree)
    monkeypatch.setattr(iso, "pinned_sha", lambda root: SHA)
    monkeypatch.setattr(iso, "default_pool_dir", lambda root: tmp_path / "default-pool")
    # restored on teardown whatever a shard run sets it to
    monkeypatch.setenv("RSR_BATTERY_THREADS", "7")
    return ms


@pytest.fixture
def work(tmp_path):
    return tmp_path / "work"


def _write_shard(work, k, n, rows, *, rc=0, done=True, **override):
    """What shard ``k`` of ``n`` leaves behind: its rows file and its done record."""
    p = bs.shard_paths(work, k, n)
    p.dir.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        p.rows.write_text(_serial(rows))
    if done:
        record = {
            "run_id": RUN,
            "index": k,
            "of": n,
            "sha": SHA,
            "table": bs.table_fingerprint(mb.MUTATIONS),
            "rc": rc,
        }
        record.update(override)
        p.done.write_text(json.dumps(record))


def _write_all(work, n, ms, skip=()):
    for k, idx in enumerate(bs.partition(len(ms), n)):
        if k not in skip:
            _write_shard(work, k, n, [_proven(ms[i]) for i in idx])


def _merge_argv(work, n, *extra) -> list[str]:
    return ["merge", "--shards", str(n), "--run-id", RUN, "--work-dir", str(work), *extra]


def _merge(work, n, *extra) -> tuple[int, list[dict]]:
    rc = bs.main(_merge_argv(work, n, *extra))
    return int(rc), json.loads((work / "merged.json").read_text())


def _verdicts(rows: list[dict]) -> list[str]:
    return [r["verdict"] for r in rows]


# ------------------------------------------------------------- the partition


def test_the_slices_cover_every_mutation_exactly_once():
    for total in [*range(1, 41), 349]:
        for n in range(1, min(total, 8) + 1):
            parts = bs.partition(total, n)
            assert len(parts) == n
            flat = [i for p in parts for i in p]
            assert sorted(flat) == list(range(total)), (total, n)
            sizes = [len(p) for p in parts]
            assert max(sizes) - min(sizes) <= 1, (total, n, sizes)
            # within a shard, MUTATIONS order -- as the serial battery runs them
            assert all(p == sorted(p) for p in parts)
    # the rule is the stride: shard k of n gets k, k+n, k+2n, ...
    assert bs.partition(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]


@pytest.mark.parametrize(
    ("parts", "why"),
    [
        ([[0], [2]], r"missing indices \[1\]"),
        ([[0, 1], [1, 2]], r"run twice \[1\]"),
        ([[0, 1], [2, 3]], r"unknown indices \[3\]"),
    ],
    ids=["missing", "twice", "out-of-range"],
)
def test_a_partition_that_is_not_exactly_once_is_refused(parts, why):
    with pytest.raises(bs.Refused, match=why):
        bs.check_partition(parts, 3)


def test_more_shards_than_mutations_is_refused(table, work):
    with pytest.raises(bs.Refused, match="no mutations"):
        bs.partition(5, 6)
    with pytest.raises(SystemExit) as e:
        bs.main(_merge_argv(work, 6))
    assert e.value.code == 3


# ------------------------------------------------------------- the layout


def test_every_shard_has_its_own_pool(work):
    n = 6
    pools = [bs.shard_paths(work, k, n).pool for k in range(n)]
    assert len(set(pools)) == n
    for a in pools:
        assert iso.is_under(a, work)
        for b in pools:
            # I1: a pool's prune removes every worktree registered under it
            assert a == b or not iso.is_under(a, b)


def test_a_work_dir_inside_the_invoking_tree_is_refused(table):
    inside = mb.ROOT / "shards"
    with pytest.raises(SystemExit) as e:
        bs.main(_merge_argv(inside, 2))
    assert e.value.code == 3
    assert not inside.exists()


def test_a_work_dir_inside_the_default_pool_is_refused(table, tmp_path):
    """The serial battery's teardown prunes every worktree under the default pool."""
    inside = tmp_path / "default-pool" / "i2"
    with pytest.raises(SystemExit) as e:
        bs.main(_merge_argv(inside, 2))
    assert e.value.code == 3
    assert not inside.exists()


def test_a_dirty_invoking_tree_is_did_not_run_not_a_traceback(table, work, monkeypatch):
    def dirty(root):
        raise iso.Unisolated(f"{root} has uncommitted changes")

    monkeypatch.setattr(iso, "pinned_sha", dirty)
    with pytest.raises(SystemExit) as e:
        bs.main(_merge_argv(work, 2))
    assert e.value.code == 3


# ------------------------------------------------------------- one shard


def _fake_battery(seen: dict):
    def main():
        seen["argv"] = list(sys.argv)
        seen["names"] = [m.name for m in mb.MUTATIONS]
        seen["threads"] = os.environ.get("RSR_BATTERY_THREADS")
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(_serial([_proven(m) for m in mb.MUTATIONS]))
        return mb.Exit.OK

    return main


def _shard(work, k, n, *extra):
    argv = ["shard", "--index", str(k), "--of", str(n), "--run-id", RUN]
    return bs.main([*argv, "--work-dir", str(work), *extra])


def test_a_shard_runs_only_its_own_slice(table, work, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(mb, "main", _fake_battery(seen))
    assert _shard(work, 1, 2) == mb.Exit.OK
    assert seen["names"] == ["m1", "m3"]
    # the table is put back: the driver's own view of it is never a slice
    assert mb.MUTATIONS is table


def test_a_shard_passes_its_own_shard_dir_to_the_battery(table, work, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(mb, "main", _fake_battery(seen))
    _shard(work, 1, 2)
    p = bs.shard_paths(work, 1, 2)
    argv = seen["argv"]
    assert argv[argv.index("--shard-dir") + 1] == str(p.pool)
    assert argv[argv.index("--json") + 1] == str(p.rows)
    assert "--check" not in argv and "--keep-shards" not in argv


def test_a_shard_suite_runs_at_one_thread_by_default(table, work, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(mb, "main", _fake_battery(seen))
    _shard(work, 0, 2)
    assert seen["threads"] == "1"


def test_a_shard_records_what_it_ran_when_it_finishes(table, work, monkeypatch):
    monkeypatch.setattr(mb, "main", _fake_battery({}))
    _shard(work, 0, 2)
    done = json.loads(bs.shard_paths(work, 0, 2).done.read_text())
    assert {k: done[k] for k in ("run_id", "index", "of", "sha", "table", "rc")} == {
        "run_id": RUN,
        "index": 0,
        "of": 2,
        "sha": SHA,
        "table": bs.table_fingerprint(table),
        "rc": 0,
    }
    assert done["n_mutations"] == 3 and done["threads"] == 1 and done["wall_s"] >= 0


def test_a_shard_whose_battery_refused_records_the_refusal(table, work, monkeypatch):
    def refused():
        mb.refuse(mb.Exit.DID_NOT_RUN, "the suite is not green before mutating")

    monkeypatch.setattr(mb, "main", refused)
    assert _shard(work, 0, 2) == mb.Exit.DID_NOT_RUN
    p = bs.shard_paths(work, 0, 2)
    assert json.loads(p.done.read_text())["rc"] == 3
    assert not p.rows.exists()


def test_a_shard_whose_battery_crashed_leaves_no_done_record(table, work, monkeypatch):
    def crashed():
        raise RuntimeError("boom")

    monkeypatch.setattr(mb, "main", crashed)
    with pytest.raises(RuntimeError):
        _shard(work, 0, 2)
    assert not bs.shard_paths(work, 0, 2).done.exists()
    assert mb.MUTATIONS is table


def test_a_shard_never_overwrites_an_earlier_result(table, work, monkeypatch):
    def must_not_run():
        raise AssertionError("the battery ran over an earlier result")

    monkeypatch.setattr(mb, "main", must_not_run)
    _write_shard(work, 0, 2, [_proven(table[i]) for i in (0, 2, 4)])
    before = bs.shard_paths(work, 0, 2).rows.read_text()
    with pytest.raises(SystemExit) as e:
        _shard(work, 0, 2)
    assert e.value.code == 3
    assert bs.shard_paths(work, 0, 2).rows.read_text() == before


# ------------------------------------------------------------- the merge


def test_merged_rows_are_in_table_order_with_the_serial_schema(table, work, monkeypatch):
    """Shards (fake batteries) then the merge: byte for byte the serial ``--json``."""
    monkeypatch.setattr(mb, "main", _fake_battery({}))
    for k in range(3):
        assert _shard(work, k, 3) == mb.Exit.OK
    rc, rows = _merge(work, 3)
    assert rc == 0
    assert [r["mutation"] for r in rows] == [m.name for m in table]
    assert (work / "merged.json").read_text() == _serial([_proven(m) for m in table])


def test_a_shard_with_no_done_record_is_did_not_run(table, work):
    """A SIGKILLed shard: whatever rows file is there, it is not a finished run."""
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[i]) for i in (1, 3)], done=False)
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert {r["mutation"]: r["verdict"] for r in rows} == {
        "m0": "PROVEN",
        "m1": "DID_NOT_RUN",
        "m2": "PROVEN",
        "m3": "DID_NOT_RUN",
        "m4": "PROVEN",
    }
    by_name = {r["mutation"]: r for r in rows}
    assert "no done record" in by_name["m1"]["did_not_run"]
    assert "shard 1 of 2" in by_name["m1"]["did_not_run"]


def test_a_shard_that_wrote_no_rows_is_did_not_run(table, work):
    """A shard whose battery refused (a red baseline, no isolation): rc 3, no rows."""
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, None, rc=3)
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m1"]["verdict"] == by_name["m3"]["verdict"] == "DID_NOT_RUN"
    assert "has no rows" in by_name["m1"]["did_not_run"]


def test_a_missing_row_is_did_not_run_never_a_pass(table, work):
    _write_all(work, 2, table, skip=(0,))
    _write_shard(work, 0, 2, [_proven(table[0]), _proven(table[2])])  # m4 is missing
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m4"]["verdict"] == "DID_NOT_RUN"
    assert "2 rows for 3 mutations" in by_name["m4"]["did_not_run"]
    # the other shard's verdicts stand
    assert by_name["m1"]["verdict"] == by_name["m3"]["verdict"] == "PROVEN"


def test_a_row_for_another_mutation_is_did_not_run(table, work):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[1]), _proven(table[4])])  # m4, not m3
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m3"]["verdict"] == "DID_NOT_RUN"
    assert "'m4'" in by_name["m3"]["did_not_run"]
    assert by_name["m1"]["verdict"] == "PROVEN"


def test_a_row_without_the_serial_rows_fields_is_did_not_run(table, work):
    _write_all(work, 2, table, skip=(1,))
    short = {k: v for k, v in _proven(table[3]).items() if k != "off_gate"}
    _write_shard(work, 1, 2, [_proven(table[1]), short])
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m3"]["verdict"] == "DID_NOT_RUN"
    assert "off_gate" in by_name["m3"]["did_not_run"]


@pytest.mark.parametrize(
    "stale",
    [
        {"run_id": "run-0"},
        {"sha": "b" * 40},
        {"table": "0" * 64},
        {"index": 0},
        {"of": 3},
    ],
    ids=["another-run", "another-sha", "another-table", "another-index", "another-n"],
)
def test_a_shard_record_from_another_run_is_did_not_run(table, work, stale):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[i]) for i in (1, 3)], **stale)
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m1"]["verdict"] == by_name["m3"]["verdict"] == "DID_NOT_RUN"
    assert by_name["m0"]["verdict"] == "PROVEN"


@pytest.mark.parametrize("rc", [1, 2, 137, 143, None], ids=str)
def test_a_shard_that_exited_outside_0_and_3_is_did_not_run(table, work, rc):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[i]) for i in (1, 3)], rc=rc)
    got, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert got == 3
    assert by_name["m1"]["verdict"] == by_name["m3"]["verdict"] == "DID_NOT_RUN"


def test_a_shards_own_did_not_run_row_is_kept_with_its_reason(table, work):
    """Exit 3 with rows: one of the shard's mutations did not run, the rest did."""
    _write_all(work, 2, table, skip=(1,))
    dnr = mb._row(table[3], {}, "pytest was killed by signal 9")
    _write_shard(work, 1, 2, [_proven(table[1]), dnr], rc=3)
    rc, rows = _merge(work, 2)
    by_name = {r["mutation"]: r for r in rows}
    assert rc == 3
    assert by_name["m1"]["verdict"] == "PROVEN"
    assert by_name["m3"]["did_not_run"] == "pytest was killed by signal 9"


def test_zero_mutations_is_unknown_not_a_pass(table, work, monkeypatch):
    monkeypatch.setattr(mb, "MUTATIONS", ())
    assert bs.main(_merge_argv(work, 1)) == mb.Exit.UNKNOWN
    assert bs.main(["run", "--shards", "1", "--work-dir", str(work)]) == mb.Exit.UNKNOWN


def test_check_fails_on_an_unproven_merged_gate(table, work):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[1]), mb._row(table[3], {}, None)])
    rc, rows = _merge(work, 2)
    assert rc == 0
    assert {r["mutation"]: r["verdict"] for r in rows}["m3"] == "ADDS NOTHING"
    rc, _ = _merge(work, 2, "--check")
    assert rc == 1


# ------------------------------------------------------------- launching


def test_a_shard_is_launched_under_one_cpu_det_slot(work, monkeypatch):
    monkeypatch.setattr(bs, "SHARD_PREFIX", ["PY", "battery_shards.py"])
    argv, env = bs.launch(
        1, 4, RUN, work, threads=1, slot_wait=600.0, base={"PYTHONPATH": "/else"}
    )
    cut = argv.index("--")
    slot = [sys.executable, "-m", "orchestrator.slot", "run", "--lane", "cpu-det"]
    assert argv[: cut - 1] == [*slot, "--slots", "1", "--wait", "600", "--job-id"]
    shard = ["PY", "battery_shards.py", "shard", "--index", "1", "--of", "4"]
    tail = ["--run-id", RUN, "--work-dir", str(work), "--threads", "1"]
    assert argv[cut + 1 :] == [*shard, *tail]
    assert env["PYTHONPATH"].split(os.pathsep) == [str(mb.ROOT / "scripts"), "/else"]
    # without a slot: the bare shard command, the environment untouched
    assert bs.launch(1, 4, RUN, work, threads=1, slot_wait=None, base={"A": "b"}) == (
        [*shard, *tail],
        {"A": "b"},
    )


def test_every_shard_gets_its_own_slot_job_id(work):
    ids = set()
    for k in range(6):
        argv, _ = bs.launch(k, 6, RUN, work, threads=1, slot_wait=0.0, base={})
        ids.add(argv[argv.index("--job-id") + 1])
    assert len(ids) == 6


def test_plan_prints_one_command_per_shard_and_the_merge(
    table, work, capsys, monkeypatch
):
    def must_not_run(*a, **k):
        raise AssertionError("plan started a process")

    monkeypatch.setattr(subprocess, "Popen", must_not_run)
    argv = ["plan", "--shards", "2", "--run-id", RUN, "--work-dir", str(work)]
    assert bs.main([*argv, "--slot-wait", "60"]) == mb.Exit.OK
    out = capsys.readouterr().out
    assert out.count("-m orchestrator.slot run") == 2
    assert "shard 0 of 2: 3 mutations" in out and "shard 1 of 2: 2 mutations" in out
    assert f"merge --shards 2 --run-id {RUN}" in out
    assert not work.exists()


# ------------------------------------------------------------- run (fake children)


def _child(tmp_path, body: str) -> list[str]:
    """A stand-in shard process; ``body`` sees ``k``, ``n`` and its shard dir ``d``."""
    script = tmp_path / "child.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "a = sys.argv\n"
        "k, n = int(a[a.index('--index') + 1]), int(a[a.index('--of') + 1])\n"
        "d = Path(a[a.index('--work-dir') + 1]) / f'shard-{k}-of-{n}'\n" + body
    )
    return [sys.executable, str(script)]


def test_a_shard_process_that_dies_makes_its_mutations_did_not_run(
    table, work, tmp_path, monkeypatch
):
    monkeypatch.setattr(bs, "SHARD_PREFIX", _child(tmp_path, "sys.exit(9 if k else 0)\n"))
    out = tmp_path / "out.json"
    rc = bs.main(["run", "--shards", "2", "--work-dir", str(work), "--json", str(out)])
    assert rc == mb.Exit.DID_NOT_RUN
    rows = json.loads(out.read_text())
    # shard 0 exited 0 and returned nothing: that is not a pass either
    assert _verdicts(rows) == ["DID_NOT_RUN"] * 5
    assert sorted(r["mutation"] for r in rows) == [m.name for m in table]
    summary = json.loads((work / "summary.json").read_text())
    assert [s["process_rc"] for s in summary["shards"]] == [0, 9]


def test_run_refuses_a_work_dir_holding_an_earlier_result(table, work, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("run started a shard over an earlier result")

    monkeypatch.setattr(subprocess, "Popen", must_not_run)
    _write_all(work, 2, table)
    with pytest.raises(SystemExit) as e:
        bs.main(["run", "--shards", "2", "--work-dir", str(work)])
    assert e.value.code == 3


def test_run_starts_every_shard_before_waiting_on_any(table, work, tmp_path, monkeypatch):
    """Concurrent, not serial: each child waits until ALL children have started."""
    body = (
        "(d / 'started').write_text('')\n"
        "end = time.monotonic() + 5\n"
        "while time.monotonic() < end:\n"
        "    if all((d.parent / f'shard-{j}-of-{n}' / 'started').exists()"
        " for j in range(n)):\n"
        "        sys.exit(0)\n"
        "    time.sleep(0.02)\n"
        "sys.exit(7)\n"
    )
    monkeypatch.setattr(bs, "SHARD_PREFIX", _child(tmp_path, body))
    bs.main(["run", "--shards", "3", "--work-dir", str(work)])
    summary = json.loads((work / "summary.json").read_text())
    assert [s["process_rc"] for s in summary["shards"]] == [0, 0, 0]


# ------------------------------------------------------------- end to end (stub)


def _porcelain(root: Path) -> str:
    return iso.git(root, "status", "--porcelain").stdout


@pytest.fixture
def stub(tmp_path):
    return build_stub(tmp_path / "stub").resolve()


def _env(tmp_path, **extra: str) -> dict[str, str]:
    """``clean_env`` with HOME moved: the DEFAULT pool is ``~/.cache/rsr-battery``,
    and a driver that failed to pass ``--shard-dir`` must not reach the real one."""
    (tmp_path / "home").mkdir(exist_ok=True)
    return clean_env(HOME=str(tmp_path / "home"), **extra)


def _run_argv(stub, names, work, merged) -> list[str]:
    """The real driver's ``run`` with 2 shards, on the stub."""
    stub_argv = [sys.executable, str(SHARDS_STUB), "drive", str(stub), "good"]
    driver = ["run", "--shards", "2", "--work-dir", str(work), "--json", str(merged)]
    return [*stub_argv, ",".join(names), *driver]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _wait_marker(marker: Path, proc, timeout: float = 60.0) -> tuple[int, int]:
    """``(pytest pid, grandchild pid)`` once the slow mutation's test is running."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if marker.exists():
            a, b = marker.read_text().split()
            return int(a), int(b)
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"driver exited {proc.returncode} early:\n{out}\n{err}")
        time.sleep(0.05)
    raise AssertionError("the slow mutation's pytest never started")


def _parent(pid: int) -> int:
    out = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


class _Slow:
    """The driver on ``[slow, proven]``: shard 0 sleeps in its suite, shard 1 ends.

    On exit nothing it started is left running, whatever the test did or however it
    failed: the driver and the slow shard's battery are asked to stop (each tears
    its own children down), and the slow suite's process group is killed.
    """

    def __init__(self, stub: Path, tmp_path: Path):
        self.marker = tmp_path / "pytest.pid"
        self.merged = tmp_path / "merged.json"
        self.work = tmp_path / "work"
        self.argv = _run_argv(stub, ["slow", "proven"], self.work, self.merged)
        self.env = _env(tmp_path, STUB_MARKER=str(self.marker))
        self.pids: list[int] = []

    def __enter__(self):
        self.proc = subprocess.Popen(
            self.argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            start_new_session=True,
        )
        try:
            self.pytest_pid, self.grandchild = _wait_marker(self.marker, self.proc)
            self.battery = _parent(self.pytest_pid)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.pids = [self.battery, self.pytest_pid, self.grandchild]
        return self

    def __exit__(self, *exc):
        for pid in (self.proc.pid, *self.pids[:1]):
            if _alive(pid):
                os.kill(pid, signal.SIGTERM)
        if self.proc.poll() is None:
            try:
                self.proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.communicate()
        if self.pids and not _wait_gone(self.battery, 60):
            os.kill(self.battery, signal.SIGKILL)
        if self.pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pytest_pid, signal.SIGKILL)


def test_sharded_verdicts_equal_the_serial_ones_on_the_stub(stub, tmp_path):
    names = ["proven", "leaks", "adds-nothing"]
    serial = tmp_path / "serial.json"
    proc = subprocess.run(
        [
            *[sys.executable, str(STUB), "drive", str(stub), "good", ",".join(names)],
            *["--shard-dir", str(tmp_path / "pool"), "--json", str(serial)],
        ],
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged, work = tmp_path / "merged.json", tmp_path / "work"
    proc = subprocess.run(
        _run_argv(stub, names, work, merged),
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(merged.read_text()) == json.loads(serial.read_text())
    assert _verdicts(json.loads(merged.read_text())) == [
        "PROVEN",
        "LEAKS",
        "ADDS NOTHING",
    ]
    # each shard ran in its own pool, and nothing is left behind
    pools = set()
    for k in range(2):
        p = bs.shard_paths(work, k, 2)
        done = json.loads(p.done.read_text())
        assert done["rc"] == 0
        assert (p.pool / "pool.lock").exists(), f"shard {k} did not use its own pool"
        pools.add(done["pool"])
    assert len(pools) == 2
    assert not (tmp_path / "home" / ".cache").exists()
    assert _porcelain(stub) == ""
    assert iso._registered(stub) == {stub}


def test_a_sigkilled_shard_is_did_not_run_and_the_other_shards_verdicts_stand(
    stub, tmp_path
):
    with _Slow(stub, tmp_path) as run:
        os.kill(run.battery, signal.SIGKILL)  # shard 0's battery: no handler runs
        out, err = run.proc.communicate(timeout=60)
        assert run.proc.returncode == 3, out + err
        rows = {r["mutation"]: r for r in json.loads(run.merged.read_text())}
        assert rows["slow"]["verdict"] == "DID_NOT_RUN"
        assert "no done record" in rows["slow"]["did_not_run"]
        assert rows["proven"]["verdict"] == "PROVEN"
        assert _porcelain(stub) == ""


def test_a_stopped_driver_stops_its_shards_and_writes_no_verdict(stub, tmp_path):
    with _Slow(stub, tmp_path) as run:
        os.kill(run.proc.pid, signal.SIGTERM)
        out, err = run.proc.communicate(timeout=60)
        assert run.proc.returncode == 128 + signal.SIGTERM, out + err
        assert "INTERRUPTED by SIGTERM" in err
        assert not run.merged.exists()
        assert _wait_gone(run.battery) and _wait_gone(run.pytest_pid)
        assert _wait_gone(run.grandchild)
        assert _porcelain(stub) == ""
        assert iso._registered(stub) == {stub}
