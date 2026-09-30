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
"""

from __future__ import annotations

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
    monkeypatch.setattr(mb, "anchor_problems", lambda: [])
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


def _merge(work, n, *extra) -> tuple[int, list[dict]]:
    out = work / "merged.json"
    rc = bs.main(
        ["merge", "--shards", str(n), "--run-id", RUN, "--work-dir", str(work), *extra]
    )
    return int(rc), json.loads(out.read_text())


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
            assert len(set(flat)) == len(flat), (total, n)
            sizes = [len(p) for p in parts]
            assert max(sizes) - min(sizes) <= 1, (total, n, sizes)
            # within a shard, MUTATIONS order -- as the serial battery runs them
            assert all(p == sorted(p) for p in parts)


def test_the_partition_is_the_stride_rule():
    assert bs.partition(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert bs.partition(3, 1) == [[0, 1, 2]]


@pytest.mark.parametrize(
    ("parts", "why"),
    [
        ([[0], [2]], "missing"),
        ([[0, 1], [1, 2]], "twice"),
        ([[0, 1, 2, 3]], "unknown"),
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
        bs.main(["merge", "--shards", "6", "--run-id", RUN, "--work-dir", str(work)])
    assert e.value.code == 3


@pytest.mark.parametrize("n", ["0", "-1"])
def test_a_shard_count_below_one_is_refused(table, work, n):
    with pytest.raises(SystemExit) as e:
        bs.main(["merge", "--shards", n, "--run-id", RUN, "--work-dir", str(work)])
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


def test_a_work_dir_inside_the_invoking_tree_is_refused(table, tmp_path):
    inside = mb.ROOT / "shards"
    with pytest.raises(SystemExit) as e:
        bs.main(["merge", "--shards", "2", "--run-id", RUN, "--work-dir", str(inside)])
    assert e.value.code == 3
    assert not inside.exists()


def test_a_work_dir_inside_the_default_pool_is_refused(table, tmp_path):
    """The serial battery's teardown prunes every worktree under the default pool."""
    inside = tmp_path / "default-pool" / "i2"
    with pytest.raises(SystemExit) as e:
        bs.main(["merge", "--shards", "2", "--run-id", RUN, "--work-dir", str(inside)])
    assert e.value.code == 3


# ------------------------------------------------------------- one shard


def _fake_battery(seen: dict, rc=mb.Exit.OK, write=True):
    def main():
        seen["argv"] = list(sys.argv)
        seen["names"] = [m.name for m in mb.MUTATIONS]
        seen["threads"] = os.environ.get("RSR_BATTERY_THREADS")
        if write:
            out = Path(sys.argv[sys.argv.index("--json") + 1])
            out.write_text(_serial([_proven(m) for m in mb.MUTATIONS]))
        return rc

    return main


def _shard(work, k, n, *extra):
    return bs.main(
        [
            "shard",
            "--index",
            str(k),
            "--of",
            str(n),
            "--run-id",
            RUN,
            "--work-dir",
            str(work),
            *extra,
        ]
    )


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
    assert json.loads(bs.shard_paths(work, 0, 2).done.read_text())["threads"] == 1


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
    assert done["n_mutations"] == 3 and done["wall_s"] >= 0


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


def test_shards_then_merge_equal_the_serial_json(table, work, monkeypatch):
    monkeypatch.setattr(mb, "main", _fake_battery({}))
    for k in range(3):
        assert _shard(work, k, 3) == mb.Exit.OK
    rc, _rows = _merge(work, 3)
    assert rc == 0
    assert (work / "merged.json").read_text() == _serial([_proven(m) for m in table])


# ------------------------------------------------------------- the merge


def test_merged_rows_are_in_table_order_with_the_serial_schema(table, work):
    _write_all(work, 2, table)
    rc, rows = _merge(work, 2)
    assert rc == 0
    assert [r["mutation"] for r in rows] == [m.name for m in table]
    assert (work / "merged.json").read_text() == _serial([_proven(m) for m in table])


def test_a_shard_with_no_done_record_is_did_not_run(table, work):
    """A SIGKILLed shard: whatever rows file is there, it is not a finished run."""
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[i]) for i in (1, 3)], done=False)
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert _verdicts(rows) == [
        "PROVEN",
        "DID_NOT_RUN",
        "PROVEN",
        "DID_NOT_RUN",
        "PROVEN",
    ]
    assert "no done record" in rows[1]["did_not_run"]
    assert "shard 1 of 2" in rows[1]["did_not_run"]


def test_a_shard_that_never_started_is_did_not_run(table, work):
    _write_all(work, 2, table, skip=(0,))
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert _verdicts(rows) == [
        "DID_NOT_RUN",
        "PROVEN",
        "DID_NOT_RUN",
        "PROVEN",
        "DID_NOT_RUN",
    ]


def test_a_shard_that_wrote_no_rows_is_did_not_run(table, work):
    """A shard whose battery refused (a red baseline, no isolation): rc 3, no rows."""
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, None, rc=3)
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert _verdicts(rows)[1::2] == ["DID_NOT_RUN", "DID_NOT_RUN"]
    assert "wrote no rows" in rows[1]["did_not_run"]


def test_a_missing_row_is_did_not_run_never_a_pass(table, work):
    _write_all(work, 2, table, skip=(0,))
    _write_shard(work, 0, 2, [_proven(table[0]), _proven(table[2])])  # m4 is missing
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert rows[4]["verdict"] == "DID_NOT_RUN"
    assert "2 rows for 3 mutations" in rows[4]["did_not_run"]
    # the other shard's verdicts stand
    assert rows[1]["verdict"] == rows[3]["verdict"] == "PROVEN"


def test_a_row_for_another_mutation_is_did_not_run(table, work):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[1]), _proven(table[4])])  # m4, not m3
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert _verdicts(rows) == ["PROVEN", "PROVEN", "PROVEN", "DID_NOT_RUN", "PROVEN"]
    assert rows[3]["mutation"] == "m3" and "'m4'" in rows[3]["did_not_run"]


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "GREEN"},
        {"reddened_gate": None},
        {"off_gate": None},
    ],
    ids=["unknown-verdict", "no-reddened-gate", "no-off-gate"],
)
def test_a_malformed_row_is_did_not_run(table, work, bad):
    _write_all(work, 2, table, skip=(1,))
    row = {**_proven(table[3]), **bad}
    _write_shard(work, 1, 2, [_proven(table[1]), row])
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert _verdicts(rows) == ["PROVEN", "PROVEN", "PROVEN", "DID_NOT_RUN", "PROVEN"]


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
    assert rc == 3
    assert _verdicts(rows)[1::2] == ["DID_NOT_RUN", "DID_NOT_RUN"]
    assert _verdicts(rows)[0::2] == ["PROVEN", "PROVEN", "PROVEN"]


@pytest.mark.parametrize("rc", [1, 2, 137, 143, None], ids=str)
def test_a_shard_that_exited_outside_0_and_3_is_did_not_run(table, work, rc):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[i]) for i in (1, 3)], rc=rc)
    got, rows = _merge(work, 2)
    assert got == 3
    assert _verdicts(rows)[1::2] == ["DID_NOT_RUN", "DID_NOT_RUN"]


def test_a_shards_own_did_not_run_row_is_kept_with_its_reason(table, work):
    _write_all(work, 2, table, skip=(1,))
    dnr = mb._row(table[3], {}, "pytest was killed by signal 9")
    _write_shard(work, 1, 2, [_proven(table[1]), dnr], rc=3)
    rc, rows = _merge(work, 2)
    assert rc == 3
    assert rows[1]["verdict"] == "PROVEN"
    assert rows[3]["did_not_run"] == "pytest was killed by signal 9"


def test_zero_mutations_is_unknown_not_a_pass(table, work, monkeypatch):
    monkeypatch.setattr(mb, "MUTATIONS", ())
    rc = bs.main(["merge", "--shards", "1", "--run-id", RUN, "--work-dir", str(work)])
    assert rc == mb.Exit.UNKNOWN
    rc = bs.main(["run", "--shards", "1", "--work-dir", str(work)])
    assert rc == mb.Exit.UNKNOWN


def test_check_fails_on_an_unproven_merged_gate(table, work):
    _write_all(work, 2, table, skip=(1,))
    _write_shard(work, 1, 2, [_proven(table[1]), mb._row(table[3], {}, None)])
    rc, rows = _merge(work, 2)
    assert rc == 0 and rows[3]["verdict"] == "ADDS NOTHING"
    rc, _ = _merge(work, 2, "--check")
    assert rc == 1


# ------------------------------------------------------------- launching


def test_a_shard_command_names_its_index_run_and_work_dir(work, monkeypatch):
    monkeypatch.setattr(bs, "SHARD_PREFIX", ["PY", "battery_shards.py"])
    argv, env = bs.launch(1, 4, RUN, work, threads=1, slot_wait=None, base={"A": "b"})
    assert argv == [
        "PY",
        "battery_shards.py",
        "shard",
        "--index",
        "1",
        "--of",
        "4",
        "--run-id",
        RUN,
        "--work-dir",
        str(work),
        "--threads",
        "1",
    ]
    assert env == {"A": "b"}


def test_a_shard_is_launched_under_one_cpu_det_slot(work, monkeypatch):
    monkeypatch.setattr(bs, "SHARD_PREFIX", ["PY", "battery_shards.py"])
    argv, env = bs.launch(
        1, 4, RUN, work, threads=1, slot_wait=600.0, base={"PYTHONPATH": "/else"}
    )
    cut = argv.index("--")
    assert argv[:cut] == [
        sys.executable,
        "-m",
        "orchestrator.slot",
        "run",
        "--lane",
        "cpu-det",
        "--slots",
        "1",
        "--wait",
        "600",
        "--job-id",
        f"{RUN}-s1of4",
    ]
    assert argv[cut + 1 : cut + 4] == ["PY", "battery_shards.py", "shard"]
    assert env["PYTHONPATH"].split(os.pathsep) == [str(mb.ROOT / "scripts"), "/else"]


def test_every_shard_gets_its_own_slot_job_id(work):
    ids = set()
    for k in range(6):
        argv, _ = bs.launch(k, 6, RUN, work, threads=1, slot_wait=0.0, base={})
        ids.add(argv[argv.index("--job-id") + 1])
    assert len(ids) == 6


def test_plan_prints_one_slot_command_per_shard_and_runs_nothing(
    table, work, capsys, monkeypatch
):
    def must_not_run(*a, **k):
        raise AssertionError("plan started a process")

    monkeypatch.setattr(subprocess, "Popen", must_not_run)
    rc = bs.main(
        [
            "plan",
            "--shards",
            "2",
            "--run-id",
            RUN,
            "--work-dir",
            str(work),
            "--slot-wait",
            "60",
        ]
    )
    out = capsys.readouterr().out
    assert rc == mb.Exit.OK
    assert out.count("orchestrator.slot run --lane cpu-det --slots 1") == 2
    assert "shard 0 of 2: 3 mutations" in out and "shard 1 of 2: 2 mutations" in out
    assert f"merge --shards 2 --run-id {RUN}" in out
    assert not work.exists()


# ------------------------------------------------------------- run (fake children)


def _child(tmp_path, body: str) -> list[str]:
    """A stand-in shard process: ``body`` sees ``rows``/``done`` paths and ``rec``."""
    script = tmp_path / "child.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(_REPO / 'scripts')!r})\n"
        "a = sys.argv\n"
        "k, n = int(a[a.index('--index') + 1]), int(a[a.index('--of') + 1])\n"
        "d = Path(a[a.index('--work-dir') + 1]) / f'shard-{k}-of-{n}'\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "run_id = a[a.index('--run-id') + 1]\n" + body
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
    assert [r["mutation"] for r in rows] == [m.name for m in table]
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


def test_run_refuses_a_stale_anchor_before_any_shard_starts(table, work, monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("run started a shard on a stale table")

    monkeypatch.setattr(subprocess, "Popen", must_not_run)
    monkeypatch.setattr(mb, "anchor_problems", lambda: ["'m0': anchor occurs 0 times"])
    with pytest.raises(SystemExit) as e:
        bs.main(["run", "--shards", "2", "--work-dir", str(work)])
    assert e.value.code == 3


def test_run_starts_every_shard_before_waiting_on_any(table, work, tmp_path, monkeypatch):
    """Concurrent, not serial: each child waits until ALL children have started."""
    body = (
        "import time\n"
        "(d / 'started').write_text('')\n"
        "end = time.monotonic() + 60\n"
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


def _wait_marker(marker: Path, proc, timeout: float = 120.0) -> tuple[int, int]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if marker.exists():
            a, b = marker.read_text().split()
            return int(a), int(b)
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError(f"driver exited {proc.returncode} early:\n{out}\n{err}")
        time.sleep(0.05)
    proc.kill()
    raise AssertionError("the slow mutation's pytest never started")


def _parent(pid: int) -> int:
    out = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, check=True
    )
    return int(out.stdout.strip())


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
        env=clean_env(),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged, work = tmp_path / "merged.json", tmp_path / "work"
    proc = subprocess.run(
        _run_argv(stub, names, work, merged),
        capture_output=True,
        text=True,
        env=clean_env(),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(merged.read_text()) == json.loads(serial.read_text())
    assert _verdicts(json.loads(merged.read_text())) == [
        "PROVEN",
        "LEAKS",
        "ADDS NOTHING",
    ]
    # each shard ran in its own pool, at one thread, and nothing is left behind
    for k in range(2):
        done = json.loads(bs.shard_paths(work, k, 2).done.read_text())
        assert done["rc"] == 0 and done["threads"] == 1
        assert done["pool"] == str(bs.shard_paths(work, k, 2).pool)
    assert _porcelain(stub) == ""
    assert iso._registered(stub) == {stub}


def test_a_sigkilled_shard_is_did_not_run_and_the_other_shards_verdicts_stand(
    stub, tmp_path
):
    marker, merged = tmp_path / "pytest.pid", tmp_path / "merged.json"
    work = tmp_path / "work"
    proc = subprocess.Popen(
        _run_argv(stub, ["slow", "proven"], work, merged),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=clean_env(STUB_MARKER=str(marker)),
        start_new_session=True,
    )
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(_parent(pytest_pid), signal.SIGKILL)  # shard 0's battery process
    out, err = proc.communicate(timeout=120)
    # SIGKILL cannot be handled: the dead shard's suite is still asleep
    os.killpg(pytest_pid, signal.SIGKILL)
    assert _wait_gone(pytest_pid) and _wait_gone(grandchild)
    assert proc.returncode == 3, out + err
    rows = {r["mutation"]: r for r in json.loads(merged.read_text())}
    assert rows["slow"]["verdict"] == "DID_NOT_RUN"
    assert "no done record" in rows["slow"]["did_not_run"]
    assert rows["proven"]["verdict"] == "PROVEN"
    assert _porcelain(stub) == ""


def test_a_stopped_driver_stops_its_shards_and_writes_no_verdict(stub, tmp_path):
    marker, merged = tmp_path / "pytest.pid", tmp_path / "merged.json"
    work = tmp_path / "work"
    proc = subprocess.Popen(
        _run_argv(stub, ["slow", "proven"], work, merged),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=clean_env(STUB_MARKER=str(marker)),
        start_new_session=True,
    )
    pytest_pid, grandchild = _wait_marker(marker, proc)
    os.kill(proc.pid, signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 128 + signal.SIGTERM, out + err
    assert "INTERRUPTED by SIGTERM" in err
    assert not merged.exists()
    assert _wait_gone(pytest_pid) and _wait_gone(grandchild)
    assert _porcelain(stub) == ""
    assert iso._registered(stub) == {stub}
