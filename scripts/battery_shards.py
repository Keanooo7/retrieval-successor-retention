"""I2 (PLAN-v4 §4; red-team M-13): N isolated battery shards, one merged verdict.

`scripts/mutation_battery.py` runs every mutation serially in one I1 shard. This
driver runs ``N`` battery processes at once, each on a disjoint slice of
``MUTATIONS`` and each in its **own** I1 pool (`battery_isolation`: a pool's
``flock`` admits one battery), and merges their verdict JSONs into the JSON a
serial ``--json`` run writes -- same row schema, ``MUTATIONS`` order -- so a serial
and a sharded run at one SHA can be compared mutation by mutation.

    .venv/bin/python scripts/battery_shards.py run --shards N --work-dir W --check
    .venv/bin/python scripts/battery_shards.py plan --shards N --work-dir W \\
        --slot-wait S                     # print the per-shard slot commands
    .venv/bin/python scripts/battery_shards.py shard --index K --of N \\
        --run-id ID --work-dir W          # one shard (what `run` spawns N of)
    .venv/bin/python scripts/battery_shards.py merge --shards N --run-id ID \\
        --work-dir W --check              # merge shards launched by hand

**Partition -- by stride.** Shard ``k`` of ``N`` gets ``MUTATIONS[k::N]``. Every
mutation costs one full suite run, so equal counts are equal work to first order;
the stride also spreads a run of neighbouring entries (one file, one feature, the
same fast or slow failure) over all shards instead of loading one, and it needs no
cost model and no recorded timings. Sizes differ by at most one, and each shard
runs its slice in ``MUTATIONS`` order. `check_partition` refuses anything that is
not "every index exactly once, no empty shard" before a shard starts and again
before a merge.

**Merge -- a row is evidence only if its shard finished this run.** Each shard
leaves ``W/shard-K-of-N/rows.json`` (the battery's own ``--json``) and, only when
its battery returned, ``done.json``: the run id, its index and N, the pinned SHA,
a fingerprint of the whole table, and the battery's exit status. `merge_rows`
takes a shard's rows only if that record exists, names THIS run, shard, SHA and
table, and carries exit status 0 or 3 (3: some mutation of its own did not run --
its other rows are real); and then only a row that is at its mutation's position,
names that mutation and gate, and has every field of the serial row. **Every other
mutation becomes a ``DID_NOT_RUN`` row** built by the battery's own ``_row``, with
the reason. A missing row never reads as a pass: there is no code path that drops
a mutation from the merged list.

**Exit code -- the serial battery's, on the merged rows.** The merged rows go
through `mutation_battery._report`, so the rule is the serial one by construction:
3 if any row is ``DID_NOT_RUN``, else 1 under ``--check`` if any gate is unproven,
else 0. Zero mutations is 2 (nothing to compare), never 0. A refusal before any
shard starts -- a work dir inside the tree or the default pool, an earlier result
in it, more shards than mutations -- is 3.

**Under the lane scheduler.** With ``--slot-wait S`` every shard is launched as

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det \\
        --slots 1 --wait S --job-id <run-id>-s<K>of<N> -- \\
        .venv/bin/python scripts/battery_shards.py shard --index K --of N ...

one cpu-det slot per shard, so the scheduler's thread cap (1) and this driver's
``--threads`` (default 1, exported as ``RSR_BATTERY_THREADS``) agree. A slot that
is refused exits 3 and starts no shard: no done record, so its mutations are
``DID_NOT_RUN``. ``plan`` prints the same commands for launching them by hand,
followed by the ``merge`` that collects them.

⚠️ `_report`, `_row`, `signal_handlers` and `_Signalled` are `mutation_battery`'s
own. Importing them is the point: the merged JSON and the exit rule cannot drift
from the serial battery's while they are the same functions.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent / "src"))
sys.path.insert(0, str(_SCRIPTS))

import battery_isolation as iso  # noqa: E402
import mutation_battery as mb  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status  # noqa: E402

__all__ = [
    "Refused",
    "check_partition",
    "launch",
    "main",
    "merge_rows",
    "partition",
    "shard_paths",
    "table_fingerprint",
]

SHARD_PREFIX: list[str] = [sys.executable, str(Path(__file__).resolve())]
"""What runs one shard, before ``shard --index ...``. A test points it at a stub."""

ROWS_EXIT = (int(Exit.OK), int(Exit.DID_NOT_RUN))
"""The shard exit statuses after which its rows file is a finished battery's."""

_IDENTITY = ("run_id", "index", "of", "sha", "table")


class Refused(Exception):
    """The sharded battery cannot start, or cannot merge: DID NOT RUN (3)."""


# ------------------------------------------------------------------ partition


def partition(total: int, n: int) -> list[list[int]]:
    """``n`` slices of ``range(total)`` by stride: shard ``k`` gets ``k, k+n, ...``.

    ``n < 1`` yields no slices, which `check_partition` refuses as missing them all.
    """
    parts = [list(range(k, total, n)) for k in range(n)]
    check_partition(parts, total)
    return parts


def check_partition(parts: list[list[int]], total: int) -> None:
    """Every index of ``range(total)`` in exactly one slice, and no empty slice."""
    flat = sorted(i for p in parts for i in p)
    if flat != list(range(total)):
        unknown = sorted(set(flat) - set(range(total)))
        missing = sorted(set(range(total)) - set(flat))
        twice = sorted({i for i in flat if flat.count(i) > 1})
        raise Refused(
            f"not a partition of {total} mutations: missing indices {missing}, "
            f"indices run twice {twice}, unknown indices {unknown}"
        )
    empty = [k for k, p in enumerate(parts) if not p]
    if empty:
        raise Refused(
            f"shard(s) {empty} of {len(parts)} would have no mutations "
            f"({total} mutations): use fewer shards"
        )


def table_fingerprint(mutations) -> str:
    """One hash of the whole table -- a shard that imported another table is stale."""
    doc = json.dumps([dataclasses.astuple(m) for m in mutations], sort_keys=True)
    return hashlib.sha256(doc.encode()).hexdigest()


# --------------------------------------------------------------------- layout


@dataclass(frozen=True)
class ShardPaths:
    dir: Path
    rows: Path
    done: Path
    pool: Path
    log: Path


def shard_paths(work: Path, k: int, n: int) -> ShardPaths:
    """Where shard ``k`` of ``n`` keeps everything. Its pool is its own (I1)."""
    d = Path(work) / f"shard-{k}-of-{n}"
    return ShardPaths(d, d / "rows.json", d / "done.json", d / "pool", d / "log.txt")


def check_work_dir(work: Path) -> Path:
    """Outside the invoking tree (it must stay clean) and outside the default pool
    (a serial battery's teardown prunes every worktree registered under that)."""
    work = Path(work).resolve()
    if iso.is_under(work, mb.ROOT):
        raise Refused(
            f"--work-dir {work} is inside the invoking tree {mb.ROOT}: every shard "
            f"requires a clean tree, and its pool must be outside it"
        )
    default = iso.default_pool_dir(mb.ROOT)
    if iso.is_under(work, default):
        raise Refused(
            f"--work-dir {work} is inside the default shard pool {default}, whose "
            f"teardown and --prune-shards remove every worktree under it"
        )
    return work


def pinned_sha() -> str:
    """The clean invoking tree's HEAD (`battery_isolation.pinned_sha`), or `Refused`:
    a dirty tree is DID NOT RUN (3), not a traceback (1)."""
    try:
        return iso.pinned_sha(mb.ROOT)
    except iso.Unisolated as e:
        raise Refused(f"not isolated: {e}") from e


def _write_json(path: Path, doc: object) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ one shard


def run_shard(args) -> int:
    """Run the real battery on slice ``index`` of ``of``; record that it finished.

    The done record is written only when the battery RETURNED -- a status or a
    refusal. A battery that was killed, or that raised, leaves none, and the
    merge then counts every mutation of this shard as ``DID_NOT_RUN``.
    """
    table = mb.MUTATIONS
    parts = partition(len(table), args.of)
    paths = shard_paths(check_work_dir(args.work_dir), args.index, args.of)
    for old in (paths.rows, paths.done):
        if old.exists():
            raise Refused(
                f"{old} exists: shard {args.index} of {args.of} already has a result "
                f"there. Use a fresh --work-dir; nothing was overwritten."
            )
    sha = pinned_sha()
    paths.dir.mkdir(parents=True, exist_ok=True)

    argv = ["mutation_battery.py", "--json", str(paths.rows)]
    argv += ["--shard-dir", str(paths.pool)]
    if args.keep_shards:
        argv.append("--keep-shards")
    os.environ["RSR_BATTERY_THREADS"] = str(args.threads)
    saved_argv = sys.argv
    started, t0 = _utc(), time.perf_counter()
    mb.MUTATIONS = tuple(table[i] for i in parts[args.index])
    sys.argv = argv
    try:
        try:
            rc = int(status(mb.main()))
        except SystemExit as e:
            # a refusal (3), or 128+N after a handled signal: the battery's own word
            rc = e.code if isinstance(e.code, int) else int(Exit.FAIL)
    finally:
        mb.MUTATIONS = table
        sys.argv = saved_argv
    _write_json(
        paths.done,
        {
            "run_id": args.run_id,
            "index": args.index,
            "of": args.of,
            "sha": sha,
            "table": table_fingerprint(table),
            "rc": rc,
            "n_mutations": len(parts[args.index]),
            "threads": args.threads,
            "pool": str(paths.pool),
            "started_utc": started,
            "wall_s": time.perf_counter() - t0,
        },
    )
    return rc


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------- merge


def shard_problem(paths: ShardPaths, want: dict, n_rows: int) -> tuple[list, str | None]:
    """``(rows, None)`` if this shard's files are a finished run of ``want``, else
    ``([], why not)``."""
    done = _read_json(paths.done)
    if not isinstance(done, dict):
        return [], (
            f"no done record at {paths.done} (the shard was killed, was refused "
            f"its slot, or never started)"
        )
    got = {k: done.get(k) for k in _IDENTITY}
    if got != want:
        diff = {k: got[k] for k in _IDENTITY if got[k] != want[k]}
        return [], f"its done record is not this run's: {diff} (want {want})"
    rc = done.get("rc")
    if isinstance(rc, bool) or rc not in ROWS_EXIT:
        return [], f"its battery exited {rc!r}, not 0 or 3: it did not finish"
    rows = _read_json(paths.rows)
    if not isinstance(rows, list) or len(rows) != n_rows:
        n = len(rows) if isinstance(rows, list) else "no"
        return [], (
            f"its battery exited {rc} and {paths.rows} has {n} rows for "
            f"{n_rows} mutations"
        )
    return rows, None


def row_problem(row: object, m: mb.Mutation) -> str | None:
    """Why ``row`` is not the serial battery's row for ``m``, or None."""
    got = row if isinstance(row, dict) else {}
    if got.get("mutation") != m.name or got.get("gate") != m.gate:
        return (
            f"the row at this mutation's position is for {got.get('mutation')!r} "
            f"(gate {got.get('gate')!r})"
        )
    absent = sorted(set(mb._row(m, {}, None)) - set(got))
    if absent:
        return f"its row lacks the serial row's fields {absent}"
    return None


def merge_rows(
    work: Path, n: int, run_id: str, sha: str, mutations
) -> tuple[list[dict], list[str | None]]:
    """One row per mutation, in table order, and each shard's problem (or None).

    🔴 Every mutation gets a row. One that its shard did not return as a finished,
    matching battery row is ``DID_NOT_RUN`` -- built by the battery's own `_row`, so
    `_report` exits 3 on it.
    """
    parts = partition(len(mutations), n)
    table = table_fingerprint(mutations)
    merged: dict[int, dict] = {}
    problems: list[str | None] = []
    for k, idx in enumerate(parts):
        want = {"run_id": run_id, "index": k, "of": n, "sha": sha, "table": table}
        rows, problem = shard_problem(shard_paths(work, k, n), want, len(idx))
        problems.append(problem)
        for j, i in enumerate(idx):
            m = mutations[i]
            why = problem if problem is not None else row_problem(rows[j], m)
            if why is None:
                merged[i] = rows[j]
            else:
                merged[i] = mb._row(m, {}, f"shard {k} of {n}: {why}")
    return [merged[i] for i in range(len(mutations))], problems


def report(args, rows: list[dict]) -> Exit:
    """The serial battery's report and exit rule, on the merged rows."""
    out = args.json or (args.work_dir / "merged.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    view = argparse.Namespace(json=out, markdown=args.markdown, check=args.check)
    return status(mb._report(view, rows))


def run_merge(args) -> Exit:
    work = check_work_dir(args.work_dir)
    sha = pinned_sha()
    rows, problems = merge_rows(work, args.shards, args.run_id, sha, mb.MUTATIONS)
    _print_shards(args.shards, problems)
    args.work_dir = work
    return report(args, rows)


def _print_shards(n: int, problems: list[str | None]) -> None:
    for k, problem in enumerate(problems):
        print(f"shard {k} of {n}: {'rows read' if problem is None else problem}")


# ------------------------------------------------------------------ launching


def launch(
    k: int,
    n: int,
    run_id: str,
    work: Path,
    *,
    threads: int,
    slot_wait: float | None,
    base: dict[str, str],
    keep: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` that runs shard ``k`` of ``n`` -- under one cpu-det slot of
    the lane scheduler when ``slot_wait`` is given."""
    argv = [*SHARD_PREFIX, "shard", "--index", str(k), "--of", str(n)]
    argv += ["--run-id", run_id, "--work-dir", str(work), "--threads", str(threads)]
    if keep:
        argv.append("--keep-shards")
    env = dict(base)
    if slot_wait is None:
        return argv, env
    slot = [sys.executable, "-m", "orchestrator.slot", "run"]
    slot += ["--lane", "cpu-det", "--slots", "1", "--wait", f"{slot_wait:g}"]
    slot += ["--job-id", f"{run_id}-s{k}of{n}", "--"]
    path = [str(mb.ROOT / "scripts"), *env.get("PYTHONPATH", "").split(os.pathsep)]
    env["PYTHONPATH"] = os.pathsep.join(p for p in path if p)
    return [*slot, *argv], env


def _prepare(args) -> tuple[Path, str, str]:
    """``(work dir, pinned sha, run id)``, or `Refused` before anything starts."""
    work = check_work_dir(args.work_dir)
    parts = partition(len(mb.MUTATIONS), args.shards)
    sha = pinned_sha()
    for k in range(len(parts)):
        p = shard_paths(work, k, args.shards)
        for old in (p.rows, p.done):
            if old.exists():
                raise Refused(
                    f"{old} exists: --work-dir {work} holds an earlier result. "
                    f"Use a fresh one; nothing was overwritten."
                )
    run_id = args.run_id or (
        f"i2-{sha[:12]}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}"
    )
    return work, sha, run_id


def run_plan(args) -> Exit:
    work, sha, run_id = _prepare(args)
    parts = partition(len(mb.MUTATIONS), args.shards)
    print(f"run id {run_id} at {sha[:12]}: {len(mb.MUTATIONS)} mutations, stride")
    for k, idx in enumerate(parts):
        argv, env = launch(
            k,
            args.shards,
            run_id,
            work,
            threads=args.threads,
            slot_wait=args.slot_wait,
            base={},
            keep=args.keep_shards,
        )
        pre = "".join(f"{key}={shlex.quote(v)} " for key, v in env.items())
        print(f"shard {k} of {args.shards}: {len(idx)} mutations")
        print(f"  {pre}{shlex.join(argv)}")
    merge = [*SHARD_PREFIX, "merge", "--shards", str(args.shards)]
    merge += ["--run-id", run_id, "--work-dir", str(work), "--check"]
    print(f"then:\n  {shlex.join(merge)}")
    return Exit.OK


def run_all(args) -> Exit:
    work, sha, run_id = _prepare(args)
    # A stale anchor is refused by every shard's own battery before its baseline
    # suite (exit 3, no rows): every mutation is then DID_NOT_RUN here.
    n = args.shards
    print(f"run id {run_id} at {sha[:12]}: {len(mb.MUTATIONS)} mutations, {n} shards")
    t0 = time.perf_counter()
    procs: list[subprocess.Popen] = []
    try:
        with mb.signal_handlers(), contextlib.ExitStack() as logs:
            try:
                for k in range(n):
                    paths = shard_paths(work, k, n)
                    paths.dir.mkdir(parents=True, exist_ok=True)
                    argv, env = launch(
                        k,
                        n,
                        run_id,
                        work,
                        threads=args.threads,
                        slot_wait=args.slot_wait,
                        base=dict(os.environ),
                        keep=args.keep_shards,
                    )
                    log = logs.enter_context(paths.log.open("w"))
                    procs.append(
                        subprocess.Popen(
                            argv,
                            cwd=mb.ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    )
                for p in procs:
                    p.wait()
            finally:
                _stop(procs)
    except mb._Signalled as e:
        name = signal.Signals(e.signum).name
        print(
            f"INTERRUPTED by {name}: {len(procs)} shard(s) stopped, no merged "
            f"verdicts; the invoking tree was never mutated",
            file=sys.stderr,
        )
        raise SystemExit(128 + e.signum) from None
    wall = time.perf_counter() - t0

    rows, shard_problems = merge_rows(work, n, run_id, sha, mb.MUTATIONS)
    _print_shards(n, shard_problems)
    print(f"sharded battery: {n} shards, wall {wall:.1f} s")
    _write_json(
        work / "summary.json",
        {
            "run_id": run_id,
            "sha": sha,
            "shards": [
                {
                    "index": k,
                    "process_rc": procs[k].returncode,
                    "problem": shard_problems[k],
                    "done": _read_json(shard_paths(work, k, n).done),
                }
                for k in range(n)
            ],
            "n_shards": n,
            "threads": args.threads,
            "wall_s": wall,
        },
    )
    args.work_dir = work
    return report(args, rows)


def _stop(procs: list[subprocess.Popen]) -> None:
    """SIGTERM every shard still running -- each battery tears its own pool down
    (and a slot wrapper forwards the signal) -- then reap them all."""
    for p in procs:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                p.send_signal(signal.SIGTERM)
    for p in procs:
        p.wait()


# ------------------------------------------------------------------------ CLI


def _parser() -> ArgumentParser:
    ap = ArgumentParser(prog="battery_shards.py")
    sub = ap.add_subparsers(dest="action", required=True)

    def common(p, *, shards: bool, run_id_required: bool):
        p.add_argument("--work-dir", type=Path, required=True)
        p.add_argument("--run-id", required=run_id_required, default=None)
        if shards:
            p.add_argument("--shards", type=int, required=True)

    def launching(p):
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--keep-shards", action="store_true")
        p.add_argument(
            "--slot-wait",
            type=float,
            default=None,
            help="launch every shard under `orchestrator.slot run --lane cpu-det "
            "--slots 1 --wait SECONDS`",
        )

    def reporting(p):
        p.add_argument("--json", type=Path, default=None)
        p.add_argument("--markdown", type=Path, default=None)
        p.add_argument("--check", action="store_true")

    run = sub.add_parser("run", help="launch N shards, wait, merge")
    common(run, shards=True, run_id_required=False)
    launching(run)
    reporting(run)
    plan = sub.add_parser("plan", help="print the per-shard commands; run nothing")
    common(plan, shards=True, run_id_required=False)
    launching(plan)
    shard = sub.add_parser("shard", help="run one shard's slice of the battery")
    common(shard, shards=False, run_id_required=True)
    shard.add_argument("--index", type=int, required=True)
    shard.add_argument("--of", type=int, required=True)
    shard.add_argument("--threads", type=int, default=1)
    shard.add_argument("--keep-shards", action="store_true")
    merge = sub.add_parser("merge", help="merge shards that already ran")
    common(merge, shards=True, run_id_required=True)
    reporting(merge)
    return ap


def main(argv: list[str] | None = None) -> Exit:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not mb.MUTATIONS:
        # Zero mutations is NOTHING TO COMPARE (2), not a pass: "0/0 proven" has
        # no unproven gate in it -- as in the serial main().
        print("UNKNOWN: the battery has no mutations to run", file=sys.stderr)
        return Exit.UNKNOWN
    try:
        if args.action == "shard":
            rc = run_shard(args)
            if rc not in tuple(int(e) for e in Exit):
                # 128+N after a handled signal: the battery's code, verbatim
                raise SystemExit(rc)
            return Exit(rc)
        action = {"run": run_all, "plan": run_plan, "merge": run_merge}[args.action]
        return action(args)
    except Refused as e:
        refuse(Exit.DID_NOT_RUN, str(e))


if __name__ == "__main__":
    run_main(main)
