"""B5 substrate convergence: does arm B's stream answer loss plateau after ckpt 3000?

Pre-registration: `experiments/b5-convergence/PREREG.md`, committed alone at dfda530,
ahead of this file. PLAN-v4 §2B B5, a robustness secondary: **ckpt3000 stays primary
for everything; choosing another substrate checkpoint is D2, owner-only.**

**No new training code.** The child is fresh-stream's resume child
(`experiments/fresh-stream/run.py::single("B")`, imported, not copied): the corpus-size
N = 64 ``train()`` call with ``stream=`` and ``resume=``, inside fresh-stream's
``ResumeCheck``. What is new is what fresh-escape hard-coded:

* **threads** are an argument (``--threads``; PREREG 4 per seed, 3 seeds = 12, the
  cpu-det lane), set on every BLAS/OpenMP variable and on ``torch.set_num_threads``;
* **the arm and the output directory** are arguments. The start checkpoint is COPIED
  into the run's own directory (sha256 checked against the T0 manifest), and every
  segment resumes from the **latest checkpoint in its own directory**. A killed run
  is resumed by running the same command again; it loses at most one window
  (``ckpt_every = WINDOW = 250``; the writes are ``rsr.train.checkpoint``'s atomic ones);
* **protection:** no output directory may lie inside a directory the T0 manifest
  names, or inside ``.worktrees/fresh-stream`` (``assert_writable``).

Modes::

    run.py --run [--report PATH]      # parent: controls, 3 children, measure, ledger
    run.py --prove-resume --workdir W # PREREG control 6 (a)(b)(c), before launch
    run.py --time-segment --steps N --workdir W   # 3 seeds x 4 threads: s/step
    run.py --status                   # rule trace from the heartbeats, read only
    run.py --write-ledger             # rebuild runs/b5-convergence/ledger.json from files
    run.py --render-results           # RESULTS.md from the ledger

Exit codes (`rsr.exit_codes`): 0 a classification or a clean in-progress state was
recorded; 3 did not run or inconclusive (a refused precondition, a control failed,
a measurement raised).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main, status  # noqa: E402

EXPERIMENT = "experiments/b5-convergence/run.py"
RUN_ID = "b5-convergence"
PREREG = "experiments/b5-convergence/PREREG.md"
PREREG_COMMIT = "dfda530"


def _load(name: str, rel: str):
    """Import an experiment script by path. Imported, never copied."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: fresh-stream: the resume child, the instrument, the readouts. ONE instance.
FS = _load("_b5_fs", "experiments/fresh-stream/run.py")
ST = FS.ST
CSC = FS.CSC
RC = FS.RC
S003 = FS.S003

#: THE measurement (PREREG control 3 / readouts): fresh-stream's, by identity.
measure_checkpoint = FS.measure_checkpoint

# --------------------------------------------------------------------------- #
# 🔒 PREREG.md, transcribed. Changing any is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = [0, 1, 2]
ARM = "B"
RESUME_STEP = 3000
CAP = 9000
WINDOW = 250
N_WINDOWS = 16
REL_BOUND = 1.0  # % per 1000 steps, the CI must lie strictly inside (-1, +1)
T_975_14 = 2.1447866879  # Student t, 0.975, 14 df
EVAL_STEPS = tuple(range(RESUME_STEP + WINDOW * N_WINDOWS, CAP + 1, WINDOW))
TRAJECTORY = (4000, 5000, 6000, 7000, 8000, 9000)
THREADS_PER_SEED = 4
CKPT_EVERY = WINDOW
DELTA = FS.DELTA
C_THRESHOLD = FS.C_THRESHOLD
BUCKET = FS.BUCKET
N_TRAIN = FS.CONTROL_ARM
STREAM = FS.STREAM
BATCH = FS.BATCH
HELDOUT = FS.HELDOUT
PROBE = FS.PROBE
VOCAB_DOCUMENTS = FS.VOCAB_DOCUMENTS
DEVICE = FS.DEVICE
REPRO_TOL = 1e-6

MAIN_REPO = Path.home() / "retrieval-successor-retention"
SOURCE_ROOT = MAIN_REPO / ".worktrees" / "fresh-stream" / "runs" / "fresh-stream"
T0_MANIFEST = Path.home() / "rsr-substrate" / "2026-09-27" / "MANIFEST.sha256"
REF_LEDGER = "runs/fresh-stream/ledger.json"
REF_MANIFEST = "runs/fresh-stream/manifest.json"
CONTROL_PREFIX = f"{ARM}.ckpt{RESUME_STEP}."

QUESTION = (
    "Does fresh-stream arm B's stream answer loss reach a plateau after its ckpt 3000 "
    "if the same run is continued on the same never-repeating stream, and if so at "
    "which common step, and does that checkpoint still retrieve through memory?"
)
DECISION_RULE = (
    "A failed control or a measurement that raised makes the result inconclusive "
    "(exit 3). E* is the first evaluation step E in {7000, 7250, ..., 9000} at which "
    "the plateau rule holds on every seed. PLATEAU at E* if E* exists and R(E*) holds "
    "on every seed; PLATEAU_WITHOUT_RETRIEVAL at E* if E* exists and R(E*) fails on "
    "some seed; NO_PLATEAU by 9000 if no E* exists by the cap. In every case ckpt3000 "
    "stays primary; making another checkpoint the substrate is D2, owner-only."
)
EXPECTED = (
    "NO_PLATEAU by 9000 about 55 %; PLATEAU about 40 % (most likely late, E* >= "
    "8000); PLATEAU_WITHOUT_RETRIEVAL about 5 %."
)
READING = {
    "PLATEAU": "the stream answer loss is flat within +/-1 %/1000 steps on every seed "
    "from E*, and that checkpoint still retrieves through memory",
    "PLATEAU_WITHOUT_RETRIEVAL": "flat, but not a retrieval substrate",
    "NO_PLATEAU": "no common plateau by the cap; this is itself the result",
    "IN_PROGRESS": "the run has not reached the cap and no measured E* exists yet",
}
POLL_S = 30.0


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(s: str) -> None:
    print(f"{now_iso()} {s}", flush=True)


# --------------------------------------------------------------------------- #
# the plateau rule (PREREG "The plateau rule")
# --------------------------------------------------------------------------- #


def ols_rule(ys: list[float]) -> dict:
    """OLS of 16 window means on midpoints 0.25 apart (thousands of steps); the 95 %
    CI of the slope over the mean, in % per 1000 steps, strictly inside (-1, +1)."""
    if len(ys) != N_WINDOWS:
        raise ValueError(f"the rule reads exactly {N_WINDOWS} windows, got {len(ys)}")
    xs = [(WINDOW * i + WINDOW / 2) / 1000 for i in range(N_WINDOWS)]
    mx, my = sum(xs) / N_WINDOWS, sum(ys) / N_WINDOWS
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    rss = sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys, strict=True))
    se = math.sqrt(rss / (N_WINDOWS - 2) / sxx)
    lo, hi = b - T_975_14 * se, b + T_975_14 * se
    rel = (100 * lo / my, 100 * hi / my)
    return {
        "slope_per_1000": b,
        "se": se,
        "mean": my,
        "rel_slope_pct": 100 * b / my,
        "rel_ci_pct": [rel[0], rel[1]],
        "holds": bool(rel[0] > -REL_BOUND and rel[1] < REL_BOUND),
    }


def stream_windows(heartbeat: Path) -> dict[int, float]:
    """fresh-stream's ``stream_loss_windows`` at WINDOW, B5's own steps only."""
    w = FS.stream_loss_windows(Path(heartbeat), window=WINDOW)
    return {int(w0): float(v) for w0, v in w.items() if int(w0) >= RESUME_STEP}


def rule_at(windows: dict[int, float], e: int) -> dict | None:
    """The rule for one seed at evaluation step E over ``[E - 4000, E)``; None if any
    of the 16 windows is absent or precedes B5's own steps."""
    starts = list(range(e - WINDOW * N_WINDOWS, e, WINDOW))
    if starts[0] < RESUME_STEP or any(w not in windows for w in starts):
        return None
    return {**ols_rule([windows[w] for w in starts]), "windows": starts}


def rule_trace(by_seed: dict[int, dict[int, float]]) -> dict[int, dict[int, dict]]:
    """``{E: {seed: rule}}`` for every E evaluable on at least one seed."""
    out: dict[int, dict[int, dict]] = {}
    for e in EVAL_STEPS:
        row = {s: r for s in SEEDS if (r := rule_at(by_seed.get(s) or {}, e)) is not None}
        if row:
            out[e] = row
    return out


def common_step(trace: dict[int, dict[int, dict]]) -> int | None:
    """E*: the first E at which the rule holds on EVERY seed."""
    for e in sorted(trace):
        row = trace[e]
        if all(s in row and row[s]["holds"] for s in SEEDS):
            return e
    return None


def classify(*, e_star: int | None, r_at_e: bool | None, reached_cap: bool) -> str:
    """PREREG classification table (controls are checked by ``verdict``)."""
    if e_star is not None and r_at_e is not None:
        return "PLATEAU" if r_at_e else "PLATEAU_WITHOUT_RETRIEVAL"
    if e_star is None and reached_cap:
        return "NO_PLATEAU"
    return "IN_PROGRESS"


# --------------------------------------------------------------------------- #
# protection, the start copy, resume
# --------------------------------------------------------------------------- #


class ProtectedPath(RuntimeError):
    """An output path inside a T0-protected directory."""


class StartCheckpointRefused(RuntimeError):
    """A start checkpoint is absent, stores another step, or is not the T0 file."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def t0_entries(manifest: Path = T0_MANIFEST) -> dict[str, str]:
    """``{repo-relative path: sha256}`` from the T0 manifest."""
    out = {}
    for line in Path(manifest).read_text().splitlines():
        if line.strip():
            sha, rel = line.split(None, 1)
            out[rel.strip()] = sha
    return out


def protected_dirs(entries: dict[str, str], repo: Path = MAIN_REPO) -> set[Path]:
    """Every directory directly holding a T0 file, and the fresh-stream worktree."""
    prot = {(Path(repo) / rel).parent for rel in entries}
    prot.add(Path(repo) / ".worktrees" / "fresh-stream")
    return prot


def assert_writable(path: Path, protected: set[Path]) -> Path:
    """Refuse a write target equal to or inside a protected directory."""
    p = Path(os.path.abspath(path))
    for d in protected:
        d = Path(os.path.abspath(d))
        if p == d or d in p.parents:
            raise ProtectedPath(f"{p} is inside protected {d} (T0): refusing to write")
    return p


def source_ckpt(
    seed: int, step: int = RESUME_STEP, *, arm: str = ARM, source_root: Path = SOURCE_ROOT
) -> Path:
    return RC.ckpt_path(Path(source_root) / arm / f"seed{seed}", step)


def t0_sha(path: Path, entries: dict[str, str], repo: Path = MAIN_REPO) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(repo))
    if rel not in entries:
        raise StartCheckpointRefused(f"{path} has no line in the T0 manifest")
    return entries[rel]


def copy_start(
    src: Path, dest_dir: Path, *, expected_sha: str, step: int = RESUME_STEP
) -> Path:
    """Copy a start checkpoint into ``dest_dir`` (read-only on ``src``): its sha256
    must equal ``expected_sha`` and it must store ``step``; the copy's sha256 must
    equal the source's. An existing identical copy is kept."""
    src, dest_dir = Path(src), Path(dest_dir)
    if not src.exists():
        raise StartCheckpointRefused(f"{src} is absent")
    sha = sha256_file(src)
    if sha != expected_sha:
        raise StartCheckpointRefused(f"{src} sha256 {sha} != expected {expected_sha}")
    if RC.stored_step(src) != step:
        raise StartCheckpointRefused(f"{src} does not store step {step}")
    dest = dest_dir / src.name
    if dest.exists() and sha256_file(dest) == sha:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f".{src.name}.copy.{os.getpid()}"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)
    if sha256_file(dest) != sha:
        raise StartCheckpointRefused(f"copy {dest} does not match {src}")
    return dest


def latest_ckpt(seed_dir: Path) -> Path | None:
    """The checkpoint with the largest step in ``seed_dir`` (dotted temporaries of
    an interrupted atomic write are not checkpoints)."""
    d = Path(seed_dir)
    if not d.is_dir():
        return None
    found = [p for p in d.glob("ckpt-*.pt") if not p.name.startswith(".")]
    return max(found, key=lambda p: int(p.stem.split("-")[1])) if found else None


# --------------------------------------------------------------------------- #
# the child: fresh-stream's resume child, threads / dir / cadence as arguments
# --------------------------------------------------------------------------- #


def child_env(threads: int) -> dict[str, str]:
    from orchestrator.lanes import THREAD_ENV_VARS

    return {**os.environ, **dict.fromkeys(THREAD_ENV_VARS, str(threads))}


def child_argv(
    seed: int, out_dir: Path, *, threads: int, iters: int, ckpt_every: int
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
        "--threads",
        str(threads),
        "--iters",
        str(iters),
        "--ckpt-every",
        str(ckpt_every),
    ]


@contextlib.contextmanager
def _set(mod, **overrides):
    """Some module names set for one call, restored after; a typo raises."""
    old = {k: getattr(mod, k) for k in overrides}
    for k, v in overrides.items():
        setattr(mod, k, v)
    try:
        yield mod
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


def single(
    seed: int,
    out_dir: Path,
    *,
    threads: int,
    iters: int,
    ckpt_every: int,
    protected: set[Path] | None = None,
) -> dict:
    """One segment: resume ``out_dir``'s latest checkpoint and train to ``iters``
    through fresh-stream's ``single("B")`` (its ``ResumeCheck`` included)."""
    torch.set_num_threads(threads)
    out_dir = Path(out_dir)
    if protected is not None:
        assert_writable(out_dir, protected)
    start = latest_ckpt(out_dir)
    if start is None:
        raise StartCheckpointRefused(f"{out_dir} holds no checkpoint to resume")
    prior = out_dir / "resume_check.json"
    if prior.exists():  # keep every segment's control 5 record
        os.replace(prior, out_dir / f"resume_check.before-{start.stem}.json")
    with (
        _set(FS, start_ckpt=lambda s, root=None: start),
        _set(CSC, CKPT_EVERY=ckpt_every),
    ):
        r = FS.single("B", seed, out_dir, iters=iters)
    r.update(
        arm=ARM,
        resumed_from=str(start),
        threads=threads,
        torch_num_threads=torch.get_num_threads(),
        ckpt_every=ckpt_every,
    )
    return r


class Child:
    """One training child; ``poll``/``terminate``/``kill``/``wait`` as Popen's."""

    def __init__(
        self, seed: int, out_dir: Path, *, threads: int, iters: int, ckpt_every: int
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.argv = child_argv(
            seed, out_dir, threads=threads, iters=iters, ckpt_every=ckpt_every
        )
        self._log = open(out_dir / "train.log", "a")  # noqa: SIM115
        self._log.write(f"\n=== segment {now_iso()} argv {self.argv}\n")
        self._log.flush()
        self._p = subprocess.Popen(
            self.argv,
            cwd=ROOT,
            stdout=self._log,
            stderr=self._log,
            env=child_env(threads),
        )
        self.pid = self._p.pid

    def poll(self):
        return self._p.poll()

    def terminate(self):
        self._p.terminate()

    def kill(self):
        self._p.kill()

    def wait(self, timeout=None):
        rc = self._p.wait(timeout=timeout)
        self._log.close()
        return rc


# --------------------------------------------------------------------------- #
# control 6: the resume proof
# --------------------------------------------------------------------------- #


def compare_payloads(a: dict, b: dict) -> dict:
    """Exact equality of two checkpoint payloads: step, model, optimizer, policy and
    the CPU / python RNG state (fresh-stream's ``_same`` for the tensors)."""
    acc: dict[str, Any] = {"n_tensors": 0, "mismatches": []}
    if a.get("step") != b.get("step"):
        acc["mismatches"].append(f"step {a.get('step')} != {b.get('step')}")
    for k in ("model", "optimizer", "policy"):
        FS._same(a.get(k), b.get(k), k, acc)
    ra, rb = a.get("rng") or {}, b.get("rng") or {}
    FS._same(ra.get("cpu"), rb.get("cpu"), "rng.cpu", acc)
    if ra.get("python") != rb.get("python"):
        acc["mismatches"].append("rng.python")
    return {
        "equal": not acc["mismatches"],
        "n_tensors": acc["n_tensors"],
        "mismatches": acc["mismatches"][:20],
    }


def _load_payload(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


def prove_resume(workdir: Path, *, seed: int = 0, protected: set[Path]) -> dict:
    """PREREG control 6: (a) 2900 -> 3000 at 5 threads == arm B's ckpt-003000;
    (b) the same at 4 threads (reported); (c) 4 threads, own 2950 -> 3000 ==
    uninterrupted 4-thread 3000. (a) and (b) run in parallel, then (c)."""
    workdir = assert_writable(Path(workdir), protected)
    entries = t0_entries()
    src2900, src3000 = source_ckpt(seed, 2900), source_ckpt(seed, 3000)
    dirs = {k: workdir / k for k in ("a", "b", "c")}
    for k in ("a", "b"):
        copy_start(src2900, dirs[k], expected_sha=t0_sha(src2900, entries), step=2900)
    t0 = time.time()
    ch = {
        "a": Child(seed, dirs["a"], threads=5, iters=3000, ckpt_every=50),
        "b": Child(seed, dirs["b"], threads=4, iters=3000, ckpt_every=50),
    }
    rcs = {k: c.wait() for k, c in ch.items()}
    shutil.copyfile(dirs["b"] / "ckpt-002950.pt", _mk(dirs["c"]) / "ckpt-002950.pt")
    rcs["c"] = Child(seed, dirs["c"], threads=4, iters=3000, ckpt_every=50).wait()
    ref = _load_payload(src3000)
    out = {
        "seed": seed,
        "exit_codes": rcs,
        "elapsed_s": time.time() - t0,
        "reference": str(src3000),
        "reference_sha256": sha256_file(src3000),
    }
    for k in ("a", "b"):
        f = dirs[k] / "ckpt-003000.pt"
        out[k] = (
            compare_payloads(_load_payload(f), ref) if f.exists() else {"equal": None}
        )
    fb, fc = dirs["b"] / "ckpt-003000.pt", dirs["c"] / "ckpt-003000.pt"
    out["c"] = (
        compare_payloads(_load_payload(fc), _load_payload(fb))
        if fb.exists() and fc.exists()
        else {"equal": None}
    )
    out["launch_allowed"] = out["c"].get("equal") is True
    (workdir / "proof.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _mk(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# timing: 3 seeds x 4 threads for N steps, measured, before any estimate
# --------------------------------------------------------------------------- #


def beat_elapsed(heartbeat: Path) -> dict[int, float]:
    out = {}
    if Path(heartbeat).exists():
        for line in Path(heartbeat).read_text().splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                r = json.loads(line)
                if r.get("kind") == "beat":
                    out[int(r["step"])] = float(r["elapsed_s"])
    return out


def s_per_step(heartbeat: Path, skip: int = 5) -> dict:
    """Seconds per step from the beats' ``elapsed_s`` of the LAST segment, skipping
    its first ``skip`` steps (warm-up)."""
    el = beat_elapsed(heartbeat)
    steps = sorted(el)
    if len(steps) <= skip + 1:
        return {"n": 0, "s_per_step": None}
    a, b = steps[skip], steps[-1]
    return {"n": b - a, "s_per_step": (el[b] - el[a]) / (b - a), "from": a, "to": b}


def time_segment(workdir: Path, steps: int, *, protected: set[Path]) -> dict:
    workdir = assert_writable(Path(workdir), protected)
    entries = t0_entries()
    t0 = time.time()
    children = {}
    for s in SEEDS:
        src = source_ckpt(s)
        copy_start(src, workdir / f"seed{s}", expected_sha=t0_sha(src, entries))
    for s in SEEDS:
        children[s] = Child(
            s,
            workdir / f"seed{s}",
            threads=THREADS_PER_SEED,
            iters=RESUME_STEP + steps,
            ckpt_every=steps,
        )
    rcs = {s: c.wait() for s, c in children.items()}
    wall = time.time() - t0
    per = {s: s_per_step(workdir / f"seed{s}" / "heartbeat.jsonl") for s in SEEDS}
    vals = [p["s_per_step"] for p in per.values() if p["s_per_step"] is not None]
    out = {
        "steps": steps,
        "threads_per_seed": THREADS_PER_SEED,
        "exit_codes": rcs,
        "wall_s": wall,
        "per_seed": per,
        "s_per_step_max": max(vals) if vals else None,
    }
    if vals:
        out["projected_h_3000_to_9000"] = max(vals) * (CAP - RESUME_STEP) / 3600
    (workdir / "timing.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


# --------------------------------------------------------------------------- #
# controls 3 and 4 (measurement path; stream), the manifest
# --------------------------------------------------------------------------- #


def reference_rows(doc: dict) -> dict[str, list[float]]:
    if doc.get("seeds_actually_run") != SEEDS:
        raise ValueError(f"reference seeds {doc.get('seeds_actually_run')} != {SEEDS}")
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in doc["rows"]
        if r.get("kind") == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def remeasured_rows(per: dict[int, dict[int, dict]]) -> dict[str, list[float]]:
    col = ST._Rows()
    with _set(FS, CHECKPOINTS={ARM: (RESUME_STEP,)}):
        FS.arm_rows(col, ARM, per)
    return {
        r["key"]: [float(x) for x in r["samples"]]
        for r in col.rows
        if r["kind"] == "statistic" and r["key"].startswith(CONTROL_PREFIX)
    }


def control_measurement_path(root: Path, reference: dict) -> dict:
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    try:
        for s in SEEDS:
            per[s][RESUME_STEP] = measure_checkpoint(
                root / ARM / f"seed{s}", s, RESUME_STEP, N_TRAIN
            )
            log(f"  control 3: measured seed {s} ckpt{RESUME_STEP} (copy)")
        out = ST.reproduction_control(reference, remeasured_rows(per), REPRO_TOL)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "per_seed": {}}
    out["error"] = None
    return out


def own_windows(iters: int = CAP) -> list[range]:
    return [STREAM.window(t, BATCH) for t in range(RESUME_STEP, iters)]


def preflight() -> dict:
    out: dict[str, Any] = {"ok": False, "error": None, "closure": {}}
    try:
        ids = [i for w in own_windows() for i in w]
        u = set(ids)
        d = {
            "n_ids": len(ids),
            "first_id": min(ids),
            "last_id": max(ids),
            "repeats": len(ids) - len(u),
            "ids_in_probe": sorted(u & set(range(*PROBE)))[:10],
            "ids_in_vocab_documents": sorted(u & set(range(*VOCAB_DOCUMENTS)))[:10],
            "ids_in_heldout": sorted(u & set(range(*HELDOUT)))[:10],
        }
        d["ok"] = not (
            d["repeats"]
            or d["ids_in_probe"]
            or d["ids_in_vocab_documents"]
            or d["ids_in_heldout"]
        )
        out["disjointness"] = d
        if not d["ok"]:
            out["error"] = "the stream is not disjoint or repeats a document"
            return out
        with _set(FS, stream_windows=own_windows):
            for s in SEEDS:
                out["closure"][s] = FS.vocabulary_closure(s, CAP)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["ok"] = True
    return out


def manifest_config(start_sha: dict, ref_sha: str, parent_threads: int) -> dict:
    return {
        **S003.CONFIG,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "decision_rule": DECISION_RULE,
        "expected": EXPECTED,
        "reading": READING,
        "instrument": f"{CSC.EXPERIMENT}::measure_checkpoint (imported, identity), "
        f"n={N_TRAIN}",
        "child": f"{FS.EXPERIMENT}::single('B'), resume = latest ckpt in own dir",
        "seeds": SEEDS,
        "arm": ARM,
        "resume_step": RESUME_STEP,
        "cap": CAP,
        "window": WINDOW,
        "n_windows": N_WINDOWS,
        "rel_bound_pct_per_1000": REL_BOUND,
        "t_975_14": T_975_14,
        "eval_steps": list(EVAL_STEPS),
        "trajectory_checkpoints": list(TRAJECTORY),
        "ckpt_every": CKPT_EVERY,
        "threads_per_seed": THREADS_PER_SEED,
        "seeds_trained_in_parallel": True,
        "parent_torch_num_threads": parent_threads,
        "stream": dataclasses.asdict(STREAM),
        "stream_documents": [
            STREAM.window(RESUME_STEP, BATCH).start,
            STREAM.window(CAP - 1, BATCH).stop,
        ],
        "heldout_documents": list(HELDOUT),
        "probe_documents": list(PROBE),
        "delta": DELTA,
        "c_threshold": C_THRESHOLD,
        "bucket": BUCKET,
        "reproduction_tolerance": REPRO_TOL,
        "device": DEVICE,
        "source_root": str(SOURCE_ROOT),
        "start_checkpoint_sha256": start_sha,
        "t0_manifest": str(T0_MANIFEST),
        "reference_ledger": REF_LEDGER,
        "reference_ledger_sha256": ref_sha,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
    }


# --------------------------------------------------------------------------- #
# state on disk -> readouts
# --------------------------------------------------------------------------- #


def _read_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def windows_by_seed(root: Path) -> dict[int, dict[int, float]]:
    return {s: stream_windows(root / ARM / f"seed{s}" / "heartbeat.jsonl") for s in SEEDS}


def measurement_path(root: Path, seed: int, c: int) -> Path:
    return root / "measurements" / f"seed{seed}-ckpt{c:06d}.json"


def load_measurements(root: Path) -> dict[int, dict[int, dict]]:
    per: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}
    d = root / "measurements"
    if d.is_dir():
        for f in sorted(d.glob("seed*-ckpt*.json")):
            s = int(f.name[4 : f.name.index("-")])
            c = int(f.stem.split("ckpt")[1])
            per[s][c] = json.loads(f.read_text())
    return per


def steps_reached(root: Path) -> dict[int, int]:
    out = {}
    for s in SEEDS:
        el = beat_elapsed(root / ARM / f"seed{s}" / "heartbeat.jsonl")
        out[s] = (max(el) + 1) if el else RESUME_STEP
    return out


def state(root: Path) -> dict:
    """Everything the verdict reads, from files only."""
    wins = windows_by_seed(root)
    trace = rule_trace(wins)
    e_star = common_step(trace)
    per = load_measurements(root)
    readout = FS.arm_readout(per, sorted({*TRAJECTORY, *([e_star] if e_star else [])}))
    reached = steps_reached(root)
    results = {s: _read_json(root / ARM / f"seed{s}" / "result.json") for s in SEEDS}
    done = {s: (results[s] or {}).get("steps_done") for s in SEEDS}
    reached_cap = all(d == CAP for d in done.values())
    r_at_e = readout[e_star]["R"] if e_star is not None and e_star in readout else None
    return {
        "windows": wins,
        "trace": trace,
        "e_star": e_star,
        "per": per,
        "readout": readout,
        "reached": reached,
        "results": results,
        "steps_done": done,
        "reached_cap": reached_cap,
        "r_at_e": r_at_e,
        "classification": classify(e_star=e_star, r_at_e=r_at_e, reached_cap=reached_cap),
    }


def pending_measurements(st: dict) -> list[int]:
    want = sorted({*TRAJECTORY, *([st["e_star"]] if st["e_star"] else [])})
    return [c for c in want if not all(c in st["per"][s] for s in SEEDS)]


def status_line(st: dict) -> str:
    last = {s: (max(w) if w else None) for s, w in st["windows"].items()}
    parts = []
    for s in SEEDS:
        w0 = last[s]
        loss = f"{st['windows'][s][w0]:.4f}@{w0}" if w0 is not None else "-"
        parts.append(f"s{s} step {st['reached'][s]} loss {loss}")
    e = max(st["trace"]) if st["trace"] else None
    rule = (
        "rule not yet evaluable (first E = 7000)"
        if e is None
        else f"rule@{e}: "
        + ", ".join(
            f"s{s} CI [{r['rel_ci_pct'][0]:+.2f},{r['rel_ci_pct'][1]:+.2f}]%"
            f" {'HOLDS' if r['holds'] else 'no'}"
            for s, r in sorted(st["trace"][e].items())
        )
    )
    return f"{' | '.join(parts)} | {rule} | E*={st['e_star']} | {st['classification']}"


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


def verdict(root: Path, st: dict) -> dict:
    why = []
    c3 = _read_json(root / "control_measurement_path.json")
    if not c3 or not c3.get("ok"):
        why.append("control 3 (measurement path unchanged) failed or did not run")
    pf = _read_json(root / "preflight.json")
    if not pf or not pf.get("ok"):
        why.append("control 4 (stream disjointness / closure) failed or did not run")
    rc = {s: _read_json(root / ARM / f"seed{s}" / "resume_check.json") for s in SEEDS}
    bad = [f"seed{s}" for s in SEEDS if not (rc[s] or {}).get("ok")]
    if bad:
        why.append(f"control 5 (resume is exact) failed or missing on {bad}")
    errs = _read_json(root / "measurement_errors.json") or []
    if errs:
        why.append(f"a measurement raised: {errs[0]}")
    cls = st["classification"]
    if why:
        return {
            "classification": "inconclusive",
            "reported_as": "inconclusive",
            "exit": Exit.DID_NOT_RUN,
            "detail": "; ".join(why),
        }
    e = st["e_star"]
    reported = {
        "PLATEAU": f"PLATEAU at {e}",
        "PLATEAU_WITHOUT_RETRIEVAL": f"PLATEAU_WITHOUT_RETRIEVAL at {e}",
        "NO_PLATEAU": f"NO_PLATEAU by {CAP}",
        "IN_PROGRESS": f"IN_PROGRESS (steps reached {st['reached']})",
    }[cls]
    return {
        "classification": cls,
        "reported_as": reported,
        "exit": Exit.OK,
        "detail": f"E* = {e}; R(E*) = {st['r_at_e']}; reached cap: "
        f"{st['reached_cap']}: {reported} -- {READING[cls]}",
    }


def build_ledger(root: Path, led=None):
    import ledger as ledger_mod

    if led is None:
        led = ledger_mod.Ledger(RUN_ID, question=QUESTION, runs_root=root.parent)
        man = root / "manifest.json"
        if man.exists():
            blob = man.read_bytes()
            led.doc["config_hash"] = hashlib.sha256(blob).hexdigest()
            led.doc["manifest_written_utc"] = (
                _read_json(root / "launch.json") or {}
            ).get("manifest_written_utc")
    st = state(root)
    v = verdict(root, st)
    measured_all = [c for c in sorted(st["readout"])]
    with _set(FS, CHECKPOINTS={ARM: tuple(measured_all)}):
        FS.arm_rows(
            led, ARM, {s: {c: st["per"][s][c] for c in measured_all} for s in SEEDS}
        )
    led.run_meta(
        device=DEVICE,
        seeds_actually_run=SEEDS,
        steps_requested=CAP,
        steps_done=min(st["reached"].values()),
    )
    for s in SEEDS:
        led.note(
            f"seed{s}.steps_reached", st["reached"][s], how="max heartbeat beat step + 1"
        )
        led.note(
            f"seed{s}.steps_done",
            st["steps_done"][s],
            how="child result.json steps_done (None while running)",
        )
        led.note(
            f"seed{s}.s_per_step",
            s_per_step(root / ARM / f"seed{s}" / "heartbeat.jsonl"),
            how="beats' elapsed_s over the last segment, first 5 steps skipped",
        )
    led.note(
        "stream_answer_loss_window_means",
        {
            f"seed{s}": {str(w): x for w, x in sorted(st["windows"][s].items())}
            for s in SEEDS
        },
        how=f"heartbeat loss_answer_tokens, mean over each complete {WINDOW}-step "
        f"window from step {RESUME_STEP}, keyed by its first step",
    )
    led.note(
        "rule_trace",
        {
            str(e): {
                f"seed{s}": {
                    k: r[k]
                    for k in (
                        "slope_per_1000",
                        "se",
                        "mean",
                        "rel_slope_pct",
                        "rel_ci_pct",
                        "holds",
                    )
                }
                for s, r in sorted(row.items())
            }
            for e, row in sorted(st["trace"].items())
        },
        how="ols_rule over the 16 windows [E-4000, E), per seed",
    )
    led.note("E_star", st["e_star"], how="first E holding on every seed; None if none")
    led.note("R_at_E_star", st["r_at_e"], how="R(E*) on every seed; None if unmeasured")
    led.note(
        "R_by_checkpoint",
        {str(c): r["R"] for c, r in st["readout"].items()},
        how="fresh-stream arm_readout: live - slots_zeroed >= DELTA every seed",
    )
    led.note(
        "C_by_checkpoint",
        {str(c): r["C"] for c, r in st["readout"].items()},
        how="fresh-stream arm_readout (reported only)",
    )
    led.note("reached_cap", st["reached_cap"], how="every seed steps_done == CAP")
    c3 = _read_json(root / "control_measurement_path.json") or {}
    for f in ("ok", "n_keys_compared", "max_abs_diff"):
        ps = c3.get("per_seed") or {}
        led.note(
            f"control_measurement_path.{f}",
            [(ps.get(str(s)) or {}).get(f) for s in SEEDS],
            how=f"{CONTROL_PREFIX}* of {REF_LEDGER} vs the re-measured copies",
        )
    led.note("control_measurement_path.passed", c3.get("ok"), how="all seeds ok")
    led.note("control_measurement_path.error", c3.get("error"), how="None if none")
    pf = _read_json(root / "preflight.json") or {}
    led.note("preflight.ok", pf.get("ok"), how="stream disjointness + closure")
    led.note("stream_disjointness", pf.get("disjointness"), how="preflight()")
    rcs = {}
    for s in SEEDS:
        d = root / ARM / f"seed{s}"
        rcs[f"seed{s}"] = {
            f.name: (_read_json(f) or {}).get("ok")
            for f in sorted(d.glob("resume_check*.json"))
        }
    led.note(
        "control_resume_exact",
        rcs,
        how="every segment's ResumeCheck (model + optimizer == loaded ckpt)",
    )
    for k in ("launch.json", "proof/proof.json", "timing/timing.json", "segments.json"):
        led.note(Path(k).stem, _read_json(root / k), how=f"runs/{RUN_ID}/{k}")
    led.note(
        "measurement_errors",
        _read_json(root / "measurement_errors.json") or [],
        how="measurements that raised",
    )
    led.note("classification", v["classification"], how="PREREG table")
    led.note("classification_reported_as", v["reported_as"], how="PREREG decision rule")
    led.note("classification_detail", v["detail"], how="verdict()")
    led.note("reading", READING.get(v["classification"]), how="PREREG table")
    led.note("decision_rule", DECISION_RULE, how="PREREG front matter")
    led.note("expected", EXPECTED, how="PREREG author's expectation")
    led.note("delta", DELTA, how="fresh-stream DELTA, imported")
    led.note(
        "primary_substrate",
        "ckpt3000 (R-2026-09-27-retrieval-shown); D2 is owner-only",
        how="PREREG status",
    )
    crashed = any(st["results"][s] is None and _child_failed(root, s) for s in SEEDS)
    if st["reached_cap"] and not pending_measurements(st):
        led.status("ok")
    else:
        led.status("crashed" if crashed else "partial")
    return led, st, v


def _child_failed(root: Path, s: int) -> bool:
    segs = _read_json(root / "segments.json") or []
    last = segs[-1] if segs else {}
    rc = (last.get("exit_codes") or {}).get(str(s))
    return rc not in (None, 0)


def write_ledger(root: Path) -> tuple[Path | None, dict, dict]:
    import ledger as ledger_mod

    led, st, v = build_ledger(root)
    try:
        p = led.write()
    except ledger_mod.DirtyTree as e:
        log(f"  ledger NOT written: {e}")
        p = None
    return p, st, v


# --------------------------------------------------------------------------- #
# the parent
# --------------------------------------------------------------------------- #


def _append(path: Path | None, text: str) -> None:
    if path is not None:
        with open(path, "a") as fh:
            fh.write(text + "\n")


def parent(root: Path, *, report: Path | None, protected: set[Path]) -> Exit:
    import ledger as ledger_mod

    assert_writable(root, protected)
    dirty = ledger_mod._dirty_source_paths()
    if dirty:
        refuse(Exit.DID_NOT_RUN, f"uncommitted source {dirty}: the ledger would refuse")
    entries = t0_entries()
    root.mkdir(parents=True, exist_ok=True)
    first = not (root / "manifest.json").exists()
    ref_path = ROOT / REF_LEDGER
    ref_threads = int(
        json.loads((ROOT / REF_MANIFEST).read_text())["parent_torch_num_threads"]
    )
    torch.set_num_threads(ref_threads)
    if first:
        start_sha = {}
        for s in SEEDS:
            src = source_ckpt(s)
            try:
                dest = copy_start(
                    src, root / ARM / f"seed{s}", expected_sha=t0_sha(src, entries)
                )
            except StartCheckpointRefused as e:
                refuse(Exit.DID_NOT_RUN, f"start checkpoint refused: {e}")
            start_sha[f"seed{s}"] = sha256_file(dest)
        led = ledger_mod.Ledger(RUN_ID, question=QUESTION, runs_root=root.parent)
        m = led.manifest(manifest_config(start_sha, sha256_file(ref_path), ref_threads))
        (root / "launch.json").write_text(
            json.dumps(
                {
                    "manifest_written_utc": led.doc["manifest_written_utc"],
                    "git_sha": led.doc["provenance"].get("git_sha"),
                    "start_checkpoint_sha256": start_sha,
                },
                indent=2,
            )
            + "\n"
        )
        log(f"manifest frozen: {m} config_hash {led.doc['config_hash']}")
        c3 = control_measurement_path(
            root, reference_rows(json.loads(ref_path.read_text()))
        )
        (root / "control_measurement_path.json").write_text(
            json.dumps(c3, indent=2, default=str) + "\n"
        )
        log(f"control 3 measurement path unchanged: ok={c3.get('ok')}")
        if not c3.get("ok"):
            write_ledger(root)
            refuse(Exit.DID_NOT_RUN, f"control 3 failed: {c3.get('error')}")
        pf = preflight()
        (root / "preflight.json").write_text(json.dumps(pf, indent=2, default=str) + "\n")
        log(f"control 4 stream/closure: ok={pf['ok']}")
        if not pf["ok"]:
            write_ledger(root)
            refuse(Exit.DID_NOT_RUN, f"control 4 failed: {pf['error']}")
    segs = _read_json(root / "segments.json") or []
    seg = {
        "started": now_iso(),
        "git_sha": ledger_mod.git_provenance().get("git_sha"),
        "parent_pid": os.getpid(),
        "from_step": {},
        "child_pid": {},
        "exit_codes": {},
    }
    children = {}
    for s in SEEDS:
        d = root / ARM / f"seed{s}"
        lc = latest_ckpt(d)
        seg["from_step"][str(s)] = int(lc.stem.split("-")[1]) if lc else None
        if lc is not None and int(lc.stem.split("-")[1]) >= CAP:
            continue
        children[s] = Child(
            s, d, threads=THREADS_PER_SEED, iters=CAP, ckpt_every=CKPT_EVERY
        )
        seg["child_pid"][str(s)] = children[s].pid
    segs.append(seg)
    (root / "segments.json").write_text(json.dumps(segs, indent=2) + "\n")
    log(f"segment {len(segs)}: children {seg['child_pid']} from {seg['from_step']}")
    _append(
        report,
        f"- {now_iso()} segment {len(segs)} started: child pids "
        f"{seg['child_pid']}, from steps {seg['from_step']}",
    )

    errors: list[str] = _read_json(root / "measurement_errors.json") or []
    last_line = None
    while True:
        st = state(root)
        for c in pending_measurements(st):
            for s in SEEDS:
                if c in st["per"][s]:
                    continue
                ck = RC.ckpt_path(root / ARM / f"seed{s}", c)
                if not ck.exists():
                    continue
                try:
                    m = measure_checkpoint(root / ARM / f"seed{s}", s, c, N_TRAIN)
                    m["measured_at"] = now_iso()
                    p = measurement_path(root, s, c)
                    _mk(p.parent)
                    p.write_text(json.dumps(m, indent=2, default=str) + "\n")
                    log(f"  measured seed {s} ckpt{c}")
                except Exception as e:  # recorded; the verdict reads it
                    errors.append(f"seed {s} ckpt{c}: {type(e).__name__}: {e}")
                    (root / "measurement_errors.json").write_text(
                        json.dumps(errors, indent=2) + "\n"
                    )
                    log(f"  measurement raised: {errors[-1]}")
        st = state(root)
        line = status_line(st)
        if line != last_line:
            write_ledger(root)
            _append(report, f"- {now_iso()} {line}")
            log(line)
            last_line = line
        alive = [s for s, c in children.items() if c.poll() is None]
        if not alive and not [
            c
            for c in pending_measurements(st)
            if all(RC.ckpt_path(root / ARM / f"seed{s}", c).exists() for s in SEEDS)
        ]:
            break
        time.sleep(POLL_S)
    seg["exit_codes"] = {str(s): c.wait() for s, c in children.items()}
    seg["ended"] = now_iso()
    segs[-1] = seg
    (root / "segments.json").write_text(json.dumps(segs, indent=2) + "\n")
    path, st, v = write_ledger(root)
    _append(
        report,
        f"- {now_iso()} segment {len(segs)} ended: exit codes "
        f"{seg['exit_codes']}; {v['reported_as']}; ledger {path}",
    )
    log(f"ledger: {path}  classification={v['reported_as']}  exit={int(v['exit'])}")
    return v["exit"]


# --------------------------------------------------------------------------- #
# RESULTS.md, rendered from the ledger
# --------------------------------------------------------------------------- #


def render_results(doc: dict) -> str:
    rows = RC._rows(doc)
    val, fmt = RC._val, RC._fmt
    prov = doc.get("provenance", {})
    rid = doc.get("run_id")
    out = [
        "# B5 substrate convergence -- RESULTS",
        "",
        f"<!-- GENERATED by `{EXPERIMENT}` from runs/{rid}/ledger.json. Do not edit: "
        "regenerate with `--render-results`. -->",
        "",
        f"Ledger `status` of run `{rid}`: **{doc.get('status')}**"
        + (
            " -- **a status RESULTS, not a result: the run has not reached the cap.**"
            if doc.get("status") != "ok"
            else ""
        )
        + "",
        "",
        f"Provenance: `provenance.git_sha` `{prov.get('git_sha')}` · `device` "
        f"{doc.get('device')} · `provenance.platform` `{prov.get('platform')}` · "
        f"`seeds_actually_run` {doc.get('seeds_actually_run')} · `steps_done` (min "
        f"over seeds) {doc.get('steps_done')} of `steps_requested` "
        f"{doc.get('steps_requested')} · `config_hash` `{doc.get('config_hash')}`.",
        "",
        "**ckpt3000 stays primary for everything; making another checkpoint the "
        "substrate is D2, owner-only** (`primary_substrate`).",
        "",
        f"## Classification: `classification_reported_as` "
        f"**{val(rows, 'classification_reported_as')}**",
        "",
        f"`classification_detail`: {val(rows, 'classification_detail')}",
        "",
        f"`E_star` {fmt(val(rows, 'E_star'))} · `R_at_E_star` "
        f"{fmt(val(rows, 'R_at_E_star'))} · `reached_cap` "
        f"{fmt(val(rows, 'reached_cap'))}.",
        "",
        "## Progress",
        "",
        "| seed | `steps_reached` | `steps_done` | `s_per_step` |",
        "|---|---|---|---|",
    ]
    for s in SEEDS:
        sp = val(rows, f"seed{s}.s_per_step") or {}
        out.append(
            f"| {s} | {fmt(val(rows, f'seed{s}.steps_reached'))} | "
            f"{fmt(val(rows, f'seed{s}.steps_done'))} | "
            f"{fmt(sp.get('s_per_step'))} |"
        )
    wins = val(rows, "stream_answer_loss_window_means") or {}
    starts = sorted({int(w) for d in wins.values() for w in d})
    out += [
        "",
        "## Stream answer loss, 250-step window means "
        "(`stream_answer_loss_window_means`)",
        "",
        "| window start | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|",
    ]
    for w in starts:
        out.append(
            f"| {w} | "
            + " | ".join(fmt((wins.get(f"seed{s}") or {}).get(str(w))) for s in SEEDS)
            + " |"
        )
    trace = val(rows, "rule_trace") or {}
    out += [
        "",
        "## Plateau rule (`rule_trace`): relative slope and 95 % CI, % per 1000 steps",
        "",
        "| E | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|",
    ]
    if not trace:
        out.append("| — | not yet evaluable: the first E is 7000 | | |")
    for e, row in sorted(trace.items(), key=lambda kv: int(kv[0])):
        cells = []
        for s in SEEDS:
            r = row.get(f"seed{s}")
            cells.append(
                "—"
                if r is None
                else f"{r['rel_slope_pct']:+.3f} [{r['rel_ci_pct'][0]:+.3f}, "
                f"{r['rel_ci_pct'][1]:+.3f}] {'**holds**' if r['holds'] else 'no'}"
            )
        out.append(f"| {e} | " + " | ".join(cells) + " |")
    out += ["", f"## Checkpoints measured (bucket `{BUCKET}`, per-seed samples)", ""]
    anyc = False
    for c in sorted(
        {*TRAJECTORY, *([val(rows, "E_star")] if val(rows, "E_star") else [])}
    ):
        e = f"{ARM}.ckpt{c}"
        if val(rows, f"{e}.R_quantity") is None:
            continue
        anyc = True
        out += [
            f"- `{e}.heldout.live.{BUCKET}.answer_acc` "
            f"{fmt(val(rows, f'{e}.heldout.live.{BUCKET}.answer_acc'))} · "
            f"`{e}.heldout.slots_zeroed.{BUCKET}.answer_acc` "
            f"{fmt(val(rows, f'{e}.heldout.slots_zeroed.{BUCKET}.answer_acc'))} · "
            f"`{e}.R_quantity` {fmt(val(rows, f'{e}.R_quantity'))} · `{e}.R_holds` "
            f"{fmt(val(rows, f'{e}.R_holds'))} · `{e}.C_quantity` "
            f"{fmt(val(rows, f'{e}.C_quantity'))}.",
            "",
        ]
    if not anyc:
        out += ["No checkpoint has been measured on every seed yet.", ""]
    out += [
        "## Controls",
        "",
        f"Measurement path unchanged (`{CONTROL_PREFIX}*`): "
        f"`control_measurement_path.passed` "
        f"{fmt(val(rows, 'control_measurement_path.passed'))} · `max_abs_diff` "
        f"{val(rows, 'control_measurement_path.max_abs_diff')!r} · `n_keys_compared` "
        f"{val(rows, 'control_measurement_path.n_keys_compared')!r}.",
        "",
        f"Stream: `preflight.ok` {fmt(val(rows, 'preflight.ok'))}. Resume exact per "
        f"segment (`control_resume_exact`): {val(rows, 'control_resume_exact')!r}.",
        "",
        f"Resume proof (`proof`, control 6): {_proof_line(val(rows, 'proof'))}.",
        "",
        f"Timing (`timing`): {_timing_line(val(rows, 'timing'))}.",
        "",
        f"Measurement errors (`measurement_errors`): "
        f"{val(rows, 'measurement_errors')!r}.",
        "",
    ]
    return "\n".join(out)


def _proof_line(p) -> str:
    if not p:
        return "not recorded"
    return "; ".join(f"({k}) equal={(p.get(k) or {}).get('equal')}" for k in "abc")


def _timing_line(t) -> str:
    if not t:
        return "not recorded"
    return (
        f"{t.get('steps')} steps x 3 seeds x {t.get('threads_per_seed')} threads, "
        f"max s/step {t.get('s_per_step_max')}, projected 3000->9000 "
        f"{t.get('projected_h_3000_to_9000')} h"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(prog="b5-convergence")
    for flag in (
        "--run",
        "--single",
        "--prove-resume",
        "--time-segment",
        "--status",
        "--write-ledger",
        "--render-results",
    ):
        ap.add_argument(flag, action="store_true")
    ap.add_argument("--seed", type=int, choices=SEEDS, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=THREADS_PER_SEED)
    ap.add_argument("--iters", type=int, default=CAP)
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)
    protected = protected_dirs(t0_entries())
    root = ROOT / "runs" / RUN_ID
    if a.single:
        if a.out_dir is None or a.seed is None:
            refuse(Exit.DID_NOT_RUN, "--single needs --out-dir and --seed")
        r = single(
            a.seed,
            a.out_dir,
            threads=a.threads,
            iters=a.iters,
            ckpt_every=a.ckpt_every,
            protected=protected,
        )
        (a.out_dir / "result.json").write_text(
            json.dumps(r, indent=2, default=str) + "\n"
        )
        return Exit.OK
    if a.prove_resume or a.time_segment:
        if a.workdir is None:
            refuse(Exit.DID_NOT_RUN, "--workdir is required")
        if a.prove_resume:
            out = prove_resume(a.workdir, seed=a.seed or 0, protected=protected)
            print(json.dumps(out, indent=2, default=str))
            return Exit.OK if out["launch_allowed"] else Exit.DID_NOT_RUN
        out = time_segment(a.workdir, a.steps, protected=protected)
        print(json.dumps(out, indent=2, default=str))
        return (
            Exit.OK
            if all(v == 0 for v in out["exit_codes"].values())
            else Exit.DID_NOT_RUN
        )
    if a.status:
        print(status_line(state(root)))
        return Exit.OK
    if a.write_ledger:
        p, _, v = write_ledger(root)
        print(f"ledger: {p} {v['reported_as']}")
        return Exit.OK if p else Exit.DID_NOT_RUN
    if a.render_results:
        p = root / "ledger.json"
        if not p.exists():
            refuse(Exit.DID_NOT_RUN, f"{p} is absent")
        out = ROOT / "experiments" / "b5-convergence" / "RESULTS.md"
        out.write_text(render_results(json.loads(p.read_text())))
        print(f"wrote {out}")
        return Exit.OK
    if a.run:
        return status(parent(root, report=a.report, protected=protected))
    refuse(Exit.DID_NOT_RUN, "no mode given")


if __name__ == "__main__":
    run_main(main)
