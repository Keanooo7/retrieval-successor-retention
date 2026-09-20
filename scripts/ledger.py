"""The per-run measurement ledger: `runs/<id>/ledger.json`.

**Why this exists.** The overnight protocol's rule is that *every number in a report
must trace to a ledger row*. A number typed into prose is a claim; a number a
process wrote next to the command, the git sha and the wall clock that produced it
is a measurement. This module is the only supported way to create the second kind,
and it deliberately makes the first kind inconvenient.

Distinct from `measurements/ledger.json` (§4.5, `rsr.constants.Registry`), which
holds *constants* a named experiment has measured and frozen. This one holds the
raw observations of a single run. A constant graduates from here to there; nothing
travels the other way.

## What cycle 0 of the 2026-09-19 run added, and why

Of the 13 ledgers the 2026-09-18 loop wrote, **9 cannot be re-executed from any
commit.** Seven `commands[].argv` entries point at a session-scoped scratch
directory belonging to a different session; one is the literal placeholder
`<scratch>/train_arm.py`; one is a bare `run_trained_alpha.py` that is not in the
tree. There was no path from any commit to most of that night's numbers.

📌 **`runs/cycle-srep-hinge/ledger.json` has `git_dirty: false` and is still
unreproducible**, because its modified trainer was never committed. So the gate is
not *"the tree is clean"* -- it is *"the code that ran is committed."* A clean-tree
check would have passed that ledger, which is why `command()` resolves the entry
point against the recorded sha rather than against the filesystem.

Six refusals, each because the alternative lets a bad number through:

* **`record()` stamps the git sha from inside this tree.** Never accepted as an
  argument -- corpus-sheet defect 8 is a caller-typed sha, which is a provenance
  *claim*.
* **A git failure is not a clean tree.** The old `_git()` returned `""` when git
  failed and `git_dirty` was `bool("")` -> `false`, so a broken git was
  indistinguishable from a clean checkout. `git_provenance()` returns
  `{"git_sha": None, "dirty": None, "provenance_error": ...}` instead, the shape
  `src/rsr/constants.py` already uses.
* **`command()` refuses an entry point that is not in the tree at the recorded
  sha.** `allow_unreproducible=True` is the escape hatch; it writes a loud row and
  **caps `verdict.outcome` at `inconclusive`**, because a number you cannot re-run
  is not evidence that survived.
* **`exit_code` is required.** Seven of 13 ledgers carried `"exit_code": null` --
  they never learned whether the process exited 0.
* **`write()` refuses a dirty `src/`, `scripts/` or `experiments/`.** A dirty
  `runs/` is normal and fine.
* **A multi-seed statistic must carry its per-seed values**, and `sd` of a single
  sample is `None`, not `0.0`. Zero reads as "measured, and it did not vary".

`device`, `seeds_actually_run`, `steps_requested`, `steps_done`, `status` and a
frozen `manifest.json` with its `config_hash` are written because
`.claude/agents/rsr-manager.md` rejects reports on them and nothing emitted them --
the two evidence formats disagreed and the reviewer's checklist silently could not
run.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import posixpath
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = [
    "DirtyTree",
    "Ledger",
    "MissingStatus",
    "SingleSample",
    "Unreproducible",
    "entry_point",
    "git_provenance",
]

_REPO = Path(__file__).resolve().parents[1]

#: `write()` gates on these. `runs/` churns by design and does not gate.
SOURCE_ROOTS = ("src/", "scripts/", "experiments/")

#: Console scripts whose *code* is in this repository even though their name is
#: not a path in it. Declared with a reason, never widened silently: a gate so
#: strict that the manager's own `pytest` verification has to use the escape hatch
#: is a gate that gets routed around, and the escape hatch caps the verdict.
#:
#: ⚠️ **This is a deviation from the dispatch's literal §5.1**, which says the
#: entry point must be "a repo-relative path existing at the recorded sha". It is
#: recorded here rather than argued in prose so it can be reversed in one line.
TOOL_ENTRY_POINTS = {
    "pytest": "the code it runs is tests/ and src/, both committed; the runner "
    "itself is pinned in pyproject.toml",
    "ruff": "reads pyproject.toml's [tool.ruff] and the committed tree; the "
    "invocation carries no code of its own",
}

#: `status` vocabulary. "unknown" is deliberately absent -- it is the silent-fail
#: shape this field exists to remove.
STATUSES = ("ok", "partial", "crashed", "did_not_run")


class SingleSample(ValueError):
    """Raised when a statistic is asked for across fewer than two samples."""


class Unreproducible(ValueError):
    """Raised when a command's entry point is not in the tree at the recorded sha."""


class DirtyTree(RuntimeError):
    """Raised when `write()` is called over uncommitted source."""


class MissingStatus(RuntimeError):
    """Raised when `write()` is called before `status()`."""


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def _git_exe() -> str | None:
    """Homebrew git, explicitly. Apple's refuses to run until an Xcode licence is
    accepted and fails as an *empty answer rather than an error* -- which has
    already produced two false "everything is clean" reports on this project."""
    for candidate in ("/opt/homebrew/bin/git", "/usr/local/bin/git"):
        if Path(candidate).exists():
            return candidate
    return shutil.which("git")


def _run_git(*args: str) -> subprocess.CompletedProcess[str] | None:
    exe = _git_exe()
    if exe is None:
        return None
    try:
        return subprocess.run(
            [exe, "-C", str(_REPO), *args],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_provenance() -> dict[str, Any]:
    """`{"git_sha", "dirty"}`, or both `None` plus `provenance_error`.

    🔴 Never returns `""`. `bool("")` is `False`, and that is how a git failure
    became a clean bill of health.
    """
    if _git_exe() is None:
        return {"git_sha": None, "dirty": None, "provenance_error": "git not found"}
    sha = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    if sha is None or status is None or sha.returncode != 0 or status.returncode != 0:
        err = (
            (sha.stderr.strip() if sha else "")
            or (status.stderr.strip() if status else "")
            or "git invocation failed"
        )
        return {"git_sha": None, "dirty": None, "provenance_error": err}
    return {"git_sha": sha.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _porcelain() -> str | None:
    r = _run_git("status", "--porcelain")
    if r is None or r.returncode != 0:
        return None
    return r.stdout


def _dirty_source_paths() -> list[str]:
    """Uncommitted paths under `SOURCE_ROOTS`.

    A `None` from `_porcelain()` means git did not answer. That is *not* "clean" --
    it is reported as a single sentinel so `write()` refuses rather than passes.
    """
    out = _porcelain()
    if out is None:
        return ["<git did not answer; cannot prove the tree is clean>"]
    dirty = []
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:  # a rename records both sides
            path = path.split(" -> ")[-1]
        path = path.strip('"')
        if path.startswith(SOURCE_ROOTS):
            dirty.append(path)
    return sorted(dirty)


def _exists_at_sha(path: str, sha: str) -> bool:
    r = _run_git("cat-file", "-e", f"{sha}:{path}")
    return r is not None and r.returncode == 0


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #

_SHELL_TOKENS = {"for", "while", "if", "do", "done", "then", "fi", "cd", "export"}


def _is_python(tok: str) -> bool:
    name = posixpath.basename(tok)
    return name.startswith("python") or name == "pypy"


def entry_point(argv: list[str] | str) -> str | None:
    """The file that carries the code being measured, repo-relative, or `None`.

    `None` means "this command names no file in this repository", which covers
    `python -c '...'`, a shell loop, a `<scratch>` placeholder and an absolute path
    outside the tree. All four are shapes that appear in last night's ledgers.
    """
    try:
        toks = shlex.split(argv) if isinstance(argv, str) else [str(t) for t in argv]
    except ValueError:  # unbalanced quotes
        return None
    if not toks:
        return None
    if any(t in _SHELL_TOKENS for t in toks[:1]) or any(
        c in (argv if isinstance(argv, str) else " ".join(toks)) for c in (";", "|", "&")
    ):
        return None

    i = 0
    while i < len(toks) and "=" in toks[i] and not toks[i].startswith("-"):
        i += 1  # leading VAR=value assignments
    if i >= len(toks):
        return None

    prog = toks[i]
    if posixpath.basename(prog) == "uv":  # `uv run python foo.py`
        i += 1
        while i < len(toks) and (toks[i] == "run" or toks[i].startswith("-")):
            i += 1
        if i >= len(toks):
            return None
        prog = toks[i]

    if _is_python(prog):
        i += 1
        while i < len(toks):
            t = toks[i]
            if t in ("-c", "-"):
                return None  # inline code names no file
            if t == "-m":
                if i + 1 >= len(toks):
                    return None
                return _module_to_path(toks[i + 1])
            if t.startswith("-"):
                i += 1
                continue
            return _repo_relative(t)
        return None
    return _repo_relative(prog)


def _module_to_path(module: str) -> str | None:
    rel = module.replace(".", "/")
    for cand in (f"src/{rel}.py", f"{rel}.py", f"src/{rel}/__main__.py"):
        if (_REPO / cand).exists():
            return cand
    return f"src/{rel}.py"  # let the sha check refuse it


def _repo_relative(tok: str) -> str | None:
    if "<" in tok or ">" in tok:  # `<scratch>/train_arm.py`
        return None
    p = Path(tok)
    if p.is_absolute():
        try:
            return p.relative_to(_REPO).as_posix()
        except ValueError:
            return None  # outside the tree entirely
    if tok.startswith(".."):
        return None
    return posixpath.normpath(tok)


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #


class Ledger:
    """Append-only JSON record for one run.

    >>> led = Ledger("my-run", question="does X differ from Y?")
    >>> led.command(".venv/bin/python scripts/canary.py 4", exit_code=0)
    >>> led.note("victims_rsr", [3, 1, 0], how="policy.records victim field")
    >>> led.stat("loss_final", [1.10, 1.14, 1.09], how="heartbeat last beat, 3 seeds")
    >>> led.status("ok")
    >>> p = led.write()
    """

    def __init__(
        self,
        run_id: str,
        *,
        question: str,
        cycle: int | None = None,
        runs_root: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.runs_root = Path(runs_root) if runs_root is not None else _REPO / "runs"
        self.path = self.runs_root / run_id / "ledger.json"
        prov = git_provenance()
        self.doc: dict[str, Any] = {
            "run_id": run_id,
            "cycle": cycle,
            "question": question,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": {
                **prov,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            # The fields `.claude/agents/rsr-manager.md` reviews on. Written
            # always, `null` when unset, so a missing check is visible as a null
            # rather than as an absent key.
            "device": None,
            "seeds_actually_run": None,
            "steps_requested": None,
            "steps_done": None,
            "status": None,
            "config_hash": None,
            "manifest_written_utc": None,
            "commands": [],
            "rows": [],
            "verdict": None,
        }

    @property
    def _sha(self) -> str | None:
        return self.doc["provenance"].get("git_sha")

    # -- what was run -------------------------------------------------------- #

    def command(
        self,
        argv: list[str] | str,
        *,
        exit_code: int | None,
        note: str = "",
        allow_unreproducible: bool = False,
    ) -> None:
        """The literal command line, and whether it can be run again.

        `exit_code` is keyword-*required* (it may still be `None` for a process
        genuinely not yet finished, but the caller has to say so). Seven of last
        night's 13 ledgers carried `null` here by omission.
        """
        ep = entry_point(argv)
        sha = self._sha
        if ep in TOOL_ENTRY_POINTS:
            why = ""
        elif ep is None:
            why = "names no file in this repository"
        elif sha is None:
            why = (
                f"the sha is unknown ("
                f"{self.doc['provenance'].get('provenance_error')}), so "
                f"{ep!r} cannot be resolved"
            )
        elif not _exists_at_sha(ep, sha):
            why = f"{ep!r} does not exist at {sha[:7]}"
        else:
            why = ""

        if why and not allow_unreproducible:
            raise Unreproducible(
                f"{argv!r}: {why}. The code that ran must be committed -- "
                f"`git_dirty: false` is not enough (runs/cycle-srep-hinge). "
                f"Commit the entry point, or pass allow_unreproducible=True and "
                f"accept that this run's verdict is capped at `inconclusive`."
            )
        self.doc["commands"].append(
            {
                "argv": argv if isinstance(argv, str) else " ".join(argv),
                "entry_point": ep,
                "reproducible": not why,
                "unreproducible_because": why or None,
                "exit_code": exit_code,
                "note": note,
                "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    # -- what the run actually did ------------------------------------------- #

    def run_meta(
        self,
        *,
        device: str | None = None,
        seeds_actually_run: list[int] | None = None,
        steps_requested: int | None = None,
        steps_done: int | None = None,
    ) -> None:
        """What was *run*, as against what was configured (§12.4). A five-seed mean
        from a one-seed run is the failure `seeds_actually_run` exists for."""
        for k, v in (
            ("device", device),
            ("seeds_actually_run", seeds_actually_run),
            ("steps_requested", steps_requested),
            ("steps_done", steps_done),
        ):
            if v is not None:
                self.doc[k] = v

    def status(self, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status={status!r} is not one of {STATUSES}")
        self.doc["status"] = status

    def manifest(self, config: dict[str, Any]) -> Path:
        """Freeze the config to `runs/<id>/manifest.json` and hash it.

        Written when called, not at `write()`, so `manifest_written_utc` can be
        compared against the run -- the reviewer rejects a manifest written after
        the fact.
        """
        blob = json.dumps(config, indent=2, sort_keys=True, default=str) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        (self.path.parent / "manifest.json").write_text(blob)
        self.doc["config_hash"] = hashlib.sha256(blob.encode()).hexdigest()
        self.doc["manifest_written_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        return self.path.parent / "manifest.json"

    # -- numbers ------------------------------------------------------------- #

    def note(self, key: str, value: Any, *, how: str) -> None:
        """One observed value. `how` says which artefact it was read out of, so a
        reader can go and look at that artefact instead of trusting the row."""
        self.doc["rows"].append(
            {"key": key, "kind": "observation", "value": value, "how": how}
        )

    def stat(
        self, key: str, samples: list[float], *, how: str, allow_single: bool = False
    ) -> dict[str, Any]:
        """Mean and sample sd over `samples`, with the samples kept.

        Refuses one sample unless asked explicitly, and never reports `sd = 0.0`
        for it. A zero sd over seeds that should differ is reported as-is and
        flagged -- it is real, and it usually means the seed did not reach the RNG.
        """
        xs = [float(x) for x in samples]
        if len(xs) < 2 and not allow_single:
            raise SingleSample(
                f"{key!r}: a mean/sd over {len(xs)} sample(s) is not a statistic. "
                f"Pass >=2 samples, or allow_single=True and accept sd=None."
            )
        mean = sum(xs) / len(xs)
        if len(xs) < 2:
            sd: float | None = None
        else:
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
        row = {
            "key": key,
            "kind": "statistic",
            "n": len(xs),
            "samples": xs,
            "mean": mean,
            "sd": sd,
            "how": how,
            "sd_exactly_zero": sd is not None and sd == 0.0,
        }
        self.doc["rows"].append(row)
        return row

    # -- the answer ---------------------------------------------------------- #

    def verdict(self, *, falsifier: str, outcome: str, detail: str) -> None:
        """`outcome` is one of `survived`, `falsified`, `inconclusive`.

        `survived` means the falsifier was run and the claim did not die. It is not
        a synonym for "it worked" -- a falsifier that could not run at all is
        `inconclusive`, and saying so is the honest row.

        🔴 **Capped at `inconclusive` if any command was recorded under
        `allow_unreproducible`.** A claim resting on a command nobody can run again
        has not survived anything; it has not been tested.
        """
        if outcome not in {"survived", "falsified", "inconclusive"}:
            raise ValueError(
                f"outcome={outcome!r} is not one of survived / falsified / inconclusive"
            )
        unrepro = [c for c in self.doc["commands"] if not c["reproducible"]]
        capped_from = None
        if unrepro and outcome != "inconclusive":
            capped_from, outcome = outcome, "inconclusive"
        self.doc["verdict"] = {
            "falsifier": falsifier,
            "outcome": outcome,
            "detail": detail,
            "capped_from": capped_from,
            "capped_because": (
                [c["unreproducible_because"] for c in unrepro] if capped_from else None
            ),
        }

    def write(self) -> Path:
        """Refuses over uncommitted source. The gate is "the code that ran is
        committed", not "the tree is clean" -- but an uncommitted `src/`,
        `scripts/` or `experiments/` fails both."""
        if self.doc["status"] is None:
            raise MissingStatus(
                f"{self.run_id}: call status(...) with one of {STATUSES} before "
                f"write(). A run that never recorded whether it finished is the "
                f"shape `exit_code: null` had."
            )
        dirty = _dirty_source_paths()
        if dirty:
            raise DirtyTree(
                f"{self.run_id}: refusing to write a ledger over uncommitted "
                f"source: {dirty}. Commit them first -- a number measured against "
                f"an uncommitted tree is not reproducible from its sha. "
                f"(A dirty runs/ is fine and does not appear here.)"
            )
        self.doc["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=2, default=str) + "\n")
        return self.path
