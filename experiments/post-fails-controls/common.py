"""Shared machinery for the post-FAILS controls, phase 1 (items 1a-1e).

PREREG ``experiments/post-fails-controls/PREREG.md``, committed alone at 9f22d87. This
module holds what every item needs and nothing item-specific:

* the PREREG front matter, machine-read (G-C1(d): the sealed bounds are read from the
  file, never retyped in code; the registered seeds likewise);
* **G-C1 (a)/(b)** the seal guard: ``rsr.data.synthetic._generate_document`` and
  ``generate`` wrapped for the whole process, raising :class:`SealedRange` for any id
  in TUNE or EVAL2 at any seed and any ``SyntheticConfig``, every module-level
  reference to the originals rebound (so ``from rsr.data.synthetic import
  _generate_document`` done before or after :func:`install_seal` reaches the guard),
  and a per-seed census of every id generated;
* **G-C1 (e)** the launch-time re-run of PREREG §1.1's search over every local
  branch, for any integer inside TUNE or EVAL2;
* **G-C2** the substrate manifest check (``shasum -a 256 -c``, rc read directly) and
  checkpoint pins;
* **G-C3** the census of a full pytest run's log;
* **G-C4** ``rsr.constants.record`` refused for the process; gradient check;
* ledger helpers (``scripts/ledger.py``) and the sha256 pins of every input read.

Imports nothing heavy at module level, so ``install_seal`` can run before any harness
module is loaded. The seal is process-wide and cannot be uninstalled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PREREG_REL = "experiments/post-fails-controls/PREREG.md"
PREREG_PATH = ROOT / PREREG_REL
PREREG_COMMIT = "9f22d87"
RUN_ID_PREFIX = "post-fails-controls-"
GIT = "/opt/homebrew/bin/git"  # never bare git (CLAUDE.md)

#: G-C1(e): paths whose sealed-range literals are phase 1's own registration and
#: generated outputs, not "a new hit". Everything else is a new hit.
SEARCH_ALLOWED = (
    re.compile(r"^experiments/post-fails-controls/PREREG\.md$"),
    re.compile(r"^experiments/post-fails-controls/RESULTS[^/]*\.md$"),
    re.compile(r"^runs/post-fails-controls-[^/]+/.*\.json$"),
    re.compile(r"^runs/post-fails-controls-[^/]+/RESULTS[^/]*\.md$"),
)


class SealedRange(RuntimeError):
    """A sealed document id (TUNE or EVAL2) was about to be generated."""


class ControlFailure(RuntimeError):
    """A registered control failed: the item exits 3."""


# --------------------------------------------------------------------------- #
# The PREREG front matter (G-C1(d))
# --------------------------------------------------------------------------- #

_RANGE = re.compile(r"^\[(\d+), (\d+)\)$")


def front_matter(path: Path | str = PREREG_PATH) -> dict[str, Any]:
    """The YAML block between the first two ``---`` lines of the PREREG."""
    import yaml

    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ControlFailure(f"{path}: no front matter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    return yaml.safe_load("\n".join(lines[1:end]))


def sealed_ranges(path: Path | str = PREREG_PATH) -> dict[str, tuple[int, int]]:
    """``{"TUNE": (lo, hi), "EVAL2": (lo, hi)}``, half-open, read from the front
    matter's ``sealed:`` mapping. Refuses anything not of the form ``[a, b)`` with
    ``a < b``."""
    fm = front_matter(path)
    sealed = fm.get("sealed")
    if not isinstance(sealed, dict) or not sealed:
        raise ControlFailure(f"{path}: front matter has no sealed: mapping")
    out: dict[str, tuple[int, int]] = {}
    for name, spec in sealed.items():
        m = _RANGE.match(str(spec).strip())
        if not m:
            raise ControlFailure(f"sealed {name}: {spec!r} is not a half-open [a, b)")
        lo, hi = int(m.group(1)), int(m.group(2))
        if not lo < hi:
            raise ControlFailure(f"sealed {name}: empty range {spec!r}")
        out[str(name)] = (lo, hi)
    return out


def registered_seeds(path: Path | str = PREREG_PATH) -> tuple[int, ...]:
    return tuple(int(s) for s in json.loads(str(front_matter(path)["seeds"])))


def sealed_name(doc_id: int, ranges: dict[str, tuple[int, int]]) -> str | None:
    for name, (lo, hi) in ranges.items():
        if lo <= int(doc_id) < hi:
            return name
    return None


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


# --------------------------------------------------------------------------- #
# G-C1 (a), (b), (c): the seal guard
# --------------------------------------------------------------------------- #

_SEAL: dict[str, Any] = {
    "installed": False,
    "ranges": None,
    "orig": {},  # attribute name -> original function
    "guard": {},  # attribute name -> guarded function
    "generated": defaultdict(set),  # seed -> ids generated (only non-sealed ever land)
    "refused": [],  # (seed, id, range) of every refused call
}


def _rebind_everywhere() -> list[str]:
    """Replace every module-level reference to an original generator by its guard."""
    orig, guard = _SEAL["orig"], _SEAL["guard"]
    rebound: list[str] = []
    for mname, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for attr, val in list(d.items()):
            for k, f in orig.items():
                if val is f:
                    setattr(mod, attr, guard[k])
                    rebound.append(f"{mname}.{attr}")
    return rebound


def install_seal(path: Path | str = PREREG_PATH) -> dict:
    """Install the process-wide guard (idempotent). Returns what it did."""
    import rsr.data.synthetic as syn

    ranges = sealed_ranges(path)
    if not _SEAL["installed"]:
        _SEAL["ranges"] = ranges
        orig_doc, orig_gen = syn._generate_document, syn.generate

        def _generate_document(doc_id, cfg):
            name = sealed_name(doc_id, _SEAL["ranges"])
            if name is not None:
                _SEAL["refused"].append((int(cfg.seed), int(doc_id), name))
                raise SealedRange(
                    f"document {doc_id} (seed {cfg.seed}) is in sealed range {name} "
                    f"{_SEAL['ranges'][name]} (PREREG §1, G-C1)"
                )
            _SEAL["generated"][int(cfg.seed)].add(int(doc_id))
            return orig_doc(doc_id, cfg)

        def generate(cfg=None):
            cfg = cfg or syn.SyntheticConfig()
            for name, rng in _SEAL["ranges"].items():
                if _overlaps((0, int(cfg.n_documents)), rng):
                    _SEAL["refused"].append((int(cfg.seed), int(rng[0]), name))
                    raise SealedRange(
                        f"generate(n_documents={cfg.n_documents}) would reach sealed "
                        f"range {name} {rng} (PREREG §1, G-C1)"
                    )
            return orig_gen(cfg)

        _generate_document.__pfc_seal__ = True  # type: ignore[attr-defined]
        generate.__pfc_seal__ = True  # type: ignore[attr-defined]
        _generate_document.__wrapped__ = orig_doc  # type: ignore[attr-defined]
        generate.__wrapped__ = orig_gen  # type: ignore[attr-defined]
        _SEAL["orig"] = {"_generate_document": orig_doc, "generate": orig_gen}
        _SEAL["guard"] = {"_generate_document": _generate_document, "generate": generate}
        syn._generate_document = _generate_document
        syn.generate = generate
        _SEAL["installed"] = True
    elif ranges != _SEAL["ranges"]:
        raise ControlFailure(
            f"seal already installed with {_SEAL['ranges']}, not {ranges}"
        )
    rebound = _rebind_everywhere()
    return {"installed": True, "ranges": dict(_SEAL["ranges"]), "rebound_now": rebound}


def seal_leaks() -> list[str]:
    """Every module attribute still holding an original (unguarded) generator."""
    if not _SEAL["installed"]:
        return ["seal not installed"]
    out = []
    origs = list(_SEAL["orig"].values())
    for mname, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for attr, val in list(d.items()):
            if any(val is f for f in origs):
                out.append(f"{mname}.{attr}")
    return out


def seal_census(named: dict[str, tuple[int, int]] | None = None) -> dict:
    """G-C1(b): the ids generated in this process, per seed and named range (count,
    min, max), with the sealed ranges always listed (count must be 0)."""
    ranges = dict(_SEAL["ranges"] or {})
    named = dict(named or {})
    per_seed: dict[str, dict] = {}
    n_sealed = 0
    for seed, ids in sorted(_SEAL["generated"].items()):
        rows: dict[str, dict] = {}
        rest = set(ids)
        for name, (lo, hi) in [*named.items(), *ranges.items()]:
            inside = sorted(i for i in ids if lo <= i < hi)
            rest -= set(inside)
            rows[name] = {
                "range": [lo, hi],
                "count": len(inside),
                "min": inside[0] if inside else None,
                "max": inside[-1] if inside else None,
            }
            if name in ranges:
                n_sealed += len(inside)
        other = sorted(rest)
        rows["other"] = {
            "count": len(other),
            "min": other[0] if other else None,
            "max": other[-1] if other else None,
        }
        per_seed[str(seed)] = rows
    leaks = seal_leaks()
    return {
        "per_seed": per_seed,
        "sealed_ranges": {k: list(v) for k, v in ranges.items()},
        "n_sealed_generated": n_sealed,
        "refused_calls": [list(x) for x in _SEAL["refused"]],
        "leaks": leaks,
        "ok": bool(_SEAL["installed"]) and n_sealed == 0 and not leaks,
    }


# --------------------------------------------------------------------------- #
# G-C1 (e): PREREG §1.1's search, re-run at launch on every local branch
# --------------------------------------------------------------------------- #

#: An integer token: not preceded by a word character, a dot (a fraction's digits) or
#: an underscore, not followed by a word character. Decimal or ``_``-grouped.
_INT_TOKEN = re.compile(r"(?<![0-9A-Za-z_.])([0-9]{1,3}(?:_?[0-9]{3})+)(?![0-9A-Za-z_])")


def search_pattern(ranges: dict[str, tuple[int, int]]) -> str:
    """A ``git grep -E`` prefilter for every integer in the ranges, in decimal or
    ``_``-grouped form. The exact range test is done in Python on each token."""
    heads: set[str] = set()
    for lo, hi in ranges.values():
        for n in (lo, hi - 1):
            if len(str(n)) != 6:
                raise ControlFailure(f"search_pattern supports 6-digit bounds, not {n}")
        heads |= {str(n)[:3] for n in range(lo, hi)}
    return "(" + "|".join(sorted(heads)) + ")_?[0-9]{3}"


def search_allowed(path: str) -> bool:
    return any(p.match(path) for p in SEARCH_ALLOWED)


def branch_search(
    ranges: dict[str, tuple[int, int]] | None = None,
    *,
    repo: Path | str = ROOT,
    git: str = GIT,
) -> dict:
    """G-C1(e): every local branch, ``git grep`` for any integer in TUNE or EVAL2.
    A hit outside :data:`SEARCH_ALLOWED` is a NEW hit (exit 3, to the PM). A git
    failure is recorded as did-not-run (never as "no hit")."""
    ranges = ranges or sealed_ranges()
    pat = search_pattern(ranges)
    r = subprocess.run(
        [git, "for-each-ref", "--format=%(refname)", "refs/heads"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    out: dict[str, Any] = {
        "git": git,
        "pattern": pat,
        "token_regex": _INT_TOKEN.pattern,
        "allowed": [p.pattern for p in SEARCH_ALLOWED],
        "for_each_ref_rc": r.returncode,
    }
    if r.returncode != 0:
        return {**out, "ok": False, "error": r.stderr.strip()[:500]}
    refs = [x for x in r.stdout.split() if x]
    hits: dict[tuple, dict] = {}
    errors = []
    for ref in refs:
        g = subprocess.run(
            [git, "grep", "-I", "-n", "-E", pat, ref, "--"],
            cwd=repo,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if g.returncode not in (0, 1):  # 1 = no match; anything else did not run
            errors.append({"ref": ref, "rc": g.returncode, "stderr": g.stderr[:300]})
            continue
        for line in g.stdout.splitlines():
            # <ref>:<path>:<lineno>:<content>
            rest = line[len(ref) + 1 :] if line.startswith(ref + ":") else line
            parts = rest.split(":", 2)
            if len(parts) < 3:
                continue
            path, lineno, content = parts
            for m in _INT_TOKEN.finditer(content):
                v = int(m.group(1).replace("_", ""))
                name = sealed_name(v, ranges)
                if name is None:
                    continue
                key = (path, m.group(1), content.strip()[:200])
                h = hits.setdefault(
                    key,
                    {
                        "path": path,
                        "token": m.group(1),
                        "value": v,
                        "range": name,
                        "line": int(lineno),
                        "excerpt": content.strip()[:200],
                        "allowed": search_allowed(path),
                        "refs": [],
                    },
                )
                h["refs"].append(ref)
    hl = sorted(hits.values(), key=lambda h: (h["path"], h["line"], h["token"]))
    new = [h for h in hl if not h["allowed"]]
    return {
        **out,
        "n_refs": len(refs),
        "refs": refs,
        "errors": errors,
        "n_hits": len(hl),
        "hits": hl,
        "n_new_hits": len(new),
        "new_hits": new,
        "ok": not errors and not new and len(refs) > 0,
    }


# --------------------------------------------------------------------------- #
# G-C2: substrate manifest and checkpoint pins
# --------------------------------------------------------------------------- #


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_check(manifest: Path | str, expected_sha256: str, cwd: Path | str) -> dict:
    """``shasum -a 256 -c <manifest>`` from ``cwd`` (the manifest's paths are relative
    to it), its rc read directly from the process; and the manifest's own sha256
    against ``expected_sha256``."""
    own = sha256_file(manifest)
    r = subprocess.run(
        ["shasum", "-a", "256", "-c", str(manifest)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    lines = r.stdout.splitlines()
    bad = [x for x in lines if not x.endswith(": OK")]
    return {
        "command": f"shasum -a 256 -c {manifest}",
        "cwd": str(cwd),
        "rc": r.returncode,
        "n_ok": sum(1 for x in lines if x.endswith(": OK")),
        "n_lines": len(lines),
        "not_ok": bad[:20],
        "stderr": r.stderr.strip()[:500],
        "manifest_sha256": own,
        "expected_manifest_sha256": expected_sha256,
        "ok": r.returncode == 0 and own == expected_sha256 and not bad,
    }


def checkpoint_pin(path: Path | str, expected_sha256: str) -> dict:
    got = sha256_file(path)
    return {
        "path": str(path),
        "sha256": got,
        "pin": expected_sha256,
        "ok": got == expected_sha256,
    }


class Pins:
    """sha256 of every input file, at first read; :meth:`recheck` at the end."""

    def __init__(self) -> None:
        self.first: dict[str, str] = {}

    def pin(self, path: Path | str) -> str:
        p = str(Path(path))
        sha = sha256_file(p)
        if p in self.first and self.first[p] != sha:
            raise ControlFailure(f"{p} changed between reads")
        self.first.setdefault(p, sha)
        return sha

    def recheck(self) -> dict:
        rows = {}
        for p, sha in sorted(self.first.items()):
            try:
                now = sha256_file(p)
            except OSError as e:
                now = f"unreadable: {e}"
            rows[p] = {"first": sha, "end": now, "ok": now == sha}
        return {
            "files": rows,
            "n_files": len(rows),
            "ok": bool(rows) and all(r["ok"] for r in rows.values()),
        }


# --------------------------------------------------------------------------- #
# G-C3: the suite census; G-C4: frozen
# --------------------------------------------------------------------------- #

_CENSUS_WORDS = ("passed", "failed", "skipped", "errors", "error", "xfailed",
                 "xpassed", "deselected", "warnings", "warning")  # fmt: skip


def suite_census(log_text: str, rc: int | None) -> dict:
    """The counts of a pytest run: the repo's own census line (``tests/conftest.py``:
    ``passed=N failed=N skipped=N errors=N``) when present, else pytest's final
    summary line; and every SKIPPED line (``-rs`` and the census's ``SKIPPED[n]``).
    ``rc`` is the process status, read directly by the caller."""
    summary = None
    counts = dict.fromkeys(("passed", "failed", "skipped", "errors"), 0)
    lines = log_text.splitlines()
    census = [
        x for x in lines if re.match(r"^passed=\d+ failed=\d+ skipped=\d+ errors=\d+", x)
    ]
    if census:
        summary = census[-1].strip()
        for k, n in re.findall(r"([a-z]+)=(\d+)", summary):
            counts[k] = int(n)
    else:
        for line in reversed(lines):
            s = line.strip().strip("=").strip()
            if re.search(r"\b(passed|failed|error|errors|skipped)\b", s) and " in " in s:
                summary = s
                break
        if summary:
            for n, word in re.findall(r"(\d+) ([a-z]+)", summary):
                w = {"error": "errors", "warning": "warnings"}.get(word, word)
                if w in _CENSUS_WORDS:
                    counts[w] = int(n)
    skipped = [x.strip() for x in lines if x.lstrip().startswith("SKIPPED")]
    return {
        "rc": rc,
        "summary": summary,
        **counts,
        "skipped_tests": skipped,
        "ok": rc == 0
        and summary is not None
        and counts["failed"] == 0
        and counts["errors"] == 0,
    }


def forbid_constants_record() -> dict:
    """G-C4: ``rsr.constants.record`` (module function and registry method) raise for
    the rest of the process. Returns what was replaced."""
    from rsr import constants as C

    def refuse(*_a, **_k):
        raise ControlFailure("G-C4: rsr.constants.record() was called in phase 1")

    done = []
    if hasattr(C, "record"):
        C.record = refuse
        done.append("rsr.constants.record")
    reg = getattr(C, "REGISTRY", None)
    if reg is not None and hasattr(reg, "record"):
        reg.record = refuse  # instance attribute shadows the method
        done.append("rsr.constants.REGISTRY.record")
    return {"replaced": done, "ok": bool(done)}


def no_grads(module) -> dict:
    """G-C4: no parameter of ``module`` holds a gradient."""
    with_grad = [n for n, p in module.named_parameters() if p.grad is not None]
    return {
        "n_params": sum(1 for _ in module.parameters()),
        "with_grad": with_grad[:10],
        "ok": not with_grad,
    }


# --------------------------------------------------------------------------- #
# Ledger helpers
# --------------------------------------------------------------------------- #


def load_module(name: str, rel: str):
    """Import an experiment script by path, as its own module instance. Never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ledger_mod():
    import ledger as _ledger  # scripts/ledger.py

    return _ledger


def ledger_rows(path: Path | str) -> dict[str, Any]:
    return {r["key"]: r for r in json.loads(Path(path).read_text())["rows"]}


def git_head(repo: Path | str = ROOT) -> str | None:
    r = subprocess.run(
        [GIT, "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    )
    return r.stdout.strip() if r.returncode == 0 else None


def git_dirty_paths(repo: Path | str = ROOT) -> list[str] | None:
    r = subprocess.run(
        [GIT, "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    return [
        x[3:] for x in r.stdout.splitlines() if x[3:] and not x[3:].startswith("runs/")
    ]


def flatten(prefix: str, x: Any, depth: int = 0, max_depth: int = 6):
    """``(dotted key, leaf)`` pairs; dicts deeper than ``max_depth`` stay whole."""
    if isinstance(x, dict) and x and depth < max_depth:
        for k, v in x.items():
            yield from flatten(f"{prefix}.{k}", v, depth + 1, max_depth)
    else:
        yield prefix, x


EXIT_RANK = {0: 0, 1: 1, 3: 2}  # worst wins: 3 > 1 > 0 (exit 5 is not used, §0)


def worst_exit(codes) -> int:
    codes = [int(c) for c in codes]
    if not codes:
        return 3
    for c in codes:
        if c not in EXIT_RANK:
            raise ValueError(f"exit {c} is not used in phase 1 (§0)")
    return max(codes, key=lambda c: EXIT_RANK[c])


LEDGER_STATUS = {0: "ok", 1: "failed", 3: "did_not_run"}
