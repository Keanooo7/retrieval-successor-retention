"""The verifier's two mechanical steps: the seeded claim draw, and the record check.

Usage::

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.verify draw <run_id> [--k N]
    PYTHONPATH=scripts .venv/bin/python -m orchestrator.verify check <run_id>

**Why a seeded draw.** The manager used to choose which of a researcher's claims to
re-execute (`rsr-manager.md`, "Verify by a different path"). A chooser who has read
the report can -- without meaning to -- pick the claims it already believes. The
verifier (`.claude/agents/rsr-verifier.md`) does not choose: the claims come from
`runs/<run_id>/claims.json`, the draw is seeded with ``sha256(run_id)``, and anyone
can recompute which claims were due. A draw that cannot be recomputed is a choice.

`claims.json` is a JSON list of ``{"claim", "command", "expected"}`` written by the
researcher (its role file says so). The verifier never reads the report prose.

The record, `runs/<run_id>/verification.json`, is::

    {"status": "ok" | "failed" | "inconclusive",
     "seed": "<sha256(run_id) hex>",
     "claims": [{"claim", "command", "expected", "observed", "match"}],
     "verifier_sha": "<40-hex git sha the verifier ran at>"}

``check`` refuses a record whose seed is not this run's, whose status disagrees
with its own `match` column (``ok`` with a mismatch, ``failed`` with none), or whose
claims are not exactly the drawn ones.

Exit codes: ``draw`` exits `3` when there is no claims list and `2` when it is
empty (nothing to compare); ``check`` exits `1` on an invalid record, `3` when there
is none.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from rsr.exit_codes import (  # noqa: E402
    ArgumentParser,
    Exit,
    did_not_run,
    run_main,
    status,
)

STATUSES = ("ok", "failed", "inconclusive")
CLAIM_KEYS = ("claim", "command", "expected")
RECORD_CLAIM_KEYS = (*CLAIM_KEYS, "observed", "match")
DEFAULT_K = 3


def seed_for(run_id: str) -> str:
    """`sha256(run_id)`, hex. Stored as a string: a 256-bit int does not survive
    a JSON reader that parses numbers as doubles."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()


def draw_indices(run_id: str, n: int, k: int = DEFAULT_K) -> list[int]:
    """Which of `n` claims are re-executed: `min(k, n)` of them, seeded by run id."""
    rng = random.Random(int(seed_for(run_id), 16))
    return sorted(rng.sample(range(n), min(k, n)))


def load_claims(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of claims")
    for i, c in enumerate(data):
        missing = [k for k in CLAIM_KEYS if not isinstance(c, dict) or k not in c]
        if missing:
            raise ValueError(f"{path}: claim {i} lacks {missing}")
    return data


def validate_record(
    rec: object, run_id: str | None = None, drawn: list[dict] | None = None
) -> list[str]:
    """Every reason `rec` is not a valid verification record. Empty == valid."""
    if not isinstance(rec, dict):
        return ["the record is not a JSON object"]
    problems = []
    st = rec.get("status")
    if st not in STATUSES:
        problems.append(f"status {st!r} is not one of {STATUSES}")
    if run_id is not None and rec.get("seed") != seed_for(run_id):
        problems.append(f"seed is not sha256({run_id!r})")
    sha = rec.get("verifier_sha")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)):
        problems.append("verifier_sha is not a 40-hex git sha")
    claims = rec.get("claims")
    if not isinstance(claims, list):
        return [*problems, "claims is not a list"]
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            problems.append(f"claims[{i}] is not an object")
            continue
        missing = [k for k in RECORD_CLAIM_KEYS if k not in c]
        if missing:
            problems.append(f"claims[{i}] lacks {missing}")
        elif not isinstance(c["match"], bool):
            problems.append(f"claims[{i}].match is not a bool")
    matches = [c.get("match") for c in claims if isinstance(c, dict)]
    if st == "ok" and (not claims or not all(m is True for m in matches)):
        problems.append("status ok needs at least one claim and every match true")
    if st == "failed" and not any(m is False for m in matches):
        problems.append("status failed needs at least one match false")
    if drawn is not None:
        want = [c["claim"] for c in drawn]
        got = [c.get("claim") for c in claims if isinstance(c, dict)]
        if got != want:
            problems.append(f"claims re-executed {got} are not the drawn ones {want}")
    return problems


def _claims_path(root: Path, run_id: str) -> Path:
    return root / "runs" / run_id / "claims.json"


def cmd_draw(root: Path, run_id: str, k: int) -> Exit:
    path = _claims_path(root, run_id)
    if not path.is_file():
        return did_not_run(f"no claims list at {path}")
    claims = load_claims(path)
    if not claims:
        print(f"UNKNOWN: {path} lists no claims; nothing to re-execute", file=sys.stderr)
        return Exit.UNKNOWN
    idx = draw_indices(run_id, len(claims), k)
    out = {
        "run_id": run_id,
        "seed": seed_for(run_id),
        "n_claims": len(claims),
        "k": len(idx),
        "chosen": [{"index": i, **claims[i]} for i in idx],
    }
    print(json.dumps(out, indent=2))
    return Exit.OK


def cmd_check(root: Path, run_id: str, k: int) -> Exit:
    path = root / "runs" / run_id / "verification.json"
    if not path.is_file():
        return did_not_run(f"no verification record at {path}")
    drawn = None
    cpath = _claims_path(root, run_id)
    if cpath.is_file():
        claims = load_claims(cpath)
        drawn = [claims[i] for i in draw_indices(run_id, len(claims), k)]
    problems = validate_record(json.loads(path.read_text()), run_id, drawn)
    for p in problems:
        print(f"INVALID: {p}")
    if problems:
        return Exit.FAIL
    print(f"OK: {path} is a valid record")
    return Exit.OK


def main() -> Exit:
    ap = ArgumentParser(prog="orchestrator.verify")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="repo root")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="claims to re-execute")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("draw", "check"):
        sub.add_parser(name).add_argument("run_id")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.cmd == "draw":
        return status(cmd_draw(root, args.run_id, args.k))
    return status(cmd_check(root, args.run_id, args.k))


if __name__ == "__main__":
    run_main(main)
