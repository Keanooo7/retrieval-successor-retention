"""Tell the owner something -- on ONE pinned GitHub issue labelled `rsr-digest`.

    PYTHONPATH=scripts .venv/bin/python -m orchestrator.notify "<title>" <body-file>

R-2026-09-22-owner-out-of-loop: the owner is contacted only for results and major
updates, and only here. **No other service is ever contacted** -- not mail, not
chat, not a webhook. The first notice creates the issue (and pins it); every later
one is a comment on it.

Dedupe: the sha256 of (title, body) is recorded in
`.orchestrator/state/notified.json`; the same content is never posted twice, so a
tick that re-derives the same idle digest every ten minutes posts it once.

`gh` is `RSR_GH_BIN` if set (the tests point it at a stub), else `gh` on PATH.

Exit: 0 posted or already posted · 1 `gh` refused · 3 `gh` missing, or the body
file is unreadable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from orchestrator import loopcore as lc
from rsr.exit_codes import ArgumentParser, Exit, run_main, status

LABEL = "rsr-digest"
ISSUE_TITLE = "RSR orchestrator digest"


def gh_bin() -> str:
    return os.environ.get("RSR_GH_BIN") or "gh"


def _gh(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [gh_bin(), *args], cwd=root, capture_output=True, text=True, check=False
    )


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256(json.dumps([title, body]).encode()).hexdigest()


def post(root: Path, title: str, body: str, *, dry_run: bool = False) -> Exit:
    """Post `title`/`body` once. Returns the protocol code; never raises for `gh`."""
    state_path = lc.sub(root, "state") / "notified.json"
    seen: dict = lc.load_json(state_path, {})
    h = content_hash(title, body)
    if h in seen:
        lc.log(f"notify: already posted ({h[:12]}) -- {title}")
        return Exit.OK
    if dry_run:
        lc.log(f"notify (DRY_RUN, not posted): {title}")
        return Exit.OK
    text = f"## {title}\n\n{body}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(text)
        body_file = f.name
    try:
        try:
            found = _gh(
                root,
                "issue", "list", "--label", LABEL, "--state", "open",
                "--json", "number", "--limit", "1",
            )  # fmt: skip
        except FileNotFoundError:
            print(f"DID NOT RUN: `{gh_bin()}` is not installed", file=sys.stderr)
            return Exit.DID_NOT_RUN
        if found.returncode != 0:
            print(f"gh issue list failed (rc={found.returncode}): {found.stderr}")
            return Exit.FAIL
        try:
            issues = json.loads(found.stdout or "[]")
        except json.JSONDecodeError:
            issues = []
        if issues:
            number = str(issues[0]["number"])
            proc = _gh(root, "issue", "comment", number, "--body-file", body_file)
        else:
            _gh(root, "label", "create", LABEL, "--force", "--color", "5319e7")
            proc = _gh(
                root,
                "issue", "create", "--title", ISSUE_TITLE,
                "--body-file", body_file, "--label", LABEL,
            )  # fmt: skip
            if proc.returncode == 0:
                url = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
                number = url.rstrip("/").rsplit("/", 1)[-1]
                pin = _gh(root, "issue", "pin", number)
                if pin.returncode != 0:
                    lc.log(f"notify: pin refused (rc={pin.returncode}); issue unpinned")
        if proc.returncode != 0:
            print(f"gh refused (rc={proc.returncode}): {proc.stderr}", file=sys.stderr)
            return Exit.FAIL
    finally:
        Path(body_file).unlink(missing_ok=True)
    seen[h] = {"title": title, "at": lc.now().isoformat(timespec="seconds")}
    lc.write_json(state_path, seen)
    lc.log(f"notify: posted -- {title}")
    return Exit.OK


def main() -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("title")
    ap.add_argument("body_file", type=Path)
    args = ap.parse_args()
    try:
        body = args.body_file.read_text()
    except OSError as e:
        print(f"DID NOT RUN: cannot read the body file: {e}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    dry = os.environ.get("DRY_RUN") == "1"
    return status(post(lc.root(), args.title, body, dry_run=dry))


if __name__ == "__main__":
    run_main(main)
