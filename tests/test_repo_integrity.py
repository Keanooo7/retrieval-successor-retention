"""Everything the working tree has, a fresh clone must have too.

🔴 **Found the hard way, 2026-09-17.** `.gitignore` line 9 was `data/` -- an
**unanchored** pattern, which git matches against a directory of that name at
*any* depth. It therefore swallowed:

* `src/rsr/data/` -- the synthetic-corpus package (gauntlet 2.5), invisible since
  the scaffold commit. A commit that claimed to add the generator added only its
  tests; a fresh clone had `tests/test_synthetic.py` and nothing to import.
* `third_party/ThoughtGestaltCode/tg/data/` -- five files of the pinned reference,
  so the vendored tree **did not match the pin** in the repository, while
  `git status` was clean and the local checkout was complete.

Both failures are silent in exactly the same way: ignored files do not appear in
`git status`, `git add -A` skips them without a word, and every local test passes.
The gauntlet's own rule covers it -- *"did not run" is not "found nothing"* -- and
so does the brief's: a confident negative that was never checked with the right
command. The gauntlet said `src/rsr/data/` "is not there", and against the
repository it was **right**.

These tests check the class of defect, not the two instances.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git() -> str | None:
    for candidate in ("/opt/homebrew/bin/git", "/usr/local/bin/git"):
        if Path(candidate).exists():
            return candidate
    return shutil.which("git")


@pytest.fixture(scope="module")
def tracked() -> set[str]:
    git = _git()
    if git is None:
        pytest.skip("git not available")
    out = subprocess.run(
        [git, "-C", str(ROOT), "ls-files"], capture_output=True, text=True
    )
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return {line for line in out.stdout.splitlines() if line}


def _sources(directory: str) -> list[Path]:
    return sorted(
        p
        for p in (ROOT / directory).rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    )


@pytest.mark.parametrize("directory", ["src", "tests", "scripts"])
def test_every_python_source_is_tracked(tracked, directory):
    """The general form. A module that exists only on this machine is a module the
    tests import locally and no clone can."""
    missing = [
        str(p.relative_to(ROOT))
        for p in _sources(directory)
        if str(p.relative_to(ROOT)) not in tracked
    ]
    assert not missing, (
        "these files exist on disk but not in git, so a fresh clone does not have "
        "them and `git status` will not say so:\n  " + "\n  ".join(missing)
    )


def test_the_synthetic_corpus_package_is_tracked(tracked):
    """The instance that cost the most: gauntlet 2.5's generator."""
    for name in ("__init__.py", "synthetic.py", "coref.py", "pg19.py"):
        assert f"src/rsr/data/{name}" in tracked, name


def test_the_vendored_reference_is_complete(tracked):
    """Gauntlet 2.1's PASS is "the vendored tree's commit equals the pin,
    **verified**". That claim is about the repository, not the working tree.

    The clone at `f220b109` had 42 files. Five of them -- `tg/data/*.py` -- were
    ignored, so the repository held 37 and the pin claim was false for anyone but
    this machine.
    """
    vendored = {t for t in tracked if t.startswith("third_party/ThoughtGestaltCode/")}
    assert len(vendored) == 42, (
        f"the pinned tree has 42 files; git tracks {len(vendored)}. The vendored "
        f"reference must match the pin in the REPOSITORY, not just on disk."
    )
    for name in ("__init__", "gist", "gpt2", "tg", "tg_flat"):
        assert f"third_party/ThoughtGestaltCode/tg/data/{name}.py" in vendored, name


def test_no_source_directory_is_ignored():
    """The rule that caused it, stated as a check.

    `data/` matches at any depth; `/data/` matches only at the repository root.
    Any ignore pattern for a build artefact that could also name a source
    directory must be anchored.
    """
    git = _git()
    if git is None:
        pytest.skip("git not available")
    probes = [
        "src/rsr/data/synthetic.py",
        "src/rsr/model/tg/config.py",
        "third_party/ThoughtGestaltCode/tg/data/tg.py",
        "tests/fixtures/tg_d128_seed0.npz",
    ]
    out = subprocess.run(
        [git, "-C", str(ROOT), "check-ignore", "-v", *probes],
        capture_output=True,
        text=True,
    )
    offending = [
        line
        for line in out.stdout.splitlines()
        if line and not line.split("\t")[0].split(":")[-1].startswith("!")
    ]
    assert not offending, "these paths are ignored and must not be:\n  " + "\n  ".join(
        offending
    )


def test_the_corpus_directory_is_still_ignored(tmp_path):
    """The other direction: anchoring must not stop ignoring what it should.

    `/data/`, `/runs/`, `/outputs/` at the repository root hold corpora and
    disposable artefacts and are never versioned.
    """
    git = _git()
    if git is None:
        pytest.skip("git not available")
    for path in ("data/corpus.bin", "runs/run-1/log.txt", "outputs/x.json"):
        out = subprocess.run(
            [git, "-C", str(ROOT), "check-ignore", "-q", path], capture_output=True
        )
        assert out.returncode == 0, f"{path} should still be ignored"
