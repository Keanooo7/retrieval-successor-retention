"""PREREG post-fails-controls phase 1: the shared module ``common.py`` (G-C1 to G-C4).

A naming-only split of PREREG §0's single ``tests/test_post_fails_controls.py``: this
file covers the shared controls, ``test_post_fails_controls_1a.py`` covers item 1a,
and the items 1b-1e have their own files.

🔴 **No sealed document is generated here.** The sealed bounds are read from the
PREREG's front matter (G-C1(d)); no test file types them. Every call that names a
sealed id is expected to raise *before* the document is built. The only documents
generated are U's (ids 64, 65 at seed 0: already inspected by B0, CF, offline-psi).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "post-fails-controls" / "common.py"
GIT = "/opt/homebrew/bin/git"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def C():
    mod = _load("pfc_common", COMMON)
    mod.install_seal()
    return mod


@pytest.fixture(scope="module")
def R1(C):
    return C.load_module("_pfc_test_r1", "experiments/rsr-run1/run.py")


# --------------------------------------------------------------------------- #
# G-C1(d): the bounds are machine-read
# --------------------------------------------------------------------------- #


def test_sealed_ranges_are_read_from_the_front_matter(C, R1):
    r = C.sealed_ranges()
    assert set(r) == {"TUNE", "EVAL2"}
    (a, b), (c, d) = r["TUNE"], r["EVAL2"]
    assert (b - a, d - c) == (512, 2048)  # PREREG §1's sizes
    assert not C._overlaps(r["TUNE"], r["EVAL2"])
    # below GEN_SEED_K, so seed*K + id cannot collide across seeds (PREREG §1)
    assert max(b, d) <= R1.GEN_SEED_K
    for lo, hi in r.values():
        for name, used in R1.USED_RANGES.items():
            assert not C._overlaps((lo, hi), used), name
        assert not C._overlaps((lo, hi), R1.EVAL)
        assert not C._overlaps((lo, hi), R1.POOL)


def test_front_matter_refuses_a_malformed_range(C, tmp_path):
    p = tmp_path / "P.md"
    p.write_text('---\nseeds: "[0, 1, 2]"\nsealed:\n  TUNE: "[5, 3)"\n---\n# x\n')
    with pytest.raises(C.ControlFailure):
        C.sealed_ranges(p)
    p.write_text('---\nsealed:\n  TUNE: "5..9"\n---\n')
    with pytest.raises(C.ControlFailure):
        C.sealed_ranges(p)
    p.write_text("# no front matter\n")
    with pytest.raises(C.ControlFailure):
        C.sealed_ranges(p)


def test_registered_seeds(C):
    assert C.registered_seeds() == (0, 1, 2)


def test_no_test_or_code_file_types_a_sealed_bound(C):
    """G-C1(d): the phase-1 code and tests read the bounds; none retypes them."""
    files = [*sorted((ROOT / "experiments" / "post-fails-controls").glob("*.py"))]
    files += sorted((ROOT / "tests").glob("test_post_fails_controls*.py"))
    for lo, hi in C.sealed_ranges().values():
        for n in (lo, hi):
            forms = (str(n), f"{n // 1000}_{n % 1000:03d}")
            for f in files:
                text = f.read_text()
                assert not any(x in text for x in forms), (f.name, n)


# --------------------------------------------------------------------------- #
# G-C1(a)-(c): the guard
# --------------------------------------------------------------------------- #


def _syn():
    import rsr.data.synthetic as syn

    return syn


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "knobs",
    [{}, {"geometric_p": 0.10, "heavy_tail_weight": 0.70}, {"geometric_p": 0.80}],
)
def test_guard_raises_on_every_sealed_bound(C, seed, knobs):
    syn = _syn()
    cfg = syn.SyntheticConfig(sentences_per_document=48, seed=seed, **knobs)
    for lo, hi in C.sealed_ranges().values():
        for i in (lo, (lo + hi) // 2, hi - 1):
            with pytest.raises(C.SealedRange):
                syn._generate_document(i, cfg)
        assert C.sealed_name(lo - 1, C.sealed_ranges()) is None
        assert C.sealed_name(hi, C.sealed_ranges()) is None


def test_generate_refuses_before_building_anything(C):
    syn = _syn()
    lo = min(lo for lo, _ in C.sealed_ranges().values())
    with pytest.raises(C.SealedRange):
        syn.generate(syn.SyntheticConfig(seed=0, n_documents=lo + 1))


def test_harness_paths_reach_the_guard(C, R1):
    """``R1.docs_by_id`` (a ``from ... import`` binding) and lookahead-room's
    ``documents`` / corpus-size-curve's ``_generate`` (``generate``) all raise."""
    lo, hi = C.sealed_ranges()["EVAL2"]
    cfg = R1.prereg_config(1, 3000)
    with pytest.raises(C.SealedRange):
        R1.docs_by_id(cfg, [lo])
    import rsr.train.loop as loop

    assert getattr(loop._generate_document, "__pfc_seal__", False)
    assert getattr(loop.generate, "__pfc_seal__", False)
    LR = C.load_module("_pfc_test_lr", "experiments/lookahead-room/run.py")
    with pytest.raises(C.SealedRange):
        LR.CSC._generate(0, hi)
    t_lo, _ = C.sealed_ranges()["TUNE"]
    with pytest.raises(C.SealedRange):
        LR.documents(2, (t_lo, t_lo + 1))


def test_a_reference_taken_before_install_is_rebound(C):
    syn = _syn()
    orig = syn._generate_document.__wrapped__
    fake = types.ModuleType("_pfc_fake_early_importer")
    fake.gd = orig
    fake.gen = syn.generate.__wrapped__
    sys.modules[fake.__name__] = fake
    try:
        assert "_pfc_fake_early_importer.gd" in C.seal_leaks()
        rep = C.install_seal()
        assert "_pfc_fake_early_importer.gd" in rep["rebound_now"]
        assert fake.gd is syn._generate_document
        assert fake.gen is syn.generate
        assert not [x for x in C.seal_leaks() if x.startswith("_pfc_fake")]
    finally:
        del sys.modules[fake.__name__]


def test_guard_passes_unsealed_documents_through_unchanged_and_counts_them(C):
    syn = _syn()
    cfg = syn.SyntheticConfig(sentences_per_document=48, seed=0)
    a = syn._generate_document(64, cfg)
    b = syn._generate_document.__wrapped__(64, cfg)
    assert a == b
    syn._generate_document(65, cfg)
    cen = C.seal_census({"U": (64, 1088)})
    assert cen["per_seed"]["0"]["U"]["count"] >= 2
    assert cen["per_seed"]["0"]["U"]["min"] <= 64
    assert cen["n_sealed_generated"] == 0
    assert set(cen["sealed_ranges"]) == {"TUNE", "EVAL2"}
    assert cen["refused_calls"]  # the refusals above were recorded
    assert cen["ok"], cen["leaks"]


def test_reinstall_with_other_bounds_is_refused(C, tmp_path):
    p = tmp_path / "P.md"
    p.write_text('---\nsealed:\n  TUNE: "[10, 20)"\n---\n')
    with pytest.raises(C.ControlFailure):
        C.install_seal(p)


# --------------------------------------------------------------------------- #
# G-C1(e): the branch search
# --------------------------------------------------------------------------- #


def test_search_pattern_prefilters_every_sealed_integer(C):
    import re

    r = C.sealed_ranges()
    pat = re.compile(C.search_pattern(r))
    for lo, hi in r.values():
        for n in range(lo, hi):
            assert pat.search(str(n)) and pat.search(f"{n // 1000}_{n % 1000:03d}")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [GIT, "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_branch_search_finds_new_hits_and_ignores_allowed_paths(C, tmp_path):
    lo, _hi = C.sealed_ranges()["EVAL2"]
    t_lo, _ = C.sealed_ranges()["TUNE"]
    n, m = lo + 5, t_lo + 7
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "clean.txt").write_text(f"sha a{n}f, fraction 0.{n}, int {lo - 1}\n")
    (repo / "experiments" / "post-fails-controls").mkdir(parents=True)
    (repo / "experiments" / "post-fails-controls" / "PREREG.md").write_text(f"[{n}, x)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a")
    rep = C.branch_search(repo=repo)
    assert rep["ok"] and rep["n_new_hits"] == 0 and rep["n_hits"] == 1, rep
    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "leak.py").write_text(
        f"EVAL2 = ({n}, 0)\nTUNE = {m // 1000}_{m % 1000:03d}\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "b")
    rep = C.branch_search(repo=repo)
    assert not rep["ok"] and rep["n_refs"] == 2
    got = sorted((h["path"], h["value"], h["range"]) for h in rep["new_hits"])
    assert got == [("leak.py", n, "EVAL2"), ("leak.py", m, "TUNE")]
    assert rep["new_hits"][0]["refs"] == ["refs/heads/other"]


def test_branch_search_git_failure_is_not_a_clean_bill(C, tmp_path):
    rep = C.branch_search(repo=tmp_path)  # not a repository
    assert rep["ok"] is False


# --------------------------------------------------------------------------- #
# G-C2, pins, G-C3, G-C4
# --------------------------------------------------------------------------- #


def test_manifest_check_reads_rc_and_own_sha(C, tmp_path):
    (tmp_path / "d").mkdir()
    f = tmp_path / "d" / "x.bin"
    f.write_bytes(b"abc")
    man = tmp_path / "MANIFEST.sha256"
    man.write_text(f"{C.sha256_file(f)}  d/x.bin\n")
    own = C.sha256_file(man)
    ok = C.manifest_check(man, own, tmp_path)
    assert ok["ok"] and ok["rc"] == 0 and ok["n_ok"] == 1
    assert not C.manifest_check(man, "0" * 64, tmp_path)["ok"]
    f.write_bytes(b"abd")
    bad = C.manifest_check(man, own, tmp_path)
    assert not bad["ok"] and bad["rc"] != 0


def test_pins_recheck_detects_a_change(C, tmp_path):
    f = tmp_path / "a.json"
    f.write_text("1")
    P = C.Pins()
    P.pin(f)
    assert P.recheck()["ok"]
    f.write_text("2")
    assert not P.recheck()["ok"]
    with pytest.raises(C.ControlFailure):
        P.pin(f)
    assert not C.Pins().recheck()["ok"]  # nothing pinned is not a pass


def test_checkpoint_pin(C, tmp_path):
    f = tmp_path / "c.pt"
    f.write_bytes(b"x")
    assert C.checkpoint_pin(f, C.sha256_file(f))["ok"]
    assert not C.checkpoint_pin(f, "0" * 64)["ok"]


def test_suite_census(C):
    log = (
        "....s.\nSKIPPED [1] tests/test_x.py:3: needs a thing\n"
        "= 5 passed, 1 skipped in 2.0s =\n"
    )
    c = C.suite_census(log, 0)
    assert (c["passed"], c["skipped"], c["failed"], c["errors"]) == (5, 1, 0, 0)
    assert c["skipped_tests"] == ["SKIPPED [1] tests/test_x.py:3: needs a thing"]
    assert c["ok"]
    c = C.suite_census("1 failed, 2 passed, 1 error in 3.1s\n", 1)
    assert (c["failed"], c["passed"], c["errors"]) == (1, 2, 1) and not c["ok"]
    assert not C.suite_census("", 0)["ok"]  # no summary line is not a pass
    assert not C.suite_census("3 passed in 1s\n", None)["ok"]
    rsr = (
        "...\n---- rsr test census ----\n"
        "passed=7 failed=0 skipped=2 errors=0\n  SKIPPED[2] why\n"
    )
    c = C.suite_census(rsr, 0)
    assert (c["passed"], c["skipped"], c["failed"]) == (7, 2, 0) and c["ok"]
    assert c["skipped_tests"] == ["SKIPPED[2] why"]


def test_forbid_constants_record(C, monkeypatch):
    from rsr import constants as K

    monkeypatch.setattr(K, "record", K.record)
    monkeypatch.setattr(K.REGISTRY, "record", K.REGISTRY.record)
    rep = C.forbid_constants_record()
    assert rep["ok"]
    with pytest.raises(C.ControlFailure):
        K.record("x", 1)
    with pytest.raises(C.ControlFailure):
        K.REGISTRY.record("x", 1)


def test_no_grads_and_worst_exit(C):
    import torch

    lin = torch.nn.Linear(2, 2)
    assert C.no_grads(lin)["ok"]
    lin(torch.ones(1, 2)).sum().backward()
    assert not C.no_grads(lin)["ok"]
    assert C.worst_exit([0, 1, 0]) == 1
    assert C.worst_exit([1, 3, 0]) == 3
    assert C.worst_exit([]) == 3
    with pytest.raises(ValueError):
        C.worst_exit([5])
