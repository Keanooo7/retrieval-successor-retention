"""The canary split (2026-09-22): a bit-exact CPU variant, and `loss_path_hash`.

**Why.** At `945b501` the MPS canary read MOVED on 6 of 6 beats, while the same
script at `7254080` reproduced `runs/canary/baseline.json` exactly on the same
machine (`docs/lab-notes/for-brendan-2026-09-21.md`). The *code* moved, not the
environment, and the canary had no way to say so: it exited `1`. These tests pin:

* a changed loss path is `2` ("not comparable: code changed"), **never `1`**;
* a baseline written before the hash existed is `2` too, and is not rewritten;
* the CPU variant is bit-exact (`rel_tol = 0`), trains on one thread, and has its
  own baseline file;
* the MPS configuration is frozen exactly as it was.

No test here trains: `train()` is replaced by a stub that writes beats, and every
path is under `tmp_path`. Nothing writes `runs/canary/`.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import canary  # noqa: E402
import ledger as ledger_mod  # noqa: E402


def _stub(monkeypatch, tmp_path, readings, seen_threads=None):
    """Replace train() with a stub writing `readings` in order; point every path
    at tmp_path."""
    import torch

    import rsr.train.loop as loop

    it = iter(readings)

    def fake_train(*, out_dir, **_cfg):
        if seen_threads is not None:
            seen_threads.append(torch.get_num_threads())
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "heartbeat.jsonl").write_text(
            "".join(json.dumps({"kind": "beat", "loss": x}) + "\n" for x in next(it))
        )

    monkeypatch.setattr(loop, "train", fake_train)
    monkeypatch.setattr(canary, "_REPO", tmp_path)
    monkeypatch.setattr(canary, "BASELINE", tmp_path / "baseline.json")
    monkeypatch.setattr(canary, "BASELINE_CPU", tmp_path / "baseline-cpu.json")
    monkeypatch.setattr(
        canary, "Ledger", functools.partial(ledger_mod.Ledger, runs_root=tmp_path / "L")
    )
    monkeypatch.setattr(ledger_mod, "_dirty_source_paths", lambda: [])


# --------------------------------------------------------------------------- #
# loss_path_hash: a code change is `2`, never `1`
# --------------------------------------------------------------------------- #


def test_canary_not_comparable_when_the_loss_path_hash_differs(tmp_path, monkeypatch):
    """The 2026-09-21 case: losses that differ on every beat, because the code
    changed. Compared, that is MOVED (`1`). It must be `2`, and it must print the
    old-sha command -- the environment check a code change leaves available."""
    _stub(monkeypatch, tmp_path, [[4.0, 3.0, 2.0]])
    before = json.dumps(
        {"cycle": 4, "losses": [9.0, 9.0, 9.0], "loss_path_hash": "0" * 64}, indent=2
    )
    (tmp_path / "baseline.json").write_text(before)

    r = canary.run(16)

    assert r["verdict"] == "not_comparable"
    assert r["exit_code"] == 2
    assert "code changed" in r["detail"]
    assert "worktree add" in r["detail"] and "scripts/canary.py 16" in r["detail"]
    # The ledger row's exit code agreeing with the process is
    # test_the_canary_ledger_row_records_the_exit_code_it_returns's gate.
    doc = json.loads(Path(r["ledger"]).read_text())
    assert doc["verdict"]["outcome"] == "inconclusive"
    assert (tmp_path / "baseline.json").read_text() == before


def test_canary_not_comparable_when_the_baseline_has_no_hash(tmp_path, monkeypatch):
    """`runs/canary/baseline.json` predates the field. It cannot say which code it
    measured, so a reading against it is `2` with a message -- and the file is not
    rewritten (re-baselining is the owner's call)."""
    _stub(monkeypatch, tmp_path, [[4.0, 3.0, 2.0]])
    before = json.dumps({"cycle": 4, "losses": [4.0, 3.0, 2.0]}, indent=2) + "\n"
    (tmp_path / "baseline.json").write_text(before)

    r = canary.run(17)

    assert r["verdict"] == "not_comparable"
    assert r["exit_code"] == 2
    assert "no loss_path_hash" in r["detail"]
    assert (tmp_path / "baseline.json").read_text() == before


def test_a_first_reading_records_the_hash_and_a_matching_reading_holds(
    tmp_path, monkeypatch
):
    _stub(monkeypatch, tmp_path, [[4.0, 3.0], [4.0, 3.0], [4.0, 3.5]], [])
    first = canary.run(1)
    base = json.loads((tmp_path / "baseline.json").read_text())
    # Verdicts only: the verdict -> exit-code mapping is
    # test_a_first_canary_reading_exits_2_nothing_to_compare's gate, and asserting
    # it here too would make its battery mutations leak into this test.
    assert first["verdict"] == "baseline"
    assert base["loss_path_hash"] == first["loss_path_hash"]
    assert "scripts/canary.py" in base["loss_path_files"]
    assert canary.run(2)["verdict"] == "held"
    moved = canary.run(3)
    assert moved["verdict"] == "MOVED"


def test_the_old_sha_is_derived_from_the_baseline_cycle_ledger(tmp_path):
    """A pre-hash baseline records no sha; the cycle that wrote it does."""
    led = tmp_path / "canary" / "cycle-04" / "ledger.json"
    led.parent.mkdir(parents=True)
    led.write_text(json.dumps({"provenance": {"git_sha": "abc123def456"}}))
    ok, why = canary.comparability(
        {"cycle": 4, "losses": []}, "f" * 64, cycle=16, device="mps", runs_root=tmp_path
    )
    assert not ok
    assert "abc123def456" in why


def test_loss_path_hash_moves_with_the_code(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "rsr").mkdir(parents=True)
    (tmp_path / "scripts" / "canary.py").write_text("x = 1\n")
    (tmp_path / "src" / "rsr" / "loop.py").write_text("y = 1\n")
    files = ["scripts/canary.py", "src/rsr/loop.py"]
    h0 = canary.loss_path_hash(files, tmp_path)
    assert h0 == canary.loss_path_hash(list(reversed(files)), tmp_path)
    (tmp_path / "src" / "rsr" / "loop.py").write_text("y = 2\n")
    assert canary.loss_path_hash(files, tmp_path) != h0


def test_loss_path_files_are_what_was_actually_imported():
    import rsr.train.loop  # noqa: F401

    files = canary.loss_path_files()
    assert files == sorted(files)
    assert "scripts/canary.py" in files
    assert "src/rsr/train/loop.py" in files
    assert all(f.startswith("src/rsr/") or f == "scripts/canary.py" for f in files)


# --------------------------------------------------------------------------- #
# the CPU variant
# --------------------------------------------------------------------------- #


def test_the_cpu_canary_is_bit_exact():
    cfg, tol, path = canary.settings("cpu")
    assert tol == 0.0
    assert cfg["device"] == "cpu"
    assert path.name == "baseline-cpu.json"
    assert {k: v for k, v in cfg.items() if k != "device"} == {
        k: v for k, v in canary.CONFIG.items() if k != "device"
    }
    verdict, moved, _ = canary.compare([1.0, 0.5], [1.0, 0.5 + 2**-40], tol)
    assert verdict == "MOVED" and len(moved) == 1
    assert canary.compare([1.0, 0.5], [1.0, 0.5], tol)[0] == "held"


def test_the_cpu_canary_trains_on_one_thread(tmp_path, monkeypatch):
    import torch

    threads: list[int] = []
    _stub(monkeypatch, tmp_path, [[1.0]], threads)
    before = torch.get_num_threads()
    r = canary.run(5, device="cpu")
    assert threads == [1]
    assert torch.get_num_threads() == before
    assert (tmp_path / "baseline-cpu.json").exists()
    assert not (tmp_path / "baseline.json").exists()
    assert r["ledger"].endswith("cycle-05-cpu/ledger.json")


def test_the_mps_canary_config_is_frozen_as_it_was():
    """Unchanged from before the split, literally."""
    assert (
        dict(
            d=128,
            steps_per_stream=48,
            batch=8,
            iters=6,
            seed=0,
            memory_slots=16,
            lr=1e-3,
            device="mps",
            beat_every=1,
            ckpt_every=0,
        )
        == canary.CONFIG
    )
    assert canary.REL_TOL == 1e-4
    assert canary.settings("mps")[2] == canary.BASELINE
    assert canary.BASELINE.name == "baseline.json"


def test_an_unknown_canary_device_did_not_run():
    with pytest.raises(SystemExit) as e:
        canary.main(["3", "--device", "gpu"])
    assert e.value.code == 3
