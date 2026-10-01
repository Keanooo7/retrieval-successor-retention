"""Run-1 readiness on top of ADR-0009 slice 1 (review record
`docs/reviews/feat-rsr-slice1/7be03038a905.md`, items m-1 .. m-8 and P-2).

Rulings: `docs/owner/rulings/R-2026-09-30-rsr-run-1-scope.md` (D2(a) arm B
ckpt3000, D2(b) frozen, 31(b) one pass over the pool) and
`docs/owner/rulings/R-2026-09-30-adr0009-signoff.md` (L6: the §7.1 diagnostics in
every RESULTS; L11; L12).

🔴 **Every checkpoint here is a tiny fixture this file trains itself.** Nothing
reads `~/rsr-substrate` or the real 435-checkpoint substrate. Nothing trains for
evaluation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

import rsr.model.tg.policy_loop as pl
import rsr.retention.rsr as rsr_mod
import rsr.train.loop as loop
from rsr.metrics import vacuity
from rsr.retention.instrumentation import RankShiftLog
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import AGE_ONLY, RSRConfig, RSRPolicy, _Obs
from rsr.retention.value_head import AgeOnlyValueHead, BilinearValueHead

# train()-level shapes: the S0-01 tiny config (as tests/test_rsr_learning_path.py).
TINY = dict(
    d=32,
    steps_per_stream=48,
    batch=2,
    max_tokens=16,
    memory_slots=4,
    iters=1,
    device="cpu",
    beat_every=1,
    ckpt_every=0,
)
#: Labelled overrides (ADR-0009 L12). Not values for any run.
OV = dict(nu=0.0, beta=1.0, gamma=0.5, b_enabled=False, shadow_enabled=False)
D, M = 32, 4


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spy_build_policy(monkeypatch) -> list:
    built, real = [], loop.build_policy

    def spy(*a, **k):
        built.append(real(*a, **k))
        return built[-1]

    monkeypatch.setattr(loop, "build_policy", spy)
    return built


def _spy_models(monkeypatch) -> list:
    """Every TGModel train() builds. Same __init__, so no RNG is moved."""
    models, real = [], loop.TGModel

    def spy(cfg):
        models.append(real(cfg))
        return models[-1]

    monkeypatch.setattr(loop, "TGModel", spy)
    return models


def _spy_loop_entries(monkeypatch) -> list:
    calls, real = [], loop.run_policy_loop

    def spy(model, ids, mask, lengths, policy, **kw):
        calls.append({"ids": ids.clone(), "rng": torch.get_rng_state().clone()})
        return real(model, ids, mask, lengths, policy, **kw)

    monkeypatch.setattr(loop, "run_policy_loop", spy)
    return calls


def _beats(out_dir: Path) -> list[dict]:
    lines = (out_dir / "heartbeat.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r.get("kind") == "beat"]


# --------------------------------------------------------------------------- #
# Fixtures: tiny FIFO checkpoints and a T0-style manifest over them
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def substrate(tmp_path_factory) -> dict:
    """A miniature `~/rsr-substrate`: FIFO checkpoints at steps 2 and 3, a
    non-FIFO checkpoint, and `MANIFEST.sha256` whose paths are relative to `root`.

    Only a FIFO arm is trained. The non-FIFO source is the step-2 FIFO payload
    with its policy name rewritten to "rsr" -- all the refusal reads -- so no
    warm-start test depends on the RSR arm building (a mutation of the RSR path
    would otherwise error every test that uses this fixture)."""
    root = tmp_path_factory.mktemp("substrate")
    loop.train(
        policy_name="fifo",
        out_dir=root / "runs" / "fifo",
        **{**TINY, "iters": 3, "ckpt_every": 1},
    )
    # A tiny run may measure INERT and quarantine its checkpoints
    # (R-2026-09-22-inert-checkpoint-quarantine), which `ck.load` then refuses.
    # The fixture copies its own files out to plain paths: these are this test's
    # fixtures, not a quarantined substrate being routed around.
    src = root / "src"
    src.mkdir()

    def pick(run: str, name: str) -> Path:
        (found,) = sorted((root / "runs" / run).rglob(name))
        return Path(shutil.copy2(found, src / f"{run}-{name}"))

    fifo3 = pick("fifo", "ckpt-000003.pt")
    fifo2 = pick("fifo", "ckpt-000002.pt")
    payload = torch.load(fifo2, map_location="cpu", weights_only=False)
    payload["policy"] = {**payload["policy"], "__policy_name__": "rsr"}
    rsr1 = src / "rsr-ckpt-000002.pt"
    torch.save(payload, rsr1)
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(
        "".join(
            f"{_sha(p)}  {p.relative_to(root).as_posix()}\n" for p in (fifo2, fifo3, rsr1)
        )
    )
    return {
        "root": root,
        "manifest": manifest,
        "fifo3": fifo3,
        "fifo2": fifo2,
        "rsr1": rsr1,
    }


def _ws(sub: dict, which: str = "fifo3") -> loop.WarmStart:
    return loop.WarmStart(
        checkpoint=str(sub[which]), manifest=str(sub["manifest"]), root=str(sub["root"])
    )


#: A pool of 4 documents at batch 2: one epoch = 2 optimizer steps = T_warm (§3.4).
SMALL_POOL = loop.DocumentPool(offset=0, length=4)


# --------------------------------------------------------------------------- #
# m-1: the weights-only warm start
# --------------------------------------------------------------------------- #


def test_the_warm_start_loads_the_source_weights_bitwise(
    substrate, tmp_path, monkeypatch
):
    """m-1: after the load (iters = 0, nothing stepped), every model tensor --
    transformer and W_sent alike -- is bitwise the source checkpoint's."""
    models = _spy_models(monkeypatch)
    loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        warm_start=_ws(substrate),
        out_dir=tmp_path,
        **{**TINY, "iters": 0},
    )
    src = torch.load(substrate["fifo3"], map_location="cpu", weights_only=False)
    got = models[0].state_dict()
    assert set(got) == set(src["model"])
    assert any(k.startswith("srep_head") for k in got)  # W_sent is among them
    differ = [k for k, v in src["model"].items() if not torch.equal(got[k], v)]
    assert differ == [], differ


def test_the_warm_start_builds_phi_fresh(substrate, tmp_path, monkeypatch):
    """m-1 / L11: phi is BilinearValueHead(d) from phi_generator(seed), whatever
    the source held."""
    built = _spy_build_policy(monkeypatch)
    loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        warm_start=_ws(substrate),
        out_dir=tmp_path,
        **{**TINY, "iters": 0},
    )
    want = BilinearValueHead(TINY["d"], generator=loop.phi_generator(0)[0])
    for (n, p), (_, q) in zip(
        built[0].head.named_parameters(), want.named_parameters(), strict=True
    ):
        assert torch.equal(p.detach().cpu(), q.detach()), n


def test_the_warm_start_runs_t_warm_at_relative_steps(substrate, tmp_path, monkeypatch):
    """m-1, §3.4 / correction 31: the source is at step 3 and T_warm is 2 steps
    (pool 4 / batch 2). k counts from 0 AT THE WARM START: steps 0 and 1 are
    `fifo_warmup`, step 2 is scored. Counting from the source's step would start
    at k = 3 >= T_warm and skip the warmup silently (the review's m-1)."""
    built = _spy_build_policy(monkeypatch)
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        warm_start=_ws(substrate, "fifo3"),
        pool=SMALL_POOL,
        out_dir=tmp_path,
        **{**TINY, "iters": 3},
    )
    assert r["config"]["warm_start"]["source_step"] == 3
    assert r["config"]["epoch"]["t_warm_steps"] == 2.0
    by_k: dict[int, set] = {}
    for rec in built[0].records:
        by_k.setdefault(rec.train_step, set()).add((rec.warm, rec.attribution))
    assert sorted(by_k) == [0, 1, 2]
    assert by_k[0] == {(True, "fifo_warmup")}
    assert by_k[1] == {(True, "fifo_warmup")}
    assert by_k[2] == {(False, "psi")}


def test_the_warm_start_restores_weights_only(substrate, tmp_path, monkeypatch):
    """m-1: no RNG, no optimizer state, no step comes from the source. The
    global RNG at the loop's entry is the cold run's (same seed, same config),
    and the optimizer after one step has stepped once, not 3 + 1 times."""
    calls = _spy_loop_entries(monkeypatch)
    kw = {**TINY, "iters": 1, "ckpt_every": 1}
    loop.train(policy_name="rsr", policy_overrides=OV, out_dir=tmp_path / "cold", **kw)
    loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        warm_start=_ws(substrate),
        out_dir=tmp_path / "warm",
        **kw,
    )
    assert torch.equal(calls[0]["rng"], calls[1]["rng"])
    assert torch.equal(calls[0]["ids"], calls[1]["ids"])
    (path,) = sorted((tmp_path / "warm").rglob("ckpt-000001.pt"))
    state = torch.load(path, weights_only=False)
    steps = {float(s["step"]) for s in state["optimizer"]["state"].values()}
    assert steps == {1.0}
    assert state["step"] == 1


def test_a_wrong_sha_is_refused(substrate, tmp_path):
    """m-1: the T0 manifest is the authority. A sha256 that differs, a file the
    manifest does not list, and a file outside its root are each refused before
    any weight is read into a model."""
    root, ck3 = substrate["root"], substrate["fifo3"]
    bad = tmp_path / "BAD.sha256"
    bad.write_text(f"{'0' * 64}  {ck3.relative_to(root).as_posix()}\n")
    with pytest.raises(loop.WarmStartRefused, match="sha256"):
        loop.verify_warm_start(loop.WarmStart(str(ck3), str(bad), str(root)))
    with pytest.raises(loop.WarmStartRefused, match="sha256"):
        loop.train(
            policy_name="rsr",
            policy_overrides=OV,
            warm_start=loop.WarmStart(str(ck3), str(bad), str(root)),
            out_dir=tmp_path / "run",
            **{**TINY, "iters": 0},
        )
    empty = tmp_path / "EMPTY.sha256"
    empty.write_text("")
    with pytest.raises(loop.WarmStartRefused, match="not in the manifest"):
        loop.verify_warm_start(loop.WarmStart(str(ck3), str(empty), str(root)))
    with pytest.raises(loop.WarmStartRefused, match="not under the manifest root"):
        loop.verify_warm_start(
            loop.WarmStart(str(ck3), str(substrate["manifest"]), str(tmp_path))
        )
    ok = loop.verify_warm_start(_ws(substrate))
    assert ok["sha256"] == _sha(ck3)
    assert ok["manifest_entry"] == "src/fifo-ckpt-000003.pt"


def test_a_frozen_warm_started_arm_moves_no_transformer_weight(substrate, tmp_path):
    """m-1 + D2(b): two optimizer steps later, every model tensor is still bitwise
    the source's, phi has moved, and the frozen mode is recorded (m-4)."""
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        warm_start=_ws(substrate),
        freeze_transformer=True,
        out_dir=tmp_path,
        **{**TINY, "iters": 2, "ckpt_every": 2},
    )
    src = torch.load(substrate["fifo3"], map_location="cpu", weights_only=False)
    (path,) = sorted(tmp_path.rglob("ckpt-000002.pt"))
    out = torch.load(path, map_location="cpu", weights_only=False)
    moved = [k for k, v in src["model"].items() if not torch.equal(out["model"][k], v)]
    assert moved == [], moved
    phi0 = BilinearValueHead(TINY["d"], generator=loop.phi_generator(0)[0])
    assert not torch.equal(out["policy"]["__head__"]["W"], phi0.W.detach())
    assert r["config"]["frozen_model_mode"] == "train (dropout live)"
    assert "+frozen+warm[fifo-ckpt-000003.pt@" in r["arm_label"]


def test_a_frozen_warm_start_measures_liveness_against_the_untrained_decoy(
    substrate, tmp_path
):
    """m-1, the review's liveness note: frozen FROM INIT, the decoy IS the trained
    model (`ratio == 1.0 exactly`, invalid, exit 3 -- SLICE1-build's
    frozen_liveness.log). Frozen from a warm start, the decoy is still "the
    untrained model at the same seed and config" -- copied before the load -- so
    the measurement compares the substrate against an untrained model."""
    kw = {**TINY, "iters": 0}
    cold = loop.train(
        policy_name="fifo", freeze_transformer=True, out_dir=tmp_path / "c", **kw
    )
    warm = loop.train(
        policy_name="fifo",
        freeze_transformer=True,
        warm_start=_ws(substrate),
        out_dir=tmp_path / "w",
        **kw,
    )
    assert cold["liveness"]["band"] == "invalid"
    assert "decoy is the trained model" in cold["liveness"]["reason"]
    assert warm["liveness"]["band"] in ("live", "inert", "inconclusive")
    assert warm["liveness"]["ratio"] != 1.0


def test_the_warm_start_is_stamped_into_frozen_hash_and_label(substrate, tmp_path):
    """m-1: the source's path and sha256 are in the frozen config, so in the hash
    and the run_id; the label names the file and its sha prefix."""
    kw = {**TINY, "iters": 0}
    a = loop.train(
        policy_name="fifo",
        warm_start=_ws(substrate, "fifo3"),
        freeze_transformer=True,
        out_dir=tmp_path / "a",
        **kw,
    )
    b = loop.train(
        policy_name="fifo",
        warm_start=_ws(substrate, "fifo2"),
        freeze_transformer=True,
        out_dir=tmp_path / "b",
        **kw,
    )
    sha3 = _sha(substrate["fifo3"])
    ws = a["config"]["warm_start"]
    assert ws["sha256"] == sha3
    assert ws["checkpoint"] == str(substrate["fifo3"].resolve())
    assert ws["manifest_entry"] == "src/fifo-ckpt-000003.pt"
    assert ws["source_policy"] == "fifo" and ws["source_step"] == 3
    # MAJOR-1 / the seed ruling: the source's seed and vocabulary are recorded.
    assert ws["source_seed"] == 0 and ws["seed_matches"] is True
    assert ws["vocab_checked"] is True and ws["source_config_origin"] == "checkpoint meta"
    assert ws["source_vocab_sha256"] == ws["run_vocab_sha256"]
    assert a["arm_label"] == (
        f"fifo+frozen+warm[fifo-ckpt-000003.pt@{sha3[:12]},src_seed=0]"
    )
    assert a["config_hash"] != b["config_hash"] and a["run_id"] != b["run_id"]


def test_a_non_fifo_source_is_refused(substrate, tmp_path):
    """m-1: run 1's substrate is a FIFO checkpoint; an RSR checkpoint's phi would
    be silently dropped by a weights-only load, so it is refused."""
    with pytest.raises(loop.WarmStartRefused, match="policy 'rsr'"):
        loop.train(
            policy_name="rsr",
            policy_overrides=OV,
            warm_start=_ws(substrate, "rsr1"),
            out_dir=tmp_path,
            **{**TINY, "iters": 0},
        )


def _fifo_source(tmp_path: Path, name: str, **kw) -> Path:
    """A tiny FIFO checkpoint at ``tmp_path/<name>.pt`` (copied out of its run
    directory, so a quarantine of the tiny run cannot move it)."""
    run = tmp_path / f"run-{name}"
    loop.train(policy_name="fifo", out_dir=run, **{**TINY, "ckpt_every": 1, **kw})
    (found,) = sorted(run.rglob(f"ckpt-{kw.get('iters', TINY['iters']):06d}.pt"))
    return Path(shutil.copy2(found, tmp_path / f"{name}.pt"))


def _manifest(root: Path, *files: Path) -> Path:
    man = root / "MANIFEST.sha256"
    man.write_text(
        "".join(f"{_sha(f)}  {f.relative_to(root).as_posix()}\n" for f in files)
    )
    return man


def _legacy(src: Path, dst: Path) -> tuple[Path, dict]:
    """``src`` rewritten as a pre-run1-ready checkpoint: ``meta`` keeps only the
    run_id (as arm B's does). Returns the path and the stripped config."""
    payload = torch.load(src, map_location="cpu", weights_only=False)
    cfg = payload["meta"].pop("config")
    payload["meta"].pop("vocab_sha256")
    torch.save(payload, dst)
    return dst, cfg


def _ws_for(ck: Path, root: Path, **kw) -> loop.WarmStart:
    return loop.WarmStart(str(ck), str(_manifest(root, ck)), str(root), **kw)


def test_a_source_with_a_different_vocabulary_map_is_refused(tmp_path):
    """MAJOR-1: the token -> id MAP is compared, not a document count. A source
    whose vocabulary came from 2 documents (133 words at S = 48, seed 0) is not
    this run's 172-word map: refused before any weight is copied."""
    ck2 = _fifo_source(tmp_path, "two-docs", n_documents=2)
    with pytest.raises(loop.WarmStartRefused, match="token->id map"):
        loop.train(
            policy_name="fifo",
            warm_start=_ws_for(ck2, tmp_path),
            out_dir=tmp_path / "run",
            **{**TINY, "iters": 0},
        )


def test_an_identical_map_from_another_document_count_is_accepted(tmp_path, monkeypatch):
    """MAJOR-1: 32 vocabulary documents give the SAME 172-word map as 64 at
    S = 48, seed 0 (the reviewer's measurement). The old document-count guard
    refused this; the map check accepts it, and the weights load bitwise."""
    ck32 = _fifo_source(tmp_path, "thirty-two", n_documents=32)
    models = _spy_models(monkeypatch)
    r = loop.train(
        policy_name="fifo",
        warm_start=_ws_for(ck32, tmp_path),
        freeze_transformer=True,
        out_dir=tmp_path / "run",
        **{**TINY, "iters": 0},
    )
    ws = r["config"]["warm_start"]
    assert ws["vocab_checked"] is True
    assert ws["source_vocab_sha256"] == ws["run_vocab_sha256"]
    src = torch.load(ck32, map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(models[0].state_dict()[k], v) for k, v in src.items())


def test_a_legacy_source_is_checked_through_its_hash_bound_config(substrate, tmp_path):
    """MAJOR-1 for checkpoints written before the vocabulary was recorded (arm B's
    kind): no config in the file -> refused; its frozen config supplied as a
    JSON file whose hash IS the run_id's -> the map is rebuilt and checked; a
    config that does not hash to the run_id -> refused as not the source's."""
    ck, cfg = _legacy(substrate["fifo3"], tmp_path / "legacy.pt")
    with pytest.raises(loop.WarmStartRefused, match="records no config"):
        loop.train(
            policy_name="fifo",
            warm_start=_ws_for(ck, tmp_path),
            out_dir=tmp_path / "r0",
            **{**TINY, "iters": 0},
        )
    good = tmp_path / "result.json"
    good.write_text(json.dumps({"config": cfg}))
    r = loop.train(
        policy_name="fifo",
        warm_start=_ws_for(ck, tmp_path, source_config=str(good)),
        out_dir=tmp_path / "r1",
        **{**TINY, "iters": 0},
    )
    ws = r["config"]["warm_start"]
    assert ws["vocab_checked"] is True and ws["source_seed"] == 0
    assert ws["source_config_origin"].startswith(str(good.resolve()))
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps({"config": {**cfg, "lr": 0.5}}))
    with pytest.raises(loop.WarmStartRefused, match="not the run that wrote"):
        loop.train(
            policy_name="fifo",
            warm_start=_ws_for(ck, tmp_path, source_config=str(bad)),
            out_dir=tmp_path / "r2",
            **{**TINY, "iters": 0},
        )


def test_an_unrecorded_vocabulary_needs_the_labelled_flag(substrate, tmp_path):
    """MAJOR-1: an unrecorded source is accepted ONLY with the labelled flag,
    which enters the frozen config, the hash and the label. Its seed is then
    unrecorded too, so the seed flag is needed as well (seed ruling)."""
    ck, _ = _legacy(substrate["fifo3"], tmp_path / "legacy.pt")
    kw = {**TINY, "iters": 0}
    with pytest.raises(loop.WarmStartRefused, match="seed"):
        loop.train(
            policy_name="fifo",
            warm_start=_ws_for(ck, tmp_path, allow_unrecorded_vocabulary=True),
            out_dir=tmp_path / "r0",
            **kw,
        )
    r = loop.train(
        policy_name="fifo",
        warm_start=_ws_for(
            ck, tmp_path, allow_unrecorded_vocabulary=True, allow_seed_mismatch=True
        ),
        out_dir=tmp_path / "r1",
        **kw,
    )
    ws = r["config"]["warm_start"]
    assert ws["vocab_checked"] is False and ws["allow_unrecorded_vocabulary"] is True
    assert ws["source_seed"] is None and ws["allow_seed_mismatch"] is True
    assert r["arm_label"].endswith(
        ",src_seed=unrecorded,vocab=unchecked,seed_mismatch=allowed]"
    )


def test_a_seed_mismatch_is_refused_unless_labelled(substrate, tmp_path):
    """Seed ruling: the source's seed (0) is recorded; a seed-1 run is refused,
    and accepted only with allow_seed_mismatch, which is in the hash and label.
    (At S = 48 the seed-1 map is the seed-0 map, so only the seed check bites.)"""
    kw = {**TINY, "iters": 0, "seed": 1}
    with pytest.raises(loop.WarmStartRefused, match="seed is 0 and this run's is 1"):
        loop.train(
            policy_name="fifo",
            warm_start=_ws(substrate),
            out_dir=tmp_path / "r0",
            **kw,
        )
    flagged = dataclasses.replace(_ws(substrate), allow_seed_mismatch=True)
    r = loop.train(policy_name="fifo", warm_start=flagged, out_dir=tmp_path / "r1", **kw)
    ws = r["config"]["warm_start"]
    assert ws["source_seed"] == 0 and ws["seed_matches"] is False
    assert r["arm_label"].endswith(",src_seed=0,seed_mismatch=allowed]")


def test_the_bytes_loaded_are_the_bytes_hashed(substrate, tmp_path, monkeypatch):
    """MINOR-5: read once, hash, load from memory. The file is swapped for other
    (valid, differently-weighted) checkpoint bytes right after the hash: the
    model must hold the HASHED bytes' weights, not the file's new contents."""
    ck = Path(shutil.copy2(substrate["fifo3"], tmp_path / "swap.pt"))
    ws = _ws_for(ck, tmp_path)
    real = loop._verified_bytes

    def swap_after_hash(w):
        out = real(w)
        shutil.copy2(substrate["fifo2"], ck)  # the path now holds other bytes
        return out

    monkeypatch.setattr(loop, "_verified_bytes", swap_after_hash)
    models = _spy_models(monkeypatch)
    loop.train(
        policy_name="fifo", warm_start=ws, out_dir=tmp_path / "r", **{**TINY, "iters": 0}
    )
    hashed = torch.load(substrate["fifo3"], map_location="cpu", weights_only=False)[
        "model"
    ]
    other = torch.load(substrate["fifo2"], map_location="cpu", weights_only=False)[
        "model"
    ]
    assert any(not torch.equal(hashed[k], other[k]) for k in hashed)  # non-vacuous
    got = models[0].state_dict()
    assert all(torch.equal(got[k], v) for k, v in hashed.items())


def test_an_unexpected_sha256_is_refused(substrate):
    """MINOR-7's CLI requires the expected sha256; a mismatch is refused even
    when the manifest agrees with the file."""
    bad = dataclasses.replace(_ws(substrate), expected_sha256="f" * 64)
    with pytest.raises(loop.WarmStartRefused, match="caller expected"):
        loop.verify_warm_start(bad)
    good = dataclasses.replace(_ws(substrate), expected_sha256=_sha(substrate["fifo3"]))
    assert loop.verify_warm_start(good)["expected_sha256"] == _sha(substrate["fifo3"])


# --------------------------------------------------------------------------- #
# m-2: the document pool
# --------------------------------------------------------------------------- #


def test_two_pools_give_two_hashes_and_two_batches(tmp_path, monkeypatch):
    """m-2: a pool is an offset and a length (or `from_range`). Two pools: two
    config hashes, two first batches. The same pool twice: one of each."""
    calls = _spy_loop_entries(monkeypatch)
    pa = loop.DocumentPool(offset=100, length=4)
    pb = loop.DocumentPool.from_range(200, 204)
    runs = [
        loop.train(
            policy_name="rsr",
            policy_overrides=OV,
            pool=p,
            out_dir=tmp_path / name,
            **TINY,
        )
        for name, p in (("a", pa), ("b", pb), ("a2", pa))
    ]
    a, b, a2 = runs
    assert a["config"]["pool"] == {
        "offset": 100,
        "length": 4,
        "vocab_documents": 64,
        "ids": [100, 104],
    }
    assert a["config"]["epoch"]["pool_documents"] == 4
    assert a["config"]["epoch"]["steps_per_epoch"] == 2.0
    assert a["config_hash"] != b["config_hash"]
    assert not torch.equal(calls[0]["ids"], calls[1]["ids"])
    assert a["config_hash"] == a2["config_hash"]
    assert torch.equal(calls[0]["ids"], calls[2]["ids"])


def test_a_pool_excludes_n_documents_and_the_stream(tmp_path):
    with pytest.raises(ValueError, match="pool replaces"):
        loop.train(pool=SMALL_POOL, n_documents=8, out_dir=tmp_path, **TINY)
    with pytest.raises(ValueError, match="pool replaces"):
        loop.train(
            pool=SMALL_POOL,
            stream=loop.DocumentStream(offset=4160, stride=2),
            out_dir=tmp_path,
            **TINY,
        )


def test_the_liveness_batch_of_a_pool_run_is_the_pool():
    """m-2: liveness is measured on the run's own training corpus -- the pool's
    first documents, in the vocabulary of [0, vocab_documents)."""
    cfg = {
        "seed": 0,
        "steps_per_stream": 48,
        "tg": {"max_sentence_tokens": 16},
        "pool": {"offset": 300, "length": 4, "vocab_documents": 64},
    }
    ids, _ = loop.liveness_batch(cfg)
    sc = loop.SyntheticConfig(sentences_per_document=48, seed=0)
    vocab = loop.build_vocab(
        loop.generate(loop.SyntheticConfig(sentences_per_document=48, seed=0))
    )
    want, _ = loop.encode(
        loop.pool_documents(loop.DocumentPool(300, 4), sc), vocab, max_tokens=16, steps=48
    )
    assert ids.shape[0] == 4 and torch.equal(ids, want)


# --------------------------------------------------------------------------- #
# m-3, m-4, m-5, P-2
# --------------------------------------------------------------------------- #


def test_the_epoch_records_sampling_with_replacement(tmp_path):
    """m-3: the count follows 31(b); the record says the draws repeat."""
    r = loop.train(
        policy_name="rsr", policy_overrides=OV, out_dir=tmp_path, **{**TINY, "iters": 0}
    )
    assert r["config"]["epoch"]["sampling"] == (
        "with replacement (torch.randint over the pool)"
    )


def test_frozen_mode_records_that_dropout_stays_on(tmp_path):
    """m-4: freezing removes the update only; the record names the mode."""
    kw = {**TINY, "iters": 0}
    f = loop.train(
        policy_name="fifo", freeze_transformer=True, out_dir=tmp_path / "f", **kw
    )
    n = loop.train(policy_name="rsr", policy_overrides=OV, out_dir=tmp_path / "n", **kw)
    assert f["config"]["frozen_model_mode"] == "train (dropout live)"
    assert "frozen_model_mode" not in n["config"]


def test_no_warmup_ablation_on_the_fifo_arm_is_refused(tmp_path):
    """m-5: a switch that is set and does nothing is refused, not ignored."""
    with pytest.raises(ValueError, match="FIFO has no warmup"):
        loop.build_policy("fifo", d_model=8, steps_per_epoch=4.0, no_warmup_ablation=True)
    with pytest.raises(ValueError, match="FIFO has no warmup"):
        loop.train(policy_name="fifo", no_warmup_ablation=True, out_dir=tmp_path, **TINY)


def test_the_adamw_eps_is_pinned_at_1e_8_and_recorded(tmp_path):
    """P-2: eps is pinned to B1's measured regime, not inherited; the optimizer
    train() builds uses it; torch's own default is recorded beside it."""
    assert loop.ADAMW_EPS == 1e-8
    rec = loop.optimizer_record()
    assert rec["eps"] == 1e-8
    assert rec["torch_default_eps"] == loop.TORCH_DEFAULT_ADAMW_EPS
    assert "pinned" in rec["eps_source"]
    loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        out_dir=tmp_path,
        **{**TINY, "ckpt_every": 1},
    )
    (path,) = sorted(tmp_path.rglob("ckpt-000001.pt"))
    state = torch.load(path, weights_only=False)
    assert {g["eps"] for g in state["optimizer"]["param_groups"]} == {1e-8}


# --------------------------------------------------------------------------- #
# m-7: CUDA synchronisation (code path only; no CUDA here)
# --------------------------------------------------------------------------- #


def test_synchronize_waits_on_cuda_and_mps(monkeypatch):
    """m-7: the dispatch, checked without a CUDA device -- a `torch.device`
    object needs none -- with both synchronisers replaced by recorders."""
    seen: list = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda d=None: seen.append(("cuda", d))
    )
    monkeypatch.setattr(torch.mps, "synchronize", lambda: seen.append(("mps", None)))
    pl.synchronize(torch.device("cuda", 0))
    pl.synchronize(torch.device("mps"))
    pl.synchronize(torch.device("cpu"))
    assert seen == [("cuda", torch.device("cuda", 0)), ("mps", None)]


def test_eval_forward_synchronises_through_the_dispatch(monkeypatch):
    seen: list = []
    monkeypatch.setattr(pl, "synchronize", lambda d: seen.append(d.type))
    cfg = loop.TGConfig(
        D=D,
        H=2,
        V=64,
        max_sentence_tokens=8,
        max_sentences_in_short_term=M,
        pad_id=60,
        bos_id=61,
        eos_id=62,
        eod_id=63,
    )
    torch.manual_seed(0)
    model = loop.TGModel(cfg)
    mem = pl.init_memory(2, cfg, device="cpu", dtype=torch.float32)
    ids = torch.randint(0, 59, (2, cfg.L))
    mask = torch.ones(2, cfg.L, dtype=torch.long)
    bos = torch.zeros(2, cfg.D)
    pl.eval_forward(
        model, ids, mask, mem, bos, torch.zeros(2, dtype=torch.bool), capture=False
    )
    assert seen == ["cpu"]


# --------------------------------------------------------------------------- #
# m-8: the §7.1 diagnostics -- hand-computable values
# --------------------------------------------------------------------------- #


def test_rankdata_averages_ties():
    got = vacuity.rankdata(torch.tensor([10.0, 20.0, 20.0, 30.0]))
    assert got.tolist() == [1.0, 2.5, 2.5, 4.0]
    assert vacuity.rankdata(torch.tensor([3, 1, 2])).tolist() == [3.0, 1.0, 2.0]


def test_partial_spearman_matches_the_first_order_formula():
    """Three permutations of 1..5, so ranks are the values and Spearman is
    1 - 6 sum(d^2) / (n (n^2 - 1)) = 1 - 6 sum(d^2) / 120:
    r_xy: d = (-1, 1, -1, 1, 0), sum 4 -> 0.8;   r_xz: d = (0, -1, 1, -1, 1) -> 0.8;
    r_yz: d = (1, -2, 2, -2, 1), sum 14 -> 0.3.
    rho_xy.z = (0.8 - 0.8 * 0.3) / sqrt((1 - 0.64)(1 - 0.09)) = 0.56 / sqrt(0.3276)."""
    x = torch.tensor([1.0, 2, 3, 4, 5])
    y = torch.tensor([2.0, 1, 4, 3, 5])
    z = torch.tensor([1.0, 3, 2, 5, 4])
    assert vacuity.spearman(x, y) == pytest.approx(0.8, abs=1e-12)
    assert vacuity.spearman(y, z) == pytest.approx(0.3, abs=1e-12)
    want = 0.56 / (0.36 * 0.91) ** 0.5
    assert vacuity.partial_spearman(x, y, z) == pytest.approx(want, abs=1e-12)
    assert vacuity.partial_correlation_with_age(x, y, z.unsqueeze(1)) == pytest.approx(
        want, abs=1e-12
    )


def test_partial_rho_is_one_for_a_score_that_is_age_and_none_when_undefined():
    """A score that IS age has partial rho 1 given any content that does not
    determine age; content that determines age leaves nothing to correlate
    (None); too few samples for the covariates is None, not a number."""
    age = torch.tensor([4.0, 3, 2, 1, 4, 2, 3, 1])
    content = torch.tensor([[0.5], [-1.0], [2.0], [0.0], [1.5], [-0.5], [1.0], [3.0]])
    assert vacuity.partial_correlation_with_age(age, age, content) == pytest.approx(1.0)
    assert vacuity.partial_correlation_with_age(age * 2, age, age.unsqueeze(1)) is None
    assert vacuity.partial_correlation_with_age(age[:3], age[:3], content[:3]) is None


def _age_only_policy(**kw) -> RSRPolicy:
    cfg = RSRConfig(
        nu=0.0,
        beta=1.0,
        gamma=kw.pop("gamma", 0.5),
        t_warm=0.0,
        a_max=M,
        psi_override=AGE_ONLY,
        b_enabled=False,
        shadow_enabled=False,
    )
    return RSRPolicy(cfg, d_model=D, **kw)


def _full_memory(step: int = 4, gestalts=None) -> MemoryState:
    """Four live slots written at 0..3, read at `step`: ages (4, 3, 2, 1) and
    ranks (1, 2, 3, 4) at step 4."""
    return MemoryState(
        gestalts=torch.zeros(M, D) if gestalts is None else gestalts,
        written_at=torch.tensor([0, 1, 2, 3]),
        live=torch.ones(M, dtype=torch.bool),
        step=step,
    )


def test_victim_age_and_rank_histograms_by_hand():
    """§7.1 / ADR-0006: two evictions chosen by a hand-set age table.
    Ages (4, 3, 2, 1). theta = (5, 5, 0, 5, 5): psi = (5, 5, 0, 5) -> slot 2,
    age 2, rank 3, displacement 2. theta = (5, 5, 5, 5, 0): psi = (0, 5, 5, 5)
    -> slot 0, age 4, rank 1, displacement 0."""
    pol = _age_only_policy(diagnostics=True)
    pol.reset()
    with torch.no_grad():
        pol.head.table.copy_(torch.tensor([5.0, 5, 0, 5, 5]))
    assert pol.select_eviction(_full_memory(), torch.zeros(D), 4) == 2
    with torch.no_grad():
        pol.head.table.copy_(torch.tensor([5.0, 5, 5, 5, 0]))
    assert pol.select_eviction(_full_memory(), torch.zeros(D), 4) == 0
    r0, r1 = pol.records
    assert (r0.victim_age, r0.victim_rank, r0.rank_shift) == (2, 3, 2)
    assert (r1.victim_age, r1.victim_rank, r1.rank_shift) == (4, 1, 0)
    want = {
        "n_evictions": 2,
        "shift_distribution": {0: 1, 2: 1},
        "victim_rank_distribution": {1: 1, 3: 1},
        "victim_age_distribution": {2: 1, 4: 1},
    }
    for summ in (pol.rank_shifts.summary(), pol.step_diagnostics()):
        assert {k: summ[k] for k in want} == want
        assert summ["fraction_shifting"] == 0.5 and summ["mean_shift"] == 1.0
    diag = pol.step_diagnostics()
    # 2 decisions x 4 live slots; D = 32 zero content columns + intercept is
    # more regressors than samples: undefined, so None -- never a number.
    assert diag["n_score_samples"] == 8
    assert diag["partial_rho_score_age_given_content"] is None


def test_the_rank_shift_log_counts_victim_age():
    log = RankShiftLog()
    assert log.summary()["victim_age_distribution"] == {}


def test_step_diagnostics_cover_one_stream_only():
    """`reset()` (one train() step) starts the window: a second stream's
    diagnostics do not include the first's evictions."""
    pol = _age_only_policy(diagnostics=True)
    pol.reset()
    pol.select_eviction(_full_memory(), torch.zeros(D), 4)
    pol.reset()
    pol.select_eviction(_full_memory(), torch.zeros(D), 4)
    pol.select_eviction(_full_memory(), torch.zeros(D), 4)
    assert pol.step_diagnostics()["n_evictions"] == 2
    assert pol.rank_shifts.n == 3


def test_the_section_7_1_diagnostics_are_logged_every_step(tmp_path):
    """m-8 through train(). TINY: 2 rows, S = 48, M = 4, every sentence writes,
    so memory is full from t = 4: 44 evictions per row, 88 per step. In warmup
    (T_warm = 32 steps, pool 64 / batch 2) every victim is the oldest: written
    at t - 4, so age 4, rank 1, displacement 0. Each beat counts ITS step only."""
    r = loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        out_dir=tmp_path,
        **{**TINY, "iters": 2},
    )
    beats = _beats(tmp_path)
    assert len(beats) == 2
    for b in beats:
        d = b["diag"]
        assert d["n_evictions"] == 88
        assert d["shift_distribution"] == {"0": 88}
        assert d["victim_rank_distribution"] == {"1": 88}
        assert d["victim_age_distribution"] == {"4": 88}
        assert d["warm"] == [True] and d["diagnostics"] is True
        # 88 decisions x 4 live slots = 352 samples > 32 content columns + 2.
        assert d["n_score_samples"] == 352
        assert -1.0 <= d["partial_rho_score_age_given_content"] <= 1.0
        assert -1.0 <= d["spearman_rho_score_age"] <= 1.0
    assert r["diagnostics"]["victim_age_distribution"] == {4: 176}
    assert r["diagnostics"]["shift_distribution"] == {0: 176}


def test_a_directly_built_policy_collects_no_diagnostics():
    """E0b's path builds RSRPolicy directly: diagnostics are off by default and
    nothing is collected (they are on only through train()'s build_policy)."""
    red = RSRPolicy(RSRConfig.reduction_to_tg(M), d_model=D)
    assert red.diagnostics is False
    red.select_eviction(_full_memory(), torch.zeros(D), 4)
    assert red._diag == []
    assert red.step_diagnostics()["partial_rho_score_age_given_content"] is None
    assert (
        loop.build_policy("rsr", d_model=8, steps_per_epoch=4.0, overrides=OV).diagnostics
        is True
    )


# --------------------------------------------------------------------------- #
# m-8: the age-only head arm (spec §7.1 item 2) -- and age never enters psi_hat
# --------------------------------------------------------------------------- #


def test_the_age_only_arm_regresses_and_evicts_on_age_alone():
    """L_MC by hand. gamma = 0, so G = r. One (row, t) = (0, 2) with slots
    written at 0 and 1: ages 2 and 1, r = (0.5, 0.25). At theta = 0:
    L_MC = (0 - 0.5)^2 + (0 - 0.25)^2 = 0.3125 (one group). With
    theta[2] = 0.5, theta[1] = 0.25: L_MC = 0. The gestalts and the context are
    garbage on purpose: the arm must not read them."""
    pol = _age_only_policy(gamma=0.0)
    obs = _Obs(
        step=2,
        keys=torch.tensor([0, 1]),
        gestalts=torch.randn(2, D),
        context=torch.randn(D),
        r=torch.tensor([0.5, 0.25]),
    )
    pol._mc = {0: [obs]}
    assert float(pol.retention_loss()) == pytest.approx(0.3125)
    with torch.no_grad():
        pol.head.table.copy_(torch.tensor([0.0, 0.25, 0.5, 0.0, 0.0]))
    assert float(pol.retention_loss()) == pytest.approx(0.0)
    # Eviction: argmin theta[age]. Ages (4, 3, 2, 1) and theta = (9, 9, 9, 1, 2)
    # give psi = (2, 1, 9, 9): slot 1, the age-3 slot.
    with torch.no_grad():
        pol.head.table.copy_(torch.tensor([9.0, 9, 9, 1, 2]))
    assert pol.select_eviction(_full_memory(), torch.zeros(D), 4) == 1  # age 3
    assert pol.records[-1].attribution == "age_only"


def test_the_content_head_never_reads_age():
    """CLAUDE.md: age is excluded from psi_hat. The content arm's score is
    unchanged when only the ages change (same gestalts, same context); the
    age-only arm's is unchanged when only the gestalts and context change."""
    g = torch.Generator().manual_seed(3)
    gestalts = torch.randn(M, D, generator=g)
    ctx = torch.randn(D, generator=g)
    content = RSRPolicy(
        RSRConfig(
            nu=0.0,
            beta=1.0,
            gamma=0.5,
            t_warm=0.0,
            a_max=M,
            psi_override=None,
            b_enabled=False,
            shadow_enabled=False,
        ),
        d_model=D,
        generator=torch.Generator().manual_seed(7),
    )
    young = _full_memory(step=4, gestalts=gestalts)
    old = MemoryState(
        gestalts=gestalts,
        written_at=torch.tensor([3, 0, 2, 1]),
        live=torch.ones(M, dtype=torch.bool),
        step=40,
    )
    assert not torch.equal(young.ages(), old.ages())
    with torch.no_grad():
        assert torch.equal(content._psi(young, ctx), content._psi(old, ctx))
    ages_only = _age_only_policy()
    with torch.no_grad():
        ages_only.head.table.copy_(torch.arange(M + 1, dtype=torch.float32))
        a = ages_only._psi(_full_memory(gestalts=gestalts), ctx)
        b = ages_only._psi(_full_memory(gestalts=-gestalts), -ctx)
    assert torch.equal(a, b) and a.tolist() == [4.0, 3.0, 2.0, 1.0]


def test_the_age_only_arm_through_train_is_labelled_and_learns(tmp_path, monkeypatch):
    """§7.1 item 2 as a separate labelled arm (L12): `psi=age_only` in the label
    and the hash, its own value_head group, L_MC computed, the table moved."""
    built = _spy_build_policy(monkeypatch)
    r = loop.train(
        policy_name="rsr",
        policy_overrides={**OV, "psi_override": AGE_ONLY},
        out_dir=tmp_path / "age",
        **TINY,
    )
    c = loop.train(
        policy_name="rsr", policy_overrides=OV, out_dir=tmp_path / "content", **TINY
    )
    assert r["arm_label"] == "rsr[b=off,beta=1,gamma=0.5,nu=0,psi=age_only,shadow=off]"
    assert r["config_hash"] != c["config_hash"]
    assert isinstance(built[0].head, AgeOnlyValueHead)
    assert r["final"]["loss_mc"] is not None
    assert float(built[0].head.table.detach().abs().sum()) > 0.0


def test_the_age_only_head_draws_no_rng():
    before = torch.get_rng_state()
    AgeOnlyValueHead(16)
    _age_only_policy()
    assert torch.equal(before, torch.get_rng_state())


def test_the_reduction_still_pins_neg_age():
    """No RSRConfig field was added: the age-only arm is a psi_override VALUE,
    and §3.7's switch for that field is still neg_age (E0b's path)."""
    assert rsr_mod.REDUCTION_SWITCHES["psi_override"] == "neg_age"
    arm = dataclasses.replace(RSRConfig.reduction_to_tg(M), psi_override=AGE_ONLY)
    assert not arm.is_reduction(M)


# --------------------------------------------------------------------------- #
# Review fixes: MINOR-1 (age-only refuses nu), MINOR-3 (every step written),
# MINOR-7 (CLI flags)
# --------------------------------------------------------------------------- #


def test_the_age_only_arm_refuses_nu():
    """MINOR-1: -nu * max cos(s_i, s_j) reads the gestalts, so an age-only arm
    with nu != 0 is not age-only. Refused by the policy and through build_policy."""
    cfg = RSRConfig(
        nu=0.5,
        beta=1.0,
        gamma=0.5,
        t_warm=0.0,
        a_max=M,
        psi_override=AGE_ONLY,
        b_enabled=False,
        shadow_enabled=False,
    )
    with pytest.raises(ValueError, match=r"requires nu=0\.0"):
        RSRPolicy(cfg, d_model=D)
    with pytest.raises(ValueError, match=r"requires nu=0\.0"):
        loop.build_policy(
            "rsr",
            d_model=D,
            steps_per_epoch=4.0,
            overrides={**OV, "nu": 0.5, "psi_override": AGE_ONLY},
        )
    assert _age_only_policy().config.nu == 0.0  # nu = 0 still builds


def test_the_diagnostics_are_written_every_step_whatever_beat_every(tmp_path):
    """MINOR-3: computed every step, so written every step -- diagnostics.jsonl
    has steps 0, 1, 2 at beat_every = 2, and each beat's `diag` is its step's."""
    loop.train(
        policy_name="rsr",
        policy_overrides=OV,
        out_dir=tmp_path,
        **{**TINY, "iters": 3, "beat_every": 2},
    )
    lines = [
        json.loads(x) for x in (tmp_path / loop.DIAGNOSTICS_FILE).read_text().splitlines()
    ]
    assert [x["step"] for x in lines] == [0, 1, 2]
    assert all(x["diag"]["n_evictions"] == 88 for x in lines)
    assert all(
        x["diag"]["partial_rho_score_age_given_content"] is not None for x in lines
    )
    beats = {b["step"]: b["diag"] for b in _beats(tmp_path)}
    assert sorted(beats) == [0, 2]
    for step, d in beats.items():
        assert json.loads(json.dumps(lines[step]["diag"])) == d


def test_the_cli_routes_the_pool_and_the_warm_start(monkeypatch, tmp_path):
    """MINOR-7: one reproducible command. The flags become the same DocumentPool
    and WarmStart objects train() validates; partial flag sets are usage errors
    (exit 3), never silently dropped."""
    seen: dict = {}
    monkeypatch.setattr(
        "rsr.train.loop.train",
        lambda **kw: seen.update(kw) or {"run_id": "x", "liveness": {"band": "live"}},
    )
    ws_argv = [
        "--warm-start", "/x/ckpt-003000.pt",
        "--warm-start-sha256", "ab" * 32,
        "--warm-start-manifest", "/x/MANIFEST.sha256",
        "--warm-start-root", "/x",
        "--warm-start-source-config", "/x/result.json",
        "--allow-seed-mismatch",
    ]  # fmt: skip
    argv = ["--out-dir", str(tmp_path), "--pool-offset", "100", "--pool-length", "256"]
    assert loop.main(argv + ws_argv) == 0
    assert seen["pool"] == loop.DocumentPool(100, 256, 64)
    assert seen["warm_start"] == loop.WarmStart(
        checkpoint="/x/ckpt-003000.pt",
        manifest="/x/MANIFEST.sha256",
        root="/x",
        expected_sha256="ab" * 32,
        source_config="/x/result.json",
        allow_unrecorded_vocabulary=False,
        allow_seed_mismatch=True,
    )
    seen.clear()
    assert loop.main(["--out-dir", str(tmp_path)]) == 0
    assert seen["pool"] is None and seen["warm_start"] is None
    for bad in (
        ["--pool-offset", "100"],
        ["--pool-offset", "100", "--pool-length", "0"],
        [
            "--warm-start",
            "/x/c.pt",
            "--warm-start-manifest",
            "/m",
            "--warm-start-root",
            "/",
        ],
        ["--allow-seed-mismatch"],
    ):
        with pytest.raises(SystemExit) as e:
            loop.main(["--out-dir", str(tmp_path), *bad])
        assert e.value.code == 3, bad


def test_the_cli_warm_start_reaches_the_real_verification(substrate, tmp_path):
    """MINOR-7, routed for real (no stub): a wrong --warm-start-sha256 is refused
    by verify_warm_start inside train()."""
    argv = [
        "--out-dir", str(tmp_path), "--iters", "0", "--d", "32", "--batch", "2",
        "--memory-slots", "4", "--device", "cpu", "--ckpt-every", "0",
        "--warm-start", str(substrate["fifo3"]),
        "--warm-start-sha256", "0" * 64,
        "--warm-start-manifest", str(substrate["manifest"]),
        "--warm-start-root", str(substrate["root"]),
    ]  # fmt: skip
    with pytest.raises(loop.WarmStartRefused, match="caller expected"):
        loop.main(argv)
