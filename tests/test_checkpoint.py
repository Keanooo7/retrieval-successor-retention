"""Checkpoint and resume (gauntlet 3.6, 3.7).

Three properties, each of which fails silently without a test:

* **3.6a** every stateful object round-trips, **policy state included**
* **3.6b** the write survives `SIGKILL` mid-save
* **3.7** a mid-stream resume reproduces the uninterrupted run **exactly**

"Exactly" is the word that does the work. A resume that restores parameters but
not the RNG produces a run that is *plausible* and not the same, and that is the
worst kind of failure because nothing looks wrong -- the loss curve diverges
slowly and reads as noise.
"""

from __future__ import annotations

import io
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from torch import nn

from rsr.baselines.lru import LRUPolicy
from rsr.model.tg import TGConfig, TGModel
from rsr.model.tg.policy_loop import run_policy_loop
from rsr.retention.policy import AttentionTrace, MemoryState
from rsr.retention.rsr import RSRConfig, RSRPolicy
from rsr.train import checkpoint as ckpt


def _tiny_cfg(**kw) -> TGConfig:
    base = dict(
        D=32,
        H=1,
        N=2,
        V=64,
        max_sentence_tokens=6,
        max_sentences_in_short_term=4,
        block_config=("S", "C"),
        pad_id=60,
        bos_id=61,
        eos_id=62,
        eod_id=63,
    )
    base.update(kw)
    return TGConfig(**base)


def _batch(cfg: TGConfig, batch=2, s=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 60, (batch, s, cfg.L), generator=g)
    ids[:, :, 0] = cfg.bos_id
    ids[:, :, -1] = cfg.eos_id
    return ids, torch.ones(batch, s, cfg.L, dtype=torch.long), torch.full((batch,), s)


def _loss_fn(t, out, ids, mask, row_valid):
    return out.logits.float().pow(2).mean()


def _rsr(cfg: TGConfig, seed=0) -> RSRPolicy:
    return RSRPolicy(
        RSRConfig(
            nu=0.0,
            beta=0.0,
            gamma=0.0,
            t_warm=0.0,
            a_max=cfg.M,
            psi_override=None,
            b_enabled=False,
            shadow_enabled=False,
        ),
        d_model=cfg.D,
        generator=torch.Generator().manual_seed(seed),
    )


# --- 3.6a: everything round-trips, policy included -------------------------- #


def test_model_and_optimizer_round_trip(tmp_path):
    cfg = _tiny_cfg()
    model = TGModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids, mask, lengths = _batch(cfg)
    run_policy_loop(model, ids, mask, lengths, _rsr(cfg), step_fn=_loss_fn).backward()
    opt.step()

    path = ckpt.save(tmp_path / "c.pt", step=7, model=model, optimizer=opt)

    restored = TGModel(cfg)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    payload = ckpt.load(path, model=restored, optimizer=restored_opt)
    assert payload["step"] == 7
    for (n, a), (_, b) in zip(
        model.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert torch.equal(a, b), n
    assert restored_opt.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


def test_rsr_policy_state_round_trips(tmp_path):
    """🔴 The eviction log is per-run state. §3.4 requires per-eviction attribution
    from the first policy run; a resume that drops it has already broken that."""
    cfg = _tiny_cfg()
    model = TGModel(cfg)
    policy = _rsr(cfg)
    ids, mask, lengths = _batch(cfg)
    run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss_fn)
    assert len(policy.records) > 0
    before = policy.rank_shifts.distribution()

    path = ckpt.save(tmp_path / "c.pt", step=1, model=model, policy=policy)
    restored = _rsr(cfg, seed=999)  # deliberately a different head seed
    ckpt.load(path, model=TGModel(cfg), policy=restored)

    assert len(restored.records) == len(policy.records)
    assert restored.rank_shifts.distribution() == before
    assert restored.attribution_counts() == policy.attribution_counts()
    for a, b in zip(policy.head.parameters(), restored.head.parameters(), strict=True):
        assert torch.equal(a, b), "phi was not restored"


def test_lru_policy_state_round_trips(tmp_path):
    """LRU is §7.1's vacuity test and one of E7's two controls. Its `last_used` is
    exactly the state gauntlet 0.4 showed does not survive being kept elsewhere."""
    policy = LRUPolicy()
    slots = MemoryState(
        gestalts=torch.zeros(4, 2),
        written_at=torch.arange(4),
        live=torch.ones(4, dtype=torch.bool),
        step=9,
    )
    alpha = torch.zeros(1, 1, 4)
    alpha[:, :, 2] = 1.0
    policy.observe(
        slots,
        AttentionTrace(
            alpha=alpha,
            wo_v=torch.zeros(1, 1, 4, 2),
            live=torch.ones(4, dtype=torch.bool),
            step=9,
            eval_mode=True,
        ),
        9,
    )
    assert policy._last_used

    path = ckpt.save(tmp_path / "c.pt", step=1, model=nn.Linear(2, 2), policy=policy)
    restored = LRUPolicy()
    ckpt.load(path, policy=restored)
    assert restored._last_used == policy._last_used
    assert restored.select_eviction(slots, torch.zeros(2), 9) == policy.select_eviction(
        slots, torch.zeros(2), 9
    )


def test_resuming_into_the_wrong_arm_is_refused(tmp_path):
    """E3's arms differ only in the eviction rule (§3.7). A cross-arm resume would
    succeed silently and produce a run that is neither."""
    path = ckpt.save(tmp_path / "c.pt", step=1, model=nn.Linear(2, 2), policy=LRUPolicy())
    with pytest.raises(ckpt.CheckpointError, match="neither"):
        ckpt.load(path, policy=_rsr(_tiny_cfg()))


def test_rng_state_round_trips(tmp_path):
    torch.manual_seed(1234)
    torch.randn(5)
    path = ckpt.save(tmp_path / "c.pt", step=1, model=nn.Linear(2, 2))
    expected = torch.randn(5)

    torch.manual_seed(9999)  # somewhere else entirely
    ckpt.load(path, restore_rng=True)
    assert torch.equal(torch.randn(5), expected)


# --- 3.6b: the write survives SIGKILL mid-save ------------------------------ #


_KILLER = textwrap.dedent(
    """
    import sys, threading
    from pathlib import Path
    sys.path.insert(0, {src!r})
    import torch
    from rsr.train import checkpoint as ckpt

    target, phase = Path(sys.argv[1]), sys.argv[2]
    big = {{"format_version": 1, "step": 1, "model": {{"w": torch.randn(4_000_000)}},
            "optimizer": None, "policy": {{}}, "stream": {{}}, "meta": {{}}, "rng": None}}

    # Synchronisation, not timing: at the requested point in the write, tell the
    # parent and park forever. The parent SIGKILLs us while we are parked, so the
    # kill lands inside the window on every run, on any machine.
    def hook(at):
        if at == phase:
            sys.stdout.write("AT " + at + chr(10))
            sys.stdout.flush()
            threading.Event().wait()
    ckpt._write_hook = hook
    ckpt.atomic_write(target, big)
    print("COMPLETED", flush=True)
    """
)

_OLD_PAYLOAD = {
    "format_version": 1,
    "step": 0,
    "model": {},
    "optimizer": None,
    "policy": {},
    "stream": {},
    "meta": {},
    "rng": None,
}

#: Where the child is killed, and what the final path must hold afterwards.
#:
#: * `mid_write` -- half the bytes are out. A straight-to-final-path write
#:   (`tmp = path`, the battery's mutation) leaves a torn file here; the atomic
#:   write leaves the OLD checkpoint byte-identical.
#: * `before_replace` -- the new bytes are complete and fsynced but not renamed.
#:   The final path must still be the old checkpoint.
#: * `after_replace` -- killed after the rename, before the directory fsync and
#:   the sweep. The final path must be the complete NEW checkpoint ("never
#:   partial", not "never new" -- the old delay-sweep flake, 4/110).
#:
#: Until Brief 0b (2026-09-21) this was a sweep of wall-clock kill delays, which
#: caught the `tmp = path` mutation in 1 of 7 battery observations.
_KILL_AT = {"mid_write": "old", "before_replace": "old", "after_replace": "new"}


@pytest.mark.parametrize("phase", sorted(_KILL_AT))
def test_a_sigkill_mid_save_never_leaves_a_corrupt_checkpoint(tmp_path, phase):
    """🔴 Gauntlet 3.6. `torch.save` straight to the final path leaves a truncated
    file when the process dies during it, and the next resume loads garbage --
    after the run has already been lost.

    The child is SIGKILLed while parked at a named point inside `atomic_write`
    (see `checkpoint._write_hook`). The final path is then either **the old
    checkpoint, byte-identical** or **the complete new one** -- never a partial one.
    """
    src = str(Path(__file__).resolve().parents[1] / "src")
    target = tmp_path / "ckpt.pt"
    # A previous good checkpoint: a reader must never see it replaced by rubble.
    ckpt.atomic_write(target, _OLD_PAYLOAD)
    good = target.read_bytes()

    stderr_log = tmp_path / "child.stderr"
    with open(stderr_log, "w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", _KILLER.format(src=src), str(target), phase],
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        try:
            line = proc.stdout.readline()  # blocks until the child is parked
            reached = line.strip() == f"AT {phase}"
            if reached:
                proc.send_signal(signal.SIGKILL)
        finally:
            if proc.poll() is None and not reached:
                proc.kill()
            proc.wait()
            proc.stdout.close()
    assert reached, (
        f"the child never reached {phase!r} (read {line!r}, rc={proc.returncode}); "
        f"stderr: {stderr_log.read_text()[-2000:]}"
    )
    assert proc.returncode == -signal.SIGKILL, proc.returncode

    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as exc:
        pytest.fail(
            f"SIGKILL at {phase!r} left an unloadable checkpoint "
            f"({type(exc).__name__}: {exc})"
        )
    assert payload["format_version"] == 1
    if _KILL_AT[phase] == "old":
        assert target.read_bytes() == good, (
            f"SIGKILL at {phase!r} replaced the old checkpoint "
            f"(step={payload['step']}) before the atomic rename"
        )
    else:
        assert payload["step"] == 1 and payload["model"]["w"].numel() == 4_000_000

    # A temporary file MAY survive: SIGKILL cannot be caught, so no cleanup handler
    # runs. What must be true is that it cannot be mistaken for a checkpoint, and
    # that the next successful write sweeps it.
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    for name in leftovers:
        assert name.startswith(f".{target.name}.tmp."), name
    ckpt.atomic_write(
        target,
        {
            "format_version": 1,
            "step": 99,
            "model": {},
            "optimizer": None,
            "policy": {},
            "stream": {},
            "meta": {},
            "rng": None,
        },
    )
    assert not [p.name for p in tmp_path.iterdir() if ".tmp." in p.name], (
        "the next successful write did not sweep the stale temporary"
    )
    assert torch.load(target, map_location="cpu", weights_only=False)["step"] == 99


def test_the_write_hook_is_inert_in_production(tmp_path):
    """The SIGKILL test's seam (`checkpoint._write_hook`) must change nothing when
    unset. Without the hook, `atomic_write` and `save` produce exactly the bytes
    of one `torch.save` to the final file -- which is what they wrote before the
    seam existed (Brief 0b, 2026-09-21)."""
    assert ckpt._write_hook is ckpt._no_op

    torch.manual_seed(7)
    payload = dict(_OLD_PAYLOAD, step=3, model={"w": torch.randn(300_001)})
    expected = io.BytesIO()
    torch.save(payload, expected)
    ckpt.atomic_write(tmp_path / "a.pt", payload)
    assert (tmp_path / "a.pt").read_bytes() == expected.getvalue()

    model = nn.Linear(3, 5)
    path = ckpt.save(tmp_path / "s.pt", step=4, model=model, capture_rng=False)
    reference = io.BytesIO()
    torch.save(ckpt.Checkpoint(step=4, model=model.state_dict()).payload(), reference)
    assert path.read_bytes() == reference.getvalue()
    assert not [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]


def test_a_live_writers_temporary_is_never_swept(tmp_path):
    """The sweep must not disturb a concurrent writer -- only temporaries whose
    owning process is gone are removed."""
    target = tmp_path / "c.pt"
    # pid 1 (launchd/init) is always alive and is never this process, so it stands
    # in for a concurrent writer without colliding with `atomic_write`'s own
    # temporary -- which is suffixed with THIS pid and would be consumed by the
    # write itself.
    live = tmp_path / f".{target.name}.tmp.1"
    live.write_bytes(b"in progress")
    dead = tmp_path / f".{target.name}.tmp.999999"
    dead.write_bytes(b"orphaned")

    ckpt.atomic_write(
        target,
        {
            "format_version": 1,
            "step": 1,
            "model": {},
            "optimizer": None,
            "policy": {},
            "stream": {},
            "meta": {},
            "rng": None,
        },
    )
    assert ckpt._process_alive(1) and not ckpt._process_alive(999999)
    assert live.exists(), "swept a live process's temporary"
    assert not dead.exists(), "did not sweep an orphaned temporary"


def test_the_write_is_atomic_not_in_place(tmp_path):
    """The mechanism, asserted directly: a temp file in the same directory, then
    `os.replace`. Writing to the final path and hoping is the defect."""
    import inspect

    source = inspect.getsource(ckpt.atomic_write)
    assert "os.replace" in source
    assert "os.fsync" in source
    # The directory fsync is the subtle one: without it the file's contents are
    # durable but the rename may not be.
    assert source.count("os.fsync") >= 2, "the directory is not fsynced"


def test_a_truncated_checkpoint_is_refused_not_half_loaded(tmp_path):
    path = tmp_path / "c.pt"
    ckpt.save(path, step=1, model=nn.Linear(2, 2))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ckpt.CheckpointError, match=r"truncated|could not be read"):
        ckpt.load(path, model=nn.Linear(2, 2))


# --- 3.7: a mid-stream resume reproduces the uninterrupted run exactly ------- #


def _run_streams(model, policy, cfg, n_streams=4, *, stop_after=None, resume=None):
    """Run `n_streams` streams, optionally checkpointing or resuming partway."""
    losses = []
    start = 0
    if resume is not None:
        start = resume
    for stream in range(start, n_streams):
        ids, mask, lengths = _batch(cfg, seed=100 + stream)
        loss = run_policy_loop(model, ids, mask, lengths, policy, step_fn=_loss_fn)
        losses.append(round(float(loss), 10))
        policy.reset()
        if stop_after is not None and stream == stop_after:
            return losses, stream + 1
    return losses, n_streams


def test_a_mid_stream_resume_reproduces_the_uninterrupted_run_exactly(tmp_path):
    """🔴 Gauntlet 3.7. Not 'close' -- the same numbers."""
    cfg = _tiny_cfg()

    torch.manual_seed(7)
    straight_model, straight_policy = TGModel(cfg), _rsr(cfg)
    uninterrupted, _ = _run_streams(straight_model, straight_policy, cfg)

    torch.manual_seed(7)
    model, policy = TGModel(cfg), _rsr(cfg)
    first_half, next_stream = _run_streams(model, policy, cfg, stop_after=1)
    path = ckpt.save(
        tmp_path / "c.pt",
        step=next_stream,
        model=model,
        policy=policy,
        stream={"next_stream": next_stream},
    )

    torch.manual_seed(31337)  # the resuming process knows nothing
    resumed_model, resumed_policy = TGModel(cfg), _rsr(cfg, seed=4242)
    payload = ckpt.load(path, model=resumed_model, policy=resumed_policy)
    second_half, _ = _run_streams(
        resumed_model,
        resumed_policy,
        cfg,
        resume=payload["stream"]["next_stream"],
    )

    assert first_half + second_half == uninterrupted


def test_the_resume_test_would_fail_without_the_rng(tmp_path):
    """**The mutation.** If the run were deterministic regardless of RNG state, the
    test above would pass with the RNG dropped and would be proving nothing.

    Dropout is what makes it bite, so this runs the model in train mode.
    """
    cfg = _tiny_cfg()
    model = TGModel(cfg)
    model.train()
    ids, mask, lengths = _batch(cfg)

    torch.manual_seed(5)
    a = float(run_policy_loop(model, ids, mask, lengths, _rsr(cfg), step_fn=_loss_fn))
    torch.manual_seed(5)
    b = float(run_policy_loop(model, ids, mask, lengths, _rsr(cfg), step_fn=_loss_fn))
    c = float(run_policy_loop(model, ids, mask, lengths, _rsr(cfg), step_fn=_loss_fn))

    assert a == b, "same seed must give the same loss"
    assert a != c, (
        "the forward is insensitive to RNG state, so the resume test above cannot "
        "distinguish a restored generator from a fresh one -- it proves nothing"
    )


def test_the_checkpoint_records_provenance(tmp_path):
    """§12.4: a documented configuration is not evidence of what was run."""
    path = ckpt.save(
        tmp_path / "c.pt",
        step=3,
        model=nn.Linear(2, 2),
        meta={"git_sha": "abc", "device": "cpu"},
    )
    payload = ckpt.load(path)
    assert payload["meta"]["git_sha"] == "abc"
    assert payload["meta"]["torch"] == torch.__version__
    assert "platform" in payload["meta"]


def test_the_ledger_is_recorded_by_digest_not_copied(tmp_path):
    """A checkpoint carrying its own copy of the constants would let a resumed run
    disagree with the registry about what was measured."""
    ledger = tmp_path / "ledger.json"
    ledger.write_text('[{"name": "tau", "value": 0.25}]')
    digest = ckpt.ledger_digest(ledger)
    assert digest and len(digest) == 64
    assert ckpt.ledger_digest(tmp_path / "absent.json") is None
