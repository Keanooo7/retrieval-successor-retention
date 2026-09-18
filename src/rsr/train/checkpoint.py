"""Checkpoint and resume (gauntlet 3.6, 3.7).

Three properties, and each one fails silently without a test:

1. **Every stateful object round-trips, policy state included.** A checkpoint that
   restores the model and the optimizer and forgets the policy resumes a *different
   experiment*: LRU's `last_used`, H2O's accumulated attention and RSR's eviction
   log are all per-run state, and losing them silently re-runs the arm from a
   different starting condition. §3.4 requires per-eviction attribution from the
   first policy run, and a resume that drops the log has already broken that.

2. **The write survives `SIGKILL` mid-save.** `torch.save` straight to the final
   path leaves a truncated file when the process dies during it, and the next
   resume loads garbage or refuses -- after the run has already been lost. Writes
   go to a temporary file in the same directory, are `fsync`ed, and are then
   `os.replace`d, which is atomic on POSIX. The directory itself is fsynced so the
   rename is durable.

3. **A mid-stream resume reproduces the uninterrupted run exactly.** That requires
   the RNG, not just the weights. Dropout masks, data order and `phi`'s
   initialisation all come from generator state, and a resume that restores
   parameters but restarts the RNG produces a run that is *plausible* and not the
   same -- which is the worst kind, because nothing looks wrong.

## What is deliberately NOT in the payload

**The constants ledger.** It is provenance, it lives in git and in
`measurements/ledger.json`, and a checkpoint that carried its own copy would let a
resumed run disagree with the registry about what was measured. The checkpoint
records the ledger's digest so a mismatch is *detectable*, and refuses nothing --
detecting is the registry's job.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "Checkpoint",
    "CheckpointError",
    "RngState",
    "atomic_write",
    "load",
    "save",
]

FORMAT_VERSION = 1


class CheckpointError(Exception):
    """A checkpoint could not be written, read or trusted."""


@dataclass
class RngState:
    """Everything that makes a run reproducible rather than merely plausible.

    `torch.cuda` and `torch.mps` keep their own generators. Restoring only the CPU
    one gives identical parameters and different dropout, which shows up as a loss
    curve that diverges slowly from the uninterrupted run -- and looks like noise.
    """

    cpu: torch.Tensor
    cuda: list[torch.Tensor] = field(default_factory=list)
    mps: torch.Tensor | None = None
    python: tuple[Any, ...] | None = None
    numpy: Any = None

    @classmethod
    def capture(cls) -> RngState:
        import random

        state = cls(cpu=torch.get_rng_state(), python=random.getstate())
        if torch.cuda.is_available():
            state.cuda = torch.cuda.get_rng_state_all()
        if torch.backends.mps.is_available():
            try:
                state.mps = torch.mps.get_rng_state()
            except (AttributeError, RuntimeError):
                state.mps = None
        try:
            import numpy as np

            state.numpy = np.random.get_state()
        except ImportError:
            state.numpy = None
        return state

    def restore(self) -> None:
        import random

        torch.set_rng_state(self.cpu)
        if self.python is not None:
            random.setstate(self.python)
        if self.cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda)
        if self.mps is not None and torch.backends.mps.is_available():
            with contextlib.suppress(AttributeError, RuntimeError):
                torch.mps.set_rng_state(self.mps)
        if self.numpy is not None:
            with contextlib.suppress(ImportError):
                import numpy as np

                np.random.set_state(self.numpy)


@dataclass
class Checkpoint:
    """One resumable point in a run."""

    step: int
    model: dict[str, Any]
    optimizer: dict[str, Any] | None = None
    policy: dict[str, Any] = field(default_factory=dict)
    """🔴 Not optional in practice. See the module docstring."""

    rng: RngState | None = None
    stream: dict[str, Any] = field(default_factory=dict)
    """Position in the data: epoch, stream index, sentence index."""

    meta: dict[str, Any] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    def payload(self) -> dict[str, Any]:
        out = {
            "format_version": self.format_version,
            "step": self.step,
            "model": self.model,
            "optimizer": self.optimizer,
            "policy": self.policy,
            "stream": self.stream,
            "meta": {
                **self.meta,
                "torch": torch.__version__,
                "python": platform.python_version(),
                "platform": f"{platform.system()} {platform.machine()}",
            },
        }
        out["rng"] = asdict(self.rng) if self.rng is not None else None
        return out


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write so that `SIGKILL` at any instant leaves `path` intact or absent.

    The sequence matters and every step of it is load-bearing:

    1. serialise to memory first, so a serialisation failure never touches disk;
    2. write to a temporary file **in the same directory**, because `os.replace`
       is only atomic within a filesystem;
    3. `flush` + `fsync` the file, so the bytes are on the medium and not merely
       in the page cache;
    4. `os.replace`, which is atomic on POSIX -- a reader sees the old file or the
       new one, never a mixture;
    5. `fsync` the **directory**, so the rename itself survives power loss.

    Skipping (5) is the subtle one: the file's contents are durable but the
    directory entry pointing at them may not be.

    **Stale temporaries are swept on the next successful write.** `SIGKILL` cannot
    be caught, so no cleanup handler runs and the temporary file from a killed
    process survives. It can never be mistaken for a checkpoint -- the name is
    dotted and pid-suffixed and `load()` takes an explicit path -- but across many
    crashes it is unbounded disk. Only temporaries whose owning process is gone are
    removed, so a concurrent writer is never disturbed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        _sweep_stale_temporaries(path)
    except BaseException:
        # SIGKILL never reaches here; this covers everything that does.
        tmp.unlink(missing_ok=True)
        raise


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _sweep_stale_temporaries(path: Path) -> None:
    """Remove `.<name>.tmp.<pid>` files left by processes that no longer exist."""
    prefix = f".{path.name}.tmp."
    for candidate in path.parent.glob(f"{prefix}*"):
        suffix = candidate.name[len(prefix) :]
        if not suffix.isdigit():
            continue
        if _process_alive(int(suffix)):
            continue
        with contextlib.suppress(OSError):
            candidate.unlink()


def save(
    path: Path | str,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    policy: Any = None,
    stream: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    capture_rng: bool = True,
) -> Path:
    """Write a checkpoint atomically. Returns the path written."""
    policy_state = {}
    if policy is not None:
        policy_state = policy_state_dict(policy)
    checkpoint = Checkpoint(
        step=step,
        model=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        policy=policy_state,
        rng=RngState.capture() if capture_rng else None,
        stream=dict(stream or {}),
        meta=dict(meta or {}),
    )
    path = Path(path)
    atomic_write(path, checkpoint.payload())
    return path


def load(
    path: Path | str,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    policy: Any = None,
    restore_rng: bool = True,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Read a checkpoint and restore into whatever was passed.

    Raises `CheckpointError` on a truncated or unreadable file rather than
    returning a half-restored model -- a partially restored run is the failure
    that looks like a training instability three days later.
    """
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # truncation, corruption, version skew
        raise CheckpointError(
            f"{path} could not be read ({type(exc).__name__}: {exc}). A truncated "
            f"checkpoint means the atomic-write path was bypassed -- check for a "
            f"direct torch.save to the final filename."
        ) from exc

    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        raise CheckpointError(
            f"{path} is format version {version!r}, this build writes "
            f"{FORMAT_VERSION}. Refusing rather than guessing at the difference."
        )

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if policy is not None:
        load_policy_state(policy, payload.get("policy") or {})
    if restore_rng and payload.get("rng") is not None:
        RngState(**payload["rng"]).restore()
    return payload


# --------------------------------------------------------------------------- #
# Policy state
# --------------------------------------------------------------------------- #

_POLICY_STATE_ATTRS = ("_last_used", "records", "rank_shifts", "total", "_accum")
"""Attributes a policy may own. Enumerated rather than pickling `__dict__`, so a
policy that grows a tensor field does not silently start carrying device state."""


def policy_state_dict(policy: Any) -> dict[str, Any]:
    """Per-run policy state, as plain data.

    A policy that implements `state_dict()` owns its own format; otherwise this
    collects the known attributes. **`name` is always recorded**, so a resume into
    the wrong arm is detectable rather than silent -- restoring an LRU checkpoint
    into an RSR policy would otherwise succeed and produce a run that is neither.
    """
    if hasattr(policy, "state_dict"):
        state = dict(policy.state_dict())
    else:
        state = {
            attr: getattr(policy, attr)
            for attr in _POLICY_STATE_ATTRS
            if hasattr(policy, attr)
        }
    state["__policy_name__"] = getattr(policy, "name", type(policy).__name__)
    if getattr(policy, "head", None) is not None:
        state["__head__"] = policy.head.state_dict()
    return state


def load_policy_state(policy: Any, state: dict[str, Any]) -> None:
    """Restore per-run policy state, refusing a cross-arm resume."""
    state = dict(state)
    saved_name = state.pop("__policy_name__", None)
    own_name = getattr(policy, "name", type(policy).__name__)
    if saved_name is not None and saved_name != own_name:
        raise CheckpointError(
            f"this checkpoint was written by policy {saved_name!r} and is being "
            f"restored into {own_name!r}. E3's arms differ only in the eviction "
            f"rule (§3.7), so resuming across them produces a run that is neither."
        )
    head_state = state.pop("__head__", None)
    if head_state is not None and getattr(policy, "head", None) is not None:
        policy.head.load_state_dict(head_state)
    if hasattr(policy, "load_state_dict"):
        policy.load_state_dict(state)
        return
    for attr, value in state.items():
        setattr(policy, attr, value)


def ledger_digest(path: Path | str) -> str | None:
    """SHA-256 of the measurement ledger, for the checkpoint's `meta`.

    Recorded, not enforced: the registry owns what a constant is, and a checkpoint
    that carried its own copy would let a resumed run disagree with it. This makes
    a disagreement visible.
    """
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
