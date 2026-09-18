"""Policies must follow the memory's device (E0C, 2026-09-17).

🔴 **Found by measuring, not by testing.** `RSRPolicy` held its value head on the
CPU while the memory lived on the accelerator, and every tensor it created was a
CPU tensor. Both paths failed on the first MPS run:

```
neg_age:      RuntimeError: Expected all tensors to be on the same device,
              but found at least two devices, mps:0 and cpu
learned head: RuntimeError: Tensor for argument #2 'mat2' is on CPU, but
              expected it to be on GPU
```

`neg_age` has no head at all, so this was not "the head was not moved" -- it was
`torch.full(...)` with no `device=`. **E0b could not catch it**: §3.7's reduction
runs on CPU by design (ADR-0001 D3, for bit-exact determinism), so the entire RSR
arm would have failed at the first accelerated run with every test green.

Two tests, deliberately: a **live** one where an accelerator exists, and a
**static** one that runs everywhere including CI, because the live one is a skip on
any machine without a GPU and a skipped test is not a passing test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from rsr.baselines.fifo import FIFOPolicy
from rsr.baselines.lru import LRUPolicy
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import RSRConfig, RSRPolicy

SRC = Path(__file__).resolve().parents[1] / "src" / "rsr"

_FACTORIES = {
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "tensor",
    "randn",
    "rand",
    "randint",
    "eye",
    "linspace",
}
"""Factories that default to CPU. The `*_like` forms inherit their argument's
device and are therefore safe -- and preferable."""

_CONSTRUCTION = {"__init__", "reset_parameters"}
"""Exempt: module construction legitimately allocates on CPU and `.to(device)`
moves it -- that is how every `nn.Module` in PyTorch works. The hazard is a factory
on a path that runs PER STEP, where nothing moves it and nothing looks.

The scan flagged `nn.Parameter(torch.empty(...))` in `value_head.py.__init__`
before this exemption existed. A check that cries wolf gets deleted, so the
exemption is narrow and `test_the_scan_exempts_module_construction` pins it."""


def _accelerator() -> str | None:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _state(device: str, capacity: int = 8, d: int = 32) -> MemoryState:
    return MemoryState(
        gestalts=torch.randn(capacity, d, device=device),
        written_at=torch.arange(capacity, device=device),
        live=torch.ones(capacity, dtype=torch.bool, device=device),
        step=capacity + 1,
    )


def _learned(capacity: int, d: int) -> RSRConfig:
    return RSRConfig(
        nu=0.0,
        beta=0.0,
        gamma=0.0,
        t_warm=0.0,
        a_max=capacity,
        psi_override=None,
        b_enabled=False,
        shadow_enabled=False,
    )


# --- static: runs everywhere, including CI with no accelerator -------------- #


def _cpu_pinned_factories(path: Path) -> list[str]:
    """Tensor factories called without `device=`, which silently produce CPU."""
    tree = ast.parse(path.read_text())
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _CONSTRUCTION:
            exempt.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno in exempt:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _FACTORIES:
            continue
        if ast.unparse(func.value) != "torch":
            continue
        if any(kw.arg == "device" for kw in node.keywords):
            continue
        found.append(f"{path.name}:{node.lineno} {ast.unparse(node)[:70]}")
    return found


@pytest.mark.parametrize(
    "path",
    sorted((SRC / "retention").glob("*.py")) + sorted((SRC / "baselines").glob("*.py")),
    ids=lambda p: p.name,
)
def test_no_cpu_pinned_tensor_factory_in_a_policy(path):
    """Every `torch.zeros/full/arange/...` in a policy must say `device=`, or use
    a `*_like` form. A CPU tensor built inside a policy is invisible until the
    first accelerated run, and then it is fatal."""
    offences = _cpu_pinned_factories(path)
    assert not offences, (
        "these create CPU tensors regardless of where the memory lives; pass "
        "device= or use a *_like form:\n  " + "\n  ".join(offences)
    )


def test_the_scan_catches_the_defect_it_was_written_for(tmp_path):
    """Mutation. A check nothing reddens adds nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\ndef f(n):\n    return torch.full((n,), float('inf'))\n")
    assert len(_cpu_pinned_factories(bad)) == 1


def test_the_scan_exempts_module_construction(tmp_path):
    """`nn.Parameter(torch.empty(...))` in `__init__` is correct and must not be
    flagged -- the scan reported it on `value_head.py` before this exemption, and a
    check that cries wolf gets deleted."""
    f = tmp_path / "f.py"
    f.write_text(
        "import torch\n"
        "class M:\n"
        "    def __init__(self, d):\n"
        "        self.w = torch.empty(d, d)\n"
        "    def step(self, d):\n"
        "        return torch.zeros(d)\n"
    )
    offences = _cpu_pinned_factories(f)
    assert len(offences) == 1 and "zeros" in offences[0]


def test_the_scan_accepts_the_correct_forms(tmp_path):
    good = tmp_path / "good.py"
    good.write_text(
        "import torch\n"
        "def f(x, n):\n"
        "    a = torch.zeros(n, device=x.device)\n"
        "    b = torch.zeros_like(x)\n"
        "    return a, b\n"
    )
    assert _cpu_pinned_factories(good) == []


def test_rsr_policy_exposes_a_to_method():
    """It is not an `nn.Module`, so nothing moves it automatically."""
    policy = RSRPolicy(RSRConfig.reduction_to_tg(capacity=8), d_model=32)
    assert hasattr(policy, "to")
    assert policy.to("cpu") is policy  # chains, like nn.Module.to


# --- live: only where an accelerator exists --------------------------------- #

ACCEL = _accelerator()
requires_accel = pytest.mark.skipif(
    ACCEL is None,
    reason=(
        "no accelerator on this machine. The STATIC scan above covers the same "
        "defect class and runs everywhere, so this skip does not leave the "
        "property unchecked."
    ),
)


@requires_accel
@pytest.mark.parametrize("override", ["neg_age", None])
def test_rsr_policy_runs_where_the_memory_lives(override):
    """Both paths. `neg_age` has no head, and it failed too -- the defect was a
    bare `torch.full`, not an unmoved module."""
    capacity, d = 8, 32
    cfg = (
        RSRConfig.reduction_to_tg(capacity=capacity)
        if override == "neg_age"
        else _learned(capacity, d)
    )
    policy = RSRPolicy(cfg, d_model=d, generator=torch.Generator().manual_seed(0))
    slots = _state(ACCEL, capacity, d)
    victim = policy.select_eviction(slots, torch.randn(d, device=ACCEL), capacity + 1)
    assert 0 <= victim < capacity
    assert bool(slots.live[victim])


@requires_accel
def test_the_value_head_follows_the_memory():
    capacity, d = 8, 32
    policy = RSRPolicy(_learned(capacity, d), d_model=d)
    assert next(policy.head.parameters()).device.type == "cpu"
    policy.select_eviction(_state(ACCEL, capacity, d), torch.randn(d, device=ACCEL), 9)
    assert next(policy.head.parameters()).device.type == ACCEL


@requires_accel
@pytest.mark.parametrize("policy_factory", [FIFOPolicy, LRUPolicy])
def test_the_baselines_run_where_the_memory_lives(policy_factory):
    """LRU is E7's control and §7.1's vacuity test; a comparator that cannot run
    on the accelerator is a comparator that silently does not run."""
    capacity, d = 8, 32
    slots = _state(ACCEL, capacity, d)
    victim = policy_factory().select_eviction(
        slots, torch.randn(d, device=ACCEL), capacity + 1
    )
    assert 0 <= victim < capacity
