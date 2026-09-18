"""Building a training config must go through the registry.

Gauntlet 0.2, 0.3 and 0.5 are one defect seen from three sides:

* **0.2** -- the DEFAULT `RSRConfig` had `t_warm = inf`, so a default-constructed
  RSR policy was stock TG forever while reporting `name = "rsr"`. Arm and control
  silently became one arm.
* **0.3** -- it also hardcoded `nu = 0.0` and `beta = 1.0`, both MEASURED by E1 and
  both refused by the registry *by name*. `nu = 0.0` is additionally section 3.7's
  *disabled* value, so the default arm shipped with the redundancy term off. This
  is defect D-1's exact shape one import from the module built to prevent it.
* **0.5** -- nothing in `src/` read the registry at all, so the machinery that
  refuses an unmeasured constant governed nothing.

The fix is the same for all three: there are **no defaults** for registry-owned
fields, and `from_registry` is the only supported constructor for a training run.
"""

from __future__ import annotations

import pytest
import torch

from rsr import constants as C
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import ContextSource, RSRConfig, RSRPolicy, SentPosIndex


@pytest.fixture
def empty(tmp_path):
    return C.Registry(definitions=C.REGISTRY.definitions, ledger_path=tmp_path / "l.json")


@pytest.fixture
def measured(empty):
    for name, value in (("nu", 0.3), ("beta", 0.1), ("gamma", 0.97)):
        empty.record(name, value, experiment="E1", run_id="e1-001", scope="synthetic")
    return empty


# --- 0.3 / 0.5: the config path raises on an empty ledger -------------------- #


def test_building_a_training_config_on_an_empty_ledger_raises(empty):
    with pytest.raises(C.UnmeasuredConstant) as exc:
        RSRConfig.from_registry("synthetic", steps_per_epoch=1200, registry=empty)
    assert "E1" in str(exc.value)


def test_a_populated_ledger_builds_a_config(measured):
    cfg = RSRConfig.from_registry("synthetic", steps_per_epoch=1200, registry=measured)
    assert (cfg.nu, cfg.beta, cfg.gamma) == (0.3, 0.1, 0.97)
    assert cfg.t_warm == 1200.0  # D-I: a float number of steps
    assert cfg.a_max == 48  # ADR-0004: A_max = S on synthetic


def test_from_registry_enforces_the_gamma_horizon(measured):
    """Section 4.5's `1/(1-gamma) <= min(S, A_max)`, checked at build time rather
    than discovered as a flat curve in week 7."""
    with pytest.raises(C.ConstraintViolation, match="horizon"):
        RSRConfig.from_registry(
            "synthetic", steps_per_epoch=1200, registry=measured, a_max=16
        )


def test_the_registry_is_read_from_a_non_test_module():
    """Gauntlet 0.5. `cli.py` read it, but the *config path* -- the one that
    governs a run -- did not, so the refusal machinery governed nothing."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "rsr"
    callers = set()
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            reads_registry = (
                isinstance(node, ast.Attribute)
                and node.attr == "get"
                and ast.unparse(node.value)
                in {"reg", "registry", "REGISTRY", "C.REGISTRY"}
            )
            if reads_registry:
                callers.add(path.name)
    assert "rsr.py" in callers, f"the config path does not read the registry: {callers}"


# --- 0.2: no default-constructible config, and the score actually decides ----- #


def test_there_is_no_default_constructible_config():
    """Gauntlet 0.2, fixed at the root rather than by changing a default value.

    A default for a MEASURED constant is a frozen unmeasured constant; the only
    safe default is none.
    """
    with pytest.raises(TypeError):
        RSRConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ["nu", "beta", "gamma", "t_warm", "a_max"])
def test_no_registry_owned_field_has_a_default(field):
    import dataclasses

    f = {x.name: x for x in dataclasses.fields(RSRConfig)}[field]
    assert f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


def test_one_hundred_evictions_are_attributable_to_the_score(measured):
    """Gauntlet 0.2's PASS. A run whose records are all `fifo_warmup` is stock TG
    reporting itself as RSR."""
    cfg = RSRConfig.from_registry(
        "synthetic", steps_per_epoch=0.5, registry=measured, b_enabled=False
    )
    assert cfg.t_warm == 0.5  # warmup ends immediately; dispatch is t < T_warm
    policy = RSRPolicy(cfg, d_model=4, generator=torch.Generator().manual_seed(0))
    capacity, written = 8, list(range(8))
    g = torch.Generator().manual_seed(0)
    for step in range(capacity, capacity + 100):
        st = MemoryState(
            gestalts=torch.randn(capacity, 4, generator=g),
            written_at=torch.tensor(written, dtype=torch.long),
            live=torch.ones(capacity, dtype=torch.bool),
            step=step,
        )
        v = policy.select_eviction(st, torch.randn(4, generator=g), step)
        written[v] = step
    counts = policy.attribution_counts()
    assert counts.get("fifo_warmup", 0) == 0, counts
    assert counts["psi-nu"] == 100, counts


def test_warmup_is_attributed_to_fifo_not_to_the_score(measured):
    """The converse, so the attribution field is proved to discriminate."""
    cfg = RSRConfig.from_registry(
        "synthetic", steps_per_epoch=10_000, registry=measured, b_enabled=False
    )
    policy = RSRPolicy(cfg, d_model=4, generator=torch.Generator().manual_seed(0))
    written = list(range(8))
    for step in range(8, 28):
        st = MemoryState(
            gestalts=torch.zeros(8, 4),
            written_at=torch.tensor(written, dtype=torch.long),
            live=torch.ones(8, dtype=torch.bool),
            step=step,
        )
        written[policy.select_eviction(st, torch.zeros(4), step)] = step
    assert policy.attribution_counts() == {"fifo_warmup": 20}


# --- D-C and D-D switches ---------------------------------------------------- #


def test_running_context_is_named_and_raises(measured):
    """D-C: "a named enum with one value implemented and the other raising"."""
    cfg = RSRConfig.from_registry(
        "synthetic",
        steps_per_epoch=1200,
        registry=measured,
        b_enabled=False,
        context_source=ContextSource.RUNNING_CONTEXT,
    )
    with pytest.raises(NotImplementedError, match="ablation"):
        RSRPolicy(cfg, d_model=4)


def test_the_absolute_age_index_exists_and_is_off_by_default(measured):
    """D-D item 3: a config-gated absolute-age variant, default off."""
    assert set(SentPosIndex) == {SentPosIndex.RANK, SentPosIndex.ABSOLUTE_AGE}
    cfg = RSRConfig.from_registry(
        "synthetic", steps_per_epoch=1200, registry=measured, b_enabled=False
    )
    assert cfg.sent_pos_index is SentPosIndex.RANK


def test_b_enabled_without_a_bias_is_refused(measured):
    """A silently absent `b` is defect D-1's shape: the control loop reported as
    unnecessary rather than as absent."""
    cfg = RSRConfig.from_registry("synthetic", steps_per_epoch=1200, registry=measured)
    assert cfg.b_enabled is True
    with pytest.raises(ValueError, match="E0e"):
        RSRPolicy(cfg, d_model=4)
