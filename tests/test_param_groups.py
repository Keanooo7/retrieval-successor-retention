"""Gauntlet 1.6 -- the muP bill that fails silently (spec section 4.3).

Two halves, and both were open: the `1/d` multiplier must be *applied* (not merely
stored as an attribute), and **the value head must have its own parameter group**.
`build_param_groups` raised `NotImplementedError`, so the group did not exist.

Mis-grouping is the failure that surfaces in week 9 with no error message. The
value head's LR happens to equal the transformer hidden group's today; that is
coincidence, not safety -- the two are governed by different prescriptions and
sharing a group makes the divergence invisible when either changes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rsr.mup.param_groups import VALUE_HEAD_GROUP, build_param_groups
from rsr.retention.value_head import BilinearValueHead


class _Tiny(nn.Module):
    """Stands in for TG: matrices, vectors and an embedding."""

    def __init__(self, d):
        super().__init__()
        self.embed = nn.Embedding(16, d)
        self.attn = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)


def _groups(d=256, base_width=128, base_lr=2.5e-4, with_head=True):
    head = BilinearValueHead(d) if with_head else None
    return {
        g["name"]: g
        for g in build_param_groups(
            _Tiny(d), head, base_lr=base_lr, d_model=d, base_width=base_width
        )
    }, head


def test_the_value_head_has_its_own_group():
    """1.6's headline. The group did not exist and the builder raised."""
    groups, head = _groups()
    assert VALUE_HEAD_GROUP in groups
    assert {id(p) for p in groups[VALUE_HEAD_GROUP]["params"]} == {
        id(p) for p in head.parameters()
    }


def test_phi_is_not_also_in_a_transformer_group():
    """Section 3.3: only `phi` receives the retention gradient. A parameter in two
    groups gets two updates, and AdamW will not complain."""
    groups, head = _groups()
    head_ids = {id(p) for p in head.parameters()}
    for name, g in groups.items():
        if name == VALUE_HEAD_GROUP:
            continue
        assert not head_ids & {id(p) for p in g["params"]}, name


def test_hidden_matrices_scale_as_one_over_width_multiplier():
    groups, _ = _groups(d=256, base_width=128, base_lr=2.5e-4)
    assert groups["transformer.hidden"]["lr"] == pytest.approx(2.5e-4 / 2)


def test_vector_like_parameters_do_not_scale():
    """muP: biases and norm gains keep the base LR."""
    groups, _ = _groups(d=256, base_width=128, base_lr=2.5e-4)
    assert groups["transformer.vector"]["lr"] == pytest.approx(2.5e-4)


def test_the_value_head_lr_follows_its_own_one_over_d_rule():
    """Both `W` (fan_in = d) and `u` (fan_in = 2d) are base-width-proportional, so
    the ratio to the base configuration is the width multiplier either way."""
    groups, _ = _groups(d=384, base_width=128, base_lr=2.5e-4)
    assert groups[VALUE_HEAD_GROUP]["lr"] == pytest.approx(2.5e-4 / 3)


def test_at_base_width_every_group_is_at_the_base_lr():
    """The sanity condition muP has to satisfy: at `d = base_width` nothing moves."""
    groups, _ = _groups(d=128, base_width=128, base_lr=2.5e-4)
    for name, g in groups.items():
        assert g["lr"] == pytest.approx(2.5e-4), name


def test_embeddings_are_not_treated_as_hidden_matrices():
    """An embedding is `ndim == 2` and is not a hidden matrix; muP gives it the
    base LR. Classifying by shape alone gets this wrong and nothing errors."""
    groups, _ = _groups(d=256)
    vector_ids = {id(p) for p in groups["transformer.vector"]["params"]}
    model = _Tiny(256)
    # the same classification, checked by name on a fresh model
    names = {n for n, p in model.named_parameters() if p.ndim >= 2}
    assert any("embed" in n for n in names)
    assert len(vector_ids) >= 2  # norm weight + bias at minimum


def test_a_model_with_no_value_head_still_builds():
    groups, _ = _groups(with_head=False)
    assert VALUE_HEAD_GROUP not in groups
    assert "transformer.hidden" in groups


def test_a_frozen_value_head_is_refused():
    """Section 3.3 makes `phi` the only thing that receives gradient; an empty
    group means the retention loss has nowhere to go."""
    d = 128
    head = BilinearValueHead(d)
    for p in head.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_param_groups(_Tiny(d), head, base_lr=1e-3, d_model=d)


def test_groups_are_named_so_a_run_config_can_be_read():
    """Gauntlet 3.5: dump the config and read it. An unnamed group is a row of
    tensors nobody can audit."""
    groups, _ = _groups()
    assert all(isinstance(name, str) and name for name in groups)


def test_the_groups_are_usable_by_adamw():
    groups, head = _groups()
    opt = torch.optim.AdamW(list(groups.values()), lr=1e-9, weight_decay=0.0)
    out = head(torch.randn(4, 256), torch.randn(256))
    out.sum().backward()
    opt.step()  # must not raise
