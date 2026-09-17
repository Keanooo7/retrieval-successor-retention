"""Attention-share Gini -- the collapse monitor (spec sections 5.5, 7.4).

Rising Gini with flat loss is retention collapse in progress. The bias loop
(section 3.5) is a control system with no convergence proof in this setting, so
this is monitored rather than assumed.
"""

from __future__ import annotations

__all__ = ["attention_share_gini"]


def attention_share_gini(*args, **kwargs):
    raise NotImplementedError("Sprint 2. Spec sections 5.5, 7.4.")
