"""Baseline retention policies. §5.4."""

from rsr.baselines.simple import FIFOPolicy, LRUPolicy, RandomPolicy
from rsr.baselines.stubs import (
    ExpireSpanPolicy,
    H2OPolicy,
    LeadingEdgePolicy,
    OraclePolicy,
)

__all__ = [
    "ExpireSpanPolicy",
    "FIFOPolicy",
    "H2OPolicy",
    "LRUPolicy",
    "LeadingEdgePolicy",
    "OraclePolicy",
    "RandomPolicy",
]
