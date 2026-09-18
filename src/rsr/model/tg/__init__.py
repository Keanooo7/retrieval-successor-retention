"""The PyTorch TG transcription (ADR-0001, gauntlet 2.4).

The reference is the released code, not the paper (D-A). See `model.py` for the
five places a transcription of *this* model goes wrong, and `docs/code-vs-paper.md`
for every divergence found between the two.
"""

from rsr.model.tg.config import TGConfig
from rsr.model.tg.loading import load_reference_params
from rsr.model.tg.model import (
    Memory,
    StepOutput,
    TGModel,
    init_memory,
    memory_positions,
    push_memory,
    run_sentence_loop,
    sinusoidal_key_pe,
)

__all__ = [
    "Memory",
    "StepOutput",
    "TGConfig",
    "TGModel",
    "init_memory",
    "load_reference_params",
    "memory_positions",
    "push_memory",
    "run_sentence_loop",
    "sinusoidal_key_pe",
]
