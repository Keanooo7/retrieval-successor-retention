"""Configuration for the PyTorch TG transcription (gauntlet 2.4).

A deliberate subset of the reference `tg/models/tg_config.py`: the fields this
transcription honours, with the reference's own defaults and the same derived
properties. **Fields the transcription does not implement are absent rather than
ignored** -- an accepted-and-unused option is the shape of defect the gauntlet
calls a decorative switch.

Not implemented, and therefore not present: `multi_eos_*` (K > 1 memory entries
per sentence), `memory_mode='in_context'`, `srep_pool_mode='mean_pool'`,
`segmentation='span'`, `block_config` orderings beyond `S`/`C`, and muP's
`8_over_d` attention scale. Each raises if requested via `from_reference_dict`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["TGConfig", "compute_ffn_dim"]


def _round_to_multiple(x: float, align_to: int) -> int:
    return int(align_to * round(float(x) / align_to))


def compute_ffn_dim(d: int, activation: str = "swiglu", align_to: int = 8) -> int:
    """Mirror of the reference `compute_ffn_dim` (`models/ffn.py`)."""
    base = int(4 * d)
    act = (activation or "gelu").lower()
    if act == "swiglu":
        return _round_to_multiple((2.0 / 3.0) * float(base), align_to)
    if act == "gelu":
        return base
    raise ValueError(f"Unsupported FFN activation: {activation}")


@dataclass(frozen=True)
class TGConfig:
    """Geometry and the handful of behavioural switches the transcription honours."""

    D: int = 128
    H: int = 2
    """Head COUNT is the width axis under muP; head DIM is not. The reference is
    `D = 768 / H = 12`, i.e. `head_dim = 64`, and `H = 2` preserves that at
    `D = 128`."""

    N: int = 12
    V: int = 512
    F: int | None = None  # None => compute_ffn_dim(D, ffn_activation)

    max_sentence_tokens: int = 64
    sentence_tail_len: int = 1
    max_sentences_in_short_term: int = 40

    block_config: tuple[str, ...] = field(default_factory=lambda: ("S", "C") * 6)
    ffn_activation: str = "swiglu"
    layer_norm_epsilon: float = 1e-6

    memory_gate_init: float = 1.0

    dropout: float = 0.1
    attn_dropout: float = 0.2
    srep_dropout: float = 0.15
    token_dropout: float = 0.15
    memory_dropout: float = 0.0
    dropout_scale: float = 1.0

    srep_extraction_layer: int = 6
    """**0-indexed** block whose output feeds the S_REP head. The spec's "layer 7"
    (§5.1, 1-indexed) and the reference README's "layer 6" are the same block --
    verified against `tg_config.py:146`, not guessed. See `docs/code-vs-paper.md`
    row 6."""

    srep_norm_target: float = 1.0
    srep_norm_margin: float = 0.1
    srep_norm_reg_weight: float = 0.01

    stm_cross_pos_mode: str = "sinusoidal"  # 'sinusoidal' | 'none'
    stm_positional_weight: float = 1.0
    """D-D / ADR-0006 item 2: the `P^(sent)`-ablated arm is
    `stm_cross_pos_mode='none'`, or equivalently `stm_positional_weight=0.0`. Both
    knobs are carried so the ablation costs a config line and cannot drift from the
    main arm."""

    detach_sreps_for_memory: bool = False
    """🔴 MUST stay False. The reference's own comment: "gradients have to flow from
    later sentences back through memory." This is §3.1 fact 1, and it is the
    property the gradient fixtures exist to check."""

    stm_backprop_window: int | None = None
    bos_replacement_mode: str = "copy"  # 'copy' | 'off'
    bos_context_detach: bool = False
    use_memory: bool = True

    pad_id: int = 50257
    bos_id: int = 50258
    eos_id: int = 50259
    eod_id: int = 50260

    def __post_init__(self) -> None:
        if self.D % self.H:
            raise ValueError(f"D {self.D} not divisible by H {self.H}")
        if self.D % 2:
            raise ValueError("D must be even for the sinusoidal STM positions")
        if len(self.block_config) != self.N:
            raise ValueError(
                f"block_config length ({len(self.block_config)}) must match N ({self.N})"
            )
        for b in self.block_config:
            if b not in ("S", "C"):
                raise ValueError(
                    f"block type {b!r} is not transcribed. The reference also has "
                    f"'SC', 'CS' and 'P' -- the per-layer-capacity ablation -- and "
                    f"they are absent here rather than silently treated as 'S'."
                )
        if self.stm_cross_pos_mode not in ("sinusoidal", "none"):
            raise ValueError(f"stm_cross_pos_mode={self.stm_cross_pos_mode!r}")
        if self.bos_replacement_mode not in ("copy", "off"):
            raise ValueError(f"bos_replacement_mode={self.bos_replacement_mode!r}")
        if self.F is None:
            object.__setattr__(self, "F", compute_ffn_dim(self.D, self.ffn_activation))
        object.__setattr__(self, "block_config", tuple(self.block_config))

    # --- derived ----------------------------------------------------------- #

    @property
    def L(self) -> int:
        """Fixed sentence length: 1 ([BOS]) + tokens + tail."""
        return 1 + self.max_sentence_tokens + self.sentence_tail_len

    @property
    def M(self) -> int:
        return self.max_sentences_in_short_term

    @property
    def head_dim(self) -> int:
        return self.D // self.H

    @property
    def attention_logit_scale(self) -> float:
        return 1.0 / math.sqrt(float(self.head_dim))

    @property
    def srep_layer_idx(self) -> int:
        if self.srep_extraction_layer < 0:
            return self.N
        return min(self.srep_extraction_layer, self.N - 1)

    @property
    def token_dropout_now(self) -> float:
        return self.token_dropout * self.dropout_scale

    @property
    def memory_dropout_now(self) -> float:
        return self.memory_dropout * self.dropout_scale

    @property
    def srep_dropout_now(self) -> float:
        return self.srep_dropout * self.dropout_scale

    @property
    def cross_attention_layers(self) -> tuple[int, ...]:
        """Which blocks carry cross-attention -- **six**, not twelve (D-E)."""
        return tuple(i for i, b in enumerate(self.block_config) if b == "C")

    @classmethod
    def from_reference_dict(cls, d: dict) -> TGConfig:
        """Build from a golden-fixture sidecar's `config` block.

        Raises on any field the transcription does not implement, rather than
        dropping it: a fixture generated under an unimplemented option would be
        compared against a different model.
        """
        unsupported = {
            "multi_eos_count": 1,
            "memory_mode": "external",
            "srep_pool_mode": "eos",
            "segmentation": "sentence",
            "mup_enabled": False,
        }
        for key, expected in unsupported.items():
            if key in d and d[key] != expected:
                raise NotImplementedError(
                    f"the fixture sets {key}={d[key]!r}; this transcription "
                    f"implements only {key}={expected!r} (gauntlet 2.4 scope)."
                )
        return cls(
            D=d["D"],
            H=d["H"],
            N=d["N"],
            V=d["V"],
            max_sentences_in_short_term=d["M"],
            max_sentence_tokens=d["L"] - 2,
            block_config=tuple(d["block_config"]),
            srep_extraction_layer=d["srep_extraction_layer"],
            srep_norm_target=d["srep_norm_target"],
        )
