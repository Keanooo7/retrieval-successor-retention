"""Expire-Span -- learned soft expiry [P7] (spec section 5.4).

**The referendum, not a comparison** (defect D-4, falsifier 6). Soft differentiable
expiry trains retention by the LM loss directly and makes sections 3.2-3.4's entire
apparatus unnecessary. It is the one baseline whose *success invalidates the design
rather than the result*.

Runs at the **week-4 gate on synthetic** (release condition 3). The demotion path
was deleted.

## The mechanism, ported onto TG's slot memory

Each gestalt `h_i` in memory gets a span

    e_i = L · sigmoid(w · h_i + b)                        (spans)

-- a *static* span, a function of the gestalt alone (spec section 2's "learned
*static* span"). At sentence step `t` the slot has age `t - written_at_i` and

    r_i(t) = e_i - (t - written_at_i)                     (remaining)
    m_i(t) = clamp(1 + r_i(t) / R, 0, 1)                  (mask_from_remaining)

and the mask enters every cross-attention layer as `a_i ∝ m_i · exp(s_i)`
(`rsr.model.tg.model.CrossAttention`, `mem_weight`). Because `m` is piecewise
linear in `e`, the LM loss has a gradient on `w, b` wherever a slot is on the ramp:
**that gradient is Expire-Span.** An auxiliary `alpha · Σ e_i` charges each admitted
gestalt its span once, so spans do not grow without cost.

TG's memory has a hard capacity `M`, which [P7]'s does not. When it is full the
slot with the **least remaining span** is evicted (`select_eviction`) -- the one
closest to, or furthest past, its own expiry. Expired slots that are not yet
evicted keep their rank but receive zero attention. Whether to free them early is
ADR-0010 question 5.

## What is deliberately NOT decided here

* **`grad_path`** -- whether the span gradient may reach the gestalt, and so the
  transformer and `W_sent` (`through_gestalt`), stops at the predictor
  (`predictor`), or is off (`none`). CLAUDE.md's stop-gradient rule governs RSR's
  `phi`; whether a baseline may do otherwise is an owner decision. The config has
  **no default** for it, or for anything else.
* `L`, `R`, `alpha`, the dropout rate and the initial span are unregistered in
  `src/rsr/constants.py`, and B-3 says the referendum must tune them on an equal
  budget. They come in through `ExpireSpanConfig`, never from a literal here.

ADR-0010 (PROPOSED) lists every open choice. **The [P7] source is not in
`~/research-corpus`**; each formula above is from the spec's description plus the
paper as recalled, and is marked UNVERIFIED there.

## Gauntlet 0.4 -- why this policy has no per-slot state to reset

The span is recomputed each step from the occupant's own gestalt, and the age from
`written_at`, both of which the memory owns and `write_at` updates. A new occupant
therefore has its own span and age 0 **by construction**: there is no cached value
that could describe the previous tenant. `run_policy_loop` passes `on_write` the
victim index, which is not where the newcomer lands (`write_at` compacts), and is
0 for a row that was not full -- a cache keyed on that index would be wrong twice.

## The E0b RNG trap

Construction draws nothing: `w` starts at zero and `b` at `logit(init fraction)`,
so every gestalt starts with the same span and the policy **evicts exactly FIFO at
initialisation**. Structured dropout draws from a private CPU generator seeded by
`cfg.seed`, then moves the draw to the memory's device, so the global stream -- and
with it data order and model dropout -- is untouched, on CPU or CUDA alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rsr.retention.policy import AttentionTrace, MemoryState

__all__ = ["GRAD_PATHS", "ExpireSpanConfig", "ExpireSpanPolicy"]

GRAD_PATHS = ("none", "predictor", "through_gestalt")
"""The span gradient's reach (ADR-0010 question 1 -- the owner's choice).

* `none` -- the mask is computed under `no_grad`; `w, b` never train. An ablation:
  fixed spans, soft expiry.
* `predictor` -- the LM loss and the span loss reach `w, b`; the gestalt enters the
  predictor detached, so nothing flows into the transformer or `W_sent` through
  the span.
* `through_gestalt` -- the gestalt enters with its graph: span gradient reaches the
  transformer and `W_sent`. The [P7]-faithful reading (UNVERIFIED).
"""


@dataclass(frozen=True)
class ExpireSpanConfig:
    """Every field is required. None of these is registered in `constants.py`, and
    B-3 makes their values the referendum's tuning budget -- a default would be an
    agent choosing it."""

    max_span: float
    """`L`, in sentence steps. The span is `L · sigmoid(·)`."""

    ramp: float
    """`R`, in sentence steps: how far past expiry the mask falls from 1 to 0."""

    loss_coef: float
    """`alpha` on the span loss `Σ e_i` (per admitted gestalt, averaged over rows)."""

    dropout: float
    """Structured dropout: the probability a live slot's mask is zeroed for one
    step in training, one draw per `(row, slot)` shared by every layer and head."""

    grad_path: str
    """One of `GRAD_PATHS`. No default, deliberately (ADR-0010)."""

    init_span_fraction: float
    """Initial span as a fraction of `L`: `b = logit(fraction)`, `w = 0`."""

    seed: int
    """Seeds the private dropout generator."""

    def __post_init__(self) -> None:
        if self.grad_path not in GRAD_PATHS:
            raise ValueError(f"grad_path={self.grad_path!r}; one of {GRAD_PATHS}")
        if not self.max_span > 0:
            raise ValueError(f"max_span must be > 0, got {self.max_span}")
        if not self.ramp > 0:
            raise ValueError(f"ramp must be > 0, got {self.ramp}")
        if not self.loss_coef >= 0:
            raise ValueError(f"loss_coef must be >= 0, got {self.loss_coef}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 < self.init_span_fraction < 1.0:
            raise ValueError(
                f"init_span_fraction must be in (0, 1), got {self.init_span_fraction}"
            )


class ExpireSpanPolicy(nn.Module):
    """Learned soft expiry with a hard capacity. See the module docstring."""

    name = "expire_span"

    def __init__(self, cfg: ExpireSpanConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        # 🔴 Zero / constant init: no draw from the global RNG (E0b), and every
        # gestalt starts with one span -- FIFO at initialisation.
        self.weight = nn.Parameter(torch.zeros(d_model))
        frac = cfg.init_span_fraction
        self.bias = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac))))
        self._gen = torch.Generator().manual_seed(cfg.seed)
        self.victims: list[int] = []
        """Per-stream eviction log, cleared by `reset`. Instrumentation only."""

    # ------------------------------------------------------------------ #
    # The span, the remaining span, the mask
    # ------------------------------------------------------------------ #

    def spans(self, gestalts: Tensor) -> Tensor:
        """`e = L · sigmoid(h · w + b)`, over the last axis of `gestalts`.

        The caller decides what `gestalts` carries; `_span_input` applies the
        gradient switch."""
        z = gestalts.to(self.weight.dtype) @ self.weight + self.bias
        return self.cfg.max_span * torch.sigmoid(z)

    def mask_from_remaining(self, remaining: Tensor) -> Tensor:
        """`m = clamp(1 + r / R, 0, 1)`."""
        return (1.0 + remaining / self.cfg.ramp).clamp(0.0, 1.0)

    def _span_input(self, kv: Tensor) -> Tensor:
        if self.cfg.grad_path == "through_gestalt":
            return kv
        # `predictor` (and `none`, which additionally runs under no_grad): the
        # gestalt is a constant to the predictor, so no span gradient reaches the
        # transformer or `W_sent`.
        return kv.detach()

    def memory_weight(
        self,
        mem_kv: Tensor,
        mem_valid: Tensor,
        mem_step: Tensor,
        step: int,
        *,
        training: bool,
    ) -> tuple[Tensor, Tensor]:
        """The `[B, M]` mask for step `step`'s cross-attention, and the span loss.

        `mem_kv`, `mem_valid`, `mem_step` are the memory the forward pass is about
        to attend over (pre-write). `training` is the **model's** mode, passed by
        the loop -- the policy's own `nn.Module.training` flag is not consulted,
        so `model.eval()` cannot leave structured dropout on.

        The span loss charges `alpha · e_i` for each gestalt admitted at step
        `step - 1` -- once per gestalt, at the first step it can be read --
        summed over slots and averaged over rows.
        """
        if self.cfg.grad_path == "none":
            with torch.no_grad():
                return self._weight(mem_kv, mem_valid, mem_step, step, training)
        return self._weight(mem_kv, mem_valid, mem_step, step, training)

    def _weight(self, mem_kv, mem_valid, mem_step, step, training):
        e = self.spans(self._span_input(mem_kv))  # [B, M]
        age = (step - mem_step).to(e.dtype)
        m = self.mask_from_remaining(e - age)
        live = mem_valid.to(e.dtype)
        weight = m * live
        if training and self.cfg.dropout > 0.0:
            keep = torch.rand(weight.shape, generator=self._gen) >= self.cfg.dropout
            weight = weight * keep.to(device=weight.device, dtype=weight.dtype)
        new = ((mem_step == step - 1) & mem_valid).to(e.dtype)
        aux = self.cfg.loss_coef * (e * new).sum() / mem_kv.shape[0]
        return weight, aux

    # ------------------------------------------------------------------ #
    # RetentionPolicy
    # ------------------------------------------------------------------ #

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        """Evict the live slot with the least remaining span `e_i - age_i`.

        Ties go to the lowest index, which in TG's oldest-first prefix is the
        oldest -- so equal spans (the zero-weight init) evict exactly FIFO."""
        with torch.no_grad():
            e = self.spans(slots.gestalts.detach().to(self.weight.device))
            age = (step - slots.written_at.to(e.device)).to(e.dtype)
            r = e - age
            r = torch.where(slots.live.to(e.device), r, torch.full_like(r, math.inf))
            victim = int(r.argmin().item())
        self.victims.append(victim)
        return victim

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        """No-op: Expire-Span learns from the LM loss, not from realised attention."""
        return None

    def on_write(self, slots: MemoryState, slot: int, step: int) -> None:
        """No-op, and genuinely so (gauntlet 0.4): the span is recomputed from the
        occupant's own gestalt and the age from `written_at`, so a new occupant
        cannot inherit the previous tenant's span. See the module docstring for
        why a cache keyed on `slot` would be wrong."""
        return None

    def reset(self) -> None:
        """Stream boundary. Spans are stateless; only the instrumentation log clears.
        The dropout generator keeps advancing, or every stream would share masks."""
        self.victims.clear()

    # ------------------------------------------------------------------ #
    # Checkpointing: the dropout stream survives a resume
    # ------------------------------------------------------------------ #

    def get_extra_state(self) -> Any:
        return {"generator": self._gen.get_state()}

    def set_extra_state(self, state: Any) -> None:
        self._gen.set_state(state["generator"])
