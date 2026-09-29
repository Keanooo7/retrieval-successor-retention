"""Expire-Span -- learned soft expiry [P7] (spec section 5.4).

**The referendum, not a comparison** (defect D-4, falsifier 6). Soft differentiable
expiry trains retention by the LM loss directly and makes sections 3.2-3.4's entire
apparatus unnecessary. It is the one baseline whose *success invalidates the design
rather than the result*.

Runs at the **week-4 gate on synthetic** (release condition 3). The demotion path
was deleted.

## The mechanism, ported onto TG's slot memory

"P7:L<n>" is a line of `~/research-corpus/sources/memory-retention/
sukhbaatar-2021-expire-span.md` (arXiv:2105.06548v2); the line-by-line check is
`~/Documents/RSR-2026-09-29-day/reports/P7-verification.md`.

Each gestalt `h_i` gets one span **per cross-attention layer `l`** -- "done
independently for each layer" (P7:L81-82), "an EXPIRE-SPAN at each layer that is
shared amongst the heads" (P7:L355-356):

    e_li = L · sigmoid(w_l · h_i + b_l)            Eq. 3, P7:L259-265   (spans)
    r_li(t) = e_li - (t - written_at_i)             P7:L267-268          (remaining)
    m_li(t) = clamp(1 + r_li(t) / R, 0, 1)          Eq. 5, P7:L284       (mask)
    a'_i = m_li a_i / Σ_j m_lj a_j                  Eq. 4, P7:L272-279

The mask reaches layer `l`'s cross-attention through `CrossAttention`'s
`mem_weight`. Because `m` is piecewise linear in `e`, the LM loss has a gradient on
`w_l, b_l` wherever a slot is on the ramp: **that gradient is Expire-Span.**

**Span loss** (Eq. 7, P7:L330-335; timing §4.2 "Loss Computation", P7:L368-382):
`alpha · e_li` is charged at every step where `0 < m_li < 1` -- "at the same time as
negative gradients", not once at admission, which the paper reports "empirically
results in poor performance". Summed over layers and slots, averaged over rows.
Eq. 7's `/T` is absorbed: the loop's LM objective is a **sum** over the `T` sentence
steps, not a mean, so both sides of Eq. 7 are multiplied by `T` (verification 5b).

**Structured dropout** (§4.2 "Regularization", P7:L387-392; App. A.2 P7:L1055-1064):
"For each batch, we sample l~U(0,L) and set a_ti = 0 for all t-i > l only during
training." One cutoff per sentence step, shared by every row, layer and head. A
switch, not a rate: the paper's form has none.

**Not in [P7] -- the port's own choices** (ADR-0010):
TG's memory has a hard capacity `M`; [P7] has none and deletes by expiry only
(P7:L289-291). When memory is full the slot whose **longest** remaining span over
layers is least is evicted (q6) -- a slot any layer still reads is alive, as "a
memory cannot be removed if it is used by any of the heads" (P7:L353-355) is for
heads. Expired slots that capacity has not forced out keep their rank and get zero
attention (q5).

## What is deliberately NOT decided here

* **`grad_path`** -- whether the span gradient may reach the gestalt, and so the
  transformer and `W_sent` (`through_gestalt`), stops at the predictor
  (`predictor`), or is off (`none`). CLAUDE.md's stop-gradient rule governs RSR's
  `phi`; whether a baseline may do otherwise is an owner decision (q1). The config
  has **no default** for it, or for anything else.
* `L`, `R`, `alpha` and the initial span are unregistered in `src/rsr/constants.py`,
  and B-3 says the referendum must tune them on an equal budget. They come in
  through `ExpireSpanConfig`, never from a literal here.

## Gauntlet 0.4 -- why this policy has no per-slot state to reset

The span is recomputed each step from the occupant's own gestalt, and the age from
`written_at`, both of which the memory owns and `write_at` updates. A new occupant
therefore has its own span and age 0 **by construction**: there is no cached value
that could describe the previous tenant. `run_policy_loop` passes `on_write` the
victim index, which is not where the newcomer lands (`write_at` compacts), and is
0 for a row that was not full -- a cache keyed on that index would be wrong twice.

## The E0b RNG trap

Construction draws nothing: `w` starts at zero and `b` at `logit(init fraction)`,
so every gestalt starts with the same span in every layer and the policy **evicts
exactly FIFO at initialisation**. Structured dropout's cutoff comes from a private
CPU generator seeded by `cfg.seed`, so the global stream -- and with it data order
and model dropout -- is untouched, on CPU or CUDA alike.
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
  transformer and `W_sent` from every earlier step of the stream.

**Neither is established as [P7]-faithful.** From the verification report's
gradient-flow section (`P7-verification.md` §2): "The paper does not say. It never
mentions a stop-gradient or detach, and it never states whether ∂e_i/∂h_i is
propagated." The paper does say, of the mask, that memories are forgotten "in a
gradually differentiable way to retain end-to-end training with backpropagation"
(P7:L78-81), and names only `w` and `b` as trainable parameters (P7:L264). The
report's verdict is "UNVERIFIED from the paper text". The official code (context,
not the paper) detaches its hidden-state cache at every block boundary, so there
the span gradient reaches the hidden states only within the current block.
`predictor` is stricter than that; `through_gestalt` is looser, because TG keeps
the graph across the whole stream.
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
    """`alpha` on the span loss: `alpha · e_li` at every step where `0 < m_li < 1`,
    summed over layers and slots, averaged over rows (P7:L368-382)."""

    structured_dropout: bool
    """[P7] §4.2's structured dropout (P7:L387-392): each training step, sample
    `l ~ U(0, L)` and zero attention to every memory older than `l`. A switch,
    because the paper's form has no rate parameter."""

    grad_path: str
    """One of `GRAD_PATHS`. No default, deliberately (ADR-0010)."""

    init_span_fraction: float
    """Initial span as a fraction of `L`: `b = logit(fraction)`, `w = 0`. Must be
    below 0.5 so `b < 0`: [P7] "initialize[s] the bias term b with a negative
    value" (App. A.1, P7:L1013-1015). `w = 0` is not in the paper."""

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
        if not isinstance(self.structured_dropout, bool):
            raise ValueError(
                f"structured_dropout is a switch, got {self.structured_dropout!r}: "
                f"[P7]'s form (l ~ U(0, L) per step) has no rate"
            )
        if not 0.0 < self.init_span_fraction < 0.5:
            raise ValueError(
                f"init_span_fraction must be in (0, 0.5) so the bias starts negative "
                f"([P7] App. A.1), got {self.init_span_fraction}"
            )


class ExpireSpanPolicy(nn.Module):
    """Learned soft expiry with a hard capacity. See the module docstring."""

    name = "expire_span"

    def __init__(self, cfg: ExpireSpanConfig, d_model: int, n_layers: int) -> None:
        """`n_layers` is the number of cross-attention (`C`) blocks: one span
        predictor each (P7:L81-82). Six in TG."""
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.cfg = cfg
        self.n_layers = n_layers
        # 🔴 Zero / constant init: no draw from the global RNG (E0b), and every
        # gestalt starts with one span in every layer -- FIFO at initialisation.
        self.weight = nn.Parameter(torch.zeros(n_layers, d_model))
        frac = cfg.init_span_fraction
        self.bias = nn.Parameter(torch.full((n_layers,), math.log(frac / (1.0 - frac))))
        self._gen = torch.Generator().manual_seed(cfg.seed)
        self.victims: list[int] = []
        """Per-stream eviction log, cleared by `reset`. Instrumentation only."""

    # ------------------------------------------------------------------ #
    # The span, the remaining span, the mask
    # ------------------------------------------------------------------ #

    def spans(self, gestalts: Tensor) -> Tensor:
        """`e_l = L · sigmoid(h · w_l + b_l)`: `[..., d]` -> `[n_layers, ...]`.

        The caller decides what `gestalts` carries; `_span_input` applies the
        gradient switch."""
        z = gestalts.to(self.weight.dtype) @ self.weight.T + self.bias
        return self.cfg.max_span * torch.sigmoid(z).movedim(-1, 0)

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
        """The `[n_layers, B, M]` mask for step `step`'s cross-attention, and the
        span loss.

        `mem_kv`, `mem_valid`, `mem_step` are the memory the forward pass is about
        to attend over (pre-write). `training` is the **model's** mode, passed by
        the loop -- the policy's own `nn.Module.training` flag is not consulted,
        so `model.eval()` cannot leave structured dropout on.

        The span loss charges `alpha · e_li` for every live memory whose
        (pre-dropout) mask is on the ramp, `0 < m_li < 1` (P7:L368-382), summed
        over layers and slots and averaged over rows. The ramp condition selects;
        it is not differentiated.
        """
        if self.cfg.grad_path == "none":
            with torch.no_grad():
                return self._weight(mem_kv, mem_valid, mem_step, step, training)
        return self._weight(mem_kv, mem_valid, mem_step, step, training)

    def _weight(self, mem_kv, mem_valid, mem_step, step, training):
        e = self.spans(self._span_input(mem_kv))  # [n_layers, B, M]
        age = (step - mem_step).to(e.dtype)
        m = self.mask_from_remaining(e - age)
        live = mem_valid.to(e.dtype)
        weight = m * live
        on_ramp = ((m > 0.0) & (m < 1.0) & mem_valid).to(e.dtype).detach()
        aux = self.cfg.loss_coef * (e * on_ramp).sum() / mem_kv.shape[0]
        if training and self.cfg.structured_dropout:
            # CPU on purpose: the private generator is a CPU generator (E0b trap:
            # never the global stream), so the draw is identical on CPU and CUDA.
            cutoff = float(
                torch.rand((), generator=self._gen, device="cpu") * self.cfg.max_span
            )
            weight = weight * (age <= cutoff).to(weight.dtype)
        return weight, aux

    # ------------------------------------------------------------------ #
    # RetentionPolicy
    # ------------------------------------------------------------------ #

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        """Evict the live slot whose longest remaining span over layers,
        `max_l (e_li - age_i)`, is least. A port choice, not [P7] (ADR-0010 q6).

        Ties go to the lowest index, which in TG's oldest-first prefix is the
        oldest -- so equal spans (the zero-weight init) evict exactly FIFO. The
        undropped span decides: structured dropout shapes attention, never who is
        evicted."""
        with torch.no_grad():
            e = self.spans(slots.gestalts.detach().to(self.weight.device))
            age = (step - slots.written_at.to(e.device)).to(e.dtype)
            r = (e - age).amax(dim=0)
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
        The dropout generator keeps advancing, or every stream would share cutoffs."""
        self.victims.clear()

    # ------------------------------------------------------------------ #
    # Checkpointing: the dropout stream survives a resume
    # ------------------------------------------------------------------ #

    def get_extra_state(self) -> Any:
        return {"generator": self._gen.get_state()}

    def set_extra_state(self, state: Any) -> None:
        self._gen.set_state(state["generator"])
