"""Baselines not yet implemented, as typed stubs that raise.

They are here in Sprint 1 so the ``RetentionPolicy`` protocol is exercised by every arm from the
start and no arm gets a bespoke code path later. Each stub carries what its implementation must do,
including the corrections in ``docs/spec-corrections.md`` -- so the constraint is read at the moment
someone starts writing the body, not months later in review.

Sprint 3 (week 3) implements H2O, Expire-Span and the leading-edge strategy.
"""

from __future__ import annotations

from torch import Tensor

from rsr.retention.policy import AttentionTrace, MemoryState

__all__ = ["ExpireSpanPolicy", "H2OPolicy", "LeadingEdgePolicy", "OraclePolicy"]


class _Stub:
    name = "stub"
    _todo = "unspecified"

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        raise NotImplementedError(f"{self.name}: {self._todo}")

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        raise NotImplementedError(f"{self.name}: {self._todo}")

    def reset(self) -> None:
        raise NotImplementedError(f"{self.name}: {self._todo}")


class H2OPolicy(_Stub):
    """Heavy-Hitter Oracle [P6] -- **the bar**. RSR's delta is predicted-future vs accumulated-past.

    Implementation constraints, all load-bearing:

    * **Both halves, at the 50/50 split of M.** Verified against [P6]: its own notation is
      ``H2O-256-256`` = "256 Heavy-Hitters and 256 local tokens", and its reference pseudocode
      returns "K heavy hitters and K recent tokens". A heavy-hitter-only H2O is a crippled
      comparator -- [P6] Table 9 measures the damage at **2.85%–22.75%**. State the split in the
      paper.
    * **The accumulated score is age-biased, and [P6] says so.** App. B.2: accumulated attention
      "can lead to a potential bias favoring the least recent tokens ... because most previous
      tokens have a higher number of attention scores." They tried an averaged score instead and
      it **degraded performance**. Do not silently "fix" H2O to an average -- that is a
      different method, and a weaker one by its authors' own measurement.
    * **Report age partial-ρ for this arm too** (correction S-2). H2O's bias runs *anti*-recency
      while FIFO/LRU run pro-recency, so E2's vacuity number is uninterpretable without it.
    * **This is a regime crossing.** [P6] is an inference-time KV-cache method scoring at each
      decoding step. Here it becomes a training-time policy over gestalt slots that retain their
      computation graphs. That has no analogue in the source and belongs in §13.
    """

    name = "h2o"
    _todo = "Sprint 3. Both halves at 50/50; see docs/spec-corrections.md S-1, S-2, S-6."


class ExpireSpanPolicy(_Stub):
    """Learned soft expiry [P7] -- **the referendum, not a comparison**.

    If this matches or beats RSR on synthetic, falsifier 6 fires: the reward proxy, the MC
    machinery, the shadow buffer and the warmup all exist to work around a non-differentiability a
    soft-expiry formulation does not have. **Stop and write a different thesis** (§6, D-4). There is
    no demotion path.

    🔴 **It must be tuned, or the referendum is rigged** (correction B-3). [P7] carries a ramp
    length ``R`` and a loss coefficient ``α``, and *requires* structured dropout -- without
    regularising memory size during training it overfits easily. The spec runs it once, untuned,
    against RSR's 9-config γ×β grid plus a ν sweep. An untuned Expire-Span losing is
    uninformative; a tuned one winning ends the project. **Equal search budget per arm, reported
    per arm.**
    """

    name = "expire_span"
    _todo = "Sprint 3. MUST be tuned under an equal-budget protocol; see spec-corrections B-3."


class LeadingEdgePolicy(_Stub):
    """Kintsch & van Dijk 1978 [P11] -- **the named ancestor**, and §2's whole framing.

    Keep the most recent propositions plus those highest in the macrostructure; discard the rest,
    with reinstatement search when a needed proposition has been dropped. RSR is a learned version
    of it, and §11 requires it as an **implemented baseline, not a citation**.

    ⚠️ **It is not an afternoon.** §5.4 and §11 both price it that way. It needs propositionalized
    text and a macrostructure over both PG-19 and the synthetic corpus. Budget accordingly and keep
    the declared fallback -- a sentence-level importance heuristic, **labelled as such in the
    paper**, never passed off as the 1978 rule.

    ⚠️ [P11] is still **unverified** against the primary source (``docs/citation-audit.md``). It is
    the highest-value remaining check precisely because the implementation depends on reading it.
    """

    name = "leading_edge"
    _todo = "Sprint 3. Read [P11] first -- it is unverified and this baseline IS the reading."


class OraclePolicy(_Stub):
    """Upper bound. Exact on synthetic; offline on PG-19 via the shadow-buffer machinery (§3.4).

    Built once, used three times: E0d's LOO validation, E-feas's headroom probe, and the shadow
    buffer's counterfactual targets. **Run it first on every corpus (E-feas)** -- if oracle ≈ FIFO,
    that corpus cannot exhibit the effect. §5.4: "One day, saves six weeks."
    """

    name = "oracle"
    _todo = "Sprint 2-3. Shares the LOO harness built for E0d."
