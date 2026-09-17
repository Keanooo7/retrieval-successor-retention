"""The RSR policy: evict the slot with the lowest predicted future retrieval demand.

    i* = argmin_i [ z(psi-hat(s_i, c_t)) + b_i - nu * max_{j != i} cos(s_i, s_j) ]

subject to the hard ``A_max`` bound (§3.6a), and FIFO before ``T_warm`` (§3.4).

Sprint 1 implements the **eviction rule** and the reduction. The learning half -- realized ``r_i``,
the Monte-Carlo return ``G_i(t)``, the shadow buffer, the bias loop's update -- is Sprint 2 and
raises until then, so that a half-built policy cannot quietly be used as a whole one.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from rsr.retention.config import RetentionConfig
from rsr.retention.policy import AttentionTrace, EvictionDecision, MemoryState
from rsr.retention.value_head import BilinearValueHead, zscore_live

__all__ = ["RSRPolicy"]


class RSRPolicy:
    name = "rsr"

    def __init__(
        self,
        config: RetentionConfig,
        head: BilinearValueHead | None = None,
    ) -> None:
        if config.psi_source == "head" and head is None:
            raise ValueError("psi_source='head' needs a BilinearValueHead")
        self.config = config
        self.head = head
        self._bias: dict[int, float] = {}
        self.decisions: list[EvictionDecision] = []

    # -- scoring ---------------------------------------------------------------------------------

    def _psi(self, slots: MemoryState, context: Tensor) -> Tensor:
        if self.config.psi_source == "negative_age":
            # §3.7: psi-hat = -a_i. argmin(-a) = argmax(a) = the oldest slot = stock TG.
            return -slots.ages.to(dtype=torch.get_default_dtype())
        assert self.head is not None
        return self.head(slots.gestalts, context)

    def _redundancy(self, slots: MemoryState) -> Tensor:
        """``max_{j != i} cos(s_i, s_j)``. O(M^2 d), negligible at M <= 40."""
        if slots.n_live < 2:
            return torch.zeros(slots.n_live, device=slots.gestalts.device)
        g = torch.nn.functional.normalize(slots.gestalts, dim=-1)
        sim = g @ g.T
        sim.fill_diagonal_(-float("inf"))
        return sim.max(dim=-1).values

    def score(self, slots: MemoryState, context: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        psi = self._psi(slots, context)
        z = zscore_live(psi) if self.config.zscore else psi

        terms: dict[str, Tensor] = {"psi": psi, "z_psi": z}
        total = z

        if self.config.use_bias:
            b = torch.tensor(
                [self._bias.get(int(sid), 0.0) for sid in slots.slot_ids],
                dtype=total.dtype,
                device=total.device,
            )
            terms["b"] = b
            total = total + b

        if self.config.nu != 0.0:
            red = self._redundancy(slots)
            terms["redundancy"] = red
            total = total - self.config.nu * red

        terms["total"] = total
        return total, terms

    # -- the protocol ----------------------------------------------------------------------------

    def select_eviction(self, slots: MemoryState, context: Tensor, step: int) -> int:
        # §3.6a: the A_max bound is hard and takes precedence over the score. A slot at the bound
        # goes regardless of how valuable it looks.
        if self.config.a_max is not None:
            over = (slots.ages >= self.config.a_max).nonzero(as_tuple=True)[0]
            if over.numel() > 0:
                idx = int(over[torch.argmax(slots.ages[over])])
                self.decisions.append(
                    EvictionDecision(slot_index=idx, step=step, terms={"a_max_forced": 1.0})
                )
                return idx

        # §3.4: FIFO until T_warm. psi-hat still trains passively; the policy is inert.
        if step < self.config.t_warm:
            idx = int(torch.argmax(slots.ages))
            self.decisions.append(
                EvictionDecision(slot_index=idx, step=step, terms={"warmup_fifo": 1.0})
            )
            return idx

        total, terms = self.score(slots, context)
        idx = int(torch.argmin(total))

        # Detach before logging. The decision log outlives the step, and `total` carries the value
        # head's graph -- keeping a reference would pin one graph per eviction for the length of a
        # stream. In TG that is not a small leak: retained computation graphs ARE the memory bill
        # ([P2] App. A), and this project's whole activation budget is set by stream length.
        terms = {k: v.detach() for k, v in terms.items()}
        total = total.detach()

        # §3.4's required decision attribution, from the FIRST policy run, not added later.
        value_only = int(torch.argmin(terms["z_psi"]))
        bias_only = terms["z_psi"] + terms.get("b", torch.zeros_like(total))
        self.decisions.append(
            EvictionDecision(
                slot_index=idx,
                step=step,
                score=float(total[idx]),
                value_only_choice=value_only,
                bias_flipped=int(torch.argmin(bias_only)) != value_only,
                redundancy_flipped=idx != int(torch.argmin(bias_only)),
                terms={k: float(v[idx]) for k, v in terms.items()},
            )
        )
        return idx

    def observe(self, slots: MemoryState, attn: AttentionTrace, step: int) -> None:
        if self.config.is_reduction:
            # Under §3.7 the policy learns nothing and holds no state, so there is nothing to do
            # and nothing that could perturb the run. This is what makes the reduction exact.
            return None
        raise NotImplementedError(
            "RSRPolicy.observe is Sprint 2: realized r_i, the Monte-Carlo return G_i(t), the "
            "shadow buffer (K=64/40, NOT M), and the gradient-free bias loop with gamma_b derived "
            "from E0e's measured lifetime. See docs/spec-corrections.md C-1, C-4."
        )

    def reset(self) -> None:
        # §3.5: b resets at stream boundaries, because [P2] resets memory (stop-gradient) there.
        # D-1's whole finding turns on this -- a b that could not reach O(b_max) within a stream
        # was operationally identical to b == 0.
        self._bias.clear()

    # -- attribution -----------------------------------------------------------------------------

    def attribution(self) -> dict[str, float]:
        """§3.4: what fraction of argmins did ``b`` and ``nu`` each flip?

        "**If ``b`` flips a large share, the balance controller is the policy**" -- the v0.2
        objection that z-scoring was introduced to prevent, reopened by D-1's fix. Report this in
        the paper, paired with A5's ``gamma_b`` sweep.
        """
        scored = [d for d in self.decisions if d.value_only_choice is not None]
        if not scored:
            return {"n": 0.0, "bias_flipped": math.nan, "redundancy_flipped": math.nan}
        n = len(scored)
        return {
            "n": float(n),
            "bias_flipped": sum(d.bias_flipped for d in scored) / n,
            "redundancy_flipped": sum(d.redundancy_flipped for d in scored) / n,
        }
