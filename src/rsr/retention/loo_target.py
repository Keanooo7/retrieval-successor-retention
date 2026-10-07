"""The on-policy resample-knockout LOO target (owner ruling C3, 2026-10-07).

`docs/owner/rulings/R-2026-10-07-c3-loo-target.md`: RSR run 2 regresses `psi_hat` on

    G^Delta_i(t) = sum_k gamma^k Delta_i(t+k)

where `Delta_i(t)` is E0d's resample-knockout leave-one-out delta loss
(`rsr.metrics.loo.loo_delta_loss(mode="resample")`, spec section 3.2.1's "LOO is
truth"), in place of run 1's gated, share-of-M rescaled `r_i`. Only the TARGET
changes: the MC return (`rsr.retention.mc`), the censoring at eviction, the loss
and the stop-gradient are run 1's.

## What "on-policy" means here (and why this is not `loo_delta_loss`)

`loo_delta_loss` runs its own FIFO stream. Run 2's `Delta` is measured in the
**policy's own memory**: at step `t` of `run_policy_loop`, over the PRE-write memory
the policy has kept, with the same `bos_ctx` / `bos_valid`, in eval mode under
`no_grad`, against the base loss of the SAME eval forward whose gestalt is `c_t`
for the decision and the regression (ADR-0009 L1(a), L2). `step_delta` is
`loo_delta_loss`'s per-step body applied to that memory, line for line; on a FIFO
memory the two agree bit for bit (`tests/test_loo_target.py`).

## Design choices (a)-(d), each settled here

(a) **Donors.** `pick_donor`'s rule, unchanged: a same-kind sentence held at the
    **same rank** (slot index in the oldest-first prefix, which is what `P^(sent)`
    indexes, ADR-0006) of a **different document's** memory **at the same step**,
    never carrying the displaced sentence's own fact key, chosen by a private
    generator seeded on (seed, doc, step, role). On-policy every row's memory is
    that row's RSR memory, so the donor pool is the other rows' *policy* memories at
    this step, and "same rank" no longer implies "same age" as it does under FIFO.
    The donor's rank -- the positional key it is read under -- is the same; its age
    may differ. `donor_age_gap` records `|age(donor) - age(own)|` so that is visible.
    The donor seed is `loo_seed(run_seed, k)`: fresh draws every optimizer step.
(b) **Missing donors.** `Delta` is NaN (`STATUS_NO_DONOR`); the policy imputes 0 and
    counts it by sentence kind. A sentence with no real target (padding) is NaN too
    (`STATUS_NO_TARGET`), imputed 0, counted separately. E0d's floor (C6:
    >= 0.80 of full-memory cells receive a donor) is applied by the run harness to
    the run's totals, exit 3 below it.
(c) **Underfull steps.** No rescale. Section 3.2.1's `share * |memory|/M` corrects
    a *share's* mechanical fill-level inflation (a slot that is one of three gets
    ~1/3 of the mass whatever it does). `Delta` is a loss difference, not a share:
    nothing is normalised across live slots, so that inflation does not exist. No
    down-weighting either (CLAUDE.md; ADR-0009 L9's unweighted sum-then-mean stays).
    The fill fraction is still reported.
(d) **Censoring.** Unchanged from run 1: the return is cut at eviction
    (`RSRPolicy.mc_targets` zeros where the slot is not live); the shadow buffer is
    off by ruling. The policy logs mean `G` by the slot's eventual age at eviction
    so the self-confirmation of censoring is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rsr.metrics.loo import _ROLE_E0D, _gen, knockout_kv
from rsr.retention.targets import (
    STATUS_EMPTY,
    STATUS_NO_DONOR,
    STATUS_NO_TARGET,
    STATUS_OK,
    TARGET_LOO_RESAMPLE,
    TARGET_R_I,
    TARGETS,
)
from rsr.train.loop import lm_token_losses

__all__ = [
    "STATUS_EMPTY",
    "STATUS_NO_DONOR",
    "STATUS_NO_TARGET",
    "STATUS_OK",
    "TARGETS",
    "TARGET_LOO_RESAMPLE",
    "TARGET_R_I",
    "LooStep",
    "LooTarget",
    "loo_seed",
    "pick_donor_fast",
    "step_delta",
]


def loo_seed(run_seed: int, k: int) -> int:
    """The donor seed for optimizer step `k` of a run seeded `run_seed`. `_gen`
    hashes the text of (seed, doc, t, role), so any injective map works; this one
    keeps `run_seed` readable in the decimal digits."""
    if not 0 <= int(k) < 10**7:
        raise ValueError(f"optimizer step {k} outside [0, 1e7)")
    return int(run_seed) * 10**7 + int(k)


def pick_donor_fast(
    kind: list[list[int]],
    fact_key: list[list[int]],
    doc_id: list[int],
    mem_step: list[list[int]],
    valid: list[list[bool]],
    *,
    row: int,
    rank: int,
    want_kind: int,
    exclude_keys: set[int],
    seed: int,
    t: int,
    role: int,
) -> tuple[int, int] | None:
    """`rsr.metrics.loo.pick_donor` on Python lists: the same candidates in the same
    order and the same seeded draw (`tests/test_loo_target.py` compares the two).
    The tensor version spends ~4 scalar tensor reads per candidate; on-policy it
    runs B * M times per sentence step, every optimizer step."""
    me = doc_id[row]
    cands: list[tuple[int, int]] = []
    for b in range(len(mem_step)):
        if b == row or doc_id[b] == me or not valid[b][rank]:
            continue
        s = mem_step[b][rank]
        if kind[b][s] != want_kind or fact_key[b][s] in exclude_keys:
            continue
        cands.append((b, s))
    if not cands:
        return None
    i = int(torch.randint(len(cands), (1,), generator=_gen(seed, me, t, role)))
    return cands[i]


@dataclass
class LooStep:
    """One sentence step's on-policy LOO, for the whole batch (CPU tensors).

    `delta` `[B, M]` float64: knockout minus live mean real-target NLL of sentence
    `t`; NaN where the slot is empty, has no donor, or the sentence has no target.
    `status` `[B, M]`: `STATUS_*`, -1 where empty. `kind` `[B, M]`: the `KIND` code
    of the sentence each slot holds, -1 where empty. `donor_row` / `donor_sentence`
    `[B, M]`: -1 unless resampled. `donor_age_gap` `[B, M]`: `|age(donor) -
    age(own)|` at the shared rank, -1 unless resampled (choice (a))."""

    delta: Tensor
    status: Tensor
    kind: Tensor
    donor_row: Tensor
    donor_sentence: Tensor
    donor_age_gap: Tensor
    seconds: float = 0.0


def _sent_loss(logits: Tensor, ids_t: Tensor, mask_t: Tensor) -> Tensor:
    """`loo_delta_loss`'s `sent_loss`, verbatim: mean real-target NLL per row,
    float64, NaN on a row with no real target; on CPU."""
    B, L = ids_t.shape
    per = lm_token_losses(logits, ids_t).view(B, L - 1).double()
    real = mask_t[:, 1:]
    n = real.sum(-1)
    mean = (per * real).sum(-1) / n.clamp(min=1)
    return torch.where(n > 0, mean, float("nan")).cpu()


@torch.no_grad()
def step_delta(
    model,
    ids_t: Tensor,
    mask_t: Tensor,
    mem,
    bos_ctx: Tensor,
    bos_valid: Tensor,
    base_logits: Tensor,
    ann: dict[str, Tensor],
    *,
    seed: int,
    t: int,
) -> LooStep:
    """`loo_delta_loss(mode="resample")`'s per-step body on `mem` (the PRE-write
    memory the step's forward attended over). `base_logits` are the live eval
    forward's -- the forward whose gestalt is `c_t` (ADR-0009 L2) -- so no second
    live pass is made. One knockout forward per occupied rank, every row at once,
    in eval mode; the caller's mode is restored."""
    B, M = mem.valid.shape
    nan = float("nan")
    base = _sent_loss(base_logits, ids_t, mask_t)
    step_c, valid_c = mem.step.cpu(), mem.valid.cpu()
    step_l, valid_l = step_c.tolist(), valid_c.tolist()
    kind_l, key_l = ann["kind"].tolist(), ann["fact_key"].tolist()
    doc_l = ann["doc_id"].tolist()
    delta = torch.full((B, M), nan, dtype=torch.float64)
    status = torch.full((B, M), STATUS_EMPTY, dtype=torch.long)
    kind = torch.full((B, M), -1, dtype=torch.long)
    donor_row = torch.full((B, M), -1, dtype=torch.long)
    donor_sentence = torch.full((B, M), -1, dtype=torch.long)
    donor_age_gap = torch.full((B, M), -1, dtype=torch.long)
    was = model.training
    model.eval()
    try:
        for i in range(M):
            slot = torch.where(valid_c[:, i], i, -1)
            rep = torch.zeros_like(mem.kv[:, 0])
            for b in range(B):
                if slot[b] < 0:
                    continue
                s = step_l[b][i]
                kind[b, i] = kind_l[b][s]
                own_key = key_l[b][s]
                d = pick_donor_fast(
                    kind_l,
                    key_l,
                    doc_l,
                    step_l,
                    valid_l,
                    row=b,
                    rank=i,
                    want_kind=kind_l[b][s],
                    exclude_keys={own_key} if own_key >= 0 else set(),
                    seed=seed,
                    t=t,
                    role=_ROLE_E0D + i,
                )
                if d is None:
                    slot[b] = -1
                    status[b, i] = STATUS_NO_DONOR
                    continue
                rep[b] = mem.kv[d[0], i]
                donor_row[b, i], donor_sentence[b, i] = d
                donor_age_gap[b, i] = abs(d[1] - s)  # same t: |age diff| = |write diff|
            if not bool((slot >= 0).any()):
                continue
            kv = knockout_kv(mem.kv, slot, mode="replace", replacement=rep)
            o = model(ids_t, mask_t, kv, mem.valid, bos_ctx, bos_valid)
            d_i = _sent_loss(o.logits, ids_t, mask_t) - base
            delta[:, i] = torch.where(slot >= 0, d_i, nan)
            ok = (slot >= 0) & ~torch.isnan(d_i)
            status[:, i] = torch.where(ok, STATUS_OK, status[:, i])
            status[:, i] = torch.where(
                (slot >= 0) & torch.isnan(d_i), STATUS_NO_TARGET, status[:, i]
            )
    finally:
        model.train(was)
    return LooStep(delta, status, kind, donor_row, donor_sentence, donor_age_gap)


@dataclass
class LooTarget:
    """What `run_policy_loop` needs to form `Delta` for one stream batch: the
    synthetic annotations of the batch's documents (`stream_annotations`, kinds and
    fact keys for donor choice) and the donor seed (`loo_seed`)."""

    ann: dict[str, Tensor]
    seed: int

    def __call__(self, model, ids_t, mask_t, mem, bos_ctx, bos_valid, base_logits, t):
        return step_delta(
            model,
            ids_t,
            mask_t,
            mem,
            bos_ctx,
            bos_valid,
            base_logits,
            self.ann,
            seed=self.seed,
            t=t,
        )
