"""B0-CF: the counterfactual demand tensor Delta, in D.pt's FIFO world.

One function, ``cf_delta``, shared by the feasibility probe (``probe.py``) and the run
(``run.py``). It never trains, never calls ``rsr.constants.record()``.

For a batch of documents rolled out under FIFO (``M`` slots; slot 0 is the oldest,
``rsr.model.tg.policy_loop.write_at`` with victim 0), it returns ``Delta[b, t, i]``
for every sentence ``i < t`` -- step-major ``[S, S]`` per document, NaN at
``i >= t``, the same layout and the same world as lookahead-room's ``D.pt``
(``experiments/lookahead-room/run.py::FifoProbeRecorder``):

* **resident** ``i`` (``t - M <= i < t``, or every ``i < t`` while memory is
  underfull): ``rsr.metrics.loo.loo_delta_loss(mode="resample")``'s ``delta`` at
  the slot holding ``i`` -- the increase of sentence ``t``'s mean real-target NLL
  when ``i``'s content is replaced by a same-kind, same-rank, same-step donor from
  another document of the batch. Imported, never re-implemented. E0d's instrument.
* **evicted** ``i`` (``i < t - M``, memory full): the decrease of the same loss when
  ``i``'s gestalt is reinstated in slot 0, displacing FIFO's oldest resident
  ``t - M`` -- exactly the memory D.pt's probe reads (lookahead-room ``run.py``
  ``kvP[p, 0] = self.gest[i]``): ``L(FIFO memory) - L(slot 0 := gest_i)``. Both
  losses come from one batched forward; the FIFO-memory row is the identity probe
  (slot 0 := gest_{t-M}), checked against the live loss.

Sign: positive = the sentence helps sentence ``t``, in both kinds.

A resident cell with no resample donor (``pick_donor`` returns None; loo's NaN) is
**imputed 0** and flagged in ``missing`` (PREREG).

``delta_shift`` is the pre-registered sensitivity variant (DIAGNOSTIC, never in the
class): evicted cells get ``+ delta_resident[t, slot 0]``, i.e.
``L(slot 0 := donor of t - M) - L(slot 0 := gest_i)`` -- the evicted reading
re-baselined from "FIFO's oldest" to "a same-rank donor", as a resident reading is.
"""

from __future__ import annotations

import time

import torch

from rsr.metrics import loo
from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import write_at
from rsr.train.loop import lm_token_losses


def _sent_loss(logits, ids_t, mask_t) -> torch.Tensor:
    """Sentence ``t``'s mean real-target NLL per row (loo_delta_loss's ``sent_loss``)."""
    B, L = ids_t.shape
    per = lm_token_losses(logits, ids_t).view(B, L - 1).double()
    real = mask_t[:, 1:]
    n = real.sum(-1)
    mean = (per * real).sum(-1) / n.clamp(min=1)
    return torch.where(n > 0, mean, float("nan")).cpu()


def _repeat_rows(x: torch.Tensor, P: int) -> torch.Tensor:
    """``[B, ...] -> [B * P, ...]``, each row repeated ``P`` times consecutively."""
    B = x.shape[0]
    return x.unsqueeze(1).expand(B, P, *x.shape[1:]).reshape(B * P, *x.shape[1:])


@torch.no_grad()
def fifo_probe_losses(model, ids, mask, *, timing: dict | None = None) -> dict:
    """The FIFO live rollout (loo_delta_loss's own loop, re-run to keep the gestalts)
    and, at every full-memory step ``t``, one batched forward over every row ``b`` x
    candidate ``[t - M, 0, 1, ..., t - M - 1]`` with slot 0 := the candidate's gestalt.

    -> ``live_loss`` ``[B, S]``, ``slot_sentence`` ``[B, S, M]``, ``probe`` -- a dict
    ``t -> [B, P]`` of losses, column 0 the identity probe, column ``1 + i`` sentence
    ``i``'s reinstatement."""
    cfg = model.cfg
    was = model.training
    model.eval()
    B, S, _L = ids.shape
    M = cfg.M
    dev = ids.device
    dtype = model.embed.weight.dtype
    live_loss = torch.full((B, S), float("nan"), dtype=torch.float64)
    slot_sentence = torch.full((B, S, M), -1, dtype=torch.long)
    gest = torch.zeros(B, S, cfg.D, dtype=dtype, device=dev)
    probe: dict[int, torch.Tensor] = {}
    n_probe_forwards = n_probe_rows = 0
    t_probe = 0.0
    try:
        mem = init_memory(B, cfg, device=dev, dtype=dtype)
        bos_ctx = torch.zeros(B, cfg.D, device=dev, dtype=dtype)
        bos_valid = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
            live_loss[:, t] = _sent_loss(out.logits, ids_t, mask_t)
            step_c, valid_c = mem.step.cpu(), mem.valid.cpu()
            slot_sentence[:, t] = torch.where(valid_c, step_c, -1)
            if bool(valid_c.all()):
                t1 = time.perf_counter()
                cand = [t - M, *range(t - M)]
                P = len(cand)
                kv = mem.kv.unsqueeze(1).expand(B, P, M, cfg.D).clone()
                kv[:, :, 0] = gest[:, cand]
                rows = B * P

                def rep(x, P=P):
                    return _repeat_rows(x, P)

                o = model(
                    rep(ids_t),
                    rep(mask_t),
                    kv.reshape(rows, M, cfg.D),
                    rep(mem.valid),
                    rep(bos_ctx),
                    rep(bos_valid),
                )
                probe[t] = _sent_loss(o.logits, rep(ids_t), rep(mask_t)).view(B, P)
                n_probe_forwards += 1
                n_probe_rows += rows
                t_probe += time.perf_counter() - t1
            gest[:, t] = out.srep
            write = out.has_eos
            victim = torch.zeros(B, dtype=torch.long, device=dev)  # FIFO
            mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (t + 1 < S)
    finally:
        model.train(was)
    if timing is not None:
        timing["probe_forwards"] = n_probe_forwards
        timing["probe_rows"] = n_probe_rows
        timing["probe_seconds"] = t_probe
    return {"live_loss": live_loss, "slot_sentence": slot_sentence, "probe": probe}


def assemble(
    delta_res: torch.Tensor,
    slot_sentence: torch.Tensor,
    probe: dict[int, torch.Tensor],
    m: int,
) -> dict:
    """``delta_res`` ``[B, S, M]`` (loo, by rank), ``slot_sentence`` ``[B, S, M]``,
    ``probe`` ``t -> [B, P]`` -> ``Delta`` / ``delta_shift`` ``[B, S, S]`` step-major,
    NaN at ``i >= t``; ``missing`` / ``evicted`` ``[B, S, S]`` bool."""
    B, S, M = delta_res.shape
    if m != M:
        raise ValueError(f"delta_res has {M} ranks, M = {m}")
    nan = float("nan")
    Delta = torch.full((B, S, S), nan, dtype=torch.float64)
    missing = torch.zeros(B, S, S, dtype=torch.bool)
    evicted = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for t in range(S):
            for j in range(M):
                s = int(slot_sentence[b, t, j])
                if s < 0:
                    continue
                v = float(delta_res[b, t, j])
                if v != v:
                    missing[b, t, s] = True
                    v = 0.0
                Delta[b, t, s] = v
    shift = Delta.clone()
    for t, pl in probe.items():
        base = pl[:, 0]
        for i in range(t - m):
            ev = base - pl[:, 1 + i]
            Delta[:, t, i] = ev
            evicted[:, t, i] = True
            r0 = torch.nan_to_num(delta_res[:, t, 0], nan=0.0)
            shift[:, t, i] = ev + r0
    return {"Delta": Delta, "delta_shift": shift, "missing": missing, "evicted": evicted}


@torch.no_grad()
def cf_delta(model, docs, ids, mask, *, seed: int, timing: dict | None = None) -> dict:
    """The whole measurement for one batch of documents (module docstring)."""
    tm = {} if timing is None else timing
    t0 = time.perf_counter()
    lo = loo.loo_delta_loss(model, docs, ids, mask, mode="resample", seed=seed)
    tm["loo_seconds"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    fp = fifo_probe_losses(model, ids, mask, timing=tm)
    tm["fifo_probe_seconds"] = time.perf_counter() - t1
    M = model.cfg.M
    out = assemble(lo["delta"], fp["slot_sentence"], fp["probe"], M)
    live_eq = bool(
        torch.equal(
            torch.nan_to_num(lo["live_loss"], nan=-1.0),
            torch.nan_to_num(fp["live_loss"], nan=-1.0),
        )
    )
    slots_eq = bool(torch.equal(lo["slot_sentence"], fp["slot_sentence"]))
    ident = 0.0
    for t, pl in fp["probe"].items():
        ident = max(ident, float((pl[:, 0] - fp["live_loss"][:, t]).abs().max()))
    B, S, _ = lo["delta"].shape
    fifo_ok = all(
        fp["slot_sentence"][b, t].tolist() == list(range(t - M, t))
        for b in range(B)
        for t in range(M, S)
    )
    occupied = fp["slot_sentence"] >= 0
    out.update(
        {
            "live_loss": fp["live_loss"],
            "slot_sentence": fp["slot_sentence"],
            "delta_res": lo["delta"],
            "checks": {
                "live_loss_equal_loo": live_eq,
                "slot_sentence_equal_loo": slots_eq,
                "fifo_memory_ok": fifo_ok,
                "identity_worst": ident,
                "n_resident_cells": int(occupied.sum()),
                "n_missing_donor": int(out["missing"].sum()),
                "n_evicted_cells": int(out["evicted"].sum()),
            },
        }
    )
    tm["total_seconds"] = time.perf_counter() - t0
    return out
