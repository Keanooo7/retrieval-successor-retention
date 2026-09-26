"""Leave-one-out slot knockout -- the causal quantity (spec sections 3.2.1, 5.4).

Ablate slot `i`, measure the change in loss. This is **truth** where `r_i` is a
correlational proxy: "if they disagree, LOO is truth and `r_i` is a confound"
(section 3.2.1, E0d).

**This is the same computation as the oracle (section 5.4), so it is built once
and used several times** (ROADMAP Sprint 2: "build once, use three times"): E0d
validates `r_i` against it, E-feas probes oracle-vs-FIFO headroom with it, the
shadow buffer scores counterfactuals through it, and the retrieval-locality
readout (W5) asks whether a query's answer lives in the slot that holds its assert.
Kickoff T6: "Build the LOO harness as the shared oracle."

## One code path

Every knockout in this module is `knockout_kv(kv, slot, mode=...)`: for each row
`b`, slot `slot[b]` of `kv[b]` is replaced (zeros, or a donor gestalt), every other
element is passed through by `torch.where` and is bit-identical to the input, and
`slot[b] = -1` leaves row `b` untouched. The validity mask is **never** changed: a
knocked-out slot is still attended over, it just carries different content.

The intervention is **single-step**: at step `t` the forward pass that reads
sentence `t` attends over the knocked-out memory; the memory that step `t+1` sees,
and the bos-copy context, come from the live forward. So a delta is the effect of
the slot's content on sentence `t` alone, not on the trajectory after it.

## Two knockout kinds, and why both

* **zero** -- the slot's content becomes zeros. **Out of distribution**: the key
  becomes position-only (`0 + P^(sent)(rank)`, ADR-0006) and the value becomes the
  value projection's bias. Kept because it is what S0-03's `slots_zeroed` does to
  every slot, but labelled.
* **resample** -- the slot's content becomes the live gestalt of a **same-kind**
  sentence (assert / query / filler, `rsr.data.synthetic`) held at the **same
  rank** of a **different document**'s memory at the **same step** (so the same
  fill level). The input stays in-distribution. Donors come from the batch;
  selection is seeded per (seed, doc, step, role) and never touches the global RNG.

## Eval mode

Correction 20: `r_i` is collected in eval mode, because attention dropout zeroes
slots at random. LOO is `r_i`'s arbiter, so it is measured in the same mode:
every forward here runs under `model.eval()` and `torch.no_grad()`, and the
caller's mode is restored on exit.

Affordable only on a subsample; section 13 records that limit.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rsr.model.tg.model import init_memory
from rsr.model.tg.policy_loop import write_at
from rsr.train.loop import lm_token_losses

__all__ = [
    "CONDITIONS",
    "CTRL_NO_PENDING_ADJACENT",
    "CTRL_OWN_NOT_RESIDENT",
    "CTRL_PRESENT",
    "KIND",
    "KNOCKOUT_MODES",
    "knockout_kv",
    "loo_delta_loss",
    "loo_readout",
    "pick_donor",
    "stream_annotations",
]

KIND = {"filler": 0, "assert": 1, "query": 2}
"""Integer code per `rsr.data.synthetic.SentenceKind`, as `stream_annotations`
writes it."""

KNOCKOUT_MODES = ("zero", "replace")

CONDITIONS = ("live", "own_zero", "own_resample", "ctrl_resample", "all_slots_zeroed")
"""`loo_readout`'s conditions, **in the order its forward passes are made** at a
step that carries an answer target (the live forward first, then one per
knockout). `all_slots_zeroed` is single-step: every slot zeroed for the query's
forward only -- NOT S0-03's `slots_zeroed`, which zeroes the memory at every step
and so also changes the bos-copy context the query sees."""

CTRL_PRESENT = 0
CTRL_OWN_NOT_RESIDENT = 1
"""The own slot was evicted, so there is no rank to be adjacent to."""
CTRL_NO_PENDING_ADJACENT = 2
"""Neither rank `own +- 1` holds an assert that is still pending (queried after
`t`) and is not the queried fact itself."""

_ROLE_OWN, _ROLE_CTRL, _ROLE_CTRL_PICK = 0, 1, 2
_ROLE_E0D = 16  # + rank


# --------------------------------------------------------------------------- #
# the one code path
# --------------------------------------------------------------------------- #


def knockout_kv(
    kv: Tensor, slot: Tensor, *, mode: str, replacement: Tensor | None = None
) -> Tensor:
    """Knock out slot `slot[b]` of row `b` of `kv` `[B, M, D]`; `slot[b] = -1`
    leaves row `b` alone. `mode="zero"` writes zeros, `mode="replace"` writes
    `replacement[b]` (`[B, D]`). Returns a new tensor; `kv` is not modified, and
    every element outside the targets is passed through bit-exactly."""
    if mode not in KNOCKOUT_MODES:
        raise ValueError(f"mode={mode!r} not in {KNOCKOUT_MODES}")
    _, M, _ = kv.shape
    slot = slot.to(device=kv.device, dtype=torch.long)
    sel = torch.nn.functional.one_hot(slot.clamp(min=0), M).bool()
    sel = sel & (slot >= 0).unsqueeze(-1)  # [B, M]
    if mode == "zero":
        fill = torch.zeros_like(kv)
    else:
        if replacement is None:
            raise ValueError('mode="replace" needs a replacement [B, D]')
        fill = replacement.to(kv.dtype).unsqueeze(1).expand_as(kv)
    return torch.where(sel.unsqueeze(-1), fill, kv)


# --------------------------------------------------------------------------- #
# annotations and donors
# --------------------------------------------------------------------------- #


def _fact_key_strings(doc) -> dict[int, str]:
    """sentence index -> "entity predicate" for every assert and query."""
    out: dict[int, str] = {}
    for a, q in doc.pairs:
        qs, asrt = doc.sentences[q], doc.sentences[a]
        obj = qs.answer
        head = qs.text.split("?", 1)[0]
        if not head.startswith("What does ") or obj is None:
            raise ValueError(f"doc {doc.doc_id} sentence {q}: not a synthetic query")
        key = head[len("What does ") :]
        body = asrt.text.rstrip(".")
        if not body.endswith(" " + obj) or body[: -len(obj) - 1] != key:
            raise ValueError(
                f"doc {doc.doc_id}: assert {asrt.text!r} and query {qs.text!r} do "
                f"not share an 'entity predicate' key"
            )
        out[a] = out[q] = key
    return out


def stream_annotations(docs, *, steps: int) -> dict[str, Tensor]:
    """Per-sentence facts the knockout needs, from the synthetic `Document`s
    (section 5.3), all CPU:

    * `kind` `[n, steps]` -- `KIND` code;
    * `query_of` `[n, steps]` -- for an assert, the index of its query (may be
      `>= steps`); `-1` otherwise;
    * `fact_key` `[n, steps]` -- an integer per distinct "entity predicate" (the
      question a query asks), for asserts and queries; `-1` for filler. Two docs
      share a key iff they ask the same question;
    * `doc_id` `[n]`.
    """
    n = len(docs)
    kind = torch.zeros(n, steps, dtype=torch.long)
    query_of = torch.full((n, steps), -1, dtype=torch.long)
    per_doc = [_fact_key_strings(d) for d in docs]
    names = sorted({k for m in per_doc for k in m.values()})  # never a hash order
    code = {k: i for i, k in enumerate(names)}
    fact_key = torch.full((n, steps), -1, dtype=torch.long)
    for di, doc in enumerate(docs):
        for si, sent in enumerate(doc.sentences[:steps]):
            kind[di, si] = KIND[sent.kind]
            if si in per_doc[di]:
                fact_key[di, si] = code[per_doc[di][si]]
        for a, q in doc.pairs:
            if a < steps:
                query_of[di, a] = q
    doc_id = torch.tensor([d.doc_id for d in docs], dtype=torch.long)
    return {"kind": kind, "query_of": query_of, "fact_key": fact_key, "doc_id": doc_id}


def _gen(seed: int, doc_id: int, t: int, role: int) -> torch.Generator:
    """A private generator per choice, so a choice depends on (seed, doc, step,
    role) and on the candidate pool only -- never on the global RNG or on how
    many choices were made before it."""
    s = ((seed * 1_000_003 + doc_id) * 8_191 + t) * 131 + role
    return torch.Generator().manual_seed(s % (2**63))


def pick_donor(
    ann: dict[str, Tensor],
    mem_step: Tensor,
    valid: Tensor,
    *,
    row: int,
    rank: int,
    want_kind: int,
    exclude_keys: set[int],
    seed: int,
    t: int,
    role: int,
) -> tuple[int, int] | None:
    """A resample donor for slot `rank` of row `row`: a row `b'` of a **different
    document** whose memory (at the same step: `mem_step`/`valid` are `[B, M]`)
    holds, at the **same rank**, a sentence of kind `want_kind` whose fact key is
    not in `exclude_keys`. Returns `(b', sentence index)` or `None` -- never a
    relaxed match."""
    doc_id = ann["doc_id"]
    me = int(doc_id[row])
    cands: list[tuple[int, int]] = []
    for b in range(mem_step.shape[0]):
        if b == row or int(doc_id[b]) == me or not bool(valid[b, rank]):
            continue
        s = int(mem_step[b, rank])
        if int(ann["kind"][b, s]) != want_kind:
            continue
        if int(ann["fact_key"][b, s]) in exclude_keys:
            continue
        cands.append((b, s))
    if not cands:
        return None
    i = int(torch.randint(len(cands), (1,), generator=_gen(seed, me, t, role)))
    return cands[i]


# --------------------------------------------------------------------------- #
# scoring (S0-03's answer_readout, per target)
# --------------------------------------------------------------------------- #


def _score(logits: Tensor, ids_t: Tensor, am: Tensor, sym_ids: Tensor):
    """-> (nll, ok, nll16, brier16) at the answer targets `am` `[B, L-1]`, computed
    exactly as `experiments/s0-03-rewardable-corpus/run.py::answer_readout` does."""
    B, L = ids_t.shape
    per = lm_token_losses(logits, ids_t).view(B, L - 1)
    lg = logits[:, :-1][am]
    tgt = ids_t[:, 1:][am]
    lp16 = torch.log_softmax(lg[:, sym_ids], dim=-1)
    pos = (sym_ids.unsqueeze(0) == tgt.unsqueeze(1)).float().argmax(-1)
    nll16 = -lp16.gather(1, pos.unsqueeze(1)).squeeze(1)
    onehot = torch.nn.functional.one_hot(pos, len(sym_ids)).to(lp16.dtype)
    brier = (lp16.exp() - onehot).pow(2).sum(-1)
    return per[am], lg.argmax(-1) == tgt, nll16, brier, per


def _nan_where(x: Tensor, keep: Tensor) -> Tensor:
    x = x.double()
    return torch.where(keep, x, torch.full_like(x, float("nan")))


# --------------------------------------------------------------------------- #
# per-answer readout (W5)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def loo_readout(model, docs, ids, mask, tmask, gap, sym_ids, *, seed: int) -> dict:
    """Per answer target, the answer under `CONDITIONS` (FIFO, eval mode, one pass).

    Arguments are S0-03's (`encode` / `answer_targets` output, the 16 answer-symbol
    ids) plus the `Document`s they encode, for kinds and fact keys, and a `seed` for
    donor choice.

    Targets per query at step `t` in row `b`:

    * **own** -- the slot whose `mem.step == t - gap` (the query's assert).
      Evicted -> `own_resident = False`, `own_rank = -1`, own_* NaN.
    * **ctrl** -- a slot at rank `own_rank +- 1` holding an assert still pending
      at `t` (queried after `t`), not of the queried fact. Both adjacent ranks
      qualifying -> one is picked, seeded. None -> `ctrl_status` says why and
      ctrl_* is NaN; **no other slot is substituted**.

    Returns columnar records, one per answer target, ordered by `(t, row)` --
    exactly `answer_readout`'s order:

    * `row`, `doc_id`, `t`, `gap`;
    * `own_resident` (bool), `own_rank`, `own_sentence`, `own_donor_row`,
      `own_donor_sentence`;
    * `ctrl_status` (`CTRL_*`), `ctrl_rank`, `ctrl_sentence`, `ctrl_donor_row`,
      `ctrl_donor_sentence`;
    * per condition `c`: `c_ok` (1.0/0.0, NaN if not applied), `c_nll`, `c_nll16`,
      `c_brier16` (float64, NaN if not applied);
    * `real_token_nll` (live, all real targets -- `answer_readout`'s),
      `conditions`, `seed`.

    All indices are `-1` where absent. The live condition reproduces
    `answer_readout(cond="live")` bit-exactly (tests/test_loo.py).
    """
    cfg = model.cfg
    if not cfg.use_memory:
        raise ValueError(
            "loo_readout needs cfg.use_memory: there is no slot to knock out"
        )
    was = model.training
    model.eval()
    B, S, _ = ids.shape
    dev = ids.device
    dtype = model.embed.weight.dtype
    ann = stream_annotations(docs, steps=S)
    if ann["doc_id"].shape[0] != B:
        raise ValueError(f"{len(docs)} docs for a batch of {B}")
    cols: dict[str, list] = {}

    def put(k, v):
        cols.setdefault(k, []).append(v)

    real_sum, real_n = 0.0, 0
    try:
        mem = init_memory(B, cfg, device=dev, dtype=dtype)
        bos_ctx = torch.zeros(B, cfg.D, device=dev, dtype=dtype)
        bos_valid = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
            am = tmask[:, t, 1:]
            live = _score(out.logits, ids_t, am, sym_ids)
            per = live[4]
            real = mask_t[:, 1:]
            real_sum += float(per[real].double().sum())
            real_n += int(real.sum())
            if am.any():
                if int(am.sum(-1).max()) > 1:
                    raise ValueError(f"step {t}: more than one answer target in a row")
                _knockouts(
                    model,
                    out,
                    live,
                    t,
                    am,
                    ids_t,
                    mask_t,
                    mem,
                    bos_ctx,
                    bos_valid,
                    gap,
                    ann,
                    sym_ids,
                    seed,
                    put,
                )
            write = out.has_eos  # every row runs the full stream, as in train()
            victim = torch.zeros(B, dtype=torch.long, device=dev)  # FIFO
            mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (t + 1 < S)
    finally:
        model.train(was)
    if not cols:
        raise ValueError("no answer targets in the stream: nothing to read out")
    res = {k: torch.cat(v) for k, v in cols.items()}
    res["real_token_nll"] = real_sum / max(real_n, 1)
    res["conditions"] = CONDITIONS
    res["seed"] = seed
    return res


def _knockouts(
    model,
    out,
    live,
    t,
    am,
    ids_t,
    mask_t,
    mem,
    bos_ctx,
    bos_valid,
    gap,
    ann,
    sym_ids,
    seed,
    put,
):
    """One step's targets, their knockout forwards and records (see loo_readout)."""
    B, M = mem.valid.shape
    rows = am.any(-1).nonzero().flatten().tolist()  # ascending == am's order
    step_c, valid_c = mem.step.cpu(), mem.valid.cpu()
    kind, qof, key = ann["kind"], ann["query_of"], ann["fact_key"]
    minus = torch.full((B,), -1, dtype=torch.long)
    own_slot, own_rs, ctrl_rs = minus.clone(), minus.clone(), minus.clone()
    own_rep = torch.zeros(B, mem.kv.shape[-1], dtype=mem.kv.dtype, device=mem.kv.device)
    ctrl_rep = own_rep.clone()
    rec = {
        k: minus.clone()
        for k in (
            "own_rank",
            "own_sentence",
            "own_donor_row",
            "own_donor_sentence",
            "ctrl_rank",
            "ctrl_sentence",
            "ctrl_donor_row",
            "ctrl_donor_sentence",
        )
    }
    ctrl_status = torch.full((B,), CTRL_OWN_NOT_RESIDENT, dtype=torch.long)
    for b in rows:
        qkey = int(key[b, t])
        a = t - int(gap[b, t])
        hit = ((step_c[b] == a) & valid_c[b]).nonzero().flatten().tolist()
        if not hit:
            continue
        r = hit[0]
        own_slot[b] = r
        rec["own_rank"][b], rec["own_sentence"][b] = r, a
        d = pick_donor(
            ann,
            step_c,
            valid_c,
            row=b,
            rank=r,
            want_kind=int(kind[b, a]),
            exclude_keys={qkey},
            seed=seed,
            t=t,
            role=_ROLE_OWN,
        )
        if d is not None:
            own_rs[b] = r
            own_rep[b] = mem.kv[d[0], r]
            rec["own_donor_row"][b], rec["own_donor_sentence"][b] = d
        # control: an adjacent rank holding a still-pending assert (not this fact)
        adj = []
        for rk in (r - 1, r + 1):
            if not (0 <= rk < M and bool(valid_c[b, rk])):
                continue
            s = int(step_c[b, rk])
            pending = int(kind[b, s]) == KIND["assert"] and int(qof[b, s]) > t
            if pending and int(key[b, s]) != qkey:
                adj.append((rk, s))
        if not adj:
            ctrl_status[b] = CTRL_NO_PENDING_ADJACENT
            continue
        if len(adj) > 1:
            g = _gen(seed, int(ann["doc_id"][b]), t, _ROLE_CTRL_PICK)
            adj = [adj[int(torch.randint(len(adj), (1,), generator=g))]]
        rk, s = adj[0]
        ctrl_status[b] = CTRL_PRESENT
        rec["ctrl_rank"][b], rec["ctrl_sentence"][b] = rk, s
        d = pick_donor(
            ann,
            step_c,
            valid_c,
            row=b,
            rank=rk,
            want_kind=int(kind[b, s]),
            exclude_keys={qkey, int(key[b, s])},
            seed=seed,
            t=t,
            role=_ROLE_CTRL,
        )
        if d is not None:
            ctrl_rs[b] = rk
            ctrl_rep[b] = mem.kv[d[0], rk]
            rec["ctrl_donor_row"][b], rec["ctrl_donor_sentence"][b] = d

    kvs = {
        "own_zero": (knockout_kv(mem.kv, own_slot, mode="zero"), own_slot),
        "own_resample": (
            knockout_kv(mem.kv, own_rs, mode="replace", replacement=own_rep),
            own_rs,
        ),
        "ctrl_resample": (
            knockout_kv(mem.kv, ctrl_rs, mode="replace", replacement=ctrl_rep),
            ctrl_rs,
        ),
        "all_slots_zeroed": (torch.zeros_like(mem.kv), torch.zeros_like(own_slot)),
    }
    idx = torch.tensor(rows, dtype=torch.long)
    scores = {"live": (live, torch.ones(len(rows), dtype=torch.bool))}
    for cond in CONDITIONS[1:]:  # the documented call order
        kv, applied = kvs[cond]
        o = model(ids_t, mask_t, kv, mem.valid, bos_ctx, bos_valid)
        scores[cond] = (_score(o.logits, ids_t, am, sym_ids), applied[idx] >= 0)
    for cond, ((nll, ok, n16, b16, _), keep) in scores.items():
        keep = keep.to(nll.device)
        put(f"{cond}_nll", _nan_where(nll, keep).cpu())
        put(f"{cond}_ok", _nan_where(ok, keep).cpu())
        put(f"{cond}_nll16", _nan_where(n16, keep).cpu())
        put(f"{cond}_brier16", _nan_where(b16, keep).cpu())
    put("row", idx)
    put("doc_id", ann["doc_id"][idx])
    put("t", torch.full((len(rows),), t, dtype=torch.long))
    put("gap", gap[:, t].cpu()[idx])
    put("own_resident", own_slot[idx] >= 0)
    put("ctrl_status", ctrl_status[idx])
    for k, v in rec.items():
        put(k, v[idx])


# --------------------------------------------------------------------------- #
# per-slot delta next-sentence loss (E0d, the oracle)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def loo_delta_loss(
    model, docs, ids, mask, *, mode: str = "zero", seed: int = 0, steps=None
) -> dict:
    """Section 3.2.1's LOO: for each step `t` and each occupied rank `i`, the
    change in sentence `t`'s mean real-target NLL when slot `i` is knocked out
    (every row at once, one forward per rank), FIFO, eval mode.

    `mode` is `"zero"` or `"resample"` (same kind, different doc, same rank, same
    step -- `pick_donor` with no key exclusion; no donor -> NaN). `steps` limits
    which steps are knocked out (the live pass still runs every step).

    Returns `delta` `[B, S, M]` float64 (knockout minus live; NaN where the slot is
    empty, the step was not requested, or no donor), `live_loss` `[B, S]`,
    `slot_sentence` `[B, S, M]` (the sentence each slot holds, -1 empty), and
    `donor_row` / `donor_sentence` `[B, S, M]` (-1 unless resampled).
    """
    if mode not in ("zero", "resample"):
        raise ValueError(f'mode={mode!r}: "zero" or "resample"')
    cfg = model.cfg
    if not cfg.use_memory:
        raise ValueError("loo_delta_loss needs cfg.use_memory")
    was = model.training
    model.eval()
    B, S, L = ids.shape
    M = cfg.M
    dev = ids.device
    dtype = model.embed.weight.dtype
    ann = stream_annotations(docs, steps=S)
    want = set(range(S)) if steps is None else set(steps)
    nan = float("nan")
    delta = torch.full((B, S, M), nan, dtype=torch.float64)
    live_loss = torch.full((B, S), nan, dtype=torch.float64)
    slot_sentence = torch.full((B, S, M), -1, dtype=torch.long)
    donor_row = torch.full((B, S, M), -1, dtype=torch.long)
    donor_sentence = torch.full((B, S, M), -1, dtype=torch.long)

    def sent_loss(logits, ids_t, mask_t):
        per = lm_token_losses(logits, ids_t).view(B, L - 1).double()
        real = mask_t[:, 1:]
        return ((per * real).sum(-1) / real.sum(-1).clamp(min=1)).cpu()

    try:
        mem = init_memory(B, cfg, device=dev, dtype=dtype)
        bos_ctx = torch.zeros(B, cfg.D, device=dev, dtype=dtype)
        bos_valid = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            out = model(ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid)
            base = sent_loss(out.logits, ids_t, mask_t)
            live_loss[:, t] = base
            step_c, valid_c = mem.step.cpu(), mem.valid.cpu()
            slot_sentence[:, t] = torch.where(valid_c, step_c, -1)
            if t in want:
                for i in range(M):
                    slot = torch.where(valid_c[:, i], i, -1)
                    rep = None
                    if mode == "resample":
                        rep = torch.zeros_like(mem.kv[:, 0])
                        for b in range(B):
                            if slot[b] < 0:
                                continue
                            s = int(step_c[b, i])
                            d = pick_donor(
                                ann,
                                step_c,
                                valid_c,
                                row=b,
                                rank=i,
                                want_kind=int(ann["kind"][b, s]),
                                exclude_keys=set(),
                                seed=seed,
                                t=t,
                                role=_ROLE_E0D + i,
                            )
                            if d is None:
                                slot[b] = -1
                                continue
                            rep[b] = mem.kv[d[0], i]
                            donor_row[b, t, i], donor_sentence[b, t, i] = d
                    if not bool((slot >= 0).any()):
                        continue
                    kv = knockout_kv(
                        mem.kv,
                        slot,
                        mode="zero" if rep is None else "replace",
                        replacement=rep,
                    )
                    o = model(ids_t, mask_t, kv, mem.valid, bos_ctx, bos_valid)
                    d_i = sent_loss(o.logits, ids_t, mask_t) - base
                    delta[:, t, i] = torch.where(slot >= 0, d_i, nan)
            write = out.has_eos
            victim = torch.zeros(B, dtype=torch.long, device=dev)  # FIFO
            mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (t + 1 < S)
    finally:
        model.train(was)
    return {
        "delta": delta,
        "live_loss": live_loss,
        "slot_sentence": slot_sentence,
        "donor_row": donor_row,
        "donor_sentence": donor_sentence,
        "mode": mode,
        "seed": seed,
    }
