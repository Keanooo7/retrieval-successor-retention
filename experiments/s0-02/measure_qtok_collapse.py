"""ADR-0008's three numbers, produced by a process rather than typed into prose.

## Why this file exists

`docs/decisions/ADR-0008-qtok-collapse.md` states **`1.49e-7`**, **`2.98e-8`** and
**`0.0383`** as measured, and they are: `tests/test_capture_bridge.py` drives the
same fixture and asserts the same identities. But the tests assert **thresholds**
(`atol=1e-5`, `atol=1e-6`, `not allclose(atol=1e-3)`), not these values, and

    grep -c "1.49\\|2.98\\|0.0383" runs/s0-02-capture-bridge/ledger.json  ->  0

So on 2026-09-20 the ADR's numbers were prose with no producer, no artefact and no
ledger row -- while the ADR is the document a future cycle cites when it decides
whether `eos_only` is worth an E0d row. A threshold test is stronger than a
recorded value in one way (it fails when the property breaks, not when the third
decimal moves) and **no substitute in the way that matters**: nothing in the tree
could tell you what the divergence actually was.

This script is the producer. `experiments/s0-02/write_ledger.py` turns its output
into rows.

## The fixture is *imported*, not re-typed

ADR-0008 says *"every number in this ADR comes from that file's fixture (`D=32,
H=2, M=5`, `Q_tok=8` with a 3-token PAD tail, so `Q_real = 5`, row 0, 4 of 5 slots
live)"*. A producer that **copies** the fixture makes that a claim about two
blocks of code staying in step. This one imports `tests/test_capture_bridge.py`
and uses its `_cfg` / `_sentence` / `_memory` / `_capture` directly, so the
sentence is checkable rather than asserted -- and `_assert_fixture_shape` below
re-derives every parameter the ADR quotes and **raises** if the fixture has
drifted, so a changed test surfaces as a failed run and not as quietly different
numbers under the same ADR.

⚠️ The direction of that coupling is deliberate and one-way: `experiments/` reads
`tests/`. Nothing in `tests/` may import this file.

## Device

**CPU, and not configurable.** These are float32 identity checks at the `1e-7`
level; `1.49e-7` is a statement about float32 epsilon on a specific accumulation
order, and MPS has a different one. The throughput arms of this run are `mps` and
these rows are `cpu`; the ledger's single `device` field cannot say both, so every
row this produces carries `device=cpu` in its `how` string and
`qtok_collapse.device` records it outright.

## What this does NOT measure

- Not whether `sum` is the *right* collapse. That is E0d (Spearman rho against
  leave-one-out delta-loss), and ADR-0008 says so in "What would distinguish them".
- Not anything about a **trained** model. `_capture` builds a freshly initialised
  `TGModel`, so the `0.0383` sum-vs-EOS divergence is measured where there is no
  learned retrieval structure for the two collapses to disagree about. The ADR
  already states this; it is repeated here because a number in a ledger outlives
  the paragraph next to it.
- Not a seed sweep. **One fixture, one row, one seed** -- there is no spread to
  report and this file must not be read as though there were. `n_seeds: 1` is
  written into the artefact so the omission is a fact in the record rather than an
  absence in it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "tests"))

import test_capture_bridge as fx  # noqa: E402  (the fixture, imported not copied)
from rsr.model.tg.policy_loop import (  # noqa: E402
    Q_TOK_COLLAPSE,
    trace_for_row,
)
from rsr.retention.reward import contribution, retrieval_demand  # noqa: E402

#: The fixture ADR-0008's numbers come from, restated as an assertion. If `tests/`
#: drifts, this run fails loudly rather than filing different numbers under the
#: same ADR.
#:
#: 🔴 **Two entries here disagree with ADR-0008 as written, and the ADR is what is
#: wrong.** It says *"`Q_tok=8` with a 3-token PAD tail"*. `TGConfig.L` is
#: `1 + max_sentence_tokens + sentence_tail_len = 1 + 8 + 1 = **10**`
#: (`src/rsr/model/tg/config.py:128-130`), and `_sentence` sets `mask[0, 5:] = 0`,
#: which is a **5**-token PAD tail. The ADR read `max_sentence_tokens` as the
#: query-axis length and then derived the tail as `8 - 5`. `Q_real = 5` is right
#: either way, which is why the error was self-consistent enough to survive: the
#: numbers below are unaffected, the description of where they came from was not.
#: Corrected in the ADR in the same commit that added this file.
ADR_FIXTURE = {
    "D": 32,
    "H": 2,
    "Q_tok": 10,
    "pad_tail": 5,
    "q_real": 5,
    "M": 5,
    "row": 0,
    "n_live": 4,
    "n_cross_layers": 6,
}

ROW = ADR_FIXTURE["row"]

#: A second row, measured because ADR-0008's EOS-only table cannot bear the weight
#: it is given on row 0 alone. See `eos_only`'s docstring. Row 1 has a full mask,
#: so its `[EOS]` is a real query token; it has 2 live slots rather than 4.
SUPPLEMENTARY_ROW = 1
SUPPLEMENTARY_N_LIVE = 2


def _assert_fixture_shape(cfg, model, mem, mask, cap) -> dict:
    """Re-derive every parameter ADR-0008 names, and refuse if one has moved."""
    got = {
        "D": cfg.D,
        "H": cfg.H,
        "Q_tok": cfg.L,
        "pad_tail": int((mask[ROW] == 0).sum()),
        "q_real": int((mask[ROW] != 0).sum()),
        "M": cfg.max_sentences_in_short_term,
        "row": ROW,
        "n_live": int(mem.valid[ROW].sum()),
        "n_cross_layers": sum(1 for b in model.blocks if b.block_type == "C"),
    }
    if got != ADR_FIXTURE:
        raise SystemExit(
            f"tests/test_capture_bridge.py's fixture has drifted from the one "
            f"ADR-0008 quotes.\n  ADR:  {ADR_FIXTURE}\n  tree: {got}\n"
            f"Refusing to file numbers from a different fixture under the same "
            f"ADR. Either the ADR's fixture line or this dict is now wrong; "
            f"decide which, in the ADR, before rerunning."
        )
    if not cap.eval_mode:
        raise SystemExit("the capture is not an eval-mode capture (D-F)")
    if cap.alpha.shape[0] != ADR_FIXTURE["n_cross_layers"]:
        raise SystemExit("D-E: the profile must have one row per C block")
    return got


# --------------------------------------------------------------------------- #
# 1. the identity ADR-0008's "sum is algebra" argument rests on
# --------------------------------------------------------------------------- #


def increment_identity(model, mem, mask, out, cap) -> dict:
    """`sum_q increment_q[i]` vs `(sum_q alpha) . W_O v`, per cross layer.

    ADR-0008: *"`W_O^(l,h) v_{l,h,i}` does not depend on `q`"*, so the whole
    sentence's cross-attention increment factorises with the summed `alpha`
    pulled out. If it does not, summing over `q` is a **choice** and the ADR's
    central argument is gone -- which is why this is the first number.

    `Q_real * attn_out_proj.bias` is subtracted because the bias is added once per
    query position and is not part of `W_O v` (ADR-0008's second exclusion).
    """
    q_real = mask[ROW] != 0
    blocks = [b for b in model.blocks if b.block_type == "C"]
    per_layer = []
    for li, (block, att) in enumerate(zip(blocks, out.cross_attention, strict=True)):
        with torch.no_grad():
            v = block.cross_attn.value(mem.kv)
            o = block.cross_attn.attn_out_proj(torch.einsum("bhqm,bmhk->bqhk", att, v))
            lhs = o[ROW][q_real].sum(0) - q_real.sum() * (
                block.cross_attn.attn_out_proj.bias
            )
        rhs = torch.einsum("hm,hmd->d", cap.alpha[li, ROW], cap.wo_v[li, ROW])
        per_layer.append(
            {
                "layer": li,
                "max_abs_diff": float((lhs - rhs).abs().max()),
                "max_abs_value": float(lhs.abs().max()),
            }
        )
    return {
        "per_layer": per_layer,
        "max_abs_diff_over_layers": max(x["max_abs_diff"] for x in per_layer),
        "max_abs_value_over_layers": max(x["max_abs_value"] for x in per_layer),
        "n_layers": len(per_layer),
        "dtype": str(cap.wo_v.dtype),
        "what": (
            "reconstructed cross output summed over real query positions, less "
            "Q_real*bias, minus einsum('hm,hmd->d', alpha, wo_v); ADR-0008 "
            "'Verified numerically rather than asserted'"
        ),
    }


# --------------------------------------------------------------------------- #
# 2. mean vs sum -- the axis that cancels
# --------------------------------------------------------------------------- #


def mean_vs_sum(mem, mask, cap) -> dict:
    """`mean = sum / Q_real`, one scalar for every `(l, h, i)`, so it cancels in
    `share_i = raw_i / sum_j raw_j`. Measured, not argued."""
    n_real = int((mask[ROW] != 0).sum())
    tr = trace_for_row(cap, mem.valid, ROW, 0)
    tr_mean = dataclasses.replace(tr, alpha=tr.alpha / n_real)
    kw = {"n_live": ADR_FIXTURE["n_live"], "capacity": ADR_FIXTURE["M"]}
    r_sum = retrieval_demand(tr, **kw)
    r_mean = retrieval_demand(tr_mean, **kw)
    c_sum = contribution(tr)
    c_mean = contribution(tr_mean)
    return {
        "q_real": n_real,
        "r_i_sum_collapse": [float(x) for x in r_sum],
        "r_i_mean_collapse": [float(x) for x in r_mean],
        "max_abs_diff_r_i": float((r_sum - r_mean).abs().max()),
        "contribution_sum_collapse": [float(x) for x in c_sum],
        "contribution_mean_collapse": [float(x) for x in c_mean],
        "contribution_ratio": [
            float(a / b) for a, b in zip(c_sum, c_mean, strict=True) if float(b) != 0.0
        ],
        "what": (
            "ADR-0008 alternative 1. r_i is invariant; the unnormalised "
            "contribution() differs by exactly Q_real"
        ),
    }


# --------------------------------------------------------------------------- #
# 3. EOS-only -- the axis that does not cancel
# --------------------------------------------------------------------------- #


def _spearman_rho(a: list[float], b: list[float]) -> float | None:
    """Rho over `len(a)` points, no ties expected. Reported to make explicit that
    a rank agreement on ONE row is not evidence the two collapses rank alike.

    `None` below three points: rho over two points is `+1` or `-1` by arithmetic
    and carries no information, and a number that is always `1.0` is the shape a
    vacuous check has.
    """
    n = len(a)
    if n < 3:
        return None
    ra = [sorted(a).index(x) for x in a]
    rb = [sorted(b).index(x) for x in b]
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb, strict=True))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def eos_only(cfg, mem, ids, mask, out, cap, row: int, n_live: int) -> dict:
    """The genuinely different measurement: `alpha` read at the `[EOS]` row only.

    Not a branch in `cross_capture` -- `collapse="eos_only"` raises by design
    (ADR-0008: *"an unused branch that nobody runs is how a second specification
    survives"*). The reduction is done here, on the same captured tensor.

    🔴 **On row 0 -- the row every number in ADR-0008 comes from -- `[EOS]` is a
    PAD position.** `_sentence` writes `ids[:, -1] = eos_id` at index 9 and then
    masks `mask[0, 5:] = 0`, so row 0's EOS token is one of the five the decided
    collapse throws away. The two collapses therefore share **no query position at
    all** on that row, and `0.0383` is a divergence guaranteed by construction
    rather than one discovered in the attention. The ADR presents it as evidence
    that EOS-only "does not cancel"; it does not cancel, but this row cannot be
    what shows it.

    So `eos_is_a_real_query_token` is recorded on every comparison, and row 1 --
    full mask, EOS real, 2 live slots -- is measured alongside as the supplementary
    case where the question is actually asked. Which of the two ADR-0008 should
    publish is the owner's call, not this script's; both are in the artefact.
    """
    eos = int((ids[row] == cfg.eos_id).to(torch.int32).argmax())
    eos_is_real = bool(mask[row][eos] != 0)
    alpha_eos = torch.stack([a[row, :, eos, :] for a in out.cross_attention])
    alpha_eos = torch.where(
        mem.valid[row].view(1, 1, -1), alpha_eos, torch.zeros_like(alpha_eos)
    )
    tr = trace_for_row(cap, mem.valid, row, 0)
    kw = {"n_live": n_live, "capacity": ADR_FIXTURE["M"]}
    r_sum = retrieval_demand(tr, **kw)
    r_eos = retrieval_demand(dataclasses.replace(tr, alpha=alpha_eos), **kw)
    a = [float(x) for x in r_sum[:n_live]]
    b = [float(x) for x in r_eos[:n_live]]
    order_sum = sorted(range(n_live), key=lambda i: -a[i])
    order_eos = sorted(range(n_live), key=lambda i: -b[i])
    diff = float((r_sum - r_eos).abs().max())
    return {
        "row": row,
        "n_live": n_live,
        "q_real": int((mask[row] != 0).sum()),
        "eos_query_position": eos,
        "eos_is_a_real_query_token": eos_is_real,
        "eos_overlaps_the_summed_positions": eos_is_real,
        "r_i_sum_collapse": [float(x) for x in r_sum],
        "r_i_eos_only": [float(x) for x in r_eos],
        "max_abs_diff_r_i": diff,
        "max_abs_diff_as_frac_of_largest_entry": diff / max(a),
        "rank_order_sum": order_sum,
        "rank_order_eos": order_eos,
        "ranks_agree_on_this_row": order_sum == order_eos,
        "spearman_rho_live_slots": _spearman_rho(a, b),
        "n_rows_compared": 1,
        "model_trained": False,
        "what": (
            "ADR-0008 alternative 2. Does NOT cancel. 🔴 One untrained row: the "
            "rank agreement is too small a sample to mean anything and is NOT "
            "evidence E0d would score the two alike -- E0d over a held-out "
            "subsample is the discriminator, not this table."
        ),
        "caveat": (
            None
            if eos_is_real
            else (
                "🔴 [EOS] is at a PAD position on this row, so the sum-collapse "
                "excludes the single position the EOS-collapse reads. The two "
                "share no query position and the divergence is guaranteed by the "
                "fixture, not measured from the attention. See the supplementary "
                "row."
            )
        ),
    }


# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=None, help="where to write the JSON artefact"
    )
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    cfg = fx._cfg()
    model, mem, ids, mask, out, cap = fx._capture(cfg)
    fixture = _assert_fixture_shape(cfg, model, mem, mask, cap)

    payload = {
        "produced_by": "experiments/s0-02/measure_qtok_collapse.py",
        "for": "docs/decisions/ADR-0008-qtok-collapse.md",
        "device": "cpu",
        "torch": torch.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "q_tok_collapse": Q_TOK_COLLAPSE,
        "fixture": {
            **fixture,
            "source": "tests/test_capture_bridge.py (imported, not re-typed)",
            "model_seed": 0,
            "sentence_seed": 1,
            "memory_seed": 2,
            "trained": False,
            "n_seeds": 1,
            "spread": (
                "NONE. One fixture, one row, one seed -- these are deterministic "
                "float32 identity checks, not a sampled quantity, so there is no "
                "sd to report and this artefact must not be read as if there were."
            ),
        },
        "increment_identity": increment_identity(model, mem, mask, out, cap),
        "mean_vs_sum": mean_vs_sum(mem, mask, cap),
        "eos_only": eos_only(cfg, mem, ids, mask, out, cap, ROW, ADR_FIXTURE["n_live"]),
        "eos_only_supplementary_row": eos_only(
            cfg, mem, ids, mask, out, cap, SUPPLEMENTARY_ROW, SUPPLEMENTARY_N_LIVE
        ),
    }
    payload["eos_only_supplementary_row"]["why"] = (
        "NOT an ADR-0008 published figure. Measured because row 0's [EOS] is a "
        "PAD position, so the ADR's 0.0383 compares two collapses that share no "
        "query position. Row 1 has a full mask, so its [EOS] is one of the "
        "positions the sum collapses over and the comparison is the one the ADR "
        "means to make."
    )

    ii = payload["increment_identity"]
    mv = payload["mean_vs_sum"]
    eo = payload["eos_only"]
    print(f"Q_TOK_COLLAPSE            {Q_TOK_COLLAPSE}")
    print(f"fixture                   {fixture}")
    print()
    print(
        f"increment identity        max |lhs-rhs| = {ii['max_abs_diff_over_layers']:.6g}"
        f"  over {ii['n_layers']} layers, on values up to "
        f"{ii['max_abs_value_over_layers']:.6g}"
    )
    print(f"mean vs sum   max |dr_i| = {mv['max_abs_diff_r_i']:.6g}   (expected ~0)")
    print(
        f"EOS vs sum    max |dr_i| = {eo['max_abs_diff_r_i']:.6g}   "
        f"({100 * eo['max_abs_diff_as_frac_of_largest_entry']:.1f}% of the "
        f"largest entry)  [row {eo['row']}, eos_is_real="
        f"{eo['eos_is_a_real_query_token']}]"
    )
    sup = payload["eos_only_supplementary_row"]
    print(
        f"EOS vs sum    max |dr_i| = {sup['max_abs_diff_r_i']:.6g}   "
        f"({100 * sup['max_abs_diff_as_frac_of_largest_entry']:.1f}% of the "
        f"largest entry)  [row {sup['row']}, eos_is_real="
        f"{sup['eos_is_a_real_query_token']}]  SUPPLEMENTARY"
    )
    if eo["caveat"]:
        print(f"\n{eo['caveat']}")
    print()
    print(f"r_i sum-collapse          {[round(x, 5) for x in mv['r_i_sum_collapse']]}")
    print(f"r_i mean-collapse         {[round(x, 5) for x in mv['r_i_mean_collapse']]}")
    print(f"r_i EOS-only              {[round(x, 5) for x in eo['r_i_eos_only']]}")
    print(
        f"contribution() sum        "
        f"{[round(x, 4) for x in mv['contribution_sum_collapse']]}"
    )
    print(
        f"contribution() mean       "
        f"{[round(x, 4) for x in mv['contribution_mean_collapse']]}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
