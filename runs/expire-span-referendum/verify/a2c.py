"""Verifier's own A2-C comparison: re-run heartbeats + step-3000 checkpoints vs the original
(c59d1af) outputs. Broader than A2.3's five fields: every beat field except wall-clock timing,
and every tensor in the checkpoint (model, policy w/b, optimizer state, stream, rng)."""
import json, sys
from pathlib import Path
import torch

NEW = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/es-referendum-a2-run/runs/expire-span-referendum")
OLD = Path("/Users/keanooo7/retrieval-successor-retention/.worktrees/es-referendum-run/runs/expire-span-referendum")
TIMING = {"elapsed_s", "step_s", "sent_per_s"}
jobs = sys.argv[1:]


def beats(p):
    out = {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") == "beat":
            out[r["step"]] = {k: v for k, v in r.items() if k not in TIMING}
    return out


def same(a, b, path=""):
    """Bitwise equality over nested ckpt structures; returns list of differing paths."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            return [path]
        return [] if torch.equal(a.contiguous().view(-1).view(torch.uint8) if a.is_floating_point() else a, b.contiguous().view(-1).view(torch.uint8) if b.is_floating_point() else b) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return [path + "{keys}"]
        d = []
        for k in a:
            d += same(a[k], b[k], f"{path}.{k}")
        return d
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [path + "[len]"]
        d = []
        for i, (x, y) in enumerate(zip(a, b)):
            d += same(x, y, f"{path}[{i}]")
        return d
    if hasattr(a, "tobytes") and hasattr(a, "dtype"):
        return [] if (hasattr(b, "tobytes") and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()) else [path]
    return [] if a == b else [path]


res = {}
for job in jobs:
    bn, bo = beats(NEW / job / "heartbeat.jsonl"), beats(OLD / job / "heartbeat.jsonl")
    steps = sorted(set(bn) | set(bo))
    first = next((s for s in steps if bn.get(s) != bo.get(s)), None)
    diff_fields = sorted({k for s in steps for k in set(bn.get(s, {})) | set(bo.get(s, {}))
                          if bn.get(s, {}).get(k) != bo.get(s, {}).get(k)})
    cn = torch.load(NEW / job / "ckpt-003000.pt", map_location="cpu", weights_only=False)
    co = torch.load(OLD / job / "ckpt-003000.pt", map_location="cpu", weights_only=False)
    ck = {k: same(cn[k], co[k], k) for k in cn if k != "meta"}
    meta_diff = same(cn["meta"], co["meta"], "meta")
    res[job] = {
        "beats_new": len(bn), "beats_old": len(bo), "steps_0_2999_present": steps == list(range(3000)),
        "first_differing_step": first, "differing_fields": diff_fields,
        "ckpt_diffs": {k: v for k, v in ck.items() if v}, "ckpt_sections_compared": sorted(ck),
        "meta_diffs": meta_diff,
        "policy_w_equal": torch.equal(cn["policy"]["weight"], co["policy"]["weight"]),
        "policy_b_equal": torch.equal(cn["policy"]["bias"], co["policy"]["bias"]),
        "policy_w_finite": bool(torch.isfinite(cn["policy"]["weight"]).all()),
        "BIT_IDENTICAL": first is None and not any(ck.values()),
    }
json.dump(res, sys.stdout, indent=1)
print()
print(f"{sum(r['BIT_IDENTICAL'] for r in res.values())}/{len(res)} BIT-IDENTICAL", file=sys.stderr)
