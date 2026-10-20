"""Erratum P0.2 (2026-09-29): correct phase A's logged FIT_VAL R² without a refit.

Phase A (run b2-psi-probe-fit, 280a2ad) logged ``1 - n_val*SSE/SST`` for every
head's ``val_r2_raw`` and for ``age_decodability.{U,C}.r2`` (§6, §8.1). The exact
correction is closed-form: ``R² = 1 - (1 - logged)/n_val``.

Nothing is overwritten. For each payload ``phaseA/ckpt{L}-seed{s}.pt`` this writes
``phaseA/ckpt{L}-seed{s}.r2-corrected.json``, and for the ledger
``ledger.r2-corrected.json``: original, n_val_rows, corrected, formula.

**Independent check, no refit, no model.** The age target ``t - i`` depends only on
the row set, so its FIT_VAL SST is structural: every FIT_VAL document contributes
the same (t, i) rows (U: every i < t; C: FIFO-resident i < t, i.e. t - i <= M).
The F11 split code (``r2_split_by_index``, correct as logged, different code)
gives SSE_a + SSE_b from its two R² and the structural split SSTs; the pooled SSE is
that same sum over the same predictions, so ``1 - (SSE_a + SSE_b)/SST`` must equal
the correction to rounding. Recorded per payload. The ψ̂ heads' targets depend on
the model, so for them the correction is arithmetic only (no independent SST).
The C row set's structure is itself checked: its predicted row count and split
counts must equal the logged ones, or the check is reported as not applicable.

    .venv/bin/python experiments/b2-psi-probe/r2_erratum.py [--fit-dir DIR]
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from rsr.exit_codes import ArgumentParser, Exit, refuse, run_main

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMULA = "R2_true = 1 - (1 - R2_logged) / n_val_rows"
BUG = "logged R2 = 1 - n_val*SSE/SST, SSE already summed over the n_val rows"


def _load_run():
    spec = importlib.util.spec_from_file_location("b2_psi_probe_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b2_psi_probe_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def structural_age_rows(S: int, m: int, rowset: str) -> list[tuple[int, int]]:
    """The (t, i) rows of one document: U every i < t; C FIFO-resident (t - i <= m)."""
    return [(t, i) for t in range(1, S) for i in range(t) if rowset == "U" or t - i <= m]


def _sst(ys: list[float]) -> float:
    n = len(ys)
    mu = sum(ys) / n
    return sum((y - mu) ** 2 for y in ys)


def correct_payload(run, pt: Path, S: int, m: int) -> dict:
    a = torch.load(pt, weights_only=False)
    heads = {}
    for key, h in a["heads"].items():
        if h.get("val_r2_form") == run.R2_FORM:
            refuse(Exit.DID_NOT_RUN, f"{pt}: {key} is post-fix; nothing to correct")
        n, sel = h["n_val_rows"], h["selected"]
        heads[key] = {
            "original": h["val_r2_raw"][sel],
            "n_val_rows": n,
            "corrected": run.r2_from_logged_n_val_form(h["val_r2_raw"][sel], n),
            "selected": sel,
            "path_original": h["val_r2_raw"],
            "path_corrected": [
                run.r2_from_logged_n_val_form(x, n) for x in h["val_r2_raw"]
            ],
        }
    age = {}
    for r, dec in a["age_decodability"].items():
        n = dec["n_val_rows"]
        corrected = run.r2_from_logged_n_val_form(dec["r2"], n)
        # The payload keeps no raw-MSE list for the age predictor, and re-deriving SSE
        # from the logged R² would be circular: check against the split code instead.
        age[r] = {"original": dec["r2"], "n_val_rows": n, "corrected": corrected}
        age[r]["check"] = age_check_from_split(dec, S, m, r, corrected)
    return {
        "payload": str(pt.relative_to(ROOT)),
        "formula": FORMULA,
        "bug": BUG,
        "fixed_in": "run.py fit_heads, val_r2_form = " + run.R2_FORM,
        "heads": heads,
        "age_decodability": age,
    }


def age_check_from_split(
    dec: dict, S: int, m: int, rowset: str, corrected: float
) -> dict:
    """The corrected pooled R² vs the one the F11 split code implies (structural SSTs,
    SSE_a + SSE_b from the split R²). No stored MSE is needed, so nothing circular."""
    rows = structural_age_rows(S, m, rowset)
    n_val = dec["n_val_rows"]
    n_docs, rem = divmod(n_val, len(rows))
    split = dec["r2_split"]
    lt = [t - i for t, i in rows if i < m]
    ge = [t - i for t, i in rows if i >= m]
    if not (
        rem == 0
        and split["i_lt_M"]["n"] == n_docs * len(lt)
        and split["i_ge_M"]["n"] == n_docs * len(ge)
    ):
        return {"applicable": False, "why": "row counts do not match the structure"}
    sst = n_docs * _sst([t - i for t, i in rows])
    sse = (1 - split["i_lt_M"]["r2"]) * n_docs * _sst(lt) + (
        1 - split["i_ge_M"]["r2"]
    ) * n_docs * _sst(ge)
    from_split = 1 - sse / sst
    return {
        "applicable": True,
        "n_docs": n_docs,
        "rows_per_doc": len(rows),
        "sst_structural": sst,
        "sse_from_split_r2": sse,
        "r2_from_split_code": from_split,
        "corrected": corrected,
        "abs_diff": abs(from_split - corrected),
    }


def correct_ledger(run, ledger: Path) -> dict:
    rows = json.loads(ledger.read_text())["rows"]
    out = {}
    for e in rows:
        k, v = e["key"], e["value"]
        if ".head." in k and isinstance(v, dict) and "val_r2_raw" in v:
            logged, n = v["val_r2_raw"], v["n_val_rows"]
        elif ".age_decodability." in k and isinstance(v, dict) and "r2" in v:
            logged, n = v["r2"], v["n_val_rows"]
        else:
            continue
        out[k] = {
            "original": logged,
            "n_val_rows": n,
            "corrected": run.r2_from_logged_n_val_form(logged, n),
        }
    return {
        "ledger": str(ledger.relative_to(ROOT)),
        "formula": FORMULA,
        "bug": BUG,
        "rows": out,
    }


def main(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser()
    ap.add_argument("--fit-dir", type=Path, default=ROOT / "runs" / "b2-psi-probe-fit")
    a = ap.parse_args(argv)
    run = _load_run()
    S, m = run.S_STEPS, run.M_STEPS
    worst = 0.0
    for pt in sorted((a.fit_dir / "phaseA").glob("ckpt*-seed*.pt")):
        side = pt.with_name(pt.stem + ".r2-corrected.json")
        if side.exists():
            print(f"exists, not overwritten: {side}", file=sys.stderr)
            return Exit.DID_NOT_RUN
        rec = correct_payload(run, pt, S, m)
        side.write_text(json.dumps(rec, indent=1))
        for r, x in rec["age_decodability"].items():
            c = x["check"]
            if not c["applicable"]:
                print(f"{pt.name} age_{r}: check not applicable ({c['why']})")
                continue
            worst = max(worst, c["abs_diff"])
            print(
                f"{pt.name} age_decodability.{r}: logged {x['original']:.4f} "
                f"-> corrected "
                f"{x['corrected']:.6f}; split-code implies {c['r2_from_split_code']:.6f} "
                f"(|diff| {c['abs_diff']:.2e})"
            )
        u = rec["heads"]["U@0.0"]
        print(
            f"{pt.name} head U@0.0: logged {u['original']:.4f} n_val {u['n_val_rows']}"
            f" -> corrected {u['corrected']:.6f}"
        )
    led = a.fit_dir / "ledger.json"
    side = a.fit_dir / "ledger.r2-corrected.json"
    if side.exists():
        print(f"exists, not overwritten: {side}", file=sys.stderr)
        return Exit.DID_NOT_RUN
    side.write_text(json.dumps(correct_ledger(run, led), indent=1))
    print(f"worst |corrected - split-implied| over applicable age checks: {worst:.3e}")
    return Exit.OK if worst < 1e-9 else Exit.FAIL


if __name__ == "__main__":
    run_main(main)
