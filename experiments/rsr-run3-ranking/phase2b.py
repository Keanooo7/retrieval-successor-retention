"""Phase 2b (on-policy run 3): run 2's harness, changed only in the loss.

PREREG ``experiments/rsr-run3-ranking/PREREG.md`` (committed alone at 32475b5) §5, run
id ``rsr-run3-2b``. **Only if 2a is WIN on all 3 seeds** (§4.5); the step count is
§5.3's rule applied to 2a's ledger; ``omega_s`` per seed is machine-read from it.

**Run 2's harness is reused by import, never copied.** ``experiments/rsr-run2/run.py``
is loaded as a private instance ``R2`` against its own fresh private run-1 instance
``R1`` (phase 1's 1a pattern), and exactly these names of that private ``R1`` are
rebound -- nothing in run 1's or run 2's files, or in their own module instances,
changes:

* ``R1.RUN_ID = "rsr-run3-2b"`` -- the job ids the controls read
  (``rsr-run3-2b-<arm>-s<seed>``);
* ``R1.cli_argv`` -- run 2's command (run 1's plus ``target=loo_resample``) plus §0.3's
  switch on both RSR arms, right after the target override:
  ``--policy-override loss=rank --policy-override rank_k=3
  --policy-override rank_omega=<omega_s>``; fifo untouched (run 1's command);
* ``R1.derive_every`` -- run 1's cadence ``gcd(160, CKPT)``, further gcd'd with 3000 so
  that phi exists on disk at k = 3000 for §5.4's psiR@3000 (training is bitwise
  unaffected by the cadence: ``R1.derive_every``'s own proof). Equal to run 1's rule
  whenever 3000 is already on it (K_2b = 3000 included).

``measure`` runs run 2's ``measure_seed`` (run 1's C0-C10 and every held-out pass, run
2's C2'/C9/C11/C12/C13 and §4/§5 additions) with ``cfg.eval_range = EVAL2`` and
``iters = K_2b``, under the measure stage's seal (EVAL2 open, TUNE sealed), after every
training process has exited with a completed footer; then §5.4's descriptive passes,
§5.6's contrasts and §5.7's added controls (C11', C-K, C-omega, C-pair, C-res, C-ES).

🔴 **Not implementable as written at K_2b != 3000 (reported to the PM, not resolved
here).** §5.7 applies run 2's C10 and C12 "unchanged". C10 requires the registered R-1
(run 1's owner table: 3000) to equal the run's iters; C12 requires fifo's run_id and
config hash to equal run 1's fifo's, and both stamp ``iters``. §5.2 says fifo is "run
1's command, byte for byte", while §5.3/§5.4 run K_2b steps. At K_2b = 3000 every
reading agrees; at K_2b > 3000 C10 and C12 fail by construction (or fifo runs a
different step count from the RSR arms, which run 1's pairing and step controls do not
admit). ``plan`` therefore refuses (exit 3) at K_2b != 3000 with this reason, and 2b
waits for a PM ruling. Nothing here picks a reading.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import importlib.util  # noqa: E402


def _load_here(name: str, file: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


SS = _load_here("run3_stage_seal", "stage_seal.py")
C = SS.C

import torch  # noqa: E402

from rsr.exit_codes import Exit  # noqa: E402
from rsr.retention.rank_loss import LOSS_RANK  # noqa: E402

RUN_ID = "rsr-run3-2b"
RUN_2A = "rsr-run3-2a"
EXPERIMENT = "experiments/rsr-run3-ranking/run.py"
LAUNCHER = "experiments/rsr-run3-ranking/launch_2b.zsh"
PREREG_REL = "experiments/rsr-run3-ranking/PREREG.md"
PREREG_PATH = ROOT / PREREG_REL
PREREG_COMMIT = "32475b5"
OUT_DEFAULT = ROOT / "runs" / RUN_ID
OUT_2A = ROOT / "runs" / RUN_2A

# 🔒 PREREG §5.3, checked against the committed text by `prereg_check`.
K_BLOCK = 1500
K_MIN = 3000
RANK_K = 3
PSIR_AT = 3000  # §5.4: psiR@3000 if K_2b > 3000
SMOKE_STEPS = 48  # §5.3 S-2b
STOP_LINE_H = 24.0  # §5.3
CONV_BLOCK = 16  # §5.6 item 7
CONV_FRACTION = 0.10  # §5.6 item 7
N_BOOT = 2000
ES_REPRO = 32  # §5.7 C-ES: EVAL[0:32]
GAMMA, WD, O5 = 0.9, 0.01, "eval"  # run 2's owner values (§0.2)

# --------------------------------------------------------------------------- #
# the harness: run 2 against a fresh private run 1, rebound
# --------------------------------------------------------------------------- #

_H: dict[str, Any] = {}
_PLAN: dict[str, Any] = {"omega": {}}


def prereg_check(path: Path = PREREG_PATH) -> dict:
    t = Path(path).read_text()
    m = re.search(
        r"K_2b = the smallest multiple of (\d+) that is ≥ max\((\d+), max_s K_conv", t
    )
    got = None if m is None else (int(m.group(1)), int(m.group(2)))
    ok = got == (K_BLOCK, K_MIN)
    m2 = re.search(r"for \*\*(\d+) optimizer steps\*\* \(16 warm \+ 32 on-policy\)", t)
    ok2 = m2 is not None and int(m2.group(1)) == SMOKE_STEPS
    m3 = re.search(r"\*\*Stop line: (\d+) h\*\*", t)
    ok3 = m3 is not None and float(m3.group(1)) == STOP_LINE_H
    return {"k_rule": got, "smoke_steps": ok2, "stop_line": ok3, "ok": ok and ok2 and ok3}


def harness() -> dict[str, Any]:
    """``{"R1", "R2"}``: run 2 loaded against a FRESH private run-1 instance, so the
    rebinding below never reaches run 2's own instance or any other caller's."""
    if _H:
        return _H
    key = "_rsr_run2_r1"
    saved = sys.modules.pop(key, None)
    try:
        R2 = C.load_module("_run3_2b_r2", "experiments/rsr-run2/run.py")
    finally:
        if saved is not None:
            sys.modules[key] = saved
        else:
            sys.modules.pop(key, None)
    R1 = R2.R1
    _r2_cli = R2.cli_argv
    _every = R1.derive_every

    def cli_argv(cfg, arm: str, owner, out_dir: Path) -> list[str]:
        """Run 2's command plus §0.3's switch on both RSR arms (after the target
        override); fifo is run 1's command byte for byte."""
        argv = list(_r2_cli(cfg, arm, owner, out_dir))
        if arm in R1.RSR_ARMS:
            om = _PLAN["omega"].get(int(cfg.seed))
            if om is None:
                raise R1.Refused(f"no omega_s for seed {cfg.seed}: run `plan` first")
            i = argv.index(f"target={R2.TARGET}") + 1
            argv[i:i] = rank_tokens(om)
        return argv

    def derive_every(steps: int, ckpt: int, diag: int = R1.DIAG_STEP) -> int:
        e = _every(steps, ckpt, diag)
        return math.gcd(e, PSIR_AT) if steps > PSIR_AT and e > 0 else e

    R1.RUN_ID = RUN_ID
    R1.cli_argv = cli_argv
    R1.derive_every = derive_every
    _H.update(R1=R1, R2=R2, cli_argv=cli_argv, derive_every=derive_every)
    return _H


def rank_tokens(omega: float) -> list[str]:
    return [
        "--policy-override",
        f"loss={LOSS_RANK}",
        "--policy-override",
        f"rank_k={RANK_K}",
        "--policy-override",
        f"rank_omega={float(omega):g}",
    ]


def k_2b(kconvs) -> int:
    """§5.3: the smallest multiple of 1500 that is >= max(3000, max_s K_conv)."""
    ks = [int(k) for k in kconvs]
    if not ks:
        raise ValueError("no K_conv")
    m = max(K_MIN, *ks)
    return ((m + K_BLOCK - 1) // K_BLOCK) * K_BLOCK


def ckpt_every(steps: int) -> int:
    H = harness()
    R1 = H["R1"]
    return H["derive_every"](steps, R1.derive_ckpt(steps), R1.DIAG_STEP)


# --------------------------------------------------------------------------- #
# plan: omega_s and K_2b, machine-read from 2a's ledger (C-K, C-omega inputs)
# --------------------------------------------------------------------------- #

C10_C12_REASON = (
    "PREREG §5.7 applies run 2's C10 (registered R-1 = 3000 in run 1's owner table) "
    "and C12 (fifo's run_id / config hash = run 1's fifo's; iters is in both) "
    "unchanged, and §5.2 has fifo as run 1's command byte for byte, while §5.3 runs "
    "K_2b steps. At K_2b != 3000 these cannot all hold: C10 and C12 fail by "
    "construction, or fifo runs a different step count from the RSR arms. Not "
    "resolved here: a PM ruling is needed before 2b launches."
)


def read_2a(ledger_path: Path) -> dict:
    """2a's ledger: the summary row (exit, gate, labels) and each seed's selection
    row; the file's sha256."""
    p = Path(ledger_path)
    sha = C.sha256_file(p)
    rows = C.ledger_rows(p)
    sm = rows["summary"]["value"]
    sel = {}
    for s in sm.get("per_seed", {}):
        r = rows.get(f"seed{s}.selection")
        sel[int(s)] = None if r is None else r["value"]
    return {"path": str(p), "sha256": sha, "summary": sm, "selection": sel}


def make_plan(ledger_path: Path) -> dict:
    a = read_2a(ledger_path)
    sm = a["summary"]
    seeds = sorted(a["selection"])
    out: dict[str, Any] = {
        "run_id": RUN_ID,
        "ledger_2a": {"path": a["path"], "sha256": a["sha256"]},
        "seeds": seeds,
        "rule": "K_2b = the smallest multiple of 1500 >= max(3000, max_s K_conv(s, w_s))",
    }
    problems = []
    if sm.get("exit") != 0:
        problems.append(f"2a exit {sm.get('exit')}: 2b does not run (§4.5)")
    if sm.get("gate_to_2b") is not True:
        problems.append("2a is not WIN on all 3 seeds: 2b does not run (§4.5, §7)")
    if any(sm.get("labels", {}).get(str(s), sm.get("labels", {}).get(s)) != "WIN"
           for s in seeds):  # fmt: skip
        problems.append(f"2a labels {sm.get('labels')}: not WIN on every seed")
    omega, kconv = {}, {}
    for s in seeds:
        r = a["selection"][s] or {}
        omega[s], kconv[s] = r.get("omega_s"), r.get("k_conv")
        if omega[s] is None or kconv[s] is None:
            problems.append(f"seed {s}: no selection row in 2a's ledger")
    out.update(omega_s=omega, k_conv=kconv)
    if not problems:
        K = k_2b(kconv.values())
        out.update(K_2b=K, ckpt=harness()["R1"].derive_ckpt(K), ckpt_every=ckpt_every(K))
        out["psiR_at_3000"] = K > PSIR_AT
        if K != K_MIN:
            problems.append(f"K_2b = {K} != 3000. " + C10_C12_REASON)
    out["problems"] = problems
    out["ok"] = not problems
    return out


def set_plan(plan: dict) -> None:
    _PLAN["omega"] = {int(k): float(v) for k, v in plan["omega_s"].items()}


# --------------------------------------------------------------------------- #
# Amendment 2b-S (C-omega's pin)
# --------------------------------------------------------------------------- #


def amendment_2bS(path: Path = PREREG_PATH) -> dict | None:
    """The committed Amendment 2b-S, machine-read: its section must carry the lines
    ``- 2a ledger sha256: `<64 hex>``` and ``- K_2b: <int>``. None if absent."""
    t = Path(path).read_text()
    m = re.search(r"^## AMENDMENT 2b-S\b.*?(?=^## |\Z)", t, re.S | re.M)
    if m is None:
        return None
    sec = m.group(0)
    sha = re.search(r"^- 2a ledger sha256: `([0-9a-f]{64})`", sec, re.M)
    k = re.search(r"^- K_2b: (\d+)\s*$", sec, re.M)
    return {
        "ledger_2a_sha256": None if sha is None else sha.group(1),
        "K_2b": None if k is None else int(k.group(1)),
    }


# --------------------------------------------------------------------------- #
# §5.7 controls added to run 2's, and §5.6 item 7 (pure; tested on fixtures)
# --------------------------------------------------------------------------- #


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def c11_prime(results: dict, hbs: dict, launch: dict | None, c11: dict | None) -> dict:
    """C11': run 2's C11 (labels carry ``target=loo_resample``; every beat
    ``mc.target`` and a finite ``loo_s``) AND both RSR arms' labels carry
    ``loss=rank``; every beat has ``mc.loss == "rank"`` and a finite ``l_rank``."""
    R1 = harness()["R1"]
    per = {}
    for a in R1.RSR_ARMS:
        label = ((results.get(a) or {}).get("config") or {}).get("arm_label") or ""
        beats = (hbs.get(a) or {}).get("beats") or []
        bad = [
            b.get("step")
            for b in beats
            if (b.get("mc") or {}).get("loss") != LOSS_RANK
            or not _finite((b.get("mc") or {}).get("l_rank"))
            or not _finite(b.get("loo_s"))
        ]
        per[a] = {
            "label": label,
            "n_beats": len(beats),
            "bad_steps": bad[:20],
            "ok": f"loss={LOSS_RANK}" in label and bool(beats) and not bad,
        }
    fifo_label = ((results.get("fifo") or {}).get("config") or {}).get("arm_label") or ""
    per["fifo"] = {"label": fifo_label, "ok": "loss=" not in fifo_label}
    launch_ok = (launch or {}).get("loss") == LOSS_RANK
    return {
        "C11": c11,
        "per_arm": per,
        "launch_loss": (launch or {}).get("loss"),
        "ok": bool((c11 or {}).get("ok"))
        and launch_ok
        and all(v["ok"] for v in per.values()),
    }


def c_k(plan: dict, launch: dict | None, steps_done: dict, kconv_2a: dict) -> dict:
    """C-K: K_2b in the launch record equals §5.3's rule applied to 2a's ledger;
    ``steps_done == K_2b`` for both RSR arms."""
    want = k_2b(kconv_2a.values())
    got = None if launch is None else launch.get("steps")
    ok = (
        str(got) == str(want)
        and plan.get("K_2b") == want
        and all(steps_done.get(a) == want for a in ("psiC", "age"))
    )
    return {"rule": want, "launch": got, "plan": plan.get("K_2b"),
            "steps_done": steps_done, "ok": bool(ok)}  # fmt: skip


def c_omega(plan: dict, ledger_2a: dict, amendment: dict | None) -> dict:
    """C-omega: each seed's omega_s equals 2a's selection row; 2a's ledger sha256
    equals the one pinned in Amendment 2b-S."""
    sel = ledger_2a["selection"]
    rows = {s: (sel.get(s) or {}).get("omega_s") for s in plan["seeds"]}
    same = all(float(plan["omega_s"][s]) == float(rows[s]) for s in plan["seeds"])
    pinned = None if amendment is None else amendment.get("ledger_2a_sha256")
    return {
        "plan": plan["omega_s"],
        "ledger": rows,
        "ledger_sha256": ledger_2a["sha256"],
        "pinned": pinned,
        "ok": bool(same) and pinned is not None and pinned == ledger_2a["sha256"],
    }


def convergence_label(losses: list[float], seed: int, n_boot: int = N_BOOT) -> dict:
    """§5.6 item 7: the mean per-step rank loss over the last 10% of steps minus that
    over the preceding 10%, 95% CI from a block bootstrap over 16-step blocks
    (``torch.Generator().manual_seed(seed)``, 2000 replicates): STILL-IMPROVING iff
    the CI upper bound < 0, else PLATEAU. Descriptive; gates nothing."""
    x = torch.tensor([float(v) for v in losses], dtype=torch.float64)
    n = x.numel()
    w = int(n * CONV_FRACTION)
    if w < CONV_BLOCK or not bool(torch.isfinite(x).all()):
        return {"label": None, "why": f"{n} steps: too few for two {CONV_BLOCK}-blocks"}
    last, prev = x[n - w :], x[n - 2 * w : n - w]
    point = float(last.mean() - prev.mean())
    g = torch.Generator().manual_seed(int(seed))

    def blocks(v):
        nb = v.numel() // CONV_BLOCK
        return v[: nb * CONV_BLOCK].reshape(nb, CONV_BLOCK)

    bl, bp = blocks(last), blocks(prev)
    il = torch.randint(0, bl.shape[0], (n_boot, bl.shape[0]), generator=g)
    ip = torch.randint(0, bp.shape[0], (n_boot, bp.shape[0]), generator=g)
    rep = bl[il].mean((1, 2)) - bp[ip].mean((1, 2))
    lo, hi = float(torch.quantile(rep, 0.025)), float(torch.quantile(rep, 0.975))
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "window_steps": w,
        "label": "STILL-IMPROVING" if hi < 0 else "PLATEAU",
    }


def training_footers(out_root: Path, seeds) -> dict:
    """§5.4's precondition: every training process (3 arms x seeds) exited rc 0 with a
    completed footer (the CLI's result dict) before EVAL2 is opened."""
    R1 = harness()["R1"]
    rows, ok = {}, True
    for s in seeds:
        for a in R1.ARMS:
            k = f"{a}-s{s}"
            rcp = Path(out_root) / "logs" / f"{k}.rc"
            rc = rcp.read_text().strip() if rcp.exists() else None
            res = R1.read_cli_result(Path(out_root) / "logs" / f"{k}.log")
            good = rc == "0" and res is not None
            rows[k] = {"rc": rc, "footer": res is not None, "ok": good}
            ok = ok and good
    return {"jobs": rows, "ok": ok}


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def plan_stage(out_root: Path, ledger_2a: Path) -> int:
    """``plan``: omega_s and K_2b from 2a's ledger, written to ``plan.json``. Exit 3
    (2b does not run) unless 2a is WIN on all 3 seeds and the plan is implementable
    as written (see the module docstring)."""
    SS.install("2b", "plan")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    p = make_plan(ledger_2a)
    p["prereg_text"] = prereg_check()
    p["ok"] = p["ok"] and p["prereg_text"]["ok"]
    p["written_utc"] = utc()
    p["exit"] = 0 if p["ok"] else 3
    (out_root / "plan.json").write_text(json.dumps(p, indent=1, default=str) + "\n")
    if not p["ok"]:
        print("DID NOT RUN:", *p["problems"], sep="\n  ", file=sys.stderr)
    return p["exit"]


def owner_values(K: int):
    R1 = harness()["R1"]
    seeds = tuple(SS.C.registered_seeds(PREREG_PATH))
    return R1.OwnerValues(GAMMA, WD, int(K), O5, seeds)


def measure_cfg(seed: int, K: int):
    """Run 1's config for the seed at ``iters = K_2b``, held-out range EVAL2 (machine-
    read from the phase-1 front matter)."""
    R1 = harness()["R1"]
    ev2 = SS.sealed_ranges()["EVAL2"]
    return dataclasses.replace(R1.prereg_config(seed, K), eval_range=ev2)


def measure_stage(seed: int, out_root: Path, *, ledger_2a: Path = OUT_2A / "ledger.json"):
    """§5.4-§5.7 for one seed: the measure stage's seal (EVAL2 opened), the footer
    precondition, run 2's ``measure_seed`` on EVAL2 at K_2b, then the added controls.
    (The §5.4 descriptive passes beyond run 2's -- psiR@3000, run 2's psiC, 2a's head,
    pinned ES -- run in ``extra_passes``.)"""
    out_root = Path(out_root)
    seal = SS.install("2b", "measure")
    H = harness()
    R1, R2 = H["R1"], H["R2"]
    SS.rebind()
    C.forbid_constants_record()
    plan = json.loads((out_root / "plan.json").read_text())
    if not plan.get("ok"):
        raise R1.Refused("plan.json is not ok: 2b does not run")
    set_plan(plan)
    seeds = plan["seeds"]
    foot = training_footers(out_root, seeds)
    if not foot["ok"]:
        raise R1.Refused(f"a training process has no completed footer: {foot}")
    K = int(plan["K_2b"])
    cfg = measure_cfg(seed, K)
    es = c_es(seed, cfg)  # before EVAL2 is read (§5.7 C-ES)
    doc = R2.measure_seed(cfg, owner_values(K), out_root, threads=1)
    a2 = read_2a(ledger_2a)
    launch = doc.get("launch") or {}
    results = {
        a: R1.read_cli_result(Path((doc.get("arms") or {}).get(a, {}).get("log", "")))
        if (doc.get("arms") or {}).get(a)
        else None
        for a in R1.ARMS
    }
    hbs = {a: R1.read_heartbeat(R1.arm_dir(out_root, a, seed)) for a in R1.ARMS}
    steps_done = {a: (results.get(a) or {}).get("steps_done") for a in R1.ARMS}
    ctl = doc["controls"]
    ctl["R-SEAL.install"] = {**seal, "ok": True}
    ctl["R-SEAL.footers"] = foot
    ctl["C11prime"] = c11_prime(results, hbs, launch, ctl.get("C11"))
    kc = {s: a2["selection"][s]["k_conv"] for s in seeds}
    ctl["C-K"] = c_k(plan, launch, steps_done, kc)
    ctl["C-omega"] = c_omega(plan, a2, amendment_2bS())
    doc.setdefault("scoped_controls", {})["C-ES"] = {
        k: v for k, v in es.items() if not k.startswith("_")
    }
    if doc.get("heldout") is not None:
        ex = extra_passes(seed, out_root, cfg, plan, es)
        ctl["C-pair"] = {
            "passes": {k: v.get("C-pair") for k, v in ex.items() if "C-pair" in v},
            "ok": all(v["C-pair"]["ok"] for v in ex.values() if "C-pair" in v),
        }
        ctl["C-res"] = {
            "passes": {k: v.get("C-res") for k, v in ex.items() if "C-res" in v},
            "ok": all(v["C-res"]["ok"] for v in ex.values() if "C-res" in v),
        }
        dl = (ctl.get("Cdelta") or {}).get("delta")
        doc["run3_extra"] = {
            "passes": ex,
            "contrasts": contrasts_5_6(seed, out_root, cfg, dl),
        }
    beats = (hbs.get("psiC") or {}).get("beats") or []
    doc["run3"] = {
        "plan": plan,
        "convergence_psiR": convergence_label(
            [b.get("loss_mc") for b in sorted(beats, key=lambda b: b["step"])], seed
        ),
    }
    ctl["R-SEAL.census"] = SS.census({"POOL": tuple(R1.POOL)})
    failed = [k for k, v in ctl.items() if not (isinstance(v, dict) and v.get("ok"))]
    doc["controls_failed"] = failed
    if failed and int(doc.get("exit", 3)) == 0:
        doc.update(exit=int(Exit.DID_NOT_RUN), status="did_not_run")
    (out_root / f"measure-seed{seed}.json").write_text(
        json.dumps(doc, indent=2, default=str) + "\n"
    )
    return int(doc["exit"])


# --------------------------------------------------------------------------- #
# §5.4 descriptive passes, §5.6 contrasts, §5.7 C-ES / C-pair / C-res
# --------------------------------------------------------------------------- #

ES_PINNED = Path("/Users/keanooo7/rsr-substrate/2026-10-07-es-pinned")
ES_CKPT = "L48-k1e-2-seed{s}-ckpt-003000.pt"


def es_module():
    return C.load_module("_run3_2b_es", "experiments/expire-span-referendum/run.py")


def c_es(seed: int, model_cfg) -> dict:
    """C-ES (scoped to §5.6 item 1): run 2's C14 (the pinned manifest, ``shasum -c`` rc
    read directly; the referendum's records' sha256) and ``es_pass`` on EVAL[0:32]
    reproducing the referendum's saved records record for record -- BEFORE EVAL2 is
    read. On failure the ES contrast is omitted and printed; the class stands."""
    R1, R2 = harness()["R1"], harness()["R2"]
    ES = es_module()
    out: dict[str, Any] = {"C14": R2.c14_es_source(seed)}
    try:
        payload = R1.read_checkpoint(ES_PINNED / ES_CKPT.format(s=seed))
        model = R1.model_from_payload(payload).to(model_cfg.device)
        pol = ES.es_policy_from_payload(payload)
        cfg = R1.prereg_config(seed, 3000)
        docs = R1.docs_by_id(cfg, range(R1.EVAL[0], R1.EVAL[0] + ES_REPRO))
        got = ES.es_pass(model, pol, docs, R1.vocab_map(cfg), cfg)
        ref = R1.read_pass(Path(R2.REFS.es_out) / "eval" / f"seed{seed}", "es")[:ES_REPRO]
        out["repro"] = {"n": len(got), "equal": got == ref}
        out["ok"] = bool(out["C14"].get("ok")) and got == ref and len(got) == ES_REPRO
        out["_payload"] = payload
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


def run_pass(model, make_policy, docs: list, vmap, cfg) -> list[dict]:
    """Run 1's held-out procedure for a policy not among run 1's arms: blocks of
    ``cfg.eval_batch``, ``R1.run_batch``, a fresh policy per block."""
    R1 = harness()["R1"]
    recs: list[dict] = []
    for i in range(0, len(docs), cfg.eval_batch):
        chunk = docs[i : i + cfg.eval_batch]
        recs += R1.run_batch(model, make_policy(chunk), chunk, vmap, cfg)
    return recs


def extra_passes(seed: int, out_root: Path, cfg, plan: dict, es: dict) -> dict:
    """§5.4: psiR@3000 (if K_2b > 3000), run 2's psiC final phi, 2a's selected head
    (T_warm = 0 deploy path), pinned ES (only if C-ES passed); each with C-pair (its
    ``(doc, [(t, gap)])`` equals the fifo pass's, all documents) and C-res (run 1's
    C7 residency). Written under ``heldout/seed<s>/``."""
    R1, R2 = harness()["R1"], harness()["R2"]
    hd = Path(out_root) / "heldout" / f"seed{seed}"
    fifo = R1.read_pass(hd, "fifo")
    docs = R1.docs_by_id(cfg, list(cfg.eval_ids()))
    vmap = R1.vocab_map(cfg)
    K = int(plan["K_2b"])
    fp = R1.read_checkpoint(R1.checkpoint_path(R1.arm_dir(out_root, "fifo", seed), K))
    model = R1.model_from_payload(fp)
    passes: dict[str, Any] = {}
    if K > PSIR_AT:
        p3 = R1.read_checkpoint(
            R1.checkpoint_path(R1.arm_dir(out_root, "psiC", seed), PSIR_AT)
        )
        passes["psiR@3000"] = lambda chunk, p3=p3: R1.eval_policy("psiC", p3, cfg)
    OP = C.load_module("_run3_2b_op", "experiments/offline-psi/run.py")
    r2c = OP.run2_psic_config(OP.REFS, seed)  # run 2's psiC ckpt-003000 (sha in doc)
    r2p = r2c["payload"]
    passes["psiC_run2"] = lambda chunk: R1.eval_policy("psiC", r2p, cfg)
    sel = json.loads((OUT_2A / f"seed{seed}" / "selection.json").read_text())
    st = torch.load(sel["phi_path"], weights_only=True)
    conf = r2c["conf"]
    ov = {**conf["policy_overrides"], "loss": LOSS_RANK, "rank_k": RANK_K,
          "rank_omega": float(sel["omega_s"])}  # fmt: skip
    spe = float(conf["epoch"]["steps_per_epoch"])
    passes["head_2a"] = lambda chunk: OP.deploy_policy(st, seed, ov, spe)
    out: dict[str, Any] = {}
    for name, mk in passes.items():
        recs = run_pass(model, mk, docs, vmap, cfg)
        sha = OP.write_records(hd / f"{name}.jsonl.gz", recs)
        bad = [r["doc"] for r, d in zip(recs, docs, strict=True)
               if not R1.residency_ok(d, r, cfg.memory_slots)]  # fmt: skip
        out[name] = {
            "records_sha256": sha,
            "C-pair": {"n": [len(recs), len(fifo)],
                       "ok": len(recs) == len(fifo) and R2._tg(recs) == R2._tg(fifo)},
            "C-res": {"n_mismatched": len(bad), "ok": not bad},
        }  # fmt: skip
    if es.get("ok"):
        ES = es_module()
        pol = ES.es_policy_from_payload(es["_payload"])
        recs = ES.es_pass(R1.model_from_payload(es["_payload"]), pol, docs, vmap, cfg)
        OP.write_records(hd / "es.jsonl.gz", recs)
        out["es"] = {
            "C-pair": {"ok": R2._tg(recs) == R2._tg(fifo)},
            "C-res": {"ok": True},
        }
    else:
        out["es"] = {"omitted": "C-ES failed: the ES contrast is omitted (§5.7)"}
    out["ff"] = {"hits": "run 2's factfiller_hits in run2_heldout (model-free)"}
    return out


def contrasts_5_6(seed: int, out_root: Path, cfg, delta) -> dict:
    """§5.6 items 1-4 (descriptive, unadjusted): psiR - pinned ES (labelled with run
    2's rule against delta_seed), psiR - psiC(run 2), psiR@3000 - psiC(run 2),
    psiR - 2a's head; all-query and by bucket."""
    R1, R2 = harness()["R1"], harness()["R2"]
    hd = Path(out_root) / "heldout" / f"seed{seed}"
    names = ["psiC", "fifo", "psiC_run2", "head_2a", "psiR@3000", "es"]
    recs = {n: R1.read_pass(hd, n) for n in names if (hd / f"{n}.jsonl.gz").exists()}
    W = R1.boot_weights(len(recs["fifo"]), seed, N_BOOT)
    bk = R1.buckets_for(cfg.memory_slots)
    out: dict[str, Any] = {}
    pairs = {
        "psiR_minus_es": ("psiC", "es"),
        "psiR_minus_psiC_run2": ("psiC", "psiC_run2"),
        "psiR3000_minus_psiC_run2": ("psiR@3000", "psiC_run2"),
        "psiR_minus_head_2a": ("psiC", "head_2a"),
    }
    for key, (a, b) in pairs.items():
        if a not in recs or b not in recs:
            out[key] = {"omitted": f"{a if a not in recs else b} not run"}
            continue
        row = {}
        for bucket in ("all", "gap_2_to_M", "gap_gt_M"):
            per = {a: R1.answer_counts(recs[a], bk[bucket]),
                   b: R1.answer_counts(recs[b], bk[bucket])}  # fmt: skip
            c = R1.contrast(per, a, b, W)
            row[bucket] = R2.label_contrast(c, delta) if key == "psiR_minus_es" else c
        out[key] = row
    out["note"] = (
        "descriptive, unadjusted; gates nothing; falsifier 6 stays deferred to E1"
    )
    return out


def smoke_measure(out: Path) -> dict:
    """S-2b's numbers (§5.3) from a ``SMOKE=1`` launch at ``out``: per RSR process the
    median ``step_s``, ``loo_s``, ``eval_forward_s`` and rank-loss time over the
    on-policy steps ``k in [16, 48)``; one held-out pass of 64 already-used EVAL
    documents through the psiR payload (never EVAL2)."""
    SS.install("2b", "smoke")
    R1 = harness()["R1"]
    rows: dict[str, Any] = {}
    plan = json.loads((Path(out) / "plan.json").read_text())
    for s in plan["seeds"]:
        for a in R1.RSR_ARMS:
            hb = R1.read_heartbeat(R1.arm_dir(out, a, s))
            beats = [
                b for b in hb.get("beats") or [] if 16 <= b.get("step", -1) < SMOKE_STEPS
            ]

            def med(key, beats=beats):
                v = [b.get(key) if key != "rank_s" else (b.get("mc") or {}).get("rank_s")
                     for b in beats]  # fmt: skip
                v = [x for x in v if _finite(x)]
                return float(torch.tensor(v).median()) if v else None

            rows[f"{a}-s{s}"] = {
                k: med(k) for k in ("step_s", "loo_s", "eval_forward_s", "rank_s")
            }
    s0 = plan["seeds"][0]
    cfg = R1.prereg_config(s0, SMOKE_STEPS)
    payload = R1.read_checkpoint(
        R1.checkpoint_path(R1.arm_dir(out, "psiC", s0), SMOKE_STEPS)
    )
    model = R1.model_from_payload(payload)
    docs = R1.docs_by_id(cfg, range(R1.EVAL[0], R1.EVAL[0] + 64))
    t0 = time.perf_counter()
    run_pass(
        model,
        lambda chunk: R1.eval_policy("psiC", payload, cfg),
        docs,
        R1.vocab_map(cfg),
        cfg,
    )
    held = time.perf_counter() - t0
    med_step = max(v["step_s"] for v in rows.values() if v["step_s"] is not None)
    res = {"per_process": rows, "heldout_64_eval_s": held, "max_median_step_s": med_step,
           "census": SS.census({"EVAL": tuple(R1.EVAL)})}  # fmt: skip
    (Path(out) / "smoke-measure.json").write_text(
        json.dumps(res, indent=1, default=str) + "\n"
    )
    return res


# --------------------------------------------------------------------------- #
# launch-checks and finalize
# --------------------------------------------------------------------------- #

QUESTION = (
    "PREREG rsr-run3-ranking §5 (2b): trained on-policy with run 2's harness, changed "
    "only in the loss, does run 2's content-only bilinear psi_hat beat FIFO on the "
    "fresh sealed EVAL2?"
)


def launch_checks(out_root: Path, *, suite_log: Path, suite_rc: Path, suite_head: Path):
    """G-C3 (the suite at the launch head, rc read by the launcher), R-SEAL (d) (the
    branch search), and the frozen manifest (plan, K_2b, omega_s, sealed source)."""
    SS.install("2b", "launch-checks")
    out_root = Path(out_root)
    rep: dict[str, Any] = {"run_id": RUN_ID, "launched_utc": utc()}
    rep["head"], rep["dirty"] = C.git_head(), C.git_dirty_paths()
    try:
        rc_txt = Path(suite_rc).read_text().strip()
        rc = int(rc_txt) if rc_txt.lstrip("-").isdigit() else None
        cen = C.suite_census(Path(suite_log).read_text(), rc)
        sh = Path(suite_head).read_text().strip()
        cen.update(head=sh, head_is_launch_head=sh == rep["head"])
        cen["ok"] = bool(cen["ok"]) and cen["head_is_launch_head"]
    except OSError as e:
        cen = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    rep["G-C3"] = cen
    rep["R-SEAL.search"] = SS.branch_search()
    plan = json.loads((out_root / "plan.json").read_text())
    rep["plan_ok"] = bool(plan.get("ok"))
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    led.manifest(
        {
            "run_id": RUN_ID,
            "prereg": PREREG_REL,
            "prereg_commit": PREREG_COMMIT,
            "plan": plan,
            "sealed_source": SS.SEALED_SOURCE,
            "sealed": {k: list(v) for k, v in SS.sealed_ranges().items()},
            "stage_seals": {k: list(v) for k, v in SS.STAGES["2b"].items()},
            "owner": {"gamma": GAMMA, "wd": WD, "o5": O5},
            "launcher": LAUNCHER,
        }
    )
    rep["manifest"] = {
        "config_hash": led.doc["config_hash"],
        "manifest_written_utc": led.doc["manifest_written_utc"],
    }
    rep["ok"] = bool(cen.get("ok") and rep["R-SEAL.search"]["ok"] and rep["plan_ok"]
                     and rep["dirty"] == [])  # fmt: skip
    rep["exit"] = 0 if rep["ok"] else 3
    (out_root / "launch.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return rep["exit"]


def finalize(out_root: Path) -> int:
    """§5.5: run 2's class rule over the per-seed labels (run 2's ``summarize``); any
    failed run-level control is exit 3 and the class is NOT READ."""
    SS.install("2b", "finalize")
    R2 = harness()["R2"]
    out_root = Path(out_root)
    seeds = list(SS.C.registered_seeds(PREREG_PATH))
    docs = []
    for sd in seeds:
        f = out_root / f"measure-seed{sd}.json"
        if f.exists():
            docs.append(json.loads(f.read_text()))
    sm = R2.summarize(docs) if docs else {}
    failed = {
        d["seed"]: d.get("controls_failed") for d in docs if d.get("controls_failed")
    }
    exits = [int(d.get("exit", 3)) for d in docs]
    code = 3 if (len(docs) != len(seeds) or failed or any(e != 0 for e in exits)) else 0
    run = {
        "run_id": RUN_ID,
        "exit": code,
        "status": {0: "ok", 3: "did_not_run"}[code],
        "class": sm.get("class") if code == 0 else "NOT READ",
        "labels": sm.get("labels") if code == 0 else {s: "NOT READ" for s in seeds},
        "labels_raw_not_read": None if code == 0 else sm.get("labels"),
        "controls_failed": failed,
        "summary_run2_form": sm,
        "convergence_psiR": {
            d["seed"]: (d.get("run3") or {}).get("convergence_psiR") for d in docs
        },
        "contrasts_5_6": {
            d["seed"]: (d.get("run3_extra") or {}).get("contrasts") for d in docs
        },
        "route": "Gate A (PREREG §5.5, §7), whatever the class",
    }
    (out_root / "run.json").write_text(json.dumps(run, indent=1, default=str) + "\n")
    L = C.ledger_mod()
    led = L.Ledger(RUN_ID, question=QUESTION, runs_root=out_root.parent)
    lj = out_root / "launch.json"
    if lj.exists():
        m = json.loads(lj.read_text()).get("manifest") or {}
        led.doc["config_hash"] = m.get("config_hash")
        led.doc["manifest_written_utc"] = m.get("manifest_written_utc")
    for d in docs:
        for a, r in (d.get("arms") or {}).items():
            led.command(
                r["argv"], exit_code=r.get("rc"), note=f"arm {a} seed {d['seed']}"
            )
    led.command([".venv/bin/python", EXPERIMENT, "2b", "finalize"], exit_code=code)
    plan = json.loads((out_root / "plan.json").read_text())
    led.run_meta(device="cpu", seeds_actually_run=[d["seed"] for d in docs],
                 steps_requested=plan.get("K_2b"))  # fmt: skip
    led.note("summary", run, how=f"{EXPERIMENT} 2b finalize")
    for d in docs:
        for k, v in C.flatten(f"seed{d['seed']}", d, max_depth=4):
            led.note(k, v, how=f"runs/{RUN_ID}/measure-seed{d['seed']}.json")
    led.status(run["status"])
    led.write()
    lines = [
        f"# RESULTS: {RUN_ID} (RSR run 3, phase 2b)\n",
        f"Generated by `{EXPERIMENT} 2b finalize`; no number is typed. Written by a "
        "model; not by Brendan.\n",
        f"- exit `{code}`; class **{run['class']}**; labels {run['labels']}",
        f"- controls failed: {failed or 'none'}",
        f"- route: {run['route']}\n",
        "Descriptive (§5.6, never gates): convergence "
        f"{run['convergence_psiR']}; contrasts in run.json `contrasts_5_6`.\n",
    ]
    (out_root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return code


def main(argv: list[str]) -> int:
    from rsr.exit_codes import ArgumentParser

    torch.set_num_threads(1)
    p = ArgumentParser(prog="run.py 2b", description="PREREG rsr-run3-ranking §5")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--out-root", default=str(OUT_DEFAULT))
    pp.add_argument("--ledger-2a", default=str(OUT_2A / "ledger.json"))
    pm = sub.add_parser("measure")
    pm.add_argument("--seed", type=int, required=True)
    pm.add_argument("--out-root", default=str(OUT_DEFAULT))
    pm.add_argument("--ledger-2a", default=str(OUT_2A / "ledger.json"))
    psm = sub.add_parser("smoke-measure")
    psm.add_argument("--out", required=True)
    plc = sub.add_parser("launch-checks")
    plc.add_argument("--out-root", default=str(OUT_DEFAULT))
    pfz = sub.add_parser("finalize")
    pfz.add_argument("--out-root", default=str(OUT_DEFAULT))
    a = p.parse_args(argv)
    if a.cmd == "launch-checks":
        logs = Path(a.out_root) / "logs"
        return launch_checks(
            Path(a.out_root),
            suite_log=logs / "suite.log",
            suite_rc=logs / "suite.rc",
            suite_head=logs / "suite.head",
        )
    if a.cmd == "finalize":
        return finalize(Path(a.out_root))
    if a.cmd == "smoke-measure":
        r = smoke_measure(Path(a.out))
        print(json.dumps({k: v for k, v in r.items() if k != "census"}, indent=1))
        return 3  # a smoke is never the run
    if a.cmd == "plan":
        return plan_stage(Path(a.out_root), Path(a.ledger_2a))
    if a.cmd == "measure":
        return measure_stage(a.seed, Path(a.out_root), ledger_2a=Path(a.ledger_2a))
    return 3
