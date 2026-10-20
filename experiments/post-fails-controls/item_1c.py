"""Item 1c (post-fails-controls PREREG §5): top-k / bottom-k agreement with demand
measured strictly after the decision step.

Pre-registration: `experiments/post-fails-controls/PREREG.md`, committed alone at
9f22d87. This module is the item's harness: `main(argv)` measures and writes
`runs/post-fails-controls-1c/{manifest.json, ledger.json}`; `finalize(argv)`
renders `RESULTS.md` and `claims.json` from that ledger (no number is typed).

**Model-free except control C-C3** (E0d's r_i recomputed on 32 U documents per seed).
Reads, never writes: offline-psi's FIFO-world gestalts, B0-CF's Delta, lookahead-room's
D.pt, run 2's psiC phi, offline-psi's phi, the LS fit, the pinned Expire-Span
checkpoints. Trains nothing; never calls `rsr.constants.record()`; no gradient
anywhere (`torch.no_grad`).

Decisions (§5): U, 1088 documents per seed, FIFO's world, full-memory steps
``t in [16, 47)``; populations all-live (16 slots ``[t-16, t)``) and bos-excluded
(slot ``t-1`` removed); step subsets all decisions and Q-steps (sentence ``t`` is a
query). Reference ``F_i(t) = G[t+1][i]``, ``G = discounted_returns(Delta, 0.9)``.

Agreement (exact, no RNG): with ``p_j`` / ``q_j`` the probability that slot ``j`` is in
the scorer's / the reference's bottom-k under uniform tie-breaking, a step's agreement
is ``sum_j p_j q_j / k``. Computed in exact integer units of ``1 / LCM`` (``LCM = 3 *
lcm(1..16)^2``), so the uniform scorer reaches chance ``k / n_t`` exactly (C-C4).

Exit codes (PREREG §0): 0 ran and every control passed; 1 a defect after the controls
passed; 3 did not run, or a control failed (numbers written, NOT READ, label DROPPED).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import discounted_demand, true_demand  # noqa: E402
from rsr.exit_codes import ArgumentParser, Exit, run_main, status  # noqa: E402
from rsr.retention.policy import MemoryState  # noqa: E402
from rsr.retention.value_head import BilinearValueHead  # noqa: E402

EXPERIMENT = "experiments/post-fails-controls/item_1c.py"
ITEM = "1c"
RUN_ID = "post-fails-controls-1c"
PREREG = "experiments/post-fails-controls/PREREG.md"
PREREG_COMMIT = "9f22d87"
OUT_DEFAULT = ROOT / "runs" / RUN_ID
RUN_PY = "experiments/post-fails-controls/run.py"


def _load(name: str, rel: str):
    """Import an experiment script by path, once per process. Imported, never copied."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: psi-ls-capacity (the LS fit) and, through it, diagnostic A (`load_inputs`: the
#: gestalt and Delta pins), offline-psi (`u_documents`, `u_split`, `target_of`,
#: `score_matrices`, `c1_delta`), run 1 (`boot_weights`, `_ci`, `SUBSTRATE`), B0
#: (`verify_manifest`, `check_shape`, `d_rel`) and lookahead-room.
LSC = _load("_pfc_lscap", "experiments/psi-ls-capacity/run.py")
CV = LSC.CV
OP = CV.OP
R1 = OP.R1
LR = OP.LR
B0 = OP.B0
#: E0d's own measurement path for C-C3 (`live_pass`).
E0D = _load("_pfc_e0d", "experiments/e0d/run.py")
#: The ES referendum's loader for the pinned Expire-Span policy.
ESR = _load("_pfc_esr", "experiments/expire-span-referendum/run.py")

# --------------------------------------------------------------------------- #
# 🔒 PREREG §5, transcribed. Changing any of these is changing the PREREG.
# --------------------------------------------------------------------------- #

SEEDS = (0, 1, 2)
M = 16
S = 48
D_MODEL = 128
GAMMA = 0.9  # F's gamma (B0-CF / offline-psi target), and the ceiling's
T_LO, T_HI = M, S - 1  # decisions t in [16, 47): t = 47 has no future step
KS = (1, 3)
DIRS = ("bottom", "top")
POPS = ("all_live", "bos_excluded")
STEPSETS = ("all", "Q")
N_BOOT = 2000
CKPT = 3000
#: The scorers of the §5 table, in its order. `op` and `ls` are reported on U-val and
#: U-train only ("separately"); every other scorer on all of U as well.
SCORERS = ("ceiling", "r_i", "psiC", "op", "ls", "es", "age", "uniform")
SPLIT_ONLY = ("op", "ls")
DOCSETS = ("U", "U_val", "U_train")
PRIMARY = ("U", "all_live", "all")
C_C3_DOCS = 32
C_C5_DECISIONS = 4  # x 16 slots = 64 cells
C_C6_DECISIONS = 64
#: Exact agreement units: 3 * lcm(1..16)^2, so every p_j q_j / k and every k / n_t is
#: an integer multiple of 1 / LCM for n_t <= 16 and k in {1, 3}.
LCM16 = 720720
UNITS = 3 * LCM16 * LCM16
LABELS = ("SURVIVES", "DROPPED")

OP_LEDGER = Path(R1._R) / ".worktrees/offline-psi/runs/offline-psi/ledger.json"
OP_PHI = Path(R1._R) / ".worktrees/offline-psi/runs/offline-psi/seed{s}/phi-cf_all.pt"
LS_DIR = Path(R1._R) / ".worktrees/psi-ls-capacity/runs/psi-ls-capacity/seed{s}"
ES_PINNED = Path(OP.REFS.es_pinned)
ES_FILE = "L48-k1e-2-seed{s}-ckpt-003000.pt"
E0D_LEDGER = Path(R1._R) / ".worktrees/e0d/runs/e0d/ledger.json"
SUBSTRATE_MANIFEST = Path(B0.MANIFEST)

QUESTION = (
    "Does 'attention ranks the top of the ranking well and the bottom badly' survive "
    "once demand is measured strictly after the decision step (PREREG §5)?"
)
FALSIFIER = (
    "NONE as a spec falsifier: Tier D diagnostic. Registered reading rule (§5): "
    "SURVIVES iff on all 3 seeds and at k = 1 and k = 3 the ceiling's bottom-k CI lower "
    "bound > chance, r_i's top-k CI lower bound > chance and r_i's bottom-k CI upper "
    "bound <= chance; otherwise DROPPED. Exit 3 reads DROPPED."
)
EXPECTED = (
    "PREREG §10: DROPPED, most likely because the ceiling's bottom-k does not clear "
    "chance (E0d's true_demand bottom-1 sat at or below chance)."
)


class ControlFailure(RuntimeError):
    """A PREREG control failed: exit 3, the label is not read (DROPPED)."""


# --------------------------------------------------------------------------- #
# common.py (the 1a builder's): G-C1 seal guard, G-C2 substrate check
# --------------------------------------------------------------------------- #


def common():
    """The shared harness module (G-C1 seal guard, G-C2 manifest check, G-C3 census,
    G-C4 refusals, ledger helpers). Owned by the 1a builder; imported, never copied."""
    return _load("pfc_common", "experiments/post-fails-controls/common.py")


#: Named document ranges the G-C1 census reports per seed (besides the sealed ones).
NAMED_RANGES = {
    "U_E": (64, 1088),
    "U_P": (4096, 4160),
    "EVAL": (995000, 997048),
}


class Globals:
    """PREREG §2's global controls for one item: G-C1 (seal installed for the whole
    process, census, launch-time branch search), G-C2 (substrate manifest, start and
    end, plus checkpoint pins), G-C3 (the suite census handed in by the launcher), G-C4
    (`rsr.constants.record` refused; no gradient), G-C5 (seeds actually run)."""

    def __init__(self, led, *, reads_substrate: bool, args) -> None:
        self.led = led
        self.cm = common()
        self.reads_substrate = reads_substrate
        self.a = args
        self.failed: list[str] = []
        self.pins = self.cm.Pins()

    def _gc2(self, when: str) -> None:
        r = self.cm.manifest_check(
            SUBSTRATE_MANIFEST, R1.MANIFEST_SHA256, B0.main_checkout()
        )
        self.led.note(f"G-C2.{when}", r, how="common.manifest_check (shasum -a 256 -c)")
        if not r["ok"]:
            self.failed.append(f"G-C2 substrate manifest at {when}")

    def start(self) -> list[str]:
        seal = self.cm.install_seal()
        self.led.note("G-C1.install", seal, how="common.install_seal")
        self.led.note(
            "G-C4.record_refused", self.cm.forbid_constants_record(), how="common"
        )
        if getattr(self.a, "branch_search", True):
            bs = self.cm.branch_search()
            self.led.note("G-C1.branch_search", bs, how="common.branch_search (§1.1)")
            if not bs["ok"]:
                self.failed.append("G-C1(e): branch search new hit or git failure")
        else:
            self.failed.append("G-C1(e): branch search not run (smoke)")
        log, rc = getattr(self.a, "suite_log", None), getattr(self.a, "suite_rc", None)
        head = getattr(self.a, "suite_head", None)
        if log is not None and rc is not None:
            cen = self.cm.suite_census(Path(log).read_text(), int(rc))
            now = self.cm.git_head()
            cen.update(log=str(log), head=head, launch_head=now)
            cen["head_is_launch_head"] = head is not None and head == now
            cen["ok"] = cen["ok"] and cen["head_is_launch_head"]
            self.led.note("G-C3.suite", cen, how="common.suite_census of the launch log")
            if not cen["ok"]:
                self.failed.append("G-C3: suite not rc 0 / failures / errors")
        else:
            self.led.note("G-C3.suite", None, how="no suite log given: NOT a pass")
            self.failed.append("G-C3: no suite census given to this launch")
        if self.reads_substrate:
            self._gc2("start")
        return self.failed

    def pin_substrate(self, seed: int) -> dict:
        sub = R1.SUBSTRATE[seed]
        r = self.cm.checkpoint_pin(sub["checkpoint"], sub["sha256"])
        self.pins.pin(sub["checkpoint"])
        if not r["ok"]:
            self.failed.append(f"G-C2 substrate seed {seed} pin")
        return r

    def end(self, seeds_run: list[int], modules=()) -> list[str]:
        if self.reads_substrate:
            self._gc2("end")
        rc = self.pins.recheck() if self.pins.first else {"n_files": 0, "ok": True}
        self.led.note("G-C2.pins_end", rc, how="common.Pins.recheck")
        if not rc["ok"]:
            self.failed.append("G-C2: a pinned input changed")
        cen = self.cm.seal_census(NAMED_RANGES)
        self.led.note("G-C1.census", cen, how="common.seal_census (ids by range)")
        if not cen["ok"]:
            self.failed.append("G-C1: a sealed id was generated or a guard leaked")
        grads = [self.cm.no_grads(m) for m in modules]
        self.led.note("G-C4.no_grads", grads, how="common.no_grads")
        if not all(g["ok"] for g in grads):
            self.failed.append("G-C4: a gradient reached a frozen module")
        self.led.run_meta(seeds_actually_run=sorted(seeds_run))
        if sorted(seeds_run) != list(SEEDS):
            self.failed.append(f"G-C5: seeds run {sorted(seeds_run)} != {list(SEEDS)}")
        return self.failed


def add_global_args(ap) -> None:
    ap.add_argument("--suite-log", type=Path, default=None, help="G-C3: the launch log")
    ap.add_argument("--suite-rc", type=int, default=None, help="G-C3: its rc, direct")
    ap.add_argument("--suite-head", default=None, help="G-C3: the sha the suite ran at")
    ap.add_argument(
        "--no-branch-search",
        dest="branch_search",
        action="store_false",
        help="smoke only (marks the run did-not-run)",
    )


# --------------------------------------------------------------------------- #
# the pure pieces (tests/test_post_fails_controls_1c.py drives each by hand)
# --------------------------------------------------------------------------- #


def decision_index(t_lo: int = T_LO, t_hi: int = T_HI, m: int = M):
    """``(T [n_dec], I [n_dec, m])``: decision steps and, per step, the FIFO-resident
    sentences ``[t - m, t)`` in slot order (slot 0 the oldest)."""
    T = torch.arange(t_lo, t_hi)
    I_ = T.unsqueeze(1) - m + torch.arange(m).unsqueeze(0)
    return T, I_


def kset_units(X: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bottom-k membership under uniform random tie-breaking, exactly.

    ``X`` ``[R, n]`` float (finite). -> ``(P, nt)`` with ``p_j = P_j / nt``: ``P_j = nt``
    if ``X_j`` is strictly below the k-th smallest value, ``k - #strictly below`` if
    equal to it, 0 otherwise; ``nt`` = the number of values equal to the k-th
    smallest."""
    if not bool(torch.isfinite(X).all()):
        raise ValueError("kset_units: a non-finite score")
    if not 1 <= k <= X.shape[1]:
        raise ValueError(f"k = {k} outside [1, {X.shape[1]}]")
    kth = X.sort(dim=1).values[:, k - 1 : k]
    below = torch.lt(X, kth)
    tied = torch.eq(X, kth)
    nb = below.sum(1, keepdim=True)
    nt = tied.sum(1, keepdim=True)
    zero = torch.zeros_like(nt)
    P = torch.where(below, nt, torch.where(tied, k - nb, zero))
    return P.to(torch.int64), nt.squeeze(1).to(torch.int64)


def agreement_units(X: torch.Tensor, F: torch.Tensor, k: int, top: bool) -> torch.Tensor:
    """Per row, ``UNITS * sum_j p_j q_j / k`` (an exact int64): the expected overlap
    fraction of the scorer's and the reference's bottom-k (``top``: top-k, the same on
    the negated values) under independent uniform tie-breaking (PREREG §5)."""
    if X.shape != F.shape:
        raise ValueError(f"shapes {tuple(X.shape)} != {tuple(F.shape)}")
    if top:
        X, F = -X, -F
    P, ntp = kset_units(X, k)
    Q, ntq = kset_units(F, k)
    num = (P * Q).sum(1)
    den = k * ntp * ntq
    if bool((UNITS % den != 0).any()):
        raise ValueError("agreement_units: a denominator does not divide UNITS")
    return num * (UNITS // den)


def chance_units(n: int, k: int) -> int:
    """``UNITS * k / n`` exactly."""
    if (UNITS * k) % n:
        raise ValueError(f"k/n = {k}/{n} is not a multiple of 1/UNITS")
    return UNITS * k // n


def gather_steps(X: torch.Tensor, T: torch.Tensor, I_: torch.Tensor) -> torch.Tensor:
    """``X`` ``[S, S]`` step-major -> ``[n_dec, m]`` = ``X[t][i]`` on the live cells."""
    return X[T.unsqueeze(1), I_]


def reference_F(G: torch.Tensor, T: torch.Tensor, I_: torch.Tensor) -> torch.Tensor:
    """``F_i(t) = G[t+1][i]``: demand strictly after ``t`` (``Delta[t]`` never enters)."""
    return G[(T + 1).unsqueeze(1), I_]


def pop_cols(pop: str, m: int = M) -> list[int]:
    """Slot columns of a population: all-live, or without the age-1 slot ``t-1``."""
    if pop == "all_live":
        return list(range(m))
    if pop == "bos_excluded":
        return list(range(m - 1))
    raise ValueError(pop)


def es_scores(spans: torch.Tensor, T: torch.Tensor, I_: torch.Tensor) -> torch.Tensor:
    """``max_l (e_li - age_i)`` with ``age_i = t - i``, in the policy's own dtype and
    order of operations (`ExpireSpanPolicy.select_eviction`). ``spans`` ``[L, S]``."""
    e = spans[:, I_]  # [L, n_dec, m]
    age = (T.unsqueeze(1) - I_).to(e.dtype)
    return (e - age).amax(dim=0)


def units_table(
    scores: dict[str, torch.Tensor],
    F: torch.Tensor,
    Q: torch.Tensor,
    *,
    ks=KS,
    dirs=DIRS,
    pops=POPS,
) -> dict:
    """``scores[name]`` and ``F`` are ``[N, n_dec, m]`` (``N`` documents), ``Q``
    ``[N, n_dec]`` bool (Q-steps). -> ``{(scorer, dir, k, pop, stepset): int64 [N, 3]}``
    = per document (agreement units, chance units, decisions), exact."""
    N, nd, m = F.shape
    out = {}
    sel = {"all": torch.ones(N, nd, dtype=torch.bool), "Q": Q.to(torch.bool)}
    for pop in pops:
        cols = pop_cols(pop, m)
        Fp = F[:, :, cols].reshape(N * nd, len(cols))
        n = len(cols)
        for name, X in scores.items():
            Xp = X[:, :, cols].reshape(N * nd, len(cols))
            for d in dirs:
                for k in ks:
                    a = agreement_units(Xp, Fp, k, top=(d == "top")).reshape(N, nd)
                    c = chance_units(n, k)
                    for ss, mask in sel.items():
                        cnt = mask.sum(1).to(torch.int64)
                        out[(name, d, k, pop, ss)] = torch.stack(
                            [(a * mask).sum(1), cnt * c, cnt], dim=1
                        )
    return out


def pooled_point(units: torch.Tensor) -> dict:
    """Exact pooled agreement, chance and their difference over documents
    (``units`` int64 ``[n_docs, 3]``; sums in python ints, one rounding at the end)."""
    A = sum(int(x) for x in units[:, 0].tolist())
    C = sum(int(x) for x in units[:, 1].tolist())
    N = sum(int(x) for x in units[:, 2].tolist())
    if N == 0:
        return {"agreement": None, "chance": None, "minus_chance": None, "n": 0}
    return {
        "agreement": A / UNITS / N,
        "chance": C / UNITS / N,
        "minus_chance": (A - C) / UNITS / N,
        "n": N,
        "exact_equal_chance": A == C,
    }


def boot_minus_chance(units: torch.Tensor, W: torch.Tensor) -> dict:
    """PREREG §5: the CI of (agreement - chance), pooled per replicate (ratio of
    per-document sums), percentile 95% (run 1's ``_ci``; a replicate with no decision
    is dropped and counted)."""
    X = units.to(torch.float64)  # per-document units < 2**53: exact
    num = W @ ((X[:, 0] - X[:, 1]) / UNITS)
    den = W @ X[:, 2]
    rep = num / den
    ci = R1._ci(rep)
    pt = pooled_point(units)
    return {
        "point": pt["minus_chance"],
        "lo": ci["lo"],
        "hi": ci["hi"],
        "sd": ci["sd"],
        "dropped": ci["dropped"],
        "agreement": pt["agreement"],
        "chance": pt["chance"],
        "n_decisions": pt["n"],
        "sd_exactly_zero": ci["sd"] == 0.0,
    }


def seed_conditions(res: dict) -> dict:
    """The three §5 conditions at k = 1 and k = 3 for one seed, on the primary set."""
    ds, pop, ss = PRIMARY
    out = {}
    for k in KS:

        def g(name, d, f, k=k):
            return res[ds][pop][ss][name][d][str(k)][f]

        lo_c, lo_rt, hi_rb = (
            g("ceiling", "bottom", "lo"),
            g("r_i", "top", "lo"),
            g("r_i", "bottom", "hi"),
        )
        out[str(k)] = {
            "ceiling_bottom_lo_above_chance": _finite(lo_c) and lo_c > 0,
            "r_i_top_lo_above_chance": _finite(lo_rt) and lo_rt > 0,
            "r_i_bottom_hi_at_or_below_chance": _finite(hi_rb) and hi_rb <= 0,
        }
    return out


def label(conds: dict[int, dict] | None, controls_ok: bool) -> str:
    """PREREG §5 (tightened): SURVIVES only if all three conditions hold at k = 1 and
    k = 3 on all 3 seeds and every control passed; otherwise DROPPED."""
    if not controls_ok or conds is None:
        return "DROPPED"
    for s in SEEDS:
        c = conds.get(s)
        if c is None:
            return "DROPPED"
        for k in KS:
            if not all(c[str(k)].values()):
                return "DROPPED"
    return "SURVIVES"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def sha256_file(p: Path) -> str:
    return R1.sha256_file(Path(p))


def ledger_rows(path: Path) -> dict[str, Any]:
    return {r["key"]: r for r in json.loads(Path(path).read_text())["rows"]}


# --------------------------------------------------------------------------- #
# inputs and controls
# --------------------------------------------------------------------------- #


def es_manifest_check() -> dict:
    """C-C1 (ES): ``shasum -a 256 -c MANIFEST.sha256`` in the pinned dir, rc direct."""
    p = subprocess.run(
        ["shasum", "-a", "256", "-c", "MANIFEST.sha256"],
        cwd=ES_PINNED,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "rc": p.returncode,
        "out": p.stdout.strip().splitlines(),
        "ok": p.returncode == 0,
    }


def load_D(seed: int) -> tuple[dict, dict]:
    """lookahead-room-r2's D.pt for ckpt3000 (manifest-verified by G-C2 / C-C1)."""
    base = B0.main_checkout()
    rel = B0.d_rel(CKPT, seed)
    v = B0.verify_manifest(SUBSTRATE_MANIFEST, [rel], base)
    return torch.load(base / rel, weights_only=True), v


def psic_head(seed: int) -> tuple[BilinearValueHead, dict, dict]:
    """Run 2's psiC phi at ckpt-003000, loaded into a fresh `BilinearValueHead(128)`."""
    rc = OP.run2_psic_config(OP.REFS, seed)
    payload = rc["payload"]
    state = R1.head_state(payload)
    head = BilinearValueHead(D_MODEL)
    head.load_state_dict(state)
    head.requires_grad_(False)
    ov = payload["meta"]["config"]["policy_overrides"]
    info = {
        "path": rc["path"],
        "sha256": sha256_file(Path(rc["path"])),
        "overrides": ov,
        "no_age": ov.get("psi_override") not in ("age_only", "neg_age"),
        "dtype": str(state["W"].dtype),
    }
    return head, payload, info


def op_head(seed: int) -> tuple[BilinearValueHead, dict, dict]:
    """offline-psi's selected cf_all phi. Its ledger's ``phi_sha256`` is offline-psi's
    `state_sha` (sha256 over the tensors in name order), not the file's sha256: both are
    recorded, the state hash is the one compared with the pin."""
    p = Path(str(OP_PHI).format(s=seed))
    sha = sha256_file(p)
    want = ledger_rows(OP_LEDGER)[f"seed{seed}.selection.cf_all.phi_sha256"]["value"]
    state = torch.load(p, weights_only=True)
    st_sha = OP.state_sha(state)
    head = BilinearValueHead(D_MODEL)
    head.load_state_dict(state)
    head.requires_grad_(False)
    info = {"path": str(p), "file_sha256": sha, "state_sha256": st_sha, "pin": want}
    info["ok"] = st_sha == want
    return head, state, info


def ls_head(seed: int) -> tuple[BilinearValueHead, dict, dict]:
    d = Path(str(LS_DIR).format(s=seed))
    p = d / "phi-cf_all-ls.pt"
    sha = sha256_file(p)
    want = json.loads((d / "result.json").read_text())["arms"]["cf_all"]["phi_sha256"]
    state = torch.load(p, weights_only=True)
    head = BilinearValueHead(D_MODEL).double()
    head.load_state_dict({k: v.to(torch.float64) for k, v in state.items()})
    head.requires_grad_(False)
    return (
        head,
        state,
        {
            "path": str(p),
            "sha256": sha,
            "pin": want,
            "ok": sha == want,
            "dtype": "float64",
        },
    )


def es_policy(seed: int):
    from rsr.train import checkpoint as ck

    p = ES_PINNED / ES_FILE.format(s=seed)
    payload = ck.load(p, restore_rng=False)
    pol = ESR.es_policy_from_payload(payload)
    pol.eval()
    return pol, payload, {"path": str(p), "sha256": sha256_file(p)}


def substrate_payload(seed: int) -> tuple[dict, dict]:
    """Arm B ckpt-003000, read from bytes hashed against run 1's SUBSTRATE pin."""
    from rsr.train import checkpoint as ck

    sub = R1.SUBSTRATE[seed]
    data = Path(sub["checkpoint"]).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    info = {"path": sub["checkpoint"], "sha256": sha, "pin": sub["sha256"]}
    info["ok"] = sha == sub["sha256"]
    if not info["ok"]:
        raise ControlFailure(f"G-C2: substrate seed {seed} hashes to {sha}")
    return ck.load_bytes(data, source=sub["checkpoint"]), info


def c_c6_state(es_payload: dict, sub_payload: dict) -> dict:
    """C-C6 (a): the ES checkpoint's transformer and W_sent equal the substrate's, bit
    for bit (every model tensor, same key set)."""
    a, b = es_payload["model"], sub_payload["model"]
    keys_equal = sorted(a) == sorted(b)
    bad = [k for k in a if k in b and not torch.equal(a[k], b[k])]
    return {
        "keys_equal": keys_equal,
        "n_tensors": len(a),
        "n_differ": len(bad),
        "differ": bad[:10],
        "ok": keys_equal and not bad,
    }


def memory_state(Sg: torch.Tensor, t: int, m: int = M) -> MemoryState:
    """FIFO's full memory at step ``t``: slot ``j`` holds sentence ``t - m + j``."""
    return MemoryState(
        gestalts=Sg[t - m : t].clone(),
        written_at=torch.arange(t - m, t, dtype=torch.int64),
        live=torch.ones(m, dtype=torch.bool),
        step=int(t),
    )


def c_c6_argmin(pol, Sall: torch.Tensor, seed: int) -> dict:
    """C-C6 (b): on 64 sampled decisions the ES score's argmin (ties to the lowest
    slot, i.e. the oldest) equals `ExpireSpanPolicy.select_eviction`."""
    rng = random.Random(f"pfc-1c:cc6:{seed}")
    T, I_ = decision_index()
    cells = [(n, int(t)) for n in range(Sall.shape[0]) for t in T.tolist()]
    pick = rng.sample(cells, C_C6_DECISIONS)
    bad = []
    with torch.no_grad():
        for n, t in pick:
            e = pol.spans(Sall[n])
            j = t - T_LO
            sc = es_scores(e, T[j : j + 1], I_[j : j + 1])[0]
            mine = int(torch.argmin(sc))
            ms = memory_state(Sall[n], t)
            theirs = int(pol.select_eviction(ms, torch.zeros(D_MODEL), t))
            if mine != theirs:
                bad.append((n, t, mine, theirs))
    return {"n": len(pick), "mismatch": bad[:10], "n_mismatch": len(bad), "ok": not bad}


def c_c5(
    seed: int, inp: dict, heads: dict, psic_payload: dict, op_state, ls_state
) -> dict:
    """C-C5: each phi loads into `BilinearValueHead(128)` (no age input) and its scores
    on 64 cells equal the producing run's own forward, ``==``: run 2's
    `RSRPolicy._psi` (what `ProbePolicy` records); offline-psi's head through its
    `score_matrices`; the LS head (`ls_head`, float64) through the same."""
    rng = random.Random(f"pfc-1c:cc5:{seed}")
    T, _ = decision_index()
    N = inp["S"].shape[0]
    pick = [(rng.randrange(N), rng.choice(T.tolist())) for _ in range(C_C5_DECISIONS)]
    out: dict[str, Any] = {"cells": [[n, t] for n, t in pick]}
    cfg = R1.prereg_config(seed, CKPT)
    pol = R1.eval_policy("psiC", psic_payload, cfg)
    op_ref = CV.new_head(seed)
    op_ref.load_state_dict(op_state)
    op_ref.requires_grad_(False)
    ls_ref = LSC.ls_head(seed, ls_state)
    res = {"psiC": [], "op": [], "ls": []}
    with torch.no_grad():
        for n, t in pick:
            Sg, c = inp["S"][n], inp["C"][n, t]
            mine = heads["psiC"](Sg[t - M : t], c)
            theirs = pol._psi(memory_state(Sg, t), c)
            res["psiC"].append(bool(torch.equal(mine, theirs)))
            Pm = OP.score_matrices(heads["op"], inp["S"][n : n + 1], inp["C"][n : n + 1])
            Pt = OP.score_matrices(op_ref, inp["S"][n : n + 1], inp["C"][n : n + 1])
            res["op"].append(bool(torch.equal(Pm[0, t, t - M : t], Pt[0, t, t - M : t])))
            S64, C64 = inp["S"][n : n + 1].double(), inp["C"][n : n + 1].double()
            Lm = OP.score_matrices(heads["ls"], S64, C64)
            Lt = OP.score_matrices(ls_ref, S64, C64)
            res["ls"].append(bool(torch.equal(Lm[0, t, t - M : t], Lt[0, t, t - M : t])))
    for k, v in res.items():
        out[k] = {"n_cells": len(v) * M, "all_equal": all(v)}
    out["head_class"] = {k: type(h).__name__ for k, h in heads.items()}
    out["ok"] = all(out[k]["all_equal"] for k in res) and all(
        isinstance(h, BilinearValueHead) for h in heads.values()
    )
    return out


def c_c3(seed: int, docs: list, vmap: dict, V: int, D: dict) -> dict:
    """C-C3: on U's first 32 documents the gated r_i from E0d's `live_pass` (its batch,
    16) equals ``D[t][i]`` on every resident cell; max|diff| <= 1e-6 * max|r|."""
    from rsr.train.loop import encode

    sub = R1.SUBSTRATE[seed]
    model = LR.load_model(Path(sub["checkpoint"]), V)
    model.requires_grad_(False)
    worst, rmax, n_cells, n_exact, problems = 0.0, 0.0, 0, 0, []
    chunk_docs = docs[:C_C3_DOCS]
    for i0 in range(0, len(chunk_docs), E0D.BATCH):
        chunk = chunk_docs[i0 : i0 + E0D.BATCH]
        ids, mask = encode(chunk, vmap, max_tokens=LR.L, steps=S)
        lv = E0D.live_pass(model, ids, mask)
        if not lv["eval_mode"]:
            problems.append("live_pass not in eval mode")
        for b, doc in enumerate(chunk):
            Dd = D[doc.doc_id]
            for t in range(S):
                want = list(range(max(0, t - M), t))
                ss = [int(x) for x in lv["slot_sentence"][b, t].tolist() if int(x) >= 0]
                if sorted(ss) != want:
                    problems.append(f"doc {doc.doc_id} t={t}: memory {ss}")
                    continue
                for j in range(M):
                    s_ = int(lv["slot_sentence"][b, t, j])
                    if s_ < 0:
                        continue
                    r = float(lv["r_gated"][b, t, j])
                    d = float(Dd[t, s_])
                    rmax = max(rmax, abs(r), abs(d))
                    worst = max(worst, abs(r - d))
                    n_cells += 1
                    n_exact += int(r == d)
    ok = not problems and n_cells > 0 and worst <= 1e-6 * rmax
    return {
        "n_docs": len(chunk_docs),
        "batch": E0D.BATCH,
        "n_cells": n_cells,
        "n_bit_exact": n_exact,
        "max_abs_diff": worst,
        "max_abs_r": rmax,
        "tolerance": 1e-6 * rmax,
        "problems": problems[:10],
        "ok": ok,
    }


def c_c2(D: dict, blob: dict, ids: list[int]) -> dict:
    """C-C2: FIFO's world. D.pt and Delta hold every U id in U order, ``[48, 48]``,
    finite at every ``i < t`` and NaN at ``i >= t`` (B0 C2, offline-psi C1); the live
    index this item reads is exactly ``range(t - 16, t)`` at every decision."""
    T, I_ = decision_index()
    live_ok = all(
        I_[j].tolist() == list(range(int(t) - M, int(t))) for j, t in enumerate(T)
    )
    d_ids_ok = list(D) == list(ids)
    d_bad = [d for d in ids if not B0.check_shape(D[d])]
    c1 = OP.c1_delta(blob, "", 0, ids)  # shape / NaN / id part only (sha is C-C1's)
    return {
        "live_index_ok": live_ok,
        "D_ids_ok": d_ids_ok,
        "D_bad": d_bad[:10],
        "delta_ids_ok": c1["ids_ok"],
        "delta_n_bad": c1["n_bad"],
        "ok": live_ok and d_ids_ok and not d_bad and c1["ids_ok"] and c1["n_bad"] == 0,
    }


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #


def score_tensors(inp: dict, D: dict, heads: dict, es_pol) -> dict:
    """``{scorer: [N, n_dec, m] float64}`` and ``F``, ``Q`` and the bridge inputs."""
    T, I_ = decision_index()
    docs = inp["docs"]
    N = len(docs)
    out: dict[str, Any] = {}
    with torch.no_grad():
        P = {
            "psiC": OP.score_matrices(heads["psiC"], inp["S"], inp["C"]),
            "op": OP.score_matrices(heads["op"], inp["S"], inp["C"]),
            "ls": OP.score_matrices(heads["ls"], inp["S"].double(), inp["C"].double()),
        }
        for k, X in P.items():
            out[k] = X[:, T.unsqueeze(1), I_]
        es = []
        for n in range(N):
            es.append(es_scores(es_pol.spans(inp["S"][n]), T, I_).double())
        out["es"] = torch.stack(es)
    ceil, ri, F, Q, d_same, td_same, ri_same = [], [], [], [], [], [], []
    for doc in docs:
        dd = torch.tensor(discounted_demand(doc, GAMMA), dtype=torch.float64)
        ceil.append(reference_F(dd, T, I_))
        Dd = D[doc.doc_id]
        ri.append(gather_steps(Dd, T, I_))
        Delta = inp["blob"]["Delta"][doc.doc_id]
        G = OP.target_of(Delta)
        F.append(reference_F(G, T, I_))
        kinds = [s.kind for s in doc.sentences]
        Q.append(torch.tensor([kinds[int(t)] == "query" for t in T]))
        td = torch.tensor(true_demand(doc), dtype=torch.float64)
        d_same.append(gather_steps(Delta, T, I_))
        td_same.append(gather_steps(td, T, I_))
        ri_same.append(gather_steps(Dd, T, I_))
    out["ceiling"] = torch.stack(ceil)
    out["r_i"] = torch.stack(ri)
    age = (I_ - T.unsqueeze(1)).to(torch.float64)
    out["age"] = age.unsqueeze(0).expand(N, -1, -1).clone()
    out["uniform"] = torch.zeros(N, len(T), M, dtype=torch.float64)
    return {
        "scores": out,
        "F": torch.stack(F),
        "Q": torch.stack(Q),
        "bridge": {
            "Delta_same": torch.stack(d_same),
            "true_demand": torch.stack(td_same),
            "r_i": torch.stack(ri_same),
        },
    }


def summarize_set(
    units: dict, positions: list[int], seed: int, scorers, n_boot: int = N_BOOT
) -> dict:
    """Per (pop, stepset, scorer, dir, k): the bootstrap of (agreement - chance) over
    the documents at ``positions`` (`R1.boot_weights(len, seed, 2000)`, paired across
    every scorer, k, population and step set of the document set)."""
    W = R1.boot_weights(len(positions), seed, n_boot)
    idx = torch.tensor(positions, dtype=torch.long)
    out: dict = {}
    for pop in POPS:
        for ss in STEPSETS:
            for name in scorers:
                for d in DIRS:
                    for k in KS:
                        u = units[(name, d, k, pop, ss)][idx]
                        out.setdefault(pop, {}).setdefault(ss, {}).setdefault(
                            name, {}
                        ).setdefault(d, {})[str(k)] = boot_minus_chance(u, W)
    return out


def bridge_set(br: dict, Q: torch.Tensor, seed: int, n_boot: int = N_BOOT) -> dict:
    """E0d-style same-step bottom-1 against ``Delta[t]`` (descriptive, §5 bridge)."""
    N = Q.shape[0]
    units = units_table(
        {"r_i": br["r_i"], "true_demand": br["true_demand"]},
        br["Delta_same"],
        Q,
        ks=(1,),
        dirs=("bottom",),
    )
    W = R1.boot_weights(N, seed, n_boot)
    out: dict = {}
    for pop in POPS:
        for ss in STEPSETS:
            for name in ("r_i", "true_demand"):
                u = units[(name, "bottom", 1, pop, ss)]
                out.setdefault(pop, {}).setdefault(ss, {})[name] = boot_minus_chance(u, W)
    return out


def c_c4(units: dict, F: torch.Tensor) -> dict:
    """C-C4: the uniform scorer's pooled agreement equals pooled chance exactly at every
    k, direction, population and step set; the reference against itself gives 1."""
    out = {"uniform_equals_chance": True, "self_is_one": True, "detail": []}
    N, nd, _ = F.shape
    for pop in POPS:
        cols = pop_cols(pop)
        Fp = F[:, :, cols].reshape(N * nd, len(cols))
        for k in KS:
            for d in DIRS:
                for ss in STEPSETS:
                    u = units[("uniform", d, k, pop, ss)]
                    A = sum(int(x) for x in u[:, 0].tolist())
                    C = sum(int(x) for x in u[:, 1].tolist())
                    if A != C:
                        out["uniform_equals_chance"] = False
                        out["detail"].append(f"uniform {pop} {d}{k} {ss}: {A} != {C}")
                a = agreement_units(Fp, Fp, k, top=(d == "top"))
                tot = sum(int(x) for x in a.tolist())
                if tot != UNITS * N * nd:
                    out["self_is_one"] = False
                    out["detail"].append(
                        f"F self {pop} {d}{k}: {tot / (UNITS * N * nd)!r}"
                    )
    out["ok"] = out["uniform_equals_chance"] and out["self_is_one"]
    return out


def e0d_seen(seed: int) -> dict:
    """E0d's SEEN bottom-1 values (printed beside the bridge, never contrasted)."""
    rows = ledger_rows(E0D_LEDGER)
    v = rows[f"seed{seed}.analysis"]["value"]
    out = {}
    for pop in ("all", "bos_excluded"):
        for sc in ("gated", "pc"):
            r = v["populations"][pop]["stats"][sc]["resample"]
            for f in ("bottom1_full", "bottom1_Q"):
                out[f"{pop}.{sc}.{f}"] = r.get(f)
                out[f"{pop}.{sc}.{f}_ci"] = r.get(f + "_ci")
            out[f"{pop}.{sc}.bottom1_chance"] = r.get("bottom1_chance")
    return out


def measure_seed(seed: int, *, limit: int | None = None, n_boot: int = N_BOOT) -> dict:
    t0 = time.perf_counter()
    inp = CV.load_inputs(seed)
    docs = inp["docs"]
    ids = inp["ids"]
    _, vmap = OP.u_documents(seed)
    V = LR.documents(seed)[2]
    D, dman = load_D(seed)
    c1 = {"gestalt_delta": inp["c1"], "D_manifest": {"ok": dman["ok"]}}
    hp, psic_payload, hp_info = psic_head(seed)
    ho, op_state, ho_info = op_head(seed)
    hl, ls_state, hl_info = ls_head(seed)
    es_pol, es_payload, es_info = es_policy(seed)
    c1["psiC"] = hp_info
    c1["op"] = ho_info
    c1["ls"] = hl_info
    c1["es"] = es_info
    c1["ok"] = bool(inp["c1"]["ok"] and dman["ok"] and ho_info["ok"] and hl_info["ok"])
    heads = {"psiC": hp, "op": ho, "ls": hl}
    if limit is not None:
        inp = {
            **inp,
            "docs": docs[:limit],
            "ids": ids[:limit],
            "S": inp["S"][:limit],
            "C": inp["C"][:limit],
        }
        docs, ids = inp["docs"], inp["ids"]
        train = [p for p in inp["train"] if p < limit]
        val = [p for p in inp["val"] if p < limit]
    else:
        train, val = inp["train"], inp["val"]
    D_use = {d: D[d] for d in ids}
    c2 = c_c2(D_use, {"Delta": {d: inp["blob"]["Delta"][d] for d in ids}}, ids)
    sub_payload, sub_info = substrate_payload(seed)
    c6 = {
        "state": c_c6_state(es_payload, sub_payload),
        "argmin": c_c6_argmin(es_pol, inp["S"], seed),
    }
    c6["ok"] = c6["state"]["ok"] and c6["argmin"]["ok"]
    c5 = c_c5(seed, inp, heads, psic_payload, op_state, ls_state)
    c3 = c_c3(seed, docs, vmap, V, D)
    t1 = time.perf_counter()
    st = score_tensors(inp, D, heads, es_pol)
    F = st["F"]
    finite = {k: bool(torch.isfinite(v).all()) for k, v in st["scores"].items()}
    finite["F"] = bool(torch.isfinite(F).all())
    units = units_table(st["scores"], F, st["Q"])
    c4 = c_c4(units, F)
    every = list(range(len(docs)))
    base_scorers = [s for s in SCORERS if s not in SPLIT_ONLY]
    res = {
        "U": summarize_set(units, every, seed, base_scorers, n_boot),
        "U_val": summarize_set(units, val, seed, SCORERS, n_boot),
        "U_train": summarize_set(units, train, seed, SCORERS, n_boot),
    }
    bridge = bridge_set(st["bridge"], st["Q"], seed, n_boot)
    return {
        "seed": seed,
        "n_documents": len(docs),
        "n_U_val": len(val),
        "n_U_train": len(train),
        "doc_ids_first_last": [ids[0], ids[-1]],
        "controls": {
            "C-C1": c1,
            "C-C2": c2,
            "C-C3": c3,
            "C-C4": c4,
            "C-C5": c5,
            "C-C6": c6,
            "substrate": sub_info,
        },
        "finite": finite,
        "results": res,
        "bridge": bridge,
        "e0d_seen": e0d_seen(seed),
        "conditions": seed_conditions(res),
        "seconds_measure": time.perf_counter() - t1,
        "seconds": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def _flat(prefix: str, tree: dict, out: dict) -> None:
    for k, v in tree.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and not {"point", "lo", "hi"} <= set(v):
            _flat(key, v, out)
        else:
            out[key] = v


def write_rows(led, per: dict[int, dict], es_omitted=()) -> None:
    """Every leaf as a per-seed statistic. The ES row of a seed whose C-C6 failed is
    omitted (None) and the row is a note, never a statistic (C-C6, scoped)."""
    how = "item_1c.py::measure_seed -> boot_minus_chance; seed order 0,1,2"
    leaves: dict[int, dict] = {}
    for s in SEEDS:
        flat: dict = {}
        _flat("", per[s]["results"], flat)
        _flat("bridge", {"U": per[s]["bridge"]}, flat)
        leaves[s] = flat
    for key in leaves[SEEDS[0]]:
        for f in ("point", "lo", "hi", "agreement", "chance", "n_decisions", "dropped"):
            vals = [leaves[s][key].get(f) for s in SEEDS]
            if ".es." in f".{key}.":
                vals = [
                    None if s in es_omitted else v
                    for s, v in zip(SEEDS, vals, strict=True)
                ]
            if all(_finite(v) for v in vals):
                led.stat(f"{key}.{f}", vals, how=how)
            else:
                led.note(f"{key}.{f}", vals, how=how + " (not finite on every seed)")
    for s in SEEDS:
        led.note(
            f"seed{s}.conditions", per[s]["conditions"], how="item_1c.py::seed_conditions"
        )
        led.note(
            f"seed{s}.e0d_seen", per[s]["e0d_seen"], how=f"{E0D_LEDGER} seed{s}.analysis"
        )
        led.note(
            f"seed{s}.n",
            {k: per[s][k] for k in ("n_documents", "n_U_val", "n_U_train")},
            how="item_1c.py::measure_seed",
        )
    led.stat("n_documents", [per[s]["n_documents"] for s in SEEDS], how=how)
    led.stat("seconds_per_seed", [per[s]["seconds"] for s in SEEDS], how="wall clock")


def manifest_config(args: dict) -> dict:
    return {
        "run_id": RUN_ID,
        "item": ITEM,
        "prereg": PREREG,
        "prereg_commit": PREREG_COMMIT,
        "question": QUESTION,
        "falsifier": FALSIFIER,
        "expected": EXPECTED,
        "seeds": list(SEEDS),
        "M": M,
        "S": S,
        "gamma": GAMMA,
        "decisions": f"t in [{T_LO}, {T_HI}), FIFO world, all-live and bos-excluded",
        "ks": list(KS),
        "scorers": list(SCORERS),
        "split_only": list(SPLIT_ONLY),
        "docsets": list(DOCSETS),
        "primary": list(PRIMARY),
        "units": UNITS,
        "bootstrap": {
            "n": N_BOOT,
            "fn": "R1.boot_weights",
            "seed_base": R1.BOOT_SEED_BASE,
        },
        "inputs": {
            "gestalts": str(CV.OP_OUT / "seed{s}" / "gestalts.pt"),
            "gestalt_sha256": CV.GESTALT_SHA,
            "delta": str(Path(OP.REFS.delta_dir) / OP.DELTA_FILE),
            "delta_sha256": OP.DELTA_SHA,
            "D_pt": B0.D_REL.format(c=CKPT, s="{s}"),
            "substrate_manifest": str(SUBSTRATE_MANIFEST),
            "psiC": str(Path(OP.REFS.run2_out) / "psiC/seed{s}/ckpt-003000.pt"),
            "op_phi": str(OP_PHI),
            "op_ledger": str(OP_LEDGER),
            "ls_dir": str(LS_DIR),
            "es": str(ES_PINNED / ES_FILE),
            "e0d_ledger": str(E0D_LEDGER),
        },
        "c_c3": {"docs": C_C3_DOCS, "batch": E0D.BATCH},
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "args": args,
    }


def execute(led, a) -> tuple[Exit, dict]:
    g = Globals(led, reads_substrate=True, args=a)
    failed = g.start()
    esa = es_manifest_check()
    led.note("C-C1.es_manifest.start", esa, how="shasum -a 256 -c (ES pinned)")
    if not esa["ok"]:
        failed.append("C-C1 ES manifest at start")
    per: dict[int, dict] = {}
    if not failed or a.limit is not None:
        for s in a.seeds:
            g.pin_substrate(s)
            per[s] = measure_seed(s, limit=a.limit, n_boot=a.n_boot)
            print(f"1c seed {s} done {per[s]['seconds']:.0f}s", flush=True)
    esb = es_manifest_check()
    led.note("C-C1.es_manifest.end", esb, how="shasum -a 256 -c (ES pinned)")
    if not esb["ok"]:
        failed.append("C-C1 ES manifest at end")
    g.end(sorted(per))
    es_omitted = []
    for s, r in per.items():
        c = r["controls"]
        for name in ("C-C1", "C-C2", "C-C3", "C-C4", "C-C5"):
            if not c[name]["ok"]:
                failed.append(f"{name} seed{s}")
        if not c["C-C6"]["ok"]:
            es_omitted.append(s)
    if a.limit is not None:
        failed.append(f"smoke: --limit {a.limit} (not the registered set)")
    for s, r in per.items():
        led.note(f"seed{s}.controls", r["controls"], how="item_1c.py::measure_seed")
        led.note(f"seed{s}.finite", r["finite"], how="item_1c.py::measure_seed")
    led.note("controls_failed", failed, how="item_1c.py::execute")
    led.note("es_row_omitted_seeds", es_omitted, how="C-C6, scoped to the ES row")
    if sorted(per) == list(SEEDS):
        write_rows(led, per, es_omitted)
    conds = {s: per[s]["conditions"] for s in per} if per else None
    lab = label(conds, controls_ok=not failed)
    led.note("label", lab, how="item_1c.py::label (PREREG §5, tightened)")
    led.note("label_read", not failed, how="False: a control failed, NOT READ (DROPPED)")
    if failed:
        led.status("did_not_run")
        return Exit.DID_NOT_RUN, {"label": lab, "failed": failed}
    bad = [f"seed{s}.{k}" for s in SEEDS for k, v in per[s]["finite"].items() if not v]
    bad += defects(per)
    if bad:
        led.note("defects", bad, how="item_1c.py::defects")
        led.status("failed")
        return Exit.FAIL, {"label": lab, "failed": bad}
    led.status("ok")
    led.verdict(
        falsifier=FALSIFIER,
        outcome="inconclusive",
        detail=f"Tier D diagnostic; registered label {lab}; every control passed",
    )
    return Exit.OK, {"label": lab, "failed": []}


def defects(per: dict[int, dict]) -> list[str]:
    """Exit 1 (§0): a NaN in a reported statistic of the primary set."""
    bad = []
    ds, pop, ss = PRIMARY
    for s, r in per.items():
        tree = r["results"][ds][pop][ss]
        for name, dd in tree.items():
            for d, kk in dd.items():
                for k, v in kk.items():
                    for f in ("point", "lo", "hi"):
                        if not _finite(v.get(f)):
                            bad.append(f"seed{s}.{ds}.{pop}.{ss}.{name}.{d}.{k}.{f}")
    return bad


def _run(argv: list[str] | None = None) -> Exit:
    ap = ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--limit", type=int, default=None, help="smoke only: first N U docs")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    ap.add_argument("--run-id", default=RUN_ID)
    add_global_args(ap)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    from ledger import Ledger

    led = Ledger(a.run_id, question=QUESTION, runs_root=a.runs_root)
    led.run_meta(device="cpu", steps_requested=0, steps_done=0)
    led.manifest(manifest_config({k: str(v) for k, v in vars(a).items()}))
    try:
        rc, _info = execute(led, a)
    except Exception as e:  # recorded; a raise before the controls passed is exit 3
        import traceback

        led.note("exception", traceback.format_exc(), how="item_1c.py::main")
        led.note("label", "DROPPED", how="exception: did not run")
        led.status("crashed")
        sealed = type(e).__name__ in ("SealedRange", "ControlFailure")
        rc = Exit.DID_NOT_RUN if sealed else Exit.FAIL
    led.command(
        [sys.executable, RUN_PY, ITEM] + (argv if argv is not None else sys.argv[1:]),
        exit_code=int(rc),
        note="its exit code is the item's",
    )
    led.write()
    return status(rc)


# --------------------------------------------------------------------------- #
# finalize: RESULTS.md and claims.json from the ledger (no number typed)
# --------------------------------------------------------------------------- #


def _row(rows: dict, key: str):
    r = rows.get(key)
    if r is None:
        return None
    return r.get("samples", r.get("value"))


def _fmt(x, nd: int = 4) -> str:
    return "n/a" if not _finite(x) else f"{x:.{nd}f}"


def render(out_root: Path = OUT_DEFAULT) -> Path:
    """Render `RESULTS.md` and `claims.json` next to the ledger, from the ledger."""
    led = json.loads((Path(out_root) / "ledger.json").read_text())
    rows = {r["key"]: r for r in led["rows"]}
    lab = rows["label"]["value"]
    read = rows["label_read"]["value"]
    failed = rows["controls_failed"]["value"]
    cmd = led["commands"][-1] if led["commands"] else {}
    L = [
        "# post-fails-controls 1c: top-k / bottom-k agreement, demand strictly after t",
        "",
        f"Generated by `{EXPERIMENT}::finalize` from `{Path(out_root) / 'ledger.json'}`. "
        "No number below is typed.",
        "",
        f"- PREREG: `{PREREG}` @ {PREREG_COMMIT} (§5)",
        f"- git sha: `{led['provenance'].get('git_sha')}` · device: {led['device']} · "
        f"seeds actually run: {led['seeds_actually_run']} · status: {led['status']}",
        f"- command: `{cmd.get('argv')}` · exit code: {cmd.get('exit_code')}",
        f"- config hash: `{led['config_hash']}`",
        f"- controls failed: {failed if failed else 'none'}",
        "",
        f"## Registered label: **{lab}**"
        + ("" if read else " (NOT READ: a control failed)"),
        "",
        "Rule (§5, tightened): SURVIVES iff, on all 3 seeds and at k = 1 and k = 3 on "
        "the primary set (U, all-live, all decisions t in [16, 47), gamma 0.9): the "
        "ceiling's bottom-k CI lower bound > chance; r_i's top-k CI lower bound > "
        "chance; r_i's bottom-k CI upper bound <= chance. Otherwise DROPPED.",
        "",
        "| seed | k | ceiling bottom lo > chance | r_i top lo > chance | "
        "r_i bottom hi <= chance |",
        "|---|---|---|---|---|",
    ]
    for s in SEEDS:
        c = (rows.get(f"seed{s}.conditions") or {}).get("value") or {}
        for k in KS:
            ck = c.get(str(k), {})
            L.append(
                f"| {s} | {k} | {ck.get('ceiling_bottom_lo_above_chance')} | "
                f"{ck.get('r_i_top_lo_above_chance')} | "
                f"{ck.get('r_i_bottom_hi_at_or_below_chance')} |"
            )
    L += [
        "",
        "## Agreement minus chance, per seed: point [95% CI] (agreement; chance)",
        "",
    ]
    for ds in DOCSETS:
        for pop in POPS:
            for ss in STEPSETS:
                L += [
                    f"### {ds} · {pop} · steps {ss}",
                    "",
                    "| scorer | dir | k | seed 0 | seed 1 | seed 2 |",
                    "|---|---|---|---|---|---|",
                ]
                for name in SCORERS:
                    for d in DIRS:
                        for k in KS:
                            base = f"{ds}.{pop}.{ss}.{name}.{d}.{k}"
                            pt = _row(rows, base + ".point")
                            if pt is None:
                                continue
                            lo, hi = _row(rows, base + ".lo"), _row(rows, base + ".hi")
                            ag, ch = (
                                _row(rows, base + ".agreement"),
                                _row(rows, base + ".chance"),
                            )
                            cells = [
                                f"{_fmt(pt[i])} [{_fmt(lo[i])}, {_fmt(hi[i])}] "
                                f"({_fmt(ag[i])}; {_fmt(ch[i])})"
                                for i in range(len(SEEDS))
                            ]
                            L.append(
                                f"| {name} | {d} | {k} | " + " | ".join(cells) + " |"
                            )
                L.append("")
    L += [
        "## Bridge to E0d (descriptive; different documents, printed beside, never "
        "contrasted)",
        "",
        "Same-step bottom-1 against Delta[t] on U decisions t in [16, 47).",
        "",
        "| pop | steps | scorer | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|---|---|",
    ]
    for pop in POPS:
        for ss in STEPSETS:
            for name in ("r_i", "true_demand"):
                base = f"bridge.U.{pop}.{ss}.{name}"
                ag, pt = _row(rows, base + ".agreement"), _row(rows, base + ".point")
                lo, hi = _row(rows, base + ".lo"), _row(rows, base + ".hi")
                if ag is None:
                    continue
                cells = [
                    f"{_fmt(ag[i])} (minus chance {_fmt(pt[i])} [{_fmt(lo[i])}, "
                    f"{_fmt(hi[i])}])"
                    for i in range(len(SEEDS))
                ]
                L.append(f"| {pop} | {ss} | {name} | " + " | ".join(cells) + " |")
    L += ["", "E0d's SEEN values (`runs/e0d/ledger.json`, chance as E0d stated it):", ""]
    for s in SEEDS:
        v = (rows.get(f"seed{s}.e0d_seen") or {}).get("value") or {}
        L.append(
            f"- seed {s}: "
            + "; ".join(
                f"{k} = {_fmt(x) if _finite(x) else x}"
                for k, x in v.items()
                if not k.endswith("_ci")
            )
        )
    L += ["", "## Controls", ""]
    for s in SEEDS:
        c = (rows.get(f"seed{s}.controls") or {}).get("value") or {}
        L.append(
            f"- seed {s}: "
            + ", ".join(
                f"{k} {'pass' if (v or {}).get('ok') else 'FAIL'}"
                for k, v in c.items()
                if isinstance(v, dict) and "ok" in v
            )
        )
    es_om = (rows.get("es_row_omitted_seeds") or {}).get("value")
    L += [f"- ES row omitted on seeds: {es_om if es_om else 'none'} (C-C6, scoped)", ""]
    L += [
        "## What 1c does NOT show (PREREG §5, verbatim pointers)",
        "",
        "- F is FIFO-world LOO Delta: demand under FIFO's memory, not under any "
        "scorer's own memory.",
        "- Agreement at the top or bottom is not eviction quality; no scorer is "
        "deployed here.",
        "- One corpus, one width, in-sample in documents (U is already inspected); "
        "no claim about real models.",
        "- The psi rows are descriptive; offline-psi and LS were fit on U-train.",
        "",
    ]
    p = Path(out_root) / "RESULTS.md"
    p.write_text("\n".join(L) + "\n")
    write_claims(out_root, rows)
    return p


def write_claims(out_root: Path, rows: dict) -> Path:
    """`claims.json`: one runnable recompute per checkable claim (rsr-verifier)."""
    base = (
        "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run --lane cpu-det "
        "--slots 1 --wait 3600 -- .venv/bin/python "
        f"{RUN_PY} {ITEM} recompute"
    )
    claims = [
        {
            "claim": f"1c registered label is {rows['label']['value']}",
            "command": f"{base} --what label",
            "expected": rows["label"]["value"],
        }
    ]
    for s in SEEDS:
        for k in KS:
            for name, d in (("ceiling", "bottom"), ("r_i", "top"), ("r_i", "bottom")):
                key = f"U.all_live.all.{name}.{d}.{k}"
                for f in ("point", "lo", "hi"):
                    v = _row(rows, f"{key}.{f}")
                    if v is None:
                        continue
                    claims.append(
                        {
                            "claim": f"seed {s} {key}.{f} = {v[SEEDS.index(s)]!r}",
                            "command": f"{base} --what {key}.{f} --seed {s}",
                            "expected": v[SEEDS.index(s)],
                        }
                    )
    p = Path(out_root) / "claims.json"
    p.write_text(json.dumps(claims, indent=1) + "\n")
    return p


def recompute(what: str, seed: int | None, n_boot: int = N_BOOT) -> str:
    """Re-run one seed's measurement and print one value (the verifier's entry)."""
    if what == "label":
        conds = {s: measure_seed(s, n_boot=n_boot)["conditions"] for s in SEEDS}
        return label(conds, controls_ok=True)
    r = measure_seed(int(seed), n_boot=n_boot)
    flat: dict = {}
    _flat("", r["results"], flat)
    _flat("bridge", {"U": r["bridge"]}, flat)
    key, f = what.rsplit(".", 1)
    return repr(flat[key][f])


def finalize(argv: list[str] | None = None) -> int:
    """``run.py finalize 1c [--out DIR]``: RESULTS.md and claims.json, from the ledger
    in DIR (default ``runs/{RUN_ID}``). No number is typed."""
    ap = ArgumentParser(description="finalize 1c")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args(list(argv or []))
    print(render(a.out))
    return int(Exit.OK)


def main(argv: list[str] | None = None) -> int:
    """``run.py 1c [measure] [options]`` measures and writes the ledger;
    ``run.py 1c finalize`` / ``run.py finalize 1c`` renders; ``run.py 1c recompute
    --what KEY [--seed S]`` re-derives one value (the verifier's entry)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        return finalize(argv[1:])
    if argv[:1] == ["recompute"]:
        ap = ArgumentParser(description="recompute one 1c value")
        ap.add_argument("--what", required=True)
        ap.add_argument("--seed", type=int, default=None)
        a = ap.parse_args(argv[1:])
        common().install_seal()
        print(recompute(a.what, a.seed))
        return 0
    if argv[:1] == ["measure"]:
        argv = argv[1:]
    return int(_run(argv))


if __name__ == "__main__":
    run_main(main)
