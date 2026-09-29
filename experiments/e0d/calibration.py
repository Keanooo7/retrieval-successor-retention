"""E0d PREREG Amendment 2, Part B.1: the calibration fixture and injected proxies.

`synthetic_ledger(...)` returns per-cell arrays shaped like `run.cells_from` output
(`doc, t, rank, sentence, n_live, full, q_step, a_cell, gap, d_resample, d_zero`, plus
`pc`, `kind`, `query_of` and the fixture's own truth column `crit`). It is pure numpy,
deterministic (`default_rng(seed)`), uses no model and no document, and never touches
`D_E0d = [262144, 263168)`: there are no documents here at all, only numbers.

It is matched to set E seed 0 (`amendment-2/reviews/e0d-a2-final/m1.log`, `m5_out.json`),
as Part B.1 of `amendment-2/E0D-AMENDMENT-2-FINAL-v3.md` specifies:
- M = 16, S = 48. At t >= 16 there is one slot per age 1-16, with rank = 16 - age.
- About 43 % of full-memory steps are Q-steps. Per Q-step, k_t is 0 (26 %), 1 (72 %),
  2 (2 %) or 3 (< 1 %) under `k_profile="matched"`, and always 1 under "one_per_step".
- Critical cells by age follow the U-shaped counts of `AGE_PROFILE`; about 92 % are
  A-cells.
- Critical Delta is log-normal (median 1.2, q10 0.62). Off-critical |Delta| is log-normal
  (q50 0.0094, q90 0.081) with 58 % positive signs (`noise="continuous"`), or exactly 0
  (`noise="tied"`).
- NaN `d_resample`: under `nan_mode="matched"`, 11/663 on A-cells and 105/13,481 on
  other cells, so NaN depends on y. Under "independent", 0.8 % of cells regardless of y.
- There is a gap-1 A-cell at rank 15, so A1.3 has something to remove.

The fixture fixes the truth, `crit`. Delta is drawn so that `Delta > tau` iff `crit`
wherever Delta has a value: critical Delta >= 1.05 tau, off-critical |Delta| <= 0.95 tau.
The runner's `y = 1[Delta > tau]` therefore reproduces `crit` exactly on labelled cells.

PREREG-OPEN (fixture choices the draft does not fix): an off-critical k = 0 Q-step holds
a resident, non-critical A-cell with probability 0.7; non-A critical cells are never at
age 1 (the newest slot), since a critical non-A cell there would be dropped by A1.3.
"""

from __future__ import annotations

import math

import numpy as np

M, S = 16, 48
#: m1.log (set E seed 0, 64 docs): critical cells by age 1..16.
AGE_PROFILE = (166, 115, 92, 81, 38, 27, 24, 19, 11, 14, 2, 22, 14, 19, 20, 19)
Q_FRAC = 0.43
K_PROFILE = ((0, 0.26), (1, 0.72), (2, 0.015), (3, 0.005))
A_SHARE = 0.92
K0_A_RESIDENT = 0.7
CRIT_MEDIAN, CRIT_Q10 = 1.2, 0.62
OFF_Q50, OFF_Q90, OFF_POS = 0.0094, 0.081, 0.58
NAN_A, NAN_OTHER = 11 / 663, 105 / 13_481
NAN_INDEPENDENT = 0.008
#: seed 0's frozen tau (PREREG A2.2); labels hold at any tau the fixture is built with.
TAU_DEFAULT = 0.4157434984576128
_Z90 = 1.2815515655446004


def _lognormal(rng, median, sigma, n):
    return np.exp(math.log(median) + sigma * rng.standard_normal(n))


def synthetic_ledger(
    seed: int,
    n_docs: int,
    noise: str = "continuous",
    k_profile: str = "matched",
    nan_mode: str = "matched",
    *,
    tau: float = TAU_DEFAULT,
) -> dict:
    if noise not in ("continuous", "tied"):
        raise ValueError(noise)
    if k_profile not in ("matched", "one_per_step"):
        raise ValueError(k_profile)
    if nan_mode not in ("matched", "independent"):
        raise ValueError(nan_mode)
    rng = np.random.default_rng(seed)
    prof = np.asarray(AGE_PROFILE, dtype=float)
    p_age = prof / prof.sum()  # ages 1..16
    p_age_non1 = prof[1:] / prof[1:].sum()  # ages 2..16

    # ---- full-memory steps: (doc, t), t in [M, S) ------------------------------
    fd = np.repeat(np.arange(n_docs), S - M)
    ft = np.tile(np.arange(M, S), n_docs)
    n_steps = len(fd)
    is_q = rng.random(n_steps) < Q_FRAC
    if k_profile == "matched":
        ks, ps = zip(*K_PROFILE, strict=True)
        k = np.where(is_q, rng.choice(ks, size=n_steps, p=ps), 0)
    else:
        k = np.where(is_q, 1, 0)
    gap = np.full(n_steps, -1)
    crit = np.zeros((n_steps, M), dtype=bool)  # by age index 0..15 (age = idx + 1)
    for j in np.flatnonzero(is_q):
        a_age = int(rng.choice(np.arange(1, M + 1), p=p_age))
        if k[j] >= 1:
            a_resident = rng.random() < A_SHARE or k_profile == "one_per_step"
            if a_resident:
                gap[j] = a_age
                crit[j, a_age - 1] = True
            else:  # the critical cell is a non-A slot; the assert is not resident
                gap[j] = int(rng.integers(M + 1, M + 9))
                crit[j, int(rng.choice(np.arange(2, M + 1), p=p_age_non1)) - 1] = True
            extra = int(k[j]) - 1
            while extra > 0:
                b = int(rng.choice(np.arange(2, M + 1), p=p_age_non1))
                if not crit[j, b - 1] and b != gap[j]:
                    crit[j, b - 1] = True
                    extra -= 1
        else:
            resident = rng.random() < K0_A_RESIDENT
            gap[j] = a_age if resident else int(rng.integers(M + 1, M + 9))

    # cells of the full steps: rank i = 0..15, age = M - i
    rank = np.tile(np.arange(M), n_steps)
    age = M - rank
    sd = np.repeat(np.arange(n_steps), M)
    c_crit = crit[sd, age - 1]
    c_q = is_q[sd]
    c_gap = np.where(c_q, gap[sd], -1)
    c_a = c_q & (age == gap[sd])
    full_cells = {
        "doc": fd[sd],
        "t": ft[sd],
        "rank": rank,
        "sentence": ft[sd] - age,
        "n_live": np.full(len(sd), M),
        "q_step": c_q,
        "a_cell": c_a,
        "gap": c_gap,
        "crit": c_crit,
    }
    # ---- underfull steps t in [1, M): FIFO, slot i holds sentence i ------------
    ut, ui = [], []
    for t in range(1, M):
        ut += [t] * t
        ui += list(range(t))
    ut, ui = np.asarray(ut), np.asarray(ui)
    ud = np.repeat(np.arange(n_docs), len(ut))
    nu = len(ud)
    under = {
        "doc": ud,
        "t": np.tile(ut, n_docs),
        "rank": np.tile(ui, n_docs),
        "sentence": np.tile(ui, n_docs),
        "n_live": np.tile(ut, n_docs),
        "q_step": np.zeros(nu, dtype=bool),
        "a_cell": np.zeros(nu, dtype=bool),
        "gap": np.full(nu, -1),
        "crit": np.zeros(nu, dtype=bool),
    }
    c = {k_: np.concatenate([under[k_], full_cells[k_]]) for k_ in full_cells}
    order = np.lexsort((c["rank"], c["t"], c["doc"]))
    c = {k_: v[order] for k_, v in c.items()}
    n = len(c["doc"])
    c["full"] = c["n_live"] == M

    # ---- Delta ------------------------------------------------------------------
    sig_c = (math.log(CRIT_MEDIAN) - math.log(CRIT_Q10)) / _Z90
    sig_o = (math.log(OFF_Q90) - math.log(OFF_Q50)) / _Z90
    crit_d = np.maximum(_lognormal(rng, CRIT_MEDIAN, sig_c, n), 1.05 * tau)
    if noise == "continuous":
        mag = np.minimum(_lognormal(rng, OFF_Q50, sig_o, n), 0.95 * tau)
        sign = np.where(rng.random(n) < OFF_POS, 1.0, -1.0)
        off = sign * mag
    else:
        off = np.zeros(n)
    d = np.where(c["crit"], crit_d, off)
    dz = np.where(c["crit"], crit_d * rng.uniform(0.8, 1.2, n), off * 1.1)
    if nan_mode == "matched":
        p_nan = np.where(c["a_cell"], NAN_A, NAN_OTHER)
    else:
        p_nan = np.full(n, NAN_INDEPENDENT)
    d = np.where(rng.random(n) < p_nan, np.nan, d)
    c["d_resample"] = d
    c["d_zero"] = dz
    c["pc"] = c["a_cell"].astype(float)
    c["kind"] = np.where(c["a_cell"], "pending_assert", "filler").astype(object)
    c["query_of"] = np.full(n, -1)
    c["tau"] = tau
    return c


# --------------------------------------------------------------------------- #
# injected proxies: each returns one r per cell; each step's shares sum to 1
# unless stated
# --------------------------------------------------------------------------- #


def step_ids(c: dict) -> np.ndarray:
    key = np.asarray(c["doc"], dtype=np.int64) * (S + 1) + np.asarray(c["t"])
    return np.unique(key, return_inverse=True)[1]


def softmax_steps(x, step) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ns = int(step.max()) + 1
    mx = np.full(ns, -np.inf)
    np.maximum.at(mx, step, x)
    e = np.exp(x - mx[step])
    return e / np.bincount(step, weights=e, minlength=ns)[step]


def _crit(c):
    return np.asarray(c["crit"], dtype=float)


def _age(c):
    return np.asarray(c["t"]) - np.asarray(c["sentence"])


def f_age(c) -> np.ndarray:
    """log critical rate by age over full-memory Q-step cells (m6's form)."""
    age = _age(c)
    sel = np.asarray(c["full"]) & np.asarray(c["q_step"])
    lr = np.zeros(M + 1)
    for a in range(1, M + 1):
        m = sel & (age == a)
        lr[a] = math.log((c["crit"][m].sum() + 1) / (m.sum() + 1))
    return lr[np.clip(age, 0, M)]


def u_age(c) -> np.ndarray:
    v = f_age(c)
    return (v - v.mean()) / (v.std() + 1e-12)


def perfect_binary(c, rng=None):
    return _crit(c)


def perfect_continuous(c, rng):
    return content(c, rng, 10.0)


def content(c, rng, a):
    return softmax_steps(a * _crit(c) + rng.standard_normal(len(_crit(c))), step_ids(c))


def age_steered(c, rng, a):
    x = a * _crit(c) + u_age(c) + rng.standard_normal(len(_crit(c)))
    return softmax_steps(x, step_ids(c))


def age_only(c, rng=None):
    return f_age(c)


def recency(c, rng=None):
    return -_age(c).astype(float)


def shuffle(c, rng):
    """A within-step permutation of a plausible score, softmax(1.5 y + eps)."""
    step = step_ids(c)
    base = softmax_steps(1.5 * _crit(c) + rng.standard_normal(len(step)), step)
    grouped = np.lexsort((np.arange(len(step)), step))  # cells by step, in order
    permuted = np.lexsort((rng.random(len(step)), step))  # by step, random order
    out = np.empty_like(base)
    out[grouped] = base[permuted]  # each step's values, permuted within the step
    return out


def step_has_crit(c) -> np.ndarray:
    step = step_ids(c)
    has = np.bincount(step, weights=_crit(c)) > 0
    return has[step]


def step_temperature(c, rng, a=2.0, *, reverse=False, eps=None):
    """softmax(T_t (a y + eps)), T_t = 0.05 on steps with a critical cell, 6 elsewhere
    (and the reverse). `eps` may be passed to share draws with `content`."""
    has = step_has_crit(c)
    lo, hi = (6.0, 0.05) if reverse else (0.05, 6.0)
    T = np.where(has, lo, hi)
    e = rng.standard_normal(len(has)) if eps is None else eps
    return softmax_steps(T * (a * _crit(c) + e), step_ids(c))


def step_concentration(c, rng):
    """a = 0: flat on steps with a critical cell, peaked on a random slot elsewhere."""
    has = step_has_crit(c)
    return softmax_steps(np.where(has, 0.0, 50.0) * rng.standard_normal(len(has)),
                         step_ids(c))  # fmt: skip


def rescaled(c, rng, r):
    step = step_ids(c)
    ct = rng.uniform(0.5, 2.0, int(step.max()) + 1)
    return np.asarray(r, dtype=float) * ct[step]


def with_score(c: dict, r) -> dict:
    """`c` with `r` as both the gated and ungated score, and the descriptive columns
    `analyse_seed` reads."""
    out = dict(c)
    r = np.asarray(r, dtype=np.float64)
    out["r_gated"], out["r_ungated"] = r, r.copy()
    out["layer_gated"] = np.stack([r, r], 1)
    out["layer_ungated"] = np.stack([r, r], 1)
    out["pre_share_gated"] = np.ones(len(r))
    out["pre_share_ungated"] = np.ones(len(r))
    return out
