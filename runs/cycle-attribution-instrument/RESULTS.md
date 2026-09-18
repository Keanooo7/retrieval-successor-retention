# cycle-attribution-instrument — `EvictionRecord.attribution` reports its configuration, not its effect

- **Falsifier.** *"`EvictionRecord.attribution` distinguishes evictions that `b`
  decided from evictions that `b` did not."*
- **Verdict: FALSIFIED.** Decisively, with two of four confusion cells empty in
  every arm and every seed, and `I(attribution ; b-changed-the-victim) = 0.0000
  bits` against a ground truth carrying up to **0.9877 bits** of entropy.
- Ledger: `runs/cycle-attribution-instrument/ledger.json` — **254 rows, every
  number below is one of them.** 87 statistic rows have `sd == 0.0` and all 87 are
  classified by key in the `sd_zero_rows_reasons` row; `sd_zero_rows_unclassified`
  is `[]`.
- Provenance recorded by the ledger itself: `git_sha 026bef1`, **`git_dirty:
  false`** — stamped at process start, before this run directory existed. The only
  untracked path now is `runs/cycle-attribution-instrument/` itself; `src/`,
  `tests/` and `docs/` are untouched, and `git status --porcelain` shows nothing
  else. Python 3.12.13, macOS-26.6.2-arm64. CPU, 1 thread, **4.3 s wall**.
- ⚠️ **The brief names `4866e7b` as the baseline; `HEAD` is `026bef1`**, one commit
  ahead. That commit touches only `Projects/RSR/for-brendan-2026-09-18.md`:
  `git diff --name-only 4866e7b 026bef1 -- src tests docs` is empty. Reported
  rather than assumed away.
- Scratch harness (throwaway, per the rules):
  `/private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_attribution_instrument.py`,
  reusing cycle 2's `scratch_bias.py`. `src/`, `tests/`, `docs/` untouched; nothing
  committed.
- **`ū` is synthetic and every rate below is conditional on it.**
  `RSRPolicy.observe()` still raises `NotImplementedError`
  (`src/rsr/retention/rsr.py:433`), so no real utilisation signal exists. `ū` is a
  per-slot log-normal salience assigned at write time plus `N(0, 0.5)` noise,
  softmaxed over live slots. Ledger row `u_bar_is_synthetic`.
- Geometry **read** from the registry, not supplied: `M = 40`, `S = 80`
  (`pg19_e3`, FROZEN), `b_max = 1.0` (FROZEN). `d_model = 384`.
- **Deliberate visible bypasses**, labelled as such in the ledger and *not*
  measurements of those constants: `γ_b ∈ {0, 0.001, 0.05, 0.1}` (DERIVED, E0e),
  `ν ∈ {0, 0.1, 0.5, 1.0}` (MEASURED, E1), `τ = 0.25`, EMA half-life `10.0`,
  `RSRConfig` hand-built rather than `from_registry`, and — carried over from
  cycle 2 — a `b(slots)` accessor added to the bias object, without which no
  `b_enabled=True` eviction can run at all (`AttributeError` at `rsr.py:377`).

## Design

Cycle 2's harness, generalized. One **driver** arm (`b_enabled=False`, `nu=0.0`)
owns the memory, so the `MemoryState` sequence is *identical* for every arm; every
other arm is handed the same state and the same context and **shares one
`BilinearValueHead` object**, so `ψ̂` is bit-identical. Ground truth for "did `b`
decide this eviction" is therefore exact and per-eviction: the arm's victim differs
from the driver's. `psi_override=None`, `t_warm=0.0`, `shadow_enabled=False`.
5 seeds × 12 streams × (80 − 40) = **480 evictions/seed, 2400 per arm**.

Two stream regimes, because `ν`'s term is `max_{j≠i} cos(s_i, s_j)` and 40 iid unit
vectors in `d = 384` are near-orthogonal:

| regime | `mean_max_cosine_over_live_memory` |
|---|---|
| `iid` (cycle 2's stream) | **0.16514 ± 0.00478** |
| `redundant` (25% of writes a near-copy of a live slot, 0.15 noise) | **0.38087 ± 0.00467** |

**Cross-cycle replication.** The `iid` regime reproduces cycle 2 exactly:
`frac_term_actually_changed_victim` is **0.02125 ± 0.00539** at `γ_b = 0.001`,
**0.56417 ± 0.01313** at 0.05 and **0.59500 ± 0.01289** at 0.1 — the same three
numbers to all reported digits. The harness is the same harness.

## A — the confusion matrix (seeds 0–4 pooled, 2400 evictions per arm)

Rows: does `attribution` contain `"+b"`. Columns: did `b` change the victim.

### `iid`, `γ_b = 0.0` — **`b` is identically zero and the record still says `psi+b`**

| | changed | NOT changed |
|---|---|---|
| `"+b"` in attribution | **0** | **2400** |
| no `"+b"` | 0 | 0 |

`frac_attribution_carries_token = 1.0000 ± 0.0000`;
`frac_term_actually_changed_victim = 0.0000 ± 0.0000`. This is the whole finding in
one table: an arm in which `b ≡ 0` — which is *also* §3.7's reduction condition —
is logged on all 2400 evictions as one the balance controller participated in.

### `iid`, `γ_b = 0.001` (defect D-1's value)

| | changed | NOT changed |
|---|---|---|
| `"+b"` | **51** | **2349** |
| no `"+b"` | 0 | 0 |

`frac_attribution_carries_token = 1.0000 ± 0.0000` vs
`frac_term_actually_changed_victim = 0.02125 ± 0.00539`. **The brief predicted
`"psi+b"` on the 97.9% where `b` changed nothing; measured 97.875%** (2349/2400).

### `iid`, `γ_b = 0.05` and `0.1` (correction 4's own range)

| `γ_b` | `"+b"` ∧ changed | `"+b"` ∧ NOT changed | no `"+b"` (either) |
|---|---|---|---|
| 0.05 | 1354 | **1046** | 0 |
| 0.1 | 1428 | **972** | 0 |

`frac_changed = 0.56417 ± 0.01313` and `0.59500 ± 0.01289`. Even where `b` *is* the
policy, **40.5% ± 1.3% of the `"psi+b"` labels are false** — so the string cannot
be used in either direction.

### The `redundant` regime is the same story

`γ_b = 0.001`: 67 / 2333 / 0. `γ_b = 0.05`: 1351 / 1049 / 0. `γ_b = 0.1`:
1391 / 1009 / 0. `frac_attribution_carries_token = 1.0000 ± 0.0000` throughout.

### Information-theoretic form

Per seed, from the empirical 2×2 joint:

| regime / arm | `mutual_information_bits` | `entropy_of_ground_truth_bits` | `conditional_entropy_given_attribution_bits` |
|---|---|---|---|
| `iid` `b@0.001` | **0.0000 ± 0.0000** | 0.14757 ± 0.03012 | **0.14757 ± 0.03012** |
| `iid` `b@0.05` | **0.0000 ± 0.0000** | 0.98768 ± 0.00436 | **0.98768 ± 0.00436** |
| `iid` `b@0.1` | **0.0000 ± 0.0000** | 0.97340 ± 0.00683 | **0.97340 ± 0.00683** |
| `redundant` `b@0.1` | **0.0000 ± 0.0000** | 0.97967 ± 0.01560 | **0.97967 ± 0.01560** |

`H(changed | attribution) = H(changed)` to every reported digit, in every arm, in
every seed. Conditioning on `attribution` removes **nothing**. As a detector,
`"+b"` has `recall = 1.0000 ± 0.0000` and `precision` equal to the base rate
exactly (`precision_of_token_as_detector` and
`frac_term_actually_changed_victim` carry identical per-seed samples): it is the
rule *always say yes*.

**Mechanism, and it is not subtle.** `rsr.py:375–381` reads
`if self.config.b_enabled: ... terms += "+b"` and
`if self.config.nu != 0.0: ... terms += "-nu"`. Measured consequence:
`n_distinct_attribution_strings_max_over_arms = 1.0000 ± 0.0000` over all 16
(regime × arm) runs — **no arm ever emitted a second attribution string across its
480 evictions.** `attribution` is a pure function of `RSRConfig`. It is a config
echo with a per-row timestamp.

It is not that `b` was flat, either:
`frac_evictions_b_spread_positive_gamma_b_0p001 = 1.0000 ± 0.0000` — `b` was
non-uniform over live slots at essentially every eviction, and still the record
says the same thing.

## B — `nu` has the identical defect, with one asymmetry worth keeping

`"-nu"` is appended on the same configuration-not-effect basis, and the measurement
matches: `frac_attribution_carries_token = 1.0000 ± 0.0000` at every `ν > 0` while

| regime | `ν = 0.1` | `ν = 0.5` | `ν = 1.0` |
|---|---|---|---|
| `iid` | 0.00333 ± 0.00186 | 0.01167 ± 0.00635 | 0.02458 ± 0.00681 |
| `redundant` | 0.01250 ± 0.00390 | 0.05667 ± 0.01636 | 0.11417 ± 0.01801 |

(`frac_term_actually_changed_victim`.) `mutual_information_bits = 0.0000 ±
0.0000` in all six arms; at `redundant`/`ν = 1.0` that is against
`entropy_of_ground_truth_bits = 0.51054 ± 0.05212`. So `"psi-nu"` is written on
**274 evictions it explains and 2126 it does not**, pooled over five seeds.

**The asymmetry.** The `ν` guard is `config.nu != 0.0`, and `0.0` *is* §3.7's off
value, so the `ν = 0` control arm correctly emits no `"-nu"`
(`selftest_attribution_nu_zero` = `["psi"]`, `frac_attribution_carries_token =
0.0000 ± 0.0000`). `b`'s guard is `config.b_enabled`, which is a *different knob
from `γ_b`* — so `b_enabled=True, γ_b=0` is a provably inert bias reported as
active, and no config value of `b_enabled` can express it. `ν`'s tag is wrong about
*effect*; `b`'s tag is wrong about *presence*. That makes `b` the worse of the two,
and it is the one §3.4's test depends on.

**Secondary finding, recorded because it will be read as a null otherwise.** In the
`iid` regime `ν = 1.0` changes only **2.46% ± 0.68%** of victims, because 40 iid
unit gestalts in `d = 384` have `mean max cosine 0.165`. The redundancy term is
**4.6× more active** in the `redundant` regime at the same `ν`. An E1 `ν` sweep run
on a corpus without near-duplicate gestalts will measure `ν ≈ inert` and that will
be a property of the corpus, not of `ν`.

## C — is `score_margin` sufficient on its own? **No. Measured.**

`score_margin` is the *post-term* top-2 margin — the only margin any arm logs.
Treating `−score_margin` as a detector of "the term changed the victim":

| regime / arm | `auc_score_margin_predicts_change` | `min_threshold_error_from_margin_alone` | `base_rate_error_always_majority` | `overlap_changed_inside_unchanged_margin_range` |
|---|---|---|---|---|
| `iid` `b@0.001` | 0.97226 ± 0.00911 | 0.01875 ± 0.00589 | 0.02125 ± 0.00539 | **0.94697 ± 0.08053** |
| `iid` `b@0.05` | 0.68573 ± 0.02936 | **0.31500 ± 0.02427** | 0.43583 ± 0.01313 | **0.99169 ± 0.01072** |
| `iid` `b@0.1` | 0.69935 ± 0.03222 | **0.29500 ± 0.02304** | 0.40500 ± 0.01289 | **0.99083 ± 0.00741** |
| `redundant` `b@0.1` | 0.67156 ± 0.03909 | 0.32042 ± 0.04341 | 0.42042 ± 0.02877 | **0.99636 ± 0.00373** |
| `redundant` `nu@1` | 0.86925 ± 0.01402 | 0.10833 ± 0.01749 | 0.11417 ± 0.01801 | 0.96341 ± 0.01886 |

**`score_margin` is informative but not sufficient, and the gap is worst exactly
where it matters.**

1. **No threshold separates the classes.** `overlap` is the fraction of flipped
   evictions whose logged margin lies inside the `[min, max]` range of unflipped
   ones: **0.947–0.996**. Sufficiency would require a margin value that decides the
   question; 99% of the flipped cases sit inside the unflipped range.
2. **In correction 4's own range the best possible margin rule still misreads
   ~30% of decisions** (0.295–0.320 vs a 0.405–0.436 base rate). That is a real
   reduction — the margin is not noise — but a §3.4 test that gets 30% of its
   decisions wrong is not an adjudication.
3. **The high AUCs at small `γ_b` and small `ν` are a base-rate illusion.** At
   `γ_b = 0.001` AUC is 0.972, and yet `min_threshold_error` (0.01875) barely beats
   "assume nothing ever flipped" (0.02125). Ranking well on a 2% positive class
   buys almost no decisions.
4. **The one sound thing `score_margin` gives you is a one-sided bound.**
   `frac_changed_with_margin_above_2bmax = 0.0000 ± 0.0000` in all six `b` arms:
   no eviction `b` flipped ever carried a logged margin above `2·b_max = 2.0`. So
   `margin > 2·b_max` ⇒ `b` did not decide it, reliably. The converse is worthless,
   and that is the direction §3.4 needs.
5. **The counterfactual margin is not better, and that is itself the point.**
   `auc_counterfactual_margin_predicts_change` — the `b`-disabled driver's margin,
   which **no arm's log contains today** — is 0.64297 ± 0.03061 at `γ_b = 0.05` and
   0.62301 ± 0.04014 at 0.1, i.e. *worse* than the arm's own post-`b` margin. So
   the fix is **not** "also log the pre-bias margin." Margins are the wrong object;
   the counterfactual *victim* is the right one.

## D — specification of a sufficient `EvictionRecord`

§3.4 asks *"does `b` flip a large share of decisions?"* and §3.5 item 1 asks
*"is `ψ̂ ≪ b`?"*. Both are **counterfactual and per-slot**. The current record has
neither property. For both to be computable **from a run's log alone**, without a
second process and without re-running anything, `EvictionRecord` must carry:

1. **`victim_no_b: int`** — the argmin of `z(ψ̂) − ν·redundancy`, the same score with
   `b` omitted, on the same state. This is the *only* field that makes §3.4's
   question a count instead of an experiment: `b` decided this eviction iff
   `victim != victim_no_b`. It costs one extra `argmin` over a `[M]` tensor already
   in registers inside `_score`; it is not a second forward pass. Generalize as
   **`victim_without: dict[str, int]`**, one entry per optional term (`"b"`,
   `"nu"`), so §3.7's "every term ships a documented off-switch" acquires a
   *measurement* counterpart: every term ships its own leave-one-out victim.
2. **`attribution` becomes effect-derived, not config-derived.** The string is the
   set of terms whose removal changes the victim — `"psi"` when no term's removal
   moves it, `"psi+b"` only when `victim != victim_no_b`. `attribution_counts()`
   then answers §3.4 directly, and a `γ_b = 0` run reports `"psi"` as it should.
   Keep the configured term set on the record too, but as a **separate field**
   (`terms_enabled: tuple[str, ...]`) — it is genuinely useful and it is what the
   current string actually is. The defect is the name, not the data.
3. **Per-term contribution at the decision, for §3.5 item 1's `ψ̂ ≪ b`.** The
   victim's and runner-up's decomposed scores: **`psi_z_victim`,
   `psi_z_runner_up`, `b_victim`, `b_runner_up`, `nu_term_victim`,
   `nu_term_runner_up`** (all post-z-scoring, so `b_max = 1.0` means one SD, which
   is the whole reason §3.5 item 1 z-scores). Six floats. `ψ̂ ≪ b` is then a
   comparison of two logged columns; today it is not computable at all.
4. **`b_spread_live: float`** — `max_i b_i − min_i b_i` over live slots.
   Necessary-condition screening and the saturation diagnostic §3.5 item 3 asks
   for (`b` pinned at `±b_max` for most slots most of the time) in one number.
5. **`runner_up: int`** alongside the existing `score_margin`. A margin without the
   identity of the slot it is a margin to cannot be joined to anything.
6. **Keep** `step`, `victim`, `victim_rank`, `rank_shift`, `n_live`, `policy`,
   `warm`, `score_margin`. None of them is the problem. `warm` is the one field
   that *is* already effect-true, because the warmup branch genuinely bypasses the
   score.

With 1–5 the release-relevant tests reduce to arithmetic over one log:
§3.4 = `mean(victim != victim_without["b"])`; §3.5 item 1 =
`mean(|b_victim − b_runner_up| > |psi_z_victim − psi_z_runner_up|)`. **Marginal
cost per eviction: two extra `argmin`s over `[M]` and eight scalars.** The cheapest
possible version is field 1 alone — `victim_no_b` — and it alone converts §3.4 from
an experiment into a query.

⚠️ One consequence to name: `_score` currently returns `(score, terms)` and
computes the score by in-place accumulation, so the per-term columns require it to
keep the components rather than fold them. That is a small refactor of
`rsr.py:362–382` and it is **Brendan's to make** — this cycle modified nothing.

## E — the `sd = 0.0000` rows

87 statistic rows have `sd` exactly 0. The ledger row `sd_zero_rows_reasons` gives a
reason **per key**, produced by a classifier in the same process, and
`sd_zero_rows_unclassified` is `[]`. The categories:

- **`frac_attribution_carries_token` (16 rows) and
  `n_distinct_attribution_strings_max_over_arms`** — structural, *and they are the
  finding*: attribution is a function of `RSRConfig`, so no random variable can
  enter it.
- **`mutual_information_bits` (16 rows)** — degenerate by construction:
  `I(X;Y) = 0` exactly whenever `H(X) = 0`.
- **`recall_of_token_as_detector`** — 1.0 identically; the token is always present.
- **`frac_changed_with_margin_above_2bmax`** — a **measured** zero, finding C4.
- **off-value arms (`γ_b = 0`, `ν = 0`)** — the ground truth is the constant 0, so
  everything derived from it is 0. These are the controls, and 0 is the required
  value: `selftest_frac_changed_gamma_b_zero` and `selftest_frac_changed_nu_zero`
  are both `0.0000 ± 0.0000`, which is what proves the shared-head design really
  makes `ψ̂` bit-identical.
- **`n_evictions_per_seed`** — `12 × (80 − 40) = 480`, no RNG.
- **`frac_evictions_b_spread_positive`** — 1.0, structural.

**The seed does reach the RNG.** `frac_term_actually_changed_victim` has `sd > 0` in
all twelve arms with the term on; `mean_max_cosine_over_live_memory` has
`sd ≈ 0.005`; every AUC and every entropy row has `sd > 0`. `φ` init, the gestalt
stream, salience, the `ū` noise and the duplicate draws all move with the seed.

## Re-execution

The headline, in the smallest deterministic form — a bias that returns zeros
**cannot** change any argmin, and the record says it did:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python -c "
import torch
from rsr.retention.policy import MemoryState
from rsr.retention.rsr import RSRConfig, RSRPolicy
from rsr.retention.value_head import BilinearValueHead
M, d = 40, 384
class ZeroBias:              # b is identically 0 -> it CANNOT change any argmin
    def b(self, slots): return torch.zeros(M)
g = torch.Generator().manual_seed(0)
ge = torch.nn.functional.normalize(torch.randn(M, d, generator=g), dim=-1)
c  = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=-1)
s = MemoryState(gestalts=ge, written_at=torch.arange(M),
                live=torch.ones(M, dtype=torch.bool), step=M)
cfg = lambda be: RSRConfig(nu=0.0, beta=0.0, gamma=0.0, t_warm=0.0, a_max=M,
                           psi_override=None, b_enabled=be, shadow_enabled=False)
head = BilinearValueHead(d, generator=torch.Generator().manual_seed(1))
p0 = RSRPolicy(cfg(False), d, value_head=head)
p1 = RSRPolicy(cfg(True),  d, value_head=head, bias=ZeroBias())
v0 = p0.select_eviction(s, c, M); v1 = p1.select_eviction(s, c, M)
print('same victim (b provably inert):', v0 == v1)
print('attribution, b disabled      :', p0.records[-1].attribution)
print('attribution, b enabled but 0 :', p1.records[-1].attribution)
"
```

Prints:

```
same victim (b provably inert): True
attribution, b disabled      : psi
attribution, b enabled but 0 : psi+b
```

The full experiment re-runs in **4.3 s** and rewrites this directory's
`ledger.json`:

```bash
cd /Users/keanooo7/retrieval-successor-retention && .venv/bin/python \
  /private/tmp/claude-501/-Users-keanooo7-second-brain/1bdd6660-ee4c-49c5-99a7-28bf139ef2c9/scratchpad/run_attribution_instrument.py
```

All per-seed fractions are seeded and reproduce exactly.

## What this does NOT establish

- **Nothing about real `ū`.** `observe()` still raises; there is no
  `AttentionTrace` capture and `rsr.retention.reward`'s `r_i` was never in this
  loop. Every rate is conditional on a log-normal-salience softmax and `τ = 0.25`.
- **Nothing about `γ_b`, `ν`, `τ`, `b_max` or `E[lifetime]` as constants.** E0e's
  and E1's.
- **Nothing about whether `b` or `ν` *helps*.** `φ` is random and untrained; no
  language model ran; no loss was computed. "Changed the victim" is not "made it
  worse".
- **`"fifo_warmup"` was never exercised** (`t_warm = 0.0` throughout, and the
  driver's `attribution_counts()` is `{"psi": 480}`). `warm` is the one
  effect-true field on the record and this cycle did not test it. Gauntlet 0.2's
  all-`fifo_warmup` failure mode is untouched.
- **`"neg_age"` untested.** §3.7's reduction path emits its own attribution string
  and was not run here.
- **Only one definition of "`b` decided it."** Counterfactual victim change. A
  weaker notion — `b` reinforced a decision `ψ̂` would have made anyway — is not
  measured, and under it the true positive rate would be higher. The proposed
  `victim_without` field measures exactly the notion §3.4's sentence uses ("flips
  a large share of decisions").
- **Nothing about device placement, MPS, or a trained head.** CPU only.
- **The cycle-1 walls all still stand**, including `measurements/ledger.json`
  still not existing (`measurements_ledger_exists: false`, re-checked).
