# Roadmap — sprints, gates, and what blocks what

**Written 2026-09-20 against `1900b3d`.** Sequenced by dependency, not by the spec's §8 calendar —
that calendar was built around rented GPU windows and ADR-0007 deleted them.

> CRITICAL: **This supersedes §8's week numbering, not §6's experiment definitions or §16's
> conditions.** Where this file and the spec disagree about *order*, this file wins. Where they
> disagree about *what an experiment is*, the spec wins and this file is wrong.

---

## 0. The state, in one paragraph

287 tests pass with zero skips, E0b is green and bit-exact, E0c is measured on the machine that will
run it, and E0g passed. Against that: **the trained model's working memory is inert** (a shuffle
control moves the loss by exactly 0.0), **the synthetic corpus cannot reward retrieval** (the answer
is stored out of band and never enters the token stream), **the RSR training arm is FIFO**
(`train/loop.py` hardcodes it), and **12 of 41 source files are stubs** — including all four metrics
and leave-one-out, which §3.2.1 makes the arbiter of truth. The 13-cycle overnight run produced real
findings about the *design* while sitting on a substrate that could not have produced a result about
the *mechanism*.

**So the first sprint is not an experiment. It is making a run capable of meaning something.**

---

## 1. KEY: The blocker that was in no plan — the AttentionTrace capture bridge

`src/rsr/retention/reward.py` is 160 lines, real, with 11 passing tests, and **zero callers in
`src/`**. It computes `r_i`, the reward the entire mechanism rests on. It has no callers because the
model cannot produce its input:

| `AttentionTrace` wants | The model has |
|---|---|
| `alpha: [L, H, M]` | `[B, H, Q_tok, M]` at `model/tg/model.py:335`. **Collapsing `Q_tok` is an undesigned decision** — mean over real tokens, EOS only, or sum? It changes `r_i`, and therefore E0d's answer |
| `wo_v: [L, H, M, d_model]` | **Never captured.** `attn_out_proj` is applied to the already-α-weighted output (`model.py:337-338`), so per-slot per-head `W_O v` never exists as a tensor |
| `gate: [L]` | An `nn.Parameter` at `model.py:388`, not surfaced on `StepOutput` |

**This one item gates E0d, E0e, E0h, `RSRPolicy.observe`, H2O, the oracle and the shadow buffer
simultaneously.** Nothing downstream of it can be scheduled until it exists.

Two that will bite silently if not fixed with it:

- **`policy.observe()` is never called.** `git grep '\.observe(' -- src/` returns **zero hits**;
  the only callers are tests. Finishing `observe` is necessary and not sufficient —
  `model/tg/policy_loop.py` needs the call site. Same for `policy.reset()` at stream boundaries.
- **`RSRConfig.from_registry` reads eagerly before applying overrides** (`retention/rsr.py:236-243`),
  so on an empty ledger it raises for *every* field, including overridden ones. **E1 cannot use the
  documented constructor until E1 has already run**, and the `γ = 0` control arm — §4.5's *"the only
  arm that licenses the word 'future' in the abstract"* — is unreachable through it.

---

## 2. Sprints

Each ends on a gate that **can answer red**. No sprint begins before its predecessor's gate answers.
Sprint 1 is CPU-only and runs beside Sprint 0.

### Sprint 0 — make the substrate honest

| Task | Where |
|---|---|
| Five loop defects, **tests first** | `train/loop.py` |
| — `ignore_index`: 93.4% of 193,536 scored targets are PAD | `:210-216` |
| — `--policy`: `FIFOPolicy()` is unconditional while `policy_name` is stamped into the run id, frozen config and heartbeat | `:164-166` |
| — the `srep_norm` hinge: `StepOutput.srep_norm_penalty` is computed and discarded | `model.py:424`, `loop.py:210` |
| — `attribution_counts()`, not `attribution()`: the `hasattr` probe is always False, so attribution is permanently null | `:236` |
| — `--vocab` default `None`, not 50257, against a 160-symbol corpus | `:274` |
| Pass `value_head` into `build_param_groups` — hardcoded `None`, so ψ̂ trains outside its μP group and §4.3's `1/d` multiplier is unbacked by the optimizer | `:161` |
| Fix the `from_registry` eager read | `retention/rsr.py:236-243` |
| **The capture bridge** — surface `wo_v`, `gate`, pre-softmax logits; decide and document the `Q_tok` collapse in an ADR | `model/tg/model.py`, `retention/policy.py` |
| Call `observe()` and `reset()` | `model/tg/policy_loop.py` |
| **Make the corpus rewardable** — answer span in the token stream, a target mask, and the answer-token loss bucketed by gap | `data/synthetic.py`, `loop.py` |

**GATE: the shuffle control, as a permanent instrument, green.** Hand a row another document's
memory; the loss must move by a stated non-zero amount, reported with spread. It currently moves by
**exactly 0.0** against the base paper's own −54% for removing working memory. A run where memory
contributes ≈0 is **refused, not filed**.

### Sprint 1 — the data kill gate (CPU, parallel with Sprint 0)

E0i. `data/pg19.py` and `data/coref.py` are stubs; the `coref` extra is already declared in
`pyproject.toml` (fastcoref, wtpsplit, datasets, spacy) per ADR-0003.

🔴 **Owner action first: sign `preregistration/e0i_threshold.md`.** It is final and unsigned and says
*"it must be signed by a person, and that person is not a model."* No agent may sign it.

**GATE:** a histogram against the pre-registered floor, or **exit 3 with a stated reason.** Never
exit 0 on "did not run."

### Sprint 2 — the instruments

`metrics/loo.py` → **E0d** (the same machinery serves the oracle and the shadow scorer — build once,
use three times). **E0e** is the cheapest experiment in the project: a FIFO run plus EMA bookkeeping,
and it auto-unlocks `γ_b` via the registry, which unblocks `ProtectionBias`. Port
`mup/coord_check.py` from branch `macbook-local-2026-09-18` (API differs: `coord_check` /
`width_invariance` → `run_coord_check`) and **re-run E0a with unit-norm gestalts** — the existing
numbers fed `torch.randn` gestalts, which is the wrong premise for real TG.

**GATE:** E0d and E0a answer. Both are kill gates.

⚠️ Ordering trap: `ProtectionBias` needs `γ_b` ← `E_lifetime` ← E0e, but `RSRPolicy.__init__` refuses
`b_enabled=True` without a bias object. **So every pre-E0e RSR arm runs `b_enabled=False`, which is
also part of the §3.7 reduction condition.** An unlabelled pre-E0e RSR arm is partially reduced
toward TG. Put it in the run metadata.

### Sprint 3 — the baselines

`leading_edge.py` **on the microstructure coherence graph, not the macrostructure** (correction 26 —
KvD disclaim the macrostructure reading in the same paragraph, and §5.4's description is wrong).
`h2o.py` **both halves at the 50/50 split**. `expire_span.py` — the referendum, no demotion path.
`oracle.py`. Then **E-feas**: if oracle ≈ FIFO, that corpus cannot exhibit the effect.

**GATE:** E-feas, plus per policy a mutation that reddens **only** its own parity test. A gate no
mutation reddens adds nothing, and that is the finding.

### Sprint 4 — the epoch count, then E1/E2

🔴 **First task, and it settles the whole downstream budget: measure E3's epoch count.** Run
`loop.py` at `d=128, S=80, batch=16` on real PG-19 and find where validation loss stops moving.
**Nothing in this repo states it**, §4.1's 48 h/run is one epoch, and the reference config trains 50
(`third_party/ThoughtGestaltCode/tg/configs/tg_default.py`). §3.4 proves multi-epoch independently:
`T_warm` is *"one epoch… stated as a fraction of total epochs"* — at one epoch the whole run is FIFO
warmup and the head never takes over. Also re-run `experiments/e0c/measure.py --configs 128:80:4`:
E0c used 81,920 tokens/step against the reference's 20,000 token budget, and small-batch throughput
at `d=128` is unmeasured.

Then E1, E2, A5; freeze `γ`, `β`, `ν`.

**GATE:** the decision point. Red light, or **permission to continue** — never a green light on the
hypothesis (D-9: the synthetic generator constructs the structure that makes lookahead pay).

### Sprint 5 — E3 on PG-19

Sized by Sprint 4. Resolve two contradictions in the spec first: §4.1 budgets **A2/A4** while §8 says
**A1/A2**; and **LRU is a §7.1 gate with no arm in E3** (§6 lists RSR, FIFO, H2O, RSR(γ=0); §5.4 says
failing to beat LRU *is* vacuity).

### Sprint 6 — E7

Blocked on **ADR-0005 sign-off** (still `proposed`) and on E7's own `M=8, d=128` model. Reports three
numbers, not two — see §4.

**Owner track, throughout: §15.2.** Until it is non-empty, *"this specification is a well-audited
implementation of other people's objections."* No agent may draft it, schedule around it, or suggest
a candidate.

---

## 3. WARNING: Feasibility of weeks 5–7 is UNDETERMINED

An earlier reading of E0c concluded weeks 5–7 fit in ~19 h with 26× headroom on one Mac Studio.
**That reading is withdrawn.** It divided a measured 310 sent/s into `1.2M sentence steps` — but
1.2M is **one epoch**, and the epoch count is the unstated number the budget actually turns on.

| Epochs | h/run | Weeks 5–7 | vs 504 h |
|---|---|---|---|
| 1 | 1.08 | 19.2 h | 3.8% |
| ~15 | 16.3 | 262 h | break-even at 65% util + 25% rerun |
| 50 (reference) | 53.8 | **860 h** | **does not fit** |

Three further caveats on any such calculation: E7's own model is **not** in that figure though §8
puts it in weeks 5–7; the ~166 h of E1/E2/E5/E-feas lines in §4.1 are unanchored guesses at different
configs and cannot be rescaled by a factor derived from the E3 line; and [P2]'s `21 sent/s` — the
origin of §4.1's assumed 7 — has a unit defined only in a paper table header, so "44× faster" may be
a unit conversion rather than a speedup.

**What is solid: 1.075 h per epoch at `d=128, S=80, batch=16`, measured end to end.** Everything
above that is open until Sprint 4. **Do not write a weeks-5–7 feasibility claim into any document
before then.**

---

## 4. Staying on the cognitive claim

§2's guard: *"The primary claim is cognitive, not engineering. Drifting toward the engineering
framing is itself drift, even when every number is honest."*

Two mechanical additions:

1. **Every `RESULTS.md` names its falsifier and states whether the cognitive claim moved.** A run
   that can only move the engineering claim is fine — it just has to say so.
2. **E7 reports three numbers, not two** (correction 25): raw ρ; partial ρ controlling serial
   position; and KEY: **agreement between RSR's survival ordering and the leading-edge strategy's
   own ordering.** The third is the actual comparator. The strategy *emphasizes recency and
   frequency*, so partialling serial position out tests a policy Kintsch never proposed — and KvD's
   footnote 6 gives a published ranking to sit RSR inside: leading-edge > levels+primacy (+23%) >
   recency-only (+43%) > random (χ²(34) = 113.77).

`docs/cognitive-grounding.md` carries the standing "CLS does not map onto this" section so the
inverted claim §3.6 deleted cannot return.

---

## 5. Standing additions to every run

Four, all cheap, none new machinery:

1. **The shuffle control**, per run, refusing a run where memory contributes ≈0. It exists in prose
   only today — `git grep shuffle -- src/ tests/ scripts/` returns nothing.
2. **`E[lifetime]` and the fill fraction**, reported per run. Both are already computed and both are
   inputs to constants the registry currently refuses.
3. **"The code that ran is committed"** as the reproducibility gate — not "the tree is clean." 9 of
   13 ledgers cannot be re-executed from any commit, and the sharpest case has `git_dirty: false`
   and is *still* unreproducible, so a clean-tree check would have passed it.
4. **The leading-edge strategy as primary comparator**, not a baseline. RSR vs FIFO is uninformative
   by construction: survival time under FIFO is `min(M, S−i)`, a deterministic function of serial
   position that the partial correlation zeroes out.

---

## 6. The defect rule this project keeps relearning

> **A defect filed as a lesson recurs; a defect filed as a gate does not.**

Exit-code conflation is stated in **six files and enforced by zero code**, and has recurred at least
six times — most sharply where `_git()` returns `""` on failure so a git failure is indistinguishable
from a clean tree, **while `constants.py:571` already does it correctly in the same repo.** The
correct pattern existed, in a file the author had read, and was re-solved wrongly.

**When a defect is found, run its test over every sibling before closing it.** That is not advice —
it is what produced the v0.5 self-audit row: *"`K = M` is too small… Same disease as D-1, found by
applying D-1's test to the other frozen constants."* One fix, applied as a class, caught a second
live defect for free.

The conversion queue, highest value first:

| Class | Convert to |
|---|---|
| Exit 3 collapsing to 0 | One shared `Exit` enum and a result helper every checker uses; a test that no checker returns a bare boolean |
| A derived number typed into prose | Generate it, or stamp it with a sha and re-measure at read time |
| A guard present but unreachable | A test that the guard is **reached**, not only that it works — `mutation_battery.py`'s `off_gate` is computed, printed, and never filtered, so "the mutation must redden only it" is enforced by nothing while the doc reports 17/17 PROVEN |
| A label that is not the mechanism | Count the thing you assume is happening. `experiments/e0c/measure.py` already asserts writes and evictions — copy that shape |

KEY: **`src/rsr/constants.py` is the project's one example of doing this right.** §4.5's "never
freeze an unmeasured constant" became a count that can only improve, with a refusal that names the
experiment that owes the value. It is the template for every row above.
