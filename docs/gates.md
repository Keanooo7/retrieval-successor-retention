# Gates and ratchets

> WARNING: **THIS IS A SPECIFICATION OF INTENT, NOT A DESCRIPTION OF THIS TREE.**
> The ratchet machinery below arrived by a port from the app pipeline and is **not
> implemented here**: there is no `src/rsr/gates/`, no `.rsr/` directory, and
> `REGISTRY.unset()` -- which the last ratchet row names -- does not exist.
> **Do not report a ratchet as checked because this file lists it.** The mutation
> evidence and the exit-code taxonomy below are real and are used; the floor table
> is a plan. Checked 2026-09-20 at `b572ad6`.


Two rules govern everything here, both borrowed from the app pipeline this project's orchestration
is ported from:

1. 🔴 **A new gate is not believed green until a mutation has shown it red.** A gate that no
   mutation can redden adds nothing, and that is a finding, not a formality.
2. 🔴 **"Did not run" is not "found nothing."** Every checker distinguishes them by exit code. A
   caller that tests only `!= 0`, or only `== 1`, merges them — and in ML that is the most dangerous
   silent failure there is, because a run that produced no metric looks exactly like a run with a
   bad metric.

---

## Mutation evidence — the reduction gate (E0b, policy level)

Run 2026-09-17. Each mutation edits `RetentionConfig` and the full suite is re-run.

| Mutation | Tests reddened |
|---|---|
| *(baseline — no mutation)* | **0** — suite green |
| `psi_source` → `"head"` | **4** |
| `use_bias` → `True` | **3** |
| `nu` → `0.5` | **3** |
| `use_shadow` → `True` | **3** |
| `is_reduction` drops the `nu` clause | **2** |
| `is_reduction` drops the `use_bias` clause | **1** |
| `is_reduction` drops the `use_shadow` clause | **1** |

### What the mutation run found

The first version of this suite had a hole, and the mutation is what exposed it.

**`is_reduction` dropping its `nu` clause left the entire suite green.** Four other mutations
reddened the reduction tests; that one reddened nothing. `is_reduction` gates whether
`RSRPolicy.observe` is allowed to be a no-op — so with that hole, a policy configured with `ν = 0.5`
would report itself as the §3.7 reduction, `observe` would silently return `None` instead of
raising, and an arm that was supposed to be learning would quietly learn nothing while still being
called RSR.

`test_is_reduction_is_pinned_against_every_switch` and
`test_observe_refuses_to_no_op_outside_the_reduction` were added for that, and the run above is the
re-test. **Nothing here is believed on the strength of it having passed the first time.**

### What the reduction gate still does not cover

`test_full_model_reduction_bit_exact_loss_curve` is **skipped with a reason**, not absent. The
policy-level reduction — identical eviction choices — is a *necessary* condition for a bit-exact
loss curve, not a substitute for one. The full test needs a TG (`ADR-0001`).

📌 `test_reduction_matches_fifo_via_the_score_path` sets `T_warm = 0` on purpose. Under the full
§3.7 config every call routes through the warmup's FIFO shortcut, so a suite that tested only that
config would pass **without ever evaluating the eviction score** — green, and vacuous.

---

## Ratchets

Each metric has a **direction**. A floor that can move both ways is not a floor.

| Floor | File · key | Direction | Owner gate |
|---|---|---|---|
| `pytest` count | `.rsr/test-floor.json` · `main.count` | may rise, **never fall** | every lane |
| Reduction (E0b) | binary — CI test | must **stay green** | R1 |
| μP coordinate drift | `.rsr/gate-floors.json` · `mup_coord_drift` | may fall, **never rise** | R1 |
| Vacuity partial ρ(score, age) | `.rsr/gate-floors.json` · `partial_rho_score_age` | may fall, never rise · **hard ceiling 0.7** | R1 |
| ρ(`r_i`, LOO Δloss) | `.rsr/gate-floors.json` · `spearman_ri_loo` | may rise, never fall | R2 |
| Compute spend | `.rsr/budget.json` · `gpu_hours_used`, `usd_spent` | may rise, never fall · **hard cap $100 through week 4** | R4 |
| Unmeasured constants | derived from `rsr.constants.REGISTRY.unset()` | may fall, **never rise** | every lane |

The last row is the mechanical form of §4.5's *"Never freeze an unmeasured constant."* It is a count
that can only improve, and it currently stands at **10**.

### Exit codes — the protocol, which trunk does not yet implement

⚠️ **This heading read "the same five everywhere" until 2026-09-20, and that was false of this tree.** The only implementation is `src/rsr/gates/floors.py` on `macbook-local-2026-09-18` — `class Exit(IntEnum)` with all five codes, emitting `4` at `:118` and `:129`. That branch is **not an ancestor of `main`**, so on trunk **no site returns `4` at all**, and `decision-review.md:82`'s *"the ratchet correctly reported `UNBANKED_RISE`"* describes a run no code here could produce. The protocol below is still the protocol; porting `src/rsr/gates/` is an open reconciliation.


```
0  at or above the floor
1  FLOOR DROP              — the thing this exists to catch
2  floor UNKNOWN           — no recorded floor for this key
3  ENVIRONMENT / DID NOT RUN   <- NOT A PASS
4  UNBANKED RISE           — measured above the floor, floor not updated
5  INERT                   — a training run trained, but its memory carries no
                             row-specific content (Amendment 1 ratio <= 0.01)
```

**`5` is not `1`** (`docs/owner/rulings/R-2026-09-22-inert-exit-5.md`): `1` means the check caught a
failure and routes to a regression/debug item; `5` means nothing broke but no eviction rule has anything
to act on, and routes to a model or training-config investigation. Emitted by `rsr.train.loop.main` on
every training run (liveness-wiring), whose inconclusive band (`0.01 < ratio < 0.1`) is `1` and whose
invalid measurement is `3`. An INERT run's checkpoints are under `<out_dir>/quarantine/` and
`rsr.train.checkpoint.load` refuses them without `allow_quarantined=True`.

**`2` and `3` are separate deliberately, and `3` is where this design can be silently defeated.**
"The gate did not run" and "the gate found nothing" are different facts.

⚠️ `make` collapses non-zero exit codes. Run the checker directly and read `$?` when the distinction
matters.
