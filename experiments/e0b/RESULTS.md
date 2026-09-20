# E0B - Reduction test vs TG

**Status: RUN. GREEN, bit-exact.** Corrected 2026-09-20 -- this file said "NOT RUN"
for a gate that has been passing since `3e19dcb` (2026-09-17).

E0b is a CI test, not a script: `tests/test_reduction.py`, 20 tests. Stock TG, the
§3.7 reduction and the explicit FIFO policy all give loss `125.310546875`; the
learned head does not. Re-verified at `b572ad6`:

```
$ PYTHONPATH=src pytest tests/test_reduction.py::test_loss_curve_is_bit_exact_against_stock_tg
1 passed in 1.30s
```

Mutation evidence for this suite is in `docs/mutation-battery.md`, including the
hole a mutation exposed: dropping the `nu` clause from `is_reduction` reddened
nothing, which would have let a policy at `nu = 0.5` report itself as the exact-TG
reduction while `observe` silently no-op'd.

| | |
|---|---|
| Prediction | Bit-exact identical loss curve under the section 3.7 switches, with a separate RNG stream for `phi`. |
| Kill gate? | **Yes** |
| Sprint 1 task | A CI test, not a script: `tests/test_reduction.py` |

## Result

_Not run._

## Reproduction

| | |
|---|---|
| Command | _tbd_ |
| Git SHA | _tbd_ |
| Hardware | _tbd_ |
| Date | _tbd_ |

## Numbers

_Not run._

> Per `CLAUDE.md`, this section records the numbers that came out **wrong** as well
> as the ones that came out right, and section 12.4 applies: documented
> configuration is not evidence of what was actually run.
