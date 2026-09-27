---
id: retrieval-curve
title: 'Retrieval curve: does longer training make memory help answers at 2 <= gap <= M?'
class: exp
sprint: 0
workstream: W0
lane: cpu-det
slots: 15
requires:
- ruling: R-2026-09-22-proxy-go
- item_done: capacity-c0
- prereg: experiments/retrieval-curve/PREREG.md
brief: docs/lab-notes/dispatch-retrieval-curve.md
status: done
attempts: 0
source: experiments/s0-03-rewardable-corpus/run.py:311; experiments/decisive-shuffle/PREREG.md:73
---
# Retrieval curve

The precondition every cognitive experiment rests on: memory is live (`runs/decisive-shuffle/`)
but does not help answers at `2 <= gap <= M` at 300 iterations (`runs/s0-03-rewardable-corpus/`,
H4 failed). One factor changes: training length, to 3000. A `falsified` verdict is what licenses
`R-*-retrieval-shown`, which E0d requires (proposed `R-2026-09-22-sprint2-start`).

**Id is `retrieval-curve`, not `exp-…`:** `dispatch.py` sets `RSR_RUN_ID` to the item id and
`merge.py` reads `runs/<item>/verification.json`, so the item id must equal the run id.

**Slots 15 = 3 seeds × 5 threads**, S0-03's thread setting. The PREREG's reproduction control is
bit-exact only at the same thread count. ⚠️ If C0's `ops/lanes.json` gives `cpu-det` fewer than
15, this item never becomes ready. Report that; **do not lower the thread count** to fit.

**Status `done` (2026-09-27, docs sync at `b21f3c3`).** The run merged in #39 and #43.
`runs/retrieval-curve/ledger.json` has `status` ok and `verdict.outcome` **falsified**: longer
training makes memory help answers (`experiments/retrieval-curve/RESULTS.md`). This status is set
in the same change that marks `capacity-c0` done. Without it, `retrieval-curve` would compute as
**ready** and could be re-dispatched, although it has already run. Whether this verdict licenses
`R-*-retrieval-shown` is the owner's ruling, and this status does not write one.
