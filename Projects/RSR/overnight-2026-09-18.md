# Overnight run — 2026-09-18

**Manager:** Claude Opus 5 (1M), unattended. **Started** 00:25 PDT. **Stops** 07:00 PDT.
**Repo:** `~/retrieval-successor-retention` @ `dce75c1` at start, clean tree, `287 passed, 0 skipped`.

Protocol per cycle: dispatch a **fresh** researcher with a brief naming one falsifier →
researcher runs **one** experiment → report → manager reviews, **re-executes one claim**,
records which → retire the researcher → dispatch the next. A researcher is never given a
second brief. Canary every 4th cycle: one fixed config, one fixed seed, re-run; if the
number moves, the environment moved and everything since is suspect.

**"We learned nothing this cycle" is a writable row.** So is "the falsifier could not be
reached." An empty *verified by re-execution* field is a failed cycle and is labelled one.

---

## Cycle 0 — manager setup (no experiment)

| | |
|---|---|
| **Falsifier** | none — infrastructure |
| **Dispatched** | nobody |
| **Result** | `scripts/ledger.py` written: the `runs/<id>/ledger.json` contract the protocol requires did not exist. |
| **Verified by re-execution** | Ran the helper's own refusals. `stat()` over 1 sample raises `SingleSample`; `sd` over 1 sample is `None`, never `0.0`; `sd` over three identical samples returns `0.0` **with `sd_exactly_zero: true`** set, so the flat statistic is visible rather than silent; `git_sha` is stamped from inside the tree (`dce75c1`) and `git_dirty` recorded. |
| **Learned** | The protocol's "every number traces to `runs/<id>/ledger.json`" was unenforceable — no such file or convention existed anywhere in the repo. `measurements/ledger.json` is a *different* artefact (frozen constants, §4.5) and is currently **absent**, i.e. the constants registry is empty. That last fact turns out to be the reason the training loop cannot instantiate RSR at all (see cycle 1). |

---
