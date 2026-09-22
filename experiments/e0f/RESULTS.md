# E0F - Verify [P5]-[P14] against primary sources

**Status: PARTIAL. 13 of 14 checked, [P9] (DNC) outstanding. Not complete, and not "logged" for
the purpose of deleting spec §1's verification note.** Corrected 2026-09-22. This file said
"NOT RUN". Its content was the scaffold (`83bdf57`, 2026-09-17), and git history shows it was never
updated after either pass (`git log -- experiments/e0f`: only `83bdf57` and `99a129e`, which
touched `run.py` alone).

E0f is a literature check, not a script, and `run.py` still exits 3 (`not_implemented`). The
record of what was checked is **`docs/citation-audit.md`**:

- **Pass 1** (2026-09-17): [P2], [P5], [P6], [P7]. Run on the MacBook and ported to trunk in
  `dd78db7` (#1).
- **Pass 2** (2026-09-20): [P1], [P3], [P4], [P8], [P10], [P11], [P12], [P13], [P14]. Committed in
  `8f6f81d` (#6). **[P11] did not survive.** See corrections 25–30 in `docs/spec-corrections.md`.

The template below is left as it was. This correction supplies no command, SHA or numbers beyond
the pointers above.

| | |
|---|---|
| Prediction | Citations hold. Ten of fourteen were unchecked at drafting. Corrections 12 and 13 already record two that do not. |
| Kill gate? | No, but it is an afternoon |
| Sprint 1 task | Sprint 2 |

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
