"""E0d -- `r_i` against leave-one-out delta-loss (spec §3.2.1, §6 kill gate).

Pre-registration: `experiments/e0d/PREREG.md`, committed alone at 5357ad2, ahead of
this file; **Amendment 1** (84e21c5, erratum 268b947); **Amendment 2** (8c63ecd),
which governs wherever it and A1 differ; and **Amendment 3** (f5a8752), which
governs over both. They fix the substrate, the documents, every
statistic, every threshold, the controls and the classification. This file
implements them; it chooses none. Where they left a detail open, the choice is
marked `PREREG-OPEN:` below and listed in RESULTS.md. Amendment 2's cited material
(Part B's tests, the m1-m7 scripts and logs, the tau cells files) is vendored in
`experiments/e0d/amendment-2/`; its calibration fixture is `calibration.py`.

    uv run python experiments/e0d/run.py            # the run: runs/e0d/
    uv run python experiments/e0d/run.py --help

🔴 **C8 (A2.13) blocks the run until `docs/owner/rulings/` holds, committed and
unmodified, `R-*-retrieval-shown*`, `R-*-sprint0-gate*`, the first
`R-*-e0d-statistic*` ruling AND an `R-*-e0d-statistic-amended*` ruling**, with
exactly one `AUROC_STAR` across the last two (read from the files, never typed).
Then C12 checks the frozen per-seed tau tables against A2.2's text, and C1 needs
the T0 record (A1.7). Any refusal exits 3 **before a single document of
`D_E0d = [262144, 263168)` is generated**. `e0d_documents()` refuses that range
unless the caller has cleared C8, C12, C1 and C2.

Amendment 2 in one paragraph: per seed, the single gate is `AUROC_strat,pct` (the
age-stratified AUROC of each slot's within-step percentile of gated `r_i`, against
`y = 1[Delta_resample > tau_s]`), read through its 95 % document-bootstrap CI against
`AUROC_STAR`. Raw `AUROC_strat` is read by one row only (A2.8 row 5b): it can turn a
would-be kill into exit 2 and a HALT to Brendan, and never produces a pass. H (the
argmin-hit rate), every rho statistic (`label_A1`) and the pooled result are
reported and never gate.

What one run does, per seed of fresh-stream arm B's frozen `ckpt-003000` (read only):

1. C8 authority -> C1 substrate sha256 and the T0 record (A1.7) -> C2
   disjointness on the generator key, against every committed manifest, stream
   manifests (B5, A1.5) included.
2. Generate `D_E0d` (1024 docs), C3 vocabulary closure against `[0, 64)`'s vocab.
3. 64 batches of 16 consecutive ids. Per batch, `r_i` (gated and ungated, eval
   mode -- corrections 17, 20) from its **own** FIFO capture forward (A1.6), and
   `rsr.metrics.loo.loo_delta_loss` in `resample` (primary) and `zero` mode. C4
   alignment and C9 FIFO per batch; C7 determinism on the first batch.
4. C5 fill, C6 resample coverage, C11 age strata (A2.3). Then, on all cells and
   once more with A1.3's bos-copy exclusions (rank M-1, gap-1 Q-steps): the A2.4
   family (`AUROC_strat,pct`, raw `AUROC_strat`, `AUROC_unstrat,pct`, `C_ws`, H) for
   gated/ungated `r_i` x resample/zero, with `true_demand` as C10' (row 0), the
   amended §5 flatness, the A2.8 labels, the 5b boolean, the raw-substituted label
   (STEP_SENSITIVE), the tau sensitivity labels, and A1's rho statistics and
   `label_A1` as secondaries. One bootstrap matrix per seed.
5. §8 over the 3 seeds (A2.8, row 4b; the exit follows gated x resample on the
   excluded population; HALT on 5b; BOS_SENSITIVE, GATE_SENSITIVE, the zero
   knockout and the pooled result are reported). C1 again at the end.
6. A1.4: any exception exits 3 with its traceback in the ledger; an exit code
   leaves only after `e0d.class` is read back from the written ledger.
   `runs/<run_id>/claims.json` is written only by a run that reached a class.

It trains nothing, touches no `RSRPolicy`, no `b`, nu = 0, no shadow (C9, PLAN-v4 T3).
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import math
import os
import pickle
import re
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rsr.data.synthetic import (  # noqa: E402
    SyntheticConfig,
    _generate_document,
    generate,
    true_demand,
)
from rsr.exit_codes import (  # noqa: E402
    ArgumentParser,
    Exit,
    did_not_run,
    run_main,
    status,
)
from rsr.metrics.loo import KIND, loo_delta_loss, stream_annotations  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.model import init_memory  # noqa: E402
from rsr.model.tg.policy_loop import cross_capture, trace_for_row, write_at  # noqa: E402
from rsr.retention.reward import (  # noqa: E402
    contribution_per_layer,
    retrieval_demand,
)
from rsr.train import checkpoint as ck  # noqa: E402
from rsr.train.loop import (  # noqa: E402
    build_vocab,
    check_vocabulary_closure,
    encode,
    lm_token_losses,
)

EXPERIMENT = "experiments/e0d/run.py"
PREREG_PATH = "experiments/e0d/PREREG.md"
RUN_ID = "e0d"

# --------------------------------------------------------------------------- #
# 🔒 PREREG constants. tests/test_e0d.py reads each one back out of the PREREG.
# --------------------------------------------------------------------------- #

#: front matter `seeds`
SEEDS = (0, 1, 2)
#: §2: 1024 documents per seed, never generated by any committed run.
D_E0D = (262144, 263168)
#: §3: batches are 16 consecutive ids of D_E0d; resample donors come from them.
BATCH = 16
#: §1: M = 16, S = 48 -- read from the checkpoint's config and CHECKED against these.
PREREG_M = 16
PREREG_S = 48
#: §2.1: `_generate_document` seeds its RNG with `seed * KEY_STRIDE + doc_id`.
KEY_STRIDE = 1_000_003
#: §4: PROPOSED, NOT RATIFIED. Used only to report the PREREG's own value beside the
#: ratified one; the run reads rho* from the C8 ruling.
RHO_STAR_PROPOSED = 0.5
#: §6 C6
RESAMPLE_MIN_COVERAGE = 0.80
#: §6 C5
UNDERFULL_MAX = 0.01
#: §6 C4
LIVE_TOL = 1e-6
#: A1.1: §5 "LOO flat" is the median over A-cells of Delta_resample < 1e-2 nats
#: per sentence (replaces the q90 rule; the rationale's 0.1-nat scale is withdrawn).
LOO_FLAT_A = 1e-2
#: §5's original q90 rule, kept as a reported secondary only (A1.1).
LOO_FLAT_Q90 = 1e-3
#: §5 R_FLAT, unchanged by A1.1.
R_FLAT = 0.05
#: §3.1 "Intervals": per-document cluster bootstrap.
BOOT_N = 2000
BOOT_SEED = 20260927
CI_LEVEL = 0.95
#: §4 item 6 / A1.8: descriptive sensitivity only; changes no class.
SENSITIVITY_RHO = (0.3, 0.7)
#: A1.5: a stream manifest's exclusive step bound may not exceed the review's
#: (one-step conservative) 16123, or D_E0d is inside its stream.
B5_T_CAP_MAX = 16123

#: front matter `substrate` / `substrate_sha256`
DEFAULT_CKPT_ROOT = Path(
    "/Users/keanooo7/retrieval-successor-retention/.worktrees/fresh-stream/"
    "runs/fresh-stream/B"
)
CKPT_NAME = "ckpt-003000.pt"
CKPT_SHA256 = {
    0: "0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60",
    1: "dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8",
    2: "b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8",
}
#: §6 C1 as amended by A1.7: the T0 manifest's path is read from the T0 record, never
#: typed. PREREG-OPEN: the record is `runs/t0-substrate/manifest.json` with keys
#: `substrate_manifest` (required) and `repo_root` (optional; the manifest's paths
#: are relative to it -- RESTORE.md: "Paths are relative to the repo root").
T0_RECORD = Path("t0-substrate") / "manifest.json"
DEFAULT_SUBSTRATE_CWD = Path("/Users/keanooo7/retrieval-successor-retention")

#: §6 C8 as replaced by A2.13: `R-*-e0d-statistic*` (every such file: the first
#: ruling and its amendment) replaces `R-*-rho-star*`, which is neither required
#: nor read. At least one non-amended and at least one amended statistic ruling must
#: be present; either alone exits 3.
RULING_PATTERNS = {
    "retrieval_shown": "R-*-retrieval-shown*",
    "sprint0_gate": "R-*-sprint0-gate*",
    "e0d_statistic": "R-*-e0d-statistic*",
}
AMENDED_PATTERN = "R-*-e0d-statistic-amended*"
DEFAULT_RULINGS_DIR = ROOT / "docs" / "owner" / "rulings"
#: A2.13 key form, **not** anchored at line start: in both rulings the key sits in a
#: table cell or a numbered list item (Part D.2 item 5).
_AUROC_KEY = re.compile(
    r"(?<![A-Za-z0-9_])[`*]*AUROC_STAR[`*]*\s*[:=]\s*([0-9]*\.?[0-9]+)"
)

# --------------------------------------------------------------------------- #
# 🔒 Amendment 2 constants. tests/test_e0d.py reads each back out of the PREREG,
# and C12 re-checks the tau tables against A2.2's text at run time.
# --------------------------------------------------------------------------- #

#: A2.2: tau_s = the q = 0.995 quantile (numpy default, type 7) of |Delta| over
#: full-memory non-A Q-step cells with a value, on set E [64, 128), frozen per seed.
#: The runner reads these; it never computes a quantile on D_E0d cells.
TAU_Q = 0.995
TAU_RESAMPLE = {0: 0.4157434984576128, 1: 0.4315741845071321, 2: 0.4884946896703897}
TAU_ZERO = {0: 0.3447678400575726, 1: 0.3300262098312375, 2: 0.386515489417143}
#: A2.2 "Sensitivities (reported, never gating)"; A2.7 item 3.
TAU_SENSITIVITY = {
    "0.99": {
        "resample": {0: 0.3109549731015192, 1: 0.310573765520007, 2: 0.3491311189141148},
        "zero": {0: 0.24913927197480037, 1: 0.24216222953796368, 2: 0.2818768095970196},
    },
    "0.999": {
        "resample": {0: 0.7129988655444408, 1: 0.7854374831784502, 2: 0.7694723016164862},
        "zero": {0: 0.5874972405433865, 1: 0.6492233513593663, 2: 0.6450255164741889},
    },
}
#: A2.3: 16 bins, one per age {1}, ..., {16}, fixed before data.
AGE_BINS = tuple(range(1, PREREG_M + 1))
#: A2.8 row 3: `AUROC_strat,pct` CI upper < 0.5 is INVERTED.
AUROC_INVERTED = 0.5
#: A2.8: first matching row wins, in this order (row 5b after row 2, ahead of 3).
LABEL_ORDER = ("0", "1", "2", "5b", "3", "4", "5", "6", "7")
STEP_OR_AGE = "STEP_OR_AGE_AMBIGUOUS"

#: §2.2's table: every range C2 checks, read from the committed manifests at run
#: time. `(run, dotted key)`; a value is `[a, b]`, a dict of those, or (for
#: `arms_n_documents`) a list of corpus sizes `n` meaning `[0, n)`.
RANGE_SOURCES = (
    ("fresh-stream", "vocab_documents"),
    ("fresh-stream", "probe_documents"),
    ("fresh-stream", "heldout_documents"),
    ("fresh-stream", "stream_documents"),
    ("corpus-size-curve", "arms_n_documents"),
    ("corpus-size-curve", "probe_documents"),
    ("corpus-size-curve", "heldout_documents"),
    ("retention-readability", "sets.E"),
    ("retention-readability", "sets.P"),
    ("lookahead-room", "sets.E"),
    ("lookahead-room", "sets.P"),
    ("lookahead-room-r2", "sets.E"),
    ("lookahead-room-r2", "sets.P"),
    ("carry-forward", "sets.EXT"),
    ("carry-forward", "sets.H64"),
    ("carry-forward", "forbidden.fresh_stream"),
    ("carry-forward", "forbidden.heldout_H64"),
    ("e0e", "heldout_documents"),
    ("scaffold-dose", "vocab_documents"),
    ("scaffold-dose", "probe_documents"),
    ("scaffold-dose", "heldout_documents"),
    ("scaffold-dose", "stream_documents"),
    ("scaffold-timing", "probe_documents"),
    ("scaffold-timing", "heldout_documents"),
    ("fresh-escape", "vocab_documents"),
    ("fresh-escape", "probe_documents"),
    ("fresh-escape", "heldout_documents"),
    ("fresh-escape", "stream_documents"),
)

#: §3.1 item 6. PREREG-OPEN: an assert is *pending* at step t while its query is
#: at or after t (the query at t reads it now), *spent* once the query is behind.
CONTENT_KINDS = ("pending_assert", "spent_assert", "unpaired_assert", "query", "filler")

EXPECTED = (
    "PREREG §9 (scored per A1.8 and A2.14): class CONFOUND, second most likely MIXED, "
    "scored against class_A1 (A1's §8 on label_A1) only; no class prediction is "
    "registered for the A2.8 classification. Numeric parts on the rho secondaries -- "
    "pooled gated rho_resample 0.1-0.4 with CI upper < 0.5 on every seed; rho_rank "
    "below rho_pool; bottom-1 above 1/16 but under 0.3."
)


class ControlFailed(RuntimeError):
    """A PREREG §6 control failed: `inconclusive`, exit 3, nothing substituted."""

    def __init__(self, control: str, msg: str) -> None:
        super().__init__(f"{control}: {msg}")
        self.control = control


# --------------------------------------------------------------------------- #
# C8 authority, C1 substrate, C2 disjointness
# --------------------------------------------------------------------------- #


def _git(*args: str, cwd: Path) -> int:
    return subprocess.run(
        ["/opt/homebrew/bin/git", *args], cwd=cwd, capture_output=True
    ).returncode


def check_authority(rulings_dir: Path, *, require_committed: bool = True) -> dict:
    """C8 (A2.13): retrieval-shown, sprint0-gate, the first e0d-statistic ruling
    and its amendment exist, committed and unmodified; across every
    `R-*-e0d-statistic*` file exactly one distinct `AUROC_STAR`, in (0.5, 1).
    `HARM_RATIO_STAR` is not required and not read; `R-*-rho-star*` is neither
    required nor read."""
    rulings_dir = Path(rulings_dir)
    names = sorted(p.name for p in rulings_dir.iterdir()) if rulings_dir.is_dir() else []
    found: dict[str, list[Path]] = {}
    for key, pat in RULING_PATTERNS.items():
        hits = [rulings_dir / n for n in names if fnmatch.fnmatch(n, pat)]
        if not hits:
            raise ControlFailed(
                "C8", f"no ruling matching {pat!r} in {rulings_dir} (PREREG A2.13)"
            )
        found[key] = hits
    stat = found["e0d_statistic"]
    amended = [p for p in stat if fnmatch.fnmatch(p.name, AMENDED_PATTERN)]
    first = [p for p in stat if p not in amended]
    if not amended:
        raise ControlFailed(
            "C8", f"no ruling matching {AMENDED_PATTERN!r} in {rulings_dir}: the first "
            f"e0d-statistic ruling alone names raw AUROC_strat and the H gate, not "
            f"A2.4's statistic (PREREG A2.13; either ruling alone exits 3)"
        )  # fmt: skip
    if not first:
        raise ControlFailed(
            "C8", f"no first (non-amended) R-*-e0d-statistic* ruling in {rulings_dir}: "
            f"the amended ruling alone exits 3 (PREREG A2.13)"
        )  # fmt: skip
    found["e0d_statistic_amended"] = amended
    if require_committed:
        for hits in found.values():
            for p in hits:
                tracked = _git("ls-files", "--error-unmatch", p.name, cwd=p.parent)
                clean = _git("diff", "--quiet", "HEAD", "--", p.name, cwd=p.parent)
                if tracked != 0 or clean != 0:
                    raise ControlFailed(
                        "C8",
                        f"{p} is not a committed, unmodified ruling (tracked rc "
                        f"{tracked}, diff rc {clean}); PREREG A2.13: committed and "
                        f"unmodified before the run starts",
                    )
    values = set()
    for p in stat:
        values |= {float(v) for v in _AUROC_KEY.findall(p.read_text())}
    if len(values) != 1:
        raise ControlFailed(
            "C8",
            f"the e0d-statistic ruling(s) {[p.name for p in stat]} state "
            f"{sorted(values) or 'no'} AUROC_STAR value(s); exactly one is required "
            f"(PREREG A2.13)",
        )
    (auroc,) = values
    if not AUROC_INVERTED < auroc < 1.0:
        raise ControlFailed("C8", f"AUROC_STAR = {auroc} is not in (0.5, 1) (A2.13)")
    return {
        "rulings": {k: [str(p) for p in v] for k, v in found.items()},
        "auroc_star": auroc,
    }


def check_tau_tables(prereg: Path | None = None) -> dict:
    """C12 (A2.13): `TAU_RESAMPLE` and `TAU_ZERO` (and the sensitivity tables) hold
    exactly seeds {0, 1, 2} and equal A2.2's registered values, parsed from the
    committed PREREG text; a seed without a frozen tau is not run. Exit 3 else."""
    text = Path(ROOT / PREREG_PATH if prereg is None else prereg).read_text()
    try:
        a22 = text.split("## Amendment 2", 1)[1].split("### A2.2", 1)[1]
        a22 = a22.split("### A2.3", 1)[0]
    except IndexError as e:
        raise ControlFailed("C12", "PREREG has no Amendment 2 §A2.2") from e
    frozen = re.findall(
        r"^\s*\|\s*(\d)\s*\|\s*`[^`|]*`\s*\|\s*\*\*([0-9.]+)\*\*\s*\|[^|]*\|"
        r"\s*([0-9.]+)\s*\|",
        a22,
        re.M,
    )
    sens = re.findall(
        r"^\s*\|\s*(\d)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
        r"\s*([0-9.]+)\s*\|\s*$",
        a22,
        re.M,
    )
    reg_res = {int(s): float(r) for s, r, _ in frozen}
    reg_zero = {int(s): float(z) for s, _, z in frozen}
    reg_sens = {
        "0.99": {"resample": {int(r[0]): float(r[1]) for r in sens},
                 "zero": {int(r[0]): float(r[3]) for r in sens}},
        "0.999": {"resample": {int(r[0]): float(r[2]) for r in sens},
                  "zero": {int(r[0]): float(r[4]) for r in sens}},
    }  # fmt: skip
    want = set(SEEDS)
    problems = []
    for name, table, reg in (
        ("TAU_RESAMPLE", TAU_RESAMPLE, reg_res),
        ("TAU_ZERO", TAU_ZERO, reg_zero),
        *(
            (f"TAU_SENSITIVITY[{q}][{k}]", TAU_SENSITIVITY[q][k], reg_sens[q][k])
            for q in ("0.99", "0.999")
            for k in ("resample", "zero")
        ),
    ):
        if set(table) != want or set(reg) != want:
            problems.append(f"{name}: seeds {sorted(table)} vs registered {sorted(reg)}")
            continue
        bad = {s: (table[s], reg[s]) for s in want if table[s] != reg[s]}
        if bad:
            problems.append(f"{name} differs from A2.2: {bad}")
    if problems:
        raise ControlFailed("C12", "; ".join(problems))
    return {
        "seeds": sorted(want),
        "tau_resample": {str(s): TAU_RESAMPLE[s] for s in sorted(want)},
        "tau_zero": {str(s): TAU_ZERO[s] for s in sorted(want)},
        "source": f"{PREREG_PATH} A2.2 (parsed at run time)",
    }


def tau_for(seed: int, knockout: str, q: str | None = None) -> float:
    """A2.2: the frozen tau of `seed` for `knockout` (`q` a sensitivity quantile)."""
    if q is None:
        table = {"resample": TAU_RESAMPLE, "zero": TAU_ZERO}[knockout]
    else:
        table = TAU_SENSITIVITY[q][knockout]
    return table[seed]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_substrate(ckpt_root: Path) -> dict:
    """C1 (start): each seed's ckpt-003000.pt has the PREREG's sha256."""
    out = {}
    for s in SEEDS:
        path = Path(ckpt_root) / f"seed{s}" / CKPT_NAME
        if not path.is_file():
            raise ControlFailed("C1", f"{path} is missing")
        got = sha256(path)
        if got != CKPT_SHA256[s]:
            raise ControlFailed("C1", f"{path} sha256 {got} != PREREG {CKPT_SHA256[s]}")
        out[str(s)] = got
    return out


def substrate_recheck(manifest: Path, *, cwd: Path) -> dict:
    """C1 (end): `shasum -a 256 -c MANIFEST.sha256` has rc 0 (PLAN-v4 T0)."""
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", str(manifest)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    rc = proc.returncode
    n_ok = sum(1 for ln in proc.stdout.splitlines() if ln.endswith(": OK"))
    if rc != 0:
        bad = [ln for ln in proc.stdout.splitlines() if not ln.endswith(": OK")]
        raise ControlFailed("C1", f"shasum -c {manifest} rc {rc}: {bad[:5]}")
    return {"rc": rc, "n_ok": n_ok, "manifest": str(manifest)}


def _lookup(d: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            raise KeyError(dotted)
        d = d[part]
    return d


def used_ranges(runs_dir: Path) -> list[dict]:
    """§2.2: every used document range, from the committed manifests. A missing
    manifest or key is a refusal (C2 cannot be asserted), never a skip."""
    out = []
    for run, key in RANGE_SOURCES:
        path = Path(runs_dir) / run / "manifest.json"
        try:
            man = json.loads(path.read_text())
            val = _lookup(man, key)
        except (OSError, ValueError, KeyError) as e:
            raise ControlFailed(
                "C2", f"cannot read {key!r} from {path}: {type(e).__name__}: {e}"
            ) from e
        seeds = tuple(sorted({*SEEDS, *(int(x) for x in man.get("seeds", []))}))
        name = f"{run}.{key}"
        if key == "arms_n_documents":
            items = [(name, (0, int(n))) for n in val]
        elif isinstance(val, dict):
            items = [(f"{name}.{k}", tuple(v)) for k, v in sorted(val.items())]
        else:
            items = [(name, tuple(val))]
        for nm, rng in items:
            a, b = (int(x) for x in rng)
            if not 0 <= a < b:
                raise ControlFailed("C2", f"{nm} = {rng} is not a range")
            out.append({"name": nm, "range": (a, b), "seeds": seeds})
    return out + stream_ranges(runs_dir)


def stream_ranges(runs_dir: Path) -> list[dict]:
    """A1.5: every committed manifest that read the fresh stream (`stream:
    {offset, stride}`) -- B5 included, under whatever run id it gets -- with that
    manifest's actual range: its `stream_documents`, and, where it records
    `resume_step`/`end_step`, `[offset + stride * resume, offset + stride * end)`.

    Refuses (C2) a stream manifest that records no range, and any whose
    exclusive step bound exceeds `B5_T_CAP_MAX` (the review's bound, applied
    unchanged: it is one step on the safe side, A1.5 author's note)."""
    out = []
    for path in sorted(Path(runs_dir).glob("*/manifest.json")):
        try:
            man = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            msg = f"cannot read {path}: {type(e).__name__}: {e}"
            raise ControlFailed("C2", msg) from e
        st = man.get("stream") if isinstance(man, dict) else None
        if not isinstance(st, dict):
            continue
        run = path.parent.name
        try:
            off, stride = int(st["offset"]), int(st["stride"])
        except (KeyError, TypeError, ValueError) as e:
            raise ControlFailed("C2", f"{path} `stream` has no offset/stride: {e}") from e
        seeds = tuple(sorted({*SEEDS, *(int(x) for x in man.get("seeds", []))}))
        items = []
        sd = man.get("stream_documents")
        if isinstance(sd, dict):
            items += [(f"{run}.stream_documents.{k}", v) for k, v in sorted(sd.items())]
        elif sd is not None:
            items.append((f"{run}.stream_documents", sd))
        if "end_step" in man:
            if "resume_step" not in man:
                raise ControlFailed("C2", f"{path} records end_step but no resume_step")
            a = off + stride * int(man["resume_step"])
            b = off + stride * int(man["end_step"])
            items.append((f"{run}.stream(resume_step..end_step)", (a, b)))
        if not items:
            raise ControlFailed(
                "C2", f"{path} reads the stream but records no stream range "
                f"(stream_documents or resume_step/end_step): C2 cannot be asserted"
            )  # fmt: skip
        for nm, rng in items:
            a, b = (int(x) for x in rng)
            if not 0 <= a < b:
                raise ControlFailed("C2", f"{nm} = {rng} is not a range")
            t_cap = -(-(b - off) // stride)
            if t_cap > B5_T_CAP_MAX:
                raise ControlFailed(
                    "C2", f"{nm} = {(a, b)} reaches stream step {t_cap} > "
                    f"{B5_T_CAP_MAX}: D_E0d may not measure it (PREREG A1.5)"
                )  # fmt: skip
            out.append({"name": nm, "range": (a, b), "seeds": seeds})
    return out


def t0_manifest(runs_dir: Path) -> dict:
    """C1 (A1.7): the T0 manifest path, read from the T0 record. No record, or a
    record naming no existing manifest, is C1 failing -- exit 3 before any
    document is generated."""
    rec = Path(runs_dir) / T0_RECORD
    try:
        man = json.loads(rec.read_text())
        path = Path(man["substrate_manifest"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ControlFailed(
            "C1", f"no usable T0 record at {rec} (PREREG A1.7): {type(e).__name__}: {e}"
        ) from e
    if not path.is_file():
        raise ControlFailed("C1", f"the T0 record names {path}, which does not exist")
    return {
        "record": str(rec),
        "manifest": path,
        "cwd": Path(man.get("repo_root", DEFAULT_SUBSTRATE_CWD)),
    }


def check_disjoint(doc_range, *, seeds, used) -> dict:
    """C2 on generator keys `k = s * KEY_STRIDE + i`, across every seed pair (§2.1)."""
    lo, hi = doc_range
    if not 0 <= lo < hi <= KEY_STRIDE:
        raise ControlFailed("C2", f"D_E0d {doc_range} is not inside one key stride")
    mine = [(s * KEY_STRIDE + lo, s * KEY_STRIDE + hi) for s in seeds]
    # no id repeats: E0d's own key intervals are pairwise disjoint
    ordered = sorted(mine)
    for (_a0, b0), (a1, _b1) in itertools.pairwise(ordered):
        if a1 < b0:
            raise ControlFailed("C2", "E0d's own generator keys repeat across seeds")
    collisions = []
    for u in used:
        a, b = u["range"]
        for s2 in u["seeds"]:
            ka, kb = s2 * KEY_STRIDE + a, s2 * KEY_STRIDE + b
            for s, (ma, mb) in zip(seeds, mine, strict=True):
                n = min(mb, kb) - max(ma, ka)
                if n > 0:
                    collisions.append((u["name"], s2, s, n))
    if collisions:
        raise ControlFailed(
            "C2", f"generator-key collisions (range, used seed, E0d seed, n): "
            f"{collisions[:10]}"
        )  # fmt: skip
    return {
        "n_keys": len(seeds) * (hi - lo),
        "collisions": 0,
        "n_ranges": len(used),
        "max_used": max((u["range"][1] for u in used), default=None),
    }


# --------------------------------------------------------------------------- #
# documents (the guard) and C3
# --------------------------------------------------------------------------- #


def e0d_documents(seed: int, S: int, doc_range, *, cleared: bool):
    """Documents `[lo, hi)` of `seed`, by id (§2). **Refuses any range touching
    D_E0d unless `cleared`** -- `main()` clears it only after C8, C1 and C2."""
    lo, hi = doc_range
    if not cleared and lo < D_E0D[1] and D_E0D[0] < hi:
        raise RuntimeError(
            f"documents {doc_range} touch D_E0d {D_E0D} and the run is not cleared: "
            f"C8/C1/C2 must pass first (PREREG §6 C8)"
        )
    cfg = SyntheticConfig(sentences_per_document=S, seed=seed, answer_in_stream=True)
    return tuple(_generate_document(i, cfg) for i in range(lo, hi))


def check_vocab(docs, vocab) -> None:
    """C3 (§2.4): a word outside `[0, 64)`'s vocabulary is exit 3; no document is
    dropped or replaced. PREREG: believed, not verified -- so it is checked here."""
    try:
        check_vocabulary_closure(docs, vocab)
    except ValueError as e:
        raise ControlFailed("C3", str(e)) from e


# --------------------------------------------------------------------------- #
# the measurement: one live pass (r_i) + loo_delta_loss (resample, zero)
# --------------------------------------------------------------------------- #


def _sent_loss(logits, ids_t, mask_t):
    """`loo.loo_delta_loss`'s own sentence loss: mean real-target NLL, NaN if none."""
    B, L = ids_t.shape
    per = lm_token_losses(logits, ids_t).view(B, L - 1).double()
    real = mask_t[:, 1:]
    n = real.sum(-1)
    mean = (per * real).sum(-1) / n.clamp(min=1)
    return torch.where(n > 0, mean, float("nan")).cpu()


@torch.no_grad()
def live_pass(model, ids, mask) -> dict:
    """A1.6: `r_i`'s own FIFO forward, with `capture=True`, over the same batch
    `loo_delta_loss` knocks out (which does not capture): §3.2.1's `r_i`, gated and
    ungated, read from the forward that reads sentence `t` over that memory. C4
    asserts its per-sentence loss equals LOO's `live_loss` (<= 1e-6) and its slot
    map equals `slot_sentence`. Eval mode (correction 20)."""
    cfg = model.cfg
    was = model.training
    model.eval()
    B, S, _ = ids.shape
    M = cfg.M
    Lc = sum(1 for b in model.blocks if b.block_type == "C")
    dev, dtype = ids.device, model.embed.weight.dtype
    f64 = dict(dtype=torch.float64)
    r_g = torch.zeros(B, S, M, **f64)
    r_u = torch.zeros(B, S, M, **f64)
    lay_g = torch.zeros(B, S, Lc, M, **f64)
    lay_u = torch.zeros(B, S, Lc, M, **f64)
    pre_g = torch.zeros(B, S, **f64)
    pre_u = torch.zeros(B, S, **f64)
    live_loss = torch.full((B, S), float("nan"), **f64)
    slot_sentence = torch.full((B, S, M), -1, dtype=torch.long)
    n_live = torch.zeros(B, S, dtype=torch.long)
    wrote = torch.zeros(B, S, dtype=torch.bool)
    eval_ok = True
    try:
        mem = init_memory(B, cfg, device=dev, dtype=dtype)
        bos_ctx = torch.zeros(B, cfg.D, device=dev, dtype=dtype)
        bos_valid = torch.zeros(B, dtype=torch.bool, device=dev)
        for t in range(S):
            ids_t, mask_t = ids[:, t], mask[:, t]
            out = model(
                ids_t, mask_t, mem.kv, mem.valid, bos_ctx, bos_valid, capture=True
            )
            live_loss[:, t] = _sent_loss(out.logits, ids_t, mask_t)
            cap = cross_capture(model, out, mem.kv, mask_t, mem.valid)
            eval_ok &= bool(cap.eval_mode)
            valid_c = mem.valid.cpu()
            slot_sentence[:, t] = torch.where(valid_c, mem.step.cpu(), -1)
            n_live[:, t] = valid_c.sum(-1)
            for b in range(B):
                tr = trace_for_row(cap, mem.valid, b, t)
                n = int(n_live[b, t])
                r_g[b, t] = retrieval_demand(tr, n_live=n, capacity=M, gated=True).cpu()
                r_u[b, t] = retrieval_demand(tr, n_live=n, capacity=M, gated=False).cpu()
                for gated, lay, pre in ((True, lay_g, pre_g), (False, lay_u, pre_u)):
                    per = contribution_per_layer(tr, gated=gated).cpu().double()
                    per = torch.where(valid_c[b], per, torch.zeros_like(per))
                    lay[b, t] = per
                    pre[b, t] = per.sum()
            write = out.has_eos
            wrote[:, t] = write.cpu()
            victim = torch.zeros(B, dtype=torch.long, device=dev)  # FIFO (C9)
            mem = write_at(mem, out.srep, write, victim, t)
            if cfg.bos_replacement_mode == "copy":
                bos_ctx = out.srep
                bos_valid = write & (t + 1 < S)
    finally:
        model.train(was)
    return {
        "r_gated": r_g,
        "r_ungated": r_u,
        "layer_gated": lay_g,
        "layer_ungated": lay_u,
        "pre_share_gated": pre_g,
        "pre_share_ungated": pre_u,
        "live_loss_plain": live_loss,
        "slot_sentence": slot_sentence,
        "n_live": n_live,
        "wrote": wrote,
        "eval_mode": eval_ok,
    }


def _nan_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(
        a.nan_to_num(0.0), b.nan_to_num(0.0)
    )


def measure_batch(model, docs, ids, mask, *, seed: int) -> dict:
    """One batch: the live pass, LOO resample (primary) and zero (secondary), C4."""
    lv = live_pass(model, ids, mask)
    res = loo_delta_loss(model, docs, ids, mask, mode="resample", seed=seed)
    zer = loo_delta_loss(model, docs, ids, mask, mode="zero", seed=seed)
    plain = lv["live_loss_plain"]
    diffs, nan_ok, map_ok = [], True, True
    for o in (res, zer):
        ll = o["live_loss"]
        nan_ok &= torch.equal(ll.isnan(), plain.isnan())
        both = ~(ll.isnan() | plain.isnan())
        diffs.append(float((ll[both] - plain[both]).abs().max()) if both.any() else 0.0)
        map_ok &= torch.equal(o["slot_sentence"], lv["slot_sentence"])
    c4 = {
        "live_loss_max_abs_diff": max(diffs),
        "live_loss_nan_pattern_equal": nan_ok,
        "slot_map_equal": map_ok,
        "eval_mode": lv["eval_mode"],
    }
    if not (lv["eval_mode"] and nan_ok and map_ok and max(diffs) <= LIVE_TOL):
        raise ControlFailed("C4", f"alignment: {c4} (tolerance {LIVE_TOL})")
    return {
        **lv,
        "delta_resample": res["delta"],
        "delta_zero": zer["delta"],
        "donor_row": res["donor_row"],
        "c4": c4,
    }


def check_determinism(a: dict, b: dict) -> dict:
    """C7: the first batch re-run gives bit-identical delta and r_i."""
    keys = ("delta_resample", "delta_zero", "r_gated", "r_ungated")
    bad = [k for k in keys if not _nan_equal(a[k], b[k])]
    if bad:
        raise ControlFailed("C7", f"re-run of the first batch differs in {bad}")
    return {"bit_identical": list(keys)}


def check_fifo(slot_sentence: torch.Tensor, wrote: torch.Tensor, M: int) -> dict:
    """C9: each forward's memory is the FIFO memory given the write flags."""
    B, S, _ = slot_sentence.shape
    for b in range(B):
        held: list[int] = []
        for t in range(S):
            want = held + [-1] * (M - len(held))
            if slot_sentence[b, t].tolist() != want:
                raise ControlFailed(
                    "C9", f"row {b} step {t}: memory {slot_sentence[b, t].tolist()} "
                    f"is not FIFO's {want}"
                )  # fmt: skip
            if bool(wrote[b, t]):
                held = [*held, t][-M:]
    return {"fifo": True}


# --------------------------------------------------------------------------- #
# cells, C5, C6
# --------------------------------------------------------------------------- #


def doc_structure(docs, S: int) -> dict:
    """A1.1's strata inputs, from the generator only (sentence kinds and
    `doc.pairs`), never from Delta or `r_i`: per `[doc, t]` whether sentence `t` is
    a query and the index of its assert (`-1` if not a query); per `[doc, t, s]`
    `true_demand(doc)[t][s]` (C10's score)."""
    n = len(docs)
    q_step = np.zeros((n, S), dtype=bool)
    assert_of = np.full((n, S), -1, dtype=np.int64)
    td = np.zeros((n, S, S), dtype=np.float64)
    for b, doc in enumerate(docs):
        qa = {q: a for a, q in doc.pairs}
        for t in range(min(S, len(doc.sentences))):
            if doc.sentences[t].kind == "query":
                if t not in qa:
                    raise RuntimeError(f"doc {doc.doc_id}: query {t} has no pair")
                q_step[b, t] = True
                assert_of[b, t] = qa[t]
        m = np.asarray(true_demand(doc), dtype=np.float64)[:S, :S]
        td[b, : m.shape[0], : m.shape[1]] = m
    return {"q_step": q_step, "assert_of": assert_of, "true_demand": td}


def cells_from(batches: list[dict], docs_per_batch: list, M: int) -> dict:
    """Flat per-cell arrays. A cell is (document, step t, rank i) with slot i
    occupied and sentence t holding a real target: exactly the non-NaN cells of
    the zero knockout (§3). A1.1's strata columns come from `doc_structure`:
    `q_step` (sentence t is a query), `a_cell` (the slot holds that query's
    assert), `gap` (t minus the assert's index at a Q-step, else -1) and `pc`
    (`true_demand(doc)[t][slot_sentence]`, C10's score)."""
    cols: dict[str, list] = {k: [] for k in (
        "doc", "t", "rank", "sentence", "n_live", "d_resample", "d_zero", "r_gated",
        "r_ungated", "layer_gated", "layer_ungated", "pre_share_gated",
        "pre_share_ungated", "kind", "q_step", "a_cell", "gap", "pc", "query_of",
    )}  # fmt: skip
    names = {v: k for k, v in KIND.items()}
    offset = 0
    for m, docs in zip(batches, docs_per_batch, strict=True):
        B, S, _ = m["delta_zero"].shape
        ann = stream_annotations(docs, steps=S)
        idx = torch.nonzero(~m["delta_zero"].isnan())
        b, t, i = idx[:, 0], idx[:, 1], idx[:, 2]
        s = m["slot_sentence"][b, t, i]
        ds = doc_structure(docs, S)
        bn, tn, sn = b.numpy(), t.numpy(), s.numpy()
        qs = ds["q_step"][bn, tn]
        aof = ds["assert_of"][bn, tn]
        cols["q_step"].append(qs)
        cols["a_cell"].append(qs & (aof == sn))
        cols["gap"].append(np.where(qs, tn - aof, -1))
        cols["pc"].append(ds["true_demand"][bn, tn, sn])
        kind = ann["kind"][b, s]
        q = ann["query_of"][b, s]
        labels = []
        for k, qq, tt in zip(kind.tolist(), q.tolist(), t.tolist(), strict=True):
            nm = names[k]
            if nm == "assert":
                nm = (
                    "unpaired_assert"
                    if qq < 0
                    else ("pending_assert" if qq >= tt else "spent_assert")
                )
            labels.append(nm)
        cols["doc"].append((b + offset).numpy())
        cols["t"].append(t.numpy())
        cols["rank"].append(i.numpy())
        cols["sentence"].append(s.numpy())
        cols["n_live"].append(m["n_live"][b, t].numpy())
        cols["d_resample"].append(m["delta_resample"][b, t, i].numpy())
        cols["d_zero"].append(m["delta_zero"][b, t, i].numpy())
        cols["r_gated"].append(m["r_gated"][b, t, i].numpy())
        cols["r_ungated"].append(m["r_ungated"][b, t, i].numpy())
        cols["layer_gated"].append(m["layer_gated"][b, t, :, i].numpy())
        cols["layer_ungated"].append(m["layer_ungated"][b, t, :, i].numpy())
        cols["pre_share_gated"].append(m["pre_share_gated"][b, t].numpy())
        cols["pre_share_ungated"].append(m["pre_share_ungated"][b, t].numpy())
        cols["kind"].append(np.array(labels, dtype=object))
        cols["query_of"].append(q.numpy())  # A2.7 item 8 (sink/inversion probe)
        offset += B
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["full"] = out["n_live"] == M
    return out


def check_fill(n_live: np.ndarray, t: np.ndarray, M: int) -> dict:
    """C5: at t >= M every memory should be full; > 1 % underfull is exit 3.
    PREREG-OPEN: counted over (document, step) pairs with t >= M."""
    sel = np.asarray(t) >= M
    n = int(sel.sum())
    under = int((np.asarray(n_live)[sel] < M).sum())
    frac = under / n if n else float("nan")
    out = {"underfull": under, "of": n, "fraction": frac}
    if not n or frac > UNDERFULL_MAX:
        raise ControlFailed("C5", f"fill: {out} (max {UNDERFULL_MAX})")
    return out


def check_coverage(d_resample: np.ndarray, full: np.ndarray) -> dict:
    """C6: >= 0.80 of full-memory cells receive a resample donor. PREREG: believed,
    not verified at batch 16 -- so it is a runtime control."""
    full = np.asarray(full, dtype=bool)
    n = int(full.sum())
    got = int((~np.isnan(np.asarray(d_resample)[full])).sum())
    cov = got / n if n else float("nan")
    out = {"coverage": cov, "with_donor": got, "full_cells": n}
    if not n or not cov >= RESAMPLE_MIN_COVERAGE:
        raise ControlFailed("C6", f"resample coverage {out} < {RESAMPLE_MIN_COVERAGE}")
    return out


# --------------------------------------------------------------------------- #
# §3.1 statistics (A1.1): Spearman with average ranks, weighted for the bootstrap
# --------------------------------------------------------------------------- #


class MeasurementUndefined(ControlFailed):
    """A registered statistic has no value (e.g. no A-cell carries a Delta): the
    rule cannot be read, so the run is `inconclusive`, exit 3 (§8 row 0)."""

    def __init__(self, msg: str) -> None:
        super().__init__("§5", msg)


class _Ranker:
    """Average ranks of `x` under integer-like weights: a weight-`w` element is `w`
    copies, so a tie group of total weight `g` after `c` lighter weight gets the
    average rank `c + (g + 1) / 2` -- exactly the duplicated sample's."""

    def __init__(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        xs = x[order]
        new = np.ones(len(xs), dtype=bool)
        new[1:] = xs[1:] != xs[:-1]
        gid_sorted = np.cumsum(new) - 1
        self.gid = np.empty(len(x), dtype=np.int64)
        self.gid[order] = gid_sorted
        self.ng = int(gid_sorted[-1]) + 1 if len(x) else 0

    def ranks(self, w: np.ndarray) -> np.ndarray:
        gw = np.bincount(self.gid, weights=w, minlength=self.ng)
        avg = np.cumsum(gw) - gw + (gw + 1.0) / 2.0
        return avg[self.gid]


def _wpearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    W = w.sum()
    if W <= 0:
        return math.nan
    mx, my = (w * x).sum() / W, (w * y).sum() / W
    dx, dy = x - mx, y - my
    vx, vy = (w * dx * dx).sum(), (w * dy * dy).sum()
    if vx <= 0 or vy <= 0:
        return math.nan
    return float((w * dx * dy).sum() / math.sqrt(vx * vy))


class WeightedSpearman:
    """Spearman rho (average ranks) of the sample in which cell `j` appears
    `w[j]` times. `w = 1` is plain Spearman; a document bootstrap's counts give
    the resampled statistic without materialising the resample."""

    def __init__(self, x, y) -> None:
        self.rx, self.ry = _Ranker(x), _Ranker(y)
        self.n = len(np.asarray(x))

    def __call__(self, w) -> float:
        w = np.asarray(w, dtype=np.float64)
        if self.n < 2 or (w > 0).sum() < 2:
            return math.nan
        return _wpearson(self.rx.ranks(w), self.ry.ranks(w), w)


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    return WeightedSpearman(x, y)(np.ones(len(x)))


class RankStrata:
    """Spearman within each write-order rank, then the mean weighted by each
    rank's cell count (the summed bootstrap weight, under the bootstrap). A rank
    whose rho is undefined carries no weight (PREREG-OPEN)."""

    def __init__(self, r, d, rank, *, n_ranks: int) -> None:
        r, d, rank = (np.asarray(a) for a in (r, d, rank))
        self.sel = [rank == i for i in range(n_ranks)]
        self.ws = [WeightedSpearman(r[s], d[s]) for s in self.sel]
        self.n = len(r)

    def __call__(self, w=None) -> dict:
        w = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)
        per, counts = [], []
        for sel, ws in zip(self.sel, self.ws, strict=True):
            per.append(ws(w[sel]) if sel.any() else math.nan)
            counts.append(float(w[sel].sum()))
        pairs = [(c, p) for c, p in zip(counts, per, strict=True) if not math.isnan(p)]
        den = sum(c for c, _ in pairs)
        return {
            "rho": sum(c * p for c, p in pairs) / den if den > 0 else math.nan,
            "per_rank": per,
            "n_per_rank": [int(c) for c in counts],
        }


def bootstrap_doc_weights(n_docs: int, n_boot: int, seed: int) -> np.ndarray:
    """`[n_boot, n_docs]` counts: each replicate draws `n_docs` documents with
    replacement (§3.1 Intervals). One generator, seeded once, so every statistic of
    a seed uses the same indices."""
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(n_docs, (n_boot, n_docs), generator=g).numpy()
    W = np.zeros((n_boot, n_docs), dtype=np.float64)
    for k in range(n_boot):
        W[k] = np.bincount(idx[k], minlength=n_docs)
    return W


def percentile_ci(vals, *, level: float = CI_LEVEL) -> tuple[float, float]:
    """The percentile interval. A NaN replicate makes the interval NaN rather than
    silently narrowing it."""
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0 or np.isnan(vals).any():
        return (math.nan, math.nan)
    a = (1.0 - level) / 2.0
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1.0 - a)))


def percentile_ci_defined(vals, *, level: float = CI_LEVEL) -> tuple[float, float, int]:
    """A2.4 "Bootstrap" (review m-7): a resample in which the statistic is undefined
    is counted and reported, and the interval is read over the defined resamples.
    Returns `(lo, hi, n_undefined)`; no defined resample at all is `(nan, nan, n)`."""
    vals = np.asarray(vals, dtype=np.float64)
    n_und = int(np.isnan(vals).sum())
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return (math.nan, math.nan, n_und)
    return (*percentile_ci(vals, level=level), n_und)


def strata(c: dict) -> dict:
    """A1.1, fixed by document structure (`cells_from`'s `q_step`/`a_cell`):
    Q-steps are full-memory steps whose sentence is a query, A-cells the slot
    holding the queried assert at a Q-step, N-steps every other full step."""
    full = np.asarray(c["full"], dtype=bool)
    q = np.asarray(c["q_step"], dtype=bool)
    Q = full & q
    return {"full": full, "Q": Q, "N": full & ~q, "A": Q & np.asarray(c["a_cell"])}


def bos_keep(c: dict, M: int) -> np.ndarray:
    """A1.3, one recomputation with both exclusions: drop every cell at write-order
    rank M-1, and every cell of a Q-step whose query has gap 1."""
    return (np.asarray(c["rank"]) != M - 1) & ~(
        np.asarray(c["q_step"], dtype=bool) & (np.asarray(c["gap"]) == 1)
    )


def _step_key(c: dict) -> np.ndarray:
    return np.asarray(c["doc"], dtype=np.int64) * (int(np.max(c["t"])) + 1) + c["t"]


def _grid(c: dict, sel: np.ndarray, col, M: int):
    """Cells `sel` as `[n_steps, M]` by write-order rank (NaN where a rank has no
    cell), with each step's document."""
    col = np.asarray(col, dtype=np.float64)
    key = _step_key(c)[sel]
    uniq, inv = np.unique(key, return_inverse=True)
    grid = np.full((len(uniq), M), np.nan)
    grid[inv, np.asarray(c["rank"])[sel]] = col[sel]
    step_doc = np.zeros(len(uniq), dtype=np.int64)
    step_doc[inv] = np.asarray(c["doc"])[sel]
    return grid, step_doc


def step_stats(r_grid: np.ndarray, d_grid: np.ndarray) -> dict:
    """Per step: the within-step Spearman over its measured slots (PREREG-OPEN:
    >= 3 needed), and bottom-1 agreement `argmin r == argmin delta` over its
    measured slots (PREREG-OPEN: >= 2 needed; a slot with no Delta, or excluded
    by A1.3, is left out of its step)."""
    n = r_grid.shape[0]
    rho = np.full(n, np.nan)
    b1 = np.full(n, np.nan)
    for j in range(n):
        ok = ~(np.isnan(r_grid[j]) | np.isnan(d_grid[j]))
        if ok.sum() >= 3:
            rho[j] = spearman(r_grid[j, ok], d_grid[j, ok])
        if ok.sum() >= 2:
            b1[j] = float(np.argmin(r_grid[j, ok]) == np.argmin(d_grid[j, ok]))
    return {"per_step_rho": rho, "bottom1_ok": b1}


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    ok = ~np.isnan(v)
    den = w[ok].sum()
    return float((v[ok] * w[ok]).sum() / den) if den > 0 else math.nan


def score_stats(c: dict, score, delta, *, keep, W: np.ndarray, M: int) -> dict:
    """Every A1.1 statistic of one score against one knockout's Delta, on the kept
    cells, each with its cluster-bootstrap CI from the same `W`.

    Decision-bearing: `rho_Q` (primary) and `rho_Q_rank`. Secondary, never
    decision-bearing: `rho_pool` (all full-memory cells), `rho_N`, `rho_rank_all`,
    `rho_step_Q` / `bottom1_Q`, `rho_step_full` / `bottom1_full`, `rho_all_steps`
    (underfull included) and `mass_fraction_N`."""
    score = np.asarray(score, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    have = np.asarray(keep, dtype=bool) & ~(np.isnan(score) | np.isnan(delta))
    st = strata(c)
    full = have & st["full"]
    Q, N = have & st["Q"], have & st["N"]
    doc, rank = np.asarray(c["doc"]), np.asarray(c["rank"])
    ws = {
        nm: (WeightedSpearman(score[m], delta[m]), doc[m])
        for nm, m in (("Q", Q), ("N", N), ("pool", full), ("all_steps", have))
    }
    rk_Q = RankStrata(score[Q], delta[Q], rank[Q], n_ranks=M)
    rk_all = RankStrata(score[full], delta[full], rank[full], n_ranks=M)
    steps = {}
    for nm, m in (("Q", Q), ("full", full)):
        rg, sdoc = _grid(c, m, score, M)
        dg, _ = _grid(c, m, delta, M)
        steps[nm] = (step_stats(rg, dg), sdoc)
    boots: dict[str, list] = {k: [] for k in (
        "Q", "N", "pool", "all_steps", "Q_rank", "rank_all", "step_Q", "b1_Q",
        "step_full", "b1_full",
    )}  # fmt: skip
    for w in W:
        for nm, (f, d_) in ws.items():
            boots[nm].append(f(w[d_]))
        boots["Q_rank"].append(rk_Q(w[doc[Q]])["rho"])
        boots["rank_all"].append(rk_all(w[doc[full]])["rho"])
        for nm in ("Q", "full"):
            s, sdoc = steps[nm]
            boots[f"step_{nm}"].append(_wmean(s["per_step_rho"], w[sdoc]))
            boots[f"b1_{nm}"].append(_wmean(s["bottom1_ok"], w[sdoc]))
    point = {nm: f(np.ones(len(d_))) for nm, (f, d_) in ws.items()}
    pq, pa = rk_Q(), rk_all()
    ci = {k: percentile_ci(v) for k, v in boots.items()}
    out = {
        "rho_Q": point["Q"],
        "rho_Q_ci": ci["Q"],
        "rho_Q_rank": pq["rho"],
        "rho_Q_rank_ci": ci["Q_rank"],
        "per_rank_Q": pq["per_rank"],
        "n_per_rank_Q": pq["n_per_rank"],
        "n_Q_cells": int(Q.sum()),
        "rho_N": point["N"],
        "rho_N_ci": ci["N"],
        "n_N_cells": int(N.sum()),
        "rho_pool": point["pool"],
        "rho_pool_ci": ci["pool"],
        "n_full_cells": int(full.sum()),
        "rho_rank_all": pa["rho"],
        "rho_rank_all_ci": ci["rank_all"],
        "per_rank_all": pa["per_rank"],
        "rho_all_steps": point["all_steps"],
        "rho_all_steps_ci": ci["all_steps"],
        "n_all_cells": int(have.sum()),
        "bottom1_chance": 1.0 / M,
        "mass_fraction_N": (
            float(score[N].sum() / score[full].sum()) if score[full].sum() else math.nan
        ),
        "n_boot": len(W),
    }
    for nm in ("Q", "full"):
        s, sdoc = steps[nm]
        ones = np.ones(len(sdoc))
        out[f"rho_step_{nm}"] = _wmean(s["per_step_rho"], ones)
        out[f"rho_step_{nm}_ci"] = ci[f"step_{nm}"]
        out[f"bottom1_{nm}"] = _wmean(s["bottom1_ok"], ones)
        out[f"bottom1_{nm}_ci"] = ci[f"b1_{nm}"]
        out[f"n_steps_{nm}"] = len(sdoc)
    return out


# --------------------------------------------------------------------------- #
# §5 flatness (A1.1), §7 labels and §8 classification (A1.1, A1.2)
# --------------------------------------------------------------------------- #


def loo_flat_a(delta, a_mask) -> dict:
    """A1.1 §5 LOO flat: the median over A-cells of Delta < `LOO_FLAT_A` nats per
    sentence. A-cells without a Delta (no resample donor) are left out; none at
    all leaves the rule unreadable, which is exit 3."""
    d = np.asarray(delta, dtype=np.float64)[np.asarray(a_mask, dtype=bool)]
    d = d[~np.isnan(d)]
    if not len(d):
        raise MeasurementUndefined("§5 (A1.1): no A-cell carries a Delta")
    med = float(np.median(d))
    return {"median_a": med, "n_a": len(d), "flat": bool(med < LOO_FLAT_A)}


def loo_flat_q90(delta) -> dict:
    """§5's withdrawn q90 rule, reported as a secondary (A1.1)."""
    d = np.asarray(delta, dtype=np.float64)
    d = d[~np.isnan(d)]
    q90 = float(np.quantile(np.abs(d), 0.9)) if len(d) else math.nan
    return {"q90_abs_delta": q90, "flat": bool(q90 < LOO_FLAT_Q90)}


def r_flat(r_grid) -> dict:
    """§5 R_FLAT: the median over steps of the within-step CV (population sd /
    mean, PREREG-OPEN: ddof 0) of `r_i` over the step's slots. A slot with no
    value (A1.3's exclusion) is left out of its step."""
    cvs = []
    for row in np.asarray(r_grid, dtype=np.float64):
        v = row[~np.isnan(row)]
        if len(v) >= 2 and v.mean() > 0:
            cvs.append(float(v.std() / v.mean()))
    if not cvs:
        raise MeasurementUndefined("§5 R_FLAT: no step has two slots with an r_i")
    med = float(np.median(cvs))
    return {"median_cv": med, "n_steps": len(cvs), "flat": bool(med < R_FLAT)}


def seed_label(inp: dict, *, rho_star: float) -> str:
    """A1's §7 (A1.1), first match wins, on 95 % CIs, never a point. Retired from
    gating by A2.1: reported as `label_A1` (at the PREREG's proposed rho* = 0.5 and
    the {0.3, 0.7} sensitivities), and scored against §9's prediction (A2.14).
    PREREG-OPEN: a PC CI that is undefined (NaN) cannot reach rho*, so it reads
    CEILING; an undefined r_i CI falls through to UNRESOLVED."""
    pc_hi = inp["pc_rho_Q_ci"][1]
    q_lo, q_hi = inp["rho_Q_ci"]
    k_lo, k_hi = inp["rho_Q_rank_ci"]
    if not pc_hi >= rho_star:
        return "CEILING"
    if inp["loo_flat"] and inp["r_flat"]:
        return "DEGENERATE"
    if inp["loo_flat"]:
        return "LOO_UNINFORMATIVE"
    if q_hi < 0:
        return "INVERTED"
    if q_hi < rho_star:
        return "DISAGREE"
    if q_lo >= rho_star and k_lo >= rho_star:
        return "AGREE"
    if q_lo >= rho_star and k_hi < rho_star:
        return "AGREE_VIA_RANK"
    return "UNRESOLVED"


# --------------------------------------------------------------------------- #
# Amendment 2: C11, the Q-step population, the within-step percentile, the
# critical label, the AUROC family, C_ws and H (A2.2-A2.6)
# --------------------------------------------------------------------------- #


def critical_labels(delta, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """A2.2: `y = 1[Delta > tau_s]`, strict. A NaN Delta (no resample donor) has no
    label. Returns `(has, y)`. Computes no quantile: tau is the frozen constant."""
    d = np.asarray(delta, dtype=np.float64)
    has = ~np.isnan(d)
    y = has & (d > tau)
    return has, y


def check_ages(c: dict, M: int) -> dict:
    """C11 (A2.3, A2.13): at every full-memory step the slots hold exactly one slot
    per age 1..M, and write-order rank = M - age. Any violation is exit 3."""
    full = np.asarray(c["full"], dtype=bool)
    age = (np.asarray(c["t"]) - np.asarray(c["sentence"]))[full]
    rank = np.asarray(c["rank"])[full]
    if not len(age):
        raise ControlFailed("C11", "no full-memory step to check")
    bad_rank = int((rank != M - age).sum())
    key = _step_key(c)[full]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    in_range = (age >= 1) & (age <= M)
    pairs = inv.astype(np.int64) * (M + 2) + np.clip(age, 0, M + 1)
    dup = len(pairs) - len(np.unique(pairs))
    short = int((cnt != M).sum())
    if bad_rank or dup or short or not in_range.all():
        raise ControlFailed(
            "C11", f"age strata broken at full-memory steps: {bad_rank} cells with "
            f"rank != M - age, {dup} duplicated ages, {short} steps without {M} "
            f"slots, {int((~in_range).sum())} ages outside 1..{M} (PREREG A2.3)"
        )  # fmt: skip
    return {"ok": True, "n_full_steps": len(uniq), "n_cells": len(age)}


def q_population(c: dict, keep, *, M: int) -> dict:
    """A2.4 "Population": the cells of A1.1's Q-steps (full memory, sentence t a
    query) that `keep` retains (A1.3's exclusions for the operative population).
    Every such cell is an eligible slot, whether or not its Delta has a value.

    C11 under the population: every step holds the same number of eligible slots,
    M (all cells) or M - 1 (A1.3: rank M - 1 out); anything else is exit 3."""
    sel = strata(c)["Q"] & np.asarray(keep, dtype=bool)
    idx = np.flatnonzero(sel)
    key = _step_key(c)[idx]
    _, step, n_t = np.unique(key, return_inverse=True, return_counts=True)
    doc = np.asarray(c["doc"])[idx]
    step_doc = np.zeros(len(n_t), dtype=np.int64)
    step_doc[step] = doc
    age = (np.asarray(c["t"]) - np.asarray(c["sentence"]))[idx]
    vals = sorted({int(x) for x in n_t})
    if len(vals) > 1 or (vals and vals[0] not in (M, M - 1)):
        raise ControlFailed(
            "C11", f"eligible slots per Q-step {vals}, expected one of {M} (all "
            f"cells) or {M - 1} (A1.3) at every step (PREREG A2.4, A2.13)"
        )  # fmt: skip
    order = np.argsort(step, kind="stable")
    return {
        "idx": idx[order],
        "step": step[order],
        "n_steps": len(n_t),
        "n_t": n_t,
        "doc": doc[order],
        "step_doc": step_doc,
        "age": age[order],
        "M": M,
    }


def step_percentiles(score, pop: dict) -> np.ndarray:
    """A2.4 step 1: at each step, over ALL its eligible slots (a NaN-Delta slot
    included), the midrank of r (rank 1 = smallest, ties share the mean rank) and
    `pct = (midrank - 1) / (n_t - 1)`. A non-finite r on an eligible slot is
    `MeasurementUndefined` (exit 3)."""
    v = np.asarray(score, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise MeasurementUndefined(
            f"A2.4: {int((~np.isfinite(v)).sum())} eligible slot(s) with a non-finite r"
        )
    step = pop["step"]
    n = len(v)
    order = np.lexsort((v, step))
    vs, ss = v[order], step[order]
    start = np.zeros(pop["n_steps"], dtype=np.int64)
    start[1:] = np.cumsum(pop["n_t"])[:-1]
    pos = np.arange(n, dtype=np.float64) - start[ss] + 1.0
    new = np.ones(n, dtype=bool)
    new[1:] = (vs[1:] != vs[:-1]) | (ss[1:] != ss[:-1])
    gid = np.cumsum(new) - 1
    mid = np.bincount(gid, weights=pos) / np.bincount(gid)
    mr = np.empty(n)
    mr[order] = mid[gid]
    return (mr - 1.0) / (pop["n_t"][pop["step"]] - 1.0)


def tied_minima(score, pop: dict) -> np.ndarray:
    """A2.6: the step's tied minima of r over its eligible slots."""
    v = np.asarray(score, dtype=np.float64)
    mn = np.full(pop["n_steps"], np.inf)
    np.minimum.at(mn, pop["step"], v)
    return v == mn[pop["step"]]


class StratAUROC:
    """A2.4 steps 3-4 on labelled cells: per bin, the Mann-Whitney probability that
    a positive's value exceeds a negative's (equal values count 1/2), over every
    positive-negative pair in the bin; bins lacking a class are excluded (never 0,
    never 1/2); the bins are averaged with weight `w_a` = the positives in the bin.
    `__call__(w)` evaluates it on a sample in which cell j appears `w[j]` times."""

    def __init__(self, values, y, bins, all_bins=()) -> None:
        self.all_bins = tuple(int(x) for x in all_bins)
        v = np.asarray(values, dtype=np.float64)
        b = np.asarray(bins, dtype=np.int64)
        self.y = np.asarray(y, dtype=bool)
        self.n = len(v)
        order = np.lexsort((v, b))
        vs, bs = v[order], b[order]
        new = np.ones(self.n, dtype=bool)
        new[1:] = (vs[1:] != vs[:-1]) | (bs[1:] != bs[:-1])
        gsorted = np.cumsum(new) - 1
        self.gid = np.empty(self.n, dtype=np.int64)
        self.gid[order] = gsorted
        self.G = int(gsorted[-1]) + 1 if self.n else 0
        gbin = bs[new]
        self.bins, self.gbin = np.unique(gbin, return_inverse=True)
        self.nb = len(self.bins)
        self.bin_first = np.searchsorted(self.gbin, np.arange(self.nb))

    def per_bin(self, w=None):
        w = np.ones(self.n) if w is None else np.asarray(w, dtype=np.float64)
        yp, yn = self.y, ~self.y
        wp = np.bincount(self.gid[yp], weights=w[yp], minlength=self.G)
        wn = np.bincount(self.gid[yn], weights=w[yn], minlength=self.G)
        cn = np.cumsum(wn)
        before = np.concatenate([[0.0], cn])[self.bin_first]
        below = cn - wn - before[self.gbin]
        num = np.bincount(self.gbin, weights=wp * (below + 0.5 * wn), minlength=self.nb)
        P = np.bincount(self.gbin, weights=wp, minlength=self.nb)
        N = np.bincount(self.gbin, weights=wn, minlength=self.nb)
        return num, P, N

    def __call__(self, w=None) -> float:
        if not self.n:
            return math.nan
        num, P, N = self.per_bin(w)
        both = (P > 0) & (N > 0)
        if not both.any():
            return math.nan
        au = num[both] / (P[both] * N[both])
        wa = P[both]
        return float((wa * au).sum() / wa.sum())

    def detail(self) -> tuple[list, list]:
        """Per-bin AUROC with its counts, and the excluded bins with theirs (A2.4
        step 3, A2.7 item 4). A registered bin with no labelled cell at all (bin {1}
        under A1.3, A2.3) is logged as excluded with 0 / 0."""
        used, excluded = [], []
        seen = set()
        if self.n:
            num, P, N = self.per_bin()
            for j, b in enumerate(self.bins.tolist()):
                seen.add(int(b))
                row = {"bin": int(b), "positives": int(P[j]), "negatives": int(N[j])}
                if P[j] > 0 and N[j] > 0:
                    used.append({**row, "auroc": float(num[j] / (P[j] * N[j]))})
                else:
                    excluded.append(row)
        for b in self.all_bins:
            if b not in seen:
                excluded.append({"bin": b, "positives": 0, "negatives": 0})
        return used, sorted(excluded, key=lambda r: r["bin"])


def _wavg(x: np.ndarray, w: np.ndarray) -> float:
    den = w.sum()
    return float((w * x).sum() / den) if den > 0 else math.nan


class WithinStepC:
    """A2.5 `C_ws`: `m_a` = the mean pct over the labelled Q-step cells of age a;
    `v_t` = the mean over a step's critical cells of `pct - m_a`; `C_ws` = the mean
    of `v_t` over steps with at least one critical cell. `m_a` is re-estimated in
    every resample (T17)."""

    def __init__(self, pct, y, lab, pop) -> None:
        self.lab = np.asarray(lab, dtype=bool)
        self.pct = np.asarray(pct, dtype=np.float64)
        self.age = pop["age"]
        self.doc = pop["doc"]
        self.ages = np.unique(self.age[self.lab]) if self.lab.any() else np.array([])
        self.aix = np.searchsorted(self.ages, self.age)
        crit = np.asarray(y, dtype=bool)
        k = np.bincount(pop["step"], weights=crit, minlength=pop["n_steps"])
        self.steps = np.flatnonzero(k > 0)
        sidx = np.searchsorted(self.steps, pop["step"])
        on = crit
        self.k = k[self.steps]
        self.S = np.bincount(sidx[on], weights=self.pct[on], minlength=len(self.steps))
        # counts of critical cells by (step, age), to subtract m_a
        self.C = np.zeros((len(self.steps), len(self.ages)))
        np.add.at(self.C, (sidx[on], self.aix[on]), 1.0)
        self.step_doc = pop["step_doc"][self.steps]

    def __call__(self, w_doc=None) -> float:
        if not len(self.steps) or not len(self.ages):
            return math.nan
        wc = (np.ones(len(self.pct)) if w_doc is None else w_doc[self.doc])[self.lab]
        num = np.bincount(self.aix[self.lab], weights=wc * self.pct[self.lab],
                          minlength=len(self.ages))  # fmt: skip
        den = np.bincount(self.aix[self.lab], weights=wc, minlength=len(self.ages))
        with np.errstate(invalid="ignore", divide="ignore"):
            m_a = num / den
        v = (self.S - self.C @ np.nan_to_num(m_a)) / self.k
        ws = np.ones(len(v)) if w_doc is None else w_doc[self.step_doc]
        return _wavg(v, ws)


class ArgminHit:
    """A2.6, reported and never gating. `Q_crit` = Q-steps with k_t >= 1 critical
    cells whose every eligible slot has a Delta value (steps dropped for a missing
    value are counted). `h_t` = the critical share of the step's tied minima of r.
    H, H_random (k_t / n_t), H_age-random (pi from r's own tied minima on Q_crit,
    re-estimated per resample), R_H, R_H,uniform, H_FIFO (the oldest slot) and
    H_age-oracle (the best fixed age).

    `H_age-oracle` is the min over the ages the population actually holds: under
    A1.3 rank M - 1 (age 1) is out, and its empty column is not an age a fixed
    policy could evict (RESULTS.md erratum, 2026-09-29, post-data and reported
    only; the 1..M min read 0.0 on every seed there)."""

    def __init__(self, score, y, has, pop) -> None:
        ns, step = pop["n_steps"], pop["step"]
        y = np.asarray(y, dtype=bool)
        k = np.bincount(step, weights=y, minlength=ns)
        complete = np.bincount(step, weights=~np.asarray(has), minlength=ns) == 0
        self.dropped = int(((k >= 1) & ~complete).sum())
        qc = np.flatnonzero((k >= 1) & complete)
        self.steps = qc
        tm = tied_minima(score, pop)
        ages = np.arange(1, pop["M"] + 1)
        sidx = np.searchsorted(qc, step)
        on = np.isin(step, qc)
        nq = len(qc)
        self.Y = np.zeros((nq, len(ages)))
        self.T = np.zeros((nq, len(ages)))
        ai = pop["age"] - 1
        self.Y[sidx[on], ai[on]] = y[on]
        self.T[sidx[on], ai[on]] = tm[on]
        n_min = self.T.sum(1)
        self.T = self.T / np.where(n_min > 0, n_min, 1.0)[:, None]
        self.h = (self.T * self.Y).sum(1)
        self.kn = k[qc] / pop["n_t"][qc]
        self.step_doc = pop["step_doc"][qc]
        self.fifo = self.Y[:, pop["M"] - 1] if nq else np.zeros(0)
        self.present = np.unique(pop["age"]) - 1  # column index of each held age

    def __call__(self, w_doc=None) -> dict:
        nq = len(self.steps)
        if not nq:
            return {"H": math.nan, "H_random": math.nan, "H_age_random": math.nan,
                    "R_H": math.nan, "R_H_uniform": math.nan, "H_FIFO": math.nan,
                    "H_age_oracle": math.nan}  # fmt: skip
        w = np.ones(nq) if w_doc is None else w_doc[self.step_doc]
        H = _wavg(self.h, w)
        H_random = _wavg(self.kn, w)
        W = w.sum()
        pi = (w[:, None] * self.T).sum(0) / W if W > 0 else np.full(self.T.shape[1], 0.0)
        h_ar = _wavg((self.Y * pi[None, :]).sum(1), w)
        by_age = (w[:, None] * self.Y).sum(0) / W if W > 0 else self.Y.sum(0) * math.nan
        return {
            "H": H,
            "H_random": H_random,
            "H_age_random": h_ar,
            "R_H": H / h_ar if h_ar > 0 else math.nan,
            "R_H_uniform": H / H_random if H_random > 0 else math.nan,
            "H_FIFO": _wavg(self.fifo, w),
            "H_age_oracle": float(np.min(by_age[self.present])),
        }


def auroc_family(c: dict, score, delta, *, tau: float, keep, W, M: int,
                 replicates: bool = False) -> dict:  # fmt: skip
    """Every A2.4-A2.6 statistic of one score against one knockout's Delta on the
    population `keep`, each with its interval from the same bootstrap matrix `W`
    (`[n_boot, n_docs]` document counts): `AUROC_strat,pct` (the gate), raw
    `AUROC_strat` (read by A2.8 row 5b only), `AUROC_unstrat,pct` (row 5 only),
    `C_ws`, and H with its baselines (reported). pct is computed once per step on
    the original data (a document's steps are resampled whole); `m_a` and pi are
    re-estimated per resample. Undefined resamples are counted (m-7)."""
    pop = q_population(c, keep, M=M)
    s = np.asarray(score, dtype=np.float64)[pop["idx"]]
    pct = step_percentiles(s, pop)
    d = np.asarray(delta, dtype=np.float64)[pop["idx"]]
    has, y = critical_labels(d, tau)
    lab = has  # A2.4 step 2: a NaN Delta leaves only its own cell; the step stays
    yl, al = y[lab], pop["age"][lab]
    bins = range(1, M + 1)  # A2.3: one bin per age, {1}, ..., {M}
    f = {
        "pct": StratAUROC(pct[lab], yl, al, bins),
        "raw": StratAUROC(s[lab], yl, al, bins),
        "unstrat_pct": StratAUROC(pct[lab], yl, np.zeros(int(lab.sum()), np.int64)),
    }
    cws = WithinStepC(pct, y & lab, lab, pop)
    hit = ArgminHit(s, y, has, pop)
    doc_l = pop["doc"][lab]
    out: dict = {"tau": float(tau)}
    reps: dict[str, list] = {k: [] for k in (*f, "c_ws", "H", "H_age_random", "R_H")}
    W = np.asarray(W, dtype=np.float64)
    for k in range(len(W)):
        w_doc = W[k]
        wl = w_doc[doc_l]
        for nm, fn in f.items():
            reps[nm].append(fn(wl))
        reps["c_ws"].append(cws(w_doc))
        hk = hit(w_doc)
        for nm in ("H", "H_age_random", "R_H"):
            reps[nm].append(hk[nm])
    for nm, fn in f.items():
        lo, hi, n_und = percentile_ci_defined(reps[nm])
        out[nm] = fn()
        out[f"{nm}_ci"] = (lo, hi)
        out[f"{nm}_n_undefined"] = n_und
    for nm in ("pct", "raw"):
        out[f"{nm}_per_bin"], out[f"{nm}_excluded_bins"] = f[nm].detail()
    out["c_ws"] = cws()
    lo, hi, n_und = percentile_ci_defined(reps["c_ws"])
    out["c_ws_ci"], out["c_ws_n_undefined"] = (lo, hi), n_und
    out.update(hit())
    for nm in ("H", "R_H"):
        lo, hi, n_und = percentile_ci_defined(reps[nm])
        out[f"{nm}_ci"], out[f"{nm}_n_undefined"] = (lo, hi), n_und
    # A2.6: R_H is undefined if H_age-random is 0 in the point or any resample
    ar = np.asarray(reps["H_age_random"], dtype=np.float64)
    out["R_H_defined"] = bool(
        out["H_age_random"] > 0 and not (len(ar) and (ar[~np.isnan(ar)] <= 0).any())
    )
    out.update(
        n_q_steps=int(pop["n_steps"]),
        n_cells=len(s),
        labelled=int(lab.sum()),
        unlabelled=int((~has).sum()),
        positives=int(yl.sum()),
        negatives=int((~yl).sum()),
        n_t_values=sorted({int(x) for x in pop["n_t"]}),
        n_q_crit=len(hit.steps),
        n_q_crit_dropped_nan=hit.dropped,
        n_boot=len(W),
    )
    if replicates:
        out["_replicates"] = {k: np.asarray(v, dtype=np.float64) for k, v in reps.items()}
    return out


# --------------------------------------------------------------------------- #
# A2.8 per-seed labels, the 5b boolean, §8 classification (row 4b, HALT)
# --------------------------------------------------------------------------- #


def seed_label_a2(inp: dict, *, auroc_star: float, stat: str = "pct") -> str:
    """A2.8, first matching row wins, in the order 0, 1, 2, 5b, 3, 4, 5, 6, 7, on
    95 % CIs, never a point. `A* = auroc_star` is read from the C8 rulings.

    `stat="raw"` is A2.5's raw-substituted label (STEP_SENSITIVE): raw
    `AUROC_strat` (and the control's raw) in place of `AUROC_strat,pct`, with row
    5b skipped. PREREG-OPEN: an undefined (NaN) control CI cannot reach A*, so it
    reads CEILING, as A1's C10 did."""
    a = auroc_star
    if stat == "pct":
        lo, hi = inp["pct_ci"]
        pc_hi = inp["pc_pct_ci"][1]
    else:
        lo, hi = inp["raw_ci"]
        pc_hi = inp.get("pc_raw_ci", inp["pc_pct_ci"])[1]
    raw_lo = inp["raw_ci"][0]
    un_lo = inp["unstrat_ci"][0]
    if not pc_hi >= a:  # row 0: C10', the true_demand control
        return "CEILING"
    if inp["loo_flat"] and inp["r_flat"]:  # row 1
        return "DEGENERATE"
    if inp["loo_flat"]:  # row 2
        return "LOO_UNINFORMATIVE"
    if stat == "pct" and hi < a and raw_lo >= a:  # row 5b, ahead of row 3
        return STEP_OR_AGE
    if hi < AUROC_INVERTED:  # row 3
        return "INVERTED"
    if lo >= a:  # row 4: the only pass, on the pct form
        return "AGREE"
    if hi < a and un_lo >= a:  # row 5
        return "AGREE_VIA_RANK"
    if hi < a:  # row 6
        return "DISAGREE"
    return "UNRESOLVED"  # row 7


def row5b(inp: dict, *, auroc_star: float) -> bool:
    """A2.8: whether row 5b's condition (pct CI upper < A* and raw CI lower >= A*)
    holds, reported on every seed whatever its label (rows 0-2 may absorb it)."""
    return bool(inp["pct_ci"][1] < auroc_star and inp["raw_ci"][0] >= auroc_star)


#: §8 rows 1-6 and 4b -> exit (A1.2: RECENCY_ONLY is 2; exit 1 only from rows 3, 4).
CLASS_EXIT = {
    "DEGENERATE_UNINFORMATIVE": 2,
    "AGREE": 0,
    "CONFOUND_INVERTED": 1,
    "CONFOUND": 1,
    STEP_OR_AGE: 2,
    "RECENCY_ONLY": 2,
    "MIXED_UNRESOLVED": 2,
}
CLASS_KEY = "e0d.class"
#: §8 row 4b's label set (A2.8).
ROW_4B = ("DISAGREE", "INVERTED", "AGREE_VIA_RANK", STEP_OR_AGE)


def classify(labels) -> tuple[str, int]:
    """§8 as replaced by A2.8, rows 1-6 with 4b, over the 3 seeds' labels (row 0 --
    any control failing or anything raising -- is exit 3 in `main`). CEILING counts
    as UNRESOLVED. A STEP_OR_AGE_AMBIGUOUS seed never counts toward a kill. Also
    classifies A1's labels (which never hold 5b) for `class_A1` (A2.14)."""
    ls = ["UNRESOLVED" if x == "CEILING" else x for x in labels]
    if len(ls) != len(SEEDS):
        raise ValueError(f"§8 classifies {len(SEEDS)} seeds, got {len(ls)} labels")
    if sum(x in ("DEGENERATE", "LOO_UNINFORMATIVE") for x in ls) >= 2:
        k = "DEGENERATE_UNINFORMATIVE"
    elif all(x == "AGREE" for x in ls):
        k = "AGREE"
    elif all(x == "INVERTED" for x in ls):
        k = "CONFOUND_INVERTED"
    elif all(x in ("DISAGREE", "INVERTED") for x in ls):
        k = "CONFOUND"
    elif STEP_OR_AGE in ls and all(x in ROW_4B for x in ls):  # row 4b
        k = STEP_OR_AGE
    elif all(x in ("DISAGREE", "INVERTED", "AGREE_VIA_RANK") for x in ls) and (
        "AGREE_VIA_RANK" in ls
    ):
        k = "RECENCY_ONLY"
    else:
        k = "MIXED_UNRESOLVED"
    return k, CLASS_EXIT[k]


# --------------------------------------------------------------------------- #
# per-seed analysis and the run's classification
# --------------------------------------------------------------------------- #

KNOCKOUTS = ("resample", "zero")
SCORES = (("gated", "r_gated"), ("ungated", "r_ungated"), ("pc", "pc"))
COMBOS = tuple(f"{g}_{k}" for g in ("gated", "ungated") for k in KNOCKOUTS)
PRIMARY = "gated_resample"
TAU_SENS_Q = ("0.99", "0.999")


def _a2_label(inp: dict, *, auroc_star: float, stat: str = "pct") -> str:
    """A2.8 with A2.4's "Undefined" rule: if no bin has both classes, the statistic
    has no value; the LOO-flat rule may still label the seed (rows 1-2), else the
    label is UNDEFINED (the primary cell turns that into exit 3 in analyse_seed).
    PREREG-OPEN: the control shares the labels, so it is undefined too; row 0 is
    then not evaluable and rows 1-2 label directly."""
    if math.isnan(inp["pct"]):
        if inp["loo_flat"]:
            return "DEGENERATE" if inp["r_flat"] else "LOO_UNINFORMATIVE"
        return "UNDEFINED"
    return seed_label_a2(inp, auroc_star=auroc_star, stat=stat)


def sink_probe(c: dict, score, keep) -> dict:
    """A2.7 item 8 (reported): the mean r by sentence kind at Q-steps, and at
    full-memory steps where a fact's assert and its already-read query are both
    resident, the fraction with r_assert > r_query."""
    score = np.asarray(score, dtype=np.float64)
    st = strata(c)
    keep = np.asarray(keep, dtype=bool)
    Q = st["Q"] & keep
    kind = np.asarray(c.get("kind", np.full(len(score), "", dtype=object)))
    by_kind = {
        str(k): {"mean_r": float(score[Q & (kind == k)].mean()),
                 "n": int((Q & (kind == k)).sum())}
        for k in sorted({str(x) for x in kind[Q]})
    }  # fmt: skip
    out = {"mean_r_by_kind_Q": by_kind, "assert_above_read_query": None}
    qo = c.get("query_of")
    if qo is None:
        return out
    qo = np.asarray(qo)
    full = st["full"] & keep
    t, sent = np.asarray(c["t"]), np.asarray(c["sentence"])
    base = _step_key(c) * (int(np.max(sent)) + int(np.max(t)) + 2)
    key = base + sent
    order = np.argsort(key[full])
    fk, fr = key[full][order], score[full][order]
    a = np.flatnonzero(full & (qo >= 0) & (qo < t))
    want = base[a] + qo[a]
    j = np.searchsorted(fk, want)
    hit = (j < len(fk)) & (fk[np.minimum(j, len(fk) - 1)] == want)
    if hit.any():
        above = score[a][hit] > fr[j[hit]]
        out["assert_above_read_query"] = {"fraction": float(above.mean()),
                                          "n_pairs": int(hit.sum())}  # fmt: skip
    return out


def _population(c: dict, keep: np.ndarray, *, M: int, W, auroc_star: float,
                tau: dict, descriptive: bool) -> dict:  # fmt: skip
    st = strata(c)
    # ---- A1's rho statistics and label_A1 (A2.1: reported, never gating) -------
    stats = {
        sc: {k: score_stats(c, c[col], c[f"d_{k}"], keep=keep, W=W, M=M)
             for k in KNOCKOUTS}
        for sc, col in SCORES
    }  # fmt: skip
    A = st["A"] & keep
    # PREREG-OPEN: the zero-knockout table reads its LOO flatness on Delta_zero.
    lf = {k: loo_flat_a(c[f"d_{k}"], A) for k in KNOCKOUTS}
    fk = st["full"] & keep
    rf = {g: r_flat(_grid(c, fk, c[f"r_{g}"], M)[0]) for g in ("gated", "ungated")}
    _, first = np.unique(_step_key(c)[fk], return_index=True)
    step_q = np.asarray(c["q_step"])[fk][first]
    flat = {
        "loo_flat_resample": lf["resample"],
        "loo_flat_zero": lf["zero"],
        "q90_abs_delta_resample": loo_flat_q90(c["d_resample"][fk])["q90_abs_delta"],
        "q90_abs_delta_zero": loo_flat_q90(c["d_zero"][fk])["q90_abs_delta"],
        "r_flat_gated": rf["gated"],
        "r_flat_ungated": rf["ungated"],
    }
    for g in ("gated", "ungated"):
        pre = np.asarray(c[f"pre_share_{g}"], dtype=np.float64)[fk][first]
        flat[f"pre_share_{g}_median"] = float(np.median(pre)) if len(pre) else math.nan
        flat[f"pre_share_{g}_mass_fraction_N"] = (
            float(pre[~step_q].sum() / pre.sum()) if pre.sum() else math.nan
        )

    def inp_a1(g: str, k: str) -> dict:
        return {
            "pc_rho_Q_ci": stats["pc"][k]["rho_Q_ci"],
            "loo_flat": lf[k]["flat"],
            "r_flat": rf[g]["flat"],
            "rho_Q_ci": stats[g][k]["rho_Q_ci"],
            "rho_Q_rank_ci": stats[g][k]["rho_Q_rank_ci"],
        }

    rho0 = RHO_STAR_PROPOSED
    labels_a1 = {
        "gated_resample": seed_label(inp_a1("gated", "resample"), rho_star=rho0),
        "ungated_resample": seed_label(inp_a1("ungated", "resample"), rho_star=rho0),
        "gated_zero": seed_label(inp_a1("gated", "zero"), rho_star=rho0),
    }

    # ---- Amendment 2: the AUROC family, the A2.8 labels -------------------------
    def fam(score, k, t):
        return auroc_family(c, score, c[f"d_{k}"], tau=t, keep=keep, W=W, M=M)

    def inp_a2(r_fam: dict, pc_fam: dict, g: str, k: str) -> dict:
        return {
            "pc_pct_ci": pc_fam["pct_ci"],
            "pc_raw_ci": pc_fam["raw_ci"],
            "loo_flat": lf[k]["flat"],
            "r_flat": rf[g]["flat"],
            "pct": r_fam["pct"],
            "pct_ci": r_fam["pct_ci"],
            "raw_ci": r_fam["raw_ci"],
            "unstrat_ci": r_fam["unstrat_pct_ci"],
            "H": r_fam["H"],
            "R_H": r_fam["R_H"],
        }

    pc_fam = {k: fam(c["pc"], k, tau[k]) for k in KNOCKOUTS}
    a2, labels, raw_sub, b5 = {}, {}, {}, {}
    for g in ("gated", "ungated"):
        for k in KNOCKOUTS:
            combo = f"{g}_{k}"
            r_fam = fam(c[f"r_{g}"], k, tau[k])
            a2[combo] = {"r": r_fam, "pc": pc_fam[k]}
            inp = inp_a2(r_fam, pc_fam[k], g, k)
            labels[combo] = _a2_label(inp, auroc_star=auroc_star)
            raw_sub[combo] = _a2_label(inp, auroc_star=auroc_star, stat="raw")
            b5[combo] = row5b(inp, auroc_star=auroc_star)
    tau_sens = {}
    for q in TAU_SENS_Q:
        tq = tau["sensitivity"][q]
        pcq = {k: fam(c["pc"], k, tq[k]) for k in KNOCKOUTS}
        tau_sens[q] = {}
        for g in ("gated", "ungated"):
            for k in KNOCKOUTS:
                rq = fam(c[f"r_{g}"], k, tq[k])
                tau_sens[q][f"{g}_{k}"] = _a2_label(
                    inp_a2(rq, pcq[k], g, k), auroc_star=auroc_star
                )
    out = {
        "n_cells": int(np.asarray(keep).sum()),
        "a2": a2,
        "labels": labels,
        "row5b": b5,
        "labels_raw_substituted": raw_sub,
        "step_sensitive": {k: raw_sub[k] != labels[k] for k in COMBOS},
        "tau_sensitivity": tau_sens,
        "stats": stats,
        "flatness": flat,
        "labels_A1": labels_a1,
        "sensitivity_A1": {
            str(x): seed_label(inp_a1("gated", "resample"), rho_star=x)
            for x in SENSITIVITY_RHO
        },
        "flags": {
            "loo_flat": lf["resample"]["flat"],
            "r_flat": rf["gated"]["flat"],
            "r_flat_flag": bool(rf["gated"]["flat"] and not lf["resample"]["flat"]),
            "pc_rank_ceiling": bool(
                not stats["pc"]["resample"]["rho_Q_rank_ci"][1] >= rho0
            ),
            "pc_rho_Q_rank_ci": stats["pc"]["resample"]["rho_Q_rank_ci"],
        },
        "sink_probe": {g: sink_probe(c, c[f"r_{g}"], keep) for g in ("gated", "ungated")},
    }
    if descriptive:  # §3.1 items 5-6, unchanged by A1.1, on full-memory cells
        full = st["full"]
        d = np.asarray(c["d_resample"], dtype=np.float64)
        out["per_layer"] = {
            g: [spearman(np.asarray(c[f"layer_{g}"])[full, k], d[full])
                for k in range(np.asarray(c[f"layer_{g}"]).shape[1])]
            for g in ("gated", "ungated")
        }  # fmt: skip
        out["by_kind"] = {
            kd: {
                "rho_gated_resample": spearman(
                    np.asarray(c["r_gated"])[full & (c["kind"] == kd)],
                    d[full & (c["kind"] == kd)],
                ),
                "n": int((full & (c["kind"] == kd)).sum()),
            }
            for kd in CONTENT_KINDS
        }
    return out


def analyse_seed(c: dict, *, seed: int, M: int, n_docs: int, auroc_star: float,
                 n_boot: int | None = None) -> dict:  # fmt: skip
    """One seed, under Amendment 2: C11, then every A2.4-A2.7 statistic and the
    A2.8 labels for gated/ungated `r_i` x resample/zero with `true_demand` as the
    C10' control, at the seed's own frozen tau (A2.2, A2.10), plus A1's rho
    statistics and `label_A1` -- on all cells and, as one recomputation, with
    A1.3's bos-copy exclusions. One bootstrap matrix for every statistic of the
    seed (T17). The exit reads gated x resample on the excluded population."""
    c11 = check_ages(c, M)  # C11
    n_boot = BOOT_N if n_boot is None else n_boot
    W = bootstrap_doc_weights(n_docs, n_boot, BOOT_SEED)
    tau = {k: tau_for(seed, k) for k in KNOCKOUTS}
    tau["sensitivity"] = {q: {k: tau_for(seed, k, q) for k in KNOCKOUTS}
                          for q in TAU_SENS_Q}  # fmt: skip
    n = len(np.asarray(c["t"]))
    kw = dict(M=M, W=W, auroc_star=auroc_star, tau=tau)
    pops = {
        "all": _population(c, np.ones(n, dtype=bool), descriptive=True, **kw),
        "bos_excluded": _population(c, bos_keep(c, M), descriptive=False, **kw),
    }
    if pops["bos_excluded"]["labels"][PRIMARY] == "UNDEFINED":
        raise MeasurementUndefined(
            f"A2.4: seed {seed}'s AUROC_strat,pct has no value (no age bin holds both "
            f"a critical and a non-critical cell) and LOO is not flat: exit 3"
        )
    return {
        "populations": pops,
        "seed": seed,
        "auroc_star": auroc_star,
        "tau": tau,
        "c11": c11,
        "n_boot": n_boot,
        "bootstrap_seed": BOOT_SEED,
    }


def pooled_primary(cells_by_seed: dict, *, M: int, n_docs: dict, auroc_star: float,
                   n_boot: int) -> dict:  # fmt: skip
    """A2.7 (reported, never gating): the primary cell pooled over seeds, each
    seed's cells labelled with its own tau and ranked within its own steps, with a
    seed-stratified bootstrap (each seed's documents resampled by its own W)."""
    parts = {nm: [] for nm in ("pct", "raw", "pc", "y", "age", "doc")}
    Ws, off = [], 0
    for s in sorted(cells_by_seed):
        c = cells_by_seed[s]
        pop = q_population(c, bos_keep(c, M), M=M)
        r = np.asarray(c["r_gated"], dtype=np.float64)[pop["idx"]]
        pc = np.asarray(c["pc"], dtype=np.float64)[pop["idx"]]
        has, y = critical_labels(np.asarray(c["d_resample"])[pop["idx"]],
                                 tau_for(s, "resample"))  # fmt: skip
        parts["pct"].append(step_percentiles(r, pop)[has])
        parts["raw"].append(r[has])
        parts["pc"].append(step_percentiles(pc, pop)[has])
        parts["y"].append(y[has])
        parts["age"].append(pop["age"][has])
        parts["doc"].append(pop["doc"][has] + off)
        Ws.append(bootstrap_doc_weights(n_docs[s], n_boot, BOOT_SEED))
        off += n_docs[s]
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    W = np.concatenate(Ws, axis=1)
    f = {
        "pct": StratAUROC(cat["pct"], cat["y"], cat["age"]),
        "raw": StratAUROC(cat["raw"], cat["y"], cat["age"]),
        "unstrat_pct": StratAUROC(cat["pct"], cat["y"], np.zeros(len(cat["y"]), int)),
        "pc_pct": StratAUROC(cat["pc"], cat["y"], cat["age"]),
    }
    out = {}
    for nm, fn in f.items():
        reps = [fn(W[k][cat["doc"]]) for k in range(len(W))]
        lo, hi, n_und = percentile_ci_defined(reps)
        out[nm], out[f"{nm}_ci"], out[f"{nm}_n_undefined"] = fn(), (lo, hi), n_und
    out["label"] = seed_label_a2(
        {"pc_pct_ci": out["pc_pct_ci"], "loo_flat": False, "r_flat": False,
         "pct_ci": out["pct_ci"], "raw_ci": out["raw_ci"],
         "unstrat_ci": out["unstrat_pct_ci"]},
        auroc_star=auroc_star,
    )  # fmt: skip
    out["note"] = "reported only; never gates (ruling: each seed separately)"
    return out


def classify_run(analyses: dict, pooled: dict | None = None) -> dict:
    """§8 (A2.8) over the seeds. The exit follows gated x resample on the
    A1.3-excluded population; HALT on any STEP_OR_AGE_AMBIGUOUS seed (exit 2, the
    seeds named, the next E0d action is Brendan's). Every other table -- all
    cells, ungated, zero, raw-substituted, tau sensitivity, label_A1, pooled -- is
    reported and moves no exit."""
    seeds = sorted(analyses)

    def pop(s, p="bos_excluded"):
        return analyses[s]["populations"][p]

    def labs(combo, p="bos_excluded", key="labels"):
        return [pop(s, p)[key][combo] for s in seeds]

    labels = labs("gated_resample")
    klass, code = classify(labels)
    halt_seeds = [s for s, lab in zip(seeds, labels, strict=True) if lab == STEP_OR_AGE]
    if halt_seeds and code != 2:  # A2.8 "HALT on row 5b": exit 2, never a kill
        raise RuntimeError(f"HALT seeds {halt_seeds} but class {klass} exits {code}")
    labels_all = labs("gated_resample", "all")
    klass_all, _ = classify(labels_all)
    klass_u, _ = classify(labs("ungated_resample"))
    klass_z, _ = classify(labs("gated_zero"))
    flags = [pop(s)["flags"] for s in seeds]
    labels_a1 = labs("gated_resample", key="labels_A1")
    lf_seeds = [s for s, f in zip(seeds, flags, strict=True) if f["loo_flat"]]
    out = {
        "class": klass,
        "exit": code,
        "labels": labels,
        "row5b": {str(s): bool(pop(s)["row5b"][PRIMARY]) for s in seeds},
        "halt": bool(halt_seeds),
        "halt_seeds": halt_seeds,
        "class_all_cells": klass_all,
        "labels_all_cells": labels_all,
        "bos_sensitive": klass_all != klass,
        "class_ungated": klass_u,
        "labels_ungated": labs("ungated_resample"),
        "gate_sensitive": klass_u != klass,
        "class_zero": klass_z,
        "labels_zero": labs("gated_zero"),
        "class_ungated_all_cells": classify(labs("ungated_resample", "all"))[0],
        "class_zero_all_cells": classify(labs("gated_zero", "all"))[0],
        "labels_raw_substituted": labs(PRIMARY, key="labels_raw_substituted"),
        "step_sensitive": {
            str(s): pop(s)["labels_raw_substituted"][PRIMARY] != lab
            for s, lab in zip(seeds, labels, strict=True)
        },
        "tau_sensitivity": {
            q: {
                "labels": (ls := [pop(s)["tau_sensitivity"][q][PRIMARY] for s in seeds]),
                "class": classify(ls)[0],
            }
            for q in TAU_SENS_Q
        },
        "labels_A1": labels_a1,
        "class_A1": classify(labels_a1)[0],
        "sensitivity": {
            str(x): {
                "labels": (ls := [pop(s)["sensitivity_A1"][str(x)] for s in seeds]),
                "class": classify(ls)[0],
            }
            for x in SENSITIVITY_RHO
        },
        "rank_ceiling_seeds": [
            s
            for s, f, lab in zip(seeds, flags, labels_a1, strict=True)
            if f["pc_rank_ceiling"] and lab == "AGREE_VIA_RANK"
        ],
        "seed_flatness": {
            str(s): {"loo_flat": f["loo_flat"], "r_flat": f["r_flat"]}
            for s, f in zip(seeds, flags, strict=True)
        },
        # A1.1 author's note, kept under A2.8 (row 0 still precedes rows 1-2):
        # >= 2 seeds meet §8 row 1's condition while one is labelled CEILING.
        "degenerate_under_ceiling": bool(
            len(lf_seeds) >= 2
            and any(labels[seeds.index(s)] == "CEILING" for s in lf_seeds)
            and klass != "DEGENERATE_UNINFORMATIVE"
        ),
    }
    if pooled is not None:
        out["pooled"] = pooled
    return out


def written_class(ledger_path: Path) -> str | None:
    rows = json.loads(Path(ledger_path).read_text()).get("rows", [])
    vals = [r.get("value") for r in rows if r.get("key") == CLASS_KEY]
    return vals[-1] if vals else None


def confirm_exit(ledger_path: Path, code: int) -> int:
    """A1.4: an exit code leaves only after the class it stands for has been
    written. Reads `e0d.class` back out of the written ledger; a missing or
    disagreeing class raises (-> exit 3)."""
    k = written_class(ledger_path)
    if k is None or CLASS_EXIT.get(k) != code:
        raise RuntimeError(
            f"exit {code} without a matching written {CLASS_KEY} (found {k!r} in "
            f"{ledger_path}); A1.4"
        )
    return code


# --------------------------------------------------------------------------- #
# per seed
# --------------------------------------------------------------------------- #


def header_config(seed_dir: Path) -> dict:
    """The checkpoint's own frozen config: its heartbeat header."""
    first = (Path(seed_dir) / "heartbeat.jsonl").read_text().splitlines()[0]
    return json.loads(first)["config"]


def measure_seed(
    seed: int, *, ckpt_root: Path, doc_range, batch: int, cleared: bool
) -> dict:
    seed_dir = Path(ckpt_root) / f"seed{seed}"
    path = seed_dir / CKPT_NAME
    config = header_config(seed_dir)
    cfg = TGConfig(**config["tg"])
    M = cfg.max_sentences_in_short_term
    S = int(config["steps_per_stream"])
    n_cross = sum(1 for b in cfg.block_config if b == "C")
    if (M, S) != (PREREG_M, PREREG_S):
        raise ControlFailed("C1", f"checkpoint config (M, S) = {(M, S)} != PREREG's")
    if config.get("policy") != "fifo":
        raise ControlFailed("C9", f"substrate policy {config.get('policy')!r} != fifo")
    model = TGModel(cfg)
    ck.load(path, model=model, restore_rng=False)  # refuses a quarantined ckpt
    model.eval()
    torch.manual_seed(seed)

    n_vocab = int(config.get("stream", {}).get("vocab_documents", 64))
    vdocs = generate(
        SyntheticConfig(sentences_per_document=S, seed=seed, n_documents=n_vocab)
    )
    vocab = build_vocab(vdocs)
    docs = e0d_documents(seed, S, doc_range, cleared=cleared)
    check_vocab(docs, vocab)  # C3
    ids, mask = encode(docs, vocab, max_tokens=cfg.max_sentence_tokens, steps=S)

    batches, docs_b, c4 = [], [], []
    c7 = None
    for lo in range(0, len(docs), batch):
        sl = slice(lo, lo + batch)
        m = measure_batch(model, docs[sl], ids[sl], mask[sl], seed=seed)
        check_fifo(m["slot_sentence"], m["wrote"], M)  # C9
        if lo == 0:
            again = measure_batch(model, docs[sl], ids[sl], mask[sl], seed=seed)
            c7 = check_determinism(m, again)  # C7
        # keep only what the analysis reads (a batch's tensors are small)
        batches.append(m)
        docs_b.append(docs[sl])
        c4.append(m["c4"])
        print(
            f"seed {seed}: batch {lo // batch + 1}/{-(-len(docs) // batch)}", flush=True
        )
    cells = cells_from(batches, docs_b, M)
    n_live = torch.cat([m["n_live"] for m in batches]).numpy()
    tt = np.broadcast_to(np.arange(n_live.shape[1]), n_live.shape)
    c5 = check_fill(n_live.ravel(), tt.ravel(), M)  # C5
    c6 = check_coverage(cells["d_resample"], cells["full"])  # C6
    return {
        "seed": seed,
        "ckpt": str(path),
        "M": M,
        "S": S,
        "n_cross_attention_layers": n_cross,
        "n_docs": len(docs),
        "doc_range": list(doc_range),
        "cells": cells,
        "c4_worst_live_loss_diff": max(x["live_loss_max_abs_diff"] for x in c4),
        "c5": c5,
        "c6": c6,
        "c7": c7,
        "c9": {"fifo": True, "batches": len(batches)},
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


#: PREREG-OPEN: `scripts/ledger.py` has no status word for exit 2 ("ran, no
#: decision"); `partial` is used so a non-zero exit never sits beside `ok`.
_STATUS = {0: "ok", 1: "failed", 2: "partial"}
_OUTCOME = {0: "survived", 1: "falsified", 2: "inconclusive"}


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def write_claims(path: Path, *, ledger_path: Path, run_id: str, out: dict,
                 seeds: list[int], t0: dict) -> Path:  # fmt: skip
    """`runs/<run_id>/claims.json`: one `{claim, command, expected}` per checkable
    claim, each command runnable from a clean checkout of the head (compute is
    slot-wrapped). Written only by a run that reached a class."""
    led = _rel(ledger_path)
    py = ".venv/bin/python"
    row = (
        f"{py} -c \"import json; r=json.load(open('{led}'))['rows']; "
        "print(json.dumps([x['value'] for x in r if x['key']=='{k}'][-1]))\""
    )
    code = int(out["exit"])
    claims = [
        {"claim": f"E0d's §8 class (gated x resample, A1.3-excluded) is {out['class']}",
         "command": row.format(k=CLASS_KEY).replace("json.dumps(", "(", 1),
         "expected": out["class"]},
        {"claim": f"the run exited {code}",
         "command": f"{py} -c \"import json; print(json.load(open('{led}'))"
                    "['commands'][-1]['exit_code'])\"",
         "expected": str(code)},
        {"claim": f"the per-seed A2.8 labels are {out['labels']}",
         "command": row.format(k="e0d.labels"),
         "expected": json.dumps(out["labels"])},
        {"claim": f"the class recomputes from the ledger's labels as {out['class']}",
         "command": f"{py} -c \"import importlib.util as u, json; "
                    f"s=u.spec_from_file_location('e0d_run', '{EXPERIMENT}'); "
                    "m=u.module_from_spec(s); s.loader.exec_module(m); "
                    f"r=json.load(open('{led}'))['rows']; "
                    "print(m.classify([x['value'] for x in r "
                    "if x['key']=='e0d.labels'][-1])[0])\"",
         "expected": out["class"]},
        {"claim": f"seeds actually run: {seeds}",
         "command": f"{py} -c \"import json; print(json.load(open('{led}'))"
                    "['seeds_actually_run'])\"",
         "expected": str(seeds)},
        {"claim": f"the class without the bos-copy exclusions is "
                  f"{out['class_all_cells']}",
         "command": row.format(k="e0d.class_all_cells").replace("json.dumps(", "(", 1),
         "expected": out["class_all_cells"]},
        {"claim": f"the row-5b booleans per seed are {out['row5b']} and HALT is "
                  f"{out['halt']}",
         "command": row.format(k="e0d.row5b"),
         "expected": json.dumps(out["row5b"])},
        {"claim": "the T0 substrate manifest still verifies (C1, end)",
         "command": f"cd {t0['cwd']} && shasum -a 256 -c {t0['manifest']} "
                    "> /dev/null 2>&1; echo $?",
         "expected": "0"},
        {"claim": f"re-running the experiment reproduces exit {code}",
         "command": "PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run "
                    f"--lane cpu-det --slots 3 -- {py} {EXPERIMENT} "
                    f"--run-id {run_id}-reverify > /dev/null 2>&1; echo $?",
         "expected": str(code)},
    ]  # fmt: skip
    Path(path).write_text(json.dumps(claims, indent=2) + "\n")
    return Path(path)


#: The seed cache's own format. A change to what a record holds bumps it, so an
#: old record is a key mismatch (exit 3), never a silent partial reuse.
SEED_CACHE_FORMAT = 1


def _head_sha() -> str | None:
    r = subprocess.run(
        ["/opt/homebrew/bin/git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True,
    )  # fmt: skip
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def seed_key(seed: int, *, auroc_star: float) -> dict:
    """What a persisted seed must match to be reused: the code (git sha and this
    file's own sha256 -- a dirty run.py moves the second when not the first), the
    seed, the documents, and the analysis inputs that are not in the code."""
    return {
        "format": SEED_CACHE_FORMAT,
        "git_sha": _head_sha(),
        "run_py_sha256": sha256(Path(__file__).resolve()),
        "seed": int(seed),
        "doc_range": [int(x) for x in D_E0D],
        "auroc_star": float(auroc_star),
        "n_boot": int(BOOT_N),
        "bootstrap_seed": int(BOOT_SEED),
        "batch": int(BATCH),
    }


class SeedCache:
    """Crash safety (runner only; RESULTS.md erratum, post-data): each seed's
    `measure_seed` record (cells included) and `analyse_seed` result, pickled --
    types exact, so a reused seed ledgers byte-for-byte what a fresh one would --
    to `<run dir>/seed-cache/seed<s>.pkl` as soon as the seed completes. Written
    to a temporary name, fsynced, then `os.replace`d: a kill leaves either the
    whole record or none. Nothing here is ever deleted."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, seed: int) -> Path:
        return self.root / f"seed{seed}.pkl"

    def load(self, seed: int, key: dict):
        """`(measured, analysis)` for a record whose key equals `key`, `None` if
        there is no record; a record that cannot be read or whose key differs in
        any field is `ControlFailed("RESUME")` -> exit 3."""
        p = self.path(seed)
        if not p.exists():
            return None
        try:
            rec = pickle.loads(p.read_bytes())
            got = rec["key"]
            meas, an = rec["measured"], rec["analysis"]
        except Exception as e:  # any unreadable record is exit 3
            raise ControlFailed(
                "RESUME", f"{p}: unreadable seed record ({type(e).__name__}: {e})"
            ) from e
        if key.get("git_sha") is None:
            raise ControlFailed(
                "RESUME", f"{p}: this run's git sha is unknown, so no key can match"
            )
        if got != key:
            diff = sorted(k for k in set(got) | set(key) if got.get(k) != key.get(k))
            raise ControlFailed(
                "RESUME", f"{p}: key mismatch on {diff} (recorded "
                f"{ {k: got.get(k) for k in diff} }, this run "
                f"{ {k: key.get(k) for k in diff} }); not reused, not overwritten"
            )  # fmt: skip
        return meas, an

    def save(self, seed: int, key: dict, measured: dict, analysis: dict) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(seed)
        tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
        blob = pickle.dumps(
            {"key": key, "measured": measured, "analysis": analysis},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        with open(tmp, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p


def _measure_and_classify(a, led, argv_s: str, note: str = "") -> Exit:
    auth = check_authority(a.rulings_dir)  # C8 -- before anything is read
    led.note("c8_authority", _jsonable(auth), how="docs/owner/rulings (A2.13)")
    c12 = check_tau_tables()  # C12 -- a seed without a frozen tau is not run
    led.note("c12_tau_tables", c12, how="PREREG A2.2 parsed at run time (A2.13)")
    led.note("c1_substrate_start", check_substrate(a.ckpt_root), how="sha256")
    t0 = t0_manifest(a.runs_dir)  # C1 (A1.7): no T0 record, no documents
    led.note("c1_t0_record", _jsonable(t0), how="runs/t0-substrate (A1.7)")
    disj = check_disjoint(D_E0D, seeds=SEEDS, used=used_ranges(a.runs_dir))
    led.note("c2_disjointness", disj, how="generator keys vs committed manifests")

    per_seed: dict[int, dict] = {}
    analyses: dict[int, dict] = {}
    cache = SeedCache(Path(led.path).parent / "seed-cache")
    keys = {s: seed_key(s, auroc_star=auth["auroc_star"]) for s in SEEDS}
    # every persisted seed is checked before any seed is measured: a mismatch is
    # exit 3 with nothing new computed
    reused = {s: hit for s in SEEDS if (hit := cache.load(s, keys[s])) is not None}
    led.note(
        "seed_cache",
        {"dir": str(cache.root), "reused_seeds": sorted(reused), "keys": keys},
        how="SeedCache (runner crash safety; RESULTS.md erratum 2026-09-29)",
    )
    for s in SEEDS:
        if s in reused:
            print(f"seed {s}: reusing {cache.path(s)} (key matches)", flush=True)
            per_seed[s], analyses[s] = reused[s]
            continue
        print(f"seed {s}: measuring", flush=True)
        per_seed[s] = measure_seed(
            s, ckpt_root=a.ckpt_root, doc_range=D_E0D, batch=BATCH, cleared=True
        )
        analyses[s] = analyse_seed(
            per_seed[s]["cells"], seed=s, M=per_seed[s]["M"],
            n_docs=per_seed[s]["n_docs"], auroc_star=auth["auroc_star"], n_boot=BOOT_N,
        )  # fmt: skip
        cache.save(s, keys[s], per_seed[s], analyses[s])
    end = substrate_recheck(t0["manifest"], cwd=t0["cwd"])  # C1, end
    led.note("c1_substrate_end", end, how="shasum -a 256 -c (PLAN-v4 T0, A1.7)")

    seeds = sorted(per_seed)
    led.run_meta(seeds_actually_run=seeds)
    for s in seeds:
        meas = {k: v for k, v in per_seed[s].items() if k != "cells"}
        led.note(f"seed{s}.controls", _jsonable(meas), how="measure_seed")
        led.note(f"seed{s}.analysis", _jsonable(analyses[s]), how="analyse_seed")
    pooled = pooled_primary(
        {s: per_seed[s]["cells"] for s in seeds}, M=per_seed[seeds[0]]["M"],
        n_docs={s: per_seed[s]["n_docs"] for s in seeds},
        auroc_star=auth["auroc_star"], n_boot=BOOT_N,
    )  # fmt: skip
    out = classify_run(analyses, pooled=pooled)
    for k, v in out.items():
        if k not in ("class", "exit"):
            led.note(f"e0d.{k}", _jsonable(v), how="§8 (A2.8), classify_run")
    prim = {s: analyses[s]["populations"]["bos_excluded"] for s in seeds}
    for key in ("pct", "raw", "unstrat_pct", "c_ws", "H"):
        name = {"pct": "AUROC_strat_pct", "raw": "AUROC_strat_raw",
                "unstrat_pct": "AUROC_unstrat_pct"}.get(key, key)  # fmt: skip
        led.stat(
            f"primary.{name}",
            [prim[s]["a2"][PRIMARY]["r"][key] for s in seeds],
            how="gated r_i x resample LOO, A1.3-excluded, per seed (spread over 3)",
        )
    for key in ("rho_Q", "rho_Q_rank", "rho_pool"):
        led.stat(
            f"secondary.{key}",
            [prim[s]["stats"]["gated"]["resample"][key] for s in seeds],
            how="A1 rho statistic, reported only (A2.1)",
        )
    if out["halt"]:
        print(f"HALT to Brendan (A2.8 row 5b): seeds {out['halt_seeds']}", flush=True)
    code = int(out["exit"])
    led.note(CLASS_KEY, out["class"], how="§8 (A2.8) first matching row (A1.4)")
    led.status(_STATUS[code])
    led.verdict(
        falsifier="spec §6 E0d / §3.2.1: 'If not, LOO is truth and r_i is a confound'",
        outcome=_OUTCOME[code],
        detail=f"class {out['class']} (all cells {out['class_all_cells']}, ungated "
        f"{out['class_ungated']}, zero {out['class_zero']}); labels {out['labels']}; "
        f"row5b {out['row5b']}; halt {out['halt_seeds']}",
    )
    led.command(argv_s, exit_code=code, note=note)
    path = led.write()
    write_claims(
        Path(path).parent / "claims.json", ledger_path=Path(path), run_id=a.run_id,
        out=out, seeds=seeds, t0=t0,
    )  # fmt: skip
    print(f"class {out['class']} -> exit {code}; labels {out['labels']}; {path}")
    return Exit(confirm_exit(path, code))  # A1.4: raises -> 3 if not written


def _parent_command() -> str | None:
    """The parent process's command line, or `None` if it cannot be read."""
    try:
        r = subprocess.run(
            ["ps", "-o", "command=", "-p", str(os.getppid())],
            capture_output=True, text=True, timeout=10,
        )  # fmt: skip
    except Exception:  # provenance only; never fails a run
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def launched_command(argv: list[str] | None) -> tuple[str, str]:
    """`(argv, note)` for the ledger's `commands[]` (RESULTS.md erratum, post-data,
    provenance only): the interpreter and this process's own argv, not a typed
    `uv run python ...`. The lane-slot wrapper, when the parent process is one, is
    named in the note -- not in the argv, whose entry point must stay run.py. An
    in-process `main(argv)` call says so instead of passing its host's argv off
    as the run's."""
    parts = []
    if argv is None:
        cmd = [sys.executable, *sys.argv]
    else:
        cmd = [sys.executable, EXPERIMENT, *argv]
        parts.append(f"main() called in-process with argv {list(argv)!r}; process "
                     f"argv {sys.argv!r}")  # fmt: skip
    parent = _parent_command()
    if parent and "orchestrator.slot" in parent:
        parts.insert(0, f"launched under the lane slot wrapper: {parent}")
    return shlex.join(cmd), "; ".join(parts)


def main(argv: list[str] | None = None) -> Exit:
    """A1.4, outermost: whatever raises before the measurement's own guard can
    ledger it (the ledger's construction, the manifest write, argument parsing)
    still exits 3 -- never 1, never 0."""
    try:
        return status(_main(argv))
    except BaseException as e:  # A1.4: nothing escapes as 1
        traceback.print_exc()
        return did_not_run(f"{type(e).__name__}: {e} (before the run could ledger it)")


def _main(argv: list[str] | None) -> Exit:
    p = ArgumentParser(prog="e0d", description=__doc__)
    p.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--rulings-dir", type=Path, default=DEFAULT_RULINGS_DIR)
    p.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    p.add_argument("--run-id", default=RUN_ID)
    a = p.parse_args(argv)

    import ledger as ledger_mod

    led = ledger_mod.Ledger(
        a.run_id,
        question="E0d (§3.2.1, §6 kill gate): does r_i rank memory slots the way "
        "leave-one-out delta next-sentence loss does: AUROC_strat,pct of gated r_i "
        "against y = 1[Delta_resample > tau_s], per seed (PREREG Amendment 2)?",
    )
    led.manifest(
        {
            "experiment": EXPERIMENT,
            "prereg": PREREG_PATH,
            "prereg_amendment": "Amendment 1 (84e21c5), erratum (268b947), "
            "Amendment 2 (8c63ecd), Amendment 3 (f5a8752)",
            "run_id": a.run_id,
            "seeds": list(SEEDS),
            "ckpt_root": str(a.ckpt_root),
            "ckpt_name": CKPT_NAME,
            "ckpt_sha256": {str(k): v for k, v in CKPT_SHA256.items()},
            "documents": list(D_E0D),
            "batch": BATCH,
            "rho_star_proposed_A1_secondary": RHO_STAR_PROPOSED,
            "auroc_star_source": "C8: every R-*-e0d-statistic* ruling (A2.13)",
            "tau_q": TAU_Q,
            "tau_resample": {str(k): v for k, v in TAU_RESAMPLE.items()},
            "tau_zero": {str(k): v for k, v in TAU_ZERO.items()},
            "tau_sensitivity": _jsonable(TAU_SENSITIVITY),
            "age_bins": list(AGE_BINS),
            "label_order": list(LABEL_ORDER),
            "loo_flat_a": LOO_FLAT_A,
            "loo_flat_q90_secondary": LOO_FLAT_Q90,
            "r_flat": R_FLAT,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "level": CI_LEVEL},
            "sensitivity_rho": list(SENSITIVITY_RHO),
            "b5_t_cap_max": B5_T_CAP_MAX,
            "class_exit": CLASS_EXIT,
            "resample_min_coverage": RESAMPLE_MIN_COVERAGE,
            "underfull_max": UNDERFULL_MAX,
            "live_tol": LIVE_TOL,
            "rulings_dir": str(a.rulings_dir),
            "t0_record": str(a.runs_dir / T0_RECORD),
            "loo": "rsr.metrics.loo.loo_delta_loss (resample primary, zero secondary)",
            "r_i": "rsr.retention.reward.retrieval_demand (gated primary), eval mode, "
            "own capture forward (A1.6)",
            "policy": "fifo (no RSRPolicy, no b, nu = 0, no shadow)",
            "device": "cpu",
            "expected": EXPECTED,
            "falsifier": "spec §6 E0d kill gate / §3.2.1: 'If not, LOO is truth and "
            "r_i is a confound.' Gated on AUROC_strat,pct (A2.12 declares the "
            "departure from the named Spearman rho).",
        }
    )
    led.run_meta(device="cpu")
    argv_s, argv_note = launched_command(argv)

    def refuse_run(key: str, value, status: str, reason: str) -> Exit:
        """§8 row 0: exit 3, with the reason (and traceback) in the ledger. If
        the ledger itself cannot be written, still exit 3."""
        try:
            led.note(key, value, how="PREREG §6 / §8 row 0 / A1.4")
            led.status(status)
            led.command(argv_s, exit_code=int(Exit.DID_NOT_RUN), note=argv_note)
            led.write()
        except BaseException as e2:  # A1.4: nothing escapes as 1
            print(f"ledger write failed: {type(e2).__name__}: {e2}", file=sys.stderr)
        return did_not_run(reason)

    try:
        return status(_measure_and_classify(a, led, argv_s, argv_note))
    except ControlFailed as e:
        return refuse_run(
            "control_failed",
            {"control": e.control, "msg": str(e), "traceback": traceback.format_exc()},
            "did_not_run",
            str(e),
        )
    except BaseException as e:  # A1.4: any exception is exit 3
        msg = f"{type(e).__name__}: {e}"
        return refuse_run(
            "measurement_raised",
            {
                "type": type(e).__name__,
                "msg": str(e),
                "traceback": traceback.format_exc(),
            },
            "crashed",
            msg,
        )


if __name__ == "__main__":
    run_main(main)
