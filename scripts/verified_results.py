"""Generate `docs/results/2026-09-30-verified.md` from the run ledgers, by git ref.

**What this is for.** The project's standing rule is that every reported number
traces to a ledger key (`scripts/render_scoreboard.py` states the history). This
file is the verified-results table of 2026-09-29/30: one row per claim, each value
read from a ledger (or a run's own committed output file) **at a pinned commit**,
never typed. The committed Markdown is this script's output and nothing else.

Every source is read with ``git show <sha>:<path>``, so the output does not depend
on the working tree, on which branches are merged, or on where a branch points
today. The SHAs are pinned below; a branch name is printed for the reader only.

Run::

    .venv/bin/python scripts/verified_results.py            # --check (default)
    .venv/bin/python scripts/verified_results.py --write    # regenerate the file
    .venv/bin/python scripts/verified_results.py --stdout   # print, write nothing

Exit codes (`rsr.exit_codes`, `docs/gates.md`): ``0`` the committed file equals the
output (``--check``) or was written; ``1`` it differs; ``3`` did not run -- a pinned
commit or a path is missing, or a key the table names is absent. 🔴 A missing
source is ``3``, never a row with a blank in it.

Two sources are not ledger keys and say so in their row:

* B2's gating Δs live in gitignored ``phaseB/*.pt`` files. They are parsed from the
  committed ``experiments/b2-psi-probe/RESULTS.md`` §2.1, which
  ``results_report.py`` generated from those files; the verifier re-derived the
  classification from them, not the Δs.
* E0h's R² is read from the committed ``runs/b2-psi-probe/phaseB/e0h-ckpt*.json``.
  E0h's thresholds are **unratified** (exit 2); no class is read.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator.verify import validate_record  # noqa: E402

from rsr.exit_codes import ArgumentParser, Exit, did_not_run, run_main  # noqa: E402

OUT = "docs/results/2026-09-30-verified.md"


class Missing(Exception):
    """A pinned source could not be read: the table did not run (exit 3)."""


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def _git_exe() -> str:
    """Homebrew git first: Apple's fails as an empty answer, not an error (CLAUDE.md)."""
    for cand in (
        os.environ.get("RSR_GIT"),
        "/opt/homebrew/bin/git",
        "/usr/local/bin/git",
    ):
        if cand and Path(cand).exists():
            return cand
    found = shutil.which("git")
    if found is None:
        raise Missing("no git binary found (set RSR_GIT)")
    return found


def git_show(sha: str, path: str) -> str:
    r = subprocess.run(
        [_git_exe(), "-C", str(_REPO), "show", f"{sha}:{path}"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise Missing(
            f"git show {sha[:12]}:{path} -> rc {r.returncode}: {r.stderr.strip()}"
        )
    return r.stdout


def git_ls(sha: str, prefix: str) -> list[str]:
    r = subprocess.run(
        [_git_exe(), "-C", str(_REPO), "ls-tree", "-r", "--name-only", sha, "--", prefix],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise Missing(
            f"git ls-tree {sha[:12]} {prefix} -> rc {r.returncode}: {r.stderr.strip()}"
        )
    return sorted(x for x in r.stdout.splitlines() if x)


# --------------------------------------------------------------------------- #
# sources -- pinned commits
# --------------------------------------------------------------------------- #

NIGHT = ("night/2026-09-30", "898628a2a8333ff21af1d0a51cbabf6a06800359")


@dataclass(frozen=True)
class Source:
    name: str
    run_id: str
    branch: str
    sha: str
    run_branch: str
    """The run's own branch; its review records live under docs/reviews/<slug>/."""
    results: str
    note: str = ""


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source(
            "E0d",
            "e0d",
            "run/e0d",
            "0e51139d8e0fca1fb4272b0a0aec92885cae7e94",
            "run/e0d",
            "experiments/e0d/RESULTS.md",
            "Not merged into night/2026-09-30 at the time of writing; read from its run "
            "branch.",
        ),
        Source(
            "B2",
            "b2-psi-probe",
            *NIGHT,
            "run/b2-psi-probe",
            "experiments/b2-psi-probe/RESULTS.md",
        ),
        Source(
            "Newcomer bake-off",
            "newcomer-bakeoff",
            *NIGHT,
            "run/newcomer-bakeoff",
            "experiments/newcomer-bakeoff/RESULTS.md",
        ),
        Source(
            "B1",
            "b1-beta-inertness",
            *NIGHT,
            "run/b1-beta-inertness",
            "experiments/b1-beta-inertness/RESULTS.md",
            "A further, out-of-repo verification exists and is cited, not read, by this "
            "generator: `~/Documents/RSR-2026-09-29-day/reports/B1-replay.md` "
            "(rsr-verifier, "
            "2026-09-30): an independent training replay of seed 1, one cell per class, "
            "compared bit-exact against the ledger's `cmp.seed1.*` at every snapshot. "
            "phi_clip and beta 0.1 were not replayed.",
        ),
        Source(
            "B5",
            "b5-convergence",
            "run/b5-convergence",
            "deaeaa6f54ba8c7090e0095cc00752a13cb6332b",
            "run/b5-convergence",
            "experiments/b5-convergence/RESULTS.md",
            "Not merged into night/2026-09-30 at the time of writing; read from its run "
            "branch at deaeaa6, which adds only the review record to the verified "
            "7677bb3. "
            "The ledger's `verdict` field is null; the classification is the "
            "`classification_reported_as` row.",
        ),
        Source(
            "B0",
            "b0-ceilings",
            *NIGHT,
            "run/b0-ceilings",
            "experiments/b0-ceilings/RESULTS.md",
            "Merged at f7a6b10 (night/2026-09-27); read here from night/2026-09-30.",
        ),
    )
}


# --------------------------------------------------------------------------- #
# refs -- where each value comes from
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ref:
    """One value's source. `kind`:

    * ``row`` -- a ledger row `key`, then `sub` into its value (or its stats);
    * ``top`` -- a top-level ledger field, dotted in `key` (e.g. ``verdict.outcome``);
    * ``json`` -- a key of a committed JSON file at `path`;
    * ``md`` -- a line of a committed Markdown file that contains `key` verbatim.
    """

    source: str
    kind: str
    key: str
    path: str = ""
    sub: tuple = ()

    @property
    def file(self) -> str:
        return self.path or f"runs/{SOURCES[self.source].run_id}/ledger.json"

    def label(self) -> str:
        if self.kind == "md":
            return (
                f"`{self.file}` §{self.sub[0].split()[1]} table line (not a ledger key)"
            )
        tail = "".join(f"[{s!r}]" if isinstance(s, str) else f"[{s}]" for s in self.sub)
        where = "" if self.path == "" else f" in `{self.path}`"
        pre = "" if self.kind != "top" else "(top-level) "
        return f"{pre}`{self.key}{tail}`{where}"


_CACHE: dict[tuple[str, str], str] = {}


def _text(sha: str, path: str) -> str:
    if (sha, path) not in _CACHE:
        _CACHE[(sha, path)] = git_show(sha, path)
    return _CACHE[(sha, path)]


def _json(sha: str, path: str) -> Any:
    return json.loads(_text(sha, path))


def _dig(obj: Any, parts, what: str) -> Any:
    for p in parts:
        try:
            obj = obj[p]
        except (KeyError, IndexError, TypeError):
            raise Missing(f"{what}: no {p!r}") from None
    return obj


def _section(text: str, heading: str, what: str) -> list[str]:
    """The lines under the one heading starting with `heading`, to the next heading."""
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith(heading)]
    if len(starts) != 1:
        raise Missing(
            f"{what}: {len(starts)} headings start with {heading!r}, need exactly 1"
        )
    out = []
    for ln in lines[starts[0] + 1 :]:
        if ln.startswith("#"):
            break
        out.append(ln)
    return out


def resolve(ref: Ref) -> Any:
    """The value `ref` names, read at its source's pinned commit. Raises `Missing`."""
    src = SOURCES[ref.source]
    what = f"{src.sha[:7]}:{ref.file}"
    if ref.kind == "md":
        lines = _section(_text(src.sha, ref.file), ref.sub[0], what)
        hits = [ln for ln in lines if ref.key in ln]
        if len(hits) != 1:
            raise Missing(
                f"{what} {ref.sub[0]!r}: {len(hits)} lines contain {ref.key!r}, need 1"
            )
        return hits[0]
    doc = _json(src.sha, ref.file)
    if ref.kind == "json":
        return _dig(doc, (ref.key, *ref.sub), what)
    if ref.kind == "top":
        return _dig(doc, (*ref.key.split("."), *ref.sub), what)
    if ref.kind != "row":
        raise ValueError(ref.kind)
    rows = [r for r in doc.get("rows", []) if r.get("key") == ref.key]
    if len(rows) != 1:
        raise Missing(f"{what}: {len(rows)} rows keyed {ref.key!r}, need exactly 1")
    row = rows[0]
    base = row.get("value", row)
    val = _dig(base, ref.sub, f"{what} {ref.key}")
    if val is None:
        raise Missing(f"{what}: {ref.key} {ref.sub} is null")
    return val


# --------------------------------------------------------------------------- #
# formatting -- fixed precision, so two runs are byte-identical
# --------------------------------------------------------------------------- #


def f4(x: float) -> str:
    return f"{x:.4f}"


def s4(x: float) -> str:
    return f"{x:+.4f}"


def ci(lo: float, hi: float, fmt=f4) -> str:
    return f"[{fmt(lo)}, {fmt(hi)}]"


def seeds(vals, fmt=f4) -> str:
    return " / ".join(fmt(v) if isinstance(v, float) else str(v) for v in vals)


def labels(xs) -> str:
    return " / ".join(str(x) for x in xs)


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    source: str
    claim: str
    refs: list[Ref]
    render: Any
    """Called with the resolved values, in `refs` order; returns the values cell."""
    values: str = field(default="")
    outside: str = ""
    """Verification outside the repo, cited, not read (printed with re-executed keys)."""


#: B2's verifier re-derived the gating contrasts from the phaseB units; its report is
#: outside the repo, so it is cited, like the B1 replay, and never read here.
B2_REDERIVED = (
    "out of repo: independently re-derived from the phaseB units by the B2 verifier, "
    "2026-09-30 (`~/Documents/RSR-2026-09-29-day/reports/B2-verification.md` §2; "
    "cited, not read)"
)


def _r(source: str, key: str, *sub, kind: str = "row", path: str = "") -> Ref:
    return Ref(source, kind, key, path, tuple(sub))


def _e0d_rows() -> list[Row]:
    S = "E0d"
    pct = [
        _r(
            S,
            f"seed{s}.analysis",
            "populations",
            "bos_excluded",
            "a2",
            "gated_resample",
            "r",
            "pct_ci",
        )
        for s in range(3)
    ]
    return [
        Row(
            S,
            "E0d class (A2.8; gating; exit 0) and per-seed labels",
            [
                _r(S, "e0d.class"),
                _r(S, "e0d.labels"),
                _r(S, "verdict", "outcome", kind="top"),
            ],
            lambda c, lab, out: f"**{c}**; labels {labels(lab)}; ledger verdict `{out}`",
        ),
        Row(
            S,
            "Primary statistic AUROC_strat,pct (gated r_i x resample LOO, A1.3 "
            "population), "
            "per seed [95% CI]; gate: CI lower >= A*",
            [
                _r(S, "primary.AUROC_strat_pct", "samples"),
                *pct,
                _r(S, "c8_authority", "auroc_star"),
                _r(S, "primary.AUROC_strat_pct", "sd"),
            ],
            lambda v, c0, c1, c2, a, sd: (
                " / ".join(
                    f"{f4(x)} {ci(*c)}" for x, c in zip(v, (c0, c1, c2), strict=True)
                )
                + f"; A* = {a}; sd over seeds {f4(sd)}"
            ),
        ),
        Row(
            S,
            "E0d class under the pre-registered sensitivities "
            "(all cells, ungated r_i, zero knockout, tau q0.99 / q0.999)",
            [
                _r(S, "e0d.class_all_cells"),
                _r(S, "e0d.class_ungated"),
                _r(S, "e0d.class_zero"),
                _r(S, "e0d.tau_sensitivity", "0.99", "class"),
                _r(S, "e0d.tau_sensitivity", "0.999", "class"),
            ],
            lambda *c: " / ".join(c),
        ),
        Row(
            S,
            "A1 Spearman reading (reported, never gates): class_A1 at rho* = 0.5, "
            "rho_Q per seed",
            [
                _r(S, "e0d.class_A1"),
                _r(S, "e0d.labels_A1"),
                _r(S, "secondary.rho_Q", "samples"),
            ],
            lambda c, lab, rho: f"{c} (labels {labels(lab)}); rho_Q {seeds(rho)}",
        ),
    ]


#: The RESULTS section holding B2's gating contrasts (§9.5, ckpt3000, gamma 0.9, Tier 1).
#: The heading is matched verbatim, so it keeps the file's own gamma character.
B2_DELTA_SECTION = (
    "### 2.1 Gating contrasts (§9.5) and Q2 (§9.7), ckpt3000 γ = 0.9 Tier 1"  # noqa: RUF001
)


def _b2_delta_refs(arm: str) -> list[Ref]:
    return [
        Ref(
            "B2",
            "md",
            f"| {s} | {arm} |",
            "experiments/b2-psi-probe/RESULTS.md",
            (B2_DELTA_SECTION,),
        )
        for s in range(3)
    ]


def _b2_delta(*lines: str) -> str:
    out = []
    for s, ln in enumerate(lines):
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        # cells: seed, arm, ref, δ, Δ all [CI], Δ 2..M, Δ >M, vs random, outcome
        seed, ref, delta, d_all, outcome = s, cells[2], cells[3], cells[4], cells[8]
        out.append(f"s{seed}: {d_all} vs {ref}, δ {delta}, {outcome.strip('*')}")
    return "; ".join(out)


def _b2_rows() -> list[Row]:
    S = "B2"
    e0h = "runs/b2-psi-probe/phaseB/e0h-ckpt{}.json"
    rows = [
        Row(
            S,
            "B2 classification (§9.6; ckpt3000, gamma 0.9, Tier 1; gating)",
            [
                _r(S, "classification", k)
                for k in ("row", "class", "psiU", "psiC", "underpowered")
            ],
            lambda row, c, u, cc, up: (
                f"row {row}, **{c}**; psiU {labels(u)}; "
                f"psiC {labels(cc)}; underpowered {up}"
            ),
        ),
        Row(
            S,
            "B2 gating Δ, psiU (all-query accuracy, arm - ref, paired 95% CI)",
            _b2_delta_refs("ψ̂-U"),
            _b2_delta,
            outside=B2_REDERIVED,
        ),
        Row(
            S,
            "B2 gating Δ, psiC (all-query accuracy, arm - ref, paired 95% CI)",
            _b2_delta_refs("ψ̂-C"),
            _b2_delta,
            outside=B2_REDERIVED,
        ),
        Row(
            S,
            "B2 non-gating readings: companion (U+, C+), ckpt2500, moving",
            [
                _r(S, "classification.companion", "class"),
                _r(S, "classification.ckpt2500", "class"),
                _r(S, "classification.moving"),
            ],
            lambda a, b, m: f"companion {a}; ckpt2500 {b}; moving {m}",
        ),
    ]
    for c in (3000, 2500):
        p = e0h.format(c)
        rows.append(
            Row(
                S,
                f"E0h (falsifier 3c) within-step R² of psiU(gamma=0), ckpt{c} — "
                "**UNRATIFIED**",
                [Ref(S, "json", "e0h.r2_headline", p, (s,)) for s in ("0", "1", "2")]
                + [
                    Ref(S, "json", k, p)
                    for k in ("rc", "e0h.ratified", "e0h.class_unratified_reading")
                ],
                lambda a, b, cc, rc, rat, rd: (
                    f"{seeds([a, b, cc])}; rc {rc}; ratified {rat}; "
                    f"unratified reading {rd} (no class is read)"
                ),
            )
        )
    return rows


def _bakeoff_rows() -> list[Row]:
    S = "Newcomer bake-off"
    rows = [
        Row(
            S,
            "DQ1: does hard grace buy anything over psiC?",
            [
                _r(S, "decision.DQ1", "class"),
                *(_r(S, "decision.DQ1", "labels", g) for g in "124"),
            ],
            lambda c, g1, g2, g4: (
                f"**{c}**; g=1 {labels(g1)}; g=2 {labels(g2)}; g=4 {labels(g4)}"
            ),
        ),
        Row(
            S,
            "DQ2: does psiR1 land near psiC or psiU? (π: 0 = at U, 1 = at C)",
            [
                _r(S, "decision.DQ2", "class"),
                _r(S, "decision.DQ2", "per_seed"),
                *(_r(S, f"seed{s}.pi") for s in range(3)),
            ],
            lambda c, ps, *pi: (
                f"**{c}**; {labels(ps)}; π "
                + " / ".join(f"{f4(p['point'])} {ci(p['lo'], p['hi'])}" for p in pi)
            ),
        ),
        Row(
            S,
            "DQ3: does the harm travel with the counterfactual target?",
            [
                _r(S, "decision.DQ3", "class"),
                _r(S, "decision.DQ3", "harm_seeds"),
                _r(S, "decision.DQ3", "per_seed", "0", "label"),
                _r(S, "decision.DQ3", "per_seed", "1", "label"),
            ],
            lambda c, hs, a, b: f"**{c}**; harm seeds {hs}; s0 {a}, s1 {b}",
        ),
        Row(
            S,
            "B2 replication on fresh documents (non-binding)",
            [_r(S, "decision.replication", k) for k in ("row", "class", "psiU", "psiC")],
            lambda row, c, u, cc: f"row {row}, {c}; psiU {labels(u)}; psiC {labels(cc)}",
        ),
    ]
    arms = list(resolve(_r(S, "seed0.outcome")).keys())
    for arm in arms:
        rows.append(
            Row(
                S,
                f"Per-arm §9.5 outcome vs its ref, {arm} (Δ all-query [95% CI])",
                [_r(S, f"seed{s}.outcome", arm) for s in range(3)],
                lambda *o: "; ".join(
                    f"s{s}: {x['label']} {s4(x['point'])} {ci(x['lo'], x['hi'], s4)} vs "
                    f"{x['ref']}"
                    for s, x in enumerate(o)
                ),
            )
        )
    return rows


def _b1_rows() -> list[Row]:
    S = "B1"
    rows = [
        Row(
            S,
            "B1 verdict on ADR-0009 L7's premise (beta inert under AdamW + isolation)",
            [_r(S, "verdict", "outcome", kind="top"), _r(S, "N_ok")],
            lambda o, n: f"`{o}`; control N (beta 1 re-run bit-identical) {n}",
        )
    ]
    for arm in ("decoupled_wd", "phi_clip", "joint_clip", "coupled_l2", "P"):
        for eps in ("1e-08", "1e-12"):
            k = f"class.{arm}.eps{eps}"
            rows.append(
                Row(
                    S,
                    f"B1 class, {arm} at eps {eps} (step 3000, beta in {{0.01, 0.1}} x 3 "
                    "seeds)",
                    [
                        _r(S, k, "class"),
                        _r(S, k, "min_agreement"),
                        _r(S, k, "max_rel"),
                        *(
                            _r(
                                S,
                                f"cmp.seed{s}.{arm}.eps{eps}.beta0.01",
                                "3000",
                                "agreement",
                            )
                            for s in range(3)
                        ),
                    ],
                    lambda c, a, r, *pa: (
                        f"**{c}**; min argmin agreement {f4(a)}, max rel "
                        f"max|Δφ| {r:.3g}; agreement at beta 0.01 per seed "
                        f"{seeds(pa)}"
                    ),
                )
            )
    return rows


def _b5_rows() -> list[Row]:
    S = "B5"
    return [
        Row(
            S,
            "B5 classification (plateau rule over E in 7000..9000, every seed)",
            [
                _r(S, "classification_reported_as"),
                _r(S, "reached_cap"),
                _r(S, "steps_done", kind="top"),
                _r(S, "seeds_actually_run", kind="top"),
            ],
            lambda c, cap, st, sd: (
                f"**{c}**; reached cap {cap}; steps_done {st}; seeds {sd}"
            ),
        ),
        Row(
            S,
            "B5 plateau rule at E = 9000: relative slope, % per 1000 steps [95% CI], "
            "holds",
            [_r(S, "rule_trace", "9000", f"seed{s}") for s in range(3)],
            lambda *t: "; ".join(
                f"s{s}: {x['rel_slope_pct']:+.3f} "
                f"{ci(*x['rel_ci_pct'], fmt=lambda v: f'{v:+.3f}')} "
                f"holds {x['holds']}"
                for s, x in enumerate(t)
            ),
        ),
        *(
            Row(
                S,
                f"B5 retrieval quantity R at ckpt{c} (R holds on every seed)",
                [_r(S, f"B.ckpt{c}.R_quantity", "samples"), _r(S, f"B.ckpt{c}.R_holds")],
                lambda q, h: f"{seeds(q)}; R_holds {h}",
            )
            for c in (4000, 9000)
        ),
    ]


def _b0_rows() -> list[Row]:
    S = "B0"
    k = "B.ckpt3000.g09.cap.{}.{}"
    return [
        Row(
            S,
            "B0 kind-oracle share of the fact/filler gain (cap), ckpt3000 gamma 0.9 "
            "[paired 95% CI] — descriptive, no gate",
            [_r(S, k.format("ko", f), "samples") for f in ("point", "lo", "hi")],
            lambda p, lo, hi: " / ".join(
                f"{s4(a)} {ci(b, c, s4)}" for a, b, c in zip(p, lo, hi, strict=True)
            ),
        ),
        Row(
            S,
            "B0 age-only cap, ckpt3000 gamma 0.9 [paired 95% CI]; controls failed",
            [
                *(_r(S, k.format("age", f), "samples") for f in ("point", "lo", "hi")),
                _r(S, "controls_failed"),
            ],
            lambda p, lo, hi, cf: (
                " / ".join(
                    f"{s4(a)} {ci(b, c, s4)}" for a, b, c in zip(p, lo, hi, strict=True)
                )
                + f"; controls_failed {cf}"
            ),
        ),
    ]


def build_rows() -> list[Row]:
    rows = [
        *_e0d_rows(),
        *_b2_rows(),
        *_bakeoff_rows(),
        *_b1_rows(),
        *_b5_rows(),
        *_b0_rows(),
    ]
    for row in rows:
        row.values = row.render(*(resolve(r) for r in row.refs))
    return rows


# --------------------------------------------------------------------------- #
# verification and review records
# --------------------------------------------------------------------------- #


@dataclass
class Verification:
    path: str
    status: str
    n_match: int
    n_claims: int
    problems: list[str]
    claims: list[dict]
    commands: list[str]
    """Each drawn claim's command in verification.json and, when the claim text
    matches, the researcher's command in claims.json (the verifier sometimes
    records prose there)."""

    @property
    def verdict(self) -> str:
        if self.status == "ok" and not self.problems:
            return f"CONFIRMED ({self.n_match}/{self.n_claims} drawn claims match)"
        why = "; ".join(self.problems) or "mismatch"
        return f"NOT CONFIRMED (status {self.status!r}; {why})"


def verification(name: str) -> Verification:
    src = SOURCES[name]
    path = f"runs/{src.run_id}/verification.json"
    rec = _json(src.sha, path)
    problems = validate_record(rec, src.run_id)
    claims = rec.get("claims", [])
    cmds = [c.get("command", "") for c in claims]
    try:
        research = {
            c["claim"]: c["command"]
            for c in _json(src.sha, f"runs/{src.run_id}/claims.json")
        }
    except Missing:
        research = {}
    cmds += [research[c["claim"]] for c in claims if c.get("claim") in research]
    return Verification(
        path,
        rec.get("status"),
        sum(c.get("match") is True for c in claims),
        len(claims),
        problems,
        claims,
        cmds,
    )


def drawn(ref: Ref, ver: Verification) -> bool:
    """Whether a drawn claim's command reads this key of this file.

    A quoted full key counts; so does the quoted prefix `'a.b.'` plus the quoted
    last segment (B0's claim builds `'B.ckpt3000.g09.cap.ko.'+f` over `('lo','hi')`).
    A Markdown line never counts: no verifier claim re-derives a RESULTS table line.
    """
    if ref.kind == "md":
        return False
    for cmd in ver.commands:
        if ref.file not in cmd:
            continue
        key = ref.key
        if f"'{key}'" in cmd or f'"{key}"' in cmd:
            return True
        if "." in key:
            pre, last = key.rsplit(".", 1)
            if f"'{pre}.'" in cmd and f"'{last}'" in cmd:
                return True
    return False


@dataclass(frozen=True)
class Review:
    path: str
    sha: str
    verdict: str
    reviewed_head: str


def reviews(name: str) -> list[Review]:
    """Review records for the run's branch, in the tree at the pinned commit."""
    src = SOURCES[name]
    slug = src.run_branch.replace("/", "-")
    out = []
    for path in git_ls(src.sha, f"docs/reviews/{slug}"):
        if not path.endswith(".md"):
            continue
        head = _text(src.sha, path).split("\n---", 1)[0]
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", head, flags=re.M))
        out.append(
            Review(
                path, src.sha, meta.get("verdict", "?"), meta.get("reviewed_head", "?")
            )
        )
    return out


# --------------------------------------------------------------------------- #
# what this does NOT show -- quoted verbatim from each run's own files
# --------------------------------------------------------------------------- #

#: (source, path at the source's commit, a substring that must occur exactly once).
#: The generator refuses (exit 3) if a quote is not in the file: a scope statement
#: is quoted, never paraphrased.
SCOPE: tuple[tuple[str, str, str], ...] = (
    (
        "E0d",
        "experiments/e0d/RESULTS.md",
        "`r_i` and leave-one-out agree **at the moment of retrieval**",
    ),
    (
        "E0d",
        "experiments/e0d/RESULTS.md",
        "**E0d does NOT show that `r_i` predicts later demand.**",
    ),
    (
        "B2",
        "experiments/b2-psi-probe/PREREG.md",
        "The fit is offline, full-batch, closed form, and drawn from the FIFO world.",
    ),
    (
        "B2",
        "experiments/b2-psi-probe/PREREG.md",
        "An offline **WIN** is **necessary, not sufficient**, for online MC at μP rates",
    ),
    (
        "B2",
        "experiments/b2-psi-probe/RESULTS.md",
        "**The mechanism is an inference from co-occurring statistics; nothing here "
        "isolates it.**",
    ),
    (
        "B2",
        "experiments/b2-psi-probe/PREREG.md",
        "**🔴 This is an offline-ridge proxy, declared.**",
    ),
    ("B2", "experiments/b2-psi-probe/PREREG.md", "It does not measure the trained head."),
    (
        "Newcomer bake-off",
        "experiments/newcomer-bakeoff/RESULTS.md",
        "**It says nothing about on-policy behaviour after `T_warm`.**",
    ),
    (
        "Newcomer bake-off",
        "experiments/newcomer-bakeoff/RESULTS.md",
        "**Whether the grace rule (arm c) is acceptable under the age prohibition is "
        "OWNER-ONLY, "
        "whatever the numbers below say.**",
    ),
    (
        "B1",
        "experiments/b1-beta-inertness/PREREG.md",
        "This is **open-loop** on the FIFO trajectory",
    ),
    (
        "B1",
        "experiments/b1-beta-inertness/PREREG.md",
        "It does not measure closed-loop policy divergence, `T_warm`, `b`, or the online "
        "(non-repeating)",
    ),
    (
        "B1",
        "experiments/b1-beta-inertness/RESULTS.md",
        "It is not a decision on L7 or L8.",
    ),
    (
        "B5",
        "experiments/b5-convergence/PREREG.md",
        "A NO_PLATEAU bounds only this budget, learning rate and thread count.",
    ),
    (
        "B5",
        "experiments/b5-convergence/PREREG.md",
        "Nothing about RSR, eviction, ψ̂ or Kintsch & van Dijk: FIFO only, no retention "
        "loss.",
    ),
    (
        "B5",
        "experiments/b5-convergence/RESULTS.md",
        "**ckpt3000 stays primary for everything; making another checkpoint the "
        "substrate is D2, "
        "owner-only**",
    ),
    ("B0", "experiments/b0-ceilings/RESULTS.md", "**Descriptive only. No gate.**"),
    (
        "B0",
        "experiments/b0-ceilings/RESULTS.md",
        "**Metric:** model-free residency (`simulate`), **not** model-read accuracy.",
    ),
)


def scope_line(source: str, path: str, quote: str) -> str:
    text = _text(SOURCES[source].sha, path)
    n = text.count(quote)
    if n != 1:
        raise Missing(
            f"{SOURCES[source].sha[:7]}:{path}: quote occurs {n} times: {quote[:60]!r}"
        )
    return quote


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    rows = build_rows()
    vers = {n: verification(n) for n in SOURCES}
    revs = {n: reviews(n) for n in SOURCES}
    L = [
        "# Verified results, 2026-09-29/30",
        "",
        "<!-- GENERATED by scripts/verified_results.py from ledgers read with `git show "
        "<sha>:<path>`. Do not edit: regenerate with `.venv/bin/python "
        "scripts/verified_results.py --write`. tests/test_verified_results.py fails on "
        "any difference. -->",
        "",
        "Every value below is read from a ledger key (or, where the row says so, a run's "
        "own "
        "committed output file) at a pinned commit. None is typed. The verifier verdict "
        "is "
        "computed from `runs/<id>/verification.json` with "
        "`orchestrator.verify.validate_record`; "
        "*re-executed keys* lists the keys of that row that a drawn verifier claim read. "
        "A key "
        "not listed there was produced by a verified run but was not itself re-executed.",
        "",
        "Written by a model (Claude Opus 5.5). **Not written by Brendan.** Nothing here "
        "is a "
        "decision; the owner decisions these results feed are in "
        "`docs/RESEARCH-CONTEXT.md` §12.",
        "",
        "## Sources",
        "",
        "| run | run id | read at | verifier | review record(s) |",
        "|---|---|---|---|---|",
    ]
    for n, src in SOURCES.items():
        v = vers[n]
        rv = (
            "; ".join(
                f"`{r.path}` ({r.verdict}, reviewed head `{r.reviewed_head[:7]}`)"
                for r in revs[n]
            )
            or f"none in the tree at `{src.sha[:7]}`"
        )
        L.append(
            f"| {n} | `{src.run_id}` | `{src.branch}@{src.sha[:7]}` | "
            f"{v.verdict}; `{v.path}` | {_cell(rv)} |"
        )
    L += [
        "",
        "## The table",
        "",
        "| # | claim | value(s) per seed | source key(s) | run branch@sha | verifier | "
        "re-executed keys | review record |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(rows, 1):
        src = SOURCES[row.source]
        v = vers[row.source]
        keys = sorted({r.label() for r in row.refs})
        hit = sorted({r.key for r in row.refs if drawn(r, v)})
        rv = ", ".join(f"`{r.path}`" for r in revs[row.source]) or "none"
        L.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in (
                    str(i),
                    row.claim,
                    row.values,
                    "; ".join(keys),
                    f"`{src.branch}@{src.sha[:7]}`",
                    f"{v.verdict.split(' (')[0]} `{v.path}`",
                    "; ".join(
                        x for x in (", ".join(f"`{h}`" for h in hit), row.outside) if x
                    )
                    or "—",
                    rv,
                )
            )
            + " |"
        )
    L += [
        "",
        "E0d and B0 have no T5(b) review record: E0d's merge is blocked on the owner, "
        "and B0 was merged before T5(b) existed.",
        "",
        "## Verifier claims, as recorded",
        "",
        "Each run's drawn claims, from its `verification.json`. Where the recorded "
        "observation says more than the expected value, it is quoted verbatim.",
        "",
    ]
    for n, v in vers.items():
        L.append(f"**{n}** (`{v.path}`, {v.verdict}):")
        L.append("")
        for c in v.claims:
            obs = str(c.get("observed", ""))
            extra = "" if obs == str(c.get("expected")) else f" Observed: {obs}"
            L.append(f"- match `{c.get('match')}`: {c.get('claim')}.{extra}")
        if SOURCES[n].note:
            L.append(f"- *Note:* {SOURCES[n].note}")
        L.append("")
    L += [
        "## What this does NOT show",
        "",
        "Quoted verbatim from each run's own PREREG or RESULTS at the pinned commit; the "
        "generator refuses if a quote is not in the file.",
        "",
    ]
    for source, path, quote in SCOPE:
        L.append(f"- **{source}** (`{path}`): {scope_line(source, path, quote)}")
    L += [
        "",
        "Not shown by any row above: anything about a ψ̂ trained in the loop (by `L_MC` "
        "in "
        "`train()`, on-policy). None has been trained. B2 and the bake-off evaluate "
        "closed-form "
        "ridge fits as fixed eviction rules; B1 trains φ offline, open-loop, on fixed "
        "FIFO-world "
        "captures, to test β only.",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> Exit:
    p = ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--check", action="store_true", help="compare with the committed file (default)"
    )
    g.add_argument("--write", action="store_true", help=f"write {OUT}")
    g.add_argument("--stdout", action="store_true", help="print the table, write nothing")
    args = p.parse_args(argv)
    try:
        text = render()
    except Missing as e:
        return did_not_run(str(e))
    if args.stdout:
        sys.stdout.write(text)
        return Exit.OK
    out = _REPO / OUT
    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {OUT} ({len(text.splitlines())} lines)")
        return Exit.OK
    if not out.is_file():
        return did_not_run(f"{OUT} does not exist; run with --write")
    if out.read_text() != text:
        print(
            f"FAIL: {OUT} differs from the generator's output; regenerate with --write",
            file=sys.stderr,
        )
        return Exit.FAIL
    print(f"OK: {OUT} equals the generator's output")
    return Exit.OK


if __name__ == "__main__":
    run_main(main)
