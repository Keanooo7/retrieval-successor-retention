"""Build ``docs/claims-register.md`` (PLAN-2026-10-19 phase 0.2).

    .venv/bin/python docs/claims-register/build.py rederive [--only EXP ...]
    .venv/bin/python docs/claims-register/build.py render

``rederive`` runs every row's checks (each a command runnable from a clean checkout;
compute is lane-slot wrapped inside the command) and writes their literal output to
``docs/claims-register/rederive/<experiment>.json``. ``render`` reads the rows (every
value from a committed ledger or raw record, via ``extract/*.py``) and those outputs,
and writes the register. No number in the register is typed.

Exit codes: 0 ok; 1 a check was not equal (rederive) -- the mismatch is recorded, never
smoothed; 3 a check could not run (rc != 0 or missing output) or a rederive file is
missing/stale for render.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

GIT = "/opt/homebrew/bin/git"
OUT_MD = C.ROOT / "docs" / "claims-register.md"
RED = C.HERE / "rederive"

#: Register order: run 1, run 2 and the post-FAILS diagnostics first, then the
#: ceilings, E0d, Expire-Span and the substrate runs.
ORDER = [
    "rsr-run2",
    "rsr-run1",
    "run2-vs-random",
    "offline-psi",
    "offline-psi-converge",
    "psi-ls-capacity",
    "b0-ceilings",
    "cf-ceiling",
    "gamma-ceiling",
    "e0d",
    "expire-span-referendum",
    "decisive-shuffle",
    "fresh-stream",
    "fresh-escape",
]


def git(*a: str) -> str:
    return subprocess.run(
        [GIT, "-C", str(C.ROOT), *a], capture_output=True, text=True, check=True
    ).stdout.strip()


def modules() -> dict[str, object]:
    out = {}
    for p in sorted((C.HERE / "extract").glob("*.py")):
        if p.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_cr_x_{p.stem}", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out[m.EXPERIMENT] = m
    return out


def ordered(mods: dict) -> list[str]:
    return [e for e in ORDER if e in mods] + sorted(e for e in mods if e not in ORDER)


# --------------------------------------------------------------------------- #
# rederive
# --------------------------------------------------------------------------- #


def run_cmd(cmd: str) -> dict:
    p = subprocess.run(cmd, shell=True, cwd=C.ROOT, capture_output=True, text=True)
    lines = [x for x in p.stdout.splitlines() if x.strip()]
    return {
        "rc": p.returncode,
        "last": lines[-1] if lines else None,
        "stderr_tail": p.stderr[-600:] if p.returncode else "",
    }


def pick(last: str | None, key: str | None):
    if last is None:
        return None, "no output"
    if key is None:
        return last.strip(), None
    try:
        obj = json.loads(last)
    except json.JSONDecodeError as e:
        return None, f"not JSON: {e}"
    if key not in obj:
        return None, f"key {key!r} absent from output"
    return C.fmt_exact(obj[key]), None


def rederive(only: list[str] | None) -> int:
    mods = modules()
    worst = 0
    head, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    RED.mkdir(exist_ok=True)
    for exp in ordered(mods):
        if only and exp not in only:
            continue
        rows = mods[exp].rows()
        cache: dict[str, dict] = {}
        out = []
        for r in rows:
            for ch in r.checks:
                if ch.cmd not in cache:
                    print(f"[{exp}] {ch.cmd}", file=sys.stderr, flush=True)
                    cache[ch.cmd] = run_cmd(ch.cmd)
                res = cache[ch.cmd]
                got, err = pick(res["last"], ch.pick) if res["rc"] == 0 else (None, None)
                equal = None if (res["rc"] != 0 or err) else (got == ch.expected)
                out.append(
                    {
                        "row": r.id,
                        "label": ch.label,
                        "kind": ch.kind,
                        "cmd": ch.cmd,
                        "pick": ch.pick,
                        "rc": res["rc"],
                        "got": got,
                        "expected": ch.expected,
                        "equal": equal,
                        "error": err or (res["stderr_tail"] or None),
                    }
                )
                if equal is None:
                    worst = max(worst, 3)
                elif not equal:
                    worst = max(worst, 1) if worst != 3 else 3
        doc = {
            "experiment": exp,
            "tree_head": head,
            "tree_dirty_at_run": dirty,
            "finished_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_commands": len(cache),
            "n_checks": len(out),
            "n_equal": sum(1 for x in out if x["equal"] is True),
            "n_not_equal": sum(1 for x in out if x["equal"] is False),
            "n_did_not_run": sum(1 for x in out if x["equal"] is None),
            "checks": out,
        }
        (RED / f"{exp}.json").write_text(json.dumps(doc, indent=1) + "\n")
        print(
            f"[{exp}] checks={doc['n_checks']} equal={doc['n_equal']} "
            f"not_equal={doc['n_not_equal']} did_not_run={doc['n_did_not_run']}",
            file=sys.stderr,
        )
    return worst


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

ABBR = [(C.SLOT, "$SLOT "), (C.LEDGER_GET, "$LG")]


def short(cmd: str) -> str:
    for a, b in ABBR:
        cmd = cmd.replace(a, b)
    return cmd


def cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def per_seed(seeds, xs, f) -> str:
    return "; ".join(f"s{s} {f(x)}" for s, x in zip(seeds, xs, strict=True))


def ci_str(ci) -> str:
    return "n/a" if ci is None else f"[{C.fmt(ci[0])}, {C.fmt(ci[1])}]"


def status(r: C.Row, res: list[dict]) -> tuple[str, str]:
    """(category, cell text). category in yes/no/not-rederived/did-not-run."""
    if not r.checks:
        return "not-rederived", r.rederive_note or "not re-derived"
    if len(res) != len(r.checks):
        return "did-not-run", "DID NOT RUN (no rederive output for this row)"
    bad = [x for x in res if x["equal"] is False]
    dnr = [x for x in res if x["equal"] is None]
    kinds = "+".join(sorted({x["kind"] for x in res}))
    if bad:
        txt = "; ".join(
            f"{x['label']}: re-derived `{x['got']}` vs ledger `{x['expected']}`" for x in bad
        )
        return "no", f"**NO** ({kinds}): {txt}"
    if dnr:
        txt = "; ".join(f"{x['label']}: rc {x['rc']} {x['error'] or ''}" for x in dnr)
        return "did-not-run", f"**DID NOT RUN** ({kinds}): {cell(txt)[:300]}"
    return "yes", f"yes ({kinds}, {len(res)}/{len(res)} checks)"


def render() -> int:
    mods = modules()
    exps = ordered(mods)
    rows: list[C.Row] = []
    for e in exps:
        rows += mods[e].rows()
    ids = [r.id for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        print(f"duplicate row ids: {sorted(dup)}", file=sys.stderr)
        return 3
    red = {}
    rc = 0
    for e in exps:
        p = RED / f"{e}.json"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            rc = 3
            red[e] = {"checks": []}
        else:
            red[e] = C.jload(p)
    by_row: dict[str, list[dict]] = {}
    for e in exps:
        for x in red[e]["checks"]:
            by_row.setdefault(x["row"], []).append(x)
    # stale check: every check of every row must have a matching output (cmd+expected)
    for r in rows:
        got = by_row.get(r.id, [])
        want = {(c.cmd, c.expected, c.pick) for c in r.checks}
        have = {(x["cmd"], x["expected"], x["pick"]) for x in got}
        if want != have:
            print(f"stale rederive output for {r.id}", file=sys.stderr)
            rc = 3
            by_row[r.id] = []
    head = git("rev-parse", "HEAD")
    cats: dict[str, list[C.Row]] = {}
    lines: list[str] = []
    w = lines.append
    w("# Claims register")
    w("")
    w(
        "Generated by `docs/claims-register/build.py render` from the committed ledgers "
        "and raw records; **no number in this file is typed**. Written by a model "
        "(PLAN-2026-10-19 phase 0.2), not by Brendan."
    )
    w("")
    w(f"- Rendered at tree `{head[:12]}`; re-derivation outputs per experiment in "
      "`docs/claims-register/rederive/<experiment>.json` (each records the tree head it ran at).")
    w("- Rebuild: `.venv/bin/python docs/claims-register/build.py rederive && "
      ".venv/bin/python docs/claims-register/build.py render`.")
    w(f"- `$SLOT` = `{C.SLOT.strip()}`; `$LG` = `{C.LEDGER_GET}` (prints one ledger key).")
    w("- Raw records are read from the read-only archive `~/rsr-runs-archive/<worktree>/runs/` "
      "(sha256 manifest `~/rsr-runs-archive/MANIFEST.sha256`).")
    w("")
    w("## How to read a row")
    w("")
    w("- **value per seed / CI per seed**: from the cited ledger key (or raw record field). "
      "Display rounds to 4 decimals; the full value is at the key. Every CI is per seed; "
      "no CI is pooled across seeds. `n/a` = the run computed no CI for that number.")
    w("- **PREREG'd**: the number (statistic, data, rule) is named in a PREREG or amendment "
      "committed before the run that produced it. **EXPLORATORY**: it is not.")
    w("- **UNSEEN**: no value of this quantity on this data had been read when its analysis "
      "was fixed. **SEEN**: one had (a post-hoc or descriptive reading, or data already "
      "inspected for this quantity). The reason is given per row.")
    w("- **re-derived equal**: every check's output string-equals the ledger value in full "
      "precision. Check kinds: `raw` = recomputed from per-document / per-seed raw records "
      "through the run's own analysis code; `archive-ledger` = read from the run's own "
      "out-root ledger in the archive (not the committed copy); `ledger` = read of the "
      "committed key only (proves the register reads the right key; recomputes nothing). "
      "A mismatch is reported in the row, never smoothed.")
    w("")
    for tier, title in (("headline", "Headline contrasts"), ("secondary", "Secondary numbers")):
        w(f"## {title}")
        w("")
        for e in exps:
            rs = [r for r in rows if r.experiment == e and r.tier == tier]
            if not rs:
                continue
            w(f"### {e}")
            w("")
            w("| id | claim | value per seed | 95% CI per seed (method) | seeds (n) | "
              "source (ledger path + key) | PREREG'd / EXPLORATORY | SEEN / UNSEEN | "
              "re-derived equal | re-derivation command |")
            w("|---|---|---|---|---|---|---|---|---|---|")
            for r in rs:
                cat, st = status(r, by_row.get(r.id, []))
                cats.setdefault(cat, []).append(r)
                ci = (
                    "n/a"
                    if r.ci is None
                    else per_seed(r.seeds, r.ci, ci_str) + f" ({r.ci_method})"
                )
                cmds = list(dict.fromkeys(short(c.cmd) for c in r.checks))
                cmd_cell = "<br>".join(f"`{cell(c)}`" for c in cmds) or "none"
                claim = r.claim + (f" *({r.note})*" if r.note else "")
                w(
                    "| "
                    + " | ".join(
                        cell(x)
                        for x in (
                            f"`{r.id}`",
                            claim,
                            per_seed(r.seeds, r.values, C.fmt),
                            ci,
                            f"{','.join(map(str, r.seeds))} (n={len(r.seeds)})",
                            r.source,
                            r.prereg,
                            f"{r.seen}: {r.seen_why}",
                            st,
                        )
                    )
                    + f" | {cmd_cell} |"
                )
            w("")
    w("## Summary")
    w("")
    n = len(rows)
    w(f"- Rows: **{n}** ({sum(1 for r in rows if r.tier == 'headline')} headline, "
      f"{sum(1 for r in rows if r.tier == 'secondary')} secondary).")
    for cat, lab in (
        ("yes", "re-derived equal"),
        ("no", "re-derived NOT equal"),
        ("not-rederived", "not re-derived"),
        ("did-not-run", "re-derivation did not run"),
    ):
        w(f"- {lab}: **{len(cats.get(cat, []))}**")
    for cat in ("no", "did-not-run", "not-rederived"):
        if cats.get(cat):
            w("")
            w(f"### Rows: {cat}")
            w("")
            for r in cats[cat]:
                w(f"- `{r.id}`: {status(r, by_row.get(r.id, []))[1]}")
    kinds: dict[str, int] = {}
    for r in cats.get("yes", []):
        k = "+".join(sorted({c.kind for c in r.checks}))
        kinds[k] = kinds.get(k, 0) + 1
    w("")
    w("Equal rows by check kind: " + ", ".join(f"`{k}` {v}" for k, v in sorted(kinds.items())))
    w("")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        f"rows={n} " + " ".join(f"{k}={len(v)}" for k, v in sorted(cats.items())),
        file=sys.stderr,
    )
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("rederive")
    pr.add_argument("--only", nargs="*")
    sub.add_parser("render")
    sub.add_parser("list")
    a = ap.parse_args(argv)
    if a.cmd == "rederive":
        return rederive(a.only)
    if a.cmd == "list":
        mods = modules()
        for e in ordered(mods):
            for r in mods[e].rows():
                print(r.id, r.tier, len(r.checks), r.values)
        return 0
    return render()


if __name__ == "__main__":
    sys.exit(main())
