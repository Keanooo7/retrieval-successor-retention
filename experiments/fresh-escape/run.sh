#!/bin/zsh
# fresh-escape: the real run (experiments/fresh-escape/PREREG.md, d5b9a23).
#
# Runs the parent and writes ITS exit code to runs/fresh-escape/run.rc when it
# exits -- captured directly with `; rc=$?`, never after a pipe (CLAUDE.md,
# "Reading an exit code"). The parent's output goes to
# runs/fresh-escape.parent.log (gitignored), because runs/fresh-escape must not
# exist before the parent starts (it refuses a run directory that exists).
#
# No arguments: --dry-run and --render-results are run.py's own, run directly.
emulate -L zsh
if (( $# )); then
  print -u2 "DID NOT RUN: run.sh takes no arguments; use run.py for --dry-run / --render-results"
  exit 3
fi
cd "${0:A:h:h:h}" || exit 3
if [[ -e runs/fresh-escape ]]; then
  # Refuse without touching it: a run.rc written here would overwrite the earlier run's.
  print -u2 "DID NOT RUN: runs/fresh-escape exists; move it aside and re-run"
  exit 3
fi
uv run --extra dev python experiments/fresh-escape/run.py > runs/fresh-escape.parent.log 2>&1; rc=$?
mkdir -p runs/fresh-escape
print -- "$rc" > runs/fresh-escape/run.rc
print "fresh-escape parent exited $rc (runs/fresh-escape/run.rc; log runs/fresh-escape.parent.log)"
exit $rc
