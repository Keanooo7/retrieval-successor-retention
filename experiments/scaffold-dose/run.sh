#!/bin/zsh
# scaffold-dose: the real run (experiments/scaffold-dose/PREREG.md, 4249567).
#
# Runs the parent and writes ITS exit code to runs/scaffold-dose/run.rc when it
# exits -- captured directly with `; rc=$?`, never after a pipe (CLAUDE.md,
# "Reading an exit code"). The parent's output goes to
# runs/scaffold-dose.parent.log (gitignored), because runs/scaffold-dose must not
# exist before the parent starts (it refuses a run directory that exists).
#
# No arguments: --dry-run and --render-results are run.py's own, run directly.
emulate -L zsh
if (( $# )); then
  print -u2 "DID NOT RUN: run.sh takes no arguments; use run.py for --dry-run / --render-results"
  exit 3
fi
cd "${0:A:h:h:h}" || exit 3
if [[ -e runs/scaffold-dose ]]; then
  # Refuse without touching it: a run.rc written here would overwrite the earlier run's.
  print -u2 "DID NOT RUN: runs/scaffold-dose exists; move it aside and re-run"
  exit 3
fi
uv run --extra dev python experiments/scaffold-dose/run.py > runs/scaffold-dose.parent.log 2>&1; rc=$?
mkdir -p runs/scaffold-dose
print -- "$rc" > runs/scaffold-dose/run.rc
print "scaffold-dose parent exited $rc (runs/scaffold-dose/run.rc; log runs/scaffold-dose.parent.log)"
exit $rc
