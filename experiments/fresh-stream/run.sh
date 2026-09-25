#!/bin/zsh
# fresh-stream: the real run (experiments/fresh-stream/PREREG.md, 83128ee).
#
# Runs the parent and writes ITS exit code to runs/fresh-stream/run.rc when it
# exits -- captured directly with `; rc=$?`, never after a pipe (CLAUDE.md,
# "Reading an exit code"). The parent's output goes to
# runs/fresh-stream.parent.log (gitignored), because runs/fresh-stream must not
# exist before the parent starts (it refuses a run directory that exists).
#
# No arguments: --dry-run and --render-results are run.py's own, run directly.
emulate -L zsh
if (( $# )); then
  print -u2 "DID NOT RUN: run.sh takes no arguments; use run.py for --dry-run / --render-results"
  exit 3
fi
cd "${0:A:h:h:h}" || exit 3
if [[ -e runs/fresh-stream ]]; then
  # Refuse without touching it: a run.rc written here would overwrite the earlier run's.
  print -u2 "DID NOT RUN: runs/fresh-stream exists; move it aside and re-run"
  exit 3
fi
uv run --extra dev python experiments/fresh-stream/run.py > runs/fresh-stream.parent.log 2>&1; rc=$?
mkdir -p runs/fresh-stream
print -- "$rc" > runs/fresh-stream/run.rc
print "fresh-stream parent exited $rc (runs/fresh-stream/run.rc; log runs/fresh-stream.parent.log)"
exit $rc
