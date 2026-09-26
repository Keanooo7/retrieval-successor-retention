#!/bin/zsh
# carry-forward: the real run (experiments/carry-forward/PREREG.md, ca34aa7 + amendment f4e1c87).
#
# Runs the parent and writes ITS exit code to runs/carry-forward/run.rc -- captured
# directly with `; rc=$?`, never after a pipe (CLAUDE.md, "Reading an exit code").
# Output goes to runs/carry-forward.parent.log (gitignored). Refuses if
# runs/carry-forward exists (run.py refuses too).
emulate -L zsh
if (( $# )); then
  print -u2 "DID NOT RUN: run.sh takes no arguments; use run.py for --render-results"
  exit 3
fi
cd "${0:A:h:h:h}" || exit 3
if [[ -e runs/carry-forward ]]; then
  print -u2 "DID NOT RUN: runs/carry-forward exists; move it aside and re-run"
  exit 3
fi
uv run --extra dev python experiments/carry-forward/run.py > runs/carry-forward.parent.log 2>&1; rc=$?
mkdir -p runs/carry-forward
print -- "$rc" > runs/carry-forward/run.rc
print "carry-forward parent exited $rc (runs/carry-forward/run.rc; log runs/carry-forward.parent.log)"
exit $rc
