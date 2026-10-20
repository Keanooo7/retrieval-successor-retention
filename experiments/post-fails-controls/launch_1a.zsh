#!/bin/zsh
# PREREG post-fails-controls §3 (item 1a): the launch, in order, at this head.
#   1. the full suite (G-C3), its rc read directly and written to logs/suite.rc
#   2. launch-checks: G-C1(e) branch search, G-C3 census, the frozen manifest
#   3. measure seeds 0 and 1 in parallel (2 cpu-det slots, 1 thread each), then 2
#   4. finalize 1a (ledger, claims.json, RESULTS.md under runs/)
# Every step's rc goes to runs/post-fails-controls-1a/logs/<step>.rc, captured
# directly after the command (never after a pipe). If the launch checks fail,
# nothing is measured and finalize records exit 3.
set -u
cd "${0:A:h}/../.."
OUT=runs/post-fails-controls-1a
LOGS=$OUT/logs
if [[ -e $OUT/launch.json || -e $OUT/measure-seed0.json ]]; then
  echo "DID NOT RUN: $OUT already holds a launch" >&2
  exit 3
fi
mkdir -p $LOGS
GIT=/opt/homebrew/bin/git
PY=.venv/bin/python
RUN=experiments/post-fails-controls/run.py
slot() { PYTHONPATH=scripts $PY -m orchestrator.slot run --lane cpu-det --slots 1 --wait 7200 -- "$@"; }

$GIT rev-parse HEAD > $LOGS/suite.head
slot $PY -m pytest -rs > $LOGS/suite.log 2>&1; echo $? > $LOGS/suite.rc

slot $PY $RUN 1a launch-checks > $LOGS/launch-checks.log 2>&1; rc=$?
echo $rc > $LOGS/launch-checks.rc
if [[ $rc -eq 0 ]]; then
  (slot $PY $RUN 1a measure --seed 0 > $LOGS/1a-s0.log 2>&1; echo $? > $LOGS/1a-s0.rc) &
  (slot $PY $RUN 1a measure --seed 1 > $LOGS/1a-s1.log 2>&1; echo $? > $LOGS/1a-s1.rc) &
  wait
  slot $PY $RUN 1a measure --seed 2 > $LOGS/1a-s2.log 2>&1; echo $? > $LOGS/1a-s2.rc
else
  echo "launch checks failed (rc $rc): nothing measured" >&2
fi

slot $PY $RUN finalize 1a > $LOGS/finalize.log 2>&1; echo $? > $LOGS/finalize.rc
echo done > $LOGS/launcher.done
