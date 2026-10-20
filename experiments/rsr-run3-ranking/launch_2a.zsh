#!/bin/zsh
# PREREG rsr-run3-ranking §4 (phase 2a, run id rsr-run3-2a): the launch, in order, at
# this head. Run with /bin/zsh INSIDE tmux, from a clean worktree at the reviewed commit.
#   1. the full suite (G-C3), rc read directly -> logs/suite.rc (2 cpu-det slots: T1(b)
#      runs two 20-step processes)
#   2. launch-checks: R-SEAL (d) branch search, G-C3 census, the frozen manifest;
#      then the Amendment PM-1 slot check (exit 3 -> nothing is fitted)
#   3. per seed, concurrently: R-C2 and the 3 grid fits (one slot each), then select
#      (TUNE's first read), then deploy (EVAL once). A failed fit makes select exit 3
#      (R-C6) and a missing selection makes deploy exit 3 (the guard): nothing is
#      skipped silently.
#   4. finalize (ledger, claims.json, RESULTS.md under runs/rsr-run3-2a)
# Every step's rc goes to runs/rsr-run3-2a/logs/<step>.rc, captured directly after the
# command (never after a pipe). If the launch checks fail, nothing is fitted.
set -u
cd "${0:A:h}/../.."
OUT=runs/rsr-run3-2a
LOGS=$OUT/logs
GIT=/opt/homebrew/bin/git
PY=.venv/bin/python
RUN=experiments/rsr-run3-ranking/run.py
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -e $OUT/launch.json || -e $OUT/events.jsonl ]] && refuse "$OUT already holds a launch"
HEAD_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $LOGS
slot() {
  local k=$1; shift
  PYTHONPATH=scripts $PY -m orchestrator.slot run --lane cpu-det --slots $k \
    --wait 28800 --pinned-sha $HEAD_SHA -- "$@"
}

print -r -- $HEAD_SHA > $LOGS/suite.head
slot 2 $PY -m pytest -rs > $LOGS/suite.log 2>&1; echo $? > $LOGS/suite.rc

slot 1 $PY $RUN 2a launch-checks > $LOGS/launch-checks.log 2>&1; rc=$?
echo $rc > $LOGS/launch-checks.rc
if [[ $rc -eq 0 ]]; then
  # Amendment PM-1: record the cpu-det slots granted; with fewer than 9, the wall
  # projection is recomputed from 2a-S for that count and must be <= 8 h. Exit 3 here
  # means nothing is fitted.
  slot 1 $PY $RUN 2a slot-check --self-slots 1 > $LOGS/slot-check.log 2>&1; rc=$?
  echo $rc > $LOGS/slot-check.rc
fi
if [[ $rc -eq 0 ]]; then
  for s in 0 1 2; do
    (
      (slot 1 $PY $RUN 2a fit --seed $s --rc2 > $LOGS/fit-s$s-rc2.log 2>&1
       echo $? > $LOGS/fit-s$s-rc2.rc) &
      for w in 2 8 32; do
        (slot 1 $PY $RUN 2a fit --seed $s --omega $w > $LOGS/fit-s$s-omega$w.log 2>&1
         echo $? > $LOGS/fit-s$s-omega$w.rc) &
      done
      wait
      slot 1 $PY $RUN 2a select --seed $s > $LOGS/select-s$s.log 2>&1
      echo $? > $LOGS/select-s$s.rc
      slot 1 $PY $RUN 2a deploy --seed $s > $LOGS/deploy-s$s.log 2>&1
      echo $? > $LOGS/deploy-s$s.rc
    ) &
  done
  wait
else
  print -u2 "launch checks or the PM-1 slot check failed (rc $rc): nothing fitted"
fi

slot 1 $PY $RUN 2a finalize > $LOGS/finalize.log 2>&1; echo $? > $LOGS/finalize.rc
echo done > $LOGS/launcher.done
