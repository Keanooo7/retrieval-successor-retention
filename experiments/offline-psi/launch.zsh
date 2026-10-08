#!/bin/zsh
# offline-psi launcher: PREREG experiments/offline-psi/PREREG.md (c200a08) and its
# §10 amendment. Run inside tmux, from anywhere; it works in this worktree's root.
# Seeds run one after another, each in the cpu-det lane (PREREG §10's measured cost).
# Every status is captured directly ($? right after the command, never after a pipe).
emulate -L zsh
set -u
cd "${0:A:h}/../.." || exit 3
PY=.venv/bin/python
OUT=runs/offline-psi
THREADS=${THREADS:-1}
# Review offline-psi-8418741 L1: the tree must be clean BEFORE this script writes
# anything, and every file written before `manifest` must be git-ignored (`.log`):
# `.gitignore` un-ignores `runs/**/*.json`, and `manifest` refuses any porcelain.
porcelain=$(git status --porcelain); rc=$?
if (( rc != 0 )) || [[ -n $porcelain ]]; then echo "DID NOT RUN: tree not clean"; exit 3; fi
mkdir -p $OUT/logs
$PY experiments/offline-psi/run.py preflight > $OUT/logs/preflight.log 2>&1; rc=$?
if (( rc != 0 )); then echo "preflight rc=$rc (see $OUT/logs/preflight.log)"; exit 3; fi
$PY experiments/offline-psi/run.py manifest --out-root $OUT > $OUT/logs/manifest.log 2>&1; rc=$?
if (( rc != 0 )); then echo "manifest rc=$rc"; exit 3; fi
for s in 0 1 2; do
  PYTHONPATH=scripts $PY -m orchestrator.slot run --lane cpu-det --slots $THREADS \
    --job-id offline-psi-s$s -- \
    $PY experiments/offline-psi/run.py measure-seed --seed $s --threads $THREADS \
    --out-root $OUT > $OUT/logs/seed$s.log 2>&1
  rc=$?
  echo "seed $s rc=$rc"
done
$PY experiments/offline-psi/run.py finalize --out-root $OUT > $OUT/logs/finalize.log 2>&1
rc=$?
echo "finalize rc=$rc"
exit $rc
