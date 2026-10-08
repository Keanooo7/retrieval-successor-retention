#!/bin/zsh
# offline-psi-converge launcher (PREREG 58783a3). Seeds run in parallel, each as its own
# cpu-det slot job: C2's bit-equality with offline-psi holds only inside that lane (a
# direct run differed by ~5e-11 in the 2026-10-08 smoke). Every status is read directly.
emulate -L zsh
set -u
cd "${0:A:h}/../.." || exit 3
porcelain=$(git status --porcelain); rc=$?
if (( rc != 0 )) || [[ -n $porcelain ]]; then echo "DID NOT RUN: tree not clean"; exit 3; fi
PY=.venv/bin/python
OUT=runs/offline-psi-converge
mkdir -p $OUT/logs
typeset -A pid
for s in 0 1 2; do
  PYTHONPATH=scripts $PY -m orchestrator.slot run --lane cpu-det --slots 1 \
    --job-id offline-psi-converge-s$s -- \
    $PY experiments/offline-psi-converge/run.py run --seed $s --out-root $OUT \
    > $OUT/logs/seed$s.log 2>&1 &
  pid[$s]=$!
done
for s in 0 1 2; do
  wait ${pid[$s]}; rc=$?
  echo "seed $s rc=$rc"
done
$PY experiments/offline-psi-converge/run.py finalize --out-root $OUT > $OUT/logs/finalize.log 2>&1
rc=$?
echo "finalize rc=$rc"
exit $rc
