#!/bin/zsh
# psi-ls-capacity launcher (PREREG 87dc2ec). Seeds run in parallel, each as its own
# cpu-det slot job (diagnostic A's lane). Every status is read directly.
emulate -L zsh
set -u
cd "${0:A:h}/../.." || exit 3
porcelain=$(git status --porcelain); rc=$?
if (( rc != 0 )) || [[ -n $porcelain ]]; then echo "DID NOT RUN: tree not clean"; exit 3; fi
PY=.venv/bin/python
OUT=runs/psi-ls-capacity
mkdir -p $OUT/logs
typeset -A pid
for s in 0 1 2; do
  PYTHONPATH=scripts $PY -m orchestrator.slot run --lane cpu-det --slots 1 \
    --job-id psi-ls-capacity-s$s -- \
    $PY experiments/psi-ls-capacity/run.py run --seed $s --out-root $OUT \
    > $OUT/logs/seed$s.log 2>&1 &
  p=$!; pid[$s]=$p  # zsh 5.9: pid[$s]=$! stores the literal "$!" (diagnostic A)
done
for s in 0 1 2; do
  wait ${pid[$s]}; rc=$?
  echo "seed $s rc=$rc"
done
# finalize runs regardless: a control failure (exit 3) still reports its numbers
# (PREREG §4); a seed with no result.json makes finalize raise.
$PY experiments/psi-ls-capacity/run.py finalize --out-root $OUT > $OUT/logs/finalize.log 2>&1
rc=$?
echo "finalize rc=$rc"
exit $rc
