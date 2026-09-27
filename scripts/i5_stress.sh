#!/bin/zsh
# I5 (PLAN-v4 §4): run ONE orchestrator test alone, repeatedly, while 3 full
# suites run in parallel as load. Every compute call goes through the cpu-det
# lane (orchestrator.slot). Logs land in $OUT; nothing is deleted.
#
#   scripts/i5_stress.sh OUT_DIR ATTEMPTS [stop-on-first-failure: 1|0]
#
# Prints one line per attempt: "attempt N rc=R". rc is captured directly
# (never after a pipe).
set -u
WT=${0:A:h:h}
OUT=$1
N=$2
STOP_ON_FAIL=${3:-1}
TEST="tests/test_orch_dispatch.py::test_submit_launches_the_slot_detached_at_the_pinned_sha"
mkdir -p "$OUT"
cd "$WT"
export PYTHONPATH=scripts
SLOT=(.venv/bin/python -m orchestrator.slot run --lane cpu-det --wait 900)
STOPFILE="$OUT/.stop-load"
rm -f "$STOPFILE"

load_loop() {
  local k=$1 i=0
  while [[ ! -e $STOPFILE ]]; do
    i=$((i + 1))
    $SLOT --slots 3 -- .venv/bin/pytest -p no:cacheprovider -q --tb=line \
      > "$OUT/load-$k-$i.log" 2>&1
    echo "rc=$?" >> "$OUT/load-$k-$i.log"
  done
}

for k in 1 2 3; do load_loop $k & done
sleep 20   # let the three suites get going before the first attempt

fails=0
for a in $(seq 1 "$N"); do
  $SLOT --slots 1 -- .venv/bin/pytest -p no:cacheprovider --tb=long -q "$TEST" \
    > "$OUT/attempt-$a.log" 2>&1
  rc=$?
  echo "attempt $a rc=$rc"
  if [[ $rc -ne 0 ]]; then
    fails=$((fails + 1))
    [[ $STOP_ON_FAIL -eq 1 ]] && break
  fi
done
echo "attempts_failed=$fails"
touch "$STOPFILE"
wait
echo "load stopped"
