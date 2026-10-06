#!/bin/zsh
# Expire-Span referendum launch (PREREG experiments/expire-span-referendum/PREREG.md §9).
# Run with /bin/zsh INSIDE tmux, from a dedicated run worktree at the reviewed head.
# 27 jobs = seeds 0-2 x the 9 configs `run.py grid` reads from PREREG §4.1 / §4.2; no
# grid value is typed here. Everything else is RSR run 1's launch (its PREREG §1, §6,
# A1.4, A4.1): the arm-B ckpt3000 sha pins below (the same shape as run 1's), the pool,
# --freeze-transformer --frozen-model-mode eval (O5 = eval), --srep-norm-reg-weight 0.0,
# 3000 steps, lr 1e-3, batch 16, CPU, one cpu-det slot (1 thread) per job. Only the
# checkpoint cadence differs (--ckpt-every = STEPS: only step 3000 is read, §6.1).
# This is NOT run 1's launcher; launch_run1.zsh (sha256 cb0bef7f...) is untouched.
# DRY_RUN=1 runs every job at --iters 0 and needs OUT and JOB_SUFFIX set; a real launch
# needs RSR_ORCH_ROOT unset (or the shared root) and OUT / JOB_SUFFIX unset. Any
# refusal exits 3 ("did not run"). One rc per config x seed, captured directly.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${DRY_RUN:-} || $DRY_RUN == 1 ]] || refuse "DRY_RUN must be unset or 1, got '$DRY_RUN'"
if [[ -z ${DRY_RUN:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a dry-run hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a dry-run hook)"
  STEPS=3000
else
  [[ -n ${OUT:-} && -n ${JOB_SUFFIX:-} ]] || refuse "DRY_RUN=1 needs OUT and JOB_SUFFIX"
  STEPS=0
fi
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/expire-span-referendum}

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
GRID_OUT=$(.venv/bin/python experiments/expire-span-referendum/run.py grid); rc_grid=$?
(( rc_grid == 0 )) || refuse "run.py grid rc=$rc_grid"
GRID=("${(@f)GRID_OUT}")
(( ${#GRID} == 9 )) || refuse "run.py grid gave ${#GRID} configs, not 9"
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
LAUNCHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python -c 'import json, sys
a = sys.argv[2:]
d = dict(zip(a[0::2], a[1::2]))
d["dirty"] = False
d["grid"] = d["grid"].split("\n")
with open(sys.argv[1], "w") as f:
    json.dump(d, f)
    f.write("\n")' $OUT/launch-git.json \
  run_sha "$RUN_SHA" steps "$STEPS" grid "$GRID_OUT" \
  rsr_orch_root "$ORCH_REC" out "$OUT_REC" job_suffix "$SUFFIX_REC" \
  dry_run "${DRY_RUN:-}" launched_utc "$LAUNCHED_UTC"; rc_json=$?
(( rc_json == 0 )) || refuse "writing $OUT/launch-git.json rc=$rc_json"

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 43200 --pinned-sha $RUN_SHA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for line in "${GRID[@]}"; do
    parts=(${=line}); cid=$parts[1]; OV=()
    for kv in "${(@)parts[2,-1]}"; do OV+=(--policy-override $kv); done
    k=$cid-s$s
    "${SLOT[@]}" --job-id es-ref-a2-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $STEPS --device cpu \
        --freeze-transformer --frozen-model-mode eval \
        --srep-norm-reg-weight 0.0 \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        --policy expire_span "${OV[@]}" --out-dir $OUT/$cid/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # run 1's measured note: pid[$k]=$! stores the literal "$!"
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
