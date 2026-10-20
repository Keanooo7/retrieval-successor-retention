#!/bin/zsh
# RSR run 2 launch (PREREG experiments/rsr-run2/PREREG.md @ a406d39). Run 1's launcher v6
# (experiments/rsr-run1/launch_run1.zsh, PREREG rsr-run1 A4.3) with exactly the PREREG §0
# change: both RSR arms add `--policy-override target=loo_resample` (owner ruling C3). Job
# ids are rsr-run2-<arm>-s<seed>; OUT defaults to runs/rsr-run2. The fifo command is run
# 1's byte for byte (C12).
# Run with /bin/zsh INSIDE tmux, from the dedicated run worktree at the reviewed commit.
# Owner slots (run 1's ruled values, R-2026-10-01-rsr-run-1-owner-values; PREREG rsr-run2 §0
# cites them unchanged): GAMMA (O1), WD (O4), STEPS (R-1), O5. CKPT is DERIVED from STEPS.
# RSR_ORCH_ROOT must be unset or the shared root, and OUT / JOB_SUFFIX unset, unless
# DRY_RUN=1 (STEPS=0) or SMOKE=1. SMOKE=1 is the pre-launch pipeline check: it needs OUT
# (a scratch dir) and JOB_SUFFIX, allows 1 <= STEPS <= 30, and is recorded in
# launch-git.json, so C10 fails every smoke seed by design: a smoke can never be read as
# the run. Any unfilled or inconsistent slot exits 3 ("did not run"). One rc per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${DRY_RUN:-} || $DRY_RUN == 1 ]] || refuse "DRY_RUN must be unset or 1, got '$DRY_RUN'"
[[ -z ${SMOKE:-} || $SMOKE == 1 ]] || refuse "SMOKE must be unset or 1, got '$SMOKE'"
[[ -n ${DRY_RUN:-} && -n ${SMOKE:-} ]] && refuse "DRY_RUN and SMOKE are exclusive"
if [[ -z ${DRY_RUN:-} && -z ${SMOKE:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a dry-run / smoke hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a dry-run / smoke hook)"
fi
if [[ -n ${SMOKE:-} ]]; then
  [[ -n ${OUT:-} ]] || refuse "SMOKE=1 needs OUT (a scratch dir, never runs/rsr-run2)"
  [[ ${OUT:A} != ${${:-runs/rsr-run2}:A} ]] || refuse "SMOKE=1 may not write runs/rsr-run2"
  [[ -n ${JOB_SUFFIX:-} ]] || refuse "SMOKE=1 needs JOB_SUFFIX (never the run's job ids)"
fi
[[ -z ${CKPT:-} ]] || refuse "CKPT is derived from STEPS; do not set it"
for v in GAMMA WD STEPS O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $STEPS == <-> ]] || refuse "STEPS must be a non-negative integer, got '$STEPS'"
[[ -n ${DRY_RUN:-} ]] && (( STEPS != 0 )) && refuse "DRY_RUN=1 needs STEPS=0 (a dry run never trains), got STEPS=$STEPS"
[[ -n ${SMOKE:-} ]] && (( STEPS < 1 || STEPS > 30 )) && refuse "SMOKE=1 needs 1 <= STEPS <= 30, got STEPS=$STEPS"
(( STEPS >= 1 )) || [[ -n ${DRY_RUN:-} ]] || refuse "STEPS must be >= 1"
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
if (( STEPS > 0 && STEPS % 3 == 0 )); then CKPT=$(( STEPS / 3 )); else CKPT=$STEPS; fi
DIAG_STEP=160
if (( STEPS >= DIAG_STEP && CKPT > 0 )); then
  a=$DIAG_STEP; b=$CKPT; while (( b != 0 )); do t=$(( a % b )); a=$b; b=$t; done; EVERY=$a
else
  EVERY=$CKPT
fi
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/rsr-run2}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
[[ -e $OUT/launch-git.json ]] && refuse "$OUT was launched already (launch-git.json exists)"
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
LAUNCHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
.venv/bin/python -c 'import json, sys
a = sys.argv[2:]
d = dict(zip(a[0::2], a[1::2]))
d["dirty"] = False
d["porcelain_lines"] = 0
with open(sys.argv[1], "w") as f:
    json.dump(d, f)
    f.write("\n")' $OUT/launch-git.json \
  run_sha "$RUN_SHA" gamma "$GAMMA" wd "$WD" steps "$STEPS" ckpt "$CKPT" \
  ckpt_rule "STEPS/3 if STEPS>0 and STEPS%3==0 else STEPS (A1.8 m-9)" o5 "$O5" \
  ckpt_every "$EVERY" ckpt_every_rule "gcd(160, CKPT) if STEPS>=160 and CKPT>0 else CKPT (A4.2)" \
  diag_step "$DIAG_STEP" \
  c1_flags "${C1_FLAGS:-}" c2_flags "${C2_FLAGS:-}" c3_flags "${C3_FLAGS:-}" \
  rsr_orch_root "$ORCH_REC" out "$OUT_REC" job_suffix "$SUFFIX_REC" \
  dry_run "${DRY_RUN:-}" smoke "${SMOKE:-}" target "loo_resample" \
  launcher "experiments/rsr-run2/launch_run2.zsh" launched_utc "$LAUNCHED_UTC"; rc_json=$?
(( rc_json == 0 )) || refuse "writing $OUT/launch-git.json rc=$rc_json"

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA
     --policy-override target=loo_resample)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run2-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $EVERY --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --srep-norm-reg-weight 0.0 \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured (run 1): pid[$k]=$! stores the literal "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
