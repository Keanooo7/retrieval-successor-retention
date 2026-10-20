#!/bin/zsh
# RSR run 3, phase 2b launch (PREREG experiments/rsr-run3-ranking/PREREG.md @ 32475b5, §5).
# Run 2's launcher (experiments/rsr-run2/launch_run2.zsh) with exactly PREREG §0.3's change:
# both RSR arms add `--policy-override loss=rank --policy-override rank_k=3
# --policy-override rank_omega=<omega_s>` right after the target override, omega_s per
# seed machine-read from plan.json (`run.py 2b plan`, from 2a's ledger). Job ids are
# rsr-run3-2b-<arm>-s<seed>; OUT defaults to runs/rsr-run3-2b. The fifo command is run
# 1's byte for byte. STEPS is K_2b from plan.json (never typed); plan.json must be ok
# (2a WIN on all 3 seeds and the plan implementable as written) unless SMOKE=1.
# SMOKE=1 is S-2b (§5.3): STEPS=48, a scratch OUT and a JOB_SUFFIX, plan.json's omega_s;
# recorded in launch-git.json, so a smoke can never be read as the run.
# Owner slots (run 2's, §0.2): GAMMA (O1), WD (O4), O5. Any unfilled or inconsistent
# slot exits 3 ("did not run"). One rc per arm x seed, read directly.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
[[ -z ${SMOKE:-} || $SMOKE == 1 ]] || refuse "SMOKE must be unset or 1, got '$SMOKE'"
[[ -z ${STEPS:-} ]] || refuse "STEPS is K_2b from plan.json (or 48 under SMOKE); do not set it"
[[ -z ${CKPT:-} ]] || refuse "CKPT is derived from STEPS; do not set it"
if [[ -z ${SMOKE:-} ]]; then
  [[ -z ${RSR_ORCH_ROOT:-} || ${RSR_ORCH_ROOT:A} == ${R:A} ]] \
    || refuse "RSR_ORCH_ROOT=$RSR_ORCH_ROOT is not the shared root $R"
  [[ -z ${OUT:-} ]] || refuse "OUT=$OUT is set (a smoke hook)"
  [[ -z ${JOB_SUFFIX:-} ]] || refuse "JOB_SUFFIX=$JOB_SUFFIX is set (a smoke hook)"
else
  [[ -n ${OUT:-} ]] || refuse "SMOKE=1 needs OUT (a scratch dir, never runs/rsr-run3-2b)"
  [[ ${OUT:A} != ${${:-runs/rsr-run3-2b}:A} ]] || refuse "SMOKE=1 may not write runs/rsr-run3-2b"
  [[ -n ${JOB_SUFFIX:-} ]] || refuse "SMOKE=1 needs JOB_SUFFIX (never the run's job ids)"
fi
for v in GAMMA WD O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
PLAN=${PLAN:-runs/rsr-run3-2b/plan.json}
[[ -f $PLAN ]] || refuse "no $PLAN: run \`run.py 2b plan\` first"
PLAN_SHA=$(shasum -a 256 $PLAN); rc_ps=$?
(( rc_ps == 0 )) || refuse "shasum $PLAN rc=$rc_ps"
PLAN_SHA=${PLAN_SHA%% *}
PJ=(.venv/bin/python -c 'import json, sys
p = json.load(open(sys.argv[1])); k = sys.argv[2]
if k.startswith("omega"):
    v = p.get("omega_s", {}).get(k[5:])
    v = None if v is None else format(float(v), "g")
else:
    v = p.get(k)
print("" if v is None else v)')
PLAN_OK=$("${PJ[@]}" $PLAN ok)
if [[ -z ${SMOKE:-} ]]; then
  [[ $PLAN_OK == True ]] || refuse "$PLAN is not ok: 2b does not run (see its problems)"
  STEPS=$("${PJ[@]}" $PLAN K_2b)
else
  STEPS=48
fi
[[ $STEPS == <-> ]] || refuse "STEPS must be a non-negative integer, got '$STEPS'"
(( STEPS >= 1 )) || refuse "STEPS must be >= 1"
typeset -A OM
for s in 0 1 2; do
  OM[$s]=$("${PJ[@]}" $PLAN omega$s)
  [[ -n ${OM[$s]} ]] || refuse "no omega_s for seed $s in $PLAN"
done
if (( STEPS > 0 && STEPS % 3 == 0 )); then CKPT=$(( STEPS / 3 )); else CKPT=$STEPS; fi
DIAG_STEP=160
gcd() { local a=$1 b=$2 t; while (( b != 0 )); do t=$(( a % b )); a=$b; b=$t; done; print $a }
if (( STEPS >= DIAG_STEP && CKPT > 0 )); then EVERY=$(gcd $DIAG_STEP $CKPT); else EVERY=$CKPT; fi
if (( STEPS > 3000 && EVERY > 0 )); then EVERY=$(gcd $EVERY 3000); fi  # §5.4 psiR@3000 on disk
ORCH_REC=${RSR_ORCH_ROOT:-} ; OUT_REC=${OUT:-} ; SUFFIX_REC=${JOB_SUFFIX:-}
: ${OUT:=runs/rsr-run3-2b}
C1=(${=C1_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
[[ -e $OUT/launch-git.json ]] && refuse "$OUT was launched already (launch-git.json exists)"
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
[[ ${PLAN:A} == ${${:-$OUT/plan.json}:A} ]] || cp $PLAN $OUT/plan.json
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
  ckpt_every "$EVERY" ckpt_every_rule "gcd(160, CKPT) if STEPS>=160 and CKPT>0 else CKPT (A4.2); then gcd(., 3000) if STEPS>3000 (run 3 §5.4)" \
  diag_step "$DIAG_STEP" \
  c1_flags "${C1_FLAGS:-}" c2_flags "" c3_flags "${C3_FLAGS:-}" \
  rsr_orch_root "$ORCH_REC" out "$OUT_REC" job_suffix "$SUFFIX_REC" \
  dry_run "" smoke "${SMOKE:-}" target "loo_resample" loss "rank" rank_k "3" \
  omega_s0 "${OM[0]}" omega_s1 "${OM[1]}" omega_s2 "${OM[2]}" plan_sha256 "$PLAN_SHA" \
  launcher "experiments/rsr-run3-ranking/launch_2b.zsh" launched_utc "$LAUNCHED_UTC"; rc_json=$?
(( rc_json == 0 )) || refuse "writing $OUT/launch-git.json rc=$rc_json"

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)

typeset -A pid rcs
for s in 0 1 2; do
  B=$R/.worktrees/fresh-stream/runs/fresh-stream/B/seed$s
  RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
       --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA
       --policy-override target=loo_resample
       --policy-override loss=rank --policy-override rank_k=3
       --policy-override rank_omega=${OM[$s]})
  for a in psiC fifo age; do
    case $a in
      psiC) P=("${RSR[@]}") ;;
      fifo) P=(--policy fifo) ;;
      age)  P=("${RSR[@]}" --policy-override psi_override=age_only) ;;
    esac
    k=$a-s$s
    "${SLOT[@]}" --job-id rsr-run3-2b-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $EVERY --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --srep-norm-reg-weight 0.0 \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
