#!/bin/zsh
# RSR run 1 launch, version 2 (PREREG Amendment 2, A2.4; supersedes A1.2's text).
# Run with /bin/zsh INSIDE tmux (A2.4), from the dedicated run worktree at the final
# reviewed commit. Owner slots: GAMMA (O1), WD (O4), STEPS (R-1), O5 (eval|train).
# Derived: CKPT (A1.8 m-9). Code-change flags: C1_FLAGS (C-1), C2_FLAGS (C-2), C3_FLAGS
# (C-3, only if O4 != 0.01). OUT defaults to runs/rsr-run1. Any unfilled or inconsistent
# slot exits 3 ("did not run"). Every rc is captured directly, one per arm x seed.
R=/Users/keanooo7/retrieval-successor-retention
GIT=/opt/homebrew/bin/git
refuse() { print -u2 "DID NOT RUN: $1"; exit 3 }
for v in GAMMA WD STEPS CKPT O5; do [[ -n ${(P)v} ]] || refuse "slot $v unfilled"; done
[[ $O5 == eval || $O5 == train ]] || refuse "O5 must be eval or train, got '$O5'"
[[ $O5 == eval && -z ${C1_FLAGS:-} ]] && refuse "O5=eval needs C1_FLAGS (C-1's flag)"
[[ $O5 == train && -n ${C1_FLAGS:-} ]] && refuse "O5=train: C1_FLAGS must be empty"
[[ $WD != 0.01 && -z ${C3_FLAGS:-} ]] && refuse "O4=$WD needs C3_FLAGS (code change C-3)"
[[ $WD == 0.01 && -n ${C3_FLAGS:-} ]] && refuse "O4=0.01 is the built value: C3_FLAGS must be empty"
: ${OUT:=runs/rsr-run1}
C1=(${=C1_FLAGS:-}) ; C2=(${=C2_FLAGS:-}) ; C3=(${=C3_FLAGS:-})

RUN_SHA=$($GIT rev-parse HEAD); rc_sha=$?
PORCELAIN=$($GIT status --porcelain); rc_status=$?
if (( rc_sha != 0 || rc_status != 0 )) || [[ -n $PORCELAIN ]]; then
  refuse "not a clean committed head (rc_sha=$rc_sha rc_status=$rc_status)"
fi
mkdir -p $OUT/logs; rc_mkdir=$?
(( rc_mkdir == 0 )) || refuse "mkdir $OUT/logs rc=$rc_mkdir"
print -r -- "{\"run_sha\": \"$RUN_SHA\", \"dirty\": false, \"porcelain_lines\": 0, \"gamma\": \"$GAMMA\", \"wd\": \"$WD\", \"steps\": \"$STEPS\", \"ckpt\": \"$CKPT\", \"o5\": \"$O5\", \"c1_flags\": \"${C1_FLAGS:-}\", \"c2_flags\": \"${C2_FLAGS:-}\", \"c3_flags\": \"${C3_FLAGS:-}\", \"launched_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > $OUT/launch-git.json

typeset -A SHA
SHA[0]=0ee3f8a69b507d927c631eb85116e1cd9739ee4472c44eb7f145d739a118da60
SHA[1]=dadd1e08a3849c1211c8479394df4060f91cd2a6e188b97701b783f2068a3da8
SHA[2]=b507ebc573a54316f9d3c20337f49e664bb9c414388af925c5a8e87efd6b63b8
SLOT=(env PYTHONPATH=scripts .venv/bin/python -m orchestrator.slot run
      --lane cpu-det --slots 1 --wait 3600 --pinned-sha $RUN_SHA)
RSR=(--policy rsr --policy-override b_enabled=false --policy-override shadow_enabled=false
     --policy-override nu=0 --policy-override beta=1 --policy-override gamma=$GAMMA)

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
    "${SLOT[@]}" --job-id rsr-run1-$k${JOB_SUFFIX:-} -- \
      .venv/bin/python -m rsr.train.loop \
        --d 128 --batch 16 --iters $STEPS --steps-per-stream 48 --memory-slots 16 \
        --lr 0.001 --seed $s --beat-every 1 --ckpt-every $CKPT --device cpu \
        --freeze-transformer "${C1[@]}" "${C3[@]}" \
        --pool-offset 993000 --pool-length 256 --pool-vocab-documents 64 "${C2[@]}" \
        --warm-start $B/ckpt-003000.pt --warm-start-sha256 ${SHA[$s]} \
        --warm-start-manifest /Users/keanooo7/rsr-substrate/2026-09-27/MANIFEST.sha256 \
        --warm-start-root $R --warm-start-source-config $B/result.json \
        "${P[@]}" --out-dir $OUT/$a/seed$s \
      > $OUT/logs/$k.log 2>&1 &
    p=$!; pid[$k]=$p      # measured: pid[$k]=$! stores the literal string "$!" in zsh 5.9
  done
done
for k in ${(ok)pid}; do
  wait ${pid[$k]}; rcs[$k]=$?
  print -r -- ${rcs[$k]} > $OUT/logs/$k.rc
  print -r -- "rc[$k]=${rcs[$k]}"
done
