#!/bin/zsh
# One deterministic pass of the RSR orchestrator. launchd runs this every 600 s
# (ops/launchd/com.keanooo7.rsr.tick.plist). The pass itself is
# scripts/orchestrator/tick.py; this wrapper only finds the tree and the interpreter,
# honours HALT before anything else runs, and reports the pass's exit status.
#
# Exit statuses (rsr.exit_codes): 0 pass done (incl. HALT, lock held, idle)
# · 1 a merge guard trip · 3 did not run (caps UNSET, no interpreter, ...).
#
# CLAUDE.md, "Reading an exit code": the status is captured into rc on the SAME
# line as the command, with no pipe in between. Never ${PIPESTATUS[0]} -- that is
# bash, and in zsh it expands to the empty string.

emulate -L zsh
setopt no_unset

code_dir=${0:A:h:h}                       # .../scripts
repo=${RSR_ORCH_ROOT:-${code_dir:h}}
export RSR_ORCH_ROOT=$repo

if [[ -e $repo/.orchestrator/HALT ]]; then
  print -r -- "[tick.zsh] HALT present -- nothing done: $(tail -n 1 $repo/.orchestrator/HALT)"
  exit 0
fi

python=${RSR_PYTHON:-${code_dir:h}/.venv/bin/python}
if [[ ! -x $python ]]; then
  print -r -u2 -- "DID NOT RUN: no interpreter at $python (uv sync --frozen first)"
  exit 3
fi

mkdir -p $repo/.orchestrator/logs
export PYTHONPATH=$code_dir${PYTHONPATH:+:$PYTHONPATH}
$python -m orchestrator.tick "$@"; rc=$?
print -r -- "[tick.zsh] tick exited $rc"
exit $rc
