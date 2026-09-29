"""Where did this pytest session import ``rsr`` from? (I1, `scripts/battery_isolation.py`)

Called from ``tests/conftest.py`` at session finish, and only when the mutation
battery set ``RSR_BATTERY_PROBE``. It records three paths and judges none of them:
the battery reads the file and refuses a suite whose ``rsr`` is not under its shard
(``battery_isolation.probe_problem``). Keeping the judgement out of the suite means
a mutation to this file can only make the probe missing or wrong -- which the
battery reports as DID NOT RUN -- never make a split-brain run look isolated.

Resolution uses ``importlib.util.find_spec``, which names the file ``rsr.__file__``
would be without executing the package; if ``rsr`` is already imported in this
process, its actual ``__file__`` is used.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

_CODE = (
    "import importlib.util as u, json;"
    "s = u.find_spec('rsr'); print(json.dumps(s.origin if s else None))"
)


def _in_process() -> str | None:
    mod = sys.modules.get("rsr")
    if mod is not None and getattr(mod, "__file__", None):
        return mod.__file__
    spec = importlib.util.find_spec("rsr")
    return spec.origin if spec else None


def _child(exe: str | None) -> str | None:
    if exe is None:
        return None
    with contextlib.suppress(OSError, ValueError):
        out = subprocess.run(
            [exe, "-c", _CODE], capture_output=True, text=True, timeout=60
        ).stdout
        return json.loads(out.strip() or "null")
    return None


def write_probe() -> None:
    path = os.environ.get("RSR_BATTERY_PROBE")
    if not path:
        return
    probe = {
        "in_process": _in_process(),
        "child": _child(sys.executable),
        "path_python": _child(shutil.which("python")),
        "executable": sys.executable,
        "pid": os.getpid(),
    }
    with contextlib.suppress(OSError), open(path, "w") as f:
        json.dump(probe, f)
