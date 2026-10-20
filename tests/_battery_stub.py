"""A tiny repository and a driver that runs the REAL battery on it (I1 kill tests).

``build_stub(dir)`` makes a committed git repo with a two-line ``rsr`` package, two
tests and this tree's ``tests/_battery_probe.py``. ``python _battery_stub.py drive
STUB MODE [battery args...]`` runs ``scripts/mutation_battery.py``'s own ``main``
with ``ROOT`` pointed at the stub, ``MUTATIONS`` set to ``TABLE`` and the shard
venv built by ``stub_sync`` instead of ``uv sync`` -- everything else, the shard
pool, the probe check, the reset and the signal handling, is the real code.

The stub venv is a plain ``python -m venv`` whose ``.pth`` puts the shard's
``src`` first and this tree's site-packages (for pytest) after it, as a plain
path: that directory's own ``.pth`` files -- the host's editable ``rsr`` -- are
not processed, so ``rsr`` resolves under the shard or nowhere.

``MODE``: ``good``; ``foreign-pytest`` -- the shard's ``pytest`` runs on THIS
tree's interpreter: the stub conftest puts the shard's src first, so the suite
itself is green, but every child it spawns imports the host's ``rsr`` -- the
2026-09-27 split-brain (a suite on another checkout's venv), reproduced with a
green baseline; ``foreign-src`` --
the shard venv's own python resolves ``rsr`` to THIS tree's ``src`` (an editable
install pointing at another checkout); ``no-probe`` -- the shard's suite never
receives ``RSR_BATTERY_PROBE``, so even its baseline reports nothing.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"

CALC = "VALUE = 1\nSLOW = False\nOTHER = 2\n"

TEST_CALC = """import os
import subprocess
import sys
import time
from pathlib import Path

from rsr import calc

POISON = Path(__file__).resolve().parents[1] / "runs" / "poison"


def test_gate():
    if calc.SLOW:
        # a grandchild in pytest's process group, as the real suite's nested
        # batteries and run.py children are
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        marker = os.environ.get("STUB_MARKER")
        if marker:
            with open(marker + ".tmp", "w") as f:
                f.write(f"{os.getpid()} {child.pid}")
            os.replace(marker + ".tmp", marker)
        time.sleep(120)
    assert calc.VALUE == 1


def test_other():
    assert calc.OTHER == 2


def test_no_poison():
    # An ignored file one mutation's suite writes (runs/ is ignored, like the
    # real tree's runs/ and .orchestrator/ state) must not reach the next suite.
    if calc.OTHER == 5:
        POISON.parent.mkdir(exist_ok=True)
        POISON.write_text("written under the previous mutation")
        return
    assert not POISON.exists(), POISON.read_text()
"""

# Like the real suite's test files, the stub puts its own src first in sys.path.
# So under ``foreign-pytest`` the pytest process imports the SHARD's rsr and the
# baseline is green -- only a child of pytest reaches the host's rsr. That is the
# 09-27 topology, and it is what the child clause of the path assertion is for.
#
# It also pins calc.py's mtime. A .pyc is trusted when the source's mtime (whole
# seconds) and size match, and every stub mutation of calc.py keeps its size: with
# a fixed mtime, ANY bytecode that survives from one suite into the next is used,
# deterministically -- the same-second race of 2026-09-29, without the race.
CONFTEST = """import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
os.utime(_SRC / "rsr" / "calc.py", (1_000_000_000, 1_000_000_000))
sys.path.insert(0, str(_SRC))

from _battery_probe import write_probe  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    write_probe(exitstatus)
"""

TABLE = {
    "proven": ("test_gate", "src/rsr/calc.py", "VALUE = 1", "VALUE = 3"),
    "leaks": (
        "test_gate",
        "src/rsr/calc.py",
        CALC,
        CALC.replace("1", "3").replace("2", "5"),
    ),
    "adds-nothing": ("test_gate", "src/rsr/calc.py", "OTHER = 2", "OTHER = 5"),
    "slow": ("test_gate", "src/rsr/calc.py", "SLOW = False", "SLOW = True"),
    "no-probe": (
        "test_gate",
        "tests/conftest.py",
        "    write_probe(exitstatus)",
        "    pass",
    ),
}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    sys.path.insert(0, str(SCRIPTS))
    import battery_isolation as iso

    return iso.git(cwd, *args)


def build_stub(root: Path) -> Path:
    (root / "src" / "rsr").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "rsr" / "__init__.py").write_text("")
    (root / "src" / "rsr" / "calc.py").write_text(CALC)
    (root / "tests" / "test_calc.py").write_text(TEST_CALC)
    (root / "tests" / "conftest.py").write_text(CONFTEST)
    (root / "tests" / "_battery_probe.py").write_text(
        (HERE / "_battery_probe.py").read_text()
    )
    (root / ".gitignore").write_text(".venv/\n__pycache__/\nruns/\n")
    for args in (
        ("init", "-q"),
        ("add", "-A"),
        ("-c", "user.email=stub@rsr", "-c", "user.name=stub", "commit", "-qm", "stub"),
    ):
        proc = _git(root, *args)
        assert proc.returncode == 0, proc.stderr
    return root


def stub_sync(mode: str):
    host_site = sysconfig.get_paths()["purelib"]
    base = getattr(sys, "_base_executable", sys.executable)

    def sync(shard_root: Path) -> None:
        venv = shard_root / ".venv"
        subprocess.run(
            [base, "-m", "venv", "--without-pip", str(venv)],
            check=True,
            capture_output=True,
        )
        (site,) = (venv / "lib").glob("python3*/site-packages")
        src = HERE.parent / "src" if mode == "foreign-src" else shard_root / "src"
        (site / "_stub.pth").write_text(f"{src}\n{host_site}\n")
        foreign = mode == "foreign-pytest"
        target = Path(sys.executable) if foreign else venv / "bin" / "python"
        # no-probe: the suite runs, but never learns where to write its probe
        unset = "RSR_BATTERY_PROBE= " if mode == "no-probe" else ""
        pytest = venv / "bin" / "pytest"
        pytest.write_text(f'#!/bin/sh\n{unset}exec "{target}" -m pytest "$@"\n')
        pytest.chmod(pytest.stat().st_mode | stat.S_IEXEC)

    return sync


def drive(stub: Path, mode: str, names: list[str], argv: list[str]) -> None:
    sys.path.insert(0, str(SCRIPTS))
    import battery_isolation as iso
    import mutation_battery as mb

    mb.ROOT = stub.resolve()
    mb.MUTATIONS = tuple(mb.Mutation(n, *TABLE[n], why=n) for n in names)
    iso.sync_venv = stub_sync(mode)
    sys.argv = ["mutation_battery.py", *argv]
    mb.run_main(mb.main)


def clean_env(**extra: str) -> dict[str, str]:
    """The driver's env: nothing of an enclosing battery or venv leaks in."""
    drop = ("RSR_BATTERY_PROBE", "RSR_ORCH_ROOT", "VIRTUAL_ENV", "RSR_TEST_COUNT")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(extra)
    return env


if __name__ == "__main__":
    # drive STUB MODE NAMES(comma-separated) [battery args...]
    _, cmd, stub_, mode_, names_, *rest = sys.argv
    assert cmd == "drive", cmd
    drive(Path(stub_), mode_, [n for n in names_.split(",") if n], rest)
