"""E0E. See RESULTS.md in this directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsr.exit_codes import Exit, not_implemented, run_main


def main() -> Exit:
    # Not implemented is DID NOT RUN (3). An uncaught NotImplementedError exits 1,
    # which is a real failure -- S0-05, docs/gates.md.
    return not_implemented("E0E")


if __name__ == "__main__":
    run_main(main)
