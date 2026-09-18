"""Session-level guards on the test run itself (gauntlet 1.3).

Two failures this file exists to prevent, both of which report **exit 0**:

1. **An all-skipped suite.** `pytest` exits 0 when every test skipped, and a green
   CI badge over zero live tests is the most expensive kind of false clean bill of
   health. `tests/test_reduction.py` and `tests/test_fidelity.py` were both skipped
   at module level on 83bdf57; a plausible next step would have skipped the rest.
2. **A silently shrinking suite.** A collection error in one file, an import that
   fails under a new dependency, a `testpaths` typo -- each removes tests without
   removing green.

The session therefore fails unless at least `MIN_PASSED` tests actually passed, and
writes a machine-readable count for CI to assert on.

**`pytest -q` on top of `addopts = "-q"` is `-qq`, which prints no count at all.**
That is how the count went missing in the first place: CI ran `pytest -q`, saw a
row of dots and exit 0, and reported a pass with no number in it. CI no longer
passes `-q`.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

MIN_PASSED = 100
"""A floor, not the current count.

Deliberately below today's number so it does not need editing on every commit, and
far enough above zero to catch a suite that has collapsed. The exact count is in
`test-count.json`; this is the tripwire.
"""

COUNT_FILE = Path(os.environ.get("RSR_TEST_COUNT", "test-count.json"))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    stats = terminalreporter.stats
    counts = {
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "skipped": len(stats.get("skipped", [])),
        "errors": len(stats.get("error", [])),
        "xfailed": len(stats.get("xfailed", [])),
        "xpassed": len(stats.get("xpassed", [])),
    }
    skip_reasons: dict[str, int] = {}
    for report in stats.get("skipped", []):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else "unknown"
        skip_reasons[str(reason)] = skip_reasons.get(str(reason), 0) + 1
    counts["skip_reasons"] = skip_reasons

    # A read-only checkout must not fail the suite over bookkeeping.
    with contextlib.suppress(OSError):
        COUNT_FILE.write_text(json.dumps(counts, indent=2) + "\n")

    terminalreporter.write_sep("-", "rsr test census")
    terminalreporter.write_line(
        f"passed={counts['passed']} failed={counts['failed']} "
        f"skipped={counts['skipped']} errors={counts['errors']}"
    )
    # 🔴 A skipped test is not a passing test. Printed separately, always.
    for reason, n in sorted(skip_reasons.items()):
        terminalreporter.write_line(f"  SKIPPED[{n}] {reason[:150]}")

    # The floor is about the *whole* suite. A developer running one file is not
    # claiming the suite is green, and failing them would only teach them to
    # delete the check.
    whole_suite = not config.getoption("file_or_dir") and not config.getoption(
        "-k", default=""
    )
    if (
        whole_suite
        and counts["passed"] < MIN_PASSED
        and not config.getoption("--collect-only")
    ):
        terminalreporter.write_line(
            f"FAILED: only {counts['passed']} tests passed, floor is {MIN_PASSED}. "
            f"An all-skipped or collapsed suite exits 0 without this check "
            f"(gauntlet 1.3)."
        )
        raise SystemExit(1)
