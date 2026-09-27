"""Run scripts/mutation_battery.py's own main() with MUTATIONS filtered by name.

    python battery_subset.py ROOT OUT.json NAME [NAME ...]
"""
import sys

root, out, names = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, root + "/scripts")
import mutation_battery as mb  # noqa: E402

sel = tuple(m for m in mb.MUTATIONS if m.name in names)
missing = set(names) - {m.name for m in sel}
if missing or len(sel) != len(names):
    sys.exit(f"subset does not resolve: missing={sorted(missing)} n={len(sel)}")
mb.MUTATIONS = sel
sys.argv = ["mutation_battery.py", "--json", out]
mb.run_main(mb.main)
