"""The work queue's readiness rule, against real git objects in a tmp repo.

`scripts/orchestrator/workqueue.py` decides readiness from git objects at one base
commit so that no model ever does. Each test builds a throwaway repository under
`tmp_path` and points the module at it through `RSR_ORCH_ROOT`; nothing here reads
or writes the real checkout except the two seed-item tests at the end, which only
read.

Test names share prefixes (`test_owner_`, `test_prereg_cocommit`, `test_sprint_cap`,
`test_ruling_unsigned`, `test_claim_`) because `scripts/mutation_battery.py`
matches its gates by substring: each battery entry must redden its prefix and
nothing else.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import workqueue as wq  # noqa: E402

GIT = "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"

SIGNED = "---\nid: {id}\ndate: 2026-09-22\nstated_in: test session\n---\nRuled.\n"
PREREG_OK = (
    "---\nfalsifier: memory is inert\ndecision_rule: ratio > 1.5\n"
    "seeds: [0, 1, 2]\nthresholds: {ratio: 1.5}\n---\n# PREREG\n"
)


# --------------------------------------------------------------------------- #
# fixture repository
# --------------------------------------------------------------------------- #


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.commit({"README.md": "fixture\n"}, "init")

    def git(self, *args: str) -> str:
        p = subprocess.run(
            [
                GIT,
                "-C",
                str(self.root),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stderr
        return p.stdout.strip()

    def write(self, files: dict[str, str]) -> None:
        for rel, text in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)

    def commit(self, files: dict[str, str], msg: str = "c") -> str:
        self.write(files)
        self.git("add", *files)
        self.git("commit", "-q", "-m", msg)
        return self.git("rev-parse", "HEAD")

    def item(
        self,
        id: str,
        *,
        klass: str = "eng",
        sprint: int = 0,
        requires=(),
        status: str = "blocked",
        commit: bool = False,
        **extra,
    ) -> Path:
        meta = {
            "id": id,
            "class": klass,
            "sprint": sprint,
            "workstream": "W0",
            "lane": "none" if klass == "owner" else "agent-only",
            "slots": 0 if klass == "owner" else 1,
            "requires": list(requires),
            "status": status,
            "attempts": 0,
            "source": "tests/test_orch_workqueue.py:1",
        }
        if klass == "owner":
            meta["owner_question"] = "Decide X?"
            meta["answered_by"] = "R-*-x"
        meta.update(extra)
        rel = f"docs/queue/items/{id}.md"
        text = f"---\n{yaml.safe_dump(meta, sort_keys=False)}---\n# {id}\n\nprose body\n"
        (self.commit if commit else self.write)({rel: text})
        return self.root / rel


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Repo:
    r = Repo(tmp_path / "repo")
    monkeypatch.setenv("RSR_ORCH_ROOT", str(r.root))
    return r


def ready_ids(repo: Repo, base: str | None = None) -> set[str]:
    q = wq.Queue(repo.root, base)
    assert not q.validate(), q.validate()
    return {i.id for i in q.ready_items()}


def unmet(repo: Repo, id: str, base: str | None = None) -> list[str]:
    q = wq.Queue(repo.root, base)
    return q.readiness(q.by_id[id]).unmet


def ruling(id: str) -> dict[str, str]:
    return {f"docs/owner/rulings/{id}.md": SIGNED.format(id=id)}


# --------------------------------------------------------------------------- #
# readiness truth table (eng/exp)
# --------------------------------------------------------------------------- #

R = "R-2026-09-22-go"


def _setup_satisfied(repo: Repo, kind: str) -> dict[str, str]:
    if kind == "ruling":
        repo.commit(ruling(R))
        return {"ruling": R}
    if kind == "ruling_pattern":
        repo.commit(ruling("R-2026-09-30-go-later"))
        return {"ruling": "R-*-go-later"}
    if kind == "item_done":
        repo.item("dep", status="done")
        return {"item_done": "dep"}
    if kind == "sprint_gate":
        repo.commit(ruling("R-2026-09-22-sprint0-gate-green"))
        return {"sprint_gate": "S0"}
    if kind == "prereg":
        repo.commit({"experiments/x/PREREG.md": PREREG_OK})
        return {"prereg": "experiments/x/PREREG.md"}
    if kind == "path_exists_at_base":
        repo.commit({"src/thing.py": "x = 1\n"})
        return {"path_exists_at_base": "src/thing.py"}
    raise AssertionError(kind)


def _setup_unsatisfied(repo: Repo, kind: str) -> dict[str, str]:
    return {
        "ruling": {"ruling": R},
        "ruling_pattern": {"ruling": "R-*-go-later"},
        "item_done": (repo.item("dep", status="running"), {"item_done": "dep"})[1],
        "sprint_gate": {"sprint_gate": "S0"},
        "prereg": {"prereg": "experiments/x/PREREG.md"},
        "path_exists_at_base": {"path_exists_at_base": "src/thing.py"},
    }[kind]


KINDS = [
    "ruling",
    "ruling_pattern",
    "item_done",
    "sprint_gate",
    "prereg",
    "path_exists_at_base",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("satisfied", [True, False])
def test_truth_table_one_predicate(repo, kind, satisfied):
    pred = (_setup_satisfied if satisfied else _setup_unsatisfied)(repo, kind)
    klass = "exp" if kind == "prereg" else "eng"
    repo.item("it", klass=klass, requires=[pred])
    assert ("it" in ready_ids(repo)) is satisfied, unmet(repo, "it")


def test_truth_table_no_predicates_eng_is_ready(repo):
    repo.item("it")
    assert ready_ids(repo) == {"it"}


def test_truth_table_conjunction_needs_every_predicate(repo):
    repo.commit(ruling(R))
    repo.item("it", requires=[{"ruling": R}, {"path_exists_at_base": "nope.py"}])
    assert ready_ids(repo) == set()
    repo.commit({"nope.py": ""})
    assert ready_ids(repo) == {"it"}


@pytest.mark.parametrize(
    "status", ["claimed", "running", "collecting", "verifying", "done", "parked"]
)
def test_truth_table_only_blocked_or_ready_items_are_offered(repo, status):
    repo.item("it", status=status)
    assert ready_ids(repo) == set()


def test_truth_table_exp_item_reads_its_prereg_at_base_not_head(repo):
    base = repo.git("rev-parse", "HEAD")
    repo.commit({"experiments/x/PREREG.md": PREREG_OK})
    repo.item("it", klass="exp", requires=[{"prereg": "experiments/x/PREREG.md"}])
    assert ready_ids(repo, base) == set()
    assert ready_ids(repo) == {"it"}


# --------------------------------------------------------------------------- #
# owner items
# --------------------------------------------------------------------------- #


def test_owner_item_is_never_ready_even_when_answered(repo):
    repo.commit(ruling("R-2026-09-22-x"))
    repo.item("dec", klass="owner")
    q = wq.Queue(repo.root)
    assert q.answered(q.by_id["dec"])[0]
    assert ready_ids(repo) == set()


def test_owner_item_with_ready_status_is_still_not_ready(repo):
    repo.item("dec", klass="owner", status="ready")
    assert ready_ids(repo) == set()


def test_owner_item_cannot_be_closed_by_a_status_write(repo):
    repo.item("dec", klass="owner")
    for status in ("done", "claimed", "ready"):
        with pytest.raises(SystemExit) as e:
            wq.main(["set-status", "dec", status])
        assert e.value.code == 3


# --------------------------------------------------------------------------- #
# rulings
# --------------------------------------------------------------------------- #


def test_ruling_absent_does_not_satisfy(repo):
    repo.item("it", requires=[{"ruling": R}])
    assert ready_ids(repo) == set()
    assert "absent" in unmet(repo, "it")[0]


@pytest.mark.parametrize(
    "text",
    [
        "---\nid: R-2026-09-22-go\ndate: 2026-09-22\n---\nno stated_in\n",
        "---\nid: R-2026-09-22-go\nstated_in: s\n---\nno date\n",
        "---\nid: R-2026-09-22-other\ndate: 2026-09-22\nstated_in: s\n---\nwrong id\n",
        "no front matter at all\n",
    ],
)
def test_ruling_unsigned_does_not_satisfy(repo, text):
    repo.commit({f"docs/owner/rulings/{R}.md": text})
    repo.item("it", requires=[{"ruling": R}])
    assert ready_ids(repo) == set()


def test_ruling_unsigned_pattern_match_does_not_satisfy(repo):
    name = "R-2026-09-22-sprint0-gate"
    text = f"---\nid: {name}\ndate: 2026-09-22\n---\n"
    repo.commit({f"docs/owner/rulings/{name}.md": text})
    repo.item("it", requires=[{"sprint_gate": "S0"}])
    assert ready_ids(repo) == set()


def test_ruling_in_working_tree_only_does_not_satisfy(repo):
    repo.write(ruling(R))  # written, never committed: not at base
    repo.item("it", requires=[{"ruling": R}])
    assert ready_ids(repo) == set()


def test_ruling_pattern_star_is_one_date_only(repo):
    repo.commit(ruling("R-2026-09-22-not-go"))
    repo.item("it", requires=[{"ruling": "R-*-go"}])
    assert ready_ids(repo) == set()


def test_sprint_gate_needs_the_named_sprint(repo):
    repo.commit(ruling("R-2026-09-22-sprint1-gate"))
    repo.item("it", requires=[{"sprint_gate": "S0"}])
    assert ready_ids(repo) == set()
    repo.commit(ruling("R-2026-09-23-sprint0-gate"))
    assert ready_ids(repo) == {"it"}


def test_sprint_gate_is_not_satisfied_by_an_ordinary_ruling(repo):
    repo.commit(ruling("R-2026-09-22-sprint0-start"))
    repo.item("it", requires=[{"sprint_gate": "S0"}])
    assert ready_ids(repo) == set()


# --------------------------------------------------------------------------- #
# pre-registration
# --------------------------------------------------------------------------- #

P = "experiments/x/PREREG.md"


def test_prereg_cocommitted_with_code_does_not_satisfy(repo):
    repo.commit({P: PREREG_OK, "experiments/x/run.py": "print(1)\n"})
    repo.item("it", klass="exp", requires=[{"prereg": P}])
    assert ready_ids(repo) == set()
    assert "own commit" in unmet(repo, "it")[0]


def test_prereg_cocommitted_then_amended_alone_still_does_not_satisfy(repo):
    # The introducing commit decides; a later solo edit does not launder it.
    repo.commit({P: PREREG_OK, "experiments/x/run.py": "print(1)\n"})
    repo.commit({P: PREREG_OK + "\n## Amendment\n"})
    repo.item("it", klass="exp", requires=[{"prereg": P}])
    assert ready_ids(repo) == set()


def test_prereg_in_its_own_commit_satisfies(repo):
    repo.commit({P: PREREG_OK})
    repo.commit({"experiments/x/run.py": "print(1)\n"})
    repo.item("it", klass="exp", requires=[{"prereg": P}])
    assert ready_ids(repo) == {"it"}


def test_prereg_uncommitted_does_not_satisfy(repo):
    repo.write({P: PREREG_OK})
    repo.item("it", klass="exp", requires=[{"prereg": P}])
    assert ready_ids(repo) == set()


@pytest.mark.parametrize("drop", wq.PREREG_FIELDS)
def test_prereg_missing_front_matter_field_does_not_satisfy(repo, drop):
    meta = yaml.safe_load(PREREG_OK.split("---\n")[1])
    del meta[drop]
    repo.commit({P: f"---\n{yaml.safe_dump(meta)}---\n# PREREG\n"})
    repo.item("it", klass="exp", requires=[{"prereg": P}])
    assert ready_ids(repo) == set()


def test_prereg_is_required_of_every_exp_item(repo):
    repo.item("it", klass="exp", requires=[])
    assert "it.md" in wq.Queue(repo.root).validate()


# --------------------------------------------------------------------------- #
# validation and the §16 sprint cap
# --------------------------------------------------------------------------- #


def test_sprint_cap_validate_fails_on_sprint_5(repo, capsys):
    repo.item("late", sprint=5)
    bad = wq.Queue(repo.root).validate()
    assert "late.md" in bad and any("§16" in e for e in bad["late.md"])
    assert wq.main(["validate"]) == wq.Exit.FAIL


def test_sprint_cap_ready_refuses_to_run_with_a_sprint_5_item(repo):
    repo.item("ok")
    repo.item("late", sprint=5)
    with pytest.raises(SystemExit) as e:
        wq.main(["ready"])
    assert e.value.code == 3


def test_sprint_cap_sprint_gate_s5_is_invalid(repo):
    repo.item("it", requires=[{"sprint_gate": "S5"}])
    assert "it.md" in wq.Queue(repo.root).validate()


def test_validate_refuses_a_misspelt_requires(repo):
    repo.item("it", requires=[], requirs=[{"ruling": R}])
    assert "it.md" in wq.Queue(repo.root).validate()


def test_validate_refuses_an_unknown_predicate_and_a_dangling_item_done(repo):
    repo.item("a", requires=[{"maybe": "x"}])
    repo.item("b", requires=[{"item_done": "ghost"}])
    bad = wq.Queue(repo.root).validate()
    assert {"a.md", "b.md"} <= set(bad)


def test_validate_refuses_an_owner_item_without_a_question(repo):
    repo.item("dec", klass="owner", owner_question="")
    assert "dec.md" in wq.Queue(repo.root).validate()


def test_empty_queue_is_nothing_to_compare(repo):
    assert wq.main(["validate"]) == wq.Exit.UNKNOWN
    assert wq.main(["ready"]) == wq.Exit.UNKNOWN


# --------------------------------------------------------------------------- #
# writers and outputs
# --------------------------------------------------------------------------- #


def test_set_status_attempt_increments_and_keeps_the_prose(repo):
    p = repo.item("it")
    assert wq.main(["set-status", "it", "running", "--attempt"]) == wq.Exit.OK
    assert wq.main(["set-status", "it", "blocked", "--attempt"]) == wq.Exit.OK
    meta, body = wq.split_front_matter(p.read_text())
    assert meta["status"] == "blocked" and meta["attempts"] == 2
    assert body == "# it\n\nprose body\n"


def test_claim_of_a_blocked_item_is_refused(repo):
    repo.item("it", requires=[{"ruling": R}])
    with pytest.raises(SystemExit) as e:
        wq.main(["set-status", "it", "claimed"])
    assert e.value.code == 3


def test_claim_of_a_ready_item_is_allowed(repo):
    repo.commit(ruling(R))
    repo.item("it", requires=[{"ruling": R}])
    assert wq.main(["set-status", "it", "claimed"]) == wq.Exit.OK


def test_base_defaults_to_night_json(repo):
    base = repo.git("rev-parse", "HEAD")
    repo.commit(ruling(R))
    repo.item("it", requires=[{"ruling": R}])
    night = repo.root / ".orchestrator" / "state" / "night.json"
    night.parent.mkdir(parents=True)
    night.write_text(json.dumps({"date": "2026-09-22", "base_sha": base}))
    assert ready_ids(repo) == set()
    night.write_text(json.dumps({"base_sha": repo.git("rev-parse", "HEAD")}))
    assert ready_ids(repo) == {"it"}


def test_an_unresolvable_base_is_refused(repo):
    repo.item("it")
    with pytest.raises(SystemExit) as e:
        wq.main(["ready", "--base", "deadbeef" * 5])
    assert e.value.code == 3


def test_render_writes_queue_json(repo):
    repo.item("a")
    repo.item("b", requires=[{"ruling": R}])
    assert wq.main(["render"]) == wq.Exit.OK
    doc = json.loads((repo.root / ".orchestrator" / "queue.json").read_text())
    by = {r["id"]: r for r in doc["items"]}
    assert by["a"]["ready"] and not by["b"]["ready"] and by["b"]["unmet"]
    assert doc["base_sha"] == repo.git("rev-parse", "HEAD")


def test_ready_json_lists_only_ready_items(repo, capsys):
    repo.item("a")
    repo.item("b", requires=[{"ruling": R}])
    assert wq.main(["ready", "--json"]) == wq.Exit.OK
    doc = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in doc["ready"]] == ["a"]


def test_digest_lists_open_owner_items_and_unmet_predicates(repo, capsys):
    repo.item("dec", klass="owner")
    repo.item("b", requires=[{"ruling": R}])
    assert wq.main(["digest"]) == wq.Exit.OK
    out = capsys.readouterr().out
    assert "**dec**" in out and "Decide X?" in out
    assert "**b**" in out and f"ruling {R}: absent" in out
    repo.commit(ruling("R-2026-09-22-x"))
    wq.main(["digest"])
    assert "**dec**" not in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the committed seed items (read-only against the real checkout)
# --------------------------------------------------------------------------- #


def test_seed_items_validate(monkeypatch):
    monkeypatch.setenv("RSR_ORCH_ROOT", str(_REPO))
    q = wq.Queue(_REPO)
    assert len(q.items) >= 40
    assert q.validate() == {}


def test_seed_item_sources_resolve_to_real_lines(monkeypatch):
    """Every `path:N` citation in a seed item names a file with at least N lines."""
    q = wq.Queue(_REPO)
    cites = 0
    for it in q.items:
        for m in re.finditer(r"([\w./-]+\.(?:md|py)):(\d+)", it.meta["source"]):
            path, line = m.group(1), int(m.group(2))
            if path.startswith("origin/"):
                continue
            f = _REPO / path
            assert f.exists(), f"{it.id}: {path} does not exist"
            assert len(f.read_text().splitlines()) >= line, f"{it.id}: {path}:{line}"
            cites += 1
    assert cites >= 30


# --------------------------------------------------------------------------- #
# an unreadable ruling is loud, never a silent "unsigned" (2026-09-22)
# --------------------------------------------------------------------------- #

# The 09-22 defect, verbatim in shape: an unquoted `: ` inside stated_in.
UNPARSEABLE = (
    "---\nid: R-2026-09-22-go\ndate: 2026-09-22\n"
    'stated_in: session -- Brendan: "go"\n---\nRuled.\n'
)


def test_unreadable_ruling_names_the_yaml_error():
    why = wq.ruling_problem("R-2026-09-22-go.md", UNPARSEABLE)
    assert why is not None and "not valid YAML" in why


def test_unreadable_ruling_quoted_colon_signs():
    quoted = UNPARSEABLE.replace(
        'stated_in: session -- Brendan: "go"', "stated_in: 'session -- Brendan: \"go\"'"
    )
    assert wq.ruling_problem("R-2026-09-22-go.md", quoted) is None


def test_unreadable_ruling_at_base_stops_ready(repo, capsys):
    repo.commit({f"docs/owner/rulings/{R}.md": UNPARSEABLE})
    repo.item("ok")
    with pytest.raises(SystemExit) as e:
        wq.main(["ready"])
    assert e.value.code == 3
    assert f"UNREADABLE RULING {R}.md" in capsys.readouterr().err


def test_unreadable_ruling_in_tree_fails_validate(repo, capsys):
    repo.write({f"docs/owner/rulings/{R}.md": UNPARSEABLE})
    repo.item("ok")
    assert wq.main(["validate"]) == wq.Exit.FAIL
    assert f"INVALID RULING {R}.md" in capsys.readouterr().out
