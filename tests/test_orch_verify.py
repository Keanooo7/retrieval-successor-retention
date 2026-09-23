"""`orchestrator.verify`: the seeded claim draw and the verification record.

The verifier does not choose which claims to re-execute; `sha256(run_id)` does, so
anyone can recompute which were due. A draw that cannot be recomputed is a choice.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from orchestrator import verify  # noqa: E402


def _claims(n: int) -> list[dict]:
    return [
        {"claim": f"c{i}", "command": f"cmd {i}", "expected": str(i)} for i in range(n)
    ]


def test_draw_is_seeded_by_run_id():
    seed = hashlib.sha256(b"decisive-shuffle").hexdigest()
    assert verify.seed_for("decisive-shuffle") == seed
    want = sorted(random.Random(int(seed, 16)).sample(range(10), 3))
    assert verify.draw_indices("decisive-shuffle", 10) == want
    draws = {tuple(verify.draw_indices(f"run-{i}", 10)) for i in range(20)}
    assert len(draws) > 1  # the run id moves the draw


def test_draw_is_stable_and_bounded():
    a = verify.draw_indices("r1", 7, 3)
    assert a == verify.draw_indices("r1", 7, 3)
    assert len(a) == 3 and len(set(a)) == 3 and all(0 <= i < 7 for i in a)
    assert verify.draw_indices("r1", 2, 3) == [0, 1]


def _record(run_id="r1", status="ok", matches=(True,), claims=None) -> dict:
    claims = claims if claims is not None else _claims(len(matches))
    return {
        "status": status,
        "seed": verify.seed_for(run_id),
        "claims": [
            {**c, "observed": c["expected"], "match": m}
            for c, m in zip(claims, matches, strict=True)
        ],
        "verifier_sha": "0123456789abcdef0123456789abcdef01234567",
    }


def test_a_valid_record():
    assert verify.validate_record(_record(), "r1") == []
    assert (
        verify.validate_record(_record(status="failed", matches=(True, False)), "r1")
        == []
    )
    assert verify.validate_record(_record(status="inconclusive", matches=()), "r1") == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda r: r.update(status="passed"), "status"),
        (lambda r: r.update(seed="0" * 64), "seed"),
        (lambda r: r.update(verifier_sha="HEAD"), "verifier_sha"),
        (lambda r: r["claims"][0].pop("observed"), "lacks"),
        (lambda r: r["claims"][0].update(match="yes"), "bool"),
        (lambda r: r["claims"][0].update(match=False), "status ok"),
        (lambda r: r.update(claims=[]), "status ok"),
        (lambda r: r.update(status="failed"), "status failed"),
    ],
)
def test_an_invalid_record_names_its_defect(mutate, needle):
    rec = _record()
    mutate(rec)
    problems = verify.validate_record(rec, "r1")
    assert any(needle in p for p in problems), problems


def test_check_refuses_claims_other_than_the_drawn_ones(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    claims = _claims(6)
    (run / "claims.json").write_text(json.dumps(claims))
    drawn = [claims[i] for i in verify.draw_indices("r1", 6)]
    (run / "verification.json").write_text(
        json.dumps(_record(matches=(True,) * 3, claims=drawn))
    )
    assert int(verify.cmd_check(tmp_path, "r1", 3)) == 0
    others = [c for c in claims if c not in drawn][:3]
    (run / "verification.json").write_text(
        json.dumps(_record(matches=(True,) * 3, claims=others))
    )
    assert int(verify.cmd_check(tmp_path, "r1", 3)) == 1


def test_draw_exit_codes(tmp_path, capsys):
    assert int(verify.cmd_draw(tmp_path, "r1", 3)) == 3  # no claims list
    (tmp_path / "runs/r1").mkdir(parents=True)
    (tmp_path / "runs/r1/claims.json").write_text("[]")
    assert int(verify.cmd_draw(tmp_path, "r1", 3)) == 2  # nothing to compare
    (tmp_path / "runs/r1/claims.json").write_text(json.dumps(_claims(5)))
    capsys.readouterr()
    assert int(verify.cmd_draw(tmp_path, "r1", 3)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["seed"] == verify.seed_for("r1")
    assert [c["index"] for c in out["chosen"]] == verify.draw_indices("r1", 5)
