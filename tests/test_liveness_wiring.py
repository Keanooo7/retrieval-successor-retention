"""liveness-wiring -- every training run measures its own memory's liveness.

Brief: `docs/lab-notes/dispatch-liveness-wiring.md`. Rulings (read only):
`R-2026-09-22-inert-exit-5`, `R-2026-09-22-inert-checkpoint-quarantine`,
`R-2026-09-22-liveness-go`.

Falsifier: *"A training run whose memory is inert is indistinguishable at exit from
one whose memory is live."* Before this, liveness was measured only by a separate
script after training, so an inert run exited `0` and its checkpoint was consumed
like any other.

The decision rule is `experiments/decisive-shuffle/PREREG.md:33-45` (#17's
Amendment 1), applied to one run (one seed), and the exit mapping is the ruling's:

====================================================  =====================
condition                                             exit
====================================================  =====================
a control fails, `A_decoy == 0`, `ratio == 1.0`       `3` DID_NOT_RUN
exactly, or the measurement raises
`ratio >= 0.1`                                        `0` OK (live)
`ratio <= 0.01`                                       `5` INERT
`0.01 < ratio < 0.1` (inconclusive)                   `1` FAIL
====================================================  =====================

The rule tests below feed hand-built measurements to the band function; the
wiring tests train a real CPU model (the S0-01 `TINY` shape) and read what
`train()` returned and what its heartbeat footer holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import rsr.train.loop as loop
from rsr.exit_codes import Exit
from rsr.metrics import memory_liveness as ml
from rsr.model.tg import TGConfig, TGModel
from rsr.train import checkpoint as ck

TINY = dict(
    d=32,
    steps_per_stream=48,
    batch=2,
    max_tokens=16,
    memory_slots=4,
    iters=1,
    device="cpu",
    beat_every=1,
    ckpt_every=0,
)

#: The bar's keys: ratio, A_trained, A_decoy, the three controls, the cross-row
#: cosine, the matched-norm random-replacement ratio, and the band label.
LIVENESS_KEYS = {
    "ratio",
    "A_trained",
    "A_decoy",
    "control_memory_disabled_ok",
    "control_own_memory_ok",
    "control_live_decoy_ok",
    "cross_row_cosine",
    "random_replacement_ratio",
    "band",
    "exit_code",
}


def _m(ratio: float | None, *, a_decoy: float = 1e-2, **controls) -> dict:
    """A hand-built measurement in `measure_liveness`'s shape."""
    a_t = None if ratio is None else ratio * a_decoy
    m = {
        "A_trained": a_t,
        "A_decoy": a_decoy,
        "ratio": (a_t / a_decoy) if (a_t is not None and a_decoy) else None,
        "control_memory_disabled_ok": True,
        "control_own_memory_ok": True,
        "control_live_decoy_ok": a_decoy > 0,
    }
    m.update(controls)
    return m


# --------------------------------------------------------------------------- #
# the rule -- one test per row of the table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("r", [0.1, 0.5, 0.99])
def test_a_live_ratio_is_live_and_exits_0(r):
    band, _ = ml.liveness_band(_m(r))
    assert band == "live"
    assert loop.liveness_exit({"band": band}) is Exit.OK


@pytest.mark.parametrize("r", [0.0, 0.005, 0.01])
def test_an_inert_ratio_is_inert_and_exits_5(r):
    """`ratio <= 0.01` -- the boundary is inert (PREREG `:36`), and the code is
    `5`, never `1` (`R-2026-09-22-inert-exit-5`)."""
    band, _ = ml.liveness_band(_m(r))
    assert band == "inert"
    assert loop.liveness_exit({"band": band}) is Exit.INERT


@pytest.mark.parametrize("r", [0.0101, 0.05, 0.0999])
def test_an_inconclusive_ratio_exits_1_never_0_or_5(r):
    """PREREG `:45`: the band between is "reported as such, not rounded to either
    outcome". It is `1` because live memory was not demonstrated."""
    band, _ = ml.liveness_band(_m(r))
    assert band == "inconclusive"
    assert loop.liveness_exit({"band": band}) is Exit.FAIL


@pytest.mark.parametrize(
    "m",
    [
        _m(0.5, control_memory_disabled_ok=False),
        _m(0.5, control_own_memory_ok=False),
        _m(None, a_decoy=0.0),
        _m(1.0),
        _m(float("nan")),
    ],
    ids=["memory-disabled", "own-memory", "decoy-zero", "ratio-one", "nan"],
)
def test_an_invalid_measurement_exits_3(m):
    """A control failed, the decoy read zero, or `ratio == 1.0` exactly (decoy
    aliasing, PREREG `:35`, `:61`): the measurement is not valid, so the run did
    not demonstrate anything -- `3`, never `0`, `1` or `5`."""
    band, why = ml.liveness_band(m)
    assert band == "invalid", why
    assert loop.liveness_exit({"band": band}) is Exit.DID_NOT_RUN


def test_a_missing_or_unknown_liveness_exits_3():
    """A train() that returned no measurement did not measure: `3`, not `0`."""
    assert loop.liveness_exit(None) is Exit.DID_NOT_RUN
    assert loop.liveness_exit({}) is Exit.DID_NOT_RUN
    assert loop.liveness_exit({"band": "sort of live"}) is Exit.DID_NOT_RUN


def test_the_thresholds_are_amendment_1s():
    """Transcribed, not chosen: `ba71e86`, decisive PREREG `:36-37`."""
    assert ml.INERT_MAX_RATIO == 0.01
    assert ml.LIVE_MIN_RATIO == 0.1


# --------------------------------------------------------------------------- #
# main() exits through the protocol on the band
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("band", "code"),
    [
        ("live", Exit.OK),
        ("inert", Exit.INERT),
        ("inconclusive", Exit.FAIL),
        ("invalid", Exit.DID_NOT_RUN),
    ],
)
def test_main_exits_on_the_liveness_band(monkeypatch, tmp_path, band, code):
    monkeypatch.setattr(
        "rsr.train.loop.train",
        lambda **kw: {"run_id": "x", "liveness": {"band": band}},
    )
    assert loop.main(["--out-dir", str(tmp_path), "--device", "cpu"]) is code


# --------------------------------------------------------------------------- #
# the wiring -- a real CPU training run
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("live")
    return loop.train(out_dir=out, seed=0, **TINY), out


def _footer(out: Path) -> dict:
    rows = [json.loads(x) for x in (out / "heartbeat.jsonl").read_text().splitlines()]
    return next(r for r in rows if r.get("kind") == "footer")


def test_every_train_ends_with_the_liveness_measurement(tiny_run):
    """Bar 2: the returned dict and the heartbeat footer both carry it."""
    r, out = tiny_run
    assert set(r["liveness"]) >= LIVENESS_KEYS, set(r["liveness"])
    foot = _footer(out)
    assert set(foot["liveness"]) >= LIVENESS_KEYS, foot
    assert foot["liveness"]["band"] == r["liveness"]["band"]
    assert foot["liveness"]["exit_code"] == int(loop.liveness_exit(r["liveness"]))


def test_the_controls_read_what_they_must_on_a_real_run(tiny_run):
    """memory-disabled and own-memory read exactly 0.0; the live decoy reads > 0."""
    lv = tiny_run[0]["liveness"]
    assert lv["control_memory_disabled_ok"] is True
    assert lv["control_own_memory_ok"] is True
    assert lv["control_live_decoy_ok"] is True
    assert lv["A_decoy"] > 0
    assert lv["random_self_ok"] is True


def test_the_decoy_is_the_untrained_model_at_the_same_seed(tiny_run):
    """The decoy is the model's own init (decisive PREREG *Decoy*), recomputed
    here independently. Pointed at the trained model instead, `A_decoy` would
    equal `A_trained` and the ratio would read 1.0 exactly."""
    r, _ = tiny_run
    lv = r["liveness"]
    torch.manual_seed(r["config"]["seed"])
    fresh = TGModel(TGConfig(**r["config"]["tg"]))
    ids, mask = loop.liveness_batch(r["config"])
    want = ml.shuffle_control(fresh, ids, mask)["mean_abs_token_delta"]
    assert lv["A_decoy"] == want
    assert lv["ratio"] != 1.0


def test_a_decoy_aliased_to_the_trained_model_is_invalid(tiny_run):
    """The aliasing control itself, through the real instrument: decoy = trained
    gives `ratio == 1.0` exactly, and the rule makes that `3`."""
    r, _ = tiny_run
    torch.manual_seed(0)
    model = TGModel(TGConfig(**r["config"]["tg"]))
    ids, mask = loop.liveness_batch(r["config"])
    m = ml.measure_liveness(model, model, ids, mask, seed=0)
    assert m["ratio"] == 1.0
    assert m["band"] == "invalid"
    assert loop.liveness_exit(m) is Exit.DID_NOT_RUN


def test_a_live_run_is_not_quarantined(tmp_path, monkeypatch):
    """Only INERT quarantines: a live run's checkpoints stay where they were."""
    monkeypatch.setattr(ml, "liveness_band", lambda m: ("live", "forced"))
    out = tmp_path / "live"
    loop.train(out_dir=out, seed=0, **{**TINY, "ckpt_every": 1})
    assert (out / "ckpt-000001.pt").exists()
    assert not (out / ck.QUARANTINE_DIR).exists()
    ck.load(out / "ckpt-000001.pt", model=None, restore_rng=False)


# --------------------------------------------------------------------------- #
# quarantine -- R-2026-09-22-inert-checkpoint-quarantine
# --------------------------------------------------------------------------- #


@pytest.fixture
def inert_run(tmp_path, monkeypatch):
    """A real training run whose band is forced to inert. The band function is
    the one seam: everything after it -- quarantine, the footer, the returned
    exit -- is the code under test."""
    monkeypatch.setattr(ml, "liveness_band", lambda m: ("inert", "forced"))
    out = tmp_path / "inert"
    r = loop.train(out_dir=out, seed=0, **{**TINY, "ckpt_every": 1})
    return r, out


def test_an_inert_run_quarantines_its_checkpoints(inert_run):
    r, out = inert_run
    q = out / ck.QUARANTINE_DIR
    assert (q / "ckpt-000001.pt").exists()
    assert (q / ck.INERT_MARKER).exists()
    assert not list(out.glob("ckpt-*.pt")), "an inert checkpoint left in place"
    assert r["liveness"]["band"] == "inert"
    assert r["liveness"]["exit_code"] == int(Exit.INERT)
    assert r["liveness"]["quarantine"] == str(q)


def test_the_loader_refuses_a_quarantined_checkpoint(inert_run):
    _, out = inert_run
    path = out / ck.QUARANTINE_DIR / "ckpt-000001.pt"
    with pytest.raises(ck.QuarantinedCheckpoint, match="INERT"):
        ck.load(path, restore_rng=False)
    ck.load(path, restore_rng=False, allow_quarantined=True)


def test_the_loader_refuses_a_checkpoint_beside_an_inert_marker(tmp_path):
    """Moved out of `quarantine/` but still beside its marker: still refused."""
    path = ck.save(tmp_path / "c.pt", step=0, model=torch.nn.Linear(2, 2))
    ck.load(path, restore_rng=False)
    (tmp_path / ck.INERT_MARKER).write_text("{}")
    with pytest.raises(ck.QuarantinedCheckpoint):
        ck.load(path, restore_rng=False)
