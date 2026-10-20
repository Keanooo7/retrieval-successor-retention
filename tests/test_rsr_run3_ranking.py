"""RSR run 3, the ranking-loss test (PREREG rsr-run3-ranking @ 32475b5, §3 T1-T7).

The load-bearing ones:

* **T1** the switch off (``loss="mc"``) is run 2: (a) ``retention_loss`` bitwise equal
  to the code at run 2's launch head ``728a42f`` on the same rows, both RSR arms, loss
  and gradient; (b) run 2's psiC seed-0 command for 20 steps at ``728a42f`` and here
  give identical heartbeats (apart from run 2's ``VOLATILE_KEYS``), diagnostics,
  eviction records, checkpoint tensors and run_id (slow: two real 20-step runs);
  (c) ``reduction_to_tg()`` is unchanged; (d) 2a's fit path with ``loss="mc"``
  reproduces offline-psi's ``fit`` on a 2-document fixture (R-C2's pattern);
* **T2** isolation under ``loss="rank"``: no transformer parameter (``W_sent`` among
  them) gets a gradient, both RSR arms; the optimizer holds only phi;
* **T3** no age: the content head takes none; permuting ``written_at`` moves nothing in
  the content arm's rank loss (and does move the age arm's: the test can see it);
* **T4** the loss on hand-computed fixtures (ties at and away from the k-th value,
  ``n <= k``, a group with no pair, shift invariance, row-order invariance, omega = 1);
* **T5** a planted bilinear G: the fit drives J down and the argmin agreement to 1;
* **T6** the stage seals, each in its own process (the seal is process-wide and one
  stage per process): sealed ranges raise, opened ones pass (to a stub generator, so no
  sealed document is ever generated);
* **T7** the rank arms' config and labels.

The sealed bounds are read from the phase-1 PREREG's front matter, never typed here.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsr.data.synthetic import SyntheticConfig, generate  # noqa: E402
from rsr.metrics import loo  # noqa: E402
from rsr.model.tg import TGConfig, TGModel  # noqa: E402
from rsr.model.tg.policy_loop import run_policy_loop  # noqa: E402
from rsr.retention import loo_target as LT  # noqa: E402
from rsr.retention import rank_loss as RL  # noqa: E402
from rsr.retention import rsr as RSR  # noqa: E402
from rsr.retention.value_head import BilinearValueHead  # noqa: E402
from rsr.train import loop  # noqa: E402

GIT = "/opt/homebrew/bin/git"
RUN2_HEAD = "728a42f"
EXP = ROOT / "experiments" / "rsr-run3-ranking"
PY = ROOT / ".venv" / "bin" / "python"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p2a():
    """phase2a, with its harness supplied directly: a stage seal is process-wide and
    one stage per process, so the test process never installs one (T6 runs each
    stage in a subprocess)."""
    mod = _load("run3_phase2a_test", EXP / "phase2a.py")
    if not mod._H:
        cv = _load("_run3_cv_test", ROOT / "experiments/offline-psi-converge/run.py")
        mod._H.update(CV=cv, OP=cv.OP, R1=cv.OP.R1, R2=cv.OP.R2)
    return mod


# --------------------------------------------------------------------------- #
# a tiny world (test_loo_target.py's shape)
# --------------------------------------------------------------------------- #

S, L, M, D = 16, 10, 4, 16


def _docs(n=8, seed=0):
    return generate(
        SyntheticConfig(
            n_documents=n,
            sentences_per_document=S,
            max_gap=12,
            heavy_tail_min=6,
            seed=seed,
        )
    )


def _world(n=6, seed=0):
    docs = _docs(n, seed)
    vmap = loop.build_vocab(_docs(64, seed))
    ids, mask = loop.encode(docs, vmap, max_tokens=L, steps=S)
    torch.manual_seed(0)
    cfg = TGConfig(
        D=D,
        H=2,
        N=4,
        V=4 + len(vmap),
        block_config=("S", "C", "S", "C"),
        srep_extraction_layer=2,
        max_sentence_tokens=L,
        max_sentences_in_short_term=M,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        eod_id=3,
    )
    return TGModel(cfg), docs, ids, mask


def _kw(psi=None, **extra) -> dict:
    return dict(
        nu=0.0,
        beta=1.0,
        gamma=0.9,
        t_warm=0.0,
        a_max=M,
        psi_override=psi,
        b_enabled=False,
        shadow_enabled=False,
        target=LT.TARGET_LOO_RESAMPLE,
        **extra,
    )


def _lm(t, out, ids, mask, row_valid):
    return loop.lm_loss(out.logits, ids, mask)


def _drive(mod, cfg, model, docs, ids, mask):
    pol = mod.RSRPolicy(cfg, d_model=D, generator=torch.Generator().manual_seed(7))
    pol.set_train_step(0)
    run_policy_loop(
        model,
        ids,
        mask,
        torch.full((ids.shape[0],), S),
        pol,
        step_fn=_lm,
        observe=True,
        eval_capture=True,
        loo=LT.LooTarget(ann=loo.stream_annotations(docs, steps=S), seed=123),
    )
    return pol


@pytest.fixture(scope="module")
def rsr_728a42f(tmp_path_factory):
    """``src/rsr/retention/rsr.py`` as it was at run 2's launch head, loaded as its own
    module (its imports resolve to this tree; every module it imports is unchanged
    since 728a42f -- asserted)."""
    d = tmp_path_factory.mktemp("rsr728")
    r = subprocess.run(
        [GIT, "show", f"{RUN2_HEAD}:src/rsr/retention/rsr.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    for dep in (
        "mc.py",
        "value_head.py",
        "targets.py",
        "instrumentation.py",
        "policy.py",
        "reward.py",
    ):
        q = subprocess.run(
            [GIT, "diff", "--quiet", RUN2_HEAD, "--", f"src/rsr/retention/{dep}"],
            cwd=ROOT,
        )
        assert q.returncode == 0, f"{dep} changed since {RUN2_HEAD}"
    p = d / "rsr_728a42f.py"
    p.write_text(r.stdout)
    return _load("rsr_728a42f", p)


# --------------------------------------------------------------------------- #
# T1 the switch off is run 2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("psi", [None, RSR.AGE_ONLY])
def test_t1a_mc_retention_loss_is_bitwise_run_2s(rsr_728a42f, psi):
    model, docs, ids, mask = _world()
    model.eval()
    new = _drive(RSR, RSR.RSRConfig(**_kw(psi)), model, docs, ids, mask)
    old = _drive(rsr_728a42f, rsr_728a42f.RSRConfig(**_kw(psi)), model, docs, ids, mask)
    assert new.config.loss == "mc" and new.config.rank_k is None
    # the same stream: same buffers, same decisions
    assert sorted(new._mc) == sorted(old._mc)
    for r in new._mc:
        for a, b in zip(new._mc[r], old._mc[r], strict=True):
            assert a.step == b.step
            for f in ("keys", "gestalts", "context", "r"):
                assert torch.equal(getattr(a, f), getattr(b, f))
    ln, lo = new.retention_loss(), old.retention_loss()
    assert torch.equal(ln.detach(), lo.detach())
    ln.backward()
    lo.backward()
    for pn, po in zip(new.head.parameters(), old.head.parameters(), strict=True):
        assert torch.equal(pn.grad, po.grad)
    assert json.dumps(new.last_mc_stats, sort_keys=True, default=str) == json.dumps(
        old.last_mc_stats, sort_keys=True, default=str
    )


def test_t1c_reduction_to_tg_is_unchanged(rsr_728a42f):
    new = dataclasses.asdict(RSR.RSRConfig.reduction_to_tg(16))
    old = dataclasses.asdict(rsr_728a42f.RSRConfig.reduction_to_tg(16))
    added = set(new) - set(old)
    assert added == {"loss", "rank_k", "rank_omega"}
    assert {k: new[k] for k in old} == old
    assert (new["loss"], new["rank_k"], new["rank_omega"]) == ("mc", None, None)
    assert set(RSR.REDUCTION_SWITCHES) == {
        f.name for f in dataclasses.fields(RSR.RSRConfig)
    }
    assert RSR.RSRConfig.reduction_to_tg(16).is_reduction(16)
    # the default config is the off switch
    cfg = RSR.RSRConfig(**_kw())
    assert (cfg.loss, cfg.rank_k, cfg.rank_omega) == ("mc", None, None)


def test_t1c_test_reduction_is_in_the_suite():
    """T1(c)'s other half: tests/test_reduction.py exists and is collected by the
    full suite (it runs there, at the launch head)."""
    assert (ROOT / "tests" / "test_reduction.py").exists()


def test_t1d_fit_path_mc_is_offline_psis_fit_on_a_two_document_fixture(p2a):
    OP = p2a._H["OP"]
    g = torch.Generator().manual_seed(5)
    n, d = 12, 8
    data = {
        "s": torch.randn(2, n, d, generator=g),
        "c": torch.randn(2, n, d, generator=g),
        "G": torch.randn(2, n, n, generator=g),
        "mask": torch.ones(n, n, dtype=torch.bool).tril(-1).expand(2, n, n).clone(),
    }
    states = []
    vals = []
    for which in ("offline-psi", "2a"):
        head = BilinearValueHead(d, generator=torch.Generator().manual_seed(1))
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0.01)
        if which == "offline-psi":
            OP.fit(head, opt, data, [0, 1], [0, 1], seed=3, steps=40, every=10)
        else:
            p2a.fit_loop(
                head,
                opt,
                data,
                [0, 1],
                seed=3,
                loss="mc",
                omega=None,
                cap=40,
                every=None,
                save_steps=(),
                stop_on_convergence=False,
            )
        states.append({k: v.clone() for k, v in head.state_dict().items()})
        vals.append(
            OP.val_loss(head, data["s"], data["c"], data["G"], data["mask"], [0, 1])
        )
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
    assert vals[0] == vals[1]


@pytest.mark.slow
def test_t1b_twenty_steps_end_to_end_identical_to_728a42f(tmp_path):
    """Run 2's psiC seed-0 command (run 2's ``cli_argv``) for 20 steps, at 728a42f
    (its ``src`` archived into tmp) and at this head: identical heartbeats apart from
    ``VOLATILE_KEYS`` and the header's git sha, diagnostics, eviction records,
    checkpoint tensors and run_id. Both run concurrently, 1 thread each."""
    old_src = tmp_path / "src728"
    old_src.mkdir()
    ar = subprocess.run(
        f"{GIT} archive {RUN2_HEAD} src | tar -x -C {old_src}",
        shell=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert ar.returncode == 0, ar.stderr
    # run 2's module wraps its private run-1 instance's cli_argv at import; a second
    # run-2 instance in this process would reuse a wrapped one (phase 1's launch-attempt
    # lesson), so load ours against a fresh run 1 and restore sys.modules after.
    saved = sys.modules.pop("_rsr_run2_r1", None)
    try:
        R2 = _load("_run3_t1b_r2", ROOT / "experiments/rsr-run2/run.py")
    finally:
        if saved is not None:
            sys.modules["_rsr_run2_r1"] = saved
        else:
            sys.modules.pop("_rsr_run2_r1", None)
    R1 = R2.R1
    probe = R2.cli_argv(
        R1.prereg_config(0, 20),
        "psiC",
        R1.OwnerValues(0.9, 0.01, 20, "eval", (0,)),
        tmp_path,
    )
    assert probe.count("target=loo_resample") == 1
    owner = R1.OwnerValues(
        gamma=0.9, phi_weight_decay=0.01, iters=20, o5="eval", seeds=(0,)
    )
    cfg = R1.prereg_config(0, 20)
    procs = {}
    for name, src in (("new", ROOT / "src"), ("old", old_src / "src")):
        out = tmp_path / name
        argv = R2.cli_argv(cfg, "psiC", owner, out)
        env = R1.thread_env(1)
        env["PYTHONPATH"] = os.pathsep.join([str(src), str(ROOT / "scripts")])
        log = open(tmp_path / f"{name}.log", "w")  # noqa: SIM115
        procs[name] = (
            subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=log),
            log,
        )
    for _name, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        assert rc == 0, (tmp_path / f"{_name}.log").read_text()[-3000:]
    for src, has_switch in ((ROOT / "src", True), (old_src / "src", False)):
        # each side imported the tree it claims: rsr from `src`, and only this head
        # has the switch
        env = R1.thread_env(1)
        env["PYTHONPATH"] = os.pathsep.join([str(src), str(ROOT / "scripts")])
        probe = subprocess.run(
            [
                str(PY),
                "-c",
                "import rsr, importlib.util as u; print(rsr.__file__); "
                "print(u.find_spec('rsr.retention.rank_loss') is not None)",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        f, sw = probe.stdout.split()
        assert Path(f).resolve().is_relative_to(src.resolve()) and sw == str(has_switch)
    a, b = tmp_path / "new", tmp_path / "old"
    assert R2.comparable_heartbeat(a / "heartbeat.jsonl") == R2.comparable_heartbeat(
        b / "heartbeat.jsonl"
    )
    for f in ("diagnostics.jsonl", "evictions.jsonl"):
        ra = [R2._strip(x) for x in R2._jsonl(a / f)]
        rb = [R2._strip(x) for x in R2._jsonl(b / f)]
        assert ra == rb and len(ra) > 0, f
    ca, cb = (
        R1.read_checkpoint(a / "ckpt-000020.pt"),
        R1.read_checkpoint(b / "ckpt-000020.pt"),
    )
    for part in ("model", "policy"):
        ta, tb = _tensors(ca.get(part)), _tensors(cb.get(part))
        assert ta.keys() == tb.keys()
        assert all(torch.equal(ta[k], tb[k]) for k in ta), part
    ha = R2._jsonl(a / "heartbeat.jsonl")[0]
    hb = R2._jsonl(b / "heartbeat.jsonl")[0]
    assert ha["run_id"] == hb["run_id"]


def _tensors(x, prefix="") -> dict:
    out = {}
    if isinstance(x, torch.Tensor):
        out[prefix] = x
    elif isinstance(x, dict):
        for k, v in x.items():
            out.update(_tensors(v, f"{prefix}.{k}"))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(_tensors(v, f"{prefix}[{i}]"))
    return out


# --------------------------------------------------------------------------- #
# T2 isolation; T3 no age
# --------------------------------------------------------------------------- #


def _rank_cfg(psi=None, omega=8.0):
    return RSR.RSRConfig(**_kw(psi, loss="rank", rank_k=3, rank_omega=omega))


@pytest.mark.parametrize("psi", [None, RSR.AGE_ONLY])
def test_t2_rank_loss_reaches_phi_only(psi, p2a):
    model, docs, ids, mask = _world()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    pol = _drive(RSR, _rank_cfg(psi), model, docs, ids, mask)
    loss = pol.retention_loss()
    assert loss is not None and torch.isfinite(loss)
    assert pol.last_mc_stats["loss"] == "rank" and pol.last_mc_stats["l_mc"] is None
    loss.backward()
    names = [n for n, _ in model.named_parameters()]
    assert any("sent" in n.lower() for n in names) or names  # W_sent is a model param
    assert all(p.grad is None for p in model.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in pol.head.parameters()
    )
    if psi is None:
        frozen = model.requires_grad_(False)
        opt = p2a._H["OP"].phi_optimizer(frozen, pol.head, 1e-3)
        assert p2a._H["OP"].optimizer_holds_only_phi(opt, pol.head)["ok"]


def test_t3_content_head_takes_no_age():
    params = list(inspect.signature(BilinearValueHead.forward).parameters)
    assert params[0] == "self" and not any("age" in p for p in params)
    h = BilinearValueHead(128, generator=torch.Generator().manual_seed(0))
    assert sum(p.numel() for p in h.parameters()) == 128 * 128 + 2 * 128 == 16_640


@pytest.mark.parametrize("psi,moves", [(None, False), (RSR.AGE_ONLY, True)])
def test_t3_permuting_ages_moves_nothing_in_the_content_arm(psi, moves):
    model, docs, ids, mask = _world()
    model.eval()
    pol = _drive(RSR, _rank_cfg(psi), model, docs, ids, mask)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():  # the age table is zero at init: give it values to read
        for prm in pol.head.parameters():
            if psi is not None:
                prm.copy_(torch.randn(prm.shape, generator=g))
    base = pol.mc_targets()
    l0 = pol.retention_loss()
    perm = base["key"][torch.randperm(base["key"].numel(), generator=g)]
    shifted = {**base, "key": perm}
    pol.mc_targets = lambda: shifted  # same s, c, G; only written_at (ages) changed
    l1 = pol.retention_loss()
    assert (not torch.equal(l0.detach(), l1.detach())) == moves


# --------------------------------------------------------------------------- #
# T4 the loss
# --------------------------------------------------------------------------- #


def _brute(scores, G, valid, k, omega):
    """The §2.1 definitions, written out with loops."""
    per, n_no = [], 0
    for s, g, v in zip(scores.tolist(), G.tolist(), valid.tolist(), strict=True):
        idx = [i for i, ok in enumerate(v) if ok]
        vals = sorted(g[i] for i in idx)
        b = {}
        if len(idx) <= k:
            b = dict.fromkeys(idx, 1.0)
        else:
            kth = vals[k - 1]
            nb = sum(1 for i in idx if g[i] < kth)
            nt = sum(1 for i in idx if g[i] == kth)
            for i in idx:
                b[i] = 1.0 if g[i] < kth else ((k - nb) / nt if g[i] == kth else 0.0)
        num = den = 0.0
        for i in idx:
            for j in idx:
                if g[i] > g[j]:
                    w = 1 + (omega - 1) * max(b[i], b[j])
                    num += w * float(
                        torch.nn.functional.softplus(torch.tensor(s[j] - s[i]))
                    )
                    den += w
        if den == 0:
            n_no += 1
        else:
            per.append(num / den)
    return sum(per) / len(per), n_no


def test_t4_membership_by_hand():
    G = torch.tensor(
        [
            [3.0, 1.0, 2.0, 0.0, 0.0],  # no ties, k=3: b = 0,1,1,1 (5th invalid)
            [0.0, 1.0, 1.0, 1.0, 5.0],  # ties at the k-th: 1, 2/3, 2/3, 2/3, 0
            [0.0, 1.0, 2.0, 5.0, 5.0],  # ties away from it: 1, 1, 1, 0, 0
            [2.0, 1.0, 0.0, 0.0, 0.0],  # n = 2 <= k: all 1
        ]
    )
    V = torch.tensor(
        [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]],
        dtype=torch.bool,
    )
    b = RL.bottom_k_membership(G, V, 3)
    want = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 2 / 3, 2 / 3, 2 / 3, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(b, want, atol=0, rtol=1e-6)
    assert torch.allclose(b.sum(-1), torch.tensor([3.0, 3.0, 3.0, 2.0]))


def test_t4_loss_by_hand_and_against_the_definitions():
    G = torch.tensor([[0.0, 1.0, 4.0, 4.0, 7.0], [1.0, 1.0, 1.0, 0.0, 0.0]])
    V = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    s = torch.tensor([[0.3, -0.2, 0.5, 0.1, 0.9], [0.4, 0.0, -1.0, 0.0, 0.0]])
    r = RL.rank_loss_groups(s, G, V, k=2, omega=8.0)
    assert r.n_groups_with_pair == 1 and r.n_groups_no_pair == 1  # all-equal G: no pair
    assert r.n_pairs == 9  # 10 pairs of 5, one tie (4, 4) forms none
    want, n_no = _brute(s, G, V, 2, 8.0)
    assert n_no == 1
    assert abs(float(r.loss) - want) < 1e-6
    # literal: group 0, k=2 -> b = 1,1,0,0,0; pairs touching slots 0/1 weigh 8
    sp = torch.nn.functional.softplus
    pairs = [(i, j) for i in range(5) for j in range(5) if G[0, i] > G[0, j]]
    w = {p: (8.0 if (p[0] in (0, 1) or p[1] in (0, 1)) else 1.0) for p in pairs}
    lit = sum(w[p] * float(sp(s[0, p[1]] - s[0, p[0]])) for p in pairs) / sum(w.values())
    assert abs(float(r.loss) - lit) < 1e-6


def test_t4_random_groups_match_the_definitions_and_invariances():
    g = torch.Generator().manual_seed(0)
    n, width = 40, 9
    G = torch.randint(0, 5, (n, width), generator=g).float()  # many ties
    V = torch.rand(n, width, generator=g) > 0.2
    s = torch.randn(n, width, generator=g)
    for k, om in ((1, 2.0), (3, 8.0), (3, 32.0), (12, 2.0)):
        r = RL.rank_loss_groups(s, G, V, k=k, omega=om)
        want, n_no = _brute(s, G, V, k, om)
        assert abs(float(r.loss) - want) < 1e-5 and r.n_groups_no_pair == n_no
    # adding a constant to every score of a group changes nothing
    shift = torch.randn(n, 1, generator=g) * 10
    r0 = RL.rank_loss_groups(s, G, V, k=3, omega=8.0)
    r1 = RL.rank_loss_groups(s + shift, G, V, k=3, omega=8.0)
    assert torch.allclose(r0.per_group[r0.has_pair], r1.per_group[r1.has_pair], atol=1e-5)
    # the batch loss does not depend on the order of rows
    perm = torch.randperm(n, generator=g)
    r2 = RL.rank_loss_groups(s[perm], G[perm], V[perm], k=3, omega=8.0)
    assert abs(float(r0.loss) - float(r2.loss)) < 1e-6
    # omega = 1 is the unweighted mean over pairs
    r3 = RL.rank_loss_groups(s, G, V, k=3, omega=1.0)
    per = []
    for si, gi, vi in zip(s, G, V, strict=True):
        idx = vi.nonzero().flatten().tolist()
        t = [
            float(torch.nn.functional.softplus(si[j] - si[i]))
            for i in idx
            for j in idx
            if gi[i] > gi[j]
        ]
        if t:
            per.append(sum(t) / len(t))
    assert abs(float(r3.loss) - sum(per) / len(per)) < 1e-5


def test_t4_no_pair_anywhere_is_none_and_counted():
    r = RL.rank_loss_groups(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.ones(2, 3, dtype=torch.bool),
        k=3,
        omega=2,
    )
    assert r.loss is None and r.n_groups_no_pair == 2


def test_t4_gradient_is_finite_with_padding_and_reaches_scores_only():
    s = torch.zeros(1, 4, requires_grad=True)
    G = torch.tensor([[2.0, 1.0, float("nan"), 0.0]])
    V = torch.tensor([[True, True, False, False]])
    r = RL.rank_loss_groups(s, torch.nan_to_num(G), V, k=3, omega=8.0)
    r.loss.backward()
    assert torch.isfinite(s.grad).all() and s.grad[0, 2:].abs().sum() == 0


def test_t4_invalid_parameters_are_refused():
    with pytest.raises(ValueError):
        RL.bottom_k_membership(torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), 0)
    with pytest.raises(ValueError):
        RL.pair_weights(torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), 3, 0.5)


def test_t4_policy_rank_loss_equals_the_padded_definition():
    """2b's ragged path (RSRPolicy._rank_loss over mc_targets' rows) equals the
    definitions applied group by group."""
    model, docs, ids, mask = _world()
    model.eval()
    pol = _drive(RSR, _rank_cfg(None, omega=32.0), model, docs, ids, mask)
    rows = pol.mc_targets()
    loss = pol.retention_loss()
    with torch.no_grad():
        psi = pol.head(rows["s"].unsqueeze(1), rows["c"]).squeeze(1)
    keys = list(zip(rows["row"].tolist(), rows["t"].tolist(), strict=True))
    groups: dict = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    width = max(len(v) for v in groups.values())
    Sx = torch.zeros(len(groups), width)
    Gx = torch.zeros(len(groups), width)
    Vx = torch.zeros(len(groups), width, dtype=torch.bool)
    for gi, idx in enumerate(groups.values()):
        Sx[gi, : len(idx)] = psi[idx]
        Gx[gi, : len(idx)] = rows["target"][idx]
        Vx[gi, : len(idx)] = True
    want, _ = _brute(Sx, Gx, Vx, 3, 32.0)
    assert abs(float(loss) - want) < 1e-5
    assert pol.last_mc_stats["rank_groups"] == len(groups) == rows["n_groups"]


# --------------------------------------------------------------------------- #
# T5 sanity: a planted bilinear G
# --------------------------------------------------------------------------- #


def test_t5_planted_bilinear_target_is_learned(p2a):
    g = torch.Generator().manual_seed(0)
    N, n, d = 32, 10, 4
    Sx = torch.randn(N, n, d, generator=g)
    Cx = torch.randn(N, n, d, generator=g)
    Wst = torch.randn(d, d, generator=g)
    # G[b, t, i] = s_i^T W c_t, rounded: integer margins (ties allowed) so the argmin
    # is identifiable in finite steps
    G = torch.einsum("bid,de,bte->bti", Sx, Wst, Cx).round()
    mask = torch.ones(n, n, dtype=torch.bool).tril(-1).expand(N, n, n).clone()
    data = {"s": Sx, "c": Cx, "G": G, "mask": mask}
    head = BilinearValueHead(d, generator=torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.2, weight_decay=0.0)
    pos = list(range(N))
    r = p2a.fit_loop(
        head,
        opt,
        data,
        pos,
        seed=0,
        loss="rank",
        omega=8.0,
        cap=1500,
        every=250,
        stop_on_convergence=False,
    )
    j = [h["J"] for h in r["history"]]
    assert j[-1] < 0.05 * j[0]
    P = p2a._H["OP"].psi_matrix(head, Sx, Cx).detach()
    agree = RL.argmin_agreement(*p2a.as_groups(P, G, mask))
    assert agree == 1.0, agree


# --------------------------------------------------------------------------- #
# 2a decision logic on hand-built inputs
# --------------------------------------------------------------------------- #


def test_convergence_rule_by_hand(p2a):
    h = [{"step": 0, "J": 1.0}]
    h += [{"step": k, "J": 1.0 - 0.1 * k / 2000} for k in range(250, 2001, 250)]
    assert not p2a.converged(h, 1750)  # before the first eligible check
    assert not p2a.converged(h, 2000)  # 10% better than best(0)
    flat = [{"step": k, "J": 0.5} for k in range(0, 4001, 250)]
    assert p2a.converged(flat, 2000)
    tiny = [{"step": 0, "J": 1.0}] + [
        {"step": k, "J": 1.0 - 0.0009 * k / 2000} for k in range(250, 2001, 250)
    ]
    assert p2a.converged(tiny, 2000)  # improved < 0.1% over the window
    nan = [{"step": k, "J": float("nan")} for k in range(0, 2001, 250)]
    assert not p2a.converged(nan, 2000)


def test_fit_loop_stops_at_convergence_and_reports_cap_hit(p2a):
    g = torch.Generator().manual_seed(0)
    N, n, d = 4, 6, 3
    data = {
        "s": torch.randn(N, n, d, generator=g),
        "c": torch.randn(N, n, d, generator=g),
        "G": torch.randn(N, n, n, generator=g),
        "mask": torch.ones(n, n, dtype=torch.bool).tril(-1).expand(N, n, n).clone(),
    }
    head = BilinearValueHead(d, generator=torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.0)  # nothing moves: J flat
    r = p2a.fit_loop(
        head, opt, data, list(range(N)), seed=0, loss="rank", omega=2.0, cap=10_000
    )
    assert r["label"] == "CONVERGED" and r["k_conv"] == 2000
    head = BilinearValueHead(d, generator=torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(head.parameters(), lr=0.0)
    r = p2a.fit_loop(
        head, opt, data, list(range(N)), seed=0, loss="rank", omega=2.0, cap=1000
    )
    assert r["label"] == "CAP-HIT" and r["k_conv"] == 1000
    assert [h["step"] for h in r["history"]] == [0, 250, 500, 750, 1000]


def test_selection_rule_ties_to_the_smaller_omega_and_none(p2a):
    def caps(a, b, c):
        return {
            p2a.wtag(w): {"cap": {"point": x}}
            for w, x in zip(p2a.OMEGAS, (a, b, c), strict=True)
        }

    assert p2a.choose(caps(0.1, 0.3, 0.3))["omega"] == 8.0
    assert p2a.choose(caps(0.5, None, 0.4))["omega"] == 2.0
    assert p2a.choose(caps(-0.2, -0.1, float("nan")))["omega"] == 8.0
    r = p2a.choose(caps(None, None, None))
    assert r["omega"] is None and r["ok"] is False


def test_tune_caps_on_a_fixture(p2a):
    # 4 documents; hits as (gap, resident)
    fifo = [[(1, True), (20, False)], [(3, True)], [(30, False)], [(2, True), (2, False)]]
    ff = [[(1, True), (20, True)], [(3, True)], [(30, True)], [(2, True), (2, True)]]
    good = [[(1, True), (20, True)], [(3, True)], [(30, False)], [(2, True), (2, True)]]
    hits = {"fifo": fifo, "factfiller": ff, "omega2": fifo, "omega8": good, "omega32": ff}
    c = p2a.tune_caps(hits, seed=0, m=16, n_boot=50)
    assert c["omega2"]["cap"]["point"] == 0.0
    assert c["omega32"]["cap"]["point"] == 1.0
    assert abs(c["omega8"]["cap"]["point"] - 2 / 3) < 1e-12
    assert p2a.choose(c)["omega"] == 32.0


def test_prereg_constants_are_the_committed_texts(p2a):
    chk = p2a.prereg_check()
    assert chk["ok"], chk
    assert p2a.registered_seeds() == (0, 1, 2)


def test_order_check(p2a):
    ev = [{"seed": 0, "stage": "fit", "what": "end", "t": t} for t in (1, 2, 3, 4)]
    ev += [
        {"seed": 0, "stage": "select", "what": "start", "t": 5},
        {"seed": 0, "stage": "select", "what": "end", "t": 6},
        {"seed": 0, "stage": "deploy", "what": "start", "t": 7},
    ]
    assert p2a.order_check(ev, 0)["ok"]
    bad = [*ev, {"seed": 0, "stage": "deploy", "what": "start", "t": 8}]
    assert not p2a.order_check(bad, 0)["ok"]  # EVAL read twice
    early = [dict(e, t=0.5) if e["stage"] == "select" else e for e in ev]
    assert not p2a.order_check(early, 0)["ok"]


def test_tune_records_carry_no_accuracy_fields(p2a):
    r = {
        "doc": 1,
        "answers": [1],
        "real_sum": 1.0,
        "real_n": 1,
        "hits": [],
        "victims": {},
    }
    assert set(p2a._strip_tune(r)) == {"doc", "hits", "victims"}


def test_every_stage_installs_its_own_seal(p2a):
    """Static: each stage function opens the seal its PREREG §1 row names."""
    tree = ast.parse((EXP / "phase2a.py").read_text())
    want = {
        "rc2_seed": "fit",
        "fit_seed_omega": "fit",
        "select_seed": "select",
        "deploy_seed": "deploy",
        "finalize": "finalize",
        "launch_checks": "launch-checks",
        "time_smoke": "time",
    }
    got = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            for c in ast.walk(node):
                if (
                    isinstance(c, ast.Call)
                    and getattr(c.func, "id", None) == "open_stage"
                    and isinstance(c.args[0], ast.Constant)
                ):
                    got[node.name] = c.args[0].value
    assert got == want


# --------------------------------------------------------------------------- #
# T6 the stage seals (each stage in its own process)
# --------------------------------------------------------------------------- #

PREREG_SEALS = {  # PREREG §1's table: the ranges SEALED at each stage
    ("2a", "fit"): {"TUNE", "EVAL2"},
    ("2a", "select"): {"EVAL2"},
    ("2a", "deploy"): {"TUNE", "EVAL2"},
    ("2a", "finalize"): {"TUNE", "EVAL2"},
    ("2b", "smoke"): {"TUNE", "EVAL2"},
    ("2b", "train"): {"TUNE", "EVAL2"},
    ("2b", "measure"): {"TUNE"},
    ("2b", "finalize"): {"TUNE", "EVAL2"},
}

_T6 = textwrap.dedent(
    """
    import json, sys, types
    sys.path.insert(0, {src!r})
    import rsr.data.synthetic as syn
    calls = []
    def stub_doc(doc_id, cfg):
        calls.append(int(doc_id))
        return "REACHED"
    def stub_gen(cfg=None):
        calls.append(-1)
        return ["REACHED"]
    syn._generate_document = stub_doc   # never generate a real document here
    syn.generate = stub_gen
    early = types.ModuleType("early_importer")
    early.f = syn._generate_document    # a `from ... import` taken BEFORE the seal
    sys.modules["early_importer"] = early
    import importlib.util
    spec = importlib.util.spec_from_file_location("ss", {seal!r})
    ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
    info = ss.install({run!r}, {stage!r})
    rng = ss.sealed_ranges()
    out = {{"sealed": sorted(info["sealed"]), "result": {{}},
           "ranges": {{k: list(v) for k, v in rng.items()}}}}
    cfg = syn.SyntheticConfig(seed=0)
    for name, (lo, hi) in rng.items():
        res = []
        for f in (syn._generate_document, early.f):
            for i in (lo, hi - 1):
                try:
                    res.append(f(i, cfg))
                except ss.SealedRange:
                    res.append("SEALED")
        try:
            syn.generate(syn.SyntheticConfig(seed=0, n_documents=hi))
            res.append("REACHED")
        except ss.SealedRange:
            res.append("SEALED")
        out["result"][name] = res
    try:
        ss.install({run!r}, "finalize" if {stage!r} != "finalize" else "launch-checks")
        out["second_stage"] = "ALLOWED"
    except Exception as e:
        out["second_stage"] = type(e).__name__
    out["leaks"] = ss.C.seal_leaks()
    print(json.dumps(out))
    """
)


@pytest.mark.parametrize("run,stage", sorted(PREREG_SEALS))
def test_t6_stage_seal_raises_exactly_where_the_prereg_seals(run, stage, tmp_path):
    code = _T6.format(
        src=str(ROOT / "src"), seal=str(EXP / "stage_seal.py"), run=run, stage=stage
    )
    r = subprocess.run([str(PY), "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    want = PREREG_SEALS[(run, stage)]
    assert set(out["sealed"]) == want
    rng = {k: tuple(v) for k, v in out["ranges"].items()}
    for name, res in out["result"].items():
        expect = "SEALED" if name in want else "REACHED"
        assert res[:4] == [expect] * 4, (name, res)
        # generate(n_documents=hi) covers [0, hi): sealed iff it reaches a sealed range
        reach = any(rng[k][0] < rng[name][1] for k in want)
        assert res[4] == ("SEALED" if reach else "REACHED"), (name, res)
    assert out["second_stage"] == "ControlFailure"
    assert out["leaks"] == []


def test_t6_the_stage_table_is_the_prereg_s(p2a):
    SS = p2a.SS
    for (run, stage), want in PREREG_SEALS.items():
        assert set(SS.STAGES[run][stage]) == want
    for run, stages in SS.STAGES.items():
        for stage, sealed in stages.items():
            if (run, stage) not in PREREG_SEALS:  # helper stages seal everything
                assert set(sealed) == {"TUNE", "EVAL2"}, (run, stage)


def test_t6_no_sealed_literal_in_phase_2_code_or_tests(p2a):
    """R-SEAL (c): the bounds live in the phase-1 front matter only. Every integer in
    TUNE or EVAL2 in this experiment's code and this test file is a defect."""
    C = p2a.SS.C
    rng = p2a.SS.sealed_ranges()
    files = [*EXP.glob("*.py"), *EXP.glob("*.zsh"), Path(__file__)]
    hits = []
    for f in files:
        for m in C._INT_TOKEN.finditer(f.read_text()):
            v = int(m.group(1).replace("_", ""))
            if C.sealed_name(v, rng):
                hits.append((f.name, m.group(1)))
    assert hits == []


def test_t6_phase2_allowlist_is_narrow(p2a):
    SS = p2a.SS
    assert SS.allowed("experiments/rsr-run3-ranking/PREREG.md")
    assert SS.allowed("runs/rsr-run3-2a/seed0/select.json")
    assert not SS.allowed("experiments/rsr-run3-ranking/phase2a.py")
    assert not SS.allowed("tests/test_rsr_run3_ranking.py")
    # Amendment PM-4: committed review markdown is prose, not a read or a claim.
    assert SS.allowed("docs/reviews/rsr-run3-ranking/66352afbbcde.md")
    assert SS.allowed("docs/reviews/post-fails-controls/b7461397ce69.md")
    assert not SS.allowed("docs/reviews/rsr-run3-ranking/fixture.json")
    assert not SS.allowed("docs/lab-notes/prior-art-eviction-signals.md")
    assert not SS.allowed("docs/reviewsX/a.md")


# --------------------------------------------------------------------------- #
# T7 config
# --------------------------------------------------------------------------- #


def test_t7_rank_arms_config_and_label(p2a):
    base = {
        "b_enabled": False,
        "shadow_enabled": False,
        "nu": 0.0,
        "beta": 1.0,
        "gamma": 0.9,
        "target": "loo_resample",
    }
    for psi in (None, RSR.AGE_ONLY):
        ov = p2a.rank_overrides(base, 8.0)
        if psi:
            ov["psi_override"] = psi
        pol = loop.build_policy(
            "rsr",
            d_model=D,
            steps_per_epoch=16.0,
            overrides=ov,
            generator=torch.Generator().manual_seed(0),
        )
        c = pol.config
        assert (c.b_enabled, c.shadow_enabled, c.nu, c.target, c.gamma) == (
            False,
            False,
            0.0,
            "loo_resample",
            0.9,
        )
        assert (c.loss, c.rank_k, c.rank_omega) == ("rank", 3, 8.0)
        lab = loop.arm_label("rsr", ov)
        assert "loss=rank" in lab and "rank_k=3" in lab and "rank_omega=8" in lab
        assert "target=loo_resample" in lab
    # the CLI path: parse_overrides gives floats; accepted, same label
    cli = loop.parse_overrides(["loss=rank", "rank_k=3", "rank_omega=32"])
    assert cli == {"loss": "rank", "rank_k": 3.0, "rank_omega": 32.0}
    assert loop.arm_label("rsr", cli) == "rsr[loss=rank,rank_k=3,rank_omega=32]"


def test_t7_switch_misuse_is_refused():
    with pytest.raises(ValueError, match="not a known retention loss"):
        RSR.RSRPolicy(RSR.RSRConfig(**_kw(loss="hinge")), d_model=D)
    with pytest.raises(ValueError, match="needs rank_k and rank_omega"):
        RSR.RSRPolicy(RSR.RSRConfig(**_kw(loss="rank", rank_k=3)), d_model=D)
    with pytest.raises(ValueError, match="configure loss='rank' only"):
        RSR.RSRPolicy(RSR.RSRConfig(**_kw(rank_omega=8.0)), d_model=D)
    with pytest.raises(ValueError, match="positive integer"):
        RSR.RSRPolicy(
            RSR.RSRConfig(**_kw(loss="rank", rank_k=2.5, rank_omega=2.0)), d_model=D
        )
    with pytest.raises(ValueError, match="finite number >= 1"):
        RSR.RSRPolicy(
            RSR.RSRConfig(**_kw(loss="rank", rank_k=3, rank_omega=0.5)), d_model=D
        )


# --------------------------------------------------------------------------- #
# 2b on fixtures (its smoke needs 2a's selection; nothing here trains or reads EVAL2)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def p2b():
    mod = _load("run3_phase2b_test", EXP / "phase2b.py")
    mod.harness()
    return mod


def test_2b_k_rule(p2b):
    assert p2b.k_2b([500, 1000, 2999]) == 3000
    assert p2b.k_2b([3000]) == 3000
    assert p2b.k_2b([3001, 10]) == 4500
    assert p2b.k_2b([95500, 117500, 103500]) == 118500
    assert p2b.k_2b([200_000]) == 201_000
    assert p2b.prereg_check()["ok"]


def test_2b_cadence_keeps_3000_160_and_k_on_disk(p2b):
    R1 = p2b.harness()["R1"]
    for K in range(3000, 201_001, 1500):
        e = p2b.ckpt_every(K)
        assert K % e == 0 and 3000 % e == 0 and 160 % e == 0, K
        assert K % R1.derive_ckpt(K) == 0
    assert p2b.ckpt_every(3000) == 40  # run 1's / run 2's cadence, unchanged


def _ledger_2a(tmp_path, kconv, labels, gate=True, exit_=0):
    per = {str(s): {} for s in (0, 1, 2)}
    rows = [
        {
            "key": "summary",
            "value": {
                "exit": exit_,
                "gate_to_2b": gate,
                "labels": labels,
                "per_seed": per,
            },
        }
    ]
    for s in (0, 1, 2):
        rows.append(
            {
                "key": f"seed{s}.selection",
                "value": {"omega_s": [2.0, 8.0, 32.0][s], "k_conv": kconv[s]},
            }
        )
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_2b_plan_from_2a_ledger(p2b, tmp_path):
    win = {"0": "WIN", "1": "WIN", "2": "WIN"}
    ok = p2b.make_plan(_ledger_2a(tmp_path, [2000, 2750, 1000], win))
    assert ok["ok"] and ok["K_2b"] == 3000 and ok["omega_s"] == {0: 2.0, 1: 8.0, 2: 32.0}
    assert ok["ckpt_every"] == 40 and ok["psiR_at_3000"] is False
    # K_2b != 3000: no longer refused (Amendment PM-1); see test_pm1_* for the rest
    big = p2b.make_plan(_ledger_2a(tmp_path, [2000, 95_500, 1000], win))
    assert big["ok"] and big["K_2b"] == 96_000
    # 2a not WIN on every seed: 2b does not run
    no = p2b.make_plan(
        _ledger_2a(
            tmp_path,
            [2000, 2000, 2000],
            {"0": "WIN", "1": "EQUIV", "2": "WIN"},
            gate=False,
        )
    )
    assert not no["ok"] and "K_2b" not in no
    bad = p2b.make_plan(_ledger_2a(tmp_path, [2000, 2000, 2000], win, exit_=3))
    assert not bad["ok"]


def test_2b_cli_argv_is_run_2s_plus_the_switch_and_fifo_is_run_1s(p2b):
    H = p2b.harness()
    R1, R2 = H["R1"], H["R2"]
    p2b.set_plan({"omega_s": {"0": 2.0, "1": 8.0, "2": 32.0}})
    ow = R1.OwnerValues(0.9, 0.01, 3000, "eval", (0, 1, 2))
    for seed, om in ((0, "2"), (1, "8"), (2, "32")):
        cfg = R1.prereg_config(seed, 3000)
        for arm in R1.ARMS:
            got = R1.cli_argv(cfg, arm, ow, Path(f"runs/rsr-run3-2b/{arm}/seed{seed}"))
            r2 = R2._R1_CLI_ARGV(cfg, arm, ow, Path(f"runs/rsr-run3-2b/{arm}/seed{seed}"))
            if arm == "fifo":
                assert got == r2 and not any("loss=" in t for t in got)
                continue
            i = got.index("target=loo_resample")
            assert got[i + 1 : i + 7] == [
                "--policy-override",
                "loss=rank",
                "--policy-override",
                "rank_k=3",
                "--policy-override",
                f"rank_omega={om}",
            ]
            items = [got[j + 1] for j, t in enumerate(got) if t == "--policy-override"]
            ov = loop.parse_overrides(items)
            pol = loop.build_policy(
                "rsr",
                d_model=128,
                steps_per_epoch=16.0,
                overrides=ov,
                generator=torch.Generator().manual_seed(0),
            )
            c = pol.config
            assert (c.loss, c.rank_k, c.rank_omega, c.target, c.b_enabled, c.nu) == (
                "rank",
                3.0,
                float(om),
                "loo_resample",
                False,
                0.0,
            )
            assert "loss=rank" in loop.arm_label("rsr", ov)
    assert R1.RUN_ID == "rsr-run3-2b"


def test_2b_launcher_command_is_the_harness_s_token_for_token(p2b):
    H = p2b.harness()
    R1 = H["R1"]
    p2b.set_plan({"omega_s": {"0": 2.0, "1": 8.0, "2": 32.0}})
    text = (EXP / "launch_2b.zsh").read_text()
    i = text.index(".venv/bin/python -m rsr.train.loop")
    j = text.index('"${P[@]}" --out-dir $OUT/$a/seed$s') + len(
        '"${P[@]}" --out-dir $OUT/$a/seed$s'
    )
    tmpl = text[i:j].replace("\\\n", " ")
    ow = R1.OwnerValues(0.9, 0.01, 3000, "eval", (0, 1, 2))
    for seed, om in ((0, "2"), (1, "8"), (2, "32")):
        rsr = (
            "--policy rsr --policy-override b_enabled=false --policy-override "
            "shadow_enabled=false --policy-override nu=0 --policy-override beta=1 "
            "--policy-override gamma=0.9 --policy-override target=loo_resample "
            "--policy-override loss=rank --policy-override rank_k=3 "
            f"--policy-override rank_omega={om}"
        )
        for arm in R1.ARMS:
            p = {
                "psiC": rsr,
                "fifo": "--policy fifo",
                "age": rsr + " --policy-override psi_override=age_only",
            }[arm]
            s = R1.SUBSTRATE[seed]
            out = f"runs/rsr-run3-2b/{arm}/seed{seed}"
            cmd = tmpl
            for k, v in (
                ('"${C1[@]}"', " ".join(R1.C1_FLAGS["eval"])),
                ('"${C3[@]}"', ""),
                ('"${P[@]}"', p),
                ("${SHA[$s]}", s["sha256"]),
                ("$IT", "3000"),
                ("$EV", str(p2b.ckpt_every(3000))),
                ("$OUT/$a/seed$s", out),
                ("$B", str(Path(s["checkpoint"]).parent)),
                ("$R", R1._R),
                ("$s", str(seed)),
            ):
                cmd = cmd.replace(k, v)
            cfg = R1.prereg_config(seed, 3000)
            assert R1.cli_argv(cfg, arm, ow, Path(out)) == shlex.split(cmd), (arm, seed)


@pytest.mark.parametrize(
    ("env", "msg"),
    [
        (
            {
                "GAMMA": "0.9",
                "WD": "0.01",
                "O5": "eval",
                "C1_FLAGS": "--frozen-model-mode eval",
                "STEPS": "3000",
            },
            "STEPS is K_2b from plan.json",
        ),
        (
            {
                "GAMMA": "0.9",
                "WD": "0.01",
                "O5": "eval",
                "C1_FLAGS": "--frozen-model-mode eval",
            },
            "no runs/rsr-run3-2b/plan.json",
        ),
        ({"WD": "0.01", "O5": "eval"}, "slot GAMMA unfilled"),
        (
            {
                "GAMMA": "0.9",
                "WD": "0.01",
                "O5": "eval",
                "C1_FLAGS": "--frozen-model-mode eval",
                "SMOKE": "1",
                "JOB_SUFFIX": "-s",
            },
            "SMOKE=1 needs OUT",
        ),
    ],
)
def test_2b_launcher_refusals_exit_3_before_anything_runs(tmp_path, env, msg):
    p = subprocess.run(
        ["/bin/zsh", str(EXP / "launch_2b.zsh")],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **env},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 3, p.stderr
    assert msg in p.stderr
    assert not (tmp_path / "runs").exists()


def test_2b_c11_prime_c_k_and_c_omega_on_fixtures(p2b, tmp_path):
    lab = "rsr[b=off,beta=1,gamma=0.9,loss=rank,nu=0,rank_k=3,rank_omega=8,shadow=off,target=loo_resample]"  # noqa: E501
    good_beat = {"step": 1, "loo_s": 0.1, "mc": {"loss": "rank", "l_rank": 0.6}}
    res = {a: {"config": {"arm_label": lab}} for a in ("psiC", "age")}
    res["fifo"] = {"config": {"arm_label": "fifo+frozen[eval]"}}
    hbs = {a: {"beats": [good_beat]} for a in ("psiC", "age")}
    launch = {"loss": "rank", "target": "loo_resample", "smoke": ""}
    assert p2b.c11_prime(res, hbs, launch, {"ok": True})["ok"]
    bad = {**hbs, "age": {"beats": [{**good_beat, "mc": {"loss": "mc", "l_rank": None}}]}}
    assert not p2b.c11_prime(res, bad, launch, {"ok": True})["ok"]
    assert not p2b.c11_prime(res, hbs, {**launch, "loss": None}, {"ok": True})["ok"]
    assert not p2b.c11_prime(res, hbs, launch, {"ok": False})["ok"]
    plan = {"K_2b": 3000, "seeds": [0, 1, 2], "omega_s": {0: 2.0, 1: 8.0, 2: 32.0}}
    kc = {0: 1000, 1: 2000, 2: 3000}
    assert p2b.c_k(plan, {"steps": "3000"}, {"psiC": 3000, "age": 3000}, kc)["ok"]
    assert not p2b.c_k(plan, {"steps": "3000"}, {"psiC": 2999, "age": 3000}, kc)["ok"]
    assert not p2b.c_k(plan, {"steps": "4500"}, {"psiC": 3000, "age": 3000}, kc)["ok"]
    led = {
        "sha256": "a" * 64,
        "selection": {0: {"omega_s": 2.0}, 1: {"omega_s": 8.0}, 2: {"omega_s": 32.0}},
    }
    pr = tmp_path / "P.md"
    pr.write_text(
        "x\n\n## AMENDMENT 2b-S (date)\n\n- 2a ledger sha256: `"
        + "a" * 64
        + "`\n- K_2b: 3000\n\n## next\n"
    )
    am = p2b.amendment_2bS(pr)
    assert am == {"ledger_2a_sha256": "a" * 64, "K_2b": 3000}
    assert p2b.c_omega(plan, led, am)["ok"]
    assert not p2b.c_omega(plan, led, None)["ok"]  # no amendment: not pinned
    assert not p2b.c_omega({**plan, "omega_s": {0: 8.0, 1: 8.0, 2: 32.0}}, led, am)["ok"]
    assert p2b.amendment_2bS() is None  # not yet committed on this branch


def test_2b_convergence_label(p2b):
    g = torch.Generator().manual_seed(0)
    down = (
        torch.linspace(1.0, 0.5, 3200) + 0.01 * torch.randn(3200, generator=g)
    ).tolist()
    assert p2b.convergence_label(down, 0)["label"] == "STILL-IMPROVING"
    flat = (0.5 + 0.01 * torch.randn(3200, generator=g)).tolist()
    r = p2b.convergence_label(flat, 0)
    assert r["label"] == "PLATEAU" and r["lo"] < 0 < r["hi"]
    assert p2b.convergence_label(flat[:100], 0)["label"] is None


def test_2b_footers_must_all_be_complete(p2b, tmp_path):
    R1 = p2b.harness()["R1"]
    logs = tmp_path / "logs"
    logs.mkdir()
    for s in (0, 1, 2):
        for a in R1.ARMS:
            (logs / f"{a}-s{s}.rc").write_text("0\n")
            (logs / f"{a}-s{s}.log").write_text(
                "noise\n" + json.dumps({"run_id": "x", "liveness": {}}, indent=2) + "\n"
            )
    assert p2b.training_footers(tmp_path, [0, 1, 2])["ok"]
    (logs / "age-s2.rc").write_text("1\n")
    assert not p2b.training_footers(tmp_path, [0, 1, 2])["ok"]


def test_2b_rebinding_is_private(p2b):
    fresh = _load("rsr_run1_fresh_run3", ROOT / "experiments" / "rsr-run1" / "run.py")
    assert fresh.RUN_ID == "rsr-run1" and fresh.cli_argv is not p2b.harness()["cli_argv"]
    assert fresh.derive_every(3000, 1000) == 40
    run2_own = sys.modules.get("_rsr_run2_r1")
    if run2_own is not None:
        assert run2_own.RUN_ID == "rsr-run2"


# --------------------------------------------------------------------------- #
# Amendment PM-1 (c7f63ea): the 2a launch-slot check; C10 / C-F at K_2b > 3000
# --------------------------------------------------------------------------- #


def test_pm1_2a_smoke_record_is_machine_read_from_amendment_2aS(p2a):
    rec = p2a.amendment_2aS_record()
    assert rec["path"] == ROOT / "runs" / "rsr-run3-2a-smoke" / "time-seed0.json"
    assert rec["sha_ok"], rec


def test_pm1_wall_projection_by_slot_count(p2a):
    d = json.loads(p2a.amendment_2aS_record()["path"].read_text())
    jobs = p2a.launch_jobs(d)
    assert len(jobs) == 3 * (1 + 3 + 2)  # per seed: rc2, 3 fits, select, deploy
    w12 = p2a.wall_for_slots(d, 12)
    assert abs(w12 - d["projection"]["wall_layout_s"]) < 1e-6  # 2a-S's layout figure
    assert abs(p2a.wall_for_slots(d, 1) - sum(j["s"] for j in jobs)) < 1e-6
    walls = [p2a.wall_for_slots(d, n) for n in range(1, 13)]
    assert max(walls) == walls[0] and min(walls) == walls[-1]
    # the ruling: >= 9 slots launches; fewer launches iff the recomputed wall <= 8 h
    assert p2a.slot_decision(9, d)["ok"] and p2a.slot_decision(12, d)["ok"]
    assert not p2a.slot_decision(0, d)["ok"]
    for n in range(1, 9):
        r = p2a.slot_decision(n, d)
        assert r["recomputed"] and r["ok"] == (r["wall_h"] <= 8.0), (n, r)


def test_pm1_slot_decision_on_a_slow_record(p2a):
    d = json.loads(p2a.amendment_2aS_record()["path"].read_text())
    slow = json.loads(json.dumps(d))
    for w in p2a.OMEGAS:
        slow["projection"]["per_fit_worst_case_s"][p2a.wtag(w)] = 6 * 3600.0
    assert p2a.slot_decision(9, slow)["ok"]  # >= 9: no recompute (the ruling)
    assert not p2a.slot_decision(8, slow)["ok"]  # 6 h fits on 8 slots: two waves


def test_pm1_launcher_gates_every_fit_on_the_slot_check():
    t = (EXP / "launch_2a.zsh").read_text()
    i, j = t.index("2a slot-check"), t.index("2a fit --seed $s --rc2")
    assert i < j
    assert "slot-check.rc" in t and "nothing fitted" in t


def test_lb4_every_seeds_rc2_finishes_before_any_fit_or_select():
    """Review 01b41ea LB-4: select's pre-select gate needs EVERY seed's rc2.json; the
    slot runner polls, so the launcher runs all three R-C2 to completion (one
    ``wait``) before any grid fit, select or deploy is started."""
    t = (EXP / "launch_2a.zsh").read_text()
    rc2 = t.index("2a fit --seed $s --rc2")
    first_wait = t.index("wait", rc2)
    for later in ("2a fit --seed $s --omega", "2a select", "2a deploy"):
        assert t.index(later) > first_wait, later
    block = t[t.rindex("for s in 0 1 2", 0, rc2) : first_wait]
    assert "--omega" not in block and "select" not in block


def test_pm1_2b_plan_no_longer_refuses_at_k_above_3000(p2b, tmp_path):
    win = {"0": "WIN", "1": "WIN", "2": "WIN"}
    big = p2b.make_plan(_ledger_2a(tmp_path, [2000, 95_500, 1000], win))
    assert big["ok"], big["problems"]
    assert big["K_2b"] == 96_000 and big["fifo_iters"] == 3000
    assert big["fifo_ckpt_every"] == 40 and big["ckpt_every"] == p2b.ckpt_every(96_000)
    assert not any("C10" in x for x in big["problems"])
    assert big["unruled_at_K_gt_3000"] == []  # Amendment PM-2 rules on all five
    assert p2b.measure_precheck(big)["ok"]
    small = p2b.make_plan(_ledger_2a(tmp_path, [2000, 2000, 1000], win))
    assert small["ok"] and small["unruled_at_K_gt_3000"] == []
    assert p2b.measure_precheck(small)["ok"]


def test_pm1_fifo_command_is_run_1s_3000_step_command_at_any_k(p2b):
    H = p2b.harness()
    R1, R2 = H["R1"], H["R2"]
    p2b.set_plan({"omega_s": {"0": 2.0, "1": 8.0, "2": 32.0}, "fifo_iters": 3000})
    K = 96_000
    ow = R1.OwnerValues(0.9, 0.01, K, "eval", (0, 1, 2))
    for seed in (0, 1, 2):
        cfg = R1.prereg_config(seed, K)
        out = Path(f"runs/rsr-run3-2b/fifo/seed{seed}")
        got = R1.cli_argv(cfg, "fifo", ow, out)
        want = R2._R1_CLI_ARGV(R1.prereg_config(seed, 3000), "fifo", ow, out)
        assert got == want
        assert got[got.index("--iters") + 1] == "3000"
        assert got[got.index("--ckpt-every") + 1] == "40"
        psi = R1.cli_argv(cfg, "psiC", ow, Path(f"runs/rsr-run3-2b/psiC/seed{seed}"))
        assert psi[psi.index("--iters") + 1] == str(K)
        assert psi[psi.index("--ckpt-every") + 1] == str(p2b.ckpt_every(K))


def test_pm1_c10_checks_fifo_against_r1_and_rsr_against_k(p2b, monkeypatch):
    """The per-arm C10: run 1's c10_launch_values, unchanged, called once on fifo's
    view (iters = R-1 = 3000, its own steps / ckpt / cadence) and once on the RSR arms'
    (iters = K_2b, the registered R-1 replaced by K_2b); results merged."""
    H = p2b.harness()
    R1 = H["R1"]
    seen = []

    def fake(
        cfg, owner, launch, results, hbs, jobs, out_root, *, rcs=None, registered=None
    ):
        seen.append((cfg.iters, owner.iters, dict(launch), sorted(results), registered))
        bad = [a for a in results if results[a]["config"]["iters"] != owner.iters]
        return {
            "per_arm": {a: {"ok": a not in bad} for a in results},
            "launch_problems": [],
            "ok": not bad,
        }

    monkeypatch.setattr(p2b, "_C10_ORIG", fake)
    plan = {
        "K_2b": 96_000,
        "fifo_iters": 3000,
        "ckpt_every": p2b.ckpt_every(96_000),
        "fifo_ckpt_every": 40,
    }
    p2b.set_plan({"omega_s": {"0": 2.0, "1": 8.0, "2": 32.0}, **plan})
    cfg = R1.prereg_config(0, 96_000)
    ow = R1.OwnerValues(0.9, 0.01, 96_000, "eval", (0, 1, 2))
    launch = {
        "steps": "96000",
        "ckpt": "32000",
        "ckpt_every": str(plan["ckpt_every"]),
        "fifo_steps": "3000",
        "fifo_ckpt": "1000",
        "fifo_ckpt_every": "40",
    }
    res = {a: {"config": {"iters": 96_000 if a != "fifo" else 3000}} for a in R1.ARMS}
    reg = {"status": "filled", "values": {"iters": 3000, "gamma": 0.9}}
    out = p2b.c10_per_arm(cfg, ow, launch, res, {}, {}, Path("x"), registered=reg)
    assert out["ok"], out
    fifo_v = next(v for v in seen if v[3] == ["fifo"])
    rsr_v = next(v for v in seen if v[3] == ["age", "psiC"])
    assert fifo_v[0] == fifo_v[1] == 3000 and fifo_v[2]["steps"] == "3000"
    assert fifo_v[2]["ckpt"] == "1000" and fifo_v[2]["ckpt_every"] == "40"
    assert fifo_v[4]["values"]["iters"] == 3000  # registered R-1, unchanged
    assert rsr_v[0] == rsr_v[1] == 96_000 and rsr_v[2]["steps"] == "96000"
    assert rsr_v[4]["values"]["iters"] == 96_000  # K_2b, per PM-1
    res_bad = {**res, "fifo": {"config": {"iters": 96_000}}}
    assert not p2b.c10_per_arm(
        cfg, ow, launch, res_bad, {}, {}, Path("x"), registered=reg
    )["ok"]


def test_pm1_c_f_on_run_2s_fifo_checkpoint_and_tampered(p2b):
    H = p2b.harness()
    R1 = H["R1"]
    p = Path(R1._R) / ".worktrees/rsr-run2-run/runs/rsr-run2/fifo/seed0/ckpt-003000.pt"
    payload = R1.read_checkpoint(p)
    sub, _sha = R1.load_substrate_model(R1.prereg_config(0, 3000))
    assert p2b.c_f(payload, sub)["ok"]
    t = {**payload, "model": dict(payload["model"])}
    k = next(iter(t["model"]))
    t["model"][k] = t["model"][k].clone() + 1e-7
    assert not p2b.c_f(t, sub)["ok"]
    assert not p2b.c_f(
        {**payload, "optimizer": {"state": {}, "param_groups": [{}]}}, sub
    )["ok"]
    conf = {**payload["meta"]["config"], "freeze_transformer": False}
    assert not p2b.c_f({**payload, "meta": {**payload["meta"], "config": conf}}, sub)[
        "ok"
    ]


# --------------------------------------------------------------------------- #
# Amendment PM-2 (f9eeb48): every fifo read at R-1, every RSR read at K_2b
# --------------------------------------------------------------------------- #

PM2_PLAN = {
    "omega_s": {"0": 2.0, "1": 8.0, "2": 32.0},
    "fifo_iters": 3000,
    "K_2b": 96_000,
}


def test_pm2_fifo_checkpoint_reads_are_at_r1(p2b, tmp_path):
    R1 = p2b.harness()["R1"]
    p2b.set_plan(PM2_PLAN)
    f = R1.checkpoint_path(R1.arm_dir(tmp_path, "fifo", 0), 96_000)
    assert f.name == "ckpt-003000.pt"
    assert R1.checkpoint_path(R1.arm_dir(tmp_path, "psiC", 0), 96_000).name == (
        "ckpt-096000.pt"
    )
    assert (
        R1.checkpoint_path(R1.arm_dir(tmp_path, "fifo", 0), 40).name == "ckpt-000040.pt"
    )
    p2b.set_plan({**PM2_PLAN, "K_2b": 3000})
    assert R1.checkpoint_path(R1.arm_dir(tmp_path, "fifo", 0), 3000).name == (
        "ckpt-003000.pt"
    )


def test_pm2_c6_per_arm_against_its_own_iters(p2b):
    R1 = p2b.harness()["R1"]
    p2b.set_plan(PM2_PLAN)
    cfg = R1.prereg_config(0, 96_000)

    def arm(n):
        r = {"steps_done": n, "steps_requested": n}
        h = {"footer": {"status": "completed"}, "beats": [{}] * 3, "crash": None}
        return r, h

    res, hbs = {}, {}
    for a, n in (("psiC", 96_000), ("fifo", 3000), ("age", 96_000)):
        res[a], hbs[a] = arm(n)
    assert R1.c6_steps(res, hbs, cfg)["ok"]
    res["fifo"], hbs["fifo"] = arm(96_000)
    assert not R1.c6_steps(res, hbs, cfg)["ok"]
    res["fifo"], hbs["fifo"] = arm(3000)
    res["age"], hbs["age"] = arm(3000)
    assert not R1.c6_steps(res, hbs, cfg)["ok"]


def test_pm2_c2_views(p2b, monkeypatch):
    R1 = p2b.harness()["R1"]
    p2b.set_plan({**PM2_PLAN, "K_2b": 6000})
    seen = []

    def fake(beats, ids, cfg, sampling, *, samplings=None):
        steps = {a: sorted(b["step"] for b in v) for a, v in beats.items()}
        seen.append(
            (
                sorted(beats),
                cfg.iters,
                {a: (s[0], s[-1]) for a, s in steps.items()},
                sorted(ids),
                sorted(samplings or {}),
            )
        )
        return {
            "ok": True,
            "warm_steps_compared": 16,
            "c_loss_lm_warm_mismatch_steps": [],
            "steps_compared": len(steps),
        }

    monkeypatch.setattr(p2b, "_C2_ORIG", fake)
    beats = {
        a: [{"step": k} for k in range(6000 if a != "fifo" else 3000)] for a in R1.ARMS
    }
    ids = {a: {k: [k] for k in range(6000 if a != "fifo" else 3000)} for a in R1.ARMS}
    sam = dict.fromkeys(R1.ARMS, "perm")
    cfg = R1.prereg_config(0, 6000)
    out = R1.c2_pairing(beats, ids, cfg, "perm", samplings=sam)
    assert out["ok"] and out["warm_steps_compared"] == 16
    assert out["c_loss_lm_warm_mismatch_steps"] == []
    rsr = next(v for v in seen if v[0] == ["age", "psiC"])
    assert rsr[1] == 6000 and rsr[2] == {"age": (0, 5999), "psiC": (0, 5999)}
    for a in ("psiC", "age"):
        v = next(v for v in seen if v[0] == sorted([a, "fifo"]))
        assert v[1] == 3000 and v[2] == {a: (0, 2999), "fifo": (0, 2999)}
        assert v[3] == sorted([a, "fifo"]) and v[4] == sorted([a, "fifo"])

    def bad(beats, ids, cfg, sampling, *, samplings=None):
        r = fake(beats, ids, cfg, sampling, samplings=samplings)
        return {**r, "ok": "fifo" not in beats}

    monkeypatch.setattr(p2b, "_C2_ORIG", bad)
    assert not R1.c2_pairing(beats, ids, cfg, "perm", samplings=sam)["ok"]


def test_pm2_c4_reads_fifo_at_r1(p2b, monkeypatch):
    R1 = p2b.harness()["R1"]
    p2b.set_plan(PM2_PLAN)
    got = []
    monkeypatch.setattr(
        p2b,
        "_C4_ORIG",
        lambda d, cfg: got.append((Path(d).parent.name, cfg.iters)) or {"ok": True},
    )
    out = R1.c4_measure(R1.arm_dir(Path("x"), "fifo", 0), R1.prereg_config(0, 96_000))
    assert out["ok"] and got == [("fifo", 3000)]


def test_pm2_delta_loss_lm_is_matched_steps_and_labelled(p2b, monkeypatch):
    R1 = p2b.harness()["R1"]
    p2b.set_plan({**PM2_PLAN, "K_2b": 6000})
    calls = []

    def fake(hbs, diags, cfg):
        calls.append(cfg.iters)
        return {
            "iters": cfg.iters,
            "delta_loss_lm": {"psiC": {"how": f"to {cfg.iters}"}},
            "rsr": {"psiC": {"iters_used": cfg.iters}},
        }

    monkeypatch.setattr(p2b, "_TRAIN_ORIG", fake)
    out = R1.analyse_training({}, {}, R1.prereg_config(0, 6000))
    assert calls == [6000, 3000]
    assert out["rsr"]["psiC"]["iters_used"] == 6000  # the RSR arms at their own iters
    d = out["delta_loss_lm"]
    assert d["psiC"]["how"] == "to 3000"  # pairs and blocks over [T_warm, R-1) only
    assert d["label"] == "matched steps <= R-1 (3000); Amendment PM-2"


def test_pm2_delta_loss_lm_real_pairs_only_where_fifo_has_a_row(p2b):
    R1 = p2b.harness()["R1"]
    p2b.set_plan({**PM2_PLAN, "K_2b": 6000})
    hbs = {
        a: {
            "beats": [
                {"step": k, "loss_lm": 1.0 if a == "fifo" else 0.5, "elapsed_s": float(k)}
                for k in range(6000 if a != "fifo" else 3000)
            ]
        }
        for a in R1.ARMS
    }
    out = R1.analyse_training(
        hbs, {a: [] for a in R1.RSR_ARMS}, R1.prereg_config(0, 6000)
    )
    for a in R1.RSR_ARMS:
        blocks = out["delta_loss_lm"][a]["blocks"]
        assert blocks and max(b["steps"][1] for b in blocks) <= 3000
        assert out["delta_loss_lm"][a]["n"] == 3000 - 16


def test_pm2_precheck_before_eval2(p2b, tmp_path):
    R1 = p2b.harness()["R1"]
    plan = {**PM2_PLAN, "K_2b": 6000, "ok": True, "unruled_at_K_gt_3000": []}
    p2b.set_plan(plan)

    def make(arm, n, ckpt):
        d = R1.arm_dir(tmp_path, arm, 0)
        d.mkdir(parents=True, exist_ok=True)
        rows = [{"kind": "beat", "step": k} for k in range(n)]
        (d / "heartbeat.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (d / f"ckpt-{ckpt:06d}.pt").write_bytes(b"x")

    make("psiC", 6000, 6000)
    make("age", 6000, 6000)
    make("fifo", 3000, 3000)
    assert p2b.pm2_precheck(tmp_path, 0, plan)["ok"]
    (R1.arm_dir(tmp_path, "fifo", 0) / "ckpt-003000.pt").unlink()
    r = p2b.pm2_precheck(tmp_path, 0, plan)
    assert not r["ok"] and r["problems"]
    make("fifo", 2999, 3000)  # fifo's beats do not cover [0, R-1)
    assert not p2b.pm2_precheck(tmp_path, 0, plan)["ok"]


def test_pm2_changes_nothing_at_k_equal_r1(p2b, monkeypatch):
    R1 = p2b.harness()["R1"]
    p2b.set_plan({**PM2_PLAN, "K_2b": 3000})
    calls = []
    monkeypatch.setattr(
        p2b,
        "_C2_ORIG",
        lambda b, i, c, s, samplings=None: calls.append(sorted(b)) or {"ok": True},
    )
    monkeypatch.setattr(
        p2b, "_C6_ORIG", lambda r, h, c: calls.append(("c6", sorted(r))) or {"ok": True}
    )
    cfg = R1.prereg_config(0, 3000)
    R1.c2_pairing({a: [] for a in R1.ARMS}, {}, cfg, "perm")
    R1.c6_steps({a: {} for a in R1.ARMS}, {}, cfg)
    assert calls == [sorted(R1.ARMS), ("c6", sorted(R1.ARMS))]


# --------------------------------------------------------------------------- #
# Tier G review 66352af, stage A (2a): F0 / F2, F1, F3
# --------------------------------------------------------------------------- #


def _write(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _preselect_world(
    p2a,
    root: Path,
    *,
    rc2_ok=True,
    fit_smoke=False,
    launch_ok=True,
    slots_ok=True,
    rc2_missing=(),
):
    for s in (0, 1, 2):
        for w in p2a.OMEGAS:
            _write(
                p2a.fit_file(root, s, w),
                {"exit": 0, "smoke": fit_smoke, "controls_failed": []},
            )
        if s not in rc2_missing:
            _write(
                p2a.seed_dir(root, s) / "rc2.json",
                {"exit": 0 if rc2_ok else 3, "controls": {"R-C2": {"ok": rc2_ok}}},
            )
    _write(root / "launch.json", {"exit": 0 if launch_ok else 3})
    _write(root / "launch-slots.json", {"exit": 0 if slots_ok else 3})


def test_f0_preselect_gate(p2a, tmp_path):
    _preselect_world(p2a, tmp_path)
    assert p2a.preselect_gate(tmp_path, 0, standin=None)["ok"]
    for kw in (
        {"rc2_ok": False},
        {"rc2_missing": (2,)},
        {"fit_smoke": True},
        {"launch_ok": False},
        {"slots_ok": False},
    ):
        root = tmp_path / str(sorted(kw.items()))
        _preselect_world(p2a, root, **kw)
        g = p2a.preselect_gate(root, 0, standin=None)
        assert not g["ok"] and g["problems"], kw
    # the stand-in smoke needs only its own smoke fits (no launch, no slot check)
    root = tmp_path / "smoke"
    for w in p2a.OMEGAS:
        _write(
            p2a.fit_file(root, 0, w), {"exit": 3, "smoke": True, "controls_failed": []}
        )
    assert p2a.preselect_gate(root, 0, standin=32)["ok"]


def test_f0_select_gates_before_tune_opens_statically():
    tree = ast.parse((EXP / "phase2a.py").read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_seed"
    )
    calls = [
        c.func.id
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
    ]
    assert "preselect_gate" in calls
    src = ast.get_source_segment((EXP / "phase2a.py").read_text(), fn)
    assert (
        src.index("preselect_gate(")
        < src.index("tune_closure(")
        < src.index("docs_by_id(")
    )


def _run2a(args, tmp):
    return subprocess.run(
        [str(PY), str(EXP / "run.py"), "2a", *args, "--out-root", str(tmp)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_f0_select_refuses_without_r_c2_and_generates_no_tune_id(p2a, tmp_path):
    _preselect_world(p2a, tmp_path, rc2_missing=(1,))
    r = _run2a(["select", "--seed", "0"], tmp_path)
    assert r.returncode == 3, r.stderr[-2000:]
    d = json.loads((p2a.seed_dir(tmp_path, 0) / "select.json").read_text())
    assert "R-C2" in json.dumps(d["controls"]["R-C6.preselect"])
    cen = d["controls"]["R-SEAL.census"]
    assert cen["n_sealed_generated"] == 0
    for rows in cen["per_seed"].values():
        assert all(v["count"] == 0 for k, v in rows.items() if k.startswith("TUNE"))
    assert not (tmp_path / "tune").exists()


def test_f1_deploy_refuses_unless_select_exited_0(p2a, tmp_path):
    sd = p2a.seed_dir(tmp_path, 0)
    sel = {
        "seed": 0,
        "omega_s": 2.0,
        "phi_sha256": "0" * 64,
        "phi_path": str(sd / "x.pt"),
    }
    _write(p2a.selection_file(tmp_path, 0), sel)
    (sd / "selection.sha256").write_text(C_sha(p2a.selection_file(tmp_path, 0)) + "\n")
    _write(
        sd / "select.json", {"exit": 3, "smoke": False, "controls_failed": ["R-C7.tune"]}
    )
    r = _run2a(["deploy", "--seed", "0"], tmp_path)
    assert r.returncode == 3, r.stderr[-2000:]
    d = json.loads((sd / "deploy.json").read_text())
    assert "select" in (d.get("error") or "")
    assert "deploy" not in d and not (tmp_path / "heldout").exists()
    assert not p2a.select_done({"exit": 3, "smoke": False, "controls_failed": []}, None)
    assert p2a.select_done({"exit": 0, "smoke": False}, None)
    assert p2a.select_done({"exit": 3, "smoke": True, "controls_failed": []}, 32)
    assert not p2a.select_done({"exit": 3, "smoke": True, "controls_failed": ["x"]}, 32)
    assert not p2a.select_done(None, None)


def C_sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_f3_rank_pairs_keep_only_finite_g(p2a):
    class Doc:
        def __init__(self, i):
            self.doc_id = i

    n = 48
    D = torch.full((n, n), float("nan"))
    low = torch.ones(n, n, dtype=torch.bool).tril(-1)
    D[low] = 0.01
    D2 = D.clone()
    D2[40, 3] = float("inf")  # an infinite knockout cell
    D2[30, 5] = float("-inf")
    blob = {"Delta": {1: D, 2: D2}}
    t = p2a.rank_fit_tensors(blob, [Doc(1), Doc(2)])
    raw = torch.stack([p2a._H["OP"].target_of(blob["Delta"][i]) for i in (1, 2)])
    assert torch.equal(t["mask"], t["mask"] & torch.isfinite(raw))
    assert not bool(t["mask"][1][~torch.isfinite(raw[1])].any())
    assert bool(t["mask"][0][low].all())  # a finite document loses nothing
    assert t["n_nonfinite_excluded"] > 0
    assert torch.isfinite(t["G"][t["mask"]]).all()
    # lookahead-room's discounted_returns nan_to_num's a ±inf Delta into a HUGE finite
    # G, so "finite G" is read off Delta: every return that draws on an infinite
    # Delta[t + k][i] (k >= 0) is out of the pair set, every other cell stays in.
    m = t["mask"][1]
    assert not bool(m[4:41, 3].any()) and bool(m[41:, 3].all())
    assert not bool(m[6:31, 5].any()) and bool(m[31:, 5].all())
    assert int((low & ~m).sum()) == t["n_nonfinite_excluded"] == (41 - 4) + (31 - 6)


# --------------------------------------------------------------------------- #
# Tier G review 66352af, stage B (2b): LB-1, LB-2, footers, F6
# --------------------------------------------------------------------------- #


def test_lb1_amendment_2bS_heading_in_the_file_s_own_convention(p2b, tmp_path):
    body = "- 2a ledger sha256: `" + "b" * 64 + "`\n- K_2b: 4500\n"
    for head in (
        "## Amendment 2b-S (2026-10-21, PM): the 2b smoke and projection",
        "## AMENDMENT 2b-S (2026-10-21)",
        "## amendment 2b-s",
    ):
        pr = tmp_path / "P.md"
        pr.write_text(f"x\n\n{head}\n\nprose\n{body}\n## Amendment PM-9\n- K_2b: 1\n")
        assert p2b.amendment_2bS(pr) == {"ledger_2a_sha256": "b" * 64, "K_2b": 4500}, head
    pr.write_text("x\n\n## Amendment 2b-Sx\n" + body)
    assert p2b.amendment_2bS(pr) is None


def test_lb1_f6_measure_order_before_eval2():
    """Every check that needs no EVAL2 document, then EVAL2's closure, then the ES
    model, then run 2's measure_seed (EVAL2 read)."""
    src = (EXP / "phase2b.py").read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "measure_stage"
    )
    body = ast.get_source_segment(src, fn)
    order = [
        body.index(x)
        for x in (
            "training_footers(",
            "pre_eval2_controls(",
            "eval2_closure(",
            "c_es(",
            "R2.measure_seed(",
        )
    ]
    assert order == sorted(order), order
    pre = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "pre_eval2_controls"
    )
    seg = ast.get_source_segment(src, pre)
    for name in (
        "c6_steps(",
        "c10_launch_values(",
        "_cheap_controls(",
        "c4_measure(",
        "c_f_at_r1(",
        "pm2_precheck(",
        "c11_prime(",
        "c_k(",
        "c_omega(",
    ):
        assert name in seg, name
    sm = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "smoke_pre"
    )
    assert "pre_eval2_controls(" in ast.get_source_segment(src, sm)


def _pre_world(p2b, monkeypatch, *, c10=True, c2=True):
    H = p2b.harness()
    R1 = H["R1"]
    res = {a: {"steps_done": 3000, "config": {"arm_label": "x"}} for a in R1.ARMS}
    hbs = {a: {"beats": []} for a in R1.ARMS}
    monkeypatch.setattr(
        p2b,
        "read_arms",
        lambda cfg, out: {
            "launch": {"loss": "rank"},
            "results": res,
            "hbs": hbs,
            "jobs": dict.fromkeys(R1.ARMS),
            "rcs": dict.fromkeys(R1.ARMS, 0),
        },
    )
    monkeypatch.setattr(p2b, "training_footers", lambda out, seeds: {"ok": True})
    monkeypatch.setattr(p2b, "pm2_precheck", lambda out, s, plan: {"ok": True})
    monkeypatch.setattr(R1, "c6_steps", lambda r, h, c: {"ok": True})
    monkeypatch.setattr(R1, "c10_launch_values", lambda *a, **k: {"ok": c10})

    def cheap(doc, cfg, out, r, h):
        doc["controls"].update(C2={"ok": c2}, C8={"ok": True}, C3={"ok": True})
        return {}

    monkeypatch.setattr(R1, "_cheap_controls", cheap)
    monkeypatch.setattr(R1, "c4_measure", lambda d, c: {"ok": True})
    monkeypatch.setattr(p2b, "c_f_at_r1", lambda cfg, out: {"ok": True})
    monkeypatch.setattr(p2b, "c11_prime", lambda *a: {"ok": True})
    monkeypatch.setattr(H["R2"], "c11_target", lambda *a: {"ok": True})
    monkeypatch.setattr(p2b, "c_k", lambda *a: {"ok": True})
    monkeypatch.setattr(p2b, "c_omega", lambda *a: {"ok": True})
    monkeypatch.setattr(p2b, "read_2a", lambda p: {"selection": {0: {"k_conv": 1}}})
    plan = {"seeds": [0], "K_2b": 3000, "fifo_iters": 3000, "omega_s": {"0": 2.0}}
    return R1, plan


@pytest.mark.parametrize("c10,c2", [(True, True), (False, True), (True, False)])
def test_lb2_a_c10_or_c2_failure_stops_the_seed_before_eval2(p2b, monkeypatch, c10, c2):
    R1, plan = _pre_world(p2b, monkeypatch, c10=c10, c2=c2)
    p2b.set_plan(plan)
    cfg = R1.prereg_config(0, 3000)
    out = p2b.pre_eval2_controls(
        cfg, p2b.owner_values(3000), Path("x"), plan, ledger_2a=Path("l.json")
    )
    assert out["ok"] == (c10 and c2)
    if not c10:
        assert "C10" in out["failed"]
    if not c2:
        assert "C2" in out["failed"]


def test_lb2_footers_incomplete_measure_refuses_with_no_eval2_id(p2b, tmp_path):
    plan = {
        "ok": True,
        "unruled_at_K_gt_3000": [],
        "seeds": [0, 1, 2],
        "omega_s": {"0": 2.0, "1": 8.0, "2": 32.0},
        "K_2b": 3000,
        "fifo_iters": 3000,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    r = subprocess.run(
        [
            str(PY),
            str(EXP / "run.py"),
            "2b",
            "measure",
            "--seed",
            "0",
            "--out-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 3, r.stderr[-2000:]
    d = json.loads((tmp_path / "measure-seed0.json").read_text())
    assert d["refused_before_eval2"] is True
    assert "footers" in d["controls_failed"]
    cen = d["controls"]["R-SEAL.census"]
    assert cen["n_sealed_generated"] == 0
    for rows in cen["per_seed"].values():
        assert all(v["count"] == 0 for k, v in rows.items() if k.startswith("EVAL2"))
    assert not (tmp_path / "heldout").exists()


@pytest.mark.parametrize(
    ("env", "msg"),
    [
        ({"SMOKE_FIFO_STEPS": "48"}, "SMOKE_* needs SMOKE=1"),
        (
            {"SMOKE": "1", "OUT": "/tmp/x", "JOB_SUFFIX": "-s", "SMOKE_STEPS": "9x"},
            "SMOKE_STEPS must be a positive integer",
        ),
        (
            {"SMOKE": "1", "OUT": "/tmp/x", "JOB_SUFFIX": "-s", "SMOKE_SEEDS": "0 7"},
            "SMOKE_SEEDS must be seeds from 0 1 2",
        ),
    ],
)
def test_lb2_smoke_variant_refusals(tmp_path, env, msg):
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GAMMA": "0.9",
        "WD": "0.01",
        "O5": "eval",
        "C1_FLAGS": "--frozen-model-mode eval",
    }
    p = subprocess.run(
        ["/bin/zsh", str(EXP / "launch_2b.zsh")],
        env={**base, **env},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 3, p.stderr
    assert msg in p.stderr


def test_lb2_smoke_needs_no_fifo_iters_in_the_plan_but_the_run_does(tmp_path):
    """Run from the repo root (the launcher parses plan.json with .venv's python). The
    smoke's OUT already holds a launch-git.json, so it refuses AFTER the fifo block
    (head or already-launched) and never trains; the run refuses AT the fifo block,
    before it creates anything."""
    plan = tmp_path / "plan.json"
    om = {"0": 8.0, "1": 8.0, "2": 8.0}
    plan.write_text(json.dumps({"ok": True, "omega_s": om, "K_2b": 3000}))
    out = tmp_path / "o"
    out.mkdir()
    (out / "launch-git.json").write_text("{}")
    base = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "GAMMA": "0.9",
        "WD": "0.01",
        "O5": "eval",
        "C1_FLAGS": "--frozen-model-mode eval",
        "PLAN": str(plan),
    }
    smoke = {
        "SMOKE": "1",
        "OUT": str(out),
        "JOB_SUFFIX": "-s",
        "SMOKE_SEEDS": "0",
        "SMOKE_FIFO_STEPS": "48",
        "SMOKE_STEPS": "96",
    }
    p = subprocess.run(
        ["/bin/zsh", str(EXP / "launch_2b.zsh")],
        env={**base, **smoke},
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 3 and "no fifo_iters" not in p.stderr, p.stderr
    assert "launched already" in p.stderr or "clean committed head" in p.stderr, p.stderr
    p = subprocess.run(
        ["/bin/zsh", str(EXP / "launch_2b.zsh")],
        env=base,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 3 and "no fifo_iters" in p.stderr, p.stderr


def test_save_tensor_creates_the_seed_dir(p2a, tmp_path):
    """2a launch 2 (90baaf4): the first fit of a seed to finish crashed because
    ``seed<s>/`` did not exist when it saved phi. Every tensor save goes through
    ``save_tensor``, which creates the directory."""
    p = p2a.phi_file(tmp_path / "runs" / "rsr-run3-2a", 0, 2.0, "kconv")
    assert not p.parent.exists()
    p2a.save_tensor({"w": torch.zeros(2)}, p)
    assert torch.equal(torch.load(p)["w"], torch.zeros(2))
    src = Path(p2a.__file__).read_text()
    assert src.count("torch.save(") == 1  # only inside save_tensor itself
