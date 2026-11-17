"""Impedance deep dive D6 (ADR 0008): contact-impedance oracle, feedforward baseline, compliant
surfaces, the blocked anchor of the reward, the target preview and the distillation labels."""
import json
from dataclasses import replace

import numpy as np
import torch

from llc.controllers.factory import build_controller
from llc.controllers.learned import LearnedController
from llc.eval.cases_v2 import make_case_v2
from llc.eval.metrics import compute_metrics
from llc.eval.parallel import CaseSpec
from llc.eval.rollout import rollout
from llc.learning.actor_critic import BlockedAnchor
from llc.learning.data import surface_depth
from llc.learning.distill import collect_labels, label_loss
from llc.learning.policy import HistoryBuffer, expand_preview, load_policy
from llc.sim.nominal import NominalModel
from llc.sim.panda import PandaSim

WARMSTART = "runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt"


def _pd_spec():
    return json.load(open("runs/d1/gains_shared.json"))["spec"]


def _run(spec, case, nominal):
    ep = rollout(PandaSim(case.perturbation, seed=case.seed), build_controller(spec, nominal, case), case)
    return ep, compute_metrics(ep, case)


def test_oracle_caps_the_force_and_holds_the_contact_point_on_a_blocked_table():
    nominal = NominalModel()
    case = make_case_v2("blocked", "val", 1, nominal)
    assert case.meta["surface"] == "table"
    _, pd = _run(_pd_spec(), case, nominal)
    _, orc = _run({"kind": "oracle", "base": _pd_spec()}, case, nominal)
    assert orc["excess_force_cap"] < 0.5 * pd["excess_force_cap"]  # what remains is the blind approach impact, not the oracle's
    assert orc["blocked_drift"] < 0.005 and orc["blocked_drift"] < pd["blocked_drift"]
    assert orc["apparent_stiffness"] < 0.7 * pd["apparent_stiffness"]
    assert np.isfinite(orc["contact_impedance"])


def test_oracle_is_the_base_controller_away_from_surfaces():
    nominal = NominalModel()
    case = make_case_v2("free", "val", 0, nominal)
    ep0, _ = _run(_pd_spec(), case, nominal)
    ep1, _ = _run({"kind": "oracle", "base": _pd_spec()}, case, nominal)
    assert np.allclose(ep0.tau, ep1.tau)


def test_pd_feedforward_tracks_a_free_reference():
    nominal = NominalModel()
    case = make_case_v2("free", "val", 1, nominal)
    _, pd = _run(_pd_spec(), case, nominal)
    _, ff = _run({**_pd_spec(), "kind": "pd_feedforward"}, case, nominal)
    assert np.isfinite(ff["pos_rmse"]) and ff["pos_rmse"] < 1.2 * pd["pos_rmse"] and not ff["diverged"]


def test_compliant_surface_lets_the_tool_sink():
    nominal = NominalModel()
    rigid = make_case_v2("blocked", "val", 1, nominal)
    soft = make_case_v2("blocked", "val", 1, nominal, surface_stiffness=500.0)
    assert soft.id.endswith("-k500-001") and CaseSpec("blocked", "val", 1, protocol="v2", surface_stiffness=500.0).id == soft.id
    ep_r, m_r = _run(_pd_spec(), rigid, nominal)
    ep_s, m_s = _run(_pd_spec(), soft, nominal)
    sink = lambda ep: surface_depth(ep.tool_pos, "table")[ep.n_tool_contacts > 0].max()
    assert sink(ep_r) < 0.004 and sink(ep_s) > 0.02
    assert m_s["f_mean"] < 0.6 * m_r["f_mean"] if np.isfinite(m_s.get("f_mean", np.nan)) else True  # the PD presses far less on the soft surface


def test_blocked_anchor_holds_the_first_contact_point_only_in_blocked_windows():
    anchor = BlockedAnchor(["blocked", "contact2", "blocked"], torch.tensor([True, True, False]))
    tool = torch.tensor([[0.5, 0.0, 0.1], [0.5, 0.1, 0.1], [0.5, 0.2, 0.1]])
    ref = torch.tensor([[0.6, 0.0, 0.05], [0.6, 0.1, 0.05], [0.6, 0.2, 0.05]])
    depth = torch.tensor([0.02, 0.02, 0.02])
    tan = anchor.tangential_target(tool, ref, depth, torch.tensor([0.0, 5.0, 5.0]))  # no touch on the first: no anchor yet
    assert torch.allclose(tan, ref)
    tan = anchor.tangential_target(tool, ref, depth, torch.tensor([5.0, 5.0, 5.0]))
    assert torch.allclose(tan[0], tool[0]) and torch.allclose(tan[1], ref[1]) and torch.allclose(tan[2], ref[2])
    tan = anchor.tangential_target(tool + 0.1, ref + 0.2, depth, torch.tensor([5.0, 5.0, 5.0]))  # the anchor stays where it was set
    assert torch.allclose(tan[0], tool[0])


def test_preview_expansion_is_exact_and_the_controller_runs():
    policy, _ = load_policy(WARMSTART)
    pe = expand_preview(policy, 5, 10)
    assert pe.cfg.obs_dim == policy.cfg.obs_dim + 35
    B = 2
    q, qd = torch.randn(B, 7) * 0.2, torch.randn(B, 7) * 0.1
    qr, qdr = q + 0.03, torch.zeros(B, 7)
    b0, b1 = HistoryBuffer(policy.cfg, q, qd), HistoryBuffer(pe.cfg, q, qd)
    for _ in range(40):
        b0.push(q, qd, torch.zeros(B, 7), qr); b1.push(q, qd, torch.zeros(B, 7), qr)
    o0 = b0.observation(q, qd, qr, qdr, policy.limits); o1 = b1.observation(q, qd, qr, qdr, pe.limits, qr[:, None] + 0.1 * torch.randn(B, 5, 7))
    assert torch.allclose(policy(o0), pe(o1), atol=1e-6)
    nominal = NominalModel()
    case = make_case_v2("hold", "val", 0, nominal)
    ep = rollout(PandaSim(case.perturbation, seed=case.seed), LearnedController(pe, nominal), case)
    assert not ep.diverged


def test_distillation_labels_reproduce_the_student_in_free_space():
    """Teacher = the oracle around the warm start; away from surfaces the labels are the warm start's own
    torques, which the student (the same network at 10 ms decisions) reproduces at its decision steps."""
    teacher = {"kind": "oracle", "base": {"kind": "learned", "path": WARMSTART}}
    policy, _ = load_policy(WARMSTART)
    import tempfile
    from llc.learning.policy import Policy, save_policy

    student = Policy(replace(policy.cfg, hold_steps=10)); student.load_state_dict(policy.state_dict()); student.eval()
    with tempfile.TemporaryDirectory() as d:
        save_policy(f"{d}/s.pt", student)
        batches = collect_labels(f"{d}/s.pt", teacher, [CaseSpec("hold", "val", 0, protocol="v2")], [False], workers=1)
    data = {k: torch.as_tensor(v) for k, v in batches[0].items() if k not in ("diverged", "case_id", "regime")}
    assert data["obs"].shape[0] > 100 and data["obs"].shape[1] == student.cfg.obs_dim
    with torch.no_grad():
        assert float(label_loss(student, data)) < 1e-4
