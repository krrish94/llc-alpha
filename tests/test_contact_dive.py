"""Contact deep dive D5 (ADR 0007): contact-maintenance reward, gain-residual parameterisation,
external-torque feature, output filter, window data and the PPO environment with the new terms."""
import json
from dataclasses import replace

import numpy as np
import torch

from llc.controllers.pd import home_inertia_diag
from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.actor_critic import Critic, RewardConfig, contact_reward
from llc.learning.data import reference_pool_v2, sample_batch_v2, surface_depth
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import HistoryBuffer, OutputFilter, Policy, PolicyConfig, convert_to_gain_residual, expand_ext_torque
from llc.learning.ppo import GaussianActor, PPOConfig, PPOEnv, collect
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model


def _cfg(kind="pd_residual", **kw) -> PolicyConfig:
    shared = json.load(open("runs/d1/gains_shared.json"))["spec"]
    return PolicyConfig(kind=kind, history=4, history_stride=5, history_dense=8, rate_hz=1000.0, latency_s=0.0,
                        inertia_diag=tuple(home_inertia_diag(NominalModel()).tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]), **kw)


def test_contact_reward_prefers_the_target_force_and_charges_lost_contact():
    cfg = RewardConfig(w_press=1.0, press_stiffness=1000.0)  # the ADR 0007 target law (ADR 0008 lowered the default to 500 N/m)
    B = 4
    e = torch.zeros(B, 7)
    tool = torch.zeros(B, 3)
    tool_ref = torch.zeros(B, 3)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * B)
    depth = torch.tensor([0.01, 0.01, 0.01, 0.0])  # 1 cm inside -> target 15 N (cap); the last robot's reference is outside
    has = torch.tensor([True, True, True, True])
    touch = torch.tensor([15.0, 0.0, 22.0, 15.0])
    r = contact_reward(cfg, e, tool, tool_ref, normal, depth, has, touch) / cfg.reward_scale
    assert r[0] > r[2] > r[1]  # at the target > pressing 7 N above it > lost contact
    assert abs(r[1] - (cfg.w_feas - cfg.w_pos)) < 1e-3  # lost contact: only the tangential kernel is left (joint kernel removed)
    assert r[3] == 0.0  # outside the surface the terms are off
    # a tangential error is charged, a normal error is not
    tool_err_n = tool_ref + torch.tensor([[0.0, 0.0, 0.02]] * B)
    tool_err_t = tool_ref + torch.tensor([[0.02, 0.0, 0.0]] * B)
    rn = contact_reward(cfg, e, tool, tool_err_n, normal, depth, has, touch)
    rt = contact_reward(cfg, e, tool, tool_err_t, normal, depth, has, touch)
    assert torch.allclose(rn[:3], r[:3] * cfg.reward_scale) and (rt[:3] < rn[:3]).all()


def test_gain_residual_decodes_bounded_multipliers_and_conversion_is_exact():
    torch.manual_seed(0)
    pol = Policy(_cfg())
    with torch.no_grad():
        pol.head.weight.normal_(0, 0.5); pol.head.bias.normal_(0, 0.5)
    gp = convert_to_gain_residual(pol)
    obs = torch.randn(6, pol.cfg.obs_dim)
    h0, h1 = pol.decode(pol(obs)), gp.decode(gp(obs))
    assert torch.allclose(h1["kp_scale"], torch.ones(6, 7)) and torch.allclose(h1["kd_scale"], torch.ones(6, 7))
    assert torch.allclose(h0["residual"], h1["residual"])
    q, qd, qr, qdr, g = (torch.randn(6, 7) for _ in range(5))
    assert torch.allclose(pol.fast_torque(h0, q, qd, qr, qdr, g), gp.fast_torque(h1, q, qd, qr, qdr, g), atol=1e-5)
    out = torch.randn(100, gp.cfg.out_dim) * 50.0  # saturating outputs stay inside the ranges
    h = gp.decode(out)
    assert h["kp_scale"].min() >= gp.cfg.gain_scale_range[0] - 1e-6 and h["kp_scale"].max() <= gp.cfg.gain_scale_range[1] + 1e-6
    assert h["kd_scale"].min() >= gp.cfg.damping_scale_range[0] - 1e-6 and h["kd_scale"].max() <= gp.cfg.damping_scale_range[1] + 1e-6


def test_ext_torque_expansion_is_exact_and_feature_sees_an_external_load():
    torch.manual_seed(1)
    pol = Policy(_cfg())
    with torch.no_grad():
        for p in pol.parameters():
            p.normal_(0, 0.1)
    ext = expand_ext_torque(pol)
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    q = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]] * 2)
    qd = torch.zeros(2, 7)
    b0, b1 = HistoryBuffer(pol.cfg, q, qd), HistoryBuffer(ext.cfg, q, qd, model.inverse_dynamics, 1e-3)
    g = model.bias(q, qd)
    load = torch.zeros(2, 7); load[1, 1] = 5.0  # robot 1 applies 5 Nm more than gravity at joint 2 while staying at rest: an external load of -5 Nm
    for _ in range(12):
        b0.push(q, qd, g + load, q); b1.push(q, qd, g + load, q)
    lim = pol.limits
    assert torch.allclose(pol(b0.observation(q, qd, q, qd, lim)), ext(b1.observation(q, qd, q, qd, lim)), atol=1e-5)
    est = b1.ext[:, -1]
    assert abs(est[0, 1]) < 1e-3 and abs(est[1, 1] - 5.0) < 1e-3


def test_output_filter_smooths_and_passes_through_at_zero():
    f0, f1 = OutputFilter(0.0), OutputFilter(0.01, 1e-3)
    a = {"residual": torch.ones(2, 7)}
    b = {"residual": torch.zeros(2, 7)}
    assert f0(a) is a
    f1(b)
    y = f1(a)["residual"]
    assert torch.allclose(y, torch.full((2, 7), 0.1))  # one step of a 10 ms filter
    for _ in range(100):
        y = f1(a)["residual"]
    assert (y > 0.99).all()


def test_window_data_has_depth_sign_and_mixed_starts_contain_the_release():
    rng = np.random.default_rng(0)
    pool = reference_pool_v2(4, "train")
    ref = pool["blocked"][0]
    depth = surface_depth(ref["meta"]["tool_ref"], ref["meta"]["surface"])
    assert depth.max() > 0.03 and depth.min() < -0.02  # 4 to 8 cm inside, 3 cm off at the ends
    q_ref, qd_ref, q0, qd0, labels, f_ext, priv = sample_batch_v2(pool, {"blocked": 1.0}, 16, 2000, 1e-3, rng, start_mode="mixed")
    assert priv["tool_ref"].shape == (16, 2000, 3) and priv["depth_ref"].shape == (16, 2000) and priv["has_surface"].all()
    inside = priv["depth_ref"] > 0
    assert inside.any(1).all()  # every window contains some of the block
    first_in = inside.argmax(1)
    assert (first_in <= 220).sum() >= 4 and (first_in >= 300).sum() >= 4  # crossing starts and approach starts both present
    assert not inside[:, 0].any()  # no window starts with the reference inside the surface (no initial penetration)


def test_ppo_env_with_contact_terms_and_gain_residual_runs():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    policy = convert_to_gain_residual(Policy(_cfg()))
    policy.cfg = replace(policy.cfg, hold_steps=10, output_tau=0.02)
    B, T = 3, 200
    batch = MjBatch([sample_instance(rng, payload_mass=1.0) for _ in range(B)], seed=0, threads=1)
    pool = reference_pool_v2(4, "train")
    pcfg = PPOConfig(hold_steps=10, minibatch=16, epochs=1, start_mode="mixed", noise_rho=0.5, reflex_force=25.0, reflex_tau=0.02, reflex_terminate=False)
    rcfg = RewardConfig(w_press=1.0, force_soft=15.0, force_cap=15.0)
    env = PPOEnv(batch, policy, RigidBodyModel(load_model(), dtype=torch.float32), pool, {"contact2": 0.5, "blocked": 0.5}, T, rng, rcfg, pcfg)
    roll = collect(env, GaussianActor(policy, 0.1), Critic(policy.cfg.obs_dim, 4))
    assert roll["obs"].shape == (T // 10, B, policy.cfg.obs_dim) and roll["act"].shape[-1] == 21
    assert torch.isfinite(roll["rew"]).all()


def test_filtered_force_reward_discounts_the_impact_transient():
    """With force_tau the force penalties see the low-pass filtered force: a 5 ms impact of 300 N
    costs far less than a sustained 300 N, while a sustained force is penalised the same."""
    from llc.learning.actor_critic import step_reward

    cfg_raw, cfg_f = RewardConfig(force_soft=15.0), RewardConfig(force_soft=15.0, force_tau=0.02)
    dt, lim = 1e-3, torch.full((7,), 50.0)
    z = torch.zeros(1, 7)

    def total(cfg, force_series):
        a = min(1.0, dt / cfg.force_tau) if cfg.force_tau > 0 else 1.0
        f_prev, f_filt, tot = torch.zeros(1), torch.zeros(1), 0.0
        for f in force_series:
            f_now = torch.tensor([f])
            f_filt = f_filt + a * (f_now - f_filt) if a < 1.0 else f_now
            tot += float(step_reward(z, z, z, z, f_filt, f_prev, lim, cfg, dt).sum())
            f_prev = f_filt
        return tot

    impact = [300.0] * 5 + [0.0] * 95
    sustained = [300.0] * 500
    assert total(cfg_f, impact) > 5 * total(cfg_raw, impact)  # the transient is discounted (penalties are negative)
    assert abs(total(cfg_f, sustained) - total(cfg_raw, sustained)) / abs(total(cfg_raw, sustained)) < 0.1  # a sustained force is still charged (after the 20 ms rise)


def test_blocked_stop_and_wait_metrics_on_the_pd():
    """The blocked family scores 'stop at the surface and wait': drift from the first-contact point,
    motion while in contact, and the force terms; defined on every blocked case."""
    import json

    from llc.eval.parallel import CaseSpec, evaluate

    spec = json.load(open("runs/d1/gains_shared.json"))["spec"]
    res = evaluate({"pd": {"kind": "pd_gravity", "kp": spec["kp"], "kd": spec["kd"]}}, [CaseSpec("blocked", "val", i, protocol="v2") for i in range(3)], workers=3)["pd"]
    for m in res:
        assert np.isfinite(m["blocked_drift"]) and 0.0 <= m["blocked_drift"] < 0.3
        assert np.isfinite(m["blocked_motion"]) and 0.0 <= m["blocked_motion"] <= 1.0
        assert np.isfinite(m["composite_cap"])
