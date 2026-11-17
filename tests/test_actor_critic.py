"""Actor-critic and decoupled-gradient pieces: TD(lambda) targets, reward scale and sign, and the
straight-through step (values from the simulator, gradients from the model)."""
import numpy as np
import torch

from llc.learning.actor_critic import Critic, RewardConfig, step_reward, td_lambda_targets
from llc.learning.decoupled import DecoupledBatch, DynamicsNet, context_vector
from llc.learning.mjgrad import MjBatch
from llc.sim.panda import HOME, TORQUE_LIMITS, DynamicsPerturbation


def test_td_lambda_targets_match_monte_carlo_for_lambda_one():
    torch.manual_seed(0)
    critic = Critic(obs_dim=4, n_future=0)
    B, T, D = 2, 5, critic.priv_dim + critic.state_dim
    x = torch.randn(B, T + 1, D)
    r = torch.randn(B, T)
    gamma = 0.9
    R = td_lambda_targets(critic, x, r, gamma, lam=1.0)
    with torch.no_grad():
        v_T = critic(x[:, T])
    for b in range(B):
        for t in range(T):
            mc = sum(gamma ** (k - t) * r[b, k] for k in range(t, T)) + gamma ** (T - t) * v_T[b]
            assert torch.isclose(R[b, t], mc, atol=1e-5)


def test_step_reward_prefers_tracking_and_penalises_force():
    cfg = RewardConfig()
    limits = torch.tensor(TORQUE_LIMITS, dtype=torch.float32)
    z = torch.zeros(1, 7)
    good = step_reward(z, z, z, z, torch.zeros(1), torch.zeros(1), limits, cfg, 1e-3)
    bad_track = step_reward(torch.full((1, 7), 0.2), z, z, z, torch.zeros(1), torch.zeros(1), limits, cfg, 1e-3)
    pushing = step_reward(z, z, z, z, torch.full((1,), 120.0), torch.full((1,), 120.0), limits, cfg, 1e-3)
    assert good > bad_track and good > pushing
    floor = cfg.w_lin * 0.01 / cfg.lin_scale  # the linear restoring term at zero error (sqrt(1e-4))
    assert abs(float(good) - (cfg.w_pos + cfg.w_vel - floor) * cfg.reward_scale) < 1e-6


def test_decoupled_step_values_from_simulator_and_gradients_from_model():
    torch.manual_seed(0)
    pert = DynamicsPerturbation()
    sim_ref = MjBatch([pert], seed=0)
    sim_ref.reset(HOME[None], np.zeros((1, 7)))
    dec = DecoupledBatch(MjBatch([pert], seed=0), DynamicsNet(), np.random.default_rng(0))
    dec.perturb([pert])
    dec.reset(HOME[None], np.zeros((1, 7)))
    q, qd = (x.float() for x in dec.state())
    tau = torch.zeros(1, 7, requires_grad=True)
    q_n, qd_n, touch_n, _ = dec.step(q, qd, tau)
    with torch.no_grad():
        q_s, qd_s, t_s, _ = sim_ref.step(q.clone(), qd.clone(), torch.zeros(1, 7))
    assert torch.allclose(q_n.detach(), q_s.float()) and torch.allclose(qd_n.detach(), qd_s.float())
    (q_n.square().sum() + qd_n.square().sum()).backward()
    assert tau.grad is not None and torch.isfinite(tau.grad).all() and tau.grad.abs().sum() > 0  # gradient flowed through the model
    assert dec.replay.n == 1
    assert context_vector(pert).shape == (34,) and np.allclose(context_vector(pert), 0.0)


def test_ref_history_observation_matches_shape_and_encodes_target_motion():
    import torch
    from llc.learning.policy import HistoryBuffer, Policy, PolicyConfig
    from llc.sim.panda import TORQUE_LIMITS

    cfg = PolicyConfig(kind="pd_residual", history=3, history_stride=2, history_dense=4, rate_hz=1000.0, latency_s=0.0, ref_history=True)
    policy = Policy(cfg)
    q = torch.zeros(2, 7); lim = torch.tensor(TORQUE_LIMITS, dtype=torch.float32)
    buf = HistoryBuffer(cfg, q, q)
    for k in range(8):
        buf.push(q, q, q, q + 0.01 * k)  # target moves by 10 mrad per step, robot stays
    obs = buf.observation(q, q, q + 0.07, q, lim)
    assert obs.shape == (2, cfg.obs_dim) and policy(obs).shape == (2, 7)
    ref_part = obs[0, 4 * 7:].reshape(cfg.n_hist, 4 * 7)[:, 3 * 7:]  # per sample: q, qd, tau, target
    assert torch.allclose(ref_part[-1], torch.zeros(7))  # newest target sample is the current target
    assert torch.allclose(ref_part[-2, 0], torch.tensor(-0.01 / 0.2))  # previous dense sample: 10 mrad behind
    assert (ref_part[0, 0] < ref_part[-2, 0]).item()  # oldest strided sample is further behind
