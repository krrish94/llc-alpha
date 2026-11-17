"""PPO environment and controller decision period (deep dive D4)."""
import json

import numpy as np
import torch

from llc.controllers.pd import home_inertia_diag
from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.actor_critic import Critic, RewardConfig
from llc.learning.data import reference_pool_v2
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import Policy, PolicyConfig
from llc.learning.ppo import GaussianActor, PPOConfig, PPOEnv, collect, gae, ppo_update
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model


def _cfg(hold: int) -> PolicyConfig:
    shared = json.load(open("runs/d1/gains_shared.json"))["spec"]
    return PolicyConfig(kind="pd_residual", history=4, history_stride=5, history_dense=8, rate_hz=1000.0, latency_s=0.0, hold_steps=hold,
                        inertia_diag=tuple(home_inertia_diag(NominalModel()).tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))


def test_env_rollout_shapes_and_reflex_masking():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    policy = Policy(_cfg(10))
    B, T = 3, 200
    batch = MjBatch([sample_instance(rng, payload_mass=1.0) for _ in range(B)], seed=0, threads=1)
    pool = reference_pool_v2(4, "train", regimes=("hold", "free"))
    pcfg = PPOConfig(hold_steps=10, reflex_force=0.5, minibatch=16, epochs=1)  # any contact triggers the reflex
    env = PPOEnv(batch, policy, RigidBodyModel(load_model(), dtype=torch.float32), pool, {"hold": 0.5, "free": 0.5}, T, rng, RewardConfig(), pcfg)
    critic = Critic(policy.cfg.obs_dim, 4)
    actor = GaussianActor(policy, 0.1)
    roll = collect(env, actor, critic)
    assert roll["obs"].shape == (T // 10, B, policy.cfg.obs_dim) and roll["rew"].shape == (T // 10, B)
    assert roll["valid"][0].all()  # every robot starts active
    # free-space windows never touch: no reflex, every sample valid, bootstrap kept
    assert not roll["reflex"].any() and roll["valid"].all()
    adv, ret = gae(roll, pcfg.gamma, pcfg.lam)
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()
    stats = ppo_update(actor, critic, torch.optim.Adam(actor.parameters(), 1e-4), torch.optim.Adam(critic.parameters(), 1e-3), roll, pcfg)
    assert stats["n_samples"] == (T // 10) * B and np.isfinite(stats["actor_loss"])


def test_reflex_terminates_and_masks():
    roll = {"rew": torch.ones(4, 2), "val": torch.zeros(4, 2), "done": torch.zeros(4, 2, dtype=torch.bool), "v_last": torch.ones(2),
            "valid": torch.ones(4, 2, dtype=torch.bool)}
    roll["done"][1, 0] = True  # robot 0 reflexes at decision 1
    roll["valid"][2:, 0] = False
    adv, ret = gae(roll, gamma=0.9, lam=1.0)
    # robot 1: full return with bootstrap; robot 0: return cut at decision 1
    assert abs(ret[0, 1] - (1 + 0.9 + 0.81 + 0.729 + 0.9**4)) < 1e-5
    assert abs(ret[0, 0] - (1 + 0.9)) < 1e-5 and abs(ret[1, 0] - 1.0) < 1e-5


def test_controller_hold_steps_holds_the_output():
    from llc.controllers.learned import LearnedController

    nominal = NominalModel()
    pol1, pol10 = Policy(_cfg(1)), Policy(_cfg(10))
    torch.manual_seed(1)
    with torch.no_grad():  # a clearly non-zero residual so holding is visible
        for p in (pol1, pol10):
            p.head.weight.normal_(0, 0.5); p.head.bias.normal_(0, 0.5)
    pol10.load_state_dict(pol1.state_dict())
    c1, c10 = LearnedController(pol1, nominal), LearnedController(pol10, nominal)
    rng = np.random.default_rng(0)
    q = rng.normal(size=7) * 0.1; qd = np.zeros(7)
    taus1, taus10 = [], []
    for k in range(25):
        qk = q + 0.01 * np.sin(0.3 * k) * np.ones(7)
        taus1.append(c1(k * 1e-3, qk, qd, q, qd, qd).copy()); taus10.append(c10(k * 1e-3, qk, qd, q, qd, qd).copy())
    taus1, taus10 = np.array(taus1), np.array(taus10)
    # the held controller's residual is constant within a decision: its torque differs from the 1 kHz one
    # only through the residual, and its residual at steps 0..9 equals the one computed at step 0
    pd_part = lambda tau, qk: tau - (np.asarray(pol1.cfg.kp_shared) * (q - qk) + nominal.gravity(qk))
    r1 = np.array([pd_part(taus1[k], q + 0.01 * np.sin(0.3 * k) * np.ones(7)) for k in range(25)])
    r10 = np.array([pd_part(taus10[k], q + 0.01 * np.sin(0.3 * k) * np.ones(7)) for k in range(25)])
    assert np.allclose(r10[0], r1[0]) and np.allclose(r10[:10], r10[0][None]) and np.allclose(r10[10:20], r10[10][None])
    assert not np.allclose(r1[:10], r1[0][None])
