"""Milestone 12 extensions (ADR 0012): the actuator lag and the environment's friction in the plant, the consolidation
instance sampler, the v5 training cases (same references as v3, new meta), the deployment inputs and their exact expansion,
the residual trunk, and the uncapped label store."""
import numpy as np
import torch

from llc.eval.cases_v2 import make_case_v2
from llc.learning.distill import FIELDS, LabelStore
from llc.learning.policy import Policy, PolicyConfig, encode_extras, expand_inputs
from llc.sim.instances import sample_consolidation_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, DynamicsPerturbation, PandaSim


def test_actuator_lag_and_surface_friction():
    sim = PandaSim(DynamicsPerturbation(actuation_lag=2e-3, surface_friction=0.4))
    import mujoco

    gid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert abs(sim.model.geom_friction[gid, 0] - 0.4) < 1e-9
    sim.reset(np.zeros(NUM_JOINTS))
    applied = [sim.step(np.ones(NUM_JOINTS))[0] for _ in range(3)]
    a = 1e-3 / (2e-3 + 1e-3)
    assert abs(applied[0] - a) < 1e-9 and abs(applied[1] - (a + a * (1 - a))) < 1e-9  # first-order response to a unit step


def test_consolidation_instance_ranges():
    rng = np.random.default_rng(0)
    inst = [sample_consolidation_instance(rng, payload_mass=1.0) for _ in range(50)]
    assert all(0.0 <= i.actuation_delay <= 5e-3 and 0.0 <= i.actuation_lag <= 3e-3 and 0.3 <= i.surface_friction <= 1.0 and 0.5 <= i.object_mass_scale <= 1.5 for i in inst)
    assert all(1e-4 <= i.obs_noise_q <= 5e-4 and abs(i.obs_noise_qd - 10 * i.obs_noise_q) < 1e-12 for i in inst)
    assert max(max(abs(np.array(i.link_mass_scale) - 1)) for i in inst) > 0.10  # the wide ranges are in use


def test_v5_case_keeps_the_reference_and_adds_the_meta():
    nominal = NominalModel()
    a, b = make_case_v2("free", "train", 1234, nominal), make_case_v2("free", "train", 1234, nominal, dist="v5")
    assert np.allclose(a.reference.q, b.reference.q)  # the reference's draws come first: unchanged by the distribution
    assert "base_scale" in b.meta and "latency_s" in b.meta and "base_scale" not in a.meta
    kp_s, kd_s = np.array(b.meta["base_scale"][0]), np.array(b.meta["base_scale"][1])
    assert kp_s.shape == (NUM_JOINTS,) and 0.2 < kp_s.min() and kp_s.max() < 3.7 and 0.4 < kd_s.min() and kd_s.max() < 2.5
    assert 0.0 <= b.meta["latency_s"] <= 0.01 and b.perturbation.actuation_lag <= 3e-3
    c = make_case_v2("hold", "test", 0, nominal, base_scale=2.0)
    assert c.id.endswith("-b2-000") and np.allclose(c.meta["base_scale"][0], 2.0) and np.allclose(c.meta["base_scale"][1], np.sqrt(2.0))


def _cfg(**kw):
    params = dict(kind="pd_gain_residual", history=5, history_stride=2, history_dense=4, rate_hz=1000.0, latency_s=0.0, hidden=32, layers=2,
                  kp_shared=(100.0,) * NUM_JOINTS, kd_shared=(10.0,) * NUM_JOINTS, ext_torque=True, feedforward=True)
    params.update(kw)
    return PolicyConfig(**params)


def test_expand_inputs_is_exact_and_residual_trunk_runs():
    torch.manual_seed(0)
    base = Policy(_cfg())
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))
    wide = expand_inputs(base)
    assert wide.cfg.extra_dim == 2 * NUM_JOINTS + 1 and wide.cfg.obs_dim == base.cfg.obs_dim + wide.cfg.extra_dim
    obs = torch.randn(3, base.cfg.obs_dim)
    extras = encode_extras(wide.cfg, np.full(NUM_JOINTS, 2.0), np.full(NUM_JOINTS, 0.5), 0.005)
    assert extras.shape == (1, wide.cfg.extra_dim) and abs(float(extras[0, 0]) - np.log(2) / np.log(3)) < 1e-6 and abs(float(extras[0, -1]) - 0.5) < 1e-6
    with torch.no_grad():
        assert torch.allclose(base(obs), wide(torch.cat([obs, extras.expand(3, -1)], -1)), atol=1e-6)
    deep = Policy(_cfg(layers=10, residual_blocks=True))
    assert isinstance(deep.trunk[0], torch.nn.Linear) and len(deep.trunk) == 2 + 9
    out = deep.decode(deep(torch.randn(2, deep.cfg.obs_dim)))
    assert torch.allclose(out["kp_scale"], torch.ones_like(out["kp_scale"]), atol=0.05)  # the head starts neutral


def test_label_store_uncapped_and_half_precision():
    rng = np.random.default_rng(0)
    store = LabelStore(0, rng)
    b = {k: np.ones((10, NUM_JOINTS if k != "obs" else 20), np.float32) for k in FIELDS}
    b["regime"] = np.array(["free"])
    for _ in range(3):
        store.add([b], weight_of=lambda r: 2.0)
    assert len(store) == 30 and store.data["obs"].dtype == np.float16 and float(store.data["weight"].mean()) == 2.0
