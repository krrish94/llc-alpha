"""Tests for the risks that matter: units and limits, causality of delays, determinism of cases,
payload perturbation consistency, and metric sanity."""
import numpy as np
import pytest

from llc.controllers.pd import PDGravity, gains_from_params, home_inertia_diag
from llc.eval.cases import case_seed, make_case
from llc.eval.metrics import compute_metrics
from llc.eval.rollout import rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, TORQUE_LIMITS, DynamicsPerturbation, PandaSim


@pytest.fixture(scope="module")
def nominal():
    return NominalModel()


def test_torque_limits_are_enforced():
    sim = PandaSim()
    applied = sim.step(np.full(7, 1e4))
    assert np.allclose(applied, TORQUE_LIMITS)
    assert np.allclose(sim.data.ctrl, TORQUE_LIMITS)


def test_gravity_compensation_holds_pose(nominal):
    """With exact gravity compensation and no PD, the arm must stay (nearly) still at home."""
    sim = PandaSim()
    sim.reset(HOME)
    for _ in range(500):
        sim.step(nominal.gravity(sim.q))
    assert np.abs(sim.q - HOME).max() < 2e-3


def test_actuation_delay_is_causal():
    """A delayed plant must not respond before the delay has elapsed."""
    delayed = PandaSim(DynamicsPerturbation(actuation_delay=0.010))
    prompt = PandaSim()
    tau = np.array([0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for sim in (delayed, prompt):
        sim.reset(HOME)
    for _ in range(5):
        delayed.step(tau)
        prompt.step(tau)
    assert np.allclose(delayed.data.ctrl, 0.0)  # still applying the zero history
    assert not np.allclose(prompt.data.ctrl, 0.0)


def test_payload_changes_mass_and_com_consistently():
    sim = PandaSim(DynamicsPerturbation(payload_mass=1.0, payload_offset=(0.0, 0.0, 0.1)))
    b = sim.tool_body
    assert np.isclose(sim.model.body_mass[b], 1.7)
    # COM moves toward the payload: (0.7*0.06 + 1.0*0.10)/1.7 (nominal tool COM at z = 0.06)
    assert np.isclose(sim.model.body_ipos[b][2], (0.7 * 0.06 + 1.0 * 0.10) / 1.7)
    assert np.all(sim.model.body_inertia[b] > 0)


def test_cases_are_deterministic(nominal):
    a = make_case("smooth", "tune", 3, nominal)
    b = make_case("smooth", "tune", 3, nominal)
    assert np.array_equal(a.reference.q, b.reference.q)
    assert case_seed("tune", "smooth", 3) != case_seed("test", "smooth", 3)


def test_metrics_units_and_ranges(nominal):
    sim = PandaSim()
    kp, kd = gains_from_params(np.concatenate([np.log([400, 400, 300, 300, 100, 60, 30]), [0.0]]), home_inertia_diag(nominal))
    ctrl = PDGravity(kp, kd, nominal)
    case = make_case("transition", "tune", 0, nominal)
    ep = rollout(sim, ctrl, case)
    m = compute_metrics(ep, case)
    assert 0.0 < m["pos_rmse"] < 0.1  # rad
    assert 5.0 < m["f_peak"] < 500.0  # N: the tool does touch the surface
    assert 0.0 <= m["saturation"] <= 1.0
    assert np.isfinite(m["composite"])
    assert m["diverged"] == 0.0
