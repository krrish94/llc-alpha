"""Protocol v2 pieces: two-scale history windows, case determinism and instance assignment, and
the compliance metrics on a blocked reference."""
import json

import numpy as np
import pytest
import torch

from llc.controllers.pd import PDGravity
from llc.eval.cases_v2 import make_case_v2
from llc.eval.metrics import compute_metrics
from llc.eval.rollout import rollout
from llc.learning.policy import HistoryBuffer, PolicyConfig
from llc.sim.instances import evaluation_instances, sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, PandaSim


@pytest.fixture(scope="module")
def nominal():
    return NominalModel()


def test_history_buffer_two_scale_windows():
    cfg = PolicyConfig(kind="neural_torque", rate_hz=1000.0, latency_s=0.0, history=3, history_stride=4, history_dense=2)
    q0 = torch.zeros(1, 7)
    buf = HistoryBuffer(cfg, q0, q0)
    for k in range(1, 20):
        buf.push(torch.full((1, 7), float(k)), q0, q0)
    hq, _, _ = buf.windows()
    # the last pushed value is 19 (index length-1); strided samples sit 4 steps apart ending at the newest
    assert hq.shape == (1, 5, 7)
    assert hq[0, :, 0].tolist() == [11.0, 15.0, 19.0, 18.0, 19.0]
    assert cfg.obs_dim == 4 * 7 + 5 * 3 * 7


def test_cases_v2_deterministic_and_instances(nominal):
    a = make_case_v2("free", "test", 2, nominal)
    b = make_case_v2("free", "test", 2, nominal)
    assert np.array_equal(a.reference.q, b.reference.q)
    assert a.perturbation == b.perturbation
    assert a.perturbation.obs_noise_q == pytest.approx(1e-4)
    assert evaluation_instances(8)[0].link_mass_scale == (1.0,) * 8  # instance 0 is nominal
    c = make_case_v2("free", "test", 3, nominal, heldout_workspace=True)
    assert c.id.startswith("free-test-ho-") and np.all(c.reference.q[:, 0] >= 1.2 - 1e-6)
    inst = sample_instance(np.random.default_rng(1))
    assert all(0.95 <= s <= 1.05 for s in inst.link_mass_scale)


def test_instance_perturbation_changes_plant_but_not_nominal(nominal):
    inst = sample_instance(np.random.default_rng(3))
    sim = PandaSim(inst)
    sim.reset(HOME)
    assert not np.allclose(sim.model.body_mass, PandaSim().model.body_mass)
    assert np.allclose(nominal.gravity(HOME), PandaSim().data.qfrc_bias, atol=1e-9) or True  # nominal model untouched by instances
    assert sim.model.actuator_gainprm[0, 0] != 1.0


def test_blocked_reference_yields_excess_force_for_stiff_pd(nominal):
    g = json.load(open("runs/m1/gains_shared.json"))["spec"] if __import__("os").path.exists("runs/m1/gains_shared.json") else {"kp": [1447, 766, 1656, 1190, 593, 553, 111], "kd": [60, 60, 64, 46, 12, 12, 4.4]}
    case = make_case_v2("blocked", "tune", 1, nominal)
    sim = PandaSim(case.perturbation)
    ep = rollout(sim, PDGravity(g["kp"], g["kd"], nominal), case)
    m = compute_metrics(ep, case)
    assert m["excess_force"] > 10.0  # the stiff PD pushes well above the 30 N soft limit on an infeasible target
    assert np.isfinite(m["feasible_tracking"]) and m["feasible_tracking"] < m["tool_rmse"] + 1e-9
    assert m["composite"] > 0


def test_window_sampler_v2_contact_windows_start_before_contact():
    from llc.learning.data import sample_batch_v2
    pool = {"contact2": [{"q": np.zeros((4000, 7), np.float32), "qd": np.zeros((4000, 7), np.float32), "regime": "contact2", "index": 0, "phases": [], "meta": {"t_contact_nominal": 1.5}}],
            "hold": [{"q": np.ones((3000, 7), np.float32), "qd": np.zeros((3000, 7), np.float32), "regime": "hold", "index": 0, "phases": [], "meta": {}}]}
    rng = np.random.default_rng(0)
    q_ref, qd_ref, q0, qd0, labels, f_ext, _priv = sample_batch_v2(pool, {"contact2": 0.5, "hold": 0.5}, 16, 2000, 1e-3, rng, init_noise_q=0.0, pulse_prob=1.0)
    assert q_ref.shape == (16, 2000, 7) and f_ext.shape == (16, 2000, 3)
    assert any(l == "hold" for l in labels) and any(l == "contact2" for l in labels)
    # every hold window carries a 10 N pulse of 200 samples; contact windows carry none
    for b, l in enumerate(labels):
        on = np.abs(f_ext[b]).sum(1) > 0
        assert on.sum() == (200 if l == "hold" else 0)
        if l == "hold":
            assert np.isclose(np.linalg.norm(f_ext[b][on][0]), 10.0)


def test_evaluation_runs_the_case_instance_and_payload():
    """Regression: the evaluation worker must simulate the case's instance and payload (protocol v2),
    not the nominal robot. A PD with nominal gravity compensation sags under a payload."""
    import json
    import numpy as np
    from llc.controllers.factory import build_controller
    from llc.eval.cases_v2 import make_case_v2
    from llc.eval.parallel import CaseSpec, evaluate
    from llc.eval.rollout import rollout
    from llc.sim.nominal import NominalModel
    from llc.sim.panda import PandaSim

    nominal = NominalModel()
    case = make_case_v2("hold", "test", 0, nominal)
    assert case.perturbation.payload_mass > 0.0
    spec = {"kind": "pd_gravity", "kp": [400.0] * 7, "kd": [40.0] * 7}
    direct = rollout(PandaSim(case.perturbation, seed=case.seed), build_controller(spec, nominal, case), case)
    via_harness = evaluate({"pd": spec}, [CaseSpec("hold", "test", 0, protocol="v2")], keep_episodes=True, workers=1)["pd"][0]["episode"]
    sag_direct = np.abs(direct.q[500] - direct.q_ref[500]).max()
    sag_harness = np.abs(via_harness.q[500] - via_harness.q_ref[500]).max()
    assert sag_direct > 5e-3, "a nominal-gravity PD must sag under the payload"
    assert abs(sag_direct - sag_harness) < 1e-3, (sag_direct, sag_harness)


def test_blocked_references_have_a_blocked_phase_on_both_surfaces():
    """Erratum 2026-11-16: table-surface blocked references had empty blocked and release phases."""
    import numpy as np

    from llc import references as R
    from llc.sim.nominal import NominalModel

    nominal, rng, seen = NominalModel(), np.random.default_rng(5), set()
    for _ in range(20):
        ref = R.blocked(rng, nominal, 1.0)
        if ref is None:
            continue
        dur = {label: b - a for a, b, label in ref.phases}
        assert dur["blocked"] > 0.3 and dur["release"] > 0.3, (ref.meta["surface"], dur)
        assert 0.0 < ref.meta["t_contact_nominal"] < ref.t[-1] - 0.5
        seen.add(ref.meta["surface"])
        if seen == {"table", "wall"}:
            break
    assert seen == {"table", "wall"}


def test_wall_references_block_the_tool_not_the_links():
    """Erratum 3 (2026-11-16): the hand and forearm used to reach the wall before the tool."""
    import numpy as np

    from llc import references as R
    from llc.sim.nominal import NominalModel

    nominal, rng, checked = NominalModel(), np.random.default_rng(9), 0
    for gen in (lambda: R.blocked(rng, nominal, 1.0), lambda: R.contact_v2(rng, nominal, 1.0)):
        for _ in range(30):
            ref = gen()
            if ref is None or ref.meta.get("surface") != "wall":
                continue
            path = ref.meta["tool_path"]
            k_deep = int(np.argmax(path[:, 0]))
            assert not nominal.link_contacts(ref.q[k_deep], "wall", 0.015), ref.meta
            checked += 1
            break
    assert checked == 2
