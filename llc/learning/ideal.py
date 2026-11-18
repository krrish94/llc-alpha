"""TAM-style ideal-robot residual (ADR 0010; after Son et al., "TAM: Torque Adaptation Module", CoRL
2026; re-implemented in MuJoCo without multi-robot pretraining).

The student is a residual on the shared PD (``pd_residual``, proprioceptive history and the
external-torque estimate as input, 10 ms decisions) trained to make the sampled perturbed robot
(instance deviations, payload) reproduce the nominal robot's response. Data are free-space episodes
of the training split driven by the student itself (at initialisation the plain PD), with a smooth
random external joint torque injected in a fraction of the episodes to mimic contact-like loads
(and the hold family's tool-force pulses, when the case carries one).
The teacher label at every decision is the torque the perturbed plant needs to produce the nominal
robot's acceleration under the nominal law (the student's PD base without its learned part, ``tau_nom``;
for a direct torque network the command itself) and the same external torque:

    a0 = M0(q)^-1 (tau_nom + tau_ext + passive0(q, qd) - bias0(q, qd))      (nominal model)
    tau* = qfrc_inverse(perturbed plant at (q, qd), qacc = a0) - tau_ext     (mj_inverse, privileged)

Taking the label under the student's full command instead (the first form of this module) made the
target drift with the student's own residual and degraded free space in every DAgger iteration
(runs/m10/ideal, runs/m10r2/distill_two_v0).

and the student regresses tau* through its torque law with a Huber loss.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import mujoco
import numpy as np
import torch

from llc.controllers.learned import load_learned_controller
from llc.eval.parallel import CaseSpec, _get_case, _get_sim, _init_worker, _worker
from llc.learning.distill import FIELDS
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS

EXT_AMPLITUDE = np.array([15.0, 15.0, 15.0, 15.0, 4.0, 4.0, 4.0])  # Nm, per joint, of the injected external torque (contact-like loads reach 20 to 40 Nm at the big joints)
RESIDUAL_BOUND = 30.0  # Nm: decisions whose ideal correction exceeds this on any joint are infeasible for the actuators and are dropped


def external_torque_signal(rng: np.random.Generator, n: int, dt: float, tau_corr: float = 0.3) -> np.ndarray:
    """(n, 7) smooth random external torque: an Ornstein-Uhlenbeck process (correlation ``tau_corr`` s)
    scaled to the per-joint amplitude, ramped in over the first 0.2 s."""
    a = dt / tau_corr
    x = np.zeros(NUM_JOINTS)
    out = np.zeros((n, NUM_JOINTS))
    for k in range(n):
        x = x + a * (-x) + np.sqrt(2 * a) * rng.normal(size=NUM_JOINTS)
        out[k] = x
    out = out / (out.std(0) + 1e-6) * EXT_AMPLITUDE * rng.uniform(0.2, 1.0)
    ramp = np.minimum(1.0, np.arange(n) * dt / 0.2)
    return out * ramp[:, None]


def ideal_label(sim, nominal, q, qd, tau_cmd, tau_ext) -> np.ndarray:
    """The torque the perturbed plant needs for the nominal robot's acceleration under ``tau_cmd`` + ``tau_ext``."""
    m0 = nominal.mass_matrix(q)  # resets the model's velocity: the bias and passive terms are read after setting (q, qd)
    nominal._set(q, qd)
    a0 = np.linalg.solve(m0, tau_cmd + tau_ext + nominal.data.qfrc_passive[:NUM_JOINTS] - nominal.data.qfrc_bias[:NUM_JOINTS])
    d = sim.data
    qacc_saved = d.qacc.copy()
    d.qacc[:NUM_JOINTS] = a0
    mujoco.mj_inverse(sim.model, d)
    tau_star = np.clip(d.qfrc_inverse[:NUM_JOINTS] - tau_ext, -TORQUE_LIMITS, TORQUE_LIMITS)  # the actuators cannot exceed their limits
    d.qacc[:] = qacc_saved
    return tau_star


def collect_one(args) -> dict[str, np.ndarray]:
    student_path, case_spec, inject, seed = args[:4]
    servo = args[4] if len(args) > 4 else None  # ADR 0011: the student decides at the host frame through the profile's servo loop
    if not _worker:
        _init_worker()
    case = _get_case(case_spec)
    sim = _get_sim(case.perturbation, seed=case.seed, scene=(case.meta.get("task") or {}).get("scene"))  # a task family brings its scene (protocol v4 and v5)
    nominal = _worker["nominal"]
    options = {"latency_s": 0.0, "dt": servo["period"]} if servo else {}
    if "base_scale" in case.meta:  # ADR 0012: the episode's base gains (the nominal law of the label is the base's own)
        options["base_scale"] = case.meta["base_scale"]
    if "latency_s" in case.meta and not servo:
        options["latency_s"] = case.meta["latency_s"]
    ctrl = load_learned_controller(student_path, nominal, **options)
    runner = ctrl
    if servo:
        from llc.sim.servo import ModelInterface, ServoLoop, get_profile

        profile = get_profile(servo["profile"], **servo.get("overrides", {}))
        runner = ServoLoop(ctrl, profile, ModelInterface(profile, np.asarray(servo["kp"]), np.asarray(servo["kd"])))
    ref = case.reference
    n = len(ref.t)
    rng = np.random.default_rng(seed)
    ext = external_torque_signal(rng, n, sim.dt) if inject else np.zeros((n, NUM_JOINTS))
    sim.reset(case.q0, case.qd0)
    runner.reset()
    rows = []
    jac = np.zeros((3, sim.model.nv))
    for k in range(n):
        q_obs, qd_obs = sim.observe()
        before = ctrl.n_updates
        tau_raw = np.nan_to_num(runner(ref.t[k], q_obs, qd_obs, ref.q[k], ref.qd[k], ref.qdd[k]))
        tau_cmd = np.clip(tau_raw, -TORQUE_LIMITS, TORQUE_LIMITS)
        f_pulse = np.zeros(3)  # the case's tool-force pulse (hold family), as the evaluation rollout applies it
        if case.pulse is not None and case.pulse.t_start <= ref.t[k] < case.pulse.t_start + case.pulse.duration:
            f_pulse = np.asarray(case.pulse.force, float)
        sim.apply_tool_force(f_pulse)
        if ctrl.n_updates > before:
            q, qd = sim.q, sim.qd  # the true state for the label; the student sees the measured one
            mujoco.mj_jacBodyCom(sim.model, sim.data, jac, None, sim.tool_body)  # the pulse acts at the tool body's centre of mass
            tau_ext = ext[k] + jac[:, :NUM_JOINTS].T @ f_pulse  # the ideal robot feels the same external loads; the label does not cancel them
            # The ideal robot is the nominal robot under the nominal law (the student's PD base without its learned part): a label
            # taken under the student's full command would ask the plant to reproduce the nominal robot's response to a residual
            # that already compensates the plant, and shifts with every DAgger iteration (ADR 0010 amendment 2, item D).
            tau_nom = np.asarray(ctrl.kp_shared) * (ref.q[k] - q_obs) + np.asarray(ctrl.kd_shared) * (ref.qd[k] - qd_obs) + ctrl.base_term(q_obs, ref.qd[k], ref.qdd[k]) if ctrl.kind != "neural_torque" else tau_raw
            saturated = (np.abs(tau_nom) >= TORQUE_LIMITS).any()  # the nominal law exceeds an actuator limit (step onsets, target jumps): no feasible correction
            tau_nom = np.clip(tau_nom, -TORQUE_LIMITS, TORQUE_LIMITS)
            label = ideal_label(sim, nominal, q, qd, tau_nom, tau_ext)
            keep = not saturated and np.abs(label - tau_nom).max() <= RESIDUAL_BOUND  # the v3 rule: corrections beyond 30 Nm are infeasible for the actuators: dropped
            if "base_scale" in case.meta:  # ADR 0012 amendment 1 (the v5 distribution): keep every decision; the correction clipped to the residual's authority, none where the nominal law saturates
                authority = float(ctrl.policy.cfg.residual_fraction) * TORQUE_LIMITS
                label = np.clip(tau_nom if saturated else tau_nom + np.clip(label - tau_nom, -authority, authority), -TORQUE_LIMITS, TORQUE_LIMITS)  # the applied torque
                keep = True
            if keep:
                rows.append((ctrl.last_obs[0].numpy().astype(np.float32), q_obs.astype(np.float32), qd_obs.astype(np.float32), ref.q[k].astype(np.float32), ref.qd[k].astype(np.float32),
                             ctrl.base_term(q_obs, ref.qd[k], ref.qdd[k]).astype(np.float32), label.astype(np.float32), np.zeros(NUM_JOINTS, np.float32),
                             ctrl.base_scale[0].astype(np.float32), ctrl.base_scale[1].astype(np.float32)))  # no surface in free space: jn = 0; the episode's gain scales
        sim.data.qfrc_applied[:NUM_JOINTS] = ext[k]
        sim.step(tau_cmd)
        ctrl.last_tau = tau_cmd
        if np.abs(sim.qd).max() > 50.0:
            blown_up = True
            break
    else:
        blown_up = False
    sim.data.qfrc_applied[:] = 0.0
    sim.apply_tool_force(np.zeros(3))
    out = {k: np.stack([r[i] for r in rows]) for i, k in enumerate(FIELDS)} if rows else {k: np.zeros((0, NUM_JOINTS), np.float32) for k in FIELDS}
    # Diverged: the plant blew up, or fewer than a tenth of the decisions kept a label. The threshold counts decisions, not
    # physics steps (erratum of report 21: against the step count, every 50 Hz episode was flagged).
    out["diverged"] = np.array([blown_up or len(rows) < max(1, ctrl.n_updates // 10)])
    out["case_id"] = np.array([case.id])
    out["regime"] = np.array([case.regime])
    return out


def collect_ideal_labels(student_path: str, cases: list[CaseSpec], inject: list[bool], seeds: list[int], workers: int | None = None) -> list[dict[str, np.ndarray]]:
    jobs = [(str(student_path), c, bool(i), int(s)) for c, i, s in zip(cases, inject, seeds)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return [collect_one(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        return pool.map(collect_one, jobs, chunksize=1)


def huber_label_loss(policy, batch: dict[str, torch.Tensor], delta: float = 0.02) -> torch.Tensor:
    """Huber loss (per torque limit) between the student's torque and the ideal-robot torque."""
    held = policy.decode(policy(batch["obs"]))
    tau = policy.fast_torque(held, batch["q"], batch["qd"], batch["q_ref"], batch["qd_ref"], batch["gravity"])
    return torch.nn.functional.huber_loss(tau / policy.limits, batch["tau_teacher"] / policy.limits, delta=delta)
