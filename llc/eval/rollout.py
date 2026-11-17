"""Closed-loop rollout of a controller on a case, logging the signals the metrics need."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llc.eval.cases import Case
from llc.sim.panda import PandaSim


@dataclass
class Episode:
    case_id: str
    regime: str
    t: np.ndarray  # (T,)
    q: np.ndarray  # (T,7) true joint position
    qd: np.ndarray
    q_ref: np.ndarray
    qd_ref: np.ndarray
    tau_cmd: np.ndarray  # commanded torque before clamping
    tau: np.ndarray  # torque applied by the actuators (after clamp and delay)
    tool_pos: np.ndarray  # (T,3)
    f_normal: np.ndarray  # (T,) N
    f_tangent: np.ndarray  # (T,) N
    n_tool_contacts: np.ndarray  # (T,)
    n_other_contacts: np.ndarray  # (T,)
    diverged: bool = False
    obj_pos: np.ndarray | None = None  # (T,3) task object position (Milestone 9 task families)
    obj_quat: np.ndarray | None = None  # (T,4)
    f_object: np.ndarray | None = None  # (T,) normal force between the tool and the object
    f_object_env: np.ndarray | None = None  # (T,) normal force between the object and the environment


def rollout(sim: PandaSim, controller, case: Case) -> Episode:
    ref = case.reference
    n = len(ref.t)
    sim.reset(case.q0, case.qd0)
    controller.reset()
    task = case.meta.get("task")
    eq_hold = None
    if task is not None:  # Milestone 9 task families: place the object; a held object is welded until the release time
        import mujoco

        sim.set_object_pose(task["object"], np.asarray(task["object_pos"]), np.asarray(task["object_quat"]))
        if task.get("weld"):
            eq_hold = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_EQUALITY, "hold")
            sim.data.eq_active[eq_hold] = 1
    k_release = int(round(task["release_time"] / float(ref.t[1] - ref.t[0]))) if task is not None and task.get("weld") else -1
    log = {k: np.zeros((n, 7)) for k in ("q", "qd", "tau_cmd", "tau")}
    tool = np.zeros((n, 3))
    f_n, f_t = np.zeros(n), np.zeros(n)
    n_tool, n_other = np.zeros(n, int), np.zeros(n, int)
    obj_pos, obj_quat, f_obj, f_obj_env = (np.zeros((n, 3)), np.zeros((n, 4)), np.zeros(n), np.zeros(n)) if task is not None else (None, None, None, None)
    diverged = False
    preview = getattr(controller, "preview", 0)
    if preview:
        from llc.learning.policy import preview_indices
    for k in range(n):
        q_obs, qd_obs = sim.observe()
        if preview:
            tau_cmd = controller(ref.t[k], q_obs, qd_obs, ref.q[k], ref.qd[k], ref.qdd[k], q_ref_future=ref.q[preview_indices(controller.policy.cfg, k, n)])
        else:
            tau_cmd = controller(ref.t[k], q_obs, qd_obs, ref.q[k], ref.qd[k], ref.qdd[k])
        if case.pulse is not None and case.pulse.t_start <= ref.t[k] < case.pulse.t_start + case.pulse.duration:
            sim.apply_tool_force(np.asarray(case.pulse.force))
        else:
            sim.apply_tool_force(np.zeros(3))
        if eq_hold is not None and k == k_release:
            sim.data.eq_active[eq_hold] = 0
        tau_applied = sim.step(np.nan_to_num(tau_cmd, nan=0.0, posinf=0.0, neginf=0.0))
        c = sim.contact()
        log["q"][k], log["qd"][k] = sim.q, sim.qd
        log["tau_cmd"][k], log["tau"][k] = tau_cmd, tau_applied
        tool[k], f_n[k], f_t[k] = sim.tool_pos, c.f_normal, c.f_tangent
        n_tool[k], n_other[k] = c.n_tool_contacts, c.n_other_contacts
        if task is not None:
            obj_pos[k], obj_quat[k] = sim.object_pose(task["object"])
            f_obj[k], f_obj_env[k] = c.f_object, c.f_object_env
        if not np.all(np.isfinite(log["q"][k])) or np.abs(log["qd"][k]).max() > 50.0:
            diverged = True
            break
    return Episode(case.id, case.regime, ref.t, log["q"], log["qd"], ref.q, ref.qd, log["tau_cmd"], log["tau"],
                   tool, f_n, f_t, n_tool, n_other, diverged, obj_pos, obj_quat, f_obj, f_obj_env)
