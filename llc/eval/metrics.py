"""Metrics computed from an episode. All quantities are in SI units (rad, rad/s, Nm, N, s).

Regime-specific metrics are evaluated on the phases they refer to; metrics that do not apply
to a regime are NaN. ``composite`` is the declared tuning objective (protocol v1); it is a
weighted sum of normalized metrics and is NOT a research signal in itself.
"""
from __future__ import annotations

import numpy as np

from llc.eval.cases import Case
from llc.eval.rollout import Episode
from llc.sim.panda import TORQUE_LIMITS, VELOCITY_LIMITS
from llc.sim.plant import PLANT

METRIC_INFO = {
    "pos_rmse": ("Position RMSE", "rad"),
    "pos_max": ("Max position error", "rad"),
    "vel_rmse": ("Velocity RMSE", "rad/s"),
    "tool_rmse": ("Tool position RMSE", "m"),
    "torque_rms": ("Torque RMS", "Nm"),
    "torque_rate_rms": ("Torque rate RMS", "Nm/s"),
    "torque_rate_max": ("Peak torque rate", "Nm/s"),
    "saturation": ("Torque saturation fraction", "-"),
    "settle_time": ("Settling time after step", "s"),
    "overshoot": ("Step overshoot", "-"),
    "pulse_deflection": ("Peak deflection under 10 N pulse", "m"),
    "pulse_recovery": ("Recovery time after pulse", "s"),
    "f_mean": ("Mean normal force in contact phase", "N"),
    "f_std": ("Normal force std in contact phase", "N"),
    "f_peak": ("Peak normal force", "N"),
    "f_peak_20ms": ("Peak of the 20 ms low-pass filtered normal force (sustained force)", "N"),
    "contact_breaks": ("Contact losses of 5 ms or more during contact phase", "-"),
    "contact_fraction": ("Fraction of contact phase in contact", "-"),
    "tangential_rmse": ("Tangential tool error during slide", "m"),
    "impulse_onset": ("Impulse in first 100 ms of contact", "N s"),
    "bounces": ("Contact losses of 5 ms or more within 0.3 s of onset", "-"),
    "force_settle": ("Force settling time after onset", "s"),
    "release_recovery": ("Tracking recovery after release", "s"),
    "other_contact": ("Fraction of time with unintended contact", "-"),
    "vel_limit": ("Fraction of time above velocity limit", "-"),
    "excess_force": (f"Excess contact force above {2.0 * PLANT.force_cap:g} N (mean while in contact)", "N"),
    "excess_force_cap": (f"Excess contact force above the {PLANT.force_cap:g} N cap (mean while in contact)", "N"),
    "contact_loss_rate": ("Contact losses of 5 ms or more per second of press phase", "1/s"),
    "feasible_tracking": ("Feasible-direction tool error", "m"),
    "press_deficit": ("Press-phase force deficit below the press floor (5 N under protocol v2, 7 N under v3)", "N"),
    "contact_recovery": ("Recovery after a blocked reference returns", "s"),
    "release_reaction": ("Blocked: time after release until the tool moves toward the reference faster than 2 cm/s", "s"),
    "release_settle": ("Blocked: time after the reference reaches its final waypoint until the tool error is below 1 cm", "s"),
    "apparent_stiffness": ("Apparent stiffness: mean normal force per metre of reference depth while blocked (after the first 0.3 s of contact)", "N/m"),
    "contact_impedance": ("Contact stiffness: mean normal force per metre of actual tool error along the inward normal while the reference is inside and the tool in contact (after the first 0.3 s of contact)", "N/m"),
    "blocked_drift": ("Blocked: mean tangential distance of the tool from its first-contact point while the reference is inside the surface", "m"),
    "blocked_motion": ("Blocked: fraction of the time in contact with the tool moving faster than 1 cm/s while the reference is inside the surface", "-"),
    "push_error": ("Push: error of the box displacement along the push direction against 10 cm", "m"),
    "push_lateral": ("Push: lateral displacement of the box", "m"),
    "push_yaw": ("Push: box yaw change", "rad"),
    "object_force_peak_20ms": ("Peak of the 20 ms filtered normal force on the object", "N"),
    "object_contact_fraction": ("Fraction of the push phase with the tool on the object", "-"),
    "place_error": ("Place: horizontal error of the block against the target at the end", "m"),
    "place_tilt": ("Place: tilt of the block from upright at the end", "rad"),
    "place_force_peak_20ms": ("Place: peak of the 20 ms filtered block-table force during placing", "N"),
    "lift_rmse": ("Pick: joint tracking RMSE during the lift after the release", "rad"),
    "task_success": ("Task success (push: errors below 2 cm and yaw below 15 degrees; pick: grasped, placed within 2 cm and tilted below 15 degrees)", "-"),
    "tau_track_rmse": ("Realised-torque error against the commanded law at the true state, RMS as a fraction of the stall torque", "-"),
    "tau_track_delay": ("Lag of the realised torque behind the command on the programmed joint (cross-correlation)", "s"),
    "tau_track_bandwidth": ("Frequency at which the chirp's realised amplitude falls below 0.7 of the command", "Hz"),
    "grasp_success": ("Pick: the block was inside the fingertip gap at the grasp instant", "-"),
    "grasp_fail": ("Pick: 1 - grasp success (composite term)", "-"),
    "pregrasp_disturbance": ("Pick: block displacement before the grasp instant", "m"),
    "grasp_force_peak_20ms": ("Pick: peak of the 20 ms filtered tool-block force before the grasp", "N"),
    "insertion_depth": ("Peg: deepest insertion below the hole's top plane during the insert phase", "m"),
    "depth_deficit": ("Peg: target depth minus the depth at the end of the insert phase", "m"),
    "insertion_fail": ("Peg: 1 if less than 20 mm inserted at the end of the insert phase (composite term)", "-"),
    "travel": ("Drawer: closing travel; door: closing angle", "m or rad"),
    "travel_deficit": ("Drawer: travel short of the stop; door: angle left open", "m or rad"),
    "closed_fail": ("Drawer: 1 if more than 5 mm from the stop; door: 1 if more than 2 degrees open (composite term)", "-"),
    "diverged": ("Diverged", "-"),
    "composite": ("Composite tuning objective (protocol v1)", "-"),
    "composite_cap": (f"Composite under the {PLANT.force_cap:g} N cap (contact deep dive selection objective)", "-"),
}

# Composite objective: sum of weight * metric / scale over the listed metrics, per regime.
COMPOSITE = {
    "hold": {"pos_rmse": (1.0, 0.01), "pulse_deflection": (1.0, 0.01), "torque_rate_rms": (0.2, 200.0)},
    "smooth": {"pos_rmse": (1.0, 0.02), "torque_rate_rms": (0.2, 200.0)},
    "zoh": {"pos_rmse": (1.0, 0.05), "torque_rate_rms": (0.2, 500.0)},
    "step": {"pos_rmse": (1.0, 0.05), "settle_time": (1.0, 0.3), "overshoot": (0.5, 0.2), "torque_rate_rms": (0.2, 500.0)},
    "contact": {"tangential_rmse": (1.0, 0.01), "f_std": (1.0, 5.0), "contact_breaks": (1.0, 20.0), "torque_rate_rms": (0.2, 500.0)},
    "transition": {"f_peak": (1.0, 50.0), "bounces": (1.0, 5.0), "release_recovery": (1.0, 0.3), "pos_rmse": (1.0, 0.02), "torque_rate_rms": (0.2, 500.0)},
}
COMPOSITE_V2 = {
    "hold": {"pos_rmse": (1.0, 0.01), "pulse_deflection": (1.0, 0.01), "torque_rate_rms": (0.2, 200.0)},
    "free": {"pos_rmse": (1.0, 0.02), "torque_rate_rms": (0.2, 200.0)},
    "policy": {"pos_rmse": (1.0, 0.05), "torque_rate_rms": (0.2, 500.0)},
    "step": {"pos_rmse": (1.0, 0.05), "settle_time": (1.0, 0.3), "overshoot": (0.5, 0.2), "torque_rate_rms": (0.2, 500.0)},
    "contact2": {"feasible_tracking": (1.0, 0.01), "excess_force": (1.0, 10.0), "f_peak": (0.5, 50.0), "press_deficit": (1.0, 5.0), "contact_breaks": (0.5, 20.0), "torque_rate_rms": (0.2, 500.0)},
    "blocked": {"excess_force": (1.0, 10.0), "f_peak": (1.0, 50.0), "feasible_tracking": (1.0, 0.02), "contact_recovery": (1.0, 0.3), "torque_rate_rms": (0.2, 500.0)},
    "task": {"feasible_tracking": (1.0, 0.01), "excess_force": (1.0, 10.0), "f_peak": (0.5, 50.0), "press_deficit": (1.0, 5.0), "torque_rate_rms": (0.2, 500.0)},
}
COMPOSITE_CAP = {  # contact deep dive (ADR 0007): the v2 composite with the force terms at the 15 N cap and the sustained peak
    **{k: v for k, v in COMPOSITE_V2.items() if k in ("hold", "free", "policy", "step")},
    "contact2": {"feasible_tracking": (1.0, 0.01), "excess_force_cap": (1.0, 5.0), "f_peak_20ms": (0.5, 25.0), "press_deficit": (1.0, 5.0), "contact_loss_rate": (0.5, 10.0), "torque_rate_rms": (0.2, 500.0)},
    # blocked: "stop at the surface and wait" (advisor, 2026-11-16): hold the first-contact point, no unnecessary force, resume when the reference returns
    "blocked": {"excess_force_cap": (1.0, 5.0), "f_peak_20ms": (1.0, 25.0), "blocked_drift": (1.0, 0.02), "blocked_motion": (0.5, 0.5), "contact_recovery": (1.0, 0.3), "torque_rate_rms": (0.2, 500.0)},
    "task": {"feasible_tracking": (1.0, 0.01), "excess_force_cap": (1.0, 5.0), "f_peak_20ms": (0.5, 25.0), "press_deficit": (1.0, 5.0), "contact_loss_rate": (0.5, 10.0), "torque_rate_rms": (0.2, 500.0)},
}
COMPOSITE_V3 = {  # protocol v3 (ADR 0009): the cap composite with the release metrics in place of the recovery, plus the task families
    **{k: v for k, v in COMPOSITE_CAP.items() if k != "blocked"},
    "blocked": {"excess_force_cap": (1.0, 5.0), "f_peak_20ms": (1.0, 25.0), "blocked_drift": (1.0, 0.02), "blocked_motion": (0.5, 0.5), "release_reaction": (1.0, 0.3), "release_settle": (1.0, 0.3), "torque_rate_rms": (0.2, 500.0)},
    "push": {"push_error": (1.0, 0.02), "push_lateral": (0.5, 0.02), "object_force_peak_20ms": (1.0, 25.0), "excess_force_cap": (1.0, 5.0), "torque_rate_rms": (0.2, 500.0)},
    "pick": {"grasp_fail": (1.0, 1.0), "pregrasp_disturbance": (1.0, 0.01), "place_error": (1.0, 0.02), "place_force_peak_20ms": (1.0, 25.0), "lift_rmse": (1.0, 0.02), "torque_rate_rms": (0.2, 500.0)},
    "torque": {"tau_track_rmse": (1.0, 0.05), "tau_track_delay": (1.0, 0.01), "torque_rate_rms": (0.2, 500.0)},  # torque-tracking family (protocol v4, ADR 0011)
    # protocol v5 (ADR 0012): wiping and erasing score as press-and-slide (erasing on the block-table force); the peg by its insertion, the drawer and the door by their closing
    "wipe": dict(COMPOSITE_CAP["contact2"]), "erase": dict(COMPOSITE_CAP["contact2"]),
    "peg": {"insertion_fail": (1.0, 1.0), "depth_deficit": (1.0, 0.01), "excess_force_cap": (1.0, 5.0), "f_peak_20ms": (0.5, 25.0), "torque_rate_rms": (0.2, 500.0)},
    "drawer": {"closed_fail": (1.0, 1.0), "travel_deficit": (1.0, 0.01), "excess_force_cap": (1.0, 5.0), "f_peak_20ms": (0.5, 25.0), "contact_loss_rate": (0.5, 10.0), "torque_rate_rms": (0.2, 500.0)},
    "door": {"closed_fail": (1.0, 1.0), "travel_deficit": (1.0, 0.05), "excess_force_cap": (1.0, 5.0), "f_peak_20ms": (0.5, 25.0), "contact_loss_rate": (0.5, 10.0), "torque_rate_rms": (0.2, 500.0)},
}
PRESS_FLOOR = {"v2": 5.0, "v3": PLANT.press_floor}  # N, the press-deficit threshold per protocol (7 N on the Panda from v3; the plant's after ADR 0011)
FORCE_SOFT_LIMIT = 2.0 * PLANT.force_cap  # N, protocol v2 excess-force threshold (30 N on the Panda)
FORCE_CAP = PLANT.force_cap  # N, sustained-force cap of the contact deep dive (ADR 0007: 15 N on the Panda; the plant's force scale, ADR 0011)
REFLEX_SUSTAINED = PLANT.reflex_force  # N, 20 ms filtered force above which a hardware collision reaction would trigger (ADR 0007)
DIVERGED_PENALTY = 20.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def _settle_time(err: np.ndarray, t: np.ndarray, tol: float) -> float:
    """Time until the error norm stays below ``tol`` for the rest of the window."""
    below = np.max(np.abs(err), axis=1) < tol
    if not below[-1]:
        return float(t[-1] - t[0])
    last_bad = np.where(~below)[0]
    return float(t[last_bad[-1] + 1] - t[0]) if last_bad.size else 0.0


def _count_breaks(in_contact: np.ndarray, min_len: int) -> int:
    """Number of contact losses lasting at least ``min_len`` samples (shorter flickers are
    stick-slip micro-events at the simulator's resolution and are not counted)."""
    count, run = 0, 0
    for c in in_contact:
        if c:
            if run >= min_len:
                count += 1
            run = 0
        else:
            run += 1
    return count + (1 if run >= min_len else 0)


def compute_metrics(ep: Episode, case: Case) -> dict[str, float]:
    ref = case.reference
    n = len(ep.t)
    m: dict[str, float] = {k: float("nan") for k in METRIC_INFO}
    dt = float(ep.t[1] - ep.t[0])
    err = ep.q_ref[:n] - ep.q[:n]
    m["pos_rmse"], m["pos_max"] = _rms(err), float(np.abs(err).max())
    m["vel_rmse"] = _rms(ep.qd_ref[:n] - ep.qd[:n])
    m["torque_rms"] = _rms(ep.tau)
    m["torque_rate_rms"] = _rms(np.diff(ep.tau, axis=0) / dt) if n > 1 else float("nan")
    m["torque_rate_max"] = float(np.abs(np.diff(ep.tau, axis=0) / dt).max()) if n > 1 else float("nan")
    m["saturation"] = float(np.mean(np.abs(ep.tau_cmd) >= TORQUE_LIMITS - 1e-9))
    m["other_contact"] = float(np.mean(ep.n_other_contacts > 0))
    m["vel_limit"] = float(np.mean(np.any(np.abs(ep.qd) > VELOCITY_LIMITS, axis=1)))
    m["diverged"] = float(ep.diverged)
    tool_ref = case.meta.get("tool_ref")  # optional precomputed tool reference path (T,3)
    if tool_ref is not None:
        m["tool_rmse"] = _rms(np.linalg.norm(tool_ref[:n] - ep.tool_pos, axis=1))

    if ep.regime == "hold" and case.pulse is not None:
        k0 = int(round(case.pulse.t_start / dt))
        rest = ep.tool_pos[max(0, k0 - 50):k0].mean(0)
        dev = np.linalg.norm(ep.tool_pos[k0:] - rest, axis=1)
        m["pulse_deflection"] = float(dev.max())
        k_end = k0 + int(round(case.pulse.duration / dt))
        after = dev[k_end - k0:]
        back = np.where(after < 2e-3)[0]
        m["pulse_recovery"] = float(back[0] * dt) if back.size else float(ep.t[-1] - case.pulse.t_start - case.pulse.duration)

    if ep.regime == "step":
        settles, overs = [], []
        for t_step in ref.meta["step_times"]:
            k0 = int(round(t_step / dt))
            k1 = min(n, k0 + int(round(0.75 / dt)))
            if k1 <= k0 + 1:
                continue
            e = err[k0:k1]
            settles.append(_settle_time(e, ep.t[k0:k1], 0.02))
            dq = ep.q_ref[k0] - ep.q[k0 - 1]  # step vector from the state just before the step
            along = (ep.q[k0:k1] - ep.q_ref[k0]) @ dq / (np.dot(dq, dq) + 1e-12)
            overs.append(float(max(0.0, along.max())))
        m["settle_time"], m["overshoot"] = float(np.mean(settles)), float(np.mean(overs))

    if ep.regime in ("contact", "transition", "contact2", "blocked", "task", "wipe", "erase"):
        f_src = ep.f_object_env if ep.regime == "erase" else ep.f_normal  # erasing (ADR 0012): the held block's force on the table
        in_contact = (f_src > 0.5) if ep.regime == "erase" else (ep.n_tool_contacts > 0)
        m["f_peak"] = float(f_src.max())
        filt, a = np.zeros(n), dt / 0.02  # first-order low-pass, 20 ms: what a collision reaction on filtered torque estimates sees
        acc = 0.0
        for i in range(n):
            acc += a * (f_src[i] - acc); filt[i] = acc
        m["f_peak_20ms"] = float(filt.max())
        press = ref.phase_mask("press")[:n] | ref.phase_mask("slide")[:n]
        if press.any():
            f = f_src[press]
            m["f_mean"], m["f_std"] = float(f.mean()), float(f.std())
            c = in_contact[press]
            m["contact_fraction"] = float(c.mean())
            m["contact_breaks"] = float(_count_breaks(c, int(round(0.005 / dt))))
            m["contact_loss_rate"] = m["contact_breaks"] / (float(press.sum()) * dt)
        slide = ref.phase_mask("slide")[:n]
        if slide.any() and tool_ref is not None:
            normal = np.asarray(ref.meta["normal"])
            d = tool_ref[:n][slide] - ep.tool_pos[slide]
            d_t = d - np.outer(d @ normal, normal)
            m["tangential_rmse"] = _rms(np.linalg.norm(d_t, axis=1))
        onset = np.where(in_contact)[0]
        if onset.size:
            k_on = onset[0]
            w = slice(k_on, min(n, k_on + int(round(0.1 / dt))))
            m["impulse_onset"] = float(np.sum(f_src[w]) * dt)
            c = in_contact[k_on:min(n, k_on + int(round(0.3 / dt)))]
            m["bounces"] = float(_count_breaks(c, int(round(0.005 / dt))))
            # force settling: time after onset until the force stays within 20% of its press-phase mean
            if press.any() and m["f_mean"] > 0 and ep.regime in ("contact", "transition"):
                k_end = int(round(ref.phases[1][1] / dt))
                f = f_src[k_on:min(n, k_end)]
                within = np.abs(f - m["f_mean"]) < 0.2 * m["f_mean"] + 0.5
                bad = np.where(~within)[0]
                m["force_settle"] = float((bad[-1] + 1) * dt) if bad.size else 0.0
        rel = ref.phase_mask("release")[:n] | ref.phase_mask("free")[:n]
        if rel.any():
            k_r = np.where(rel)[0][0]
            m["release_recovery"] = _settle_time(err[k_r:], ep.t[k_r:], 0.02)

    if ep.regime in ("push", "pick") and ep.obj_pos is not None:  # Milestone 9 task families
        task = case.meta["task"]
        filt_o = np.zeros(n); acc = 0.0; a = dt / 0.02
        src = ep.f_object if ep.regime == "push" else ep.f_object_env
        for i in range(n):
            acc += a * (src[i] - acc); filt_o[i] = acc
        if ep.regime == "push":
            d = np.asarray(task["push_dir"]); p0 = np.asarray(task["object_pos"])
            disp = ep.obj_pos[-1] - p0
            along = float(disp @ d); lateral = disp - along * d
            m["push_error"] = abs(along - task["push_dist"]); m["push_lateral"] = float(np.hypot(lateral[0], lateral[1]))
            q0, q1 = np.asarray(task["object_quat"]), ep.obj_quat[-1]
            yaw = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            m["push_yaw"] = float(abs((yaw(q1) - yaw(q0) + np.pi) % (2 * np.pi) - np.pi))
            push = ref.phase_mask("push")[:n]
            m["object_contact_fraction"] = float(np.mean(ep.f_object[push] > 0.5)) if push.any() else float("nan")
            m["object_force_peak_20ms"] = float(filt_o.max())
            m["task_success"] = float(m["push_error"] < 0.02 and m["push_lateral"] < 0.02 and m["push_yaw"] < np.radians(15))
            m["f_peak"], m["f_peak_20ms"] = float(ep.f_object.max()), float(filt_o.max())
            on = ep.f_object > 0.5
            m["excess_force_cap"] = float(np.mean(np.maximum(0.0, ep.f_object[on] - FORCE_CAP))) if on.any() else 0.0
        else:
            target = np.asarray(task["target"])
            if ep.grasped is not None:  # protocol v4: the grasp
                k_g = int(round(task["grasp_time"] / dt))
                m["grasp_success"] = float(ep.grasped); m["grasp_fail"] = 1.0 - float(ep.grasped)
                m["pregrasp_disturbance"] = float(np.linalg.norm(ep.obj_pos[min(n - 1, k_g - 1)] - ep.obj_pos[0]))
                filt_t = np.zeros(n); acc_t = 0.0
                for i in range(min(n, k_g)):
                    acc_t += a * (ep.f_object[i] - acc_t); filt_t[i] = acc_t
                m["grasp_force_peak_20ms"] = float(filt_t[:max(1, min(n, k_g))].max())
            m["place_error"] = float(np.hypot(*(ep.obj_pos[-1][:2] - target[:2])))
            q1 = ep.obj_quat[-1]
            up = np.array([2 * (q1[1] * q1[3] + q1[0] * q1[2]), 2 * (q1[2] * q1[3] - q1[0] * q1[1]), 1 - 2 * (q1[1] ** 2 + q1[2] ** 2)])  # body z axis in the world
            m["place_tilt"] = float(np.arccos(np.clip(abs(up[2]), 0.0, 1.0)))  # the block's axis against the vertical, either way up (it hangs axis-down in the gripper)
            place = ref.phase_mask("place")[:n]
            m["place_force_peak_20ms"] = float(filt_o[place].max()) if place.any() else float("nan")
            lift = ref.phase_mask("lift")[:n]
            m["lift_rmse"] = _rms(err[lift]) if lift.any() else float("nan")
            m["task_success"] = float(m["place_error"] < 0.02 and m["place_tilt"] < np.radians(15) and (ep.grasped is None or ep.grasped))
            m["f_peak"], m["f_peak_20ms"] = float(src.max()), float(filt_o.max())
            m["excess_force_cap"] = float(np.mean(np.maximum(0.0, src[src > 0.5] - FORCE_CAP))) if (src > 0.5).any() else 0.0
    if ep.regime in ("peg", "drawer", "door") and ep.obj_pos is not None:  # protocol v5 (ADR 0012)
        task = case.meta["task"]
        f_src = ep.f_object_env if ep.regime == "peg" else ep.f_object  # the peg against the fixture; the tool against the panel
        filt_o = np.zeros(n); acc = 0.0; a = dt / 0.02
        for i in range(n):
            acc += a * (f_src[i] - acc); filt_o[i] = acc
        m["f_peak"], m["f_peak_20ms"] = float(f_src.max()), float(filt_o.max())
        on = f_src > 0.5
        m["excess_force_cap"] = float(np.mean(np.maximum(0.0, f_src[on] - FORCE_CAP))) if on.any() else 0.0
        if ep.regime == "peg":
            q = ep.obj_quat
            z_b = np.stack([2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2]), 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)], 1)  # the peg's axis in the world
            tip = ep.obj_pos + task["peg_half_len"] * z_b
            depth = task["hole_top_z"] - tip[:, 2]
            ins = ref.phase_mask("insert")[:n]
            d_ins = depth[ins] if ins.any() else depth
            m["insertion_depth"] = float(d_ins.max())
            d_end = float(d_ins[-1])
            m["depth_deficit"] = float(max(0.0, task["target_depth"] - d_end))
            m["insertion_fail"] = float(d_end < 0.020)
            m["task_success"] = 1.0 - m["insertion_fail"]
        else:
            push = ref.phase_mask("push")[:n]
            c = f_src[push] > 0.5
            m["object_contact_fraction"] = float(c.mean()) if push.any() else float("nan")
            m["contact_breaks"] = float(_count_breaks(c, int(round(0.005 / dt)))) if push.any() else float("nan")
            m["contact_loss_rate"] = m["contact_breaks"] / (float(push.sum()) * dt) if push.any() else float("nan")
            if ep.regime == "drawer":
                travel = float((ep.obj_pos[-1] - ep.obj_pos[0]) @ np.asarray(task["axis"]))
                m["travel"] = travel
                m["travel_deficit"] = float(max(0.0, task["travel"] - travel))
                m["closed_fail"] = float(travel < task["travel"] - 0.005)
            else:
                yaw = lambda q: np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                q_end = float((yaw(ep.obj_quat[-1]) - yaw(np.asarray(task["base_quat"])) + np.pi) % (2 * np.pi) - np.pi)  # the hinge angle left (closed at 0)
                m["travel"] = float(task["object_q0"] - q_end)
                m["travel_deficit"] = float(max(0.0, q_end - task["target_angle"]))
                m["closed_fail"] = float(q_end > np.radians(2.0))
            m["task_success"] = 1.0 - m["closed_fail"]
    if ep.regime == "torque":  # torque-tracking family (protocol v4, ADR 0011): realised torque against the upper command at the true state
        from llc.controllers.program import chirp_frequency

        prog = case.meta["program"]
        d_tau = (ep.tau - ep.tau_cmd) / TORQUE_LIMITS
        m["tau_track_rmse"] = _rms(d_tau)
        j = int(prog["joint"])
        if prog["kind"] != "hold":
            a, b = ep.tau[:, j] - ep.tau[:, j].mean(), ep.tau_cmd[:, j] - ep.tau_cmd[:, j].mean()
            lags = range(0, int(round(0.05 / dt)) + 1)
            errs = [float(np.mean((a[lag:] - b[:n - lag]) ** 2)) for lag in lags]
            m["tau_track_delay"] = float(lags[int(np.argmin(errs))] * dt)
        if prog["kind"] in ("chirp", "torque_chirp"):  # amplitude ratio per chirp cycle; bandwidth = the first frequency where two consecutive cycles fall below 0.7
            f = chirp_frequency(prog, ep.t[:n])
            inside = np.isfinite(f)
            phase = np.cumsum(np.where(inside, 2 * np.pi * np.nan_to_num(f) * dt, 0.0))
            cycle = np.floor(phase / (2 * np.pi)).astype(int)
            ratios, freqs = [], []
            for c_id in range(int(cycle[inside].min()), int(cycle[inside].max())):
                idx = np.where(inside & (cycle == c_id))[0]
                if len(idx) < 4:
                    continue
                amp_c, amp_r = float(np.std(ep.tau_cmd[idx, j])), float(np.std(ep.tau[idx, j]))
                if amp_c > 1e-6:
                    ratios.append(amp_r / amp_c); freqs.append(float(np.mean(f[idx])))
            bw = float(prog.get("f1", 125.0))
            for i in range(len(ratios) - 1):
                if ratios[i] < 0.7 and ratios[i + 1] < 0.7:
                    bw = freqs[i]
                    break
            m["tau_track_bandwidth"] = bw
    # protocol v2 metrics (defined for every regime; contact is the measured contact flag)
    in_contact = ep.n_tool_contacts > 0
    m["excess_force"] = float(np.mean(np.maximum(0.0, ep.f_normal[in_contact] - FORCE_SOFT_LIMIT))) if in_contact.any() else 0.0
    if ep.regime == "erase":  # the held block's force on the table (ADR 0012)
        on = ep.f_object_env > 0.5
        m["excess_force_cap"] = float(np.mean(np.maximum(0.0, ep.f_object_env[on] - FORCE_CAP))) if on.any() else 0.0
    elif ep.regime not in ("push", "pick", "peg", "drawer", "door"):
        m["excess_force_cap"] = float(np.mean(np.maximum(0.0, ep.f_normal[in_contact] - FORCE_CAP))) if in_contact.any() else 0.0
    if tool_ref is not None:
        d = tool_ref[:n] - ep.tool_pos
        normal = case.meta.get("normal")
        if normal is not None and in_contact.any():
            nrm = np.asarray(normal, float)
            d_c = d[in_contact] - np.outer(d[in_contact] @ nrm, nrm)
            d_all = np.concatenate([np.linalg.norm(d_c, axis=1), np.linalg.norm(d[~in_contact], axis=1)])
        else:
            d_all = np.linalg.norm(d, axis=1)
        m["feasible_tracking"] = _rms(d_all)
    press_mask = ref.phase_mask("press")[:n] | ref.phase_mask("press2")[:n]
    protocol = case.meta.get("protocol", "v1")
    if press_mask.any():
        m["press_deficit"] = float(max(0.0, PRESS_FLOOR.get(protocol, 5.0) - (ep.f_object_env if ep.regime == "erase" else ep.f_normal)[press_mask].mean()))
    if ep.regime in ("contact2", "blocked", "task") and tool_ref is not None and case.meta.get("normal") is not None and case.meta.get("surface") is not None:
        from llc.learning.data import surface_depth

        depth_ref = surface_depth(np.asarray(tool_ref[:n]), case.meta["surface"])
        nrm = np.asarray(case.meta["normal"], float)
        settled = np.ones(n, bool)
        first = np.where(in_contact)[0]
        if first.size:
            settled[: first[0] + int(round(0.3 / dt))] = False  # the impact transient is not stiffness
        # contact stiffness: force per metre of the tool's actual error along the inward normal (a spring on the error, ADR 0008)
        mask = (depth_ref > 0.0) & in_contact & settled
        e_in = (ep.tool_pos - np.asarray(tool_ref[:n])) @ nrm
        if mask.sum() > 50 and e_in[mask].mean() > 0.002:
            m["contact_impedance"] = float(ep.f_normal[mask].mean() / e_in[mask].mean())
    if ep.regime == "blocked" and tool_ref is not None and case.meta.get("normal") is not None and case.meta.get("surface") is not None:
        mask = (depth_ref > 0.0) & settled
        if mask.sum() > 50 and depth_ref[mask].mean() > 0.005:
            m["apparent_stiffness"] = float(ep.f_normal[mask].mean() / depth_ref[mask].mean())
        # stop at the surface and wait: tangential drift from the first-contact point and motion while the reference is inside
        inside = depth_ref > 0.0
        on = np.where(inside & in_contact)[0]
        if on.size > 50:
            nrm = np.asarray(case.meta["normal"], float)
            anchor = ep.tool_pos[on[0]]
            after = np.zeros(n, bool); after[on[0]:] = True
            win = inside & after
            d = ep.tool_pos[win] - anchor
            d_t = d - np.outer(d @ nrm, nrm)
            m["blocked_drift"] = float(np.linalg.norm(d_t, axis=1).mean())
            speed = np.linalg.norm(np.diff(ep.tool_pos, axis=0), axis=1) / dt
            moving = win[1:] & in_contact[1:] & (speed > 0.01)
            m["blocked_motion"] = float(moving.sum() / max(1, (win[1:] & in_contact[1:]).sum()))
    if ep.regime == "blocked":
        rel = ref.phase_mask("release")[:n]
        if rel.any() and tool_ref is not None:
            k_r = int(np.where(rel)[0][0])
            err_tool = np.linalg.norm(tool_ref[k_r:n] - ep.tool_pos[k_r:], axis=1)
            below = np.where(err_tool < 0.01)[0]
            m["contact_recovery"] = float(below[0] * dt) if below.size else float(ep.t[-1] - ep.t[k_r])
            # release reaction (ADR 0009): the tool moves toward the reference faster than 2 cm/s
            to_ref = tool_ref[k_r:n] - ep.tool_pos[k_r:]
            unit = to_ref / (np.linalg.norm(to_ref, axis=1, keepdims=True) + 1e-9)
            v_tool = np.vstack([np.zeros((1, 3)), np.diff(ep.tool_pos[k_r:], axis=0) / dt])
            toward = np.where((v_tool * unit).sum(1) > 0.02)[0]
            m["release_reaction"] = float(toward[0] * dt) if toward.size else float(ep.t[-1] - ep.t[k_r])
            # release settling: from the reference's arrival at its final waypoint (it stops moving) until the error is below 1 cm
            moving = np.linalg.norm(np.diff(tool_ref[:n], axis=0), axis=1) > 1e-6
            k_a = int(np.where(moving)[0][-1] + 1) if moving.any() else k_r
            k_a = max(k_a, k_r)
            err_a = np.linalg.norm(tool_ref[k_a:n] - ep.tool_pos[k_a:], axis=1)
            below_a = np.where(err_a < 0.01)[0]
            m["release_settle"] = float(below_a[0] * dt) if below_a.size else float(ep.t[-1] - ep.t[k_a])
    terms = (COMPOSITE_V2 if ep.regime in COMPOSITE_V2 and case.meta.get("protocol", "v1") == "v2" else COMPOSITE).get(ep.regime) or COMPOSITE_V2.get(ep.regime) or COMPOSITE_V3[ep.regime]
    comp = sum(w * (m[k] if np.isfinite(m[k]) else 0.0) / s for k, (w, s) in terms.items())
    if ep.diverged:
        comp += DIVERGED_PENALTY
    m["composite"] = float(comp)
    terms_cap = (COMPOSITE_V3 if case.meta.get("protocol") == "v3" else COMPOSITE_CAP).get(ep.regime, terms)
    comp_cap = sum(w * (m[k] if np.isfinite(m[k]) else 0.0) / s for k, (w, s) in terms_cap.items()) + (DIVERGED_PENALTY if ep.diverged else 0.0)
    m["composite_cap"] = float(comp_cap)
    return m
