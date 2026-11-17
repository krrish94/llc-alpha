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
    "excess_force": ("Excess contact force above 30 N (mean while in contact)", "N"),
    "excess_force_cap": ("Excess contact force above the 15 N cap (mean while in contact)", "N"),
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
    "task_success": ("Task success (push: errors below 2 cm and yaw below 15 degrees; place: error below 2 cm and tilt below 15 degrees)", "-"),
    "diverged": ("Diverged", "-"),
    "composite": ("Composite tuning objective (protocol v1)", "-"),
    "composite_cap": ("Composite under the 15 N cap (contact deep dive selection objective)", "-"),
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
    "pick": {"place_error": (1.0, 0.02), "place_force_peak_20ms": (1.0, 25.0), "lift_rmse": (1.0, 0.02), "torque_rate_rms": (0.2, 500.0)},
}
PRESS_FLOOR = {"v2": 5.0, "v3": 7.0}  # N, the press-deficit threshold per protocol
FORCE_SOFT_LIMIT = 30.0  # N, protocol v2 excess-force threshold
FORCE_CAP = 15.0  # N, sustained-force cap of the contact deep dive (ADR 0007): pick-and-place, peg insertion and light connectors
REFLEX_SUSTAINED = 25.0  # N, 20 ms filtered force above which a hardware collision reaction would trigger (ADR 0007)
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

    if ep.regime in ("contact", "transition", "contact2", "blocked", "task"):
        in_contact = ep.n_tool_contacts > 0
        m["f_peak"] = float(ep.f_normal.max())
        filt, a = np.zeros(n), dt / 0.02  # first-order low-pass, 20 ms: what a collision reaction on filtered torque estimates sees
        acc = 0.0
        for i in range(n):
            acc += a * (ep.f_normal[i] - acc); filt[i] = acc
        m["f_peak_20ms"] = float(filt.max())
        press = ref.phase_mask("press")[:n] | ref.phase_mask("slide")[:n]
        if press.any():
            f = ep.f_normal[press]
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
            m["impulse_onset"] = float(np.sum(ep.f_normal[w]) * dt)
            c = in_contact[k_on:min(n, k_on + int(round(0.3 / dt)))]
            m["bounces"] = float(_count_breaks(c, int(round(0.005 / dt))))
            # force settling: time after onset until the force stays within 20% of its press-phase mean
            if press.any() and m["f_mean"] > 0 and ep.regime in ("contact", "transition"):
                k_end = int(round(ref.phases[1][1] / dt))
                f = ep.f_normal[k_on:min(n, k_end)]
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
            m["place_error"] = float(np.hypot(*(ep.obj_pos[-1][:2] - target[:2])))
            q1 = ep.obj_quat[-1]
            up = np.array([2 * (q1[1] * q1[3] + q1[0] * q1[2]), 2 * (q1[2] * q1[3] - q1[0] * q1[1]), 1 - 2 * (q1[1] ** 2 + q1[2] ** 2)])  # body z axis in the world
            m["place_tilt"] = float(np.arccos(np.clip(abs(up[2]), 0.0, 1.0)))  # the block's axis against the vertical, either way up (it hangs axis-down in the gripper)
            place = ref.phase_mask("place")[:n]
            m["place_force_peak_20ms"] = float(filt_o[place].max()) if place.any() else float("nan")
            lift = ref.phase_mask("lift")[:n]
            m["lift_rmse"] = _rms(err[lift]) if lift.any() else float("nan")
            m["task_success"] = float(m["place_error"] < 0.02 and m["place_tilt"] < np.radians(15))
            m["f_peak"], m["f_peak_20ms"] = float(src.max()), float(filt_o.max())
            m["excess_force_cap"] = float(np.mean(np.maximum(0.0, src[src > 0.5] - FORCE_CAP))) if (src > 0.5).any() else 0.0
    # protocol v2 metrics (defined for every regime; contact is the measured contact flag)
    in_contact = ep.n_tool_contacts > 0
    m["excess_force"] = float(np.mean(np.maximum(0.0, ep.f_normal[in_contact] - FORCE_SOFT_LIMIT))) if in_contact.any() else 0.0
    if ep.regime not in ("push", "pick"):
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
        m["press_deficit"] = float(max(0.0, PRESS_FLOOR.get(protocol, 5.0) - ep.f_normal[press_mask].mean()))
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
