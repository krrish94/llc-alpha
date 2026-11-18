"""Reference trajectories for the evaluation regimes.

Every reference is a joint-space trajectory sampled at the physics rate, with a list of
labelled phases used by the metrics. Task-space (tool) paths are converted to joint space
with the nominal model's IK once, so every controller tracks the identical reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from llc.sim.nominal import NominalModel
from llc.sim.plant import PLANT
from llc.sim.panda import HOME, NUM_JOINTS, VELOCITY_LIMITS

TABLE_TOP_Z = PLANT.table_top_z  # m, top face of the table block (0.15 on the Panda scenes)
WALL_FACE_X = PLANT.wall_face_x  # m, face of the wall block pointing toward the robot (0.66 on the Panda scenes)
PRESS_DEPTH = PLANT.press_depth  # m, commanded press depth of the contact family (protocol v3: 7 to 12 N under 5 N + 500 N/m x depth on the Panda)
TABLE_XY, WALL_YZ = PLANT.table_xy, PLANT.wall_yz  # base-point ranges of the surface families
TOOL_RADIUS = PLANT.tool_offset  # m, from the tool point to its contact face (the sphere's radius on plant v2, 0 for the fingertips of v3)
# Free-space workspace box for the tool tip (above the table, in front of the wall).
WORKSPACE_LO = np.array(PLANT.workspace_lo)
WORKSPACE_HI = np.array(PLANT.workspace_hi)


@dataclass
class Reference:
    t: np.ndarray  # (T,) s
    q: np.ndarray  # (T,7) rad
    qd: np.ndarray  # (T,7) rad/s
    qdd: np.ndarray  # (T,7) rad/s^2 (zero where the reference is discontinuous)
    regime: str
    phases: list[tuple[float, float, str]] = field(default_factory=list)  # (t_start, t_end, label)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def phase_mask(self, label: str) -> np.ndarray:
        mask = np.zeros_like(self.t, dtype=bool)
        for t0, t1, lab in self.phases:
            if lab == label:
                mask |= (self.t >= t0) & (self.t < t1)
        return mask


def min_jerk(t: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-jerk interpolation profile s(t) in [0,1] with derivatives."""
    s = np.clip(t / duration, 0.0, 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    acc = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
    return pos, vel, acc


def _random_free_config(rng: np.random.Generator, nominal: NominalModel, near: np.ndarray | None = None) -> np.ndarray:
    """A collision-free joint configuration whose tool tip lies in the free-space workspace box."""
    for _ in range(200):
        target = rng.uniform(WORKSPACE_LO, WORKSPACE_HI)
        q0 = HOME if near is None else near
        axis = None if rng.uniform() < 0.5 else np.array([0.0, 0.0, -1.0])
        q, err = nominal.ik(target, q0, tool_axis_world=axis)
        if err < 1e-3 and not nominal.in_collision(q):
            return q
    raise RuntimeError("could not sample a free configuration")


def _path_is_free(q_path: np.ndarray, nominal: NominalModel, stride: int = 25) -> bool:
    return not any(nominal.in_collision(q) for q in q_path[::stride])


def _segment_duration(dq: np.ndarray, speed: float) -> float:
    """Duration so that the peak min-jerk joint velocity is ``speed`` times the velocity limit."""
    peak_ratio = np.max(np.abs(dq) / VELOCITY_LIMITS)
    return max(0.3, 1.875 * peak_ratio / speed)  # min-jerk peak velocity is 1.875 * dq / T


def _waypoint_trajectory(
    waypoints: list[np.ndarray], durations: list[float], dwell: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segs_q, segs_qd, segs_qdd = [], [], []
    for i, dur in enumerate(durations):
        n = int(round(dur / dt))
        tt = np.arange(n) * dt
        s, sd, sdd = min_jerk(tt, dur)
        dq = waypoints[i + 1] - waypoints[i]
        segs_q.append(waypoints[i] + s[:, None] * dq)
        segs_qd.append(sd[:, None] * dq)
        segs_qdd.append(sdd[:, None] * dq)
        n_dwell = int(round(dwell / dt))
        segs_q.append(np.repeat(waypoints[i + 1][None], n_dwell, 0))
        segs_qd.append(np.zeros((n_dwell, NUM_JOINTS)))
        segs_qdd.append(np.zeros((n_dwell, NUM_JOINTS)))
    q = np.concatenate(segs_q)
    qd = np.concatenate(segs_qd)
    qdd = np.concatenate(segs_qdd)
    t = np.arange(len(q)) * dt
    return t, q, qd, qdd


def hold(rng: np.random.Generator, nominal: NominalModel, duration: float = 3.0, dt: float = 1e-3) -> Reference:
    q0 = _random_free_config(rng, nominal)
    n = int(round(duration / dt))
    z = np.zeros((n, NUM_JOINTS))
    return Reference(np.arange(n) * dt, np.repeat(q0[None], n, 0), z, z.copy(), "hold", [(0.0, duration, "hold")])


def smooth(rng: np.random.Generator, nominal: NominalModel, speed: float, n_segments: int = 3, dt: float = 1e-3) -> Reference:
    """Minimum-jerk motion through random waypoints; ``speed`` is the peak joint speed as a
    fraction of the velocity limit."""
    for _ in range(50):
        wps = [_random_free_config(rng, nominal)]
        for _ in range(n_segments):
            wps.append(_random_free_config(rng, nominal, near=wps[-1]))
        durs = [_segment_duration(wps[i + 1] - wps[i], speed) for i in range(n_segments)]
        t, q, qd, qdd = _waypoint_trajectory(wps, durs, dwell=0.3, dt=dt)
        if _path_is_free(q, nominal):
            return Reference(t, q, qd, qdd, "smooth", [(0.0, t[-1], "motion")], {"speed": speed})
    raise RuntimeError("no collision-free smooth trajectory found")


def zoh(rng: np.random.Generator, nominal: NominalModel, speed: float, rate_hz: float = 20.0, dt: float = 1e-3) -> Reference:
    """A smooth motion sampled and held at ``rate_hz`` with zero velocity targets, as a high-level
    policy would command it."""
    ref = smooth(rng, nominal, speed, dt=dt)
    period = int(round(1.0 / (rate_hz * dt)))
    idx = (np.arange(len(ref.t)) // period) * period
    q = ref.q[idx]
    z = np.zeros_like(q)
    return Reference(ref.t, q, z, z.copy(), "zoh", [(0.0, ref.t[-1], "motion")], {"speed": speed, "rate_hz": rate_hz})


def step(rng: np.random.Generator, nominal: NominalModel, step_scale: float, n_steps: int = 4, dt: float = 1e-3) -> Reference:
    """Piecewise-constant targets: every 0.75 s the target jumps by a random vector with
    per-joint magnitude up to ``step_scale`` rad; velocity target is zero."""
    for _ in range(50):
        targets = [_random_free_config(rng, nominal)]
        ok = True
        for _ in range(n_steps):
            for _ in range(50):
                cand = targets[-1] + rng.uniform(-step_scale, step_scale, NUM_JOINTS)
                cand = np.clip(cand, nominal.joint_range[:, 0] + 0.05, nominal.joint_range[:, 1] - 0.05)
                tip = nominal.tool_pos(cand)
                if not nominal.in_collision(cand) and np.all(tip > WORKSPACE_LO - 0.05) and np.all(tip < WORKSPACE_HI + 0.05):
                    targets.append(cand)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        hold_n = int(round(0.75 / dt))
        q = np.concatenate([np.repeat(tg[None], hold_n, 0) for tg in targets])
        z = np.zeros_like(q)
        t = np.arange(len(q)) * dt
        phases = [(i * 0.75, (i + 1) * 0.75, "step" if i > 0 else "settle") for i in range(len(targets))]
        # Straight-line interpolation between consecutive targets is the path a stiff controller
        # takes; check it for collisions.
        path = np.concatenate([np.linspace(targets[i], targets[i + 1], 50) for i in range(n_steps)])
        if _path_is_free(path, nominal, stride=5):
            return Reference(t, q, z, z.copy(), "step", phases, {"step_scale": step_scale, "step_times": [i * 0.75 for i in range(1, len(targets))]})
    raise RuntimeError("no collision-free step reference found")


def _ik_multistart(nominal: NominalModel, target: np.ndarray, q0: np.ndarray, axis: np.ndarray | None,
                   rng: np.random.Generator, n_starts: int = 8) -> tuple[np.ndarray, float]:
    """IK from ``q0`` and from random perturbations of it; returns the best collision-free solution."""
    best, best_err = None, np.inf
    for i in range(n_starts):
        start = q0 if i == 0 else np.clip(q0 + rng.normal(0.0, 0.6, NUM_JOINTS), nominal.joint_range[:, 0] + 0.1, nominal.joint_range[:, 1] - 0.1)
        q, err = nominal.ik(target, start, tool_axis_world=axis, iters=300)
        if axis is not None:
            _, rot = nominal.tool_frame(q)
            err = max(err, 0.05 * float(np.linalg.norm(np.cross(rot[:, 2], axis))))  # 5 cm per radian of axis error
        if err < best_err:
            best, best_err = q, err
        if best_err < 1e-3:
            break
    return best, best_err


def _tool_path_to_joint(path: np.ndarray, nominal: NominalModel, q_start: np.ndarray, axis: np.ndarray,
                        dt: float, stride: int = 10) -> np.ndarray | None:
    """IK along a dense tool path, solved every ``stride`` samples with warm starts and cubic
    interpolation in between. Returns None if the joint path jumps (IK branch change)."""
    from scipy.interpolate import CubicSpline

    idx = np.arange(0, len(path), stride)
    if idx[-1] != len(path) - 1:
        idx = np.append(idx, len(path) - 1)
    q = q_start.copy()
    coarse = np.empty((len(idx), NUM_JOINTS))
    for k, i in enumerate(idx):
        q, err = nominal.ik(path[i], q, tool_axis_world=axis, iters=50)
        if err > 2e-3:
            return None
        coarse[k] = q
    if np.abs(np.diff(coarse, axis=0)).max() > 0.15:  # rad per 10 ms: a branch change, not motion
        return None
    return CubicSpline(idx * dt, coarse, axis=0)


def _tool_waypoints(points: list[np.ndarray], speeds: list[float], dwell: list[float], dt: float):
    """Minimum-jerk tool-space path through points at given peak speeds (m/s), with dwell times."""
    segs, phases_t = [], [0.0]
    for i in range(len(points) - 1):
        dist = np.linalg.norm(points[i + 1] - points[i])
        dur = max(0.2, 1.875 * dist / speeds[i])
        n = int(round(dur / dt))
        s, _, _ = min_jerk(np.arange(n) * dt, dur)
        segs.append(points[i] + s[:, None] * (points[i + 1] - points[i]))
        segs.append(np.repeat(points[i + 1][None], int(round(dwell[i] / dt)), 0))
        phases_t.append(phases_t[-1] + dur)
        phases_t.append(phases_t[-1] + dwell[i])
    return np.concatenate(segs), phases_t


def _contact_reference(
    nominal: NominalModel, surface: str, approach_speed: float, slide_speed: float | None, depth: float,
    rng: np.random.Generator, dt: float, regime: str,
) -> Reference | None:
    """Tool path: start 5 cm off the surface, approach, press ``depth`` into it, optionally slide
    along it, hold, then retract. ``surface`` is 'table' (contact along -z) or 'wall' (along +x)."""
    if surface == "table":
        normal = np.array([0.0, 0.0, 1.0])
        base = np.array([rng.uniform(*TABLE_XY[0]), rng.uniform(*TABLE_XY[1]), TABLE_TOP_Z + TOOL_RADIUS])
        axis = np.array([0.0, 0.0, -1.0])
        slide_dir = np.array([0.0, 1.0, 0.0])
    else:
        normal = np.array([-1.0, 0.0, 0.0])
        base = np.array([WALL_FACE_X - TOOL_RADIUS, rng.uniform(*WALL_YZ[0]), rng.uniform(*WALL_YZ[1])])
        axis = np.array([0.5, 0.0, -np.sqrt(3) / 2])  # tool tilted 60 deg down so the wrist stays clear of the wall
        slide_dir = np.array([0.0, 0.0, 1.0])
    off = base + 0.03 * normal
    pressed = base - depth * normal
    pts, speeds, dwell = [off, pressed], [approach_speed], [0.5]
    labels = ["approach", "press"]
    if slide_speed is not None:
        sign = 1.0 if rng.uniform() < 0.5 else -1.0
        slide_len = 0.12 if surface == "table" else 0.08
        pts += [pressed + sign * slide_len * slide_dir]
        speeds += [slide_speed]
        dwell += [0.3]
        labels += ["slide", "press"]
    pts += [pts[-1] + 0.03 * normal]
    speeds += [approach_speed]
    dwell += [0.5]
    labels += ["release", "free"]
    path, times = _tool_waypoints(pts, speeds, dwell, dt)
    q_start, err = _ik_multistart(nominal, off, HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = _tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.6:  # near-singular posture: reject the sample
        return None
    phases = [(times[i], times[i + 1], labels[i]) for i in range(len(labels))]
    # The expected first-contact time under perfect tracking: when the tool path crosses the surface.
    signed = (path - base) @ normal
    t_contact = float(t[np.argmax(signed < 0.0)])
    return Reference(t, q, qd, qdd, regime, phases, {
        "tool_path": path, "surface": surface, "approach_speed": approach_speed, "slide_speed": slide_speed,
        "depth": depth, "t_contact_nominal": t_contact, "normal": normal.tolist(),
    })


def contact(rng: np.random.Generator, nominal: NominalModel, slide_speed: float, depth: float = 0.01, dt: float = 1e-3) -> Reference:
    """Sustained contact: slow approach, press, slide along the table, retract."""
    for _ in range(20):
        ref = _contact_reference(nominal, "table", 0.1, slide_speed, depth, rng, dt, "contact")
        if ref is not None:
            return ref
    raise RuntimeError("no valid contact reference found")


def transition(rng: np.random.Generator, nominal: NominalModel, approach_speed: float, depth: float = 0.01, dt: float = 1e-3) -> Reference:
    """Contact transition: approach at ``approach_speed``, press, release; table or wall."""
    surface = "table" if rng.uniform() < 0.5 else "wall"
    for _ in range(20):
        ref = _contact_reference(nominal, surface, approach_speed, None, depth, rng, dt, "transition")
        if ref is not None:
            return ref
    raise RuntimeError("no valid transition reference found")


# ----------------------------------------------------------------------------- protocol v2 families
JOINT_MARGIN = PLANT.joint_margin  # rad kept from the joint limits when sampling configurations
Q1_TRAIN = PLANT.q1_train  # base-yaw range used for training; the plant's q1_heldout is the held-out workspace


def _random_joint_config(rng: np.random.Generator, nominal: NominalModel, q1_range=Q1_TRAIN, near: np.ndarray | None = None,
                         max_step: float = 1.0) -> np.ndarray:
    """Collision-free configuration sampled uniformly in joint space (tool tip above the floor); with
    ``near`` given, within ``max_step`` rad per joint of it, so that segments stay a few seconds long."""
    lo, hi = nominal.joint_range[:, 0] + JOINT_MARGIN, nominal.joint_range[:, 1] - JOINT_MARGIN
    for _ in range(500):
        if near is None:
            q = rng.uniform(lo, hi)
            q[0] = rng.uniform(*q1_range)
        else:
            q = np.clip(near + rng.uniform(-max_step, max_step, NUM_JOINTS), lo, hi)
            q[0] = float(np.clip(q[0], *q1_range))
        if nominal.in_collision(q):
            continue
        if nominal.tool_pos(q)[2] < PLANT.tool_min_z:
            continue
        return q
    raise RuntimeError("no collision-free joint configuration found")


def free_joint(rng: np.random.Generator, nominal: NominalModel, speed: float, n_segments: int = 3, q1_range=Q1_TRAIN, dt: float = 1e-3) -> Reference:
    """Minimum-jerk motion through configurations sampled uniformly in joint space (protocol v2)."""
    for _ in range(50):
        wps = [_random_joint_config(rng, nominal, q1_range)]
        for _ in range(n_segments):
            wps.append(_random_joint_config(rng, nominal, q1_range, near=wps[-1]))
        durs = [_segment_duration(wps[i + 1] - wps[i], speed) for i in range(n_segments)]
        t, q, qd, qdd = _waypoint_trajectory(wps, durs, dwell=0.2, dt=dt)
        if _path_is_free(q, nominal, stride=20):
            return Reference(t, q, qd, qdd, "free", [(0.0, t[-1], "motion")], {"speed": speed, "q1_range": list(q1_range)})
    raise RuntimeError("no collision-free free-joint trajectory found")


def policy_like(rng: np.random.Generator, nominal: NominalModel, speed: float, rate_hz: float, jitter: float, jump_prob: float,
                q1_range=Q1_TRAIN, dt: float = 1e-3) -> Reference:
    """Free-joint motion sampled and held at ``rate_hz`` with Gaussian jitter (rad) on every held
    target and occasional jumps of up to 0.2 rad, as a noisy high-level policy would command."""
    base = free_joint(rng, nominal, speed, q1_range=q1_range, dt=dt)
    period = max(1, int(round(1.0 / (rate_hz * dt))))
    n = len(base.t)
    q = np.empty_like(base.q)
    for k0 in range(0, n, period):
        target = base.q[k0] + rng.normal(0.0, jitter, NUM_JOINTS)
        if rng.uniform() < jump_prob:
            target = target + rng.uniform(-0.2, 0.2, NUM_JOINTS)
        q[k0:k0 + period] = target
    z = np.zeros_like(q)
    return Reference(base.t, q, z, z.copy(), "policy", [(0.0, base.t[-1], "motion")], {"speed": speed, "rate_hz": rate_hz, "jitter": jitter, "jump_prob": jump_prob})


def _table_axis(rng: np.random.Generator) -> np.ndarray:
    """Tool axis of the table branch of the blocked and task families: straight down on the Panda (no random draw, so the
    record's streams are kept), tilted forward within the plant's range on arms whose wrist cannot point straight down."""
    lo, hi = PLANT.blocked_tilt_deg
    if hi <= 0.0:
        return np.array([0.0, 0.0, -1.0])
    t = np.radians(rng.uniform(lo, hi))
    return np.array([np.sin(t), 0.0, -np.cos(t)])


WALL_TILT_DEG = PLANT.wall_tilt_deg  # tool axis angle below horizontal at the wall (erratum 3: at 60 deg the hand reached the wall before the tool; v2 30 to 45, v3 0 to 15)
WALL_LINK_MARGIN = 0.02  # m: no robot link other than the tool within this distance of the wall at the deepest reference pose


def blocked(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3, wall_tilt_deg: tuple[float, float] = WALL_TILT_DEG,
            wall_depth: tuple[float, float] = PLANT.wall_depth, link_margin: float = WALL_LINK_MARGIN) -> Reference | None:
    """A free-space-looking tool path that passes through the table top or the wall: the commanded
    target cannot be tracked and the controller must yield, then recover when the path returns."""
    surface = "table" if rng.uniform() < 0.6 else "wall"
    if surface == "table":
        x, y = rng.uniform(*TABLE_XY[0]), rng.uniform(*TABLE_XY[1])
        a = np.array([x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05), TABLE_TOP_Z + rng.uniform(*PLANT.approach_height)])
        b = np.array([x, y, TABLE_TOP_Z - 0.06 + rng.uniform(0.0, 0.04)])  # 2 to 6 cm below the surface, with the tool tip inside the table (the Panda's random stream kept)
        c = np.array([x + rng.uniform(-0.08, 0.08), y + rng.uniform(-0.08, 0.08), TABLE_TOP_Z + rng.uniform(*PLANT.approach_height)])
        axis, normal, offset = _table_axis(rng), np.array([0.0, 0.0, 1.0]), TABLE_TOP_Z + TOOL_RADIUS
    else:
        y, z = rng.uniform(*WALL_YZ[0]), rng.uniform(*WALL_YZ[1])
        a = np.array([WALL_FACE_X - 0.21 + rng.uniform(0.0, 0.10), y + rng.uniform(-0.05, 0.05), z + rng.uniform(-0.05, 0.05)])
        b = np.array([WALL_FACE_X - TOOL_RADIUS + rng.uniform(*wall_depth), y, z])  # 3 to 6 cm inside the wall (erratum 3: 4 to 8 cm drove the links into the wall)
        c = np.array([WALL_FACE_X - 0.21 + rng.uniform(0.0, 0.10), y + rng.uniform(-0.08, 0.08), z + rng.uniform(-0.08, 0.08)])
        tilt = np.radians(rng.uniform(*wall_tilt_deg))
        axis, normal, offset = np.array([np.cos(tilt), 0.0, -np.sin(tilt)]), np.array([-1.0, 0.0, 0.0]), -(WALL_FACE_X - TOOL_RADIUS)
    v = rng.uniform(0.08, 0.3) * speed
    path, times = _tool_waypoints([a, b, c], [v, v], [rng.uniform(0.3, 0.8), 0.4], dt)
    q_start, err = _ik_multistart(nominal, a, HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = _tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.8:
        return None
    if surface == "wall":  # the block must happen at the tool: at the deepest reference pose no link may be within the margin of the wall
        k_deep = int(np.argmax(path[:, 0]))
        if nominal.link_contacts(q[k_deep], "wall", link_margin):
            return None
    inside = (path @ normal - offset) < 0.0  # table: z below the top face; wall: x beyond the face (erratum 2026-11-16: the table test had the wrong sign)
    k_in = int(np.argmax(inside)) if inside.any() else len(t) - 1
    k_out = int(len(inside) - np.argmax(inside[::-1])) if inside.any() else len(t)
    k_deep = int(np.argmax((path @ normal - offset) * -1.0))  # deepest reference pose along the inward normal
    depth_deep = float(-(path[k_deep] @ normal - offset))
    if not _links_clear(nominal, [(q[k_in], 0.0), (q[k_deep], depth_deep)], pressed=surface):  # v3: crossing and deepest pose, the latter lifted back to the surface
        return None
    phases = [(0.0, t[k_in], "approach"), (t[k_in], t[min(k_out, len(t) - 1)], "blocked"), (t[min(k_out, len(t) - 1)], t[-1], "release")]
    return Reference(t, q, qd, qdd, "blocked", phases, {"tool_path": path, "surface": surface, "speed": speed, "approach_speed": v, "normal": normal.tolist(),
                                                        "t_contact_nominal": float(t[k_in])})


LINK_CLEARANCE = 0.002  # m, protocol v3: real clearance every non-tool geom must keep from the wall and the table (the finger meshes end 5 mm behind the pad faces)


def _links_clear(nominal: NominalModel, samples, pressed: str | None = None) -> bool:
    """Protocol v3 (ADR 0009, gripper plant): at each sampled (q, depth) of the nominal path, every
    colliding robot geom other than the tool keeps ``LINK_CLEARANCE`` from the wall and the table once
    the tool is lifted back to the pressed surface (the commanded depth is virtual: the surface blocks
    the tool), i.e. signed distance + depth >= clearance toward ``pressed`` and signed distance >=
    clearance toward the other surface. On the sphere plant (v2) the generators keep their record and
    this check is skipped."""
    if PLANT.name != "v3":
        return True
    for q, depth in samples:
        for env in ("wall", "table"):
            lift = depth if env == pressed else 0.0
            if nominal.link_clearance(q, env) + lift < LINK_CLEARANCE:
                return False
    return True


def contact_v2(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> Reference | None:
    """Contact with random tool tilt, approach speed, commanded depth and sliding direction, on the
    table or the wall. Protocol v3 (ADR 0009): the commanded depth, 0.4 to 1.4 cm, asks for 7 to 12 N
    under the press law 5 N + 500 N/m x depth, the middle of the 5 to 15 N band (protocol v2: 0.5 to
    3 cm, partly infeasible)."""
    surface = "table" if rng.uniform() < 0.65 else "wall"
    depth = float(rng.uniform(*PRESS_DEPTH))
    approach = float(rng.uniform(0.05, 0.3) * speed)
    slide_speed = float(rng.uniform(0.05, 0.2) * speed)
    if surface == "table":
        normal = np.array([0.0, 0.0, 1.0])
        base = np.array([rng.uniform(*TABLE_XY[0]), rng.uniform(*TABLE_XY[1]), TABLE_TOP_Z + TOOL_RADIUS])
        tilt = np.radians(rng.uniform(*PLANT.table_tilt_deg))
        ang = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.sin(tilt) * np.cos(ang), np.sin(tilt) * np.sin(ang), -np.cos(tilt)])
        s_ang = rng.uniform(0, 2 * np.pi)
        slide_dir = np.array([np.cos(s_ang), np.sin(s_ang), 0.0])
        slide_len = 0.10
    else:
        normal = np.array([-1.0, 0.0, 0.0])
        base = np.array([WALL_FACE_X - TOOL_RADIUS, rng.uniform(*WALL_YZ[0]), rng.uniform(*WALL_YZ[1])])
        tilt = np.radians(rng.uniform(*WALL_TILT_DEG))  # erratum 3: 50 to 70 deg put the hand on the wall
        axis = np.array([np.cos(tilt), 0.0, -np.sin(tilt)])
        s_ang = rng.uniform(0, 2 * np.pi)
        slide_dir = np.array([0.0, np.cos(s_ang), np.sin(s_ang)])
        slide_len = 0.07
    off = base + 0.03 * normal
    pressed = base - depth * normal
    pts = [off, pressed, pressed + slide_len * slide_dir, pressed + slide_len * slide_dir + 0.03 * normal]
    speeds = [approach, slide_speed, approach]
    dwell = [0.4, 0.3, 0.3]
    labels = ["approach", "press", "slide", "release", "free"]
    path, times = _tool_waypoints(pts, speeds, dwell, dt)
    q_start, err = _ik_multistart(nominal, off, HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = _tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.8:
        return None
    if surface == "wall" and nominal.link_contacts(q[int(np.argmax(path[:, 0]))], "wall", WALL_LINK_MARGIN):  # erratum 3: the press must be at the tool
        return None
    if not _links_clear(nominal, [(q[min(len(q) - 1, int(times[i] / dt))], depth) for i in (1, 3)], pressed=surface):  # v3: press pose and slide end
        return None
    # phases from the waypoint times: approach, press(dwell), slide, press(dwell), release, free
    labs = ["approach", "press", "slide", "press", "release", "free"]
    phases = [(times[i], times[i + 1], labs[i]) for i in range(len(labs))]
    signed = (path - base) @ normal
    t_contact = float(t[np.argmax(signed < 0.0)])
    return Reference(t, q, qd, qdd, "contact2", phases, {"tool_path": path, "surface": surface, "approach_speed": approach, "slide_speed": slide_speed,
                                                         "depth": depth, "tilt_deg": float(np.degrees(tilt)), "t_contact_nominal": t_contact, "normal": normal.tolist()})


def task(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> Reference | None:
    """Task-like sequence: reach, descend, press 1 cm, slide 10 cm, lift, transport, descend, press
    0.5 cm, lift; all on the table with the tool pointing down (evaluation family)."""
    v_free, v_contact = 0.4 * speed, 0.1 * speed
    p1 = np.array([rng.uniform(TABLE_XY[0][0], TABLE_XY[0][1] - 0.02), rng.uniform(TABLE_XY[1][0], TABLE_XY[1][1] - 0.1), TABLE_TOP_Z + TOOL_RADIUS])
    s_ang = rng.uniform(0, 2 * np.pi)
    p2 = p1 + 0.10 * np.array([np.cos(s_ang), np.sin(s_ang), 0.0])
    p2[0] = np.clip(p2[0], TABLE_XY[0][0] - 0.01, TABLE_XY[0][1] + 0.02); p2[1] = np.clip(p2[1], TABLE_XY[1][0] - 0.05, TABLE_XY[1][1] + 0.05)
    p3 = np.array([rng.uniform(TABLE_XY[0][0], TABLE_XY[0][1] - 0.02), rng.uniform(TABLE_XY[1][0] + 0.1, TABLE_XY[1][1]), TABLE_TOP_Z + TOOL_RADIUS])
    start = p1 + np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(*PLANT.task_start_height)])
    up = np.array([0.0, 0.0, 1.0])
    pts = [start, p1 + 0.03 * up, p1 - 0.01 * up, p2 - 0.01 * up, p2 + 0.05 * up, p3 + 0.05 * up, p3 - 0.005 * up, p3 + 0.05 * up]
    speeds = [v_free, v_contact, v_contact, v_contact, v_free, v_contact, v_contact]
    dwell = [0.2, 0.5, 0.3, 0.2, 0.2, 0.3, 0.3]
    labs = ["reach", "descend", "press", "slide", "lift", "transport", "descend2", "press2", "lift2"]
    path, times = _tool_waypoints(pts, speeds, dwell, dt)
    axis = _table_axis(rng)
    q_start, err = _ik_multistart(nominal, start, HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = _tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.8:
        return None
    if not _links_clear(nominal, [(q[min(len(q) - 1, int(times[i] / dt))], d) for i, d in ((3, 0.01), (5, 0.01), (11, 0.005))], pressed="table"):  # v3: presses and the slide end
        return None
    # phase labels: each waypoint segment + its dwell; map to the task phase names
    seg_names = ["reach", "descend", "slide", "lift", "transport", "descend2", "lift2"]
    phases = []
    for i, name in enumerate(seg_names):
        phases.append((times[2 * i], times[2 * i + 1], name))
        dwell_label = {"descend": "press", "slide": "press", "descend2": "press2"}.get(name, name)
        phases.append((times[2 * i + 1], times[2 * i + 2], dwell_label))
    return Reference(t, q, qd, qdd, "task", phases, {"tool_path": path, "surface": "table", "speed": speed, "normal": [0.0, 0.0, 1.0],
                                                     "t_contact_nominal": float(t[np.argmax((path[:, 2] - (TABLE_TOP_Z + TOOL_RADIUS)) < 0.0)])})


def wipe(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> Reference | None:
    """Wiping (protocol v5, ADR 0012): approach, press to the band, a raster of three passes 8 cm long (6 cm on the wall)
    3 cm apart at 10 cm/s, release; table (70%) or wall. The metrics are the press-and-slide family's over the raster."""
    surface = "table" if rng.uniform() < 0.7 else "wall"
    depth = float(rng.uniform(*PRESS_DEPTH))
    approach = float(rng.uniform(0.05, 0.2) * speed)
    slide_speed = 0.1 * speed
    if surface == "table":
        normal = np.array([0.0, 0.0, 1.0])
        (x_lo, x_hi), (y_lo, y_hi) = TABLE_XY
        base = np.array([rng.uniform(x_lo, x_hi - 0.06), rng.uniform(y_lo + 0.05, y_hi - 0.05), TABLE_TOP_Z + TOOL_RADIUS])
        tilt = np.radians(rng.uniform(*PLANT.table_tilt_deg))
        ang = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.sin(tilt) * np.cos(ang), np.sin(tilt) * np.sin(ang), -np.cos(tilt)])
        along, across, length = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), 0.08
    else:
        normal = np.array([-1.0, 0.0, 0.0])
        (y_lo, y_hi), (z_lo, z_hi) = WALL_YZ
        base = np.array([WALL_FACE_X - TOOL_RADIUS, rng.uniform(y_lo + 0.04, y_hi - 0.04), rng.uniform(z_lo, z_hi - 0.06)])
        axis = np.array([0.5, 0.0, -np.sqrt(3) / 2])
        along, across, length = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), 0.06
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    off = base + 0.03 * normal
    pressed = base - depth * normal
    pts, speeds, dwell, labels = [off, pressed], [approach], [0.4], ["approach", "press"]
    p = pressed.copy()
    for i in range(3):  # three passes, stepping across between them
        p = p + sign * length * along * (1 if i % 2 == 0 else -1)
        pts.append(p.copy()); speeds.append(slide_speed); dwell.append(0.1); labels += ["slide", "press"]
        if i < 2:
            p = p + 0.03 * across
            pts.append(p.copy()); speeds.append(slide_speed); dwell.append(0.1); labels += ["slide", "press"]
    pts.append(pts[-1] + 0.03 * normal); speeds.append(approach); dwell.append(0.5); labels += ["release", "free"]
    path, times = _tool_waypoints(pts, speeds, dwell, dt)
    q_start, err = _ik_multistart(nominal, off, HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = _tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.6:
        return None
    phases = [(times[i], times[i + 1], labels[i]) for i in range(len(labels))]
    samples = [(q[min(len(q) - 1, int(times[i] / dt))], depth) for i in range(2, len(times) - 2, 2)]
    if not _links_clear(nominal, samples, pressed=surface):
        return None
    signed = (path - base) @ normal
    return Reference(t, q, qd, qdd, "wipe", phases, {"tool_path": path, "surface": surface, "approach_speed": approach, "slide_speed": slide_speed, "depth": depth,
                                                     "t_contact_nominal": float(t[np.argmax(signed < 0.0)]), "normal": normal.tolist(), "passes": 3, "pass_length": length})
