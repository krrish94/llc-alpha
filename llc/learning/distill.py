"""Distillation of the privileged contact-impedance oracle into a proprioceptive controller (ADR 0008):
teacher labels along student rollouts (DAgger, Ross et al. 2011).

``TeacherStudent`` runs the student (a ``LearnedController``) and the teacher (the oracle around a
frozen base controller) side by side in the simulator loop. Per episode either the teacher's torque
or the student's is executed; both controllers keep the executed torque in their histories. At every
student decision it records the student's observation, the quantities the student's fast loop needs
and the teacher's torque at that state. ``fit_student`` regresses the student's torque on the
teacher's (normalised by the torque limits) through the student's output parameterisation.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
import torch

from llc.controllers.factory import build_controller
from llc.controllers.learned import LearnedController, load_learned_controller
from llc.eval.parallel import CaseSpec, _get_case, _get_sim, _init_worker, _worker
from llc.eval.rollout import rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS

FIELDS = ("obs", "q", "qd", "q_ref", "qd_ref", "gravity", "tau_teacher", "jn", "kp_scale", "kd_scale")  # kp_scale, kd_scale: the episode's base-gain multipliers (ADR 0012), ones elsewhere  # jn: the joint torque of a unit force along the surface normal at the decision (zero outside contact), for the force-space label term (ADR 0011 amendment 3)
FORCE_SCALE = float(__import__("llc.sim.plant", fromlist=["PLANT"]).PLANT.force_cap)  # N, the unit of the force-space label term


class TeacherStudent:
    name = "teacher_student"

    def __init__(self, student: LearnedController, teacher, nominal: NominalModel, execute_teacher: bool):
        self.student, self.teacher, self.nominal, self.execute_teacher = student, teacher, nominal, execute_teacher
        self.preview = student.preview
        self.policy = student.policy
        self.reset()

    def reset(self) -> None:
        self.student.reset()
        self.teacher.reset()
        self.logs: list[tuple] = []

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None):
        before = self.student.n_updates
        tau_s = self.student(t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=q_ref_future)
        tau_t = self.teacher(t, q, qd, q_ref, qd_ref, qdd_ref)
        if self.student.n_updates > before:
            self.logs.append((self.student.last_obs[0].numpy().astype(np.float32), q.astype(np.float32), qd.astype(np.float32), q_ref.astype(np.float32),
                              qd_ref.astype(np.float32), self.student.base_term(q, qd_ref, qdd_ref).astype(np.float32), np.clip(np.asarray(tau_t, np.float32), -TORQUE_LIMITS, TORQUE_LIMITS).astype(np.float32), np.zeros(NUM_JOINTS, np.float32),
                              self.student.base_scale[0].astype(np.float32), self.student.base_scale[1].astype(np.float32)))  # the base term the fast loop adds; the label is the teacher's applied torque (ADR 0012 amendment 1); jn filled in after the rollout; the episode's gain scales
        tau = tau_t if self.execute_teacher else tau_s
        applied = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        self.student.last_tau = applied
        base = getattr(self.teacher, "base", self.teacher)
        if hasattr(base, "last_tau"):
            base.last_tau = applied
        return tau

    def base_term(self, q, qd_ref, qdd_ref):
        return self.student.base_term(q, qd_ref, qdd_ref)

    def frame(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None) -> dict:
        """Through a servo profile (ADR 0011): the student and the teacher decide at the host frame; the executed
        frame is the teacher's or the student's; the label is the teacher's torque at the frame instant."""
        from llc.sim.servo import frame_law

        fr_s = self.student.frame(t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=q_ref_future)
        fr_t = self.teacher.frame(t, q, qd, q_ref, qd_ref, qdd_ref)
        self.logs.append((self.student.last_obs[0].numpy().astype(np.float32), q.astype(np.float32), qd.astype(np.float32), q_ref.astype(np.float32),
                          qd_ref.astype(np.float32), self.student.base_term(q, qd_ref, qdd_ref).astype(np.float32), np.asarray(frame_law(fr_t, q, qd), np.float32), np.zeros(NUM_JOINTS, np.float32),
                          self.student.base_scale[0].astype(np.float32), self.student.base_scale[1].astype(np.float32)))
        return fr_t if self.execute_teacher else fr_s

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack([rec[i] for rec in self.logs]) for i, k in enumerate(FIELDS)} if self.logs else {k: np.zeros((0, NUM_JOINTS), np.float32) for k in FIELDS}


FREE_FAMILIES = ("hold", "free", "policy", "step")


def scale_base(spec: dict, base_scale) -> dict:
    """A controller spec whose PD base (its own gains, or the base of an oracle) is scaled by the episode's multipliers (ADR 0012)."""
    kp_s, kd_s = (np.asarray(base_scale[0], float), np.asarray(base_scale[1], float))
    if "kp" in spec and "kd" in spec:
        return {**spec, "kp": (np.asarray(spec["kp"], float) * kp_s).tolist(), "kd": (np.asarray(spec["kd"], float) * kd_s).tolist()}
    if "base" in spec and isinstance(spec["base"], dict):
        return {**spec, "base": scale_base(spec["base"], base_scale)}
    return spec


def _collect_one(args) -> dict[str, np.ndarray]:
    student_path, teacher_spec, case_spec, execute_teacher = args
    if teacher_spec.get("free_teacher") == "ideal" and case_spec.regime in FREE_FAMILIES:  # item D: ideal-robot labels in free space (the student drives; the label is not a controller)
        from llc.learning.ideal import collect_one as collect_ideal

        return collect_ideal((student_path, case_spec, False, int(case_spec.index), teacher_spec.get("servo")))
    teacher_spec = {k: v for k, v in teacher_spec.items() if k != "free_teacher"}
    if not _worker:
        _init_worker()
    case = _get_case(case_spec)
    sim = _get_sim(case.perturbation, seed=case.seed, scene=(case.meta.get("task") or {}).get("scene"))  # a task family brings its scene (protocol v4 and v5)
    nominal = _worker["nominal"]
    servo = teacher_spec.get("servo")  # ADR 0011: the episode runs through a servo profile; the student decides at the host frame
    teacher_spec = {k: v for k, v in teacher_spec.items() if k != "servo"}
    options = {"latency_s": 0.0, "dt": servo["period"]} if servo else {}
    if "base_scale" in case.meta:  # ADR 0012: the episode's base gains, for the student and for the teacher's base
        options["base_scale"] = case.meta["base_scale"]
        teacher_spec = scale_base(teacher_spec, case.meta["base_scale"])
    if "latency_s" in case.meta and not servo:
        options["latency_s"] = case.meta["latency_s"]
    student = load_learned_controller(student_path, nominal, **options)
    ctrl = TeacherStudent(student, build_controller(teacher_spec, nominal, case), nominal, execute_teacher)
    stride = student.hold  # physics steps between decisions
    if servo:
        from llc.sim.servo import ModelInterface, ServoLoop, get_profile

        profile = get_profile(servo["profile"], **servo.get("overrides", {}))
        ctrl = ServoLoop(ctrl, profile, ModelInterface(profile, np.asarray(servo["kp"]), np.asarray(servo["kd"])))
        stride = profile.period_steps
    ep = rollout(sim, ctrl, case)
    if servo:
        ctrl = ctrl.inner
    out = ctrl.arrays()
    normal = case.meta.get("normal")
    if normal is not None and len(out["q"]):  # the force-space label term's direction: a unit force along the surface normal, at decisions in contact
        steps = np.minimum(np.arange(len(out["q"])) * stride, len(ep.t) - 1)
        nrm = np.asarray(normal, float)
        for i in np.where(ep.n_tool_contacts[steps] > 0)[0]:
            out["jn"][i] = nominal.tool_jacobian(out["q"][i].astype(float))[0].T @ nrm
    out["diverged"] = np.array([ep.diverged])
    out["case_id"] = np.array([case.id])
    out["regime"] = np.array([case.regime])
    return out


def collect_labels(student_path: str, teacher_spec: dict, cases: list[CaseSpec], execute_teacher: list[bool], workers: int | None = None) -> list[dict[str, np.ndarray]]:
    """Teacher-labelled decisions along one episode per case (the student's or the teacher's rollout).
    ``teacher_spec["free_teacher"] == "ideal"`` labels the free-space families with the ideal-robot torque
    instead (ADR 0010 amendment 2, item D): the student drives those episodes whatever ``execute_teacher``."""
    jobs = [(str(student_path), teacher_spec, c, bool(e)) for c, e in zip(cases, execute_teacher)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return [_collect_one(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        return pool.map(_collect_one, jobs, chunksize=1)


class LabelStore:
    """Aggregated DAgger data with a cap: beyond ``max_rows`` decisions a uniform random subset is kept; ``max_rows <= 0``
    keeps everything (ADR 0012), the observations in half precision."""

    def __init__(self, max_rows: int, rng: np.random.Generator):
        self.max_rows, self.rng = max_rows, rng
        self.data: dict[str, np.ndarray] | None = None

    def add(self, batches: list[dict[str, np.ndarray]], weight_of=None) -> None:
        """``weight_of(regime) -> float`` weights every decision of an episode (the free-space fix of ADR 0010)."""
        new = {k: np.concatenate([b[k] for b in batches if len(b["obs"])]) for k in FIELDS}
        new["obs"] = new["obs"].astype(np.float16)
        new["weight"] = np.concatenate([np.full(len(b["obs"]), float(weight_of(str(b["regime"][0])) if weight_of else 1.0), np.float32) for b in batches if len(b["obs"])])
        self.data = new if self.data is None else {k: np.concatenate([self.data[k], new[k]]) for k in list(FIELDS) + ["weight"]}
        n = len(self.data["obs"])
        if self.max_rows > 0 and n > self.max_rows:
            keep = np.sort(self.rng.choice(n, self.max_rows, replace=False))
            self.data = {k: v[keep] for k, v in self.data.items()}

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data["obs"])


def label_loss(policy, batch: dict[str, torch.Tensor], force_weight: float = 0.0) -> torch.Tensor:
    """Mean squared torque error between the student's fast-loop torque and the teacher's, per torque
    limit, weighted per decision when the batch carries weights. With ``force_weight`` the force along the
    surface normal that the torque error implies (least squares through the tool Jacobian, per the force cap)
    is added at decisions in contact: the torque loss is blind to the direction of its residual in joint
    space, the contact force is not (ADR 0011 amendment 3)."""
    held = policy.decode(policy(batch["obs"].float()))
    tau = policy.fast_torque(held, batch["q"], batch["qd"], batch["q_ref"], batch["qd_ref"], batch["gravity"], batch.get("kp_scale"), batch.get("kd_scale"))
    tau = torch.clamp(tau, -policy.limits, policy.limits)  # the applied torque (ADR 0012 amendment 1: rows where the base saturates are kept, with the saturated law as the label)
    d = tau - batch["tau_teacher"]
    err = (d / policy.limits).square().mean(-1)
    if force_weight > 0.0 and "jn" in batch:
        jn = batch["jn"]
        jn2 = (jn * jn).sum(-1)
        df = (jn * d).sum(-1) / jn2.clamp_min(1e-9)
        err = err + force_weight * (df / FORCE_SCALE).square() * (jn2 > 0).float()
    w = batch.get("weight")
    return err.mean() if w is None else (w * err).sum() / w.sum()


def fit_student(policy, opt: torch.optim.Optimizer, store: LabelStore, epochs: int, batch_size: int, rng: np.random.Generator, force_weight: float = 0.0) -> dict:
    data = {k: torch.as_tensor(v) for k, v in store.data.items()}
    n = len(data["obs"])
    policy.train()
    first, last = None, None
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for i0 in range(0, n, batch_size):
            idx = perm[i0:i0 + batch_size]
            loss = label_loss(policy, {k: v[idx] for k, v in data.items()}, force_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            first = loss.item() if first is None else first
            last = loss.item()
    policy.eval()
    return {"loss_first": first, "loss_last": last, "n_rows": n}
