"""Protocol v2 evaluation cases (deep dive, ADR 0005): new reference families, robot instances,
held-out workspace and speeds. Case ids are deterministic in (regime, split, index)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from llc import references as R
from llc.eval.cases import Case, Pulse, case_seed
from llc.sim.instances import evaluation_instances, sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, DynamicsPerturbation, surface_params

REGIMES_V2 = ("hold", "free", "policy", "step", "contact2", "blocked", "task")
from llc.sim.plant import PLANT  # noqa: E402

PROTOCOL = "v2" if PLANT.name == "v2" else "v3"  # the v3 composites and press floor apply to every plant after the sphere one (ADR 0011)
TASK_REGIMES = ("push", "pick", "wipe", "erase", "peg", "drawer", "door") if PLANT.tasks else ()  # protocol v5 (ADR 0012) adds wipe, erase, peg, drawer, door  # Milestone 9 task families (gripper plant only)  # protocol v3 (ADR 0009): gripper plant, press band 7 to 12 N, release metrics; the families and splits are those of v2
N_EVAL_INSTANCES = 8
Q1_HELDOUT = PLANT.q1_heldout


def make_case_v2(regime: str, split: str, index: int, nominal: NominalModel, speed_scale: float = 1.0,
                 heldout_workspace: bool = False, payload: float | None = None, instance: DynamicsPerturbation | None = None,
                 ood_instance: bool = False, surface_stiffness: float | None = None, dist: str = "v3", base_scale: float | None = None, lag: float | None = None) -> Case:
    """One case. Evaluation splits use the fixed instance set (instance index = case index mod 8)
    with a payload drawn from {0, 1, 2} kg; the 'train' split draws a fresh instance and payload
    in 0 to 3 kg. ``heldout_workspace`` samples free/policy references in the held-out base-yaw
    range; ``ood_instance`` doubles the instance deviation ranges (held-out parameter errors);
    ``surface_stiffness`` (N/m, effective at the tool) replaces the scene's hard table and wall by a
    compliant one (ADR 0008 held-out groups)."""
    seed = case_seed(split, regime + ("-ho" if heldout_workspace else ""), index)
    rng = np.random.default_rng(seed)
    q1 = Q1_HELDOUT if heldout_workspace else R.Q1_TRAIN
    pulse, meta = None, {}
    if regime == "hold":
        ref = R.hold(rng, nominal)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        pulse = Pulse(1.0, 0.2, tuple(PLANT.pulse_force * d))
    elif regime == "torque":  # torque-tracking family (protocol v4): a hold reference with a torque program on the upper command
        from llc.controllers.program import PROGRAMS

        ref = R.hold(rng, nominal)
        kind = PROGRAMS[index % len(PROGRAMS)]
        j = int(rng.integers(0, NUM_JOINTS))
        amp = {"hold": 0.0, "step": float(rng.uniform(0.1, 0.3)), "chirp": 0.1, "torque_chirp": 0.02}[kind]
        meta["program"] = {"kind": kind, "joint": j, "amplitude": amp, "t_start": 0.5, "f0": 0.5, "f1": 125.0, "duration": 2.0}
    elif regime == "free":
        ref = R.free_joint(rng, nominal, rng.uniform(0.2, 0.7) * speed_scale, q1_range=q1)
    elif regime == "policy":
        ref = R.policy_like(rng, nominal, rng.uniform(0.2, 0.7) * speed_scale, float(rng.choice([10.0, 20.0, 50.0])), rng.uniform(0.002, 0.01), 0.05, q1_range=q1)
    elif regime == "step":
        ref = R.step(rng, nominal, step_scale=0.25 * speed_scale)
    elif regime in ("contact2", "blocked", "task", "wipe", "push", "pick", "erase", "peg", "drawer", "door"):
        if regime in ("push", "pick", "erase", "peg", "drawer", "door"):
            from llc import tasks as T

            fn = {"push": T.push, "pick": T.pick, "erase": T.erase, "peg": T.peg, "drawer": T.drawer, "door": T.door}[regime]
        else:
            fn = {"contact2": R.contact_v2, "blocked": R.blocked, "task": R.task, "wipe": R.wipe}[regime]
        ref = None
        for _ in range(30):
            ref = fn(rng, nominal, speed_scale)
            if ref is not None:
                break
        if ref is None:
            raise RuntimeError(f"no valid {regime} reference for case {index}")
    else:
        raise ValueError(regime)
    if instance is None:
        if split == "train" and dist == "v5":  # ADR 0012: the consolidation distribution; the reference's draws above are those of v3
            from llc.sim.instances import sample_consolidation_instance

            instance = sample_consolidation_instance(rng, payload_mass=float(rng.uniform(0.0, PLANT.payload_train_max)), payload_offset=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.10))
            if surface_stiffness is None:
                surface_stiffness = [None, 2000.0, 500.0][int(rng.integers(0, 3))]
            s_kp, s_kd = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))), float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            meta["base_scale"] = [(s_kp * rng.uniform(0.8, 1.2, NUM_JOINTS)).tolist(), (s_kd * rng.uniform(0.8, 1.2, NUM_JOINTS)).tolist()]
            meta["latency_s"] = float(rng.uniform(0.0, 0.01))
        elif split == "train":
            instance = sample_instance(rng, payload_mass=float(rng.uniform(0.0, PLANT.payload_train_max)), payload_offset=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.10))
        else:
            inst_rng = np.random.default_rng(seed + 7)
            pl = float(rng.choice(list(PLANT.payloads_eval))) if payload is None else payload
            if ood_instance:
                from llc.sim.instances import InstanceRanges
                r = InstanceRanges(mass_rel=0.10, inertia_rel=0.20, com_offset_m=0.01, damping_range=(0.3, 3.0), armature_rel=0.4, friction_loss_max=0.6, torque_rel=0.06)
                instance = sample_instance(inst_rng, r, payload_mass=pl)
            else:
                instance = evaluation_instances(N_EVAL_INSTANCES, payload_mass=pl)[index % N_EVAL_INSTANCES]
    if surface_stiffness is not None:
        instance = replace(instance, **surface_params(surface_stiffness))
    if lag is not None:  # the actuator-lag group (ADR 0012): a first-order lag and a 5 ms transport delay for every controller
        instance = replace(instance, actuation_lag=float(lag), actuation_delay=5e-3)
    meta.update({k: v for k, v in ref.meta.items() if k not in ("normal", "tool_path")})
    if "task" in ref.meta:
        meta["task"] = dict(meta["task"])
    meta.update({"speed_scale": speed_scale, "heldout_workspace": heldout_workspace, "payload": instance.payload_mass, "ood_instance": ood_instance, "surface_stiffness": surface_stiffness, "protocol": PROTOCOL})
    tool_ref = ref.meta.get("tool_path")
    if tool_ref is None:
        tool_ref = np.array([nominal.tool_pos(ref.q[i]) for i in range(0, len(ref.q), 10)])
        tool_ref = np.array([np.interp(ref.t, ref.t[::10], tool_ref[:, j]) for j in range(3)]).T
    meta["tool_ref"] = tool_ref
    meta["normal"] = ref.meta.get("normal")
    if base_scale is not None:  # a held-out stiffness group (ADR 0012): the learned controllers' base at this multiple, damping by its square root
        meta["base_scale"] = [[float(base_scale)] * NUM_JOINTS, [float(np.sqrt(base_scale))] * NUM_JOINTS]
    tag = f"{regime}-{split}{'-ho' if heldout_workspace else ''}{'-ood' if ood_instance else ''}{'' if speed_scale == 1.0 else f'-x{speed_scale:g}'}{'' if surface_stiffness is None else f'-k{surface_stiffness:g}'}{'' if base_scale is None else f'-b{base_scale:g}'}{'' if lag is None else f'-lag{1e3 * lag:g}'}-{index:03d}"
    return Case(tag, regime, seed, ref, ref.q[0].copy(), np.zeros(NUM_JOINTS), pulse, instance, meta)
