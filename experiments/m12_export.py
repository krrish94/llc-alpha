"""Deployment export of a learned controller (Milestone 12, ADR 0012 decision 6): the policy as TorchScript, a specification
of its inputs and outputs at the control rate, the base controller it rides on, and the measured inference time per decision.

The exported bundle (a directory) holds ``policy.pt`` (TorchScript, observation in, raw head out), ``policy_state.pt`` (the
checkpoint), ``spec.json`` (the observation layout: current quantities, the two-scale history windows, the external-torque
estimate, the deployment inputs; the decode of the head; the base PD's gains and feedforward; the decision period, the hold and
the latency the policy was trained with; the torque limits; the plant and the training run) and ``timing.json``.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m12_export.py --ckpt runs/m12/student_l4-s0/best.pt --out runs/m12/export_l4
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from llc.learning.policy import load_policy
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS
from llc.sim.plant import PLANT


def observation_layout(cfg) -> list[dict]:
    """The observation vector's segments in order, with their scales (``llc.learning.policy.encode_obs``)."""
    n = NUM_JOINTS
    seg = [{"name": "q", "size": n, "scale": "q / 3.0"}, {"name": "qd", "size": n, "scale": "qd / 2.0"}, {"name": "q_ref - q", "size": n, "scale": "/ 0.2"}, {"name": "qd_ref", "size": n, "scale": "/ 2.0"}]
    per = ["(hist_q - q) / 0.2", "hist_qd / 2.0", "hist_tau / torque_limits"] + (["(hist_ref - q_ref) / 0.2"] if cfg.ref_history else []) + (["hist_ext / torque_limits"] if cfg.ext_torque else [])
    seg.append({"name": "history", "samples": cfg.n_hist, "per_sample": per, "size": cfg.n_hist * len(per) * n,
                "window": f"{cfg.history} samples {cfg.history_stride} steps apart then {cfg.history_dense} consecutive samples, oldest first, at the physics rate (1 kHz)"})
    if cfg.preview:
        seg.append({"name": "preview", "size": cfg.preview * n})
    if cfg.base_scale_input:
        seg.append({"name": "log(base stiffness scale) / log 3", "size": n}); seg.append({"name": "log(base damping scale) / log 2", "size": n})
    if cfg.latency_input:
        seg.append({"name": "compute latency / 10 ms", "size": 1})
    return seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing-runs", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    policy, extra = load_policy(args.ckpt)
    cfg = policy.cfg
    policy.eval()
    example = torch.zeros(1, cfg.obs_dim)
    scripted = torch.jit.trace(policy, example)
    scripted.save(str(out / "policy.pt"))
    torch.save({"config": asdict(cfg), "state_dict": policy.state_dict(), "extra": extra}, out / "policy_state.pt")
    decode = {"pd_gain_residual": "kp_scale = exp(log lo + (log hi - log lo) sigmoid(out[:7] + b)), kd_scale likewise on out[7:14], residual = fraction * limits * tanh(out[14:21]); torque = kp_scale kp (q_ref - q) + kd_scale kd (qd_ref - qd) + base term + residual",
              "pd_residual": "residual = fraction * limits * tanh(out); torque = kp (q_ref - q) + kd (qd_ref - qd) + base term + residual"}.get(cfg.kind, cfg.kind)
    spec = {"plant": PLANT.name, "checkpoint": str(args.ckpt), "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
            "policy": {"kind": cfg.kind, "layers": cfg.layers, "hidden": cfg.hidden, "residual_blocks": cfg.residual_blocks, "obs_dim": cfg.obs_dim, "out_dim": cfg.out_dim, "parameters": int(sum(p.numel() for p in policy.parameters()))},
            "observation": observation_layout(cfg),
            "decode": decode, "gain_scale_range": list(cfg.gain_scale_range), "damping_scale_range": list(cfg.damping_scale_range), "residual_fraction": cfg.residual_fraction,
            "base": {"kp": list(cfg.kp_shared), "kd": list(cfg.kd_shared), "feedforward": cfg.feedforward, "base_term": "M(q) qdd_ref + C(q, qd_ref) qd_ref + g(q) from the nominal model" if cfg.feedforward else "g(q)",
                     "note": "the base gains are multiplied by the deployment stiffness and damping scales when the policy carries them as inputs"},
            "timing": {"physics_rate_hz": 1000.0, "decision_period_ms": 1e3 * cfg.hold_steps / cfg.rate_hz if cfg.rate_hz else cfg.hold_steps, "latency_trained_s": cfg.latency_s,
                       "fast_loop": "the held output's law at 1 kHz on the measured state"},
            "limits": {"torque": TORQUE_LIMITS.tolist()},
            "inputs_on_hardware": ["joint positions", "joint velocities", "the torque the loop commanded (history)", "external-torque estimate: commanded torque minus the nominal model's M(q) qdd + bias(q, qd), with qdd from the velocity difference" if cfg.ext_torque else None,
                                   "the base stiffness and damping scales set on the robot" if cfg.base_scale_input else None, "the measured compute latency" if cfg.latency_input else None, "the reference q_ref, qd_ref, qdd_ref at every physics step"],
            "unknown_to_the_policy": ["payload", "robot instance", "surface", "contact state"]}
    spec["inputs_on_hardware"] = [x for x in spec["inputs_on_hardware"] if x]
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    torch.set_num_threads(1)
    x = torch.randn(1, cfg.obs_dim)
    with torch.no_grad():
        for _ in range(100):
            scripted(x)
        t0 = time.perf_counter()
        for _ in range(args.timing_runs):
            scripted(x)
        t_script = (time.perf_counter() - t0) / args.timing_runs
        t0 = time.perf_counter()
        for _ in range(args.timing_runs):
            policy(x)
        t_eager = (time.perf_counter() - t0) / args.timing_runs
    timing = {"machine": platform.platform(), "processor": platform.processor() or platform.machine(), "threads": 1, "runs": args.timing_runs,
              "torchscript_ms_per_decision": 1e3 * t_script, "eager_ms_per_decision": 1e3 * t_eager, "decision_period_ms": spec["timing"]["decision_period_ms"]}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"exported {out}: {spec['policy']['parameters']} parameters, obs {cfg.obs_dim}, {1e3 * t_script:.3f} ms per decision (TorchScript, one thread) against a {timing['decision_period_ms']:.0f} ms period")


if __name__ == "__main__":
    main()
