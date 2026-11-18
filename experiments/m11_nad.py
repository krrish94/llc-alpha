"""Milestone 11, Phase 0 (ADR 0011, decision 2): what the Neural Actuation Dataset identifies of the Dynamixel XM430 and the
Feetech STS3215 servo loops. Per joint: the command register (PWM duty on the XM430, the load register on the STS3215) regressed
on the position error and the reported velocity (the firmware's position loop as seen at the telemetry rate), the motor current
regressed on duty times bus voltage and on velocity (the DC-motor law i = (V_bus d - K_e N omega) / R), the feedback resolutions
observed in the data (the smallest non-zero velocity step), and the datasheet cross-checks (stall current and no-load speed from
the fitted R and K_e N). Writes runs/m11/nad_fit.json.

    PYTHONPATH=. uv run --no-sync python experiments/m11_nad.py --out runs/m11/nad_fit.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from llc.data.nad import load

XM430 = {"stall_current_A": 2.3, "stall_torque_Nm": 4.1, "no_load_rpm": 46.0, "gear_ratio": 353.5, "pwm_limit": 885, "bus_V": 12.0, "current_unit_mA": 2.69, "velocity_unit_rad_s": 0.229 * 2 * np.pi / 60, "position_counts": 4096}
STS3215 = {"stall_torque_Nm": 2.94, "no_load_rpm": 71.0, "bus_V": 12.0, "current_unit_mA": 6.5, "velocity_unit_rad_s": 50 * 2 * np.pi / 4096, "position_counts": 4096, "load_unit": 0.001}


def fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, float(1.0 - np.var(y - X @ coef) / np.var(y))


def omx(out: dict) -> None:
    d = load("omx", "train")
    joints = []
    for j in range(1, 6):
        e = d[f"goal_pos{j}"] - d[f"pos{j}"]; v = d[f"vel{j}"].astype(float); pwm = d[f"pwm{j}"].astype(float); cur = d[f"current{j}"].astype(float) / 1e3; volts = d[f"volts{j}"].astype(float)
        c1, r1 = fit(np.stack([e, v, np.ones_like(e)], 1), pwm)
        c2, r2 = fit(np.stack([pwm / XM430["pwm_limit"] * volts, v, np.ones_like(e)], 1), cur)
        R = 1.0 / c2[0] if c2[0] > 0 else float("nan")  # i = (d V) / R - (K_e N / R) omega
        ke_n = -c2[1] * R
        vu = np.unique(np.round(np.abs(v[v != 0]), 6))
        joints.append({"joint": j, "pwm_per_rad": float(c1[0]), "pwm_per_rad_s": float(c1[1]), "pwm_fit_r2": r1, "R_ohm": float(R), "KeN_Vs_per_rad": float(ke_n), "current_fit_r2": r2,
                       "stall_current_A_at_12V": float(12.0 / R), "no_load_rpm_at_12V": float(12.0 / ke_n * 60 / (2 * np.pi)), "velocity_step_rad_s": float(vu[0]) if vu.size else float("nan"),
                       "error_std_rad": float(np.std(e)), "current_range_A": [float(cur.min()), float(cur.max())]})
    Kt_n = XM430["stall_torque_Nm"] / XM430["stall_current_A"]  # Nm per A at the joint, from the datasheet stall point
    out["xm430"] = {"datasheet": XM430, "joints": joints, "Kt_N_Nm_per_A_datasheet": Kt_n,
                    "note": "the PWM regression carries the integral and the derivative terms on the firmware's internal velocity, so its R2 is low and the gain is not the firmware's P; the current regression is the DC-motor law"}


def so101(out: dict) -> None:
    d = load("so101", "train")
    joints = []
    for j in range(1, 7):
        e = d[f"goal_pos{j}"] - d[f"pos{j}"]; v = d[f"vel{j}"].astype(float); ld = d[f"load{j}"].astype(float); cur = d[f"current{j}"].astype(float)
        c1, r1 = fit(np.stack([e, v, np.ones_like(e)], 1), ld)
        vu = np.unique(np.abs(v[v != 0]))
        joints.append({"joint": j, "load_per_rad": float(c1[0]), "load_fit_r2": r1, "velocity_step_steps_s": float(vu[0]) if vu.size else float("nan"), "load_max": float(np.abs(ld).max()),
                       "current_max_units": float(cur.max()), "error_std_rad": float(np.std(e))})
    out["sts3215"] = {"datasheet": STS3215, "joints": joints, "note": "the load register is the signed PWM duty in 0.1% steps; its regression on the position error is the firmware's P term (R2 0.97 to 0.999 on the arm joints), no velocity term is visible; the current register is in 6.5 mA units"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/m11/nad_fit.json")
    args = ap.parse_args()
    out = {}
    omx(out); so101(out)
    json.dump(out, open(args.out, "w"), indent=1)
    for j in out["xm430"]["joints"]:
        print(f"XM430 joint {j['joint']}: pwm/rad {j['pwm_per_rad']:.0f} (R2 {j['pwm_fit_r2']:.2f}); R {j['R_ohm']:.2f} ohm, KeN {j['KeN_Vs_per_rad']:.2f} V s/rad (R2 {j['current_fit_r2']:.2f}) -> stall {j['stall_current_A_at_12V']:.2f} A, no-load {j['no_load_rpm_at_12V']:.0f} rpm; velocity step {j['velocity_step_rad_s']:.4f} rad/s")
    for j in out["sts3215"]["joints"]:
        print(f"STS3215 joint {j['joint']}: load/rad {j['load_per_rad']:.0f} (R2 {j['load_fit_r2']:.3f}); velocity step {j['velocity_step_steps_s']:.0f} steps/s; load max {j['load_max']:.0f}")


if __name__ == "__main__":
    main()
