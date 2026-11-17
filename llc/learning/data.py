"""Training references: a cached pool of references per regime (split 'train') and batch windows.

Windows are ``horizon_s`` long at the model step ``dt``. For hold, step, contact and transition
references the window starts at t = 0; for smooth and zoh references it starts at a random time.
References shorter than the window are padded by holding their last sample. The initial state is
the reference at the window start (zero velocity for held references) plus a small random offset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from llc.eval.cases import REGIMES, make_case
from llc.sim.nominal import NominalModel

CACHE_DIR = Path("runs/cache")


def _resample(ref_t: np.ndarray, x: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_new, ref_t, x[:, j]) for j in range(x.shape[1])], 1)


def reference_pool(n_per_regime: int, dt: float, split: str = "train", regimes=REGIMES, nominal: NominalModel | None = None) -> dict[str, list[dict]]:
    """{regime: [{"q": (T,7), "qd": (T,7), "regime", "index"}]} at the model step; cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"refpool_{split}_n{n_per_regime}_dt{dt:g}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["pool"].item()
    nominal = nominal or NominalModel()
    pool: dict[str, list[dict]] = {}
    for regime in regimes:
        pool[regime] = []
        for i in range(n_per_regime):
            case = make_case(regime, split, i, nominal)
            ref = case.reference
            t_new = np.arange(0.0, ref.duration + 1e-9, dt)
            pool[regime].append({"q": _resample(ref.t, ref.q, t_new).astype(np.float32),
                                 "qd": _resample(ref.t, ref.qd, t_new).astype(np.float32),
                                 "regime": regime, "index": i, "meta": {k: v for k, v in case.meta.items() if k != "tool_ref"}})
    np.savez(path, pool=np.array(pool, dtype=object))
    return pool


def sample_batch(pool: dict[str, list[dict]], weights: dict[str, float], batch: int, horizon_steps: int,
                 dt: float, rng: np.random.Generator, init_noise_q: float = 0.02, pulse_prob: float = 0.0):
    """Returns q_ref (B,T,7), qd_ref (B,T,7), q0 (B,7), qd0 (B,7), regimes (list) and, if
    ``pulse_prob`` > 0, f_ext (B,T,3): a 10 N force pulse of 0.2 s in a random direction at a random
    time in a fraction ``pulse_prob`` of the windows (protocol v2; matches the hold regime's pulse)."""
    regimes = list(weights)
    p = np.array([weights[r] for r in regimes], float)
    p /= p.sum()
    q_ref = np.zeros((batch, horizon_steps, 7), np.float32)
    qd_ref = np.zeros((batch, horizon_steps, 7), np.float32)
    q0 = np.zeros((batch, 7), np.float32)
    qd0 = np.zeros((batch, 7), np.float32)
    labels = []
    for b in range(batch):
        regime = regimes[rng.choice(len(regimes), p=p)]
        ref = pool[regime][rng.integers(len(pool[regime]))]
        n = len(ref["q"])
        if regime in ("smooth", "zoh") and n > horizon_steps:
            k0 = int(rng.integers(0, n - horizon_steps))
            if regime == "zoh":  # align to the 20 Hz hold grid so the window starts at a held target
                grid = int(round(0.05 / dt))
                k0 -= k0 % grid
        else:
            k0 = 0
        idx = np.minimum(np.arange(k0, k0 + horizon_steps), n - 1)
        q_ref[b], qd_ref[b] = ref["q"][idx], ref["qd"][idx]
        q0[b] = ref["q"][k0] + rng.normal(0.0, init_noise_q, 7)
        qd0[b] = ref["qd"][k0] if regime in ("smooth",) else 0.0
        labels.append(regime)
    if pulse_prob <= 0.0:
        return q_ref, qd_ref, q0, qd0, labels
    f_ext = np.zeros((batch, horizon_steps, 3), np.float32)
    n_pulse = int(round(0.2 / dt))
    for b in range(batch):
        if rng.uniform() < pulse_prob:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            k0 = int(rng.integers(int(0.2 / dt), max(int(0.2 / dt) + 1, horizon_steps - n_pulse)))
            f_ext[b, k0:k0 + n_pulse] = 10.0 * direction
    return q_ref, qd_ref, q0, qd0, labels, f_ext


# ----------------------------------------------------------------------------- protocol v2 (deep dive)
def reference_pool_v2(n_per_regime: int, split: str = "train", regimes=("hold", "free", "policy", "step", "contact2", "blocked"),
                      nominal: NominalModel | None = None) -> dict[str, list[dict]]:
    """Training references of the deep dive at the physics step (1 kHz), with their phase lists so
    that windows can start at phase boundaries; cached on disk."""
    from llc.eval.cases_v2 import make_case_v2

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    default_regimes = ("hold", "free", "policy", "step", "contact2", "blocked")
    tag = "" if tuple(regimes) == default_regimes else "_" + "-".join(regimes)
    from llc.sim.plant import PLANT

    path = CACHE_DIR / f"refpool_v4_{split}_n{n_per_regime}{tag}{PLANT.cache_tag}.npz"  # v4: corrected blocked phases and wall geometry, tool reference kept (ADR 0007); plant tag (ADR 0009)
    if path.exists():
        return np.load(path, allow_pickle=True)["pool"].item()
    nominal = nominal or NominalModel()
    pool: dict[str, list[dict]] = {}
    for regime in regimes:
        pool[regime] = []
        for i in range(n_per_regime):
            case = make_case_v2(regime, split, i, nominal)
            ref = case.reference
            meta = {k: v for k, v in case.meta.items() if k not in ("tool_ref",)}
            if regime in ("contact2", "blocked", "task"):  # tool reference of the surface families (privileged reward terms)
                meta["tool_ref"] = np.asarray(case.meta["tool_ref"], np.float32)
            pool[regime].append({"q": ref.q.astype(np.float32), "qd": ref.qd.astype(np.float32), "regime": regime, "index": i,
                                 "phases": [(float(a), float(b), l) for a, b, l in ref.phases], "meta": meta})
    np.savez(path, pool=np.array(pool, dtype=object))
    return pool


SURFACE_FAMILIES = ("contact2", "blocked", "task")


def surface_depth(tool_path: np.ndarray, surface: str) -> np.ndarray:
    """Signed depth (m) of a tool path into the table top or the wall face, positive inside."""
    from llc.references import TABLE_TOP_Z, TOOL_RADIUS, WALL_FACE_X

    if surface == "table":
        return (TABLE_TOP_Z + TOOL_RADIUS) - tool_path[:, 2]
    return tool_path[:, 0] - (WALL_FACE_X - TOOL_RADIUS)


def sample_batch_v2(pool: dict[str, list[dict]], weights: dict[str, float], batch: int, horizon_steps: int, dt: float,
                    rng: np.random.Generator, init_noise_q: float = 0.02, pulse_prob: float = 0.3, start_mode: str = "contact"):
    """Windows for the deep dive: free/policy windows start at a random time, surface-family windows
    (contact2, blocked, task) start 0.3 to 0.8 s before the nominal contact time (so the approach,
    onset and pressing fit in the window), hold and step windows start at t = 0. ``start_mode``
    "mixed" (ADR 0007) starts half of the surface windows 0.1 to 0.2 s before the reference crosses
    the surface instead, with a tenth of the start noise (the robot arrives with the reference and
    must not start inside the surface), so that the window is spent on the block and, when long
    enough, on the release and the recovery.
    Returns q_ref, qd_ref, q0, qd0, labels, f_ext and the privileged window data ``priv``:
    tool_ref (B,T,3), normal (B,3), depth_ref (B,T; signed depth of the reference into the surface,
    positive inside, 0 without a surface) and has_surface (B,)."""
    regimes = list(weights)
    p = np.array([weights[r] for r in regimes], float)
    p /= p.sum()
    q_ref = np.zeros((batch, horizon_steps, 7), np.float32)
    qd_ref = np.zeros((batch, horizon_steps, 7), np.float32)
    q0 = np.zeros((batch, 7), np.float32)
    qd0 = np.zeros((batch, 7), np.float32)
    f_ext = np.zeros((batch, horizon_steps, 3), np.float32)
    priv = {"tool_ref": np.zeros((batch, horizon_steps, 3), np.float32), "normal": np.zeros((batch, 3), np.float32),
            "depth_ref": np.zeros((batch, horizon_steps), np.float32), "has_surface": np.zeros(batch, bool)}
    labels = []
    n_pulse = int(round(0.2 / dt))
    for b in range(batch):
        regime = regimes[rng.choice(len(regimes), p=p)]
        ref = pool[regime][rng.integers(len(pool[regime]))]
        n = len(ref["q"])
        noise_scale = 1.0
        if regime in ("free", "policy") and n > horizon_steps:
            k0 = int(rng.integers(0, n - horizon_steps))
            if regime == "policy":
                k0 -= k0 % 20
        elif regime in SURFACE_FAMILIES:
            k_c = int(round(ref["meta"].get("t_contact_nominal", 0.5) / dt))
            k0 = int(max(0, k_c - rng.integers(int(0.3 / dt), int(0.8 / dt))))
            if start_mode == "mixed" and "tool_ref" in ref["meta"] and rng.uniform() < 0.5:
                inside = surface_depth(ref["meta"]["tool_ref"], ref["meta"]["surface"]) > 0.0
                if inside.any():
                    k_in = int(np.where(inside)[0][0])
                    k0 = int(max(0, k_in - rng.integers(int(0.1 / dt), int(0.2 / dt))))
                    noise_scale = 0.1
        else:
            k0 = 0
        idx = np.minimum(np.arange(k0, k0 + horizon_steps), n - 1)
        q_ref[b], qd_ref[b] = ref["q"][idx], ref["qd"][idx]
        if regime in SURFACE_FAMILIES and "tool_ref" in ref["meta"]:
            tool = ref["meta"]["tool_ref"]
            priv["tool_ref"][b] = tool[idx]
            priv["normal"][b] = np.asarray(ref["meta"]["normal"], np.float32)
            priv["depth_ref"][b] = surface_depth(tool[idx], ref["meta"]["surface"])
            priv["has_surface"][b] = True
        q0[b] = ref["q"][k0] + rng.normal(0.0, noise_scale * init_noise_q, 7)
        qd0[b] = ref["qd"][k0] if regime in ("free", "contact2", "blocked", "task") else 0.0
        labels.append(regime)
        if regime in ("hold", "free") and rng.uniform() < pulse_prob:
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            kp = int(rng.integers(int(0.2 / dt), max(int(0.2 / dt) + 1, horizon_steps - n_pulse)))
            f_ext[b, kp:kp + n_pulse] = 10.0 * d
    return q_ref, qd_ref, q0, qd0, labels, f_ext, priv
