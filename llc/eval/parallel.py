"""Parallel evaluation of controller specifications on cases, one process per CPU core.

Workers build their own simulator and nominal model once. Cases are passed by their
(regime, split, index, perturbation, speed_scale) description and rebuilt in the worker so
that no large arrays cross process boundaries.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from llc.controllers.factory import build_controller
from llc.eval.cases import Case, make_case
from llc.eval.metrics import compute_metrics
from llc.eval.rollout import Episode, rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import DynamicsPerturbation, PandaSim


@dataclass
class CaseSpec:
    regime: str
    split: str
    index: int
    perturbation: DynamicsPerturbation = field(default_factory=DynamicsPerturbation)
    speed_scale: float = 1.0
    protocol: str = "v1"  # "v2": cases from llc.eval.cases_v2 (instances, payloads and noise come with the case)
    heldout_workspace: bool = False
    ood_instance: bool = False
    payload: float | None = None
    surface_stiffness: float | None = None  # N/m effective; compliant table and wall (ADR 0008)

    @property
    def id(self) -> str:
        if self.protocol == "v2":
            return f"{self.regime}-{self.split}{'-ho' if self.heldout_workspace else ''}{'-ood' if self.ood_instance else ''}{'' if self.speed_scale == 1.0 else f'-x{self.speed_scale:g}'}{'' if self.surface_stiffness is None else f'-k{self.surface_stiffness:g}'}-{self.index:03d}"
        return f"{self.regime}-{self.split}-{self.index:03d}"


_worker: dict = {}


def _init_worker() -> None:
    _worker["nominal"] = NominalModel()
    _worker["sims"] = {}
    _worker["cases"] = {}


def _get_sim(perturbation: DynamicsPerturbation, seed: int, scene: str | None = None) -> PandaSim:
    key = (tuple(sorted(asdict(perturbation).items())), seed, scene)
    if key not in _worker["sims"]:
        _worker["sims"] = {}  # keep at most one simulator per worker
        _worker["sims"][key] = PandaSim(perturbation, seed=seed, scene=scene)
    return _worker["sims"][key]


def _get_case(spec: CaseSpec) -> Case:
    key = (spec.protocol, spec.regime, spec.split, spec.index, spec.speed_scale, spec.heldout_workspace, spec.ood_instance, spec.payload, spec.surface_stiffness)
    if key not in _worker["cases"]:
        if len(_worker["cases"]) > 64:
            _worker["cases"].clear()
        if spec.protocol == "v2":
            from llc.eval.cases_v2 import make_case_v2

            _worker["cases"][key] = make_case_v2(spec.regime, spec.split, spec.index, _worker["nominal"], speed_scale=spec.speed_scale,
                                                 heldout_workspace=spec.heldout_workspace, payload=spec.payload, ood_instance=spec.ood_instance, surface_stiffness=spec.surface_stiffness)
        else:
            _worker["cases"][key] = make_case(spec.regime, spec.split, spec.index, _worker["nominal"], speed_scale=spec.speed_scale)
    case = _worker["cases"][key]
    if spec.protocol != "v2":
        case.perturbation = spec.perturbation
    return case


def _run_one(args) -> tuple[dict, Episode | None]:
    controller_spec, case_spec, keep_episode = args
    if not _worker:
        _init_worker()
    case = _get_case(case_spec)
    # The simulator takes the case's own perturbation: for protocol v2 the robot instance, payload and
    # sensor noise are part of the case (set by make_case_v2); for v1 _get_case copies the spec's.
    sim = _get_sim(case.perturbation, seed=case.seed, scene=(case.meta.get("task") or {}).get("scene"))
    controller = build_controller(controller_spec, _worker["nominal"], case)
    ep = rollout(sim, controller, case)
    metrics = compute_metrics(ep, case)
    metrics["case_id"] = case_spec.id
    metrics["regime"] = case_spec.regime
    return metrics, (ep if keep_episode else None)


def evaluate(controller_specs: dict[str, dict], case_specs: list[CaseSpec], keep_episodes: bool = False,
             workers: int | None = None) -> dict[str, list[dict]]:
    """Run every controller on every case. Returns {controller name: [metrics per case]} in case order;
    with ``keep_episodes`` each metrics dict also carries the Episode under 'episode'."""
    jobs = [(spec, cs, keep_episodes) for spec in controller_specs.values() for cs in case_specs]
    names = [name for name in controller_specs for _ in case_specs]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) == 1:
        results = [_run_one(j) for j in jobs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(_run_one, jobs, chunksize=1)
    out: dict[str, list[dict]] = {name: [] for name in controller_specs}
    for name, (metrics, ep) in zip(names, results):
        if ep is not None:
            metrics["episode"] = ep
        out[name].append(metrics)
    return out


def mean_composite(results: list[dict]) -> float:
    return float(np.mean([r["composite"] for r in results]))


def validate_policy(ckpt_path, n_free: int, n_contact: int, objective: str = "composite", workers: int | None = None) -> dict:
    """Validation of a learned checkpoint on the protocol v2 'val' split: ``n_free`` cases per free-space
    family, ``n_contact`` per surface family; the mean over families of the per-family mean of
    ``objective`` ("composite" or "composite_cap"), the force statistics and the cap/reflex counts."""
    from llc.eval.cases_v2 import REGIMES_V2
    from llc.eval.metrics import FORCE_CAP, REFLEX_SUSTAINED

    surface = ("contact2", "blocked", "task")
    cases = [CaseSpec(r, "val", i, protocol="v2") for r in REGIMES_V2 for i in range(n_contact if r in surface else n_free)]
    res = evaluate({"c": {"kind": "learned", "path": str(ckpt_path)}}, cases, workers=workers or int(os.environ.get("LLC_VAL_WORKERS", "0")) or None)["c"]
    key = "composite_cap" if objective == "free" else objective
    per = {r: float(np.mean([m[key] for m in res if m["regime"] == r])) for r in REGIMES_V2}
    selected = [r for r in REGIMES_V2 if r in ("hold", "free", "policy", "step")] if objective == "free" else list(REGIMES_V2)  # "free": selection on the free-space families (ADR 0010 round 2)
    fin = lambda x: x == x
    return {"mean": float(np.mean([per[r] for r in selected])), "objective": objective, "per_regime": per, "diverged": int(sum(m["diverged"] for m in res)),
            "excess_force": float(np.mean([m["excess_force"] for m in res])), "excess_force_cap": float(np.mean([m["excess_force_cap"] for m in res])),
            "f_peak_max": float(np.nanmax([m["f_peak"] for m in res])),
            "over_reflex": int(sum(1 for m in res if fin(m["f_peak"]) and m["f_peak"] > 40.0)),
            "over_reflex_20ms": int(sum(1 for m in res if fin(m["f_peak_20ms"]) and m["f_peak_20ms"] > 40.0)),
            "over_cap_20ms": int(sum(1 for m in res if fin(m["f_peak_20ms"]) and m["f_peak_20ms"] > REFLEX_SUSTAINED)),
            "press_force": float(np.nanmean([m["f_mean"] for m in res if m["regime"] in surface])),
            "contact_loss_rate": float(np.nanmean([m["contact_loss_rate"] for m in res if m["regime"] in surface])),
            "n_cases": len(cases), "cap": FORCE_CAP}
