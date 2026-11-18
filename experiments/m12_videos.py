"""Clips of report 23 (Milestone 12): the five protocol v5 families, one representative test case each (the median case of the
tracking-tuned PD from the evaluation), the feedforward PD, the per-family PD, the oracle and the 4-layer student side by side.
Episodes are simulated here into runs/m12/episodes_clips; the clips and the time-series JSON go to the report's media and data.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m12_videos.py [--student runs/m12/student_l4-s0/best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import video_figure  # noqa: E402
from video import render_comparison  # noqa: E402
from videocheck import check as video_check  # noqa: E402

from experiments.handoff_videos import FIELDS, TASK_FIELDS, timeseries_json  # noqa: E402
from llc.eval.cases_v2 import REGIMES_V2, TASK_REGIMES, make_case_v2  # noqa: E402
from llc.eval.parallel import CaseSpec, evaluate  # noqa: E402
from llc.sim.nominal import NominalModel  # noqa: E402

REPORT = Path("research/reports/23-m12-consolidation")
EPISODES = Path("runs/m12/episodes_clips")
TITLES = {"wipe": "Wiping under the force band", "erase": "Erasing with a held block", "peg": "Peg-in-hole with a chamfer", "drawer": "Drawer pushed to its stop", "door": "Door pushed closed"}
FORCE_OF = {"wipe": "f_normal", "erase": "f_object_env", "peg": "f_object_env", "drawer": "f_object", "door": "f_object"}
CAPTIONS = {"wipe": "a raster of three passes under the 7 to 12 N band; the force plot is the tool-table (or tool-wall) contact force",
            "erase": "the pick's block held from the start and pressed on the table while sliding; the force plot is the block-table contact force",
            "peg": "the hanging square peg aimed 2 to 6 mm beside the chamfered hole; the force plot is the peg's contact force with the fixture and the table; the oracle's lateral spring is weak while the peg is blocked, so the chamfer guides it in",
            "drawer": "the front pushed closed along a reference that drifts sideways and continues 2 cm past the stop; the force plot is the tool-panel contact force",
            "door": "the panel pushed closed along a straight line while the contact slides on it; the force plot is the tool-panel contact force"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/m12/metrics_test.json")
    ap.add_argument("--student", default="runs/m12/student_l4-s0/best.pt")
    ap.add_argument("--student-key", default="distill_consol_l4")
    ap.add_argument("--families", nargs="*", default=["wipe", "erase", "peg", "drawer", "door"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data = json.load(open(args.metrics))
    pd = json.load(open("runs/m9/gains_shared.json"))["spec"]
    fam = {}
    for r in list(REGIMES_V2) + list(TASK_REGIMES):
        p = Path("runs/m12") / f"tune_{r}" / f"gains_{r}.json"
        if p.exists():
            fam[r] = json.load(open(p))["spec"]
    fam["shared"] = {**pd, "kind": "pd_feedforward"}
    specs = {"pd_feedforward": {**pd, "kind": "pd_feedforward"}, "pd_family": {"kind": "pd_per_family", "gains": fam}, "oracle_pd": {"kind": "oracle", "base": {**pd, "kind": "pd_feedforward"}},
             args.student_key: {"kind": "learned", "path": args.student}}
    labels = {"pd_feedforward": "PD, tracking-tuned, with feedforward", "pd_family": "Per-family PD", "oracle_pd": "Oracle (privileged ceiling)", args.student_key: "Student, 4 layers, at scale"}
    nominal = NominalModel()
    media, datadir = REPORT / "media", REPORT / "data"
    media.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True); EPISODES.mkdir(parents=True, exist_ok=True)
    reps = data.get("representative_cases", {})
    snippets = []
    for r in args.families:
        case_id = reps.get(r)
        if case_id is None:
            print("no representative case for", r); continue
        index = int(case_id.split("-")[-1])
        cs = CaseSpec(r, "test", index, protocol="v2")
        case = make_case_v2(r, "test", index, nominal)
        task = case.meta.get("task") or {}
        scene = task.get("scene")
        eps = []
        for name, spec in specs.items():
            p = EPISODES / f"{case_id}__{name}.npz"
            if not p.exists():
                ep = evaluate({name: spec}, [cs], keep_episodes=True, workers=1)[name][0]["episode"]
                arrays = {k: getattr(ep, k) for k in FIELDS}
                if scene:
                    arrays.update({k: getattr(ep, k) for k in TASK_FIELDS}); arrays["scene"] = np.array(scene); arrays["task_json"] = np.array(json.dumps({k: v for k, v in task.items() if k != "scene"}))
                np.savez_compressed(p, **arrays)
            eps.append((name, labels[name], str(p)))
        series = ("pos_err", FORCE_OF[r], "tau2")
        json.dump(timeseries_json(eps, series), open(datadir / f"ts_{r}.json", "w"))
        rec = render_comparison(eps, str(media / f"{r}.mp4"), f"{TITLES[r]}, case {case_id}", speed=1.0, camera="detail", plot_series=series, poster_dir=str(media), scene=scene)
        ok, msg = video_check(str(media / f"{r}.mp4"))
        print(msg, flush=True)
        if not ok:
            raise SystemExit(f"clip check failed: {r}")
        poster = Path(rec["posters"]["middle"]).name
        snippets.append(video_figure(f"media/{r}.mp4", f"media/{poster}", f"{TITLES[r]}, real time, case {case_id} (the median case of the tracking-tuned PD): {CAPTIONS[r]}. The gray sphere is the commanded tool point; the metrics under each view accumulate causally; the plot cursor follows the video.", f"fig-video-{r}"))
    (datadir / "videos.html").write_text("\n".join(snippets))
    print("wrote", datadir / "videos.html")


if __name__ == "__main__":
    main()
