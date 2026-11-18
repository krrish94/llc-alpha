"""Clips of report 19 (Milestone 11): the pick with a grasp on the torque interface, the feedforward PD against the two-teacher
student (seed 0) on a case with the 2 kg tool payload where the PD misses the grasp and the student takes it (selection rule: the
lowest-index test case with a 2 kg payload among those; from runs/m11/metrics_tasks_p0.json), and the same family on a case
without payload. Episodes are simulated here into runs/m11/episodes_clips; the clips and the time-series JSON go to the report's
media and data directories, the HTML snippets to data/videos.html (read by experiments.m11_narrative).

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_videos.py [--workers 4]
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
from llc.eval.cases_v2 import make_case_v2  # noqa: E402
from llc.eval.parallel import CaseSpec, evaluate  # noqa: E402
from llc.sim.nominal import NominalModel  # noqa: E402

REPORT = Path("research/reports/19-m11-interfaces")
EPISODES = Path("runs/m11/episodes_clips")
STUDENT = "runs/m10r2/distill_two-s0"
LABELS = {"pd_feedforward": "PD, tracking-tuned, with feedforward", "student": "Two-teacher student (seed 0)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    pd = json.load(open("runs/m9/gains_shared.json"))["spec"]
    specs = {"pd_feedforward": {**pd, "kind": "pd_feedforward"}, "student": {"kind": "learned", "path": f"{STUDENT}/best.pt"}}
    data = json.load(open("runs/m11/metrics_tasks_p0.json"))
    nominal = NominalModel()
    pdm = {m["case_id"]: m for m in data["groups"]["tasks"]["pd_feedforward"] if m["regime"] == "pick"}
    stm = {m["case_id"]: m for m in data["groups"]["tasks"]["distill_two-s0"] if m["regime"] == "pick"}
    payload = {c: make_case_v2("pick", "test", int(c.split("-")[-1]), nominal).perturbation.payload_mass for c in pdm}
    converted = sorted(c for c in pdm if payload[c] == 2.0 and stm[c]["grasp_success"] and not pdm[c]["grasp_success"])
    plain = sorted(c for c in pdm if payload[c] == 0.0 and pdm[c]["grasp_success"])
    clips = [{"stem": "pick_payload", "case": converted[0], "title": "Pick with a grasp, 2 kg tool payload", "caption": f"Pick-and-place with a grasp, real time, case {converted[0]}, with the 2 kg tool payload of its instance: the lowest-index such case among the {len(converted)} of 16 where the feedforward PD misses the grasp and the student takes it. The gray sphere is the commanded tool point; the force plot is the tool-block contact force. The PD sags under the unmodelled payload (10 to 19 mm over the family) and the block's centre is outside the 6 mm by 9 mm tolerance at the grasp instant, so the weld never activates and the block stays; the student infers the payload from the history, grasps, lifts, carries and places."},
             {"stem": "pick_plain", "case": plain[0], "title": "Pick with a grasp, no payload", "caption": f"The same family without a tool payload, real time, case {plain[0]} (the lowest-index case without payload that the PD grasps): the PD arrives within 1 mm and grasps; the student arrives with 4 to 6 mm of lateral error, its pads brush the block during the descent (6 to 9 N on the block) and its contact law yields, so it stops 18 mm above the grasp height and never takes the block. The contact compliance that wins the blocked family loses a grasp that touches its object on the way in."}]
    media, datadir = REPORT / "media", REPORT / "data"
    media.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True); EPISODES.mkdir(parents=True, exist_ok=True)
    snippets = []
    for clip in clips:
        cs = CaseSpec("pick", "test", int(clip["case"].split("-")[-1]), protocol="v2")
        scene = make_case_v2("pick", "test", cs.index, nominal).meta["task"]["scene"]
        eps = []
        for name in specs:
            p = EPISODES / f"{clip['case']}__{name}.npz"
            if not p.exists():
                ep = evaluate({name: specs[name]}, [cs], keep_episodes=True, workers=1)[name][0]["episode"]
                arrays = {k: getattr(ep, k) for k in FIELDS}; arrays.update({k: getattr(ep, k) for k in TASK_FIELDS}); arrays["scene"] = np.array(scene)
                np.savez_compressed(p, **arrays)
            eps.append(("distill_two" if name == "student" else name, LABELS[name], str(p)))  # palette keys of research/design.json
        stem = clip["stem"]
        json.dump(timeseries_json(eps, ("pos_err", "f_object", "tau2")), open(datadir / f"ts_{stem}.json", "w"))
        rec = render_comparison(eps, str(media / f"{stem}.mp4"), f"{clip['title']}, case {clip['case']}", speed=1.0, camera="detail", plot_series=("pos_err", "f_object", "tau2"), poster_dir=str(media), scene=scene)
        ok, msg = video_check(str(media / f"{stem}.mp4"))
        print(msg, flush=True)
        if not ok:
            raise SystemExit(f"clip check failed: {stem}")
        poster = Path(rec["posters"]["middle"]).name
        snippets.append(video_figure(f"media/{stem}.mp4", f"media/{poster}", clip["caption"] + " The gray sphere is the commanded tool point; the metrics under each view accumulate causally; the plot cursor follows the video.", f"fig-video-{stem}", timeseries_json_relpath=f"data/ts_{stem}.json", speed=1.0))
    (datadir / "videos.html").write_text("\n".join(snippets))
    print("wrote", datadir / "videos.html")


if __name__ == "__main__":
    main()
