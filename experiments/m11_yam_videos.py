"""Clips of report 20 (the YAM through profile P1): from the representative episodes saved by the YAM evaluation
(runs/m11/yam/episodes_p1, the median case of the re-tuned PD per in-distribution family), the re-tuned feedforward PD
against the host-rate student, with the oracle in the contact clips. Snippets go to the report's data/videos.html.

    LLC_PLANT=yam PYTHONPATH=. uv run --no-sync python experiments/m11_yam_videos.py [--metrics runs/m11/yam/metrics_p1_student.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("research").resolve()))
from report_api import video_figure  # noqa: E402
from video import render_comparison  # noqa: E402
from videocheck import POSTER_MIN_DIFF, check as video_check  # noqa: E402

from experiments.handoff_videos import timeseries_json  # noqa: E402

REPORT = Path("research/reports/20-m11-yam")
LABELS = {"pd_feedforward": "Re-tuned PD with feedforward", "pd_shipped": "Shipped gains", "oracle_pd": "Oracle (privileged ceiling)", "student_p1-s0": "Host-rate student (seed 0)"}
PALETTE = {"student_p1-s0": "distill_two", "pd_shipped": "pd_initial"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/m11/yam/metrics_p1_student.json")
    ap.add_argument("--episodes", default="runs/m11/yam/episodes_p1")
    args = ap.parse_args()
    data = json.load(open(args.metrics))
    rep = data["representative_cases"]
    eps_dir = Path(args.episodes)
    media, datadir = REPORT / "media", REPORT / "data"
    media.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True)
    med = "the median case of the re-tuned PD's composite"
    clips = [
        {"stem": "free", "regime": "free", "methods": ["pd_shipped", "pd_feedforward", "student_p1-s0"], "speed": 1.0, "camera": "side", "series": ("pos_err", "tau2", "tool_z"), "title": "Free joint-space motion on the YAM",
         "caption": f"Free motion on the YAM through its 250 Hz impedance interface ({med}), real time: the shipped gains, the re-tuned feedforward PD and the host-rate student."},
        {"stem": "contact2", "regime": "contact2", "methods": ["pd_feedforward", "oracle_pd", "student_p1-s0"], "speed": 0.25, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"), "title": "Press and slide on the YAM",
         "caption": f"Press and slide ({med}), quarter speed: the reference asks for 0 to 6 mm of depth (5 to 8 N under the YAM's press law); the force plot shows what each controller presses with against the 12 N cap."},
        {"stem": "blocked", "regime": "blocked", "methods": ["pd_feedforward", "oracle_pd", "student_p1-s0"], "speed": 0.25, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"), "title": "Blocked target on the YAM",
         "caption": f"Blocked target ({med}), quarter speed: the commanded path passes through the surface; the oracle and the student are meant to stop at the surface and wait at a bounded force."},
    ]
    snippets = []
    for clip in clips:
        case = rep.get(clip["regime"])
        if case is None:
            continue
        eps = [(PALETTE.get(n, n.split("-s")[0]), LABELS[n], str(eps_dir / f"{case}__{n}.npz")) for n in clip["methods"] if (eps_dir / f"{case}__{n}.npz").exists()]
        if len(eps) < 2:
            print("skipping", clip["stem"], "missing episodes"); continue
        stem = f"{clip['stem']}_{int(clip['speed'] * 100):03d}x"
        json.dump(timeseries_json(eps, clip["series"]), open(datadir / f"ts_{clip['stem']}.json", "w"))
        rec = render_comparison(eps, str(media / f"{stem}.mp4"), f"{clip['title']}, case {case}", speed=clip["speed"], camera=clip["camera"], plot_series=clip["series"], poster_dir=str(media))
        ok, msg = video_check(str(media / f"{stem}.mp4"), min_poster_diff=POSTER_MIN_DIFF)
        print(msg, flush=True)
        if not ok:
            raise SystemExit(f"clip check failed: {stem}")
        poster = Path(rec["posters"]["middle"]).name
        snippets.append(video_figure(f"media/{stem}.mp4", f"media/{poster}", clip["caption"] + " The gray sphere is the commanded tool point; the metrics under each view accumulate causally; the plot cursor follows the video.", f"fig-video-{clip['stem']}", timeseries_json_relpath=f"data/ts_{clip['stem']}.json", speed=clip["speed"]))
    (datadir / "videos.html").write_text("\n".join(snippets))
    print("wrote", datadir / "videos.html")


if __name__ == "__main__":
    main()
