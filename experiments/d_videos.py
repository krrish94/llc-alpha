"""Render the comparison clips of a deep-dive report from the representative episodes stored by
experiments/d_evaluate.py (--save-episodes): one clip per in-distribution regime at real time and
one slow-motion clip for the contact regimes, into research/reports/<id>/media/, plus the
synchronized time-series JSON for the notebook's video widget. Prints the HTML snippets to paste
through the narrative module (VIDEOS list) or returns them when imported."""
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

REGIME_LABELS = {"hold": "Hold", "free": "Free joint-space motion", "policy": "Policy-like targets", "step": "Step targets",
                 "contact2": "Contact", "blocked": "Blocked reference", "task": "Task sequence"}


def timeseries_json(episodes: list[tuple[str, str, str]], stride: int = 5) -> dict:
    eps = {key: dict(np.load(path)) for key, _, path in episodes}
    ref = next(iter(eps.values()))
    t = ref["t"][::stride]
    series = [{"key": "pos_err", "label": "Max joint tracking error", "unit": "rad"}, {"key": "f_normal", "label": "Normal contact force", "unit": "N"},
              {"key": "tau2", "label": "Joint 2 torque", "unit": "Nm"}, {"key": "tool_z", "label": "Tool height", "unit": "m"}]
    values = {s["key"]: {} for s in series}
    for key, ep in eps.items():
        n = len(ep["t"])
        values["pos_err"][key] = np.abs(ep["q_ref"][:n] - ep["q"]).max(1)[::stride].round(5).tolist()
        values["f_normal"][key] = ep["f_normal"][::stride].round(2).tolist()
        values["tau2"][key] = ep["tau"][::stride, 1].round(3).tolist()
        values["tool_z"][key] = ep["tool_pos"][::stride, 2].round(5).tolist()
    return {"t": t.round(4).tolist(), "series": series, "methods": [{"key": k, "label": l} for k, l, _ in episodes], "values": values}


CAMERA_BY_REGIME = {"contact2": "detail", "blocked": "detail", "task": "detail"}  # free-space families use the wider side camera


def render_report_videos(report_id: str, metrics_path: str, methods: list[tuple[str, str]], episodes_dir: str, camera: str = "auto",
                         slow_regimes=("contact2", "blocked", "task")) -> list[str]:
    data = json.load(open(metrics_path))
    rep = data.get("representative_cases", {})
    report = Path("research/reports") / report_id
    media, datadir = report / "media", report / "data"
    media.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True)
    snippets = []
    for regime, case_id in rep.items():
        # the palette key is the method name without its seed suffix (colors are per method, not per seed)
        eps = [(k.split("-s")[0], l, str(Path(episodes_dir) / f"{case_id}__{k}.npz")) for k, l in methods if (Path(episodes_dir) / f"{case_id}__{k}.npz").exists()]
        if not eps:
            continue
        ts = timeseries_json(eps)
        json.dump(ts, open(datadir / f"ts_{regime}.json", "w"))
        for speed in ((1.0, 0.25) if regime in slow_regimes else (1.0,)):
            stem = f"{regime}_{int(speed * 100):03d}x"
            cam = CAMERA_BY_REGIME.get(regime, "side") if camera == "auto" else camera
            rec = render_comparison(eps, str(media / f"{stem}.mp4"), f"{REGIME_LABELS.get(regime, regime)}, case {case_id}", speed=speed, camera=cam, poster_dir=str(media))
            video_check(str(media / f"{stem}.mp4"))
            posters = rec.get("posters", {})
            poster = Path(posters["middle"]).name if isinstance(posters, dict) and "middle" in posters else f"{stem}-middle.png"
            snippets.append(video_figure(f"media/{stem}.mp4", f"media/{poster}", f"{REGIME_LABELS.get(regime, regime)}: representative case {case_id} (median composite of the shared PD) at {speed:g}x; the gray sphere is the commanded tool position, metrics accumulate causally; the plot cursor follows the video.",
                                         f"fig-video-{stem}", timeseries_json_relpath=f"data/ts_{regime}.json", speed=speed))
    return snippets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--methods", nargs="+", required=True, help="key=label pairs")
    ap.add_argument("--camera", default="auto")
    args = ap.parse_args()
    methods = [tuple(m.split("=", 1)) for m in args.methods]
    snippets = render_report_videos(args.report, args.metrics, methods, args.episodes, camera=args.camera)
    out = Path("research/reports") / args.report / "data" / "videos.html"
    out.write_text("\n".join(snippets))
    print(f"wrote {len(snippets)} video figures to {out}")


if __name__ == "__main__":
    main()
