"""Comparison videos: synchronized replays of stored episodes side by side.

    PYTHONPATH=. uv run --no-sync python research/video.py --case transition-test-012 \
        --methods pd_shared pd_tuned computed_torque \
        --labels "PD, shared gains" "PD, per-regime gains" "Computed torque (privileged)" \
        --title "Contact transition, case transition-test-012" --camera side --speed 1 \
        --out media/transition-1x.mp4 --poster-dir media

Layout (docs/design.md, "Video"): one title line with the simulation clock and the playback
speed; method labels above equal-sized views from one fixed scene camera; concise causal metrics
below each view; a synchronized plot strip beneath (the full episode drawn once, a cursor at the
displayed time). Poses are the stored q samples replayed unscaled through mj_forward. The scene
is the saved MJCF; a flat light-gray sky is added in memory for rendering only. The commanded
tool position (q_ref through the nominal model) is a gray ghost sphere in every view; the
measured tool path of the last 0.5 s is a trace of small spheres fading from gray to the
method color. Colors and fonts come from research/design.json and research/plotstyle.py.

render_comparison() writes the mp4, optional first/middle/last posters and a JSON clip record
(<stem>.json next to the mp4) that research/videocheck.py verifies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import FONT_FAMILY, apply_style, design, method_color, reference_color  # noqa: E402

from llc.sim.nominal import NominalModel  # noqa: E402
from llc.sim.panda import SCENE_PATH  # noqa: E402

# Series available for the plot strip: key -> (label, unit, function of the episode arrays).
SERIES = {
    "pos_err": ("Max joint tracking error", "rad", lambda ep: np.abs(ep["q_ref"] - ep["q"]).max(axis=1)),
    "f_normal": ("Normal contact force", "N", lambda ep: ep["f_normal"]),
    "f_tangent": ("Tangential contact force", "N", lambda ep: ep["f_tangent"]),
    "tau2": ("Joint 2 torque", "Nm", lambda ep: ep["tau"][:, 1]),
    "tool_z": ("Tool height", "m", lambda ep: ep["tool_pos"][:, 2]),
    "f_object": ("Tool-object force", "N", lambda ep: ep["f_object"]),  # task scenes (Milestone 9)
    "f_object_env": ("Object-table force", "N", lambda ep: ep["f_object_env"]),
}

MARGIN, GAP = 24, 16            # frame margin and gap between views (px)
LABEL_H, METRICS_H = 28, 30     # method-label row above and metrics row below the views
PLOT_ROW_H = 230                # plot strip height per row of panels
PLOT_MIN_PANEL_W = 300          # narrower than this, panels stack vertically
GHOST_RADIUS = 0.025            # m; a little larger than the 0.02 m tool tip so it reads as a shell
GHOST_ALPHA = 0.5
TRACE_WINDOW_S, TRACE_SPACING_S, TRACE_RADIUS = 0.5, 0.02, 0.007
ENCODER_PARAMS = ["-crf", "20", "-preset", "medium", "-movflags", "+faststart"]
NOTE = ("Gray sphere: commanded tool position (q_ref through the nominal model) · "
        "trace: measured tool path, last 0.5 s · metrics accumulate from t = 0 to the displayed time")


# ---------- colors and fonts ----------


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb01(hex_color: str) -> np.ndarray:
    return np.array(_rgb(hex_color), dtype=np.float64) / 255.0


@lru_cache(maxsize=None)
def _font(size: int, weight: str = "normal") -> ImageFont.FreeTypeFont:
    """The plot font (research/plotstyle.FONT_FAMILY) for PIL; picks the face by style inside a .ttc."""
    path = fm.findfont(fm.FontProperties(family=FONT_FAMILY, weight=weight), fallback_to_default=True)
    wanted = {"normal": "Regular", "medium": "Medium", "bold": "Bold"}[weight]
    first = None
    for index in range(32):
        try:
            font = ImageFont.truetype(path, size, index=index)
        except OSError:
            break
        if font.getname()[1] == wanted:
            return font
        first = first or font
    if first is None:
        raise RuntimeError(f"cannot load font {path}")
    return first


def _fit(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> str:
    if font.getlength(text) <= max_w:
        return text
    while text and font.getlength(text + "…") > max_w:
        text = text[:-1]
    return text.rstrip() + "…"


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines, line = [], ""
    for word in text.split(" "):
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


# ---------- episodes ----------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_episodes(episodes: list[tuple[str, str, str]]) -> list[tuple[str, str, Path, dict]]:
    if not 1 <= len(episodes) <= 4:
        raise ValueError(f"one to four episodes fit the layout, got {len(episodes)}")
    loaded = []
    for key, label, path in episodes:
        method_color(key)  # raises for unknown method keys
        with np.load(path) as npz:
            ep = {name: npz[name] for name in npz.files}
        loaded.append((key, label, Path(path), ep))
    t0, q_ref0, path0 = loaded[0][3]["t"], loaded[0][3]["q_ref"], loaded[0][2]
    for _, _, path, ep in loaded[1:]:
        if ep["t"].shape != t0.shape or not np.allclose(ep["t"], t0) or not np.allclose(ep["q_ref"], q_ref0):
            raise ValueError(f"{path} is not the same case as {path0}: t or q_ref differ")
    return loaded


# ---------- scene ----------


def _render_model(view_w: int, view_h: int, scene_path: str | Path | None = None) -> mujoco.MjModel:
    """The saved scene (or a task scene with its free objects) with a flat sky in the design's separator gray, added in memory for rendering only."""
    spec = mujoco.MjSpec.from_file(str(scene_path or SCENE_PATH))
    sky = _rgb01(design()["separator"]).tolist()
    spec.add_texture(name="video_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=sky, rgb2=sky, width=32, height=32)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, view_w)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, view_h)
    return spec.compile()


def _repo_relative(path) -> str:
    """Paths in clip records are relative to the repository root (never the machine's home directory)."""
    root = Path(__file__).resolve().parents[1]
    p = Path(path).resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(path)


_IDENTITY = np.eye(3).flatten()


def _add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]),
                        np.asarray(pos, dtype=np.float64), _IDENTITY, np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def _add_trace(scene: mujoco.MjvScene, tool_pos: np.ndarray, t: np.ndarray, i: int, color: np.ndarray, gray: np.ndarray) -> None:
    """Measured tool positions over the last TRACE_WINDOW_S, oldest gray, newest in the method color."""
    dt = float(t[1] - t[0])
    spacing = max(1, int(round(TRACE_SPACING_S / dt)))
    window = int(round(TRACE_WINDOW_S / dt))
    for j in range(i - spacing, max(-1, i - window - 1), -spacing):
        age = (t[i] - t[j]) / TRACE_WINDOW_S
        rgb = color * (1.0 - age) + gray * age
        _add_sphere(scene, tool_pos[j], TRACE_RADIUS, (*rgb, 1.0))


# ---------- plot strip ----------


def _plot_strip(width: int, t: np.ndarray, series_keys: tuple[str, ...], curves: list[tuple[str, dict]],
                frame_idx: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Render the static strip once; return its RGB array and per-panel cursor geometry for every frame."""
    apply_style()
    n = len(series_keys)
    ncols = n if width / n >= PLOT_MIN_PANEL_W else 1
    nrows = math.ceil(n / ncols)
    height = PLOT_ROW_H * nrows if nrows == 1 else int(PLOT_ROW_H * 0.8) * nrows
    dpi = 100
    fig, axes = plt.subplots(nrows, ncols, figsize=(width / dpi, height / dpi), dpi=dpi, layout="constrained", squeeze=False)
    axes = list(axes.flat)
    frame_times = t[frame_idx]
    for ax, key in zip(axes, series_keys):
        label, unit, _ = SERIES[key]
        for method, values in curves:
            ax.plot(t, values[key], color=method_color(method), linewidth=1.1)
        ax.set_title(f"{label} ({unit})", loc="left")
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_xlabel("Time (s)")
        ax.margins(y=0.08)
        lo = min(float(np.nanmin(values[key])) for _, values in curves)
        hi = max(float(np.nanmax(values[key])) for _, values in curves)
        if lo >= 0:  # non-negative signals (forces, errors) start at zero; an all-zero signal keeps a visible range
            ax.set_ylim(0, max(hi * 1.08, 1e-3) if hi > 0 else 1.0)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    h_px = image.shape[0]
    panels = []
    for ax, key in zip(axes, series_keys):
        bbox = ax.get_window_extent()
        x_px = ax.transData.transform(np.column_stack([frame_times, np.zeros_like(frame_times)]))[:, 0]
        dots = {method: h_px - ax.transData.transform(np.column_stack([frame_times, values[key][frame_idx]]))[:, 1]
                for method, values in curves}
        panels.append({"top": h_px - bbox.y1, "bottom": h_px - bbox.y0, "x": x_px, "dots": dots})
    plt.close(fig)
    return image, panels


# ---------- rendering ----------


def render_comparison(episodes: list[tuple[str, str, str]], out_mp4: str, title: str, fps: int = 50, speed: float = 1.0,
                      camera: str = "front", view_size=(480, 360), plot_series=("pos_err", "f_normal", "tau2"),
                      poster_dir: str | None = None, scene: str | None = None) -> dict:
    """Render one to four episodes of the same case side by side; see the module docstring for the layout.

    `scene` replays the episodes in a task scene (Milestone 9): episodes then carry obj_pos and obj_quat,
    the pose of the scene's one free object, which is set with the arm's joints at every frame.

    `episodes` lists (method_key, label, npz_path). Frames are taken every round(speed / (fps * dt))
    samples, so `speed` is the playback rate relative to simulation time (0.25 = slow motion).
    Returns the clip record (also written as JSON next to the mp4)."""
    started = time.time()
    loaded = _load_episodes(episodes)
    for key in plot_series:
        if key not in SERIES:
            raise KeyError(f"unknown plot series {key!r}; known: {sorted(SERIES)}")
    view_w, view_h = int(view_size[0]), int(view_size[1])
    t = loaded[0][3]["t"]
    n_samples, dt = len(t), float(t[1] - t[0])
    step = max(1, int(round(speed / (fps * dt))))
    effective_speed = step * fps * dt
    if abs(effective_speed - speed) > 0.01 * speed:
        raise ValueError(f"speed {speed} is not reachable with {fps} fps at {1 / dt:.0f} Hz samples; nearest is {effective_speed:g}")
    frame_idx = np.arange(0, n_samples, step)
    n_frames = len(frame_idx)
    speed_text = f"{effective_speed:g}x"

    d = design()
    text, muted, surface, separator = _rgb(d["text"]), _rgb(d["muted"]), _rgb(d["surface"]), _rgb(d["separator"])
    gray01 = _rgb01(reference_color())
    f_title, f_clock, f_note = _font(20, "medium"), _font(18), _font(13)
    f_label, f_metrics = _font(16, "medium"), _font(14)

    # Per-method curves and causal metrics (joint RMSE in mrad and peak normal force so far).
    curves, rmse_mrad, peak_force = [], [], []
    for key, _, _, ep in loaded:
        curves.append((key, {s: np.asarray(SERIES[s][2](ep), dtype=np.float64) for s in plot_series}))
        err2 = ((ep["q_ref"] - ep["q"]) ** 2).sum(axis=1)
        rmse_mrad.append(1000.0 * np.sqrt(np.cumsum(err2) / (ep["q"].shape[1] * np.arange(1, n_samples + 1))))
        peak_force.append(np.maximum.accumulate(ep["f_normal"]))

    # Geometry.
    n_views = len(loaded)
    width = 2 * MARGIN + n_views * view_w + (n_views - 1) * GAP
    width += width % 2
    inner_w = width - 2 * MARGIN
    note_lines = _wrap(NOTE, f_note, inner_w)
    y_title = 16
    y_note = y_title + 26 + 4
    y_labels = y_note + 16 * len(note_lines) + 14
    y_views = y_labels + LABEL_H
    y_metrics = y_views + view_h + 8
    y_plot = y_metrics + METRICS_H + 6
    strip, panels = _plot_strip(inner_w, t, tuple(plot_series), curves, frame_idx)
    height = y_plot + strip.shape[0] + 12
    height += height % 2
    x_views = [MARGIN + v * (view_w + GAP) for v in range(n_views)]

    # Static base: title, note, method labels, view borders and the plot strip.
    base = Image.new("RGB", (width, height), surface)
    draw = ImageDraw.Draw(base)
    clock_w = f_clock.getlength("t = 00.000 s") + 14 + f_clock.getlength(speed_text)
    draw.text((MARGIN, y_title), _fit(title, f_title, inner_w - clock_w - 24), font=f_title, fill=text, anchor="la")
    for k, line in enumerate(note_lines):
        draw.text((MARGIN, y_note + 16 * k), line, font=f_note, fill=muted, anchor="la")
    for (key, label, _, _), x in zip(loaded, x_views):
        draw.rectangle([x, y_labels + 11, x + 15, y_labels + 14], fill=_rgb(method_color(key)))
        draw.text((x + 24, y_labels + 4), _fit(label, f_label, view_w - 24), font=f_label, fill=text, anchor="la")
        draw.rectangle([x - 1, y_views - 1, x + view_w, y_views + view_h], outline=separator)
    base.paste(Image.fromarray(strip), (MARGIN, y_plot))

    # Renderer and commanded tool positions at the frame times.
    model = _render_model(view_w, view_h, scene)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=view_h, width=view_w)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1  # the arm's meshes live in geom group 3
    nominal = NominalModel()
    q_ref = loaded[0][3]["q_ref"]
    p_ref = np.array([nominal.tool_pos(q_ref[i]) for i in frame_idx])
    ghost_rgba = (*gray01, GHOST_ALPHA)
    method_rgb01 = [_rgb01(method_color(key)) for key, _, _, _ in loaded]

    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poster_frames = {0: "first", (n_frames - 1) // 2: "middle", n_frames - 1: "last"}
    posters: dict[str, str] = {}
    if poster_dir is not None:
        Path(poster_dir).mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), format="FFMPEG", mode="I", fps=fps, codec="libx264", pixelformat="yuv420p",
                                quality=None, macro_block_size=1, ffmpeg_log_level="error", output_params=ENCODER_PARAMS)
    try:
        for f, i in enumerate(frame_idx):
            frame = base.copy()
            draw = ImageDraw.Draw(frame)
            x_right = width - MARGIN
            draw.text((x_right, y_title + 2), speed_text, font=f_clock, fill=muted, anchor="ra")
            draw.text((x_right - f_clock.getlength(speed_text) - 14, y_title + 2), f"t = {t[i]:.3f} s", font=f_clock, fill=text, anchor="ra")
            for v, (key, _, _, ep) in enumerate(loaded):
                data.qpos[:7] = ep["q"][i]
                if "obj_pos" in ep:  # task scene: the free object's joint follows the arm's seven
                    data.qpos[7:10], data.qpos[10:14] = ep["obj_pos"][i], ep["obj_quat"][i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                _add_trace(renderer.scene, ep["tool_pos"], t, int(i), method_rgb01[v], gray01)
                _add_sphere(renderer.scene, p_ref[f], GHOST_RADIUS, ghost_rgba)
                frame.paste(Image.fromarray(renderer.render()), (x_views[v], y_views))
                x = x_views[v]
                for label_text, value in (("joint RMSE", f"{rmse_mrad[v][i]:.1f} mrad"), ("peak normal force", f"{peak_force[v][i]:.1f} N")):
                    draw.text((x, y_metrics + 6), label_text, font=f_metrics, fill=muted, anchor="la")
                    x += f_metrics.getlength(label_text) + 5
                    draw.text((x, y_metrics + 6), value, font=f_metrics, fill=text, anchor="la")
                    x += f_metrics.getlength(value) + 18
            for panel in panels:
                cx = MARGIN + panel["x"][f]
                draw.line([(cx, y_plot + panel["top"]), (cx, y_plot + panel["bottom"])], fill=text, width=1)
                for (key, _, _, _), rgb in zip(loaded, method_rgb01):
                    cy = y_plot + panel["dots"][key][f]
                    if np.isfinite(cy):
                        draw.ellipse([cx - 3.5, cy - 3.5, cx + 3.5, cy + 3.5], fill=tuple(int(c * 255) for c in rgb), outline=surface, width=1)
            array = np.asarray(frame)
            writer.append_data(array)
            if poster_dir is not None and f in poster_frames:
                poster = Path(poster_dir) / f"{out_path.stem}-{poster_frames[f]}.png"
                frame.save(poster)
                posters[poster_frames[f]] = str(poster)
    finally:
        writer.close()

    record = {
        "path": str(out_path), "record": str(out_path.with_suffix(".json")),
        "frames": n_frames, "fps": fps, "speed": effective_speed, "duration_s": n_frames / fps,
        "width": width, "height": height, "resolution": [width, height], "view_size": [view_w, view_h],
        "camera": camera, "title": title, "plot_series": list(plot_series),
        "t_start": float(t[0]), "t_end": float(t[frame_idx[-1]]), "sample_step": step, "sample_rate_hz": 1.0 / dt,
        "posters": posters,
        "episodes": [{"method": key, "label": label, "npz": str(path), "sha256": _sha256(path),
                      "joint_rmse_mrad": float(rmse_mrad[v][-1]), "peak_normal_force_n": float(peak_force[v][-1])}
                     for v, (key, label, path, _) in enumerate(loaded)],
        "reference": "gray sphere: commanded tool position, q_ref through llc.sim.nominal.NominalModel",
        "trace": f"measured tool path over the last {TRACE_WINDOW_S:g} s, gray (old) to method color (new)",
        "metrics": "joint RMSE over all joints and samples from t = 0 to the displayed time (mrad); peak normal force so far (N)",
        "scene": _repo_relative(scene or SCENE_PATH), "render_seconds": round(time.time() - started, 2),
    }
    Path(record["record"]).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# ---------- command line ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a side-by-side comparison video from stored episodes.")
    parser.add_argument("--episodes-dir", type=Path, default=Path("runs/m1/episodes"))
    parser.add_argument("--case", required=True, help="case id, e.g. transition-test-012")
    parser.add_argument("--methods", nargs="+", required=True, help="method keys (research/design.json)")
    parser.add_argument("--labels", nargs="*", default=None, help="one label per method; defaults to the keys")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", required=True, help="output .mp4")
    parser.add_argument("--poster-dir", default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--view-size", type=int, nargs=2, default=(480, 360), metavar=("W", "H"))
    parser.add_argument("--series", nargs="+", default=("pos_err", "f_normal", "tau2"), choices=sorted(SERIES))
    args = parser.parse_args()
    labels = args.labels or args.methods
    if len(labels) != len(args.methods):
        parser.error("--labels must match --methods")
    episodes = [(m, lbl, str(args.episodes_dir / f"{args.case}__{m}.npz")) for m, lbl in zip(args.methods, labels)]
    record = render_comparison(episodes, args.out, args.title, fps=args.fps, speed=args.speed, camera=args.camera,
                               view_size=tuple(args.view_size), plot_series=tuple(args.series), poster_dir=args.poster_dir)
    print(f"{record['path']}: {record['frames']} frames, {record['width']}x{record['height']}, {record['duration_s']:.2f} s "
          f"at {record['speed']:g}x, rendered in {record['render_seconds']} s; record {record['record']}")


if __name__ == "__main__":
    main()
