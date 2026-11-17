"""Helpers for authoring report fragments (research/reports/<id>/report.html).

Every helper returns an HTML string. compare_figure() computes the static table for the
default selection with the same seeded bootstrap as research/assets/notebook.js, so the
static fallback and the interactive table agree to the displayed precision.
"""

from __future__ import annotations

import html
import re
from typing import Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 1000
_U32 = 0xFFFFFFFF


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt(x: float | None, precision: int = 4) -> str:
    """Python %.4g formatting; "—" for a missing value. Mirrored by fmt() in notebook.js."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x == 0:
        return "0"
    return f"{x:.{precision}g}"


# ---------- deterministic bootstrap (identical to notebook.js) ----------


def _fnv1a(text: str) -> int:
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = (h * 0x01000193) & _U32
    return h


def _mulberry32(seed: int):
    a = seed & _U32

    def rand() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & _U32
        t = ((a ^ (a >> 15)) * (1 | a)) & _U32
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & _U32)) ^ t) & _U32
        return ((t ^ (t >> 14)) & _U32) / 4294967296

    return rand


def _quantile(sorted_values: np.ndarray, q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _mean(values: Sequence[float]) -> float:
    total = 0.0
    for v in values:  # sequential sum, bit-identical to the JS loop
        total += v
    return total / len(values)


def bootstrap_ci(values: Sequence[float], seed_key: str) -> tuple[float, float]:
    """95% bootstrap CI of the mean (percentile method), seeded by a string key."""
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (values[0], values[0])
    rand = _mulberry32(_fnv1a(seed_key))
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for b in range(BOOTSTRAP_RESAMPLES):
        total = 0.0
        for _ in range(n):
            total += values[int(rand() * n)]
        means[b] = total / n
    means.sort()
    return (_quantile(means, 0.025), _quantile(means, 0.975))


def compare_stats(data: dict, regime: str, metric: str) -> list[dict]:
    """Per-method mean, CI, counts and paired difference vs the reference method."""
    values = data["values"][regime][metric]
    ref = data["reference_method"]
    ref_values = values.get(ref, [])
    stats = []
    for m in data["methods"]:
        raw = values.get(m["key"], [])
        present = [x for x in raw if x is not None]
        pairs = [x - r for x, r in zip(raw, ref_values) if x is not None and r is not None]
        stat = {
            "key": m["key"], "label": m["label"], "n": len(present), "total": len(raw),
            "mean": _mean(present) if present else float("nan"),
            "ci": bootstrap_ci(present, f"{regime}/{metric}/{m['key']}"),
            "is_ref": m["key"] == ref, "pairs": len(pairs), "diff": float("nan"), "diff_ci": (float("nan"), float("nan")),
        }
        if not stat["is_ref"] and pairs:
            stat["diff"] = _mean(pairs)
            stat["diff_ci"] = bootstrap_ci(pairs, f"{regime}/{metric}/{m['key']}-vs-{ref}")
        stats.append(stat)
    return stats


def _validate_compare(data: dict) -> None:
    for key in ("methods", "regimes", "metrics", "reference_method", "values"):
        if key not in data:
            raise ValueError(f"compare data lacks {key!r}")
    method_keys = [m["key"] for m in data["methods"]]
    if data["reference_method"] not in method_keys:
        raise ValueError(f"reference_method {data['reference_method']!r} is not in methods")
    for regime in data["regimes"]:
        for metric in data["metrics"]:
            block = data["values"].get(regime["key"], {}).get(metric["key"])
            if block is None:
                raise ValueError(f"values missing for regime {regime['key']!r}, metric {metric['key']!r}")
            lengths = {len(block.get(k, [])) for k in method_keys}
            if len(lengths) != 1:
                raise ValueError(f"case counts differ across methods for {regime['key']}/{metric['key']}: {sorted(lengths)}")
            ids = data.get("case_ids", {}).get(regime["key"])
            if ids is not None and len(ids) != lengths.pop():
                raise ValueError(f"case_ids length mismatch for regime {regime['key']!r}")


def _ci_text(ci: tuple[float, float]) -> str:
    return "—" if np.isnan(ci[0]) else f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def _selection_text(data: dict, regime: dict, metric: dict, stats: list[dict]) -> str:
    unit = f" ({metric['unit']})" if metric.get("unit") else ""
    ref = next((m["label"] for m in data["methods"] if m["key"] == data["reference_method"]), data["reference_method"])
    cases = stats[0]["total"] if stats else 0
    return (f"Selection: {regime['label']}, {metric['label']}{unit}; n = {cases} cases. "
            f"Mean with 95% bootstrap CI ({BOOTSTRAP_RESAMPLES} resamples); difference vs {ref} is paired by case, "
            "failed cases dropped from the pair.")


def compare_table(stats: list[dict]) -> str:
    headers = ["Method", "Mean", "95% CI", "n", "Δ vs reference", "95% CI (paired)", "Pairs"]
    rows = []
    for st in stats:
        n_text = str(st["n"]) if st["n"] == st["total"] else f"{st['n']} of {st['total']}"
        rows.append([st["label"], fmt(st["mean"]), _ci_text(st["ci"]), n_text,
                     "reference" if st["is_ref"] else fmt(st["diff"]),
                     "—" if st["is_ref"] else _ci_text(st["diff_ci"]),
                     "—" if st["is_ref"] else str(st["pairs"])])
    return table(headers, rows, numeric_cols=range(1, 7), css_class="compare-table")


# ---------- figure and table builders ----------


def _source_link(json_relpath: str) -> str:
    return f'<a class="source" href="{esc(json_relpath)}">Data (JSON)</a>'


def compare_figure(json_relpath: str, data: dict, caption: str, fig_id: str,
                   default_regime: str | None = None, default_metric: str | None = None) -> str:
    """Paired metric comparison. The caller writes `data` to json_relpath; this returns the <figure>."""
    _validate_compare(data)
    regime_key = default_regime or data["regimes"][0]["key"]
    metric_key = default_metric or data["metrics"][0]["key"]
    regime = next(r for r in data["regimes"] if r["key"] == regime_key)
    metric = next(m for m in data["metrics"] if m["key"] == metric_key)
    stats = compare_stats(data, regime_key, metric_key)
    return (
        f'<figure class="widget compare" id="{esc(fig_id)}" data-widget="compare" data-src="{esc(json_relpath)}" '
        f'data-regime="{esc(regime_key)}" data-metric="{esc(metric_key)}">\n'
        f'<div class="widget-body">\n<p class="selection">{esc(_selection_text(data, regime, metric, stats))}</p>\n'
        f"{compare_table(stats)}\n</div>\n"
        f"<figcaption>{caption} {_source_link(json_relpath)}</figcaption>\n</figure>"
    )


def _timeseries_static(caption: str, fallback_svg_relpath: str | None) -> tuple[str, str]:
    """(data-fallback attribute, static content) for a time-series widget without JavaScript."""
    if fallback_svg_relpath:
        return (f' data-fallback="{esc(fallback_svg_relpath)}"',
                f'<img src="{esc(fallback_svg_relpath)}" alt="{esc(_plain(caption))}">')
    return "", '<p class="missing">No static figure for this viewer; the interactive view needs JavaScript.</p>'


def timeseries_figure(json_relpath: str, caption: str, fig_id: str, fallback_svg_relpath: str | None = None) -> str:
    """Time-series viewer; `fallback_svg_relpath` is a static image shown without JavaScript."""
    fallback, static = _timeseries_static(caption, fallback_svg_relpath)
    return (
        f'<figure class="widget timeseries" id="{esc(fig_id)}" data-widget="timeseries" data-src="{esc(json_relpath)}"{fallback}>\n'
        f'<div class="widget-body">\n{static}\n</div>\n'
        f"<figcaption>{caption} {_source_link(json_relpath)}</figcaption>\n</figure>"
    )


def video_figure(mp4_relpath: str, poster_relpath: str, caption: str, fig_id: str,
                 timeseries_json_relpath: str | None = None, fallback_svg_relpath: str | None = None,
                 speed: float = 1.0, size: tuple[int, int] | None = None) -> str:
    """Native video (controls, poster, no autoplay, no loop) rendered by research/video.py.

    With `timeseries_json_relpath` (same JSON schema as timeseries_figure) a time-series widget sits
    below the video, linked by data-video: its cursor follows playback and moving the cursor seeks
    the video. `speed` is the clip's playback rate relative to simulation time (the record's "speed",
    e.g. 0.25 for slow motion) so the widget can map video time to episode time; `size` (width,
    height) reserves the layout box before the metadata loads and prevents upscaling."""
    video_id = f"{fig_id}-video"
    dims = f' width="{int(size[0])}" height="{int(size[1])}"' if size else ""
    parts = [
        f'<figure class="video wide" id="{esc(fig_id)}">',
        f'<video id="{esc(video_id)}" controls preload="metadata" poster="{esc(poster_relpath)}"{dims}>',
        f'<source src="{esc(mp4_relpath)}" type="video/mp4">',
        f'<p class="missing">Video playback is not supported here; <a href="{esc(mp4_relpath)}">download the video (MP4)</a>.</p>',
        "</video>",
    ]
    links = [f'<a class="source" href="{esc(mp4_relpath)}">Video (MP4)</a>']
    if timeseries_json_relpath:
        fallback, static = _timeseries_static(caption, fallback_svg_relpath)
        parts.append(
            f'<div class="widget timeseries" data-widget="timeseries" data-src="{esc(timeseries_json_relpath)}" '
            f'data-video="{esc(video_id)}" data-video-speed="{speed:g}"{fallback}>\n'
            f'<div class="widget-body">\n{static}\n</div>\n</div>')
        links.append(_source_link(timeseries_json_relpath))
    parts.append(f"<figcaption>{caption} {' '.join(links)}</figcaption>\n</figure>")
    return "\n".join(parts)


def figure(image_relpath: str, caption: str, fig_id: str, wide: bool = False, alt: str | None = None) -> str:
    cls = ' class="wide"' if wide else ""
    alt_text = _plain(caption) if alt is None else alt
    return (f'<figure id="{esc(fig_id)}"{cls}>\n<img src="{esc(image_relpath)}" alt="{esc(alt_text)}">\n'
            f"<figcaption>{caption}</figcaption>\n</figure>")


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], caption: str | None = None,
          numeric_cols: Sequence[int] = (), css_class: str | None = None) -> str:
    """Compact table; cells are escaped, floats formatted with fmt(), numeric columns right-aligned."""
    numeric = set(numeric_cols)
    attr = lambda i: ' class="num"' if i in numeric else ""  # noqa: E731
    cell = lambda v: fmt(v) if isinstance(v, float) or v is None else esc(v)  # noqa: E731
    lines = [f"<table{' class=' + chr(34) + esc(css_class) + chr(34) if css_class else ''}>"]
    if caption:
        lines.append(f"<caption>{caption}</caption>")
    lines.append("<thead>\n<tr>" + "".join(f"<th{attr(i)}>{esc(t)}</th>" for i, t in enumerate(headers)) + "</tr>\n</thead>")
    lines.append("<tbody>")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells, expected {len(headers)}: {row!r}")
        lines.append("<tr>" + "".join(f"<td{attr(i)}>{cell(v)}</td>" for i, v in enumerate(row)) + "</tr>")
    lines.append("</tbody>\n</table>")
    return "\n".join(lines)


def _plain(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()
