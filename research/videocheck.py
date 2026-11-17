"""Verify a comparison video against its clip record (written by research/video.py).

    PYTHONPATH=. uv run --no-sync python research/videocheck.py media/clip.mp4 [more clips...]

Decodes every frame of the mp4 and checks the frame count and resolution against the record,
checks that the first/middle/last posters have the video's resolution, differ from each other
and match the decoded first and last frames, then prints one summary line per clip.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

POSTER_MIN_DIFF = 0.5    # mean |difference| (0-255) below which two posters count as the same frame
DECODE_MAX_DIFF = 12.0   # mean |difference| allowed between a poster and its decoded (lossy) frame


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def check(arg: str, min_poster_diff: float = POSTER_MIN_DIFF) -> tuple[bool, str]:
    """Verify a clip against its record; ``min_poster_diff`` is the mean absolute difference below which two
    posters count as the same frame (a caller lowers it for a clip whose motion is small by design, such as a hold)."""
    path = Path(arg)
    record_path = path if path.suffix == ".json" else path.with_suffix(".json")
    if not record_path.is_file():
        return False, f"FAIL {arg}: no clip record at {record_path}"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    mp4 = Path(record["path"])
    if not mp4.is_absolute() and not mp4.is_file():
        mp4 = record_path.parent / mp4.name
    width, height, expected = int(record["width"]), int(record["height"]), int(record["frames"])
    problems: list[str] = []

    count, first, last = 0, None, None
    with imageio.get_reader(str(mp4), format="FFMPEG") as reader:
        meta = reader.get_meta_data()
        for frame in reader:
            if frame.shape != (height, width, 3):
                problems.append(f"frame {count} has shape {frame.shape}, expected {(height, width, 3)}")
                break
            if count == 0:
                first = frame.copy()
            last = frame
            count += 1
    if count != expected:
        problems.append(f"decoded {count} frames, record says {expected}")
    fps = meta.get("fps")
    if fps is not None and abs(float(fps) - float(record["fps"])) > 0.01:
        problems.append(f"container fps {fps}, record says {record['fps']}")

    posters = record.get("posters") or {}
    diffs, match = [], []
    if posters:
        missing = [k for k in ("first", "middle", "last") if k not in posters]
        if missing:
            problems.append(f"record lacks posters {missing}")
        else:
            images = {}
            for key in ("first", "middle", "last"):
                p = Path(posters[key])
                if not p.is_file():
                    p = record_path.parent / p.name
                if not p.is_file():
                    problems.append(f"poster {key} missing: {posters[key]}")
                    continue
                img = np.asarray(Image.open(p).convert("RGB"))
                if img.shape != (height, width, 3):
                    problems.append(f"poster {key} has shape {img.shape}, expected {(height, width, 3)}")
                images[key] = img
            if len(images) == 3:
                for a, b in (("first", "middle"), ("middle", "last"), ("first", "last")):
                    diff = mean_abs_diff(images[a], images[b])
                    diffs.append(diff)
                    if diff < min_poster_diff:
                        problems.append(f"posters {a} and {b} are the same frame (mean |diff| {diff:.2f})")
                if first is not None and last is not None and not problems:
                    for key, frame in (("first", first), ("last", last)):
                        diff = mean_abs_diff(images[key], frame)
                        match.append(diff)
                        if diff > DECODE_MAX_DIFF:
                            problems.append(f"poster {key} does not match the decoded frame (mean |diff| {diff:.1f})")

    if problems:
        return False, f"FAIL {mp4}: " + "; ".join(problems)
    summary = (f"OK {mp4}: {count} frames, {width}x{height} at {record['fps']} fps = {count / record['fps']:.2f} s "
               f"({record['speed']:g}x, episode {record['t_start']:.3f}-{record['t_end']:.3f} s)")
    if diffs:
        summary += f"; posters first/middle/last differ (mean |diff| {', '.join(f'{x:.1f}' for x in diffs)})"
    if match:
        summary += f"; first/last posters match decoded frames (mean |diff| {', '.join(f'{x:.1f}' for x in match)})"
    return True, summary


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    ok = True
    for arg in argv:
        passed, line = check(arg)
        print(line)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
