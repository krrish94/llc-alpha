"""Representative episodes for the handoff page (research/reports/18-handoff): ten side-by-side clips
spanning the protocol's families and the two task families, the best classical controller of the
family against the two-teacher student (seed 0), with the privileged oracle in the contact clips.

The in-distribution clips replay the benchmark's saved representative episodes (runs/m10r2/episodes_test,
the median case of the tracking-tuned PD per family). The held-out and task clips are simulated here for
the clip's controllers only (runs/m10r2/episodes_handoff), with the selection rule of each case stated in
its caption and computed from the benchmark tables (runs/m10r2/metrics_test.json).

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/handoff_videos.py [--workers 6]
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
from videocheck import POSTER_MIN_DIFF, check as video_check  # noqa: E402

from experiments.d_evaluate import case_groups  # noqa: E402
from llc.eval.parallel import _get_case, _init_worker, _worker, evaluate  # noqa: E402

REPORT = Path("research/reports/18-handoff")
METRICS = Path("runs/m10r2/metrics_test.json")
EPISODES_TEST = Path("runs/m10r2/episodes_test")
EPISODES_NEW = Path("runs/m10r2/episodes_handoff")
STUDENT_RUN = "distill_two-s0"
FIELDS = ("t", "q", "qd", "q_ref", "qd_ref", "tau", "tool_pos", "f_normal", "f_tangent", "n_tool_contacts", "n_other_contacts")
TASK_FIELDS = ("obj_pos", "obj_quat", "f_object", "f_object_env")
LABELS = {"pd_shared": "PD, tracking-tuned", "pd_shared_cap": "PD, cap-tuned", "pd_feedforward": "PD, tracking-tuned, with feedforward",
          "pd_shared_cap_feedforward": "PD, cap-tuned, with feedforward", "oracle_pd": "Oracle (privileged ceiling)", STUDENT_RUN: "Two-teacher student (seed 0)"}


def controller_specs() -> dict[str, dict]:
    pd = json.load(open("runs/m9/gains_shared.json"))["spec"]
    cap = json.load(open("runs/m10/gains_shared.json"))["spec"]
    return {"pd_shared": pd, "pd_shared_cap": cap, "pd_feedforward": {**pd, "kind": "pd_feedforward"}, "pd_shared_cap_feedforward": {**cap, "kind": "pd_feedforward"},
            "oracle_pd": {"kind": "oracle", "base": pd}, STUDENT_RUN: {"kind": "learned", "path": f"runs/m10r2/{STUDENT_RUN}/best.pt"}}


# ------------------------------------------------------------------------------------- case selection
def rows(data: dict, group: str, method: str, regime: str) -> list[dict]:
    return sorted([m for m in data["groups"][group][method] if m["regime"] == regime], key=lambda m: m["case_id"])


def median_case(data: dict, group: str, method: str, regime: str, key: str) -> str:
    r = sorted(rows(data, group, method, regime), key=lambda m: m[key])
    return r[len(r) // 2]["case_id"]


def push_cases(data: dict, classical: str = "pd_shared") -> tuple[str, str, dict]:
    """(a push the student converts, a push both fail): the converted case with the median yaw reduction,
    and among the pushes both fail the one with the median classical yaw. Returns the counts as well."""
    pd = {m["case_id"]: m for m in rows(data, "tasks", classical, "push")}
    st = {m["case_id"]: m for m in rows(data, "tasks", STUDENT_RUN, "push")}
    converted = sorted([c for c in pd if st[c]["task_success"] and not pd[c]["task_success"]], key=lambda c: pd[c]["push_yaw"] - st[c]["push_yaw"])
    lost = [c for c in pd if pd[c]["task_success"] and not st[c]["task_success"]]
    both_fail = sorted([c for c in pd if not pd[c]["task_success"] and not st[c]["task_success"]], key=lambda c: pd[c]["push_yaw"])
    counts = {"classical": int(sum(m["task_success"] for m in pd.values())), "student": int(sum(m["task_success"] for m in st.values())),
              "converted": len(converted), "lost": len(lost), "both_fail": len(both_fail), "n": len(pd)}
    return converted[len(converted) // 2], both_fail[len(both_fail) // 2], counts


def find_case(groups: dict, group: str, case_id: str):
    return next(c for c in groups[group] if c.id == case_id)


# ------------------------------------------------------------------------------------- episodes
def episode_path(group: str, case_id: str, name: str) -> Path:
    if group == "in_distribution":
        p = EPISODES_TEST / f"{case_id}__{name}.npz"
        if p.exists():
            return p
    return EPISODES_NEW / f"{group}__{case_id}__{name}.npz"


def ensure_episodes(specs: dict, groups: dict, group: str, case_id: str, names: list[str], workers: int) -> tuple[list[Path], str | None]:
    """Paths of the clip's episodes (simulated here when the benchmark did not save them) and the task scene, if any."""
    case_spec = find_case(groups, group, case_id)
    if "cases" not in _worker:  # the case cache of the evaluation workers, used here in the main process
        _init_worker()
    task = _get_case(case_spec).meta.get("task")
    scene = task["scene"] if task else None
    missing = [n for n in names if not episode_path(group, case_id, n).exists()]
    if missing:
        EPISODES_NEW.mkdir(parents=True, exist_ok=True)
        print(f"simulating {group} {case_id}: {missing}", flush=True)
        res = evaluate({n: specs[n] for n in missing}, [case_spec], keep_episodes=True, workers=min(workers, len(missing)))
        for n in missing:
            ep = res[n][0]["episode"]
            arrays = {k: getattr(ep, k) for k in FIELDS}
            if task:
                arrays.update({k: getattr(ep, k) for k in TASK_FIELDS})
                arrays["scene"] = np.array(scene)
            np.savez_compressed(episode_path(group, case_id, n), **arrays)
    return [episode_path(group, case_id, n) for n in names], scene


def timeseries_json(eps: list[tuple[str, str, Path]], series_keys: tuple[str, ...], stride: int = 5) -> dict:
    from video import SERIES

    loaded = {key: dict(np.load(path)) for key, _, path in eps}
    t = next(iter(loaded.values()))["t"][::stride]
    series = [{"key": k, "label": SERIES[k][0], "unit": SERIES[k][1]} for k in series_keys]
    values = {k: {key: np.asarray(SERIES[k][2](ep), float)[::stride].round(5).tolist() for key, ep in loaded.items()} for k in series_keys}
    return {"t": t.round(4).tolist(), "series": series, "methods": [{"key": k, "label": l} for k, l, _ in eps], "values": values}


# ------------------------------------------------------------------------------------- clips
def clip_plan(data: dict) -> list[dict]:
    rep = data["representative_cases"]
    free_payload = median_case(data, "payload_3kg", "pd_feedforward", "free", "pos_rmse")
    blocked_soft = median_case(data, "surface_500", "pd_shared_cap", "blocked", "composite_cap")
    push_win, push_fail, pc = push_cases(data)
    pick = median_case(data, "tasks", "pd_feedforward", "pick", "place_force_peak_20ms")
    free_rows = sorted(rows(data, "in_distribution", "pd_feedforward", "free"), key=lambda m: m["pos_rmse"])
    free_worst = free_rows[-1]["case_id"]  # the feedforward PD's worst free-motion case (where the nominal model is most wrong); the rule is recorded here, not in the caption (advisor, 2026-11-17)
    st_free = {m["case_id"]: m["pos_rmse"] for m in rows(data, "in_distribution", STUDENT_RUN, "free")}
    pd_free = {m["case_id"]: m["pos_rmse"] for m in free_rows}
    free_wins = sum(st_free[c] < pd_free[c] for c in pd_free)
    student = STUDENT_RUN
    med = "the median case of the tracking-tuned PD's composite, the benchmark's representative case"
    return [
        {"stem": "hold", "group": "in_distribution", "case": rep["hold"], "methods": ["pd_shared", student], "speed": 1.0, "camera": "side", "series": ("pos_err", "tau2", "tool_z"),
         "min_poster_diff": 0.2, "title": "Hold with a 10 N pulse", "caption": f"Hold with a 10 N tool pulse of 0.2 s at t = 1 s ({med}), real time. The learned controller holds the loaded, perturbed instance where the PD sags under the payload and the pulse: the error plot shows the deflection and its recovery; family means 1.6 against 8.1 mrad."},
        {"stem": "free", "group": "in_distribution", "case": free_worst, "methods": ["pd_feedforward", student], "speed": 1.0, "camera": "side", "series": ("pos_err", "tau2", "tool_z"),
         "title": "Free joint-space motion", "caption": f"Free motion, real time, against the feedforward PD, the best classical controller of this family, on an instance and payload on which the nominal model is wrong (the PD at {1e3 * pd_free[free_worst]:.1f} mrad, the student at {1e3 * st_free[free_worst]:.1f}). Both follow the gray commanded tool; the error plot separates them: the feedforward does not know this robot, the learned part infers it from the history. Over the sixteen cases the student (seed 0) wins {free_wins} and loses {16 - free_wins}, the losses where the PD is already at 1 to 3 mrad (Section 2.3); the family means are 2.6 against 4.9 mrad."},
        {"stem": "free_payload", "group": "payload_3kg", "case": free_payload, "methods": ["pd_feedforward", student], "speed": 1.0, "camera": "side", "series": ("pos_err", "tau2", "tool_z"),
         "title": "Free motion with a 3 kg payload (held out)", "caption": "Free motion with a 3 kg payload at the tool, a held-out group (the case with the median error of the feedforward PD in the group; the payload is a point mass and is not drawn), real time. The nominal feedforward does not know the payload and the PD droops under it; the learned controller infers it from the history (family means 5.0 against 20.6 mrad)."},
        {"stem": "contact2", "group": "in_distribution", "case": rep["contact2"], "methods": ["pd_shared_cap", "oracle_pd", student], "speed": 0.25, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"),
         "title": "Press and slide", "caption": f"Press and slide ({med}), quarter speed, against the cap-tuned PD (the best classical controller of this family) and the privileged oracle. The reference asks for 0.4 to 1.4 cm of depth; the force plot shows the PD pressing with its full stiffness (18 N mean over the family), the oracle and the student pressing inside the 7 to 12 N band, the student with a noisier force during the slide (the detachments of light sliding contact on this plant, report 14). The impact transient at the onset, 300 to 370 N for all three, is the blind approach, not the contact law."},
        {"stem": "blocked", "group": "in_distribution", "case": rep["blocked"], "methods": ["pd_shared_cap_feedforward", "oracle_pd", student], "speed": 0.25, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"),
         "title": "Blocked target", "caption": f"Blocked target ({med}), quarter speed: the commanded path passes 3 to 8 cm through the surface. The PD follows its error into the surface (23 N above the cap over the family); the oracle and the student stop at the surface and wait at a bounded force (5 N above the cap, apparent stiffness 390 N/m), then follow the reference when it returns. The impact transient at the onset is alike for all three: the approach is blind."},
        {"stem": "blocked_soft", "group": "surface_500", "case": blocked_soft, "methods": ["pd_shared_cap", "oracle_pd", student], "speed": 0.25, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"),
         "title": "Blocked target on a 500 N/m surface (held out)", "caption": "Blocked target on a compliant surface of 500 N/m, a held-out group (the case with the median composite of the cap-tuned PD in the group), quarter speed. The surface yields, so the controller's stiffness sets how deep it sinks: the PD keeps its own stiffness (contact stiffness 920 N/m over the group) and pushes hardest, the oracle holds its 500 N/m spring at 15 N, and the student lies between them in this case (group mean 450 N/m) although it never saw a compliant surface in training; it chatters briefly after the release."},
        {"stem": "task", "group": "in_distribution", "case": rep["task"], "methods": ["pd_feedforward", "oracle_pd", student], "speed": 1.0, "camera": "detail", "series": ("pos_err", "f_normal", "tau2"),
         "title": "Task sequence: reach, press, slide, lift, transport, press, lift", "caption": f"The task sequence ({med}), a family held out from training, real time: reach, press 1 cm, slide 10 cm, lift, transport, press 0.5 cm, lift. The force plot shows the two presses: the feedforward PD at 10 N above the cap over the family, the student at 2 N, with the oracle's profile; the impact transients at the two onsets (100 to 130 N) are blind approaches for all three."},
        {"stem": "push_win", "group": "tasks", "case": push_win, "methods": ["pd_shared", student], "speed": 1.0, "camera": "detail", "series": ("pos_err", "f_object", "tau2"),
         "title": "Box pushing: a case the learned controller converts", "caption": f"Box pushing with the closed gripper, real time, case {push_win}: selected as the case with the median yaw reduction among the {pc['converted']} of {pc['n']} pushes that fail for the PD (box turned beyond 15 degrees) and succeed for the student, which also loses {pc['lost']} of the PD's {pc['classical']} successes ({pc['student']} successes in all). The force plot is the tool-box contact force. The box turns when the fingertip faces meet it off-centre; the student's smaller tracking error keeps the push through the box's centre."},
        {"stem": "push_fail", "group": "tasks", "case": push_fail, "methods": ["pd_shared", student], "speed": 1.0, "camera": "detail", "series": ("pos_err", "f_object", "tau2"),
         "title": "Box pushing: a case every controller fails", "caption": f"Box pushing, real time, case {push_fail}: selected as the median PD yaw among the {pc['both_fail']} of {pc['n']} pushes that fail for both controllers. The box turns for both; the low-level controller cannot correct a push line the task policy chose badly, which is why box yaw is a task-policy problem (Section 6)."},
        {"stem": "pick", "group": "tasks", "case": pick, "methods": ["pd_feedforward", student], "speed": 1.0, "camera": "detail", "series": ("pos_err", "f_object_env", "tau2"),
         "title": "Pick-and-place: placing a held block", "caption": f"Pick-and-place, real time, case {pick}: the gripper carries a 0.3 kg block to a target, descends 2 mm below contact, releases and lifts (the case with the median placing force of the feedforward PD). The force plot is the block-table contact force: the PD presses the block onto the table with its full stiffness (14 N over the family), the student places it at 5 N and tracks the lift after the payload change with half the error."},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None, help="clip stems to render (default: all)")
    ap.add_argument("--captions-only", action="store_true", help="rewrite data/videos.html from the existing clips without simulating or rendering")
    args = ap.parse_args()
    data = json.loads(METRICS.read_text())
    groups = case_groups(data["n_cases"])
    specs = controller_specs()
    media, datadir = REPORT / "media", REPORT / "data"
    media.mkdir(parents=True, exist_ok=True); datadir.mkdir(parents=True, exist_ok=True)
    snippets = []
    for clip in clip_plan(data):
        if args.only and clip["stem"] not in args.only:
            continue
        stem = f"{clip['stem']}_{int(clip['speed'] * 100):03d}x"
        if args.captions_only:
            rec = json.loads((media / f"{stem}.json").read_text())
        else:
            paths, scene = ensure_episodes(specs, groups, clip["group"], clip["case"], clip["methods"], args.workers)
            eps = [(n.split("-s")[0], LABELS[n], str(p)) for n, p in zip(clip["methods"], paths)]  # palette keys are per method, not per seed
            json.dump(timeseries_json(eps, clip["series"]), open(datadir / f"ts_{clip['stem']}.json", "w"))
            rec = render_comparison(eps, str(media / f"{stem}.mp4"), f"{clip['title']}, case {clip['case']}", speed=clip["speed"], camera=clip["camera"],
                                    plot_series=clip["series"], poster_dir=str(media), scene=scene)
        ok, msg = video_check(str(media / f"{stem}.mp4"), min_poster_diff=clip.get("min_poster_diff", POSTER_MIN_DIFF))
        print(msg, flush=True)
        if not ok:
            raise SystemExit(f"clip check failed: {stem}")
        poster = Path(rec["posters"]["middle"]).name
        snippets.append(video_figure(f"media/{stem}.mp4", f"media/{poster}", clip["caption"] + " The gray sphere is the commanded tool position; the metrics under each view accumulate causally; the plot cursor follows the video.",
                                     f"fig-video-{clip['stem']}", timeseries_json_relpath=f"data/ts_{clip['stem']}.json", speed=clip["speed"]))
    out = datadir / "videos.html"
    if args.only:  # partial re-render: keep the other snippets
        existing = out.read_text() if out.exists() else ""
        for s in snippets:
            fig_id = s.split('id="', 1)[1].split('"', 1)[0]
            start = existing.find(f'id="{fig_id}"')
            if start >= 0:
                a = existing.rfind("<figure", 0, start); b = existing.find("</figure>", start) + len("</figure>")
                existing = existing[:a] + s + existing[b:]
            else:
                existing += ("\n" if existing else "") + s
        out.write_text(existing)
    else:
        out.write_text("\n".join(snippets))
    print(f"wrote {len(snippets)} video figures to {out}")


if __name__ == "__main__":
    main()
