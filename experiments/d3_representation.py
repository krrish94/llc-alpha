"""Stage 3 launcher: representation study for the deep dive (history window, stride, dense samples,
Fourier features) on one parameterization. Prints or runs the training commands; each run is a
d2_train.py invocation with identical budget and seed.

    uv run python experiments/d3_representation.py --kind pd_residual --threads 3 --parallel 4
"""
from __future__ import annotations

import argparse
import subprocess
import time

CONFIGS = {
    "strided_only": ["--history", "10", "--history-stride", "10", "--history-dense", "0"],
    "dense32_strided10": ["--history", "10", "--history-stride", "10", "--history-dense", "32"],  # D2 default (reuse runs/d2)
    "dense64_only": ["--history", "0", "--history-stride", "1", "--history-dense", "64"],
    "dense32_strided5ms": ["--history", "10", "--history-stride", "5", "--history-dense", "32"],
    "dense8_only": ["--history", "0", "--history-stride", "1", "--history-dense", "8"],
    "no_history": ["--history", "0", "--history-stride", "1", "--history-dense", "0"],
    "dense32_strided10_fourier": ["--history", "10", "--history-stride", "10", "--history-dense", "32", "--encoding", "fourier"],
    "strided_only_fourier": ["--history", "10", "--history-stride", "10", "--history-dense", "0", "--encoding", "fourier"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--stage-iters", nargs=3, default=["100", "100", "300"])
    ap.add_argument("--skip", nargs="*", default=["dense32_strided10"])
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    cmds = []
    for name, extra in CONFIGS.items():
        if name in args.skip:
            continue
        out = f"runs/d3/{args.kind}_{name}-s{args.seed}"
        cmd = ["uv", "run", "--no-sync", "python", "experiments/d2_train.py", "--kind", args.kind, "--seed", str(args.seed), "--out", out, "--threads", str(args.threads),
               "--stage-iters", *args.stage_iters, "--variant", name, *extra]
        cmds.append((out, cmd))
    if args.dry:
        for out, cmd in cmds:
            print(" ".join(cmd))
        return
    running = []
    for out, cmd in cmds:
        while len([p for p in running if p.poll() is None]) >= args.parallel:
            time.sleep(10)
        log = open(out + ".log", "w") if __import__("os").makedirs("runs/d3", exist_ok=True) is None else None
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**__import__("os").environ, "PYTHONPATH": "."}))
        print("launched", out, flush=True)
    for p in running:
        p.wait()
    print("all representation runs finished")


if __name__ == "__main__":
    main()
