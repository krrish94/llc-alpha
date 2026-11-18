# Learned low-level controllers for robot manipulation

Every torque-controlled manipulator runs a feedback loop between the policy's targets and the motors, usually a PD or impedance law at 100 Hz to 1 kHz with gains tuned by hand. This repository asks whether that loop can be learned in simulation so that one controller is competitive with, or better than, the best hand-tuned classical law in every operating regime at once: free motion, holding under disturbances, policy-like and step targets, pressing and sliding on surfaces, blocked targets, and two object-manipulation tasks, on a Franka Panda with a Robotiq 2F-85 in MuJoCo.

The controller that resulted does not replace the classical loop; it adds to it. A 1 kHz PD law with the nominal inverse-dynamics feedforward runs under a network that, every 10 ms, sets bounded per-joint gain multipliers and a bounded torque residual from 100 ms of proprioceptive history and an estimate of the external torque. It is trained in two stages, both in simulation: a residual by backpropagation through the plant on free-space references, then imitation of two privileged teachers, the ideal-robot torque in free space and a contact-impedance law in contact. On the pre-registered benchmark it is within 5% of, or better than, the best classical controller of each family on 28 of 33 family-and-group rows, three seeds, with a bounded contact force where the PD pushes through the surface.

## The notebook

The evidence is a research notebook of twenty-three reports under `research/reports/`, each with its question, setup, figures, tables, clips, findings and caveats. Start with the handoff page (report 18), a standalone summary of the investigation up to the torque-interface controller: the setting and the success criterion, the controller and its training, the hypothesis space with the outcome of every cell, what worked and what did not, the gaps, and what to revisit at scale. Reports 19 to 22 take the controller to position-controlled arms (servo interface profiles, a model-based and a learned torque-command interface, host-rate students, the i2rt YAM, a DC-motor layer; Checkpoint 3 is report 22). Report 23, in progress, is the consolidation on the torque-interface Panda: ten times the training data under a wider plant distribution and five new contact-rich families (wiping, erasing with a held block, peg-in-hole with a chamfer, a drawer, a door).

To read it, build the static site and open it in a browser:

```
pip install markdown
python research/build.py
python -m http.server --directory research/site
```

Serving over HTTP lets the interactive figures load their data; the pages also work from `file://` with static content.

## Layout

- `llc/`: the reusable parts. `sim/` (MuJoCo plant, robot instances, gripper, task scenes), `dynamics/` (a differentiable rigid-body model), `controllers/` (PD, feedforward PD, the privileged contact-impedance oracle, the learned controller with its latency queue), `learning/` (policy and history buffer, actor-critic through plant gradients, PPO, distillation, ideal-robot labels), `eval/` (references, cases, metrics, parallel evaluation), `references.py` and `tasks.py` (reference families and the task suite).
- `experiments/`: the scripts behind each report, in order of the milestones and deep dives (`m*_*.py`, `d*_*.py`), the benchmark (`d_evaluate.py`), the report builders (`d_report.py`, the `*_narrative.py` modules, `checkpoint_*.py`, `handoff.py`, `handoff_videos.py`).
- `research/`: the notebook sources (`reports/`, `catalog.json`, `design.json`), the site builder (`build.py`), the plotting style, the clip renderer and its checker.
- `runs/`: the launchers of the final runs (`m9/`, `m10/`, `m10r2/`) and of deep dive D5 (`d6/`). The checkpoints, metrics, logs and episodes they produce are not under version control.
- `tests/`: 65 tests of the physics, the models, the controllers, the metrics and the training code.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv sync` installs the dependencies (MuJoCo, PyTorch, NumPy, SciPy, matplotlib, CMA-ES, imageio). Run everything from the repository root with the plant selected by an environment variable:

```
PYTHONPATH=. uv run pytest    # the test suite, on the default plant (the Panda with a spherical tool, v2)
export LLC_PLANT=v3           # the Panda with the Robotiq 2F-85, the plant of reports 14 to 18 and of the launchers
```

The experiments were run on a laptop (Apple M5 Pro, CPU MuJoCo); the handoff page lists what each stage costs there.

## Reproducing the final controller

The pipeline has three stages; the exact commands with their arguments are the launchers under `runs/m9/` and `runs/m10r2/`.

1. Shared PD gains by CMA-ES on the training split: `experiments/m1_tune_gains.py` (`runs/m9/gains_shared.json`; the cap-tuned set of the classical baseline is `runs/m10/gains_shared.json`).
2. The base residual by actor-critic through the plant on the free-space families: `experiments/d2_train.py --kind pd_residual --feedforward --objective ac --median-clip 3 --pos-sigma 0.02 --w-rate 0.003 --stage-iters 100 150 0 --val-objective free` (`runs/m10r2/launch_bases.sh`).
3. The student by DAgger from the two teachers: `experiments/d7_distill.py --gain-residual --ext-torque --free-teacher ideal --oracle-f_min 7 --betas 1,0 --free-weight 3 --iters 12` (`runs/m10r2/launch_two_teachers.sh`).

The benchmark against the classical set and the privileged oracle is `experiments/d_evaluate.py` (`runs/m10r2/eval_r2.sh`), and the reports are regenerated from its metrics file by the report builders.
