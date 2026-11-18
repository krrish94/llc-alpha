"""The Neural Actuation Dataset (Dou et al., RSS 2026; MIT licence), the public servo telemetry the servo models of
ADR 0011 are identified on: OpenManipulator-X on Dynamixel XM430-W350 (goal and measured position, velocity, current in
mA, PWM duty, bus voltage, temperature at 59 Hz; payload and force-sensor subsets) and SO-101 on Feetech STS3215 (goal and
measured position, velocity in encoder steps per second, the load register, current, voltage in decivolts at 62 Hz).
The parquet files are downloaded from Hugging Face into runs/nad (not under version control) on first use.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path("runs/nad")
URL = "https://huggingface.co/datasets/frankzydou/NAD/resolve/main/parquet/{config}/{split}.parquet"
ID_COLUMNS = {"omx": ("subset", "task", "trajectory"), "so101": ("task", "payload", "trajectory")}


def load(config: str, split: str = "train") -> dict[str, np.ndarray]:
    """{column: array} of one config and split, downloaded when missing."""
    import pyarrow.parquet as pq

    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{config}_{split}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(config=config, split=split), path)
    t = pq.read_table(path)
    return {c: t.column(c).to_numpy() for c in t.column_names}


def trajectories(data: dict[str, np.ndarray], config: str) -> dict[str, np.ndarray]:
    """{trajectory id: row indices} with ids 'subset/task/trajectory' (OMX) or 'task/payload/trajectory' (SO-101)."""
    keys = ID_COLUMNS[config]
    ids = np.array(["/".join(str(data[k][i]) for k in keys) for i in range(len(data["timestamp"]))])
    return {name: np.where(ids == name)[0] for name in np.unique(ids)}
