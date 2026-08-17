#!/usr/bin/env python3
"""Convert a local RoboCasa LeRobot v3 cache from state16 quaternions to state20 rotation 6D."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_KEY = "observation.state"
SOURCE_DIM = 16
TARGET_DIM = 20
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def quaternion_xyzw_to_rotation_6d(quaternions: np.ndarray) -> np.ndarray:
    """Return the first two rotation-matrix columns for batched xyzw quaternions."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternion array (N, 4), got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Dataset contains a zero-norm quaternion")
    x, y, z, w = (quaternions / norms).T
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        axis=1,
    )


def convert_state(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != SOURCE_DIM:
        raise ValueError(f"Expected state array (N, {SOURCE_DIM}), got {states.shape}")
    converted = np.concatenate(
        [
            states[:, 0:3],
            quaternion_xyzw_to_rotation_6d(states[:, 3:7]),
            states[:, 7:10],
            quaternion_xyzw_to_rotation_6d(states[:, 10:14]),
            states[:, 14:16],
        ],
        axis=1,
    )
    if converted.shape[1] != TARGET_DIM or not np.isfinite(converted).all():
        raise ValueError("Converted state failed shape/finite validation")
    return converted


def compute_feature_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Compute every statistic required by LeRobot normalization modes."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected a non-empty feature matrix, got {values.shape}")
    stats: dict[str, list[float] | list[int]] = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }
    for name, quantile in QUANTILES.items():
        stats[name] = np.quantile(values, quantile, axis=0).tolist()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--stats-episodes", type=int, default=100)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    marker_path = root / "meta" / "kouro_rotation6d.json"

    info = json.loads(info_path.read_text(encoding="utf-8"))
    current_shape = info["features"][STATE_KEY]["shape"]
    if current_shape == [TARGET_DIM] and marker_path.is_file():
        return
    if current_shape != [SOURCE_DIM]:
        raise ValueError(f"Expected metadata state shape [{SOURCE_DIM}], got {current_shape}")

    if args.stats_episodes < 1:
        raise ValueError("--stats-episodes must be positive")
    stats_state_parts: list[np.ndarray] = []
    stats_action_parts: list[np.ndarray] = []
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No data parquet files found below {root / 'data'}")

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        converted = convert_state(np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float64))
        episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        stats_mask = episode_indices < args.stats_episodes
        stats_state_parts.append(converted[stats_mask])
        stats_action_parts.append(
            np.asarray(table["action"].to_pylist(), dtype=np.float64)[stats_mask]
        )
        state_array = pa.array(converted.tolist(), type=pa.list_(pa.float64()))
        column_index = table.column_names.index(STATE_KEY)
        table = table.set_column(column_index, STATE_KEY, state_array)
        temp_path = parquet_path.with_suffix(".rotation6d.tmp.parquet")
        pq.write_table(table, temp_path, compression="zstd")
        os.replace(temp_path, parquet_path)

    all_states = np.concatenate(stats_state_parts, axis=0)
    all_actions = np.concatenate(stats_action_parts, axis=0)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[STATE_KEY] = compute_feature_stats(all_states)
    stats["action"] = compute_feature_stats(all_actions)
    info["features"][STATE_KEY]["shape"] = [TARGET_DIM]

    info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    marker_path.write_text(
        json.dumps(
            {
                "representation": "rotation_6d_first_two_matrix_columns",
                "source_quaternion_convention": "xyzw",
                "source_state_layout": "base_pos3+base_quat4+ee_pos3+ee_quat4+gripper2",
                "target_state_layout": "base_pos3+base_rot6d6+ee_pos3+ee_rot6d6+gripper2",
                "normalization_stats_episode_range": f"0:{args.stats_episodes}",
                "normalization_stats": ["min", "max", "mean", "std", *QUANTILES],
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
