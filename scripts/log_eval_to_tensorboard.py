#!/usr/bin/env python3
"""Append numeric LeRobot evaluation results to an experiment's TensorBoard log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def flatten(prefix: str, value: Any) -> list[tuple[str, float]]:
    if isinstance(value, bool):
        return [(prefix, float(value))]
    if isinstance(value, (int, float)):
        return [(prefix, float(value))]
    if isinstance(value, dict):
        result: list[tuple[str, float]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.extend(flatten(child_prefix, child))
        return result
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_info", type=Path)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(args.eval_info.read_text(encoding="utf-8"))
    writer = SummaryWriter(log_dir=str(args.log_dir))
    for name, value in flatten("eval", payload.get("overall", payload)):
        writer.add_scalar(name, value, args.step)
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
