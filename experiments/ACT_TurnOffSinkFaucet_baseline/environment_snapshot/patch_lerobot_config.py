#!/usr/bin/env python3
"""Apply Project-Kouro patches to a local cached LeRobot source tree."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs_py", type=Path)
    parser.add_argument("train_py", type=Path)
    args = parser.parse_args()

    old = 'self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(16,))'
    new = 'self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(20,))'
    text = args.configs_py.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected exactly one RoboCasa state16 declaration, found {count}")
    args.configs_py.write_text(text.replace(old, new), encoding="utf-8")

    # Upstream builds eval_tracker but only sends it to WandB. Emit the tracker
    # through normal logging as well so the TensorBoard bridge sees periodic eval.
    train_text = args.train_py.read_text(encoding="utf-8")
    old_train = "                eval_tracker.pc_success = aggregated.pop(\"pc_success\")\n                if wandb_logger:"
    new_train = (
        "                eval_tracker.pc_success = aggregated.pop(\"pc_success\")\n"
        "                logging.info(eval_tracker)\n"
        "                if wandb_logger:"
    )
    train_count = train_text.count(old_train)
    if train_count != 1:
        raise SystemExit(f"Expected exactly one eval tracker logging insertion point, found {train_count}")
    args.train_py.write_text(train_text.replace(old_train, new_train), encoding="utf-8")


if __name__ == "__main__":
    main()
