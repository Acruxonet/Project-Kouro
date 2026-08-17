#!/usr/bin/env python3
"""Run a command, tee its output, and convert LeRobot metrics to TensorBoard."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TOKEN_RE = re.compile(
    r"(?P<name>step|smpl|ep|epch|loss|grdn|lr|updt_s|data_s|∑rwrd|success|eval_s):"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[KMB]?)"
)
TAG_MAP = {
    "smpl": "train/samples",
    "ep": "train/episodes",
    "epch": "train/epochs",
    "loss": "train/loss",
    "grdn": "train/grad_norm",
    "lr": "train/learning_rate",
    "updt_s": "train/update_seconds",
    "data_s": "train/dataload_seconds",
    "∑rwrd": "eval/avg_sum_reward",
    "success": "eval/pc_success",
    "eval_s": "eval/eval_seconds",
}


def human_number(value: str) -> float:
    scale = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
    if value[-1:] in scale:
        return float(value[:-1]) * scale[value[-1]]
    return float(value)


def parse_metrics(line: str) -> tuple[int, dict[str, float]] | None:
    clean = ANSI_RE.sub("", line).replace("\r", "\n")
    matches = {m.group("name"): human_number(m.group("value")) for m in TOKEN_RE.finditer(clean)}
    if "step" not in matches or not ({"loss", "success"} & matches.keys()):
        return None
    step = int(matches.pop("step"))
    return step, matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir), flush_secs=10)
    writer.add_text("run/command", f"`{shlex.join(command)}`", 0)

    child: subprocess.Popen[str] | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )
    assert child.stdout is not None

    with args.log_file.open("a", encoding="utf-8", buffering=1) as log_handle:
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            parsed = parse_metrics(line)
            if parsed is None:
                continue
            step, metrics = parsed
            for name, value in metrics.items():
                tag = TAG_MAP.get(name)
                if tag is not None:
                    writer.add_scalar(tag, value, step)
            writer.flush()

    return_code = child.wait()
    writer.add_text("run/exit", f"exit_code={return_code}", 0)
    writer.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
