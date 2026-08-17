#!/usr/bin/env bash

# One-shot experiment workflow: offline training with periodic evaluation,
# followed by the final 50-episode RoboCasa evaluation.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

kouro_log "Stage 1/2: training with $KOURO_INLINE_EVAL_EPISODES-episode eval every $KOURO_EVAL_FREQ steps"
bash "$script_dir/train_turnoff_sink_faucet.sh"

kouro_log "Stage 2/2: final $KOURO_FINAL_EVAL_EPISODES-episode RoboCasa evaluation"
bash "$script_dir/eval_turnoff_sink_faucet.sh"
