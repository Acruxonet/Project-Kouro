#!/usr/bin/env bash

# Start training and TensorBoard as two windows in a detached tmux session.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
project_root=$(kouro_project_root)
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

command -v tmux >/dev/null 2>&1 || kouro_die "tmux is not installed in this container"

[ "$#" -eq 0 ] || kouro_die "Usage: $0"
run_id=$KOURO_EXPERIMENT_ID
session_name=${KOURO_TMUX_SESSION:-kouro-$run_id}
kouro_validate_id "$run_id" experiment_id
kouro_validate_id "$session_name" tmux_session
tmux has-session -t "$session_name" 2>/dev/null && kouro_die "tmux session already exists: $session_name"

env_args=(env "KOURO_EXPERIMENT=$(basename -- "$experiment_dir")" \
    "KOURO_TENSORBOARD_PORT=$KOURO_TENSORBOARD_PORT")
while IFS= read -r variable_name; do
    case "$variable_name" in KOURO_EXPERIMENT|KOURO_TENSORBOARD_PORT) continue ;; esac
    if [[ -v $variable_name ]]; then
        env_args+=("$variable_name=${!variable_name}")
    fi
done < <(compgen -A variable KOURO_ | LC_ALL=C sort -u)

train_argv=("${env_args[@]}" bash "$project_root/scripts/train_and_eval_turnoff_sink_faucet.sh")
tensorboard_argv=("${env_args[@]}" bash "$project_root/scripts/tensorboard_turnoff_sink_faucet.sh" "$KOURO_TENSORBOARD_PORT")
printf -v train_shell 'cd %q && ' "$project_root"
printf -v train_joined '%q ' "${train_argv[@]}"
train_shell+="$train_joined"
printf -v tensorboard_shell 'cd %q && ' "$project_root"
printf -v tensorboard_joined '%q ' "${tensorboard_argv[@]}"
tensorboard_shell+="$tensorboard_joined"

tmux new-session -d -s "$session_name" -n train "$train_shell"
tmux set-option -t "$session_name" remain-on-exit on >/dev/null
tmux set-option -t "$session_name" history-limit 100000 >/dev/null
tmux set-option -t "$session_name" @kouro_run_id "$run_id" >/dev/null
tmux set-option -t "$session_name" @kouro_experiment_dir "$experiment_dir" >/dev/null
tmux new-window -d -t "$session_name" -n tensorboard "$tensorboard_shell"
tmux select-window -t "$session_name:train"

printf 'Started Project-Kouro experiment.\n'
printf '  experiment:   %s\n' "$run_id"
printf '  tmux session: %s\n' "$session_name"
printf '  attach:       tmux attach -t %s\n' "$session_name"
printf '  status:       bash scripts/tmux_status.sh %s\n' "$session_name"
printf '  TensorBoard:  http://127.0.0.1:%s\n' "$KOURO_TENSORBOARD_PORT"
printf '  stop:         bash scripts/tmux_stop.sh %s\n' "$session_name"
