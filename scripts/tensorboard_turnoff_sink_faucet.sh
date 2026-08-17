#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
project_root=$(kouro_project_root)
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

port=${1:-$KOURO_TENSORBOARD_PORT}
[ "$#" -le 1 ] || kouro_die "Usage: $0 [PORT]"
[[ "$port" =~ ^[0-9]+$ ]] || kouro_die "TensorBoard port must be numeric: $port"

python_bin=$(kouro_resolve_python)
log_dir="$experiment_dir/tensorboard"
wait_seconds=${KOURO_TENSORBOARD_WAIT_SECONDS:-1800}
elapsed=0
while [ ! -d "$log_dir" ] && [ "$elapsed" -lt "$wait_seconds" ]; do
    if [ $((elapsed % 30)) -eq 0 ]; then
        kouro_log "Waiting for TensorBoard log directory: $log_dir"
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
[ -d "$log_dir" ] || kouro_die "TensorBoard log directory did not appear: $log_dir"

kouro_log "TensorBoard: http://127.0.0.1:$port"
exec "$python_bin" -m tensorboard.main --logdir "$log_dir" --host 0.0.0.0 --port "$port"
