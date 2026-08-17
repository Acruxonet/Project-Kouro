#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
project_root=$(kouro_project_root)
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

[ "$#" -eq 0 ] || kouro_die "Usage: $0"
run_id=$KOURO_EXPERIMENT_ID
kouro_validate_id "$run_id" experiment_id
command -v tmux >/dev/null 2>&1 || kouro_die "tmux is not installed in this container"

session_name=${KOURO_TMUX_SESSION:-kouro-eval-$run_id}
kouro_validate_id "$session_name" tmux_session
tmux_cmd=(tmux -f /dev/null -L "$session_name")
"${tmux_cmd[@]}" has-session -t "$session_name" 2>/dev/null && \
    kouro_die "tmux session already exists: $session_name"

eval_argv=(env "KOURO_EXPERIMENT=$(basename -- "$experiment_dir")")
while IFS= read -r variable_name; do
    [ "$variable_name" = KOURO_EXPERIMENT ] && continue
    if [[ -v $variable_name ]]; then eval_argv+=("$variable_name=${!variable_name}"); fi
done < <(compgen -A variable KOURO_ | LC_ALL=C sort -u)
eval_argv+=(bash "$project_root/scripts/eval_turnoff_sink_faucet.sh")
printf -v eval_shell 'cd %q && ' "$project_root"
printf -v eval_joined '%q ' "${eval_argv[@]}"
eval_shell+="$eval_joined"

session_created=false
cleanup_incomplete_session() {
    if $session_created; then
        "${tmux_cmd[@]}" kill-session -t "$session_name" 2>/dev/null || true
    fi
}
trap cleanup_incomplete_session EXIT

# Keep the pane alive while configuring it so fast failures remain inspectable.
"${tmux_cmd[@]}" new-session -d -s "$session_name" -n eval 'exec sleep 2147483647'
session_created=true
"${tmux_cmd[@]}" set-option -w -t "$session_name:eval" remain-on-exit on >/dev/null
"${tmux_cmd[@]}" set-option -w -t "$session_name:eval" history-limit 100000 >/dev/null
"${tmux_cmd[@]}" set-option -t "$session_name" @kouro_run_id "$run_id" >/dev/null
"${tmux_cmd[@]}" set-option -t "$session_name" @kouro_experiment_dir "$experiment_dir" >/dev/null
"${tmux_cmd[@]}" respawn-pane -k -t "$session_name:eval" "$eval_shell"
session_created=false
trap - EXIT

printf 'Started RoboCasa evaluation.\n'
printf '  experiment:   %s\n' "$run_id"
printf '  tmux session: %s\n' "$session_name"
printf '  attach:       tmux -L %s attach -t %s\n' "$session_name" "$session_name"
