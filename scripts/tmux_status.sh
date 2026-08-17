#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"

if [ "$#" -eq 0 ]; then
    tmux list-sessions -F '#{session_name}  windows=#{session_windows}  attached=#{session_attached}' 2>/dev/null || \
        kouro_die "No tmux server is running"
    exit 0
fi

session_name=$1
kouro_validate_id "$session_name" tmux_session
if tmux has-session -t "$session_name" 2>/dev/null; then
    tmux_cmd=(tmux)
    socket_name=default
elif tmux -L "$session_name" has-session -t "$session_name" 2>/dev/null; then
    tmux_cmd=(tmux -L "$session_name")
    socket_name=$session_name
else
    kouro_die "No such tmux session: $session_name"
fi
run_id=$("${tmux_cmd[@]}" show-option -qv -t "$session_name" @kouro_run_id)
experiment_dir=$("${tmux_cmd[@]}" show-option -qv -t "$session_name" @kouro_experiment_dir)

printf 'session: %s\n' "$session_name"
printf 'socket: %s\n' "$socket_name"
printf 'experiment: %s\n' "${run_id:-unknown}"
"${tmux_cmd[@]}" list-windows -t "$session_name" -F 'window=#{window_index}:#{window_name} active=#{window_active} dead=#{pane_dead} command=#{pane_current_command}'

if [ -n "$experiment_dir" ] && [ -f "$experiment_dir/status.env" ]; then
    printf '\nstatus.env\n'
    sed -n '1,20p' "$experiment_dir/status.env"
fi

if "${tmux_cmd[@]}" list-windows -t "$session_name" -F '#{window_name}' | grep -qx train; then
    output_window=train
    printf '\nlast training output\n'
else
    output_window=eval
    printf '\nlast evaluation output\n'
fi
"${tmux_cmd[@]}" capture-pane -p -t "$session_name:$output_window" -S -30 | tail -30
