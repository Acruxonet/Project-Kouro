#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"

session_name=${1:-}
[ -n "$session_name" ] || kouro_die "Usage: $0 TMUX_SESSION"
kouro_validate_id "$session_name" tmux_session
tmux has-session -t "$session_name" 2>/dev/null || kouro_die "No such tmux session: $session_name"
tmux kill-session -t "$session_name"
printf 'Stopped tmux session: %s\n' "$session_name"
