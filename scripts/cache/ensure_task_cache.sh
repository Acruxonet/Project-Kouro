#!/usr/bin/env bash

# Copy one immutable LeRobot task from NAS to this container's local SSD.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

project_root=$(kouro_project_root)
cache_root=$(kouro_cache_root)
task=${1:-TurnOffSinkFaucet}
stats_episodes=${2:-100}
[[ "$stats_episodes" =~ ^[1-9][0-9]*$ ]] || kouro_die "stats episode count must be positive"
converter="$project_root/scripts/data/convert_state_quat_to_rotation6d.py"
[ -f "$converter" ] || kouro_die "Rotation 6D converter is missing: $converter"

case "$task" in
    TurnOffSinkFaucet) task_date=20250819 ;;
    *) kouro_die "Task cache is currently enabled only for TurnOffSinkFaucet, got: $task" ;;
esac

source_dir="$project_root/data/robocasa365/v1.0/pretrain/atomic/$task/$task_date/lerobot"
[ -f "$source_dir/meta/info.json" ] || kouro_die "Dataset is missing: $source_dir"
[ -d "$source_dir/data" ] || kouro_die "Dataset data directory is missing: $source_dir/data"
[ -d "$source_dir/videos" ] || kouro_die "Dataset video directory is missing: $source_dir/videos"

source_fingerprint=$(
    cd -- "$source_dir"
    find . -type f -printf '%P %s\n' | LC_ALL=C sort | sha256sum | awk '{print $1}'
)
converter_hash=$(sha256sum -- "$converter" | awk '{print $1}')
fingerprint=$(printf 'state-rotation6d-v1 %s %s stats=%s\n' \
    "$source_fingerprint" "$converter_hash" "$stats_episodes" | \
    sha256sum | awk '{print $1}')
cache_dir="$cache_root/data/$task/$task_date/${fingerprint:0:16}"
ready_file="$cache_dir/.ready"

mkdir -p "$cache_root/locks" "$(dirname -- "$cache_dir")"
exec 9>"$cache_root/locks/data-$task.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$fingerprint" ]; then
    kouro_log "Dataset cache hit: $cache_dir/lerobot"
    printf '%s\n' "$cache_dir/lerobot"
    exit 0
fi

required_kib=$(du -sk -- "$source_dir" | awk '{print $1}')
available_kib=$(df -Pk -- "$cache_root" | awk 'NR == 2 {print $4}')
required_with_margin=$((required_kib + 1024 * 1024))
[ "$available_kib" -ge "$required_with_margin" ] || \
    kouro_die "Not enough local disk: need dataset size plus 1 GiB, available ${available_kib} KiB"

partial_dir="${cache_dir}.partial.$$"
kouro_safe_remove_cache_path "$partial_dir"
if [ -e "$cache_dir" ]; then
    kouro_safe_remove_cache_path "$cache_dir"
fi
mkdir -p "$partial_dir"
trap 'kouro_safe_remove_cache_path "$partial_dir"' EXIT

kouro_log "Dataset cache miss; copying $task from NAS to local SSD..."
cp -a -- "$source_dir" "$partial_dir/lerobot"
kouro_log "Converting cached observation.state from quaternion16 to rotation6d20..."
"$(kouro_resolve_python)" "$converter" "$partial_dir/lerobot" \
    --stats-episodes "$stats_episodes"
[ -f "$partial_dir/lerobot/meta/info.json" ] || kouro_die "Copied dataset failed validation"
grep -q '"shape": \[' "$partial_dir/lerobot/meta/info.json" || kouro_die "Converted metadata is invalid"
[ -f "$partial_dir/lerobot/meta/kouro_rotation6d.json" ] || \
    kouro_die "Rotation 6D conversion marker is missing"
printf '%s\n' "$fingerprint" > "$partial_dir/.ready"
mv -- "$partial_dir" "$cache_dir"
trap - EXIT

kouro_log "Dataset cache ready: $cache_dir/lerobot"
printf '%s\n' "$cache_dir/lerobot"
