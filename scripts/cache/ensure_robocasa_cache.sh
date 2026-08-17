#!/usr/bin/env bash

# Materialize RoboCasa/RoboSuite source and assets on local SSD for evaluation.
# The packed Python 3.11 environment is intentionally not extracted: training
# and evaluation use the shared Python 3.12 Robotwin environment.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

cache_root=$(kouro_cache_root)
inspection_root=$(kouro_resolve_inspection_root)
setup_dir="$inspection_root/environment_setup"
archive="$setup_dir/artifacts/robocasa365-source-assets-1.0.1.tar.zst"
decoder="$setup_dir/zstd_decompress.sh"
source_hash=f9d0c286f06ec5dc411bbba89aaca6d690b5e3c03b57fd5998a70a5566f772ce
cache_dir="$cache_root/robocasa/${source_hash:0:16}"
ready_file="$cache_dir/.ready"

[ -f "$archive" ] || kouro_die "RoboCasa source/assets archive is missing: $archive"
[ -x "$decoder" ] || kouro_die "zstd decoder is missing or not executable: $decoder"

mkdir -p "$cache_root/locks" "$(dirname -- "$cache_dir")"
exec 9>"$cache_root/locks/robocasa-assets.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$source_hash" ]; then
    kouro_log "RoboCasa asset cache hit: $cache_dir"
    printf '%s\n' "$cache_dir"
    exit 0
fi

available_kib=$(df -Pk -- "$cache_root" | awk 'NR == 2 {print $4}')
required_kib=$((32 * 1024 * 1024))
[ "$available_kib" -ge "$required_kib" ] || \
    kouro_die "RoboCasa evaluation cache needs at least 32 GiB local free space"

partial_dir="${cache_dir}.partial.$$"
kouro_safe_remove_cache_path "$partial_dir"
if [ -e "$cache_dir" ]; then
    kouro_safe_remove_cache_path "$cache_dir"
fi
mkdir -p "$partial_dir/third_party"
trap 'kouro_safe_remove_cache_path "$partial_dir"' EXIT

kouro_log "RoboCasa cache miss; extracting source and assets from NAS (this is the long first-run step)..."
tar -I "$decoder" -xf "$archive" -C "$partial_dir/third_party"
[ -d "$partial_dir/third_party/robocasa/robocasa" ] || kouro_die "RoboCasa extraction failed"
[ -d "$partial_dir/third_party/robosuite/robosuite" ] || kouro_die "RoboSuite extraction failed"
printf '%s\n' "$source_hash" > "$partial_dir/.ready"
mv -- "$partial_dir" "$cache_dir"
trap - EXIT

kouro_log "RoboCasa asset cache ready: $cache_dir"
printf '%s\n' "$cache_dir"
