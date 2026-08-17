#!/usr/bin/env bash

# Install the exact RoboCasa-compatible MuJoCo and NumPy wheels into a local overlay.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

project_root=$(kouro_project_root)
cache_root=$(kouro_cache_root)
python_bin=$(kouro_resolve_python)
mujoco_wheel="$project_root/artifacts/wheels/mujoco-3.3.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
numpy_wheel="$project_root/artifacts/wheels/numpy-2.2.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
for wheel in "$mujoco_wheel" "$numpy_wheel"; do
    [ -f "$wheel" ] || kouro_die "Bundled RoboCasa wheel is missing: $wheel"
done

bundle_hash=$(
    sha256sum -- "$mujoco_wheel" "$numpy_wheel" | awk '{print $1}' | sha256sum | awk '{print $1}'
)
python_tag=$($python_bin -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
[ "$python_tag" = py312 ] || kouro_die "RoboCasa overlay requires Python 3.12, got $python_tag"
cache_dir="$cache_root/python/$python_tag-robocasa-${bundle_hash:0:12}"
ready_file="$cache_dir/.ready"

mkdir -p "$cache_root/locks" "$(dirname -- "$cache_dir")"
exec 9>"$cache_root/locks/robocasa-python.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$bundle_hash" ]; then
    kouro_log "RoboCasa Python overlay hit: $cache_dir"
    printf '%s\n' "$cache_dir"
    exit 0
fi

partial_dir="${cache_dir}.partial.$$"
kouro_safe_remove_cache_path "$partial_dir"
if [ -e "$cache_dir" ]; then
    kouro_safe_remove_cache_path "$cache_dir"
fi
mkdir -p "$partial_dir"
trap 'kouro_safe_remove_cache_path "$partial_dir"' EXIT

kouro_log "Installing MuJoCo 3.3.1 and NumPy 2.2.5 into local Python overlay..."
"$python_bin" -m pip install --no-deps --no-compile --target "$partial_dir" \
    "$mujoco_wheel" "$numpy_wheel" >&2
printf '%s\n' "$bundle_hash" > "$partial_dir/.ready"
mv -- "$partial_dir" "$cache_dir"
trap - EXIT

kouro_log "RoboCasa Python overlay ready: $cache_dir"
printf '%s\n' "$cache_dir"
