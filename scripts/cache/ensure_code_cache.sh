#!/usr/bin/env bash

# Copy LeRobot source to local SSD and overlay the policy selected by the experiment.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

project_root=$(kouro_project_root)
cache_root=$(kouro_cache_root)
lerobot_root=$(kouro_resolve_lerobot_root)
policy_type=${KOURO_POLICY_TYPE:-diffusion}
policy_dir=$(kouro_resolve_policy_dir "$policy_type" "${KOURO_POLICY_DIR:-}")
cache_schema=policy-package-v4
environment_dir=${KOURO_ENVIRONMENT_DIR:-$project_root/environment}
[ -d "$environment_dir" ] || kouro_die "Environment source directory is missing: $environment_dir"

[ -f "$policy_dir/__init__.py" ] || kouro_die "Policy package is missing: $policy_dir/__init__.py"
for required_file in robocasa.py patch_lerobot_config.py; do
    [ -f "$environment_dir/$required_file" ] || \
        kouro_die "Missing environment file: $environment_dir/$required_file"
done

policy_hash=$(
    cd -- "$policy_dir"
    find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | \
        LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
)
environment_hash=$(
    cd -- "$environment_dir"
    sha256sum robocasa.py patch_lerobot_config.py | sha256sum | awk '{print $1}'
)
source_commit=$(git -C "$lerobot_root" rev-parse HEAD 2>/dev/null || printf 'unknown')
source_diff_hash=$(git -C "$lerobot_root" diff --no-ext-diff -- src/lerobot 2>/dev/null | \
    sha256sum | awk '{print $1}')
source_hash=$(printf '%s %s\n' "$source_commit" "$source_diff_hash" | sha256sum | awk '{print $1}')

custom_hash=$(printf '%s %s %s %s\n' "$cache_schema" "$policy_type" "$policy_hash" "$environment_hash" | \
    sha256sum | awk '{print $1}')
cache_dir="$cache_root/code/lerobot-${source_hash:0:12}-kouro-${custom_hash:0:12}"
ready_file="$cache_dir/.ready"
ready_value="$source_hash $cache_schema $policy_type $policy_hash $environment_hash"

mkdir -p "$cache_root/locks" "$(dirname -- "$cache_dir")"
exec 9>"$cache_root/locks/lerobot-code.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$ready_value" ]; then
    kouro_log "LeRobot code cache hit: $cache_dir"
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

kouro_log "Code cache miss; copying LeRobot source to local SSD..."
cp -a -- "$lerobot_root/src/lerobot" "$partial_dir/lerobot"
kouro_log "Installing Project-Kouro policy/$policy_type from $policy_dir..."
policy_target="$partial_dir/lerobot/policies/$policy_type"
if [ -e "$policy_target" ]; then
    kouro_safe_remove_cache_path "$policy_target"
fi
mkdir -p "$policy_target"
cp -a -- "$policy_dir/." "$policy_target/"
if [ -d "$policy_dir/_shared" ]; then
    # Some policies keep original LeRobot siblings here (for example Pi05's
    # ../pi_gemma.py and ../rtc imports). Install each sibling at
    # lerobot/policies/ while keeping policy/<type>/ self-contained in Kouro.
    shopt -s dotglob nullglob
    shared_entries=("$policy_dir/_shared/"*)
    shopt -u dotglob nullglob
    for shared_entry in "${shared_entries[@]}"; do
        shared_name=$(basename -- "$shared_entry")
        shared_target="$partial_dir/lerobot/policies/$shared_name"
        if [ -e "$shared_target" ]; then
            kouro_safe_remove_cache_path "$shared_target"
        fi
        cp -a -- "$shared_entry" "$shared_target"
    done
    kouro_safe_remove_cache_path "$policy_target/_shared"
fi
if [ -f "$policy_dir/normalization.py" ]; then
    cp -a -- "$policy_dir/normalization.py" \
        "$partial_dir/lerobot/processor/normalize_processor.py"
    # This file mirrors lerobot/processor/normalize_processor.py and uses
    # processor-relative imports, so it must not be imported as a policy module
    # during discover_packages_path traversal.
    kouro_safe_remove_cache_path "$policy_target/normalization.py"
fi
cp -a -- "$environment_dir/robocasa.py" "$partial_dir/lerobot/envs/robocasa.py"
"$(kouro_resolve_python)" "$environment_dir/patch_lerobot_config.py" \
    "$partial_dir/lerobot/envs/configs.py" \
    "$partial_dir/lerobot/scripts/lerobot_train.py"
printf '%s\n' "$ready_value" > "$partial_dir/.ready"
mv -- "$partial_dir" "$cache_dir"
trap - EXIT

kouro_log "LeRobot code cache ready: $cache_dir"
printf '%s\n' "$cache_dir"
