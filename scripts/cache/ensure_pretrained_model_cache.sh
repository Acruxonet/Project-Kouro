#!/usr/bin/env bash

# Copy an immutable pretrained-model snapshot from NAS to container-local SSD.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

[ "$#" -eq 2 ] || kouro_die "Usage: $0 POLICY_TYPE SOURCE_SNAPSHOT"
policy_type=$1
source_snapshot=$(readlink -f -- "$2")
kouro_validate_id "$policy_type" policy_type
[ -d "$source_snapshot" ] || kouro_die "Pretrained snapshot is missing: $source_snapshot"
[ -f "$source_snapshot/config.json" ] || kouro_die "Missing pretrained config: $source_snapshot/config.json"
[ -f "$source_snapshot/model.safetensors" ] || \
    kouro_die "Missing pretrained weights: $source_snapshot/model.safetensors"

cache_root=$(kouro_cache_root)
fingerprint=$(
    {
        printf 'pretrained-model-v1 %s %s\n' "$policy_type" "$source_snapshot"
        find -L "$source_snapshot" -maxdepth 1 -type f -printf '%f %s %T@\n' | LC_ALL=C sort
    } | sha256sum | awk '{print $1}'
)
cache_dir="$cache_root/models/$policy_type/${fingerprint:0:16}"
model_dir="$cache_dir/model"
ready_file="$cache_dir/.ready"

mkdir -p "$cache_root/locks" "$(dirname -- "$cache_dir")"
exec 9>"$cache_root/locks/model-$policy_type.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$fingerprint" ]; then
    kouro_log "Pretrained model cache hit: $model_dir"
    printf '%s\n' "$model_dir"
    exit 0
fi

required_kib=$(du -skL -- "$source_snapshot" | awk '{print $1}')
available_kib=$(df -Pk -- "$cache_root" | awk 'NR == 2 {print $4}')
required_with_margin=$((required_kib + 1024 * 1024))
[ "$available_kib" -ge "$required_with_margin" ] || \
    kouro_die "Not enough local disk for $policy_type weights: need model size plus 1 GiB"

partial_dir="${cache_dir}.partial.$$"
kouro_safe_remove_cache_path "$partial_dir"
if [ -e "$cache_dir" ]; then
    kouro_safe_remove_cache_path "$cache_dir"
fi
mkdir -p "$partial_dir/model"
trap 'kouro_safe_remove_cache_path "$partial_dir"' EXIT

kouro_log "Copying $policy_type pretrained snapshot from NAS to local SSD..."
# Hugging Face snapshots contain relative symlinks into a separate blobs directory;
# dereference them so the standalone local snapshot remains valid.
cp -aL -- "$source_snapshot/." "$partial_dir/model/"
[ -f "$partial_dir/model/config.json" ] || kouro_die "Copied model config is missing"
[ -f "$partial_dir/model/model.safetensors" ] || kouro_die "Copied model weights are missing"
printf '%s\n' "$fingerprint" > "$partial_dir/.ready"
mv -- "$partial_dir" "$cache_dir"
trap - EXIT

kouro_log "Pretrained model cache ready: $model_dir"
printf '%s\n' "$model_dir"
