#!/usr/bin/env bash

# Seed one Hugging Face hub repository cache from NAS into container-local SSD.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/../lib/common.sh"

[ "$#" -eq 1 ] || kouro_die "Usage: $0 SOURCE_HF_REPO_CACHE"
source_repo=$(readlink -f -- "$1")
[ -d "$source_repo" ] || kouro_die "Hugging Face repository cache is missing: $source_repo"
repo_name=$(basename -- "$source_repo")
[[ "$repo_name" = models--* ]] || kouro_die "Unexpected Hugging Face cache name: $repo_name"
[ -f "$source_repo/refs/main" ] || kouro_die "Hugging Face cache has no refs/main: $source_repo"

cache_root=$(kouro_cache_root)
destination="$cache_root/huggingface/hub/$repo_name"
fingerprint=$(
    {
        printf 'hf-repo-v1 %s\n' "$repo_name"
        cat -- "$source_repo/refs/main"
        find -L "$source_repo" -type f -printf '%P %s %T@\n' | LC_ALL=C sort
    } | sha256sum | awk '{print $1}'
)
ready_file="$destination/.kouro-ready"

mkdir -p "$cache_root/locks" "$(dirname -- "$destination")"
exec 9>"$cache_root/locks/hf-$repo_name.lock"
flock 9

if [ -f "$ready_file" ] && [ "$(cat -- "$ready_file")" = "$fingerprint" ]; then
    kouro_log "Hugging Face cache hit: $destination"
    printf '%s\n' "$destination"
    exit 0
fi

partial_dir="${destination}.partial.$$"
kouro_safe_remove_cache_path "$partial_dir"
if [ -e "$destination" ]; then
    kouro_safe_remove_cache_path "$destination"
fi
mkdir -p "$partial_dir"
trap 'kouro_safe_remove_cache_path "$partial_dir"' EXIT

kouro_log "Seeding local Hugging Face cache: $repo_name"
cp -a -- "$source_repo/." "$partial_dir/"
printf '%s\n' "$fingerprint" > "$partial_dir/.kouro-ready"
mv -- "$partial_dir" "$destination"
trap - EXIT

kouro_log "Hugging Face cache ready: $destination"
printf '%s\n' "$destination"
