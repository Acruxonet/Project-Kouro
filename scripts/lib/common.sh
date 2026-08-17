#!/usr/bin/env bash

# Shared helpers for Project-Kouro launch scripts.

kouro_project_root() {
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd
}

kouro_experiment_dir() {
    local experiment_name experiment_dir
    experiment_name=${KOURO_EXPERIMENT:-DP_TurnOffSinkFaucet_baseline}
    kouro_validate_id "$experiment_name" experiment
    experiment_dir="$(kouro_project_root)/experiments/$experiment_name"
    [ -d "$experiment_dir" ] || kouro_die "Experiment directory is missing: $experiment_dir"
    printf '%s\n' "$experiment_dir"
}

kouro_resolve_policy_dir() {
    local policy_type=$1 configured_path=${2:-} project_root policy_dir
    kouro_validate_id "$policy_type" policy_type
    project_root=$(kouro_project_root)
    if [ -z "$configured_path" ]; then
        policy_dir="$project_root/policy/$policy_type"
    elif [[ "$configured_path" = /* ]]; then
        policy_dir=$configured_path
    else
        policy_dir="$project_root/$configured_path"
    fi
    [ -d "$policy_dir" ] || kouro_die "Policy source directory is missing: $policy_dir"
    readlink -f -- "$policy_dir"
}

kouro_log() {
    printf '[kouro] %s\n' "$*" >&2
}

kouro_die() {
    printf '[kouro] ERROR: %s\n' "$*" >&2
    exit 1
}

kouro_cache_root() {
    printf '%s\n' "${KOURO_CACHE_ROOT:-/tmp/project-kouro-cache}"
}

kouro_resolve_python() {
    local candidate
    if [ -n "${KOURO_PYTHON:-}" ]; then
        [ -x "$KOURO_PYTHON" ] || kouro_die "KOURO_PYTHON is not executable: $KOURO_PYTHON"
        readlink -f -- "$KOURO_PYTHON"
        return
    fi

    for candidate in \
        /mnt/data/nas/hufangchi/applications/conda_envs/Robotwin/bin/python \
        /root/main/applications/conda_envs/Robotwin/bin/python; do
        if [ -x "$candidate" ]; then
            readlink -f -- "$candidate"
            return
        fi
    done
    kouro_die "Cannot find the Python 3.12 Robotwin environment. Set KOURO_PYTHON."
}

kouro_resolve_lerobot_root() {
    local candidate
    if [ -n "${KOURO_LEROBOT_ROOT:-}" ]; then
        [ -d "$KOURO_LEROBOT_ROOT/src/lerobot" ] || \
            kouro_die "KOURO_LEROBOT_ROOT has no src/lerobot: $KOURO_LEROBOT_ROOT"
        readlink -f -- "$KOURO_LEROBOT_ROOT"
        return
    fi

    for candidate in /mnt/data/nas/hufangchi/lerobot /root/main/lerobot; do
        if [ -d "$candidate/src/lerobot" ]; then
            readlink -f -- "$candidate"
            return
        fi
    done
    kouro_die "Cannot find the shared LeRobot checkout. Set KOURO_LEROBOT_ROOT."
}

kouro_resolve_inspection_root() {
    local project_root candidate
    project_root=$(kouro_project_root)
    if [ -n "${KOURO_ROBOCASA_INSPECTION_ROOT:-}" ]; then
        [ -d "$KOURO_ROBOCASA_INSPECTION_ROOT/environment_setup/artifacts" ] || \
            kouro_die "Invalid KOURO_ROBOCASA_INSPECTION_ROOT: $KOURO_ROBOCASA_INSPECTION_ROOT"
        readlink -f -- "$KOURO_ROBOCASA_INSPECTION_ROOT"
        return
    fi

    for candidate in \
        "$project_root/../robocasa_inspection" \
        /mnt/data/nas/hufangchi/projects/robocasa_inspection \
        /root/main/projects/robocasa_inspection; do
        if [ -d "$candidate/environment_setup/artifacts" ]; then
            readlink -f -- "$candidate"
            return
        fi
    done
    kouro_die "Cannot find robocasa_inspection. Set KOURO_ROBOCASA_INSPECTION_ROOT."
}

kouro_validate_id() {
    local value=$1 label=${2:-identifier}
    [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || \
        kouro_die "$label may only contain letters, digits, dot, underscore and dash: $value"
}

kouro_safe_remove_cache_path() {
    local cache_root target
    cache_root=$(readlink -m -- "$(kouro_cache_root)")
    target=$(readlink -m -- "$1")
    case "$target" in
        "$cache_root"/*) rm -rf -- "$target" ;;
        *) kouro_die "Refusing to remove a path outside the cache root: $target" ;;
    esac
}
