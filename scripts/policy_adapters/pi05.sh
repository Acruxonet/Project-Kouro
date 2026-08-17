#!/usr/bin/env bash

# Pi05 fine-tuning from the local official lerobot/pi05_libero checkpoint.

: "${KOURO_PI05_PRETRAINED_SOURCE:=}"
: "${KOURO_PI05_PRETRAINED_REPO_CACHE:=/mnt/data/nas/hufangchi/cache/huggingface/hub/models--lerobot--pi05_libero}"
: "${KOURO_PI05_TOKENIZER_REPO_CACHE:=/mnt/data/nas/hufangchi/cache/huggingface/hub/models--google--paligemma-3b-pt-224}"
: "${KOURO_PI05_DTYPE:=bfloat16}"
: "${KOURO_PI05_CHUNK_SIZE:=50}"
: "${KOURO_PI05_N_ACTION_STEPS:=10}"
: "${KOURO_PI05_EMPTY_CAMERAS:=0}"
: "${KOURO_PI05_GRADIENT_CHECKPOINTING:=true}"
: "${KOURO_PI05_TRAIN_EXPERT_ONLY:=false}"
: "${KOURO_PI05_FREEZE_VISION_ENCODER:=false}"
: "${KOURO_PI05_COMPILE_MODEL:=false}"

kouro_pi05_resolve_pretrained_source() {
    local revision source_path
    if [ -n "$KOURO_PI05_PRETRAINED_SOURCE" ]; then
        source_path=$(readlink -f -- "$KOURO_PI05_PRETRAINED_SOURCE")
    else
        [ -f "$KOURO_PI05_PRETRAINED_REPO_CACHE/refs/main" ] || \
            kouro_die "Pi05 pretrained cache has no refs/main: $KOURO_PI05_PRETRAINED_REPO_CACHE"
        revision=$(tr -d '\r\n' < "$KOURO_PI05_PRETRAINED_REPO_CACHE/refs/main")
        source_path="$KOURO_PI05_PRETRAINED_REPO_CACHE/snapshots/$revision"
    fi
    [ -d "$source_path" ] || kouro_die "Pi05 pretrained snapshot is missing: $source_path"
    [ -f "$source_path/config.json" ] || kouro_die "Pi05 config is missing: $source_path/config.json"
    [ -f "$source_path/model.safetensors" ] || \
        kouro_die "Pi05 weights are missing: $source_path/model.safetensors"
    printf '%s\n' "$source_path"
}

kouro_policy_validate_config() {
    local boolean_name
    [[ "$KOURO_PI05_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || \
        kouro_die "KOURO_PI05_CHUNK_SIZE must be positive"
    [[ "$KOURO_PI05_N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
        kouro_die "KOURO_PI05_N_ACTION_STEPS must be positive"
    [ "$KOURO_PI05_N_ACTION_STEPS" -le "$KOURO_PI05_CHUNK_SIZE" ] || \
        kouro_die "Pi05 n_action_steps cannot exceed chunk_size"
    [[ "$KOURO_PI05_EMPTY_CAMERAS" =~ ^[0-9]+$ ]] || \
        kouro_die "KOURO_PI05_EMPTY_CAMERAS must be non-negative"
    case "$KOURO_PI05_DTYPE" in bfloat16|float32) ;; *) kouro_die "Unsupported Pi05 dtype" ;; esac
    for boolean_name in \
        KOURO_PI05_GRADIENT_CHECKPOINTING KOURO_PI05_TRAIN_EXPERT_ONLY \
        KOURO_PI05_FREEZE_VISION_ENCODER KOURO_PI05_COMPILE_MODEL; do
        case "${!boolean_name}" in true|false) ;; *) kouro_die "$boolean_name must be true or false" ;; esac
    done
    KOURO_PI05_RESOLVED_PRETRAINED_SOURCE=$(kouro_pi05_resolve_pretrained_source)
    grep -q '"type"[[:space:]]*:[[:space:]]*"pi05"' \
        "$KOURO_PI05_RESOLVED_PRETRAINED_SOURCE/config.json" || \
        kouro_die "Selected pretrained checkpoint is not Pi05"
    [ -f "$KOURO_PI05_TOKENIZER_REPO_CACHE/refs/main" ] || \
        kouro_die "Local PaliGemma tokenizer cache is missing: $KOURO_PI05_TOKENIZER_REPO_CACHE"
}

kouro_policy_prepare_local_artifacts() {
    local _project_root=$1 cache_root=$2 dry_run=${3:-false}
    if [ "$dry_run" = true ]; then
        KOURO_POLICY_PRETRAINED_PATH=$KOURO_PI05_RESOLVED_PRETRAINED_SOURCE
        kouro_log "Pi05 dry run: verified pretrained weights; skipped the 14 GiB local copy."
        return
    fi
    KOURO_POLICY_PRETRAINED_PATH=$(KOURO_CACHE_ROOT="$cache_root" \
        "$script_dir/cache/ensure_pretrained_model_cache.sh" \
        pi05 "$KOURO_PI05_RESOLVED_PRETRAINED_SOURCE")
    KOURO_PI05_LOCAL_TOKENIZER_CACHE=$(KOURO_CACHE_ROOT="$cache_root" \
        "$script_dir/cache/ensure_hf_repo_cache.sh" "$KOURO_PI05_TOKENIZER_REPO_CACHE")
}

kouro_policy_prepare_eval_artifacts() {
    local _project_root=$1 cache_root=$2
    KOURO_PI05_LOCAL_TOKENIZER_CACHE=$(KOURO_CACHE_ROOT="$cache_root" \
        "$script_dir/cache/ensure_hf_repo_cache.sh" "$KOURO_PI05_TOKENIZER_REPO_CACHE")
}

kouro_policy_append_selector_args() {
    local -n target_args=$1
    [ -n "${KOURO_POLICY_PRETRAINED_PATH:-}" ] || kouro_die "Pi05 pretrained path was not prepared"
    target_args+=(--policy.path="$KOURO_POLICY_PRETRAINED_PATH")
}

kouro_policy_append_train_args() {
    local -n target_args=$1
    target_args+=(
        --policy.input_features=null
        --policy.dtype="$KOURO_PI05_DTYPE"
        --policy.chunk_size="$KOURO_PI05_CHUNK_SIZE"
        --policy.n_action_steps="$KOURO_PI05_N_ACTION_STEPS"
        --policy.empty_cameras="$KOURO_PI05_EMPTY_CAMERAS"
        --policy.gradient_checkpointing="$KOURO_PI05_GRADIENT_CHECKPOINTING"
        --policy.train_expert_only="$KOURO_PI05_TRAIN_EXPERT_ONLY"
        --policy.freeze_vision_encoder="$KOURO_PI05_FREEZE_VISION_ENCODER"
        --policy.compile_model="$KOURO_PI05_COMPILE_MODEL"
    )
}

kouro_policy_append_required_modules() {
    local -n target_modules=$1
    target_modules+=(safetensors transformers)
}

kouro_policy_write_effective_config() {
    printf 'KOURO_PI05_PRETRAINED_SOURCE=%q\n' "$KOURO_PI05_RESOLVED_PRETRAINED_SOURCE"
    printf 'KOURO_PI05_PRETRAINED_LOCAL_PATH=%q\n' "$KOURO_POLICY_PRETRAINED_PATH"
    printf 'KOURO_PI05_TOKENIZER_REPO_CACHE=%q\n' "$KOURO_PI05_TOKENIZER_REPO_CACHE"
    printf 'KOURO_PI05_DTYPE=%q\n' "$KOURO_PI05_DTYPE"
    printf 'KOURO_PI05_CHUNK_SIZE=%q\n' "$KOURO_PI05_CHUNK_SIZE"
    printf 'KOURO_PI05_N_ACTION_STEPS=%q\n' "$KOURO_PI05_N_ACTION_STEPS"
    printf 'KOURO_PI05_EMPTY_CAMERAS=%q\n' "$KOURO_PI05_EMPTY_CAMERAS"
    printf 'KOURO_PI05_GRADIENT_CHECKPOINTING=%q\n' "$KOURO_PI05_GRADIENT_CHECKPOINTING"
    printf 'KOURO_PI05_TRAIN_EXPERT_ONLY=%q\n' "$KOURO_PI05_TRAIN_EXPERT_ONLY"
    printf 'KOURO_PI05_FREEZE_VISION_ENCODER=%q\n' "$KOURO_PI05_FREEZE_VISION_ENCODER"
    printf 'KOURO_PI05_COMPILE_MODEL=%q\n' "$KOURO_PI05_COMPILE_MODEL"
}
