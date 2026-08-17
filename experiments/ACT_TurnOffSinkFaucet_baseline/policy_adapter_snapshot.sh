#!/usr/bin/env bash

# Original LeRobot ACT defaults exposed as experiment-level overrides.

: "${KOURO_ACT_CHUNK_SIZE:=100}"
: "${KOURO_ACT_N_ACTION_STEPS:=100}"
: "${KOURO_ACT_VISION_BACKBONE:=resnet18}"
: "${KOURO_ACT_PRETRAINED_BACKBONE:=ResNet18_Weights.IMAGENET1K_V1}"
: "${KOURO_ACT_USE_VAE:=true}"
: "${KOURO_ACT_KL_WEIGHT:=10.0}"

kouro_policy_validate_config() {
    [[ "$KOURO_ACT_CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]] || \
        kouro_die "KOURO_ACT_CHUNK_SIZE must be positive"
    [[ "$KOURO_ACT_N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]] || \
        kouro_die "KOURO_ACT_N_ACTION_STEPS must be positive"
    [ "$KOURO_ACT_N_ACTION_STEPS" -le "$KOURO_ACT_CHUNK_SIZE" ] || \
        kouro_die "ACT n_action_steps cannot exceed chunk_size"
    case "$KOURO_ACT_USE_VAE" in true|false) ;; *) kouro_die "KOURO_ACT_USE_VAE must be true or false" ;; esac
}

kouro_policy_prepare_local_artifacts() {
    local project_root=$1 cache_root=$2 resnet_artifact resnet_cache
    resnet_artifact="$project_root/artifacts/pretrained/resnet18-f37072fd.pth"
    resnet_cache="$cache_root/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if [ "$KOURO_ACT_PRETRAINED_BACKBONE" != null ] && [ ! -f "$resnet_cache" ]; then
        if [ -f "$resnet_artifact" ]; then
            kouro_log "Copying bundled ResNet-18 weights to local Torch cache for ACT..."
            cp -p -- "$resnet_artifact" "$resnet_cache"
        else
            kouro_log "Bundled ResNet weights are absent; TorchVision may download them on first use."
        fi
    fi
}

kouro_policy_append_train_args() {
    local -n target_args=$1
    target_args+=(
        --policy.chunk_size="$KOURO_ACT_CHUNK_SIZE"
        --policy.n_action_steps="$KOURO_ACT_N_ACTION_STEPS"
        --policy.vision_backbone="$KOURO_ACT_VISION_BACKBONE"
        --policy.pretrained_backbone_weights="$KOURO_ACT_PRETRAINED_BACKBONE"
        --policy.use_vae="$KOURO_ACT_USE_VAE"
        --policy.kl_weight="$KOURO_ACT_KL_WEIGHT"
    )
}

kouro_policy_append_required_modules() {
    local -n target_modules=$1
    target_modules+=(torchvision)
}

kouro_policy_write_effective_config() {
    printf 'KOURO_ACT_CHUNK_SIZE=%q\n' "$KOURO_ACT_CHUNK_SIZE"
    printf 'KOURO_ACT_N_ACTION_STEPS=%q\n' "$KOURO_ACT_N_ACTION_STEPS"
    printf 'KOURO_ACT_VISION_BACKBONE=%q\n' "$KOURO_ACT_VISION_BACKBONE"
    printf 'KOURO_ACT_PRETRAINED_BACKBONE=%q\n' "$KOURO_ACT_PRETRAINED_BACKBONE"
    printf 'KOURO_ACT_USE_VAE=%q\n' "$KOURO_ACT_USE_VAE"
    printf 'KOURO_ACT_KL_WEIGHT=%q\n' "$KOURO_ACT_KL_WEIGHT"
}
