#!/usr/bin/env bash

# Diffusion-specific validation, local artifacts and LeRobot CLI arguments.
# This file is sourced by the task launcher after the experiment config.

kouro_policy_validate_config() {
    local numeric_name max_action_steps
    for numeric_name in KOURO_N_OBS_STEPS KOURO_HORIZON KOURO_N_ACTION_STEPS; do
        [[ "${!numeric_name}" =~ ^[1-9][0-9]*$ ]] || \
            kouro_die "$numeric_name must be a positive integer"
    done
    [[ "$KOURO_DROP_N_LAST_FRAMES" =~ ^[0-9]+$ ]] || \
        kouro_die "KOURO_DROP_N_LAST_FRAMES must be a non-negative integer"
    max_action_steps=$((KOURO_HORIZON - KOURO_N_OBS_STEPS + 1))
    [ "$max_action_steps" -ge 1 ] || kouro_die "Invalid horizon/observation combination"
    [ "$KOURO_N_ACTION_STEPS" -le "$max_action_steps" ] || \
        kouro_die "KOURO_N_ACTION_STEPS cannot exceed $max_action_steps for this horizon configuration"
}

kouro_policy_prepare_local_artifacts() {
    local project_root=$1 cache_root=$2 resnet_artifact resnet_cache
    resnet_artifact="$project_root/artifacts/pretrained/resnet18-f37072fd.pth"
    resnet_cache="$cache_root/torch/hub/checkpoints/resnet18-f37072fd.pth"
    if [ "$KOURO_PRETRAINED_BACKBONE" != null ] && [ ! -f "$resnet_cache" ]; then
        if [ -f "$resnet_artifact" ]; then
            kouro_log "Copying bundled ResNet-18 weights to local Torch cache..."
            cp -p -- "$resnet_artifact" "$resnet_cache"
        else
            kouro_log "Bundled ResNet weights are absent; TorchVision will download them on first use."
        fi
    fi
}

kouro_policy_append_train_args() {
    local -n target_args=$1
    target_args+=(
        --policy.pretrained_backbone_weights="$KOURO_PRETRAINED_BACKBONE"
        --policy.n_obs_steps="$KOURO_N_OBS_STEPS"
        --policy.horizon="$KOURO_HORIZON"
        --policy.n_action_steps="$KOURO_N_ACTION_STEPS"
        --policy.drop_n_last_frames="$KOURO_DROP_N_LAST_FRAMES"
    )
}

kouro_policy_append_required_modules() {
    local -n target_modules=$1
    target_modules+=(diffusers torchvision)
}

kouro_policy_write_effective_config() {
    printf 'KOURO_PRETRAINED_BACKBONE=%q\n' "$KOURO_PRETRAINED_BACKBONE"
    printf 'KOURO_N_OBS_STEPS=%q\n' "$KOURO_N_OBS_STEPS"
    printf 'KOURO_HORIZON=%q\n' "$KOURO_HORIZON"
    printf 'KOURO_N_ACTION_STEPS=%q\n' "$KOURO_N_ACTION_STEPS"
    printf 'KOURO_DROP_N_LAST_FRAMES=%q\n' "$KOURO_DROP_N_LAST_FRAMES"
}
