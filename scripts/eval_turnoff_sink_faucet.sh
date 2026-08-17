#!/usr/bin/env bash

# Evaluate one completed TurnOffSinkFaucet checkpoint in RoboCasa.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
project_root=$(kouro_project_root)
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

run_id=$KOURO_EXPERIMENT_ID
[ "$run_id" = "$(basename -- "$experiment_dir")" ] || \
    kouro_die "KOURO_EXPERIMENT_ID must match its directory name: $(basename -- "$experiment_dir")"
policy_type=$KOURO_POLICY_TYPE
kouro_validate_id "$policy_type" policy_type
[ "$KOURO_STATE_ROTATION_REPRESENTATION" = rotation_6d ] || \
    kouro_die "This experiment requires KOURO_STATE_ROTATION_REPRESENTATION=rotation_6d"
kouro_validate_id "$run_id" experiment_id
dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=true
elif [ "$#" -gt 0 ]; then
    kouro_die "Usage: $0 [--dry-run]"
fi

checkpoint_dir="$project_root/checkpoints/robocasa/$KOURO_TASK/$policy_type/$run_id"
policy_checkpoint="$checkpoint_dir/checkpoints/last/pretrained_model"
policy_snapshot="$experiment_dir/policy_snapshot"
policy_adapter_snapshot="$experiment_dir/policy_adapter_snapshot.sh"
environment_snapshot="$experiment_dir/environment_snapshot"
[ -d "$experiment_dir" ] || kouro_die "Experiment is missing: $experiment_dir"
[ -f "$policy_checkpoint/config.json" ] || kouro_die "Final checkpoint is missing: $policy_checkpoint"
[ -d "$policy_snapshot" ] || kouro_die "Training policy snapshot is missing: $policy_snapshot"
[ -d "$environment_snapshot" ] || \
    kouro_die "Training environment snapshot is missing: $environment_snapshot"

python_bin=$(kouro_resolve_python)
cache_root=$(kouro_cache_root)
code_root=$(KOURO_POLICY_TYPE="$policy_type" KOURO_POLICY_DIR="$policy_snapshot" \
    KOURO_ENVIRONMENT_DIR="$environment_snapshot" "$script_dir/cache/ensure_code_cache.sh")
robocasa_root=$("$script_dir/cache/ensure_robocasa_cache.sh")
python_overlay=$("$script_dir/cache/ensure_robocasa_python_overlay.sh")

mkdir -p \
    "$cache_root/huggingface" \
    "$cache_root/torch" \
    "$cache_root/numba" \
    "$cache_root/xdg" \
    "$cache_root/tmp"

kouro_policy_prepare_eval_artifacts() { :; }
if [ -f "$policy_adapter_snapshot" ]; then
    source "$policy_adapter_snapshot"
elif [ -f "$script_dir/policy_adapters/$policy_type.sh" ]; then
    source "$script_dir/policy_adapters/$policy_type.sh"
fi
kouro_policy_prepare_eval_artifacts "$project_root" "$cache_root"

export PYTHONPATH="$python_overlay:$code_root:$robocasa_root/third_party/robosuite:$robocasa_root/third_party/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$cache_root/huggingface/hub"
export TRANSFORMERS_CACHE="$cache_root/huggingface/transformers"
export TORCH_HOME="$cache_root/torch"
export NUMBA_CACHE_DIR="$cache_root/numba"
export XDG_CACHE_HOME="$cache_root/xdg"
export TMPDIR="$cache_root/tmp"
export MUJOCO_GL=${MUJOCO_GL:-egl}
if [ -n "${KOURO_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$KOURO_GPU"
fi

eval_id=$(date -u +%Y%m%d-%H%M%S)
eval_dir="$experiment_dir/eval/$eval_id"
eval_command=(
    "$python_bin" -m lerobot.scripts.lerobot_eval
    --policy.path="$policy_checkpoint"
    --policy.device=cuda
    --env.type=robocasa
    --env.task=TurnOffSinkFaucet
    --env.split="$KOURO_EVAL_SPLIT"
    '--env.obj_registries=[objaverse,lightwheel]'
    --eval.batch_size=1
    --eval.n_episodes="$KOURO_FINAL_EVAL_EPISODES"
    --eval.use_async_envs=false
    --output_dir="$eval_dir"
    --job_name="eval_$run_id"
    --seed="$KOURO_SEED"
)

if $dry_run; then
    kouro_log "Evaluation dry run complete. Caches are ready; no rollout was started."
    printf 'PYTHONPATH=%q ' "$PYTHONPATH"
    printf '%q ' "${eval_command[@]}"
    printf '\n'
    exit 0
fi

kouro_log "Running RoboCasa/CUDA preflight..."
"$python_bin" - <<'PY'
import torch
import mujoco
import robocasa
import robosuite
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this container")
if mujoco.__version__ != "3.3.1":
    raise SystemExit(f"Expected MuJoCo 3.3.1, got {mujoco.__version__}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"MuJoCo: {mujoco.__version__}")
print(f"RoboSuite: {robosuite.__version__}")
print(f"RoboCasa: {getattr(robocasa, '__version__', 'unknown')}")
PY

mkdir -p "$eval_dir"
eval_command_snapshot=$(printf '%q ' "${eval_command[@]}")
eval_command_snapshot=${eval_command_snapshot% }
{
    printf '# Evaluation %s\n\n' "$eval_id"
    printf -- '- Training run: `%s`\n' "$run_id"
    printf -- '- Policy: `%s`\n' "$policy_type"
    printf -- '- Task / split: `%s / %s`\n' "$KOURO_TASK" "$KOURO_EVAL_SPLIT"
    printf -- '- Episodes: `%s`\n' "$KOURO_FINAL_EVAL_EPISODES"
    printf -- '- Checkpoint: `%s`\n\n' "$policy_checkpoint"
    printf '```bash\n%s\n```\n' "$eval_command_snapshot"
} > "$eval_dir/README.md"

set +e
"${eval_command[@]}" 2>&1 | tee "$eval_dir/eval.log"
exit_code=${PIPESTATUS[0]}
set -e
printf 'exit_code=%s\nfinished_at=%s\n' "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$eval_dir/status.env"
if [ "$exit_code" -eq 0 ] && [ -f "$eval_dir/eval_info.json" ]; then
    "$python_bin" "$script_dir/log_eval_to_tensorboard.py" \
        "$eval_dir/eval_info.json" "$experiment_dir/tensorboard" --step "$KOURO_STEPS"
fi
exit "$exit_code"
