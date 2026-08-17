#!/usr/bin/env bash

# Train a configured TurnOffSinkFaucet experiment. This script never creates
# Docker; run it inside the GPU container that has the shared NAS mounted.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/lib/common.sh"
project_root=$(kouro_project_root)
experiment_dir=$(kouro_experiment_dir)
source "$experiment_dir/config.env"

dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=true
elif [ "$#" -gt 0 ]; then
    kouro_die "Usage: $0 [--dry-run]"
fi

[ "$KOURO_TASK" = TurnOffSinkFaucet ] || kouro_die "This launcher is locked to TurnOffSinkFaucet"
[ "$KOURO_STATE_ROTATION_REPRESENTATION" = rotation_6d ] || \
    kouro_die "This experiment requires KOURO_STATE_ROTATION_REPRESENTATION=rotation_6d"
run_id=$KOURO_EXPERIMENT_ID
kouro_validate_id "$run_id" run_id
[ "$run_id" = "$(basename -- "$experiment_dir")" ] || \
    kouro_die "KOURO_EXPERIMENT_ID must match its directory name: $(basename -- "$experiment_dir")"
policy_type=$KOURO_POLICY_TYPE
kouro_validate_id "$policy_type" policy_type
policy_dir=$(kouro_resolve_policy_dir "$policy_type" "${KOURO_POLICY_DIR:-}")

# Policies with no adapter use LeRobot defaults plus the common flags below.
kouro_policy_validate_config() { :; }
kouro_policy_prepare_local_artifacts() { :; }
kouro_policy_append_selector_args() {
    local -n target_args=$1
    target_args+=(--policy.type="$policy_type")
}
kouro_policy_append_train_args() { :; }
kouro_policy_append_required_modules() { :; }
kouro_policy_write_effective_config() { :; }
policy_adapter="$script_dir/policy_adapters/$policy_type.sh"
if [ -f "$policy_adapter" ]; then
    source "$policy_adapter"
fi
kouro_policy_validate_config

python_bin=$(kouro_resolve_python)
cache_root=$(kouro_cache_root)
data_root=$("$script_dir/cache/ensure_task_cache.sh" "$KOURO_TASK" "$KOURO_NUM_EPISODES")
code_root=$(KOURO_POLICY_TYPE="$policy_type" KOURO_POLICY_DIR="$policy_dir" \
    "$script_dir/cache/ensure_code_cache.sh")

mkdir -p \
    "$cache_root/huggingface" \
    "$cache_root/torch/hub/checkpoints" \
    "$cache_root/numba" \
    "$cache_root/xdg" \
    "$cache_root/tmp"

kouro_policy_prepare_local_artifacts "$project_root" "$cache_root" "$dry_run"

export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="$cache_root/huggingface"
export HF_DATASETS_CACHE="$cache_root/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$cache_root/huggingface/hub"
export TRANSFORMERS_CACHE="$cache_root/huggingface/transformers"
export TORCH_HOME="$cache_root/torch"
export NUMBA_CACHE_DIR="$cache_root/numba"
export XDG_CACHE_HOME="$cache_root/xdg"
export TMPDIR="$cache_root/tmp"
export TOKENIZERS_PARALLELISM=false
if [ -n "${KOURO_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$KOURO_GPU"
fi

episode_list='['
for ((episode_index = 0; episode_index < KOURO_NUM_EPISODES; episode_index++)); do
    if [ "$episode_index" -gt 0 ]; then episode_list+=','; fi
    episode_list+="$episode_index"
done
episode_list+=']'

checkpoint_dir="$project_root/checkpoints/robocasa/$KOURO_TASK/$policy_type/$run_id"
tensorboard_dir="$experiment_dir/tensorboard"

policy_selector_args=()
kouro_policy_append_selector_args policy_selector_args
policy_train_args=()
kouro_policy_append_train_args policy_train_args

train_command=(
    "$python_bin" -m lerobot.scripts.lerobot_train
    --policy.discover_packages_path="lerobot.policies.$policy_type"
    "${policy_selector_args[@]}"
    --policy.device=cuda
    --policy.use_amp="$KOURO_USE_AMP"
    --policy.push_to_hub=false
    "${policy_train_args[@]}"
    --dataset.repo_id=project_kouro/robocasa_TurnOffSinkFaucet
    --dataset.root="$data_root"
    --dataset.episodes="$episode_list"
    --dataset.video_backend=pyav
    --env.type=robocasa
    --env.task=TurnOffSinkFaucet
    --env.split="$KOURO_EVAL_SPLIT"
    '--env.obj_registries=[objaverse,lightwheel]'
    --output_dir="$checkpoint_dir"
    --job_name="$run_id"
    --steps="$KOURO_STEPS"
    --batch_size="$KOURO_BATCH_SIZE"
    --num_workers="$KOURO_NUM_WORKERS"
    --prefetch_factor="$KOURO_PREFETCH_FACTOR"
    --persistent_workers=true
    --log_freq="$KOURO_LOG_FREQ"
    --save_freq="$KOURO_SAVE_FREQ"
    --eval_freq="$KOURO_EVAL_FREQ"
    --eval.batch_size=1
    --eval.n_episodes="$KOURO_INLINE_EVAL_EPISODES"
    --eval.use_async_envs=false
    --seed="$KOURO_SEED"
    --wandb.enable=false
)

if $dry_run; then
    kouro_log "Dry run complete. Caches are ready; no training was started."
    printf 'PYTHONPATH=%q ' "$PYTHONPATH"
    printf '%q ' "${train_command[@]}"
    printf '\n'
    exit 0
fi

kouro_log "Preparing local RoboCasa source/assets cache for periodic evaluation..."
robocasa_root=$("$script_dir/cache/ensure_robocasa_cache.sh")
python_overlay=$("$script_dir/cache/ensure_robocasa_python_overlay.sh")
export PYTHONPATH="$python_overlay:$code_root:$robocasa_root/third_party/robosuite:$robocasa_root/third_party/robocasa${PYTHONPATH:+:$PYTHONPATH}"

[ ! -e "$checkpoint_dir" ] || kouro_die "Checkpoint directory already exists: $checkpoint_dir"
[ -f "$experiment_dir/README.md" ] || kouro_die "Experiment README is missing: $experiment_dir/README.md"
[ ! -e "$experiment_dir/status.env" ] || kouro_die "This experiment has already been launched: $experiment_dir"
[ ! -e "$experiment_dir/policy_snapshot" ] || kouro_die "Policy snapshot already exists: $experiment_dir"

kouro_log "Running Python/GPU preflight..."
required_modules=(accelerate av tensorboard torch)
kouro_policy_append_required_modules required_modules
"$python_bin" - "$code_root" "$policy_type" "${required_modules[@]}" <<'PY'
import importlib.util
import pathlib
import sys

code_root = pathlib.Path(sys.argv[1]).resolve()
policy_type = sys.argv[2]
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version}")
for module in sys.argv[3:]:
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Missing Python package: {module}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this container")
policy_spec = importlib.util.find_spec(f"lerobot.policies.{policy_type}")
if policy_spec is None:
    raise SystemExit(f"Policy package is not importable: lerobot.policies.{policy_type}")
if policy_spec.submodule_search_locations:
    policy_path = pathlib.Path(next(iter(policy_spec.submodule_search_locations))).resolve()
elif policy_spec.origin:
    policy_path = pathlib.Path(policy_spec.origin).resolve()
else:
    raise SystemExit(f"Cannot resolve policy package path: {policy_type}")
if code_root not in policy_path.parents:
    raise SystemExit(f"Policy was not loaded from the local cache: {policy_path}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
print(f"Project policy ({policy_type}): {policy_path}")
PY

mkdir -p "$(dirname -- "$checkpoint_dir")"
mkdir -p "$experiment_dir/policy_snapshot"
cp -a -- "$policy_dir/." "$experiment_dir/policy_snapshot/"
if [ -f "$policy_adapter" ]; then
    cp -a -- "$policy_adapter" "$experiment_dir/policy_adapter_snapshot.sh"
fi
mkdir -p "$experiment_dir/environment_snapshot"
cp -a -- "$project_root/environment/"*.py "$experiment_dir/environment_snapshot/"
command_snapshot=$(printf '%q ' "${train_command[@]}")
{
    printf '\n## 本次实际展开命令\n\n'
    printf -- '- 启动时间：`%s`\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf -- '- Dataset cache：`%s`\n' "$data_root"
    printf -- '- Code cache：`%s`\n' "$code_root"
    printf -- '- Policy：`%s` from `%s`\n' "$policy_type" "$policy_dir"
    printf -- '- Checkpoint：`%s`\n\n' "$checkpoint_dir"
    printf '```bash\n%s\n```\n' "$command_snapshot"
} >> "$experiment_dir/README.md"

{
    printf 'KOURO_EXPERIMENT_ID=%q\n' "$run_id"
    printf 'KOURO_TASK=%q\n' "$KOURO_TASK"
    printf 'KOURO_POLICY_TYPE=%q\n' "$policy_type"
    printf 'KOURO_POLICY_DIR=%q\n' "$policy_dir"
    printf 'KOURO_STEPS=%q\n' "$KOURO_STEPS"
    printf 'KOURO_BATCH_SIZE=%q\n' "$KOURO_BATCH_SIZE"
    printf 'KOURO_NUM_WORKERS=%q\n' "$KOURO_NUM_WORKERS"
    printf 'KOURO_SEED=%q\n' "$KOURO_SEED"
    printf 'KOURO_STATE_ROTATION_REPRESENTATION=%q\n' \
        "$KOURO_STATE_ROTATION_REPRESENTATION"
    printf 'KOURO_EVAL_FREQ=%q\n' "$KOURO_EVAL_FREQ"
    printf 'KOURO_INLINE_EVAL_EPISODES=%q\n' "$KOURO_INLINE_EVAL_EPISODES"
    printf 'KOURO_FINAL_EVAL_EPISODES=%q\n' "$KOURO_FINAL_EVAL_EPISODES"
    kouro_policy_write_effective_config
    printf 'KOURO_CACHE_ROOT=%q\n' "$cache_root"
    printf 'DATA_ROOT=%q\n' "$data_root"
    printf 'CODE_ROOT=%q\n' "$code_root"
    printf 'ROBOCASA_ROOT=%q\n' "$robocasa_root"
    printf 'CHECKPOINT_DIR=%q\n' "$checkpoint_dir"
} > "$experiment_dir/effective_config.env"

exit_code=1
write_status() {
    printf 'exit_code=%s\nfinished_at=%s\n' "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "$experiment_dir/status.env"
}
trap write_status EXIT
printf 'exit_code=running\nstarted_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$experiment_dir/status.env"

kouro_log "Starting training run: $run_id"
kouro_log "TensorBoard logdir: $tensorboard_dir"
set +e
"$python_bin" "$script_dir/tensorboard_logger.py" \
    --log-file "$experiment_dir/train.log" \
    --log-dir "$tensorboard_dir" \
    -- "${train_command[@]}"
exit_code=$?
set -e
exit "$exit_code"
