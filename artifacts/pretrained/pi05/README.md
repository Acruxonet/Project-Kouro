# Local Pi05 checkpoints

Pi05 大权重保留在共享 Hugging Face cache，不在 Project-Kouro 内再复制一份。

## 默认 fine-tuning 起点

- Hugging Face ID：`lerobot/pi05_libero`
- revision：`a217bfd3b14673cf2ce597e69997ab21866438dd`
- snapshot：
  `/mnt/data/nas/hufangchi/cache/huggingface/hub/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd`
- `model.safetensors`：14,467,165,872 bytes（约 13.5 GiB / `du` 约 14G）。
- 用途：LeRobot model card 标记为需要在目标数据上 fine-tune 的 Pi05 base model。

`scripts/policy_adapters/pi05.sh` 默认选择这个 revision。实际训练前，
`ensure_pretrained_model_cache.sh` 会将它解引用复制到当前容器的本地 SSD，
避免训练时从 NAS 持续读取。`--dry-run` 只校验该路径，不复制 14G 权重。

## 已存在但不默认使用

- Hugging Face ID：`lerobot/pi05_libero_finetuned`
- revision：`dbf8a3f794a9c4297b44f40b752712f50073d945`
- 权重：7,473,096,344 bytes（约 7.0G）。
- 用途：LIBERO 任务的已微调模型；它的 state/action 是 8D/7D，不应冒充
  RoboCasa 20D/12D baseline 的通用预训练起点。

## Tokenizer

`google/paligemma-3b-pt-224` tokenizer revision
`35e4f46485b4d07967e7e9935bc3786aad50687c` 也已完整存在共享 cache，大小约
17 MiB。Pi05 adapter 会将它自动 seed 到容器本地 Hugging Face cache。
