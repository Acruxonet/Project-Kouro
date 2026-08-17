# Policy adapters

通用训练脚本只负责 dataset、environment、checkpoint、TensorBoard 和 LeRobot 的公共参数。
某个 policy 独有的参数与依赖放在同名 adapter：

```text
policy/<type>/                    # 可直接修改的 policy 核心 Python 代码
scripts/policy_adapters/<type>.sh # 少量训练参数、依赖和本地 artifact 逻辑
```

Adapter 是可选的。没有 adapter 时，训练器传入 `--policy.type`、device、AMP 和
`push_to_hub` 等公共参数，其余使用该 policy 的 LeRobot 默认配置。

需要定制时，可以实现以下 hook；未实现的 hook 都是 no-op：

- `kouro_policy_validate_config`：检查 policy 专属实验配置；
- `kouro_policy_prepare_local_artifacts`：准备预训练权重等本地缓存；
- `kouro_policy_prepare_eval_artifacts`：在新的 eval 容器中补齐 tokenizer 等运行资产；
- `kouro_policy_append_selector_args`：选择 `--policy.type` 或预训练的 `--policy.path`；
- `kouro_policy_append_train_args`：追加 policy 专属 LeRobot CLI 参数；
- `kouro_policy_append_required_modules`：追加训练前依赖检查；
- `kouro_policy_write_effective_config`：记录 policy 专属的实际配置。

现有 adapter 分别覆盖 Diffusion、ACT 和 Pi05。Pi05 使用 selector hook 从本机已有
checkpoint fine-tune，并用 train/eval artifact hook 在每个临时 Docker 中重建本地 SSD
缓存。新增 policy 不需要修改通用训练、eval 或代码缓存脚本。
