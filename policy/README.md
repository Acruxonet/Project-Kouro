# Policy 工作区

这里保存会被训练实际加载的 policy 核心代码。启动脚本不会直接使用共享盘 LeRobot
中的 Diffusion Policy：它先将 LeRobot 源码复制到容器本地 SSD，再使用
[`diffusion`](diffusion) 中的文件覆盖本地副本。修改任一文件都会改变代码指纹，下一次
启动会自动创建新的本地代码缓存。

## Diffusion Policy 文件

- `configuration_diffusion.py`：观测/动作 horizon、归一化模式、视觉 backbone、U-Net、
  diffusion scheduler、优化器和 loss 开关。
- `modeling_diffusion.py`：ResNet 编码器、Spatial Softmax、Conditional U-Net、采样逻辑，
  以及 `DiffusionModel.compute_loss()`。
- `processor_diffusion.py`：训练前的 device/rename/normalize pipeline 和 rollout 后的
  action unnormalize pipeline。
- `normalization.py`：训练实际使用的完整归一化与反归一化实现。缓存构建时它会覆盖
  `lerobot/processor/normalize_processor.py`。

当前 baseline 的数值路径是：

```text
dataset sample
  -> image: per-camera dataset mean/std（ResNet 权重仍由 ImageNet 初始化）
  -> source state16 quaternion -> local cached state20 rotation 6D
  -> state20: 2 * (x - min) / (max - min) - 1
  -> action: 2 * (a - min) / (max - min) - 1
  -> DiffusionModel.compute_loss(): MSE(predicted_noise, sampled_noise)
  -> sampled normalized action
  -> action: (a_norm + 1) / 2 * (max - min) + min
  -> RoboCasa
```

这里的 20D state 是
`base_pos3 + base_rot6d6 + ee_pos3 + ee_rot6d6 + gripper2`。12D action 中的
`ee_rotation_delta3` 是控制器增量，不是四元数，因此 action 的维数和语义保持不变。
具体 train/eval 转换边界见 [`../environment/README.md`](../environment/README.md)。

默认使用 `epsilon` prediction。要改架构或 loss，优先查看
`modeling_diffusion.py` 的 `DiffusionModel` 和 `compute_loss()`；要改统计量或缩放公式，
直接修改 `normalization.py`；要改模式映射则修改 `configuration_diffusion.py`。

这些文件来自本地 LeRobot `5c98e80430d4a747926b45893568e388105a2400`，保留原文件的
Apache-2.0 版权头。每个训练实验还会在自己的
`experiments/<Policy>_<Task>_<Variant>/policy_snapshot/` 保存一份冻结快照，供旧
checkpoint 评测复现。

## 切换或新增 policy

每个 policy 占用一个直接子目录，目录名与 LeRobot `policy.type` 一致：

```text
policy/
├── diffusion/
├── act/
├── pi05/
└── <new_policy>/
```

实验 `config.env` 只用下面两项选择核心代码：

```bash
: "${KOURO_POLICY_TYPE:=diffusion}"
: "${KOURO_POLICY_DIR:=policy/diffusion}"
```

启动时会自动用该目录替换本地 LeRobot 缓存的
`lerobot/policies/<type>/`，并在实验中保存 `policy_snapshot/`。如果目录含
`normalization.py`，它还会覆盖训练实际使用的 LeRobot normalize processor。
使用 `policy.type` 从头训练时，launcher 会传入 `policy.discover_packages_path`，因此新
policy 只要按 LeRobot 约定在 configuration 中注册 `PreTrainedConfig`，并提供对应的
modeling 和 processor，不需要再修改 LeRobot 中央 factory。使用 `policy.path` 加载预训练
checkpoint 时不传 discovery 参数：LeRobot 会将 `policy.*` CLI 参数作为 checkpoint config
override，再传 discovery 会被错误地解析成未知字段。

如果原始 policy 还依赖 `lerobot/policies/` 下的同级文件，将其保存在
`policy/<type>/_shared/`。缓存脚本会将 `_shared` 中的文件按原层级安装到
`lerobot/policies/`，但不会把 `_shared` 暴露成 policy 自身的 Python 子包。

通用训练脚本不包含任何 Diffusion 专属参数。Policy 特有的 horizon、backbone、
依赖和 artifact 缓存等放在
[`scripts/policy_adapters`](../scripts/policy_adapters)。Adapter 可选；不需要特殊参数时，
只新增 policy 目录和修改实验中两项配置即可。

为新 policy 创建一个新的扁平实验目录后，通用 TurnOffSinkFaucet launcher 用
`KOURO_EXPERIMENT` 选择它：

```bash
KOURO_EXPERIMENT=<Policy>_TurnOffSinkFaucet_<Variant> \
KOURO_GPU=0 bash scripts/tmux_train_turnoff_sink_faucet.sh
```

未设置 `KOURO_EXPERIMENT` 时仍默认运行 `DP_TurnOffSinkFaucet_baseline`，因此现有 baseline
命令保持不变。

## 已加入的原始 policy

`act/` 保存原始 LeRobot ACT 的 configuration、modeling 和 processor。adapter 的模型
默认值仍是 `chunk_size=100`、`n_action_steps=100`、ResNet-18 ImageNet backbone、VAE 和
`kl_weight=10`；当前 ACT baseline 在它自己的 `config.env` 中显式覆盖为 `64/32`。
ACT 从头训练，没有强制的 policy checkpoint。

`pi05/` 保存原始 LeRobot Pi05 的 configuration、modeling 和 processor，以及它直接
依赖的原始 `pi_gemma.py` 和 `rtc/` 源码。Pi05 adapter 默认从本机已有的
`lerobot/pi05_base` pinned revision 开始 fine-tune，并自动将权重和 PaliGemma tokenizer
缓存到容器本地盘。权重盘点见
[`artifacts/pretrained/pi05/README.md`](../artifacts/pretrained/pi05/README.md)。

Pi05 从通用 base 适配到 RoboCasa 时，adapter 会设置 `input_features=null`，
让 LeRobot 从当前数据推断三路图像和 20D rotation-6D state；12D action 也会从数据
重建。内部 `max_state_dim=max_action_dim=32` 足以容纳这两个维度。

Pi05 原始配置对 state/action 使用 q01/q99 quantile normalization。本地 rotation-6D 数据
缓存会为前 100 个 episode 同时重算 `min/max/mean/std/q01/q10/q50/q90/q99`，因此
Diffusion、ACT 和 Pi05 可以共用同一份缓存而各自选择不同 normalization mode。

三套 policy 源码均来自本地 LeRobot commit
`5c98e80430d4a747926b45893568e388105a2400`，ACT/Pi05 复制时对应源路径无未提交修改。
Pi05 在本项目内额外将 pretrained loader 改为 fail-closed：任何权重解析或 strict
`state_dict` 加载错误都会终止训练，而不是回退到随机初始化。
