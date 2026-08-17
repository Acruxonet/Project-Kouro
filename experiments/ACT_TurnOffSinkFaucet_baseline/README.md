# ACT TurnOffSinkFaucet baseline

## 实验目的

在与 Diffusion baseline 相同的 RoboCasa 数据、rotation-6D state 和评测协议上训练原始
LeRobot ACT，得到 action-chunking transformer baseline。这里保留原始 CVAE、Transformer
和 L1+KL loss，不加入 temporal ensemble，也不修改网络结构。

这个实验用于回答：在同一套三相机 EEF+Delta 数据上，直接回归一个长 action chunk 的 ACT，
与迭代生成 action trajectory 的 Diffusion Policy 相比，收敛速度、长时 open-loop 稳定性和
最终成功率分别如何。

## 任务与评测协议

- 环境：RoboCasa365 / PandaOmron。
- 任务族：Atomic。
- 任务名：`TurnOffSinkFaucet`。
- 训练数据 split：`pretrain`；RoboCasa eval split：`pretrain`。
- 评测 object registries：`[objaverse, lightwheel]`。
- 训练中评测：每 10,000 steps 做 5 episodes。
- 最终评测：训练结束后自动做 50 episodes。

任务目标是关闭正在出水的水龙头。本实验使用前 100 个 episode，共 11,608 frames；ACT
不配置 `drop_n_last_frames`，所以每一帧都可作为训练 anchor。100k steps、batch size 8 约为
69 次 dataset pass。episode 尾部不足 100 步的 action chunk 由 dataset loader 复制最后一个
action padding，但 ACT loss 会用 `action_is_pad` 排除这些位置。

## 数据

NAS 上的权威源数据：

```text
data/robocasa365/v1.0/pretrain/atomic/TurnOffSinkFaucet/20250819/lerobot
```

- 格式：LeRobot v3.0，20 Hz。
- 原始数据：106 episodes / 12,309 frames。
- 本实验：episode index `0..99`，100 episodes / 11,608 frames。
- 图像：三个 256×256 RGB 视角：left、eye-in-hand、right。
- 语言：dataset 含 `Turn off the sink faucet.`，但 ACT 不读取 task 文本。

### State 与 action 语义

本实验仍是 **EEF + Delta**。权威数据的 16D state 为：

```text
base_position_world(3) + base_quaternion_xyzw_world(4)
+ ee_position_in_base(3) + ee_quaternion_xyzw_in_base(4) + gripper_qpos(2)
```

训练前只在 Docker 本地 cache 中将两段 quaternion 转成 rotation 6D，得到 20D：

```text
base_position_world(3) + base_rotation_6d_world(6)
+ ee_position_in_base(3) + ee_rotation_6d_in_base(6) + gripper_qpos(2)
```

rotation 6D 使用旋转矩阵前两列，顺序为
`[R00,R10,R20,R01,R11,R21]`。环境 eval adapter 做完全相同的转换。

12D action 不转换：

| Slice | 字段 | 语义 |
| --- | --- | --- |
| `[0:4]` | mobile base + torso | 三个 base velocity 和一个 torso position Delta |
| `[4]` | control mode | `-1` arm mode，`+1` base mode |
| `[5:8]` | EEF position Delta | mobile-base frame 中的 `(dx,dy,dz)` |
| `[8:11]` | EEF rotation Delta | mobile-base frame 中的 axis-angle Delta |
| `[11]` | gripper | `-1` open，`+1` close |

本任务中 `[0:5]` 与 `[11]` 都是常量，真正变化的是 `[5:11]` 的 6D EEF Delta；模型仍对
完整 12D action 建模。完整参考系和 controller scaling 见
[`../../data/README.md`](../../data/README.md)。

## Policy 核心代码

实验显式选择：

```bash
KOURO_POLICY_TYPE=act
KOURO_POLICY_DIR=policy/act
```

核心文件分别是：

- [`configuration_act.py`](../../policy/act/configuration_act.py)：feature schema、chunk、
  Transformer/CVAE、normalization 和 optimizer；
- [`modeling_act.py`](../../policy/act/modeling_act.py)：共享 ResNet、CVAE encoder、主
  encoder/decoder、action head 与 loss；
- [`processor_act.py`](../../policy/act/processor_act.py)：训练前 normalize 和 rollout 后
  action unnormalize。

源码来自本机 LeRobot commit `5c98e80430d4a747926b45893568e388105a2400`，当前仓库副本
与原文件逐字节一致。ACT policy checkpoint 从头训练；只有 ResNet18 使用本地缓存的
ImageNet-1K 初始化权重。

## 数据流与张量 shape

当前 `batch_size=8`、`n_obs_steps=1`：

| 输入 | Shape | 进入的模块 |
| --- | --- | --- |
| `observation.state` | `(8,20)` | CVAE state projection 与主 encoder state token |
| 三个 `observation.images.*` | 每个 `(8,3,256,256)` | 共享 ResNet18 |
| `action` | `(8,100,12)` | CVAE action tokens 与 L1 target |
| `action_is_pad` | `(8,100)` | CVAE attention mask 与 L1 valid mask |

### 1. Normalize processor

ACT 对 image、state、action 全部使用前 100 episodes 的 dataset `mean/std`：

```text
x_norm = (x - mean) / (std + 1e-8)
x      = x_norm * std + mean
```

ResNet 虽然由 ImageNet 权重初始化，输入 normalization 使用每个 RoboCasa camera 自己的
dataset mean/std，不是固定 ImageNet mean/std。常量 action 维度的 `std=0`，训练值映射到
0，反归一化后恢复对应的 dataset mean。

### 2. 共享视觉 backbone

三个相机逐个经过同一套 ResNet18 权重，而不是三个独立 backbone：

```text
每个 camera: (8,3,256,256)
  -> shared ResNet18, remove avgpool/fc: (8,512,8,8)
  -> shared 1x1 Conv 512->512:          (8,512,8,8)
  -> flatten spatial tokens:            64 tokens × 512D
```

ResNet 中 BatchNorm 被 `FrozenBatchNorm2d` 替换；其 scale/bias/running statistics 是 buffer，
不计入 parameter 数。三路图像一共产生 `3×64=192` 个 visual tokens，并使用无参数的 2D
sin/cos position embedding。

这里的“一个 backbone”是 **多相机共享权重**，不是只读取一个相机，也不是漏建了两个
backbone。`ACT.__init__()` 只建立一次 `self.backbone`，forward 再遍历三路 image，对每一路
调用同一个 `self.backbone`。本机 RoboTwin 中的 ACT/DETR 实现也只构造一个 backbone，
并在各个 camera 上使用 `backbones[0]`。因此本 baseline 的 11.17M 视觉参数只计一份。

这与当前 Diffusion baseline 的设定不同：DP 的
`use_separate_rgb_encoder_per_camera=true`，会为三个相机创建三套独立权重，视觉编码部分
合计 33.59M。如果将 ACT 改成三套独立 ResNet18，那是一个新的 architecture ablation，
而不再是这个 baseline。

### 3. CVAE encoder（仅训练时使用）

CVAE 输入 token 序列为：

```text
[learned CLS, state20->512, 100 × action12->512] = 102 tokens × 512D
```

4 层 VAE Transformer encoder 使用 hidden 512、8 heads、FFN 3200、dropout 0.1。CLS 输出
经 `512 -> 64` linear 拆成 32D `mu` 与 32D `log_sigma_x2`，再用

```text
z = mu + exp(log_sigma_x2 / 2) * epsilon_z,  epsilon_z ~ N(0,I)
```

得到 32D latent。推理时没有 ground-truth action，CVAE encoder 整支不执行，latent 固定为
全零；这正是 ACT 的 conditional VAE 训练/推理边界。

### 4. 主 Transformer encoder

主 encoder 输入为：

```text
[latent32->512, state20->512, 192 visual tokens] = 194 tokens × 512D
```

它同样是 4 层、8 heads、FFN 3200。latent/state 使用两个 learnable 1D position embeddings，
视觉 token 使用 2D sin/cos position embedding。输出是供 decoder cross-attention 使用的
194-token memory。

### 5. Transformer decoder 与 action head

100 个 learnable query embedding 对应 chunk 中的 100 个未来 action 位置。当前实现用 1 层
decoder：query self-attention、对 encoder memory 的 cross-attention、FFN 3200，最后
`Linear(512,12)` 输出 `(B,100,12)`。

配置中的 1 层是 LeRobot 对原始 ACT 行为的兼容选择：原始实现虽然声明 7 层，但历史 bug
实际只使用首层。当前 `temporal_ensemble_coeff=null`；rollout 每次预测 100 步并全部放入
queue，在 20 Hz 下最多 open-loop 执行约 5 秒后再规划。

## 模块与参数量

参数量按三相机、state20、action12、chunk100 的真实模型实例统计。视觉 backbone 在三个相机
之间共享，所以只计算一次。

| 模块 | 结构 | 参数量 |
| --- | --- | ---: |
| CVAE Transformer encoder | 4 × 4,333,184 | 17,332,736 |
| CVAE CLS/state/action/latent projections | 512 + 10,752 + 6,656 + 32,832 | 50,752 |
| **CVAE 支路小计** | 仅训练时执行 | **17,383,488** |
| shared ResNet18 backbone | 去掉 avgpool/fc，FrozenBatchNorm | 11,166,912 |
| 主 Transformer encoder | 4 × 4,333,184 | 17,332,736 |
| Transformer decoder | 1 layer 5,384,832 + final LayerNorm 1,024 | 5,385,856 |
| state input projection | Linear 20→512 | 10,752 |
| latent input projection | Linear 32→512 | 16,896 |
| image feature projection | Conv2d 512→512, kernel 1 | 262,656 |
| encoder 1D position embeddings | 2×512 | 1,024 |
| decoder query embeddings | 100×512 | 51,200 |
| action head | Linear 512→12 | 6,156 |
| **输入/输出投影小计** | 上述六项 | **348,684** |
| **Policy 总计** | 所有 trainable parameters | **51,617,676（51.62M）** |

所有 51,617,676 个参数当前均 `requires_grad=true`。无参数模块包括 2D/sinusoidal position
encoding、action queue、normalizer 的统计逻辑和 `action_is_pad` mask。

## Loss 计算

令 ground-truth normalized action 为 `a`、预测为 `a_hat`、有效 mask 为
`m = not action_is_pad`。重建项只平均有效 timestep 与 12 个 action 维度：

```text
L_L1 = sum(m * abs(a_hat - a)) / (sum(m) * 12)
```

CVAE latent 的 posterior 为 `N(mu, diag(exp(log_sigma_x2)))`，相对标准高斯的 KL 为：

```text
L_KL = mean_batch[-0.5 * sum_latent(
    1 + log_sigma_x2 - mu^2 - exp(log_sigma_x2)
)]

L_total = L_L1 + 10.0 * L_KL
```

所以 ACT 的 reconstruction loss 是 L1，不是 Diffusion 的 noise MSE；padding action 不进入
L1，但会在 CVAE attention 中作为 padding token mask。optimizer 为 AdamW，
`lr=1e-5`、`weight_decay=1e-4`，当前没有额外 scheduler。backbone 参数单独建 optimizer
group，但本配置 `optimizer_lr_backbone=1e-5`，与其余模型相同。

## 固定实验配置与可修改位置

| 参数 | 值 |
| --- | ---: |
| training steps | 100,000 |
| batch size | 8 |
| DataLoader workers | 8 |
| seed | 42 |
| state rotation | rotation 6D |
| `chunk_size` | 100 |
| `n_action_steps` | 100 |
| hidden / heads / FFN | 512 / 8 / 3200 |
| main encoder / decoder layers | 4 / 1 |
| VAE encoder layers / latent | 4 / 32 |
| checkpoint + inline eval interval | 10,000 steps |
| inline / final eval episodes | 5 / 50 |

修改 action horizon 编辑 [`config.env`](config.env) 的 `KOURO_ACT_CHUNK_SIZE` 和
`KOURO_ACT_N_ACTION_STEPS`；网络和 loss 在 `policy/act/modeling_act.py`，结构默认值在
`policy/act/configuration_act.py`。如需保留 baseline，应复制成新的扁平实验目录并修改
`KOURO_EXPERIMENT_ID`。

## 一键训练、TensorBoard 与最终评测

在已经挂载共享硬盘的 GPU Docker 内执行：

```bash
cd /mnt/data/nas/hufangchi/projects/project_kouro
KOURO_EXPERIMENT=ACT_TurnOffSinkFaucet_baseline \
KOURO_GPU=0 bash scripts/tmux_train_turnoff_sink_faucet.sh
```

脚本不会创建 Docker。tmux 的 `train` window 自动建立本地 SSD cache、训练、周期 eval 和
最终 50-episode eval；`tensorboard` window 监听 `0.0.0.0:6007`。

只检查缓存和展开命令，不启动训练：

```bash
KOURO_EXPERIMENT=ACT_TurnOffSinkFaucet_baseline \
bash scripts/train_turnoff_sink_faucet.sh --dry-run
```

输出位置：

- checkpoint：`checkpoints/robocasa/TurnOffSinkFaucet/act/ACT_TurnOffSinkFaucet_baseline/`；
- 实际配置：`effective_config.env`；
- 冻结代码：`policy_snapshot/`、`policy_adapter_snapshot.sh`、`environment_snapshot/`；
- 训练监控：`train.log`、`status.env`、`tensorboard/`；
- RoboCasa 结果：`eval/<timestamp>/`。
