# DP TurnOffSinkFaucet baseline

## 实验目的

建立 `TurnOffSinkFaucet` 单任务上的 Diffusion Policy baseline，得到后续修改 action horizon、
网络结构、归一化方法和 loss 时可直接比较的训练曲线与 RoboCasa 成功率。本实验先固定一套
可复现配置：batch size 32、训练 20k steps，每 2k steps 保存 checkpoint 并做一次 5-episode 小评测，
训练结束后自动使用最终 checkpoint 做 50-episode 正式评测。

这里的 baseline 使用标准 Diffusion Policy 架构和 loss，但将输入状态中的四元数统一转换成
continuous rotation 6D。这个表示选择是 baseline 配置的一部分，不是额外的网络 idea。

## 任务与评测协议

- 环境：RoboCasa365 / PandaOmron。
- 任务族：Atomic。
- 任务名：`TurnOffSinkFaucet`。
- 训练数据 split：`pretrain`。
- RoboCasa 评测 split：`pretrain`。
- 评测 object registries：`[objaverse, lightwheel]`。
- 训练中评测：每 2k steps 做 5 episodes，用于观察收敛和过拟合趋势。
- 最终评测：训练完成后自动做 50 episodes，作为本实验的正式结果。

任务目标是控制移动底盘、机械臂和夹爪，将正在出水的水龙头关闭。前 100 个 episode 在
`drop_n_last_frames=7` 后有 10,908 个有效采样窗口，且相邻窗口高度重叠。batch size 32
训练 20k steps 约等于 59 次 dataset pass；继续到 100k 将达到约 293 次，过拟合风险仍然很高。
因此本 baseline 保留 2k、4k、…、20k 的 checkpoint 和小评测结果，根据曲线选择 checkpoint，
不默认最后一步最好。

## 数据

### 本实验的状态与动作表示

本 baseline 明确归类为 **`EEF + Delta`**：机械臂由 EEF operational-space controller
控制，不使用 arm joint state/action；EEF action 是每个控制步的位姿增量，不是 relative
pose target 或 world-frame absolute pose target。

| 项目 | 本实验使用的语义 |
| --- | --- |
| base observation | Mobile Base 在 MuJoCo world frame 中的 **absolute pose** |
| EEF observation | EEF 在**当前 Mobile Base frame** 中的 **relative pose** |
| EEF action | 在**当前 Mobile Base frame** 中表达的 position/rotation **Delta** |
| gripper observation | 当前两指 joint qpos（absolute state） |

每一帧满足 `T_world_eef = T_world_base @ T_base_eef`。因此当相邻帧的 Mobile Base
发生移动时，即使 `T_base_eef` 不变，EEF 的 world-frame absolute pose 仍会变化；不能把
相邻两帧 `ee_*_relative` 的直接差值当成 EEF 的 world-frame 位移。

本任务数据中 `base_motion[0:4]` 始终为 `0`、`control_mode` 始终为 `-1`、gripper action
始终为 `-1`，实际变化的控制信号是 6D EEF Delta；policy 仍按完整 12D action 训练和预测。
完整的字段切片、参考系和控制器缩放定义见 [`../../data/README.md`](../../data/README.md)。

NAS 上的权威源数据：

```text
data/robocasa365/v1.0/pretrain/atomic/TurnOffSinkFaucet/20250819/lerobot
```

- 格式：LeRobot v3.0。
- 采集日期：2025-08-19。
- 原始数据：106 episodes，12,309 frames。
- 本实验使用：episode index `0..99`，共 100 episodes。
- 归一化统计：只使用 episode index `0..99` 重新计算，额外 6 个 episode 不进入统计量。
- 频率：20 Hz。
- 动作：12D `action`。
- 图像：256×256 RGB，三个视角：
  `robot0_agentview_left`、`robot0_eye_in_hand`、`robot0_agentview_right`。

源数据的 `observation.state` 是 16D，包含两段 `xyzw` 四元数：

```text
base_position(3) + base_quaternion_xyzw(4)
+ ee_position_relative(3) + ee_quaternion_relative_xyzw(4) + gripper_qpos(2)
```

训练脚本把数据复制到容器本地 SSD 后，将两段四元数转换成 rotation 6D，并重新计算前 100 个
episode 的 state/action 统计量。实际送入 policy 的 state 是 20D：

```text
base_position(3) + base_rotation_6d(6)
+ ee_position_relative(3) + ee_rotation_6d(6) + gripper_qpos(2)
```

rotation 6D 使用旋转矩阵的前两列，顺序为
`[R00, R10, R20, R01, R11, R21]`。NAS 上的 16D 源数据不会被修改；转换结果只存在于当前
训练容器的本地缓存中。评测环境使用同一转换，因此 train/eval 的 state 表示完全一致。

12D action 不做 rotation 6D 转换。它的布局是：

```text
base_motion(4) + control_mode(1) + ee_position_delta(3)
+ ee_rotation_delta(3) + gripper(1)
```

`ee_rotation_delta(3)` 是 RoboCasa 控制器要求的增量控制量，并不是四元数。

具体维度按 Python 0-based index 定义如下：

| Index | 字段 | 执行语义 |
| ---: | --- | --- |
| `0` | base forward | Mobile Base 前后速度命令 |
| `1` | base side | Mobile Base 左右速度命令 |
| `2` | base yaw | Mobile Base 旋转速度命令 |
| `3` | torso height | 底盘升降立柱的 joint-position Delta；归一化命令每步最多对应 ±0.05 m，该滑动关节范围为 0–0.34 m，初始位置为 0.20 m |
| `4` | control mode | `-1` 为 arm mode（从 achieved EEF pose 更新目标），`+1` 为 base mode（从 desired EEF goal 更新目标） |
| `5:8` | EEF position Delta | 当前 Mobile Base frame 中的 `(dx, dy, dz)`；controller 将每维 `[-1,1]` 缩放到 `[-0.05,0.05]` m |
| `8:11` | EEF rotation Delta | 当前 Mobile Base frame 中的 axis-angle `(drx, dry, drz)`；每维 `[-1,1]` 缩放到 `[-0.5,0.5]` rad |
| `11` | gripper | `-1` 打开，`+1` 闭合 |

这里的 torso 是 Omron 底盘与 Panda 机械臂之间的垂直升降平台，不是 Panda 的 7 个 arm
joint。本任务的 `action[0:5]` 和 `action[11]` 均为常量，torso 也没有运动。

## Policy 与固定配置

Policy：ResNet-18 ImageNet 视觉 backbone、Conditional 1D U-Net、DDPM scheduler、
epsilon prediction，以及 MSE noise-prediction loss。

本实验通过 `config.env` 显式选择：

```bash
KOURO_POLICY_TYPE=diffusion
KOURO_POLICY_DIR=policy/diffusion
```

通用 launcher 不再写死 Diffusion Policy；Diffusion 专属的 horizon、backbone 和依赖参数
由 `scripts/policy_adapters/diffusion.sh` 生成。

核心代码入口：

- [`configuration_diffusion.py`](../../policy/diffusion/configuration_diffusion.py)：horizon、
  normalization、ResNet/U-Net、DDPM 与 optimizer；
- [`modeling_diffusion.py`](../../policy/diffusion/modeling_diffusion.py)：视觉 encoder、
  Conditional U-Net、sampling 和 `compute_loss()`；
- [`processor_diffusion.py`](../../policy/diffusion/processor_diffusion.py)：normalize/unnormalize pipeline；
- [`normalization.py`](../../policy/diffusion/normalization.py)：训练实际覆盖使用的数值变换实现。

### 训练 batch 与时间窗口

模型实际使用的训练字段如下；表中 shape 使用本实验 `batch_size=32`：

| 字段 | Shape | 用途 |
| --- | --- | --- |
| `observation.state` | `(32, 2, 20)` | 两个观测时刻的 robot state |
| 三个 `observation.images.*` | 每个 `(32, 2, 3, 256, 256)` | 左外部、腕部、右外部 RGB 图像 |
| `action` | `(32, 64, 12)` | diffusion 的 clean action trajectory |
| `action_is_pad` | `(32, 64)` | episode 边界 padding 标记 |

以当前帧为 `t`，DataLoader 取：

```text
observation: [t-1, t]
action:      [t-1, t, ..., t+62]   # horizon=64
```

训练完成后的 rollout 从生成的 64 步中取 `[1:33]`，即 `[t, ..., t+31]` 共 32 步执行；
action queue 清空后才重新规划。模型不使用 task description/name、reward/done、frame/episode/task
index，也不使用不存在于数据中的 arm joint state。

`action_is_pad` 会进入 batch，但当前 `do_mask_loss_for_padding=false`，loss 不使用该 mask。
本实验将 `drop_n_last_frames` 从 31 改为 LeRobot Diffusion Policy 默认值 7。它只改变训练集
中哪些帧能作为采样 anchor，不改变 policy 的网络输出或推理执行逻辑：rollout 仍然生成 64 步，
取 `[1:33]`，即 `[t, ..., t+31]` 共 32 步执行，然后重新规划。

前 100 个 episode 的对比如下：

| 指标 | `drop=31` | `drop=7` |
| --- | ---: | ---: |
| 可采样 anchor | 8,508 | 10,908 |
| anchor 增量 | — | +2,400（约 +28.2%） |
| 每条轨迹丢弃的尾部 | 31 帧 / 1.55 s | 7 帧 / 0.35 s |
| success-state anchor | 0 | 900 |
| 完整 64-step target 的平均 padding | 约 9.1% | 约 22.1% |

这样会让最后 8～31 帧重新成为 observation，模型可以看到更接近水龙头、完成转动、末段微调和
失败纠正的状态。本数据每条 demo 的 success 信号都位于最后 16 帧；`drop=7` 会保留其中前 9 帧
作为 anchor，因此更贴近 rollout 中较晚发生重新规划时的状态分布。

代价是 episode 尾部 anchor 的 action target 会越过轨迹末端。Dataset loader 会复制最后一个
真实 action 做边界 padding；由于 padding loss mask 当前关闭，这些复制值仍参与 MSE。最极端的
anchor 是 `t=L-8`：

```text
实际执行区间：t, t+1, ..., t+7       # 8 个真实 action
padding 区间： t+8, ..., t+31         # 24 个 padding action
完整训练 target [t-1, ..., t+62]：9 个真实位置 + 55 个 padding 位置
```

这可能让模型学到“episode 结束后重复最后一个 action”的模式；不过正常 rollout 在环境检测到
success 后会立即终止，不会继续执行成功后的 padding 动作。是否启用
`do_mask_loss_for_padding=true` 应作为独立实验变量，不并入当前 baseline。

### 归一化与反归一化

三个 feature type 的配置是：

```text
VISUAL -> MEAN_STD
STATE  -> MIN_MAX
ACTION -> MIN_MAX
```

视频解码后的图像是 float `[0,1]`。虽然 ResNet-18 权重从 ImageNet 初始化，processor 实际
使用的是本任务前 100 个 episode 为每个相机分别统计的逐通道 `mean/std`，不是固定的
ImageNet mean/std：

```text
image_norm = (image - dataset_camera_mean) / (dataset_camera_std + 1e-8)
```

state20 和 action12 使用前 100 个 episode 重新计算的逐维 `min/max`：

```text
x_norm = 2 * (x - min) / (max - min) - 1
x      = (x_norm + 1) / 2 * (max - min) + min
```

磁盘中的 action 已经是 RoboCasa 的 `[-1,1]` controller input；上式是 policy 根据经验
`min/max` 做的第二层缩放。例如没有覆盖完整 controller range 的 EEF rotation 会被拉伸到
`[-1,1]`。State normalization 不 clip，所以评测状态超出训练范围时可以得到 `[-1,1]`
之外的值。

常量维度满足 `min == max` 时，代码以 `1e-8` 代替分母：其训练值稳定映射为 `-1`，推理
反归一化后回到原常量附近。因此本实验虽然对完整 12D action 计算 loss，真正变化的 action
仍只有 `action[5:11]` 的 6D EEF Delta。

### 视觉编码与 conditioning shape

ResNet-18 去掉 global average pooling 和 classification head，输出保留空间结构的 feature
map。每个相机使用独立 encoder，默认不 resize/crop：

```text
每个相机：
(B*T, 3, 256, 256) = (64, 3, 256, 256)
  -> ResNet-18 backbone:       (64, 512, 8, 8)
  -> learned 1x1 conv:         (64, 32, 8, 8)
  -> Spatial Softmax:          (64, 32, 2)
  -> flatten + Linear + ReLU:  (64, 64)
```

Spatial Softmax 的 `(x,y)` 是每个 learned keypoint 在归一化图像平面 `[-1,1]²` 上的期望
坐标。三个相机各输出 64D，因此每个观测时刻得到 `3*64=192D` visual feature；与 20D
state 拼接为 212D，再展平两个 observation steps，最终送给 Conditional 1D U-Net 的
`global_condition` shape 是 `(32, 424)`。

### 模块与参数量

下面是按本实验真实 feature schema（三个 256×256 相机、state20、action12）实例化
`DiffusionPolicy` 后统计的 trainable parameter 数；batch size 不影响参数量。三个相机默认
`use_separate_rgb_encoder_per_camera=true`，所以 ResNet 权重不共享。
模型构造时会创建三个 `DiffusionRgbEncoder` 实例并放入 `ModuleList`；forward 时三个
encoder 与三路 camera 一一对应。因此不仅 ResNet18，各路的 learned keypoint conv 和
64D projection head 也都是独立参数。这与 ACT baseline 的“三路相机共享一套
ResNet18”不同，是两个 baseline 之间需要显式记录的架构差异。

| 模块 | 结构 / 输出 | 参数量 |
| --- | --- | ---: |
| RGB encoder 0 | ResNet18 + SpatialSoftmax + 64D head | 11,197,088 |
| RGB encoder 1 | 同上，独立权重 | 11,197,088 |
| RGB encoder 2 | 同上，独立权重 | 11,197,088 |
| 三路 RGB encoder 小计 | 每路 backbone 11,176,512；keypoint conv 16,416；Linear 4,160 | 33,591,264 |
| diffusion timestep encoder | sinusoidal 128D → Linear 512 → Mish → Linear 128 | 131,712 |
| U-Net down stage 0 | 12 → 512，两个 conditional residual blocks + downsample | 5,895,168 |
| U-Net down stage 1 | 512 → 1024，两个 blocks + downsample | 24,299,520 |
| U-Net down stage 2 | 1024 → 2048，两个 blocks | 80,054,272 |
| U-Net middle block 0 | 2048D conditional residual block | 44,220,416 |
| U-Net middle block 1 | 2048D conditional residual block | 44,220,416 |
| U-Net up stage 0 | 2048 skip fusion → 1024 + upsample | 47,368,192 |
| U-Net up stage 1 | 1024 skip fusion → 512 + upsample | 12,411,904 |
| final convolution | 512D Conv/GroupNorm/Mish → 12D action | 1,318,412 |
| Conditional 1D U-Net 小计 | timestep + down/middle/up/final | 259,920,012 |
| **Policy 总计** | scheduler、normalizer 和 queues 无可训练参数 | **293,511,276（293.51M）** |

全部 293,511,276 个参数在当前配置中 `requires_grad=true`。这里最大的部分不是 ResNet，
而是 `down_dims=(512,1024,2048)` 的 Conditional U-Net；仅两个 2048D middle blocks 就有
88,440,832 个参数。若要做轻量化或检查是否过拟合，`down_dims`、是否共享相机 encoder、
SpatialSoftmax keypoint 数和 FiLM scale modulation 都是直接的改动位置。

### Diffusion loss 与 rollout 输出

训练时令归一化后的 clean action 为 `a₀ ∈ R^(64×12)`，采样
`epsilon ~ N(0,I)` 和均匀 timestep `k ∈ {0,...,99}`。DDPM 的
`squaredcos_cap_v2` schedule 产生系数后构造 `a_k`，Conditional 1D U-Net 预测加入的噪声：

```text
a_k = sqrt(alpha_bar_k) * a_0 + sqrt(1 - alpha_bar_k) * epsilon
epsilon_hat = UNet(a_k, k, global_condition_424)
loss = mean((epsilon_hat - epsilon)^2)
```

当前 `prediction_type=epsilon`，所以 target 不是 action 本身。MSE 对 batch、64 个 horizon
位置和 12 个 action 维度等权平均；`do_mask_loss_for_padding=false`，复制 padding 也参与
loss。优化器为 Adam，`lr=1e-4`、`betas=(0.95,0.999)`、`eps=1e-8`、
`weight_decay=1e-6`，cosine schedule warmup 500 steps。

推理时从高斯噪声反向采样 normalized action trajectory；`clip_sample=true` 将 denoising
sample 限制在 `[-1,1]`。`num_inference_steps` 未覆盖，因此使用完整 100 个 DDPM 反向步。
Policy postprocessor 再按 action 统计量反归一化成原始 12D
RoboCasa controller input，最后由环境 adapter 拆成 base、control mode、EEF 和 gripper
命令执行。

| 参数 | 值 |
| --- | ---: |
| training steps | 20,000 |
| batch size | 32 |
| DataLoader workers | 8 |
| seed | 42 |
| AMP | false |
| state rotation | rotation 6D |
| `n_obs_steps` | 2 |
| `horizon` | 64 |
| `n_action_steps` | 32 |
| `drop_n_last_frames` | 7 |
| checkpoint interval | 2,000 steps |
| inline eval interval | 2,000 steps |
| inline eval episodes | 5 |
| final eval episodes | 50 |
| TensorBoard train interval | 100 steps |

完整可执行配置在同目录的 `config.env`。当前 baseline 要更改 action horizon，修改其中的
`KOURO_HORIZON`、`KOURO_N_ACTION_STEPS` 和必要时的 `KOURO_N_OBS_STEPS`；
`KOURO_DROP_N_LAST_FRAMES=7` 是独立、显式的采样设置，不再由 horizon 参数自动计算。
如果要保留这个 baseline，应先创建新的扁平实验目录，修改新目录的 `KOURO_EXPERIMENT_ID`
和配置，再通过 `KOURO_EXPERIMENT` 选择它；不需要复制 launcher。

## 一键启动：训练 + 周期评测 + 最终评测

在已经挂载本项目共享硬盘的 GPU 容器中执行唯一一条启动命令：

```bash
cd /mnt/data/nas/hufangchi/projects/project_kouro
KOURO_GPU=0 bash scripts/tmux_train_turnoff_sink_faucet.sh
```

脚本不会创建或启动 Docker。它会建立一个 detached tmux session，其中：

1. `train` window 首次把 LeRobot、数据、RoboCasa 源码与 assets 缓存到该容器本地磁盘；
2. 开始 20k-step 训练，每 2k steps 保存 checkpoint 并运行 5 episodes；
3. 训练成功结束后，同一 window 自动用最终 checkpoint 运行 50 episodes；
4. `tensorboard` window 持续提供训练 loss、周期评测成功率和最终评测结果。

首次构建 RoboCasa 本地缓存需要从 NAS 复制约 32 GiB assets，因此应确保容器的 `/tmp`
所在本地盘有足够空间。后续在同一容器中会命中缓存；新临时容器会自动重建，不需要手工操作。

只构建并检查数据/代码缓存、打印训练命令，不启动训练或 RoboCasa rollout：

```bash
bash scripts/train_turnoff_sink_faucet.sh --dry-run
```

## 监控

启动脚本会打印实际 tmux session 名。常用命令：

```bash
tmux -L kouro-DP_TurnOffSinkFaucet_baseline attach -t kouro-DP_TurnOffSinkFaucet_baseline
bash scripts/tmux_status.sh kouro-DP_TurnOffSinkFaucet_baseline
bash scripts/tmux_stop.sh kouro-DP_TurnOffSinkFaucet_baseline
```

TensorBoard 默认监听 `0.0.0.0:6006`。若训练在远程机器上，可建立 SSH 端口转发后打开
`http://127.0.0.1:6006`。其中主要关注：

- `train/loss`、`train/grad_norm`、`train/learning_rate`；
- `eval/pc_success` 和 `eval/avg_sum_reward`：每 2k steps 的 5-episode 快速信号；
- 最终 50 episodes 写入的 `eval/*` 指标。

5-episode 结果方差会比较大，只用于看趋势和筛查过拟合；正式 baseline 指标以最终自动运行的
50-episode `pc_success` 为准。

## 单独重新评测

一键流程已经包含最终评测。如果训练完成后想再次用最终 checkpoint 运行另一组 50 episodes：

```bash
KOURO_GPU=0 bash scripts/tmux_eval_turnoff_sink_faucet.sh
```

每次结果都会写入独立的 `eval/<timestamp>/`，包含 `eval_info.json`、日志、逐 episode 结果和
rollout 视频，并追加到 TensorBoard。

## 输出

- Checkpoint：
  `checkpoints/robocasa/TurnOffSinkFaucet/diffusion/DP_TurnOffSinkFaucet_baseline/`。
- 实际配置与本地缓存路径：`effective_config.env`。
- 训练时 policy：`policy_snapshot/`。
- 训练时 policy adapter：`policy_adapter_snapshot.sh`。
- 训练时 RoboCasa adapter：`environment_snapshot/`。
- 训练状态与日志：`status.env`、`train.log`。
- TensorBoard：`tensorboard/`。
- RoboCasa 结果：`eval/<timestamp>/`。

训练启动时，脚本还会在本文件末尾追加本次实际展开的 LeRobot 命令和缓存路径。

## 本次实际展开命令

- 启动时间：`2026-08-17T13:28:00Z`
- Dataset cache：`/tmp/project-kouro-cache/data/TurnOffSinkFaucet/20250819/bb573179495466bd/lerobot`
- Code cache：`/tmp/project-kouro-cache/code/lerobot-d2d4f33a3555-kouro-31351ae7870e`
- Policy：`diffusion` from `/mnt/data/nas/hufangchi/projects/project_kouro/policy/diffusion`
- Checkpoint：`/mnt/data/nas/hufangchi/projects/project_kouro/checkpoints/robocasa/TurnOffSinkFaucet/diffusion/DP_TurnOffSinkFaucet_baseline`

```bash
/mnt/data/nas/hufangchi/applications/conda_envs/Robotwin/bin/python3.12 -m lerobot.scripts.lerobot_train --policy.discover_packages_path=lerobot.policies.diffusion --policy.type=diffusion --policy.device=cuda --policy.use_amp=false --policy.push_to_hub=false --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 --policy.n_obs_steps=2 --policy.horizon=64 --policy.n_action_steps=32 --policy.drop_n_last_frames=7 --dataset.repo_id=project_kouro/robocasa_TurnOffSinkFaucet --dataset.root=/tmp/project-kouro-cache/data/TurnOffSinkFaucet/20250819/bb573179495466bd/lerobot --dataset.episodes=\[0\,1\,2\,3\,4\,5\,6\,7\,8\,9\,10\,11\,12\,13\,14\,15\,16\,17\,18\,19\,20\,21\,22\,23\,24\,25\,26\,27\,28\,29\,30\,31\,32\,33\,34\,35\,36\,37\,38\,39\,40\,41\,42\,43\,44\,45\,46\,47\,48\,49\,50\,51\,52\,53\,54\,55\,56\,57\,58\,59\,60\,61\,62\,63\,64\,65\,66\,67\,68\,69\,70\,71\,72\,73\,74\,75\,76\,77\,78\,79\,80\,81\,82\,83\,84\,85\,86\,87\,88\,89\,90\,91\,92\,93\,94\,95\,96\,97\,98\,99\] --dataset.video_backend=pyav --env.type=robocasa --env.task=TurnOffSinkFaucet --env.split=pretrain --env.obj_registries=\[objaverse\,lightwheel\] --output_dir=/mnt/data/nas/hufangchi/projects/project_kouro/checkpoints/robocasa/TurnOffSinkFaucet/diffusion/DP_TurnOffSinkFaucet_baseline --job_name=DP_TurnOffSinkFaucet_baseline --steps=20000 --batch_size=32 --num_workers=8 --prefetch_factor=4 --persistent_workers=true --log_freq=100 --save_freq=2000 --eval_freq=2000 --eval.batch_size=1 --eval.n_episodes=5 --eval.use_async_envs=false --seed=42 --wandb.enable=false
```
