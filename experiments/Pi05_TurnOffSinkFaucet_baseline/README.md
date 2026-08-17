# Pi05 TurnOffSinkFaucet baseline

## 实验目的

将 LeRobot 的原始 Pi05 flow-matching policy 从官方通用 `lerobot/pi05_base`
checkpoint fine-tune 到 RoboCasa `TurnOffSinkFaucet`，建立预训练 Vision-Language-Action baseline。
本实验默认 full fine-tuning：不冻结 vision encoder，也不只训练 action expert。

这里的实现是 LeRobot 对 OpenPI Pi0.5 continuous flow-matching action head 的直接 PyTorch port。
它不包含上游尚未发布的 subtask prediction、discrete action tokenization 或 RL 组件；README
描述的是当前真正会训练的这套连续 action 模型。

## 任务与评测协议

- 环境：RoboCasa365 / PandaOmron。
- 任务族：Atomic。
- 任务名：`TurnOffSinkFaucet`。
- 训练数据 split：`pretrain`；RoboCasa eval split：`pretrain`。
- 评测 object registries：`[objaverse, lightwheel]`。
- 训练中评测：每 2,000 steps 做 5 episodes。
- 最终评测：训练结束后自动做 50 episodes。

本实验使用前 100 个 episode，共 11,608 frames。Pi05 没有 `drop_n_last_frames` 配置，因此
所有帧都可作为 anchor；episode 尾部不足 50 步时，dataset loader 会复制最后一个 action
padding。当前 Pi05 loss 不读取 `action_is_pad`，这些 padding 位置仍参与 flow-matching loss。

## 数据与 RoboCasa 接口

NAS 权威源数据：

```text
data/robocasa365/v1.0/pretrain/atomic/TurnOffSinkFaucet/20250819/lerobot
```

- 格式：LeRobot v3.0，20 Hz。
- 原始数据：106 episodes / 12,309 frames。
- 本实验：episode index `0..99`，100 episodes / 11,608 frames。
- 图像：left、eye-in-hand、right 三个 256×256 RGB 视角。
- 语言：task metadata 为 `Turn off the sink faucet.`。

### State/action 表示

权威数据的 state16 为：

```text
base_position_world(3) + base_quaternion_xyzw_world(4)
+ ee_position_in_base(3) + ee_quaternion_xyzw_in_base(4) + gripper_qpos(2)
```

训练和 eval 都转换为 state20：

```text
base_position_world(3) + base_rotation_6d_world(6)
+ ee_position_in_base(3) + ee_rotation_6d_in_base(6) + gripper_qpos(2)
```

rotation 6D 使用 `[R00,R10,R20,R01,R11,R21]`。12D action 保持 RoboCasa EEF+Delta：

| Slice | 字段 | 语义 |
| --- | --- | --- |
| `[0:4]` | mobile base + torso | base velocity 与 torso position Delta |
| `[4]` | control mode | `-1` arm mode，`+1` base mode |
| `[5:8]` | EEF position Delta | mobile-base frame 中的 Cartesian Delta |
| `[8:11]` | EEF rotation Delta | mobile-base frame 中的 axis-angle Delta |
| `[11]` | gripper | `-1` open，`+1` close |

本任务真正变化的 action 是 `[5:11]` 的 6D EEF Delta，其余维度为常量；Pi05 仍预测完整
12D。完整参考系和 controller scaling 见 [`../../data/README.md`](../../data/README.md)。

### 从通用 base schema 适配到 RoboCasa 20D/12D

通用 `pi05_base` checkpoint 声明三路 224×224 图像、32D state 和 32D action。adapter 传入
`--policy.input_features=null`，LeRobot factory 随后从当前 dataset 重建：

```text
input_features  = 三路 image + observation.state(20)
output_features = action(12)
```

Pi05 内部固定 `max_state_dim=max_action_dim=32`，所以 state20/action12 都在上限内。action
进入网络前从 12D zero-pad 到 32D，网络输出 32D 后只取前 12D。checkpoint 中的原 camera
keys 也不会直接套到 RoboCasa；当前 dataset 的 left、eye-in-hand、right 三路真实相机会取代它们。

## Policy 核心代码与 checkpoint

实验显式选择：

```bash
KOURO_POLICY_TYPE=pi05
KOURO_POLICY_DIR=policy/pi05
```

核心文件：

- [`configuration_pi05.py`](../../policy/pi05/configuration_pi05.py)：Gemma variants、flow
  时间分布、horizon、normalization、optimizer；
- [`modeling_pi05.py`](../../policy/pi05/modeling_pi05.py)：SigLIP/PaliGemma、action expert、
  flow loss 和 ODE sampling；
- [`processor_pi05.py`](../../policy/pi05/processor_pi05.py)：quantile normalization、
  state-to-text prompt、tokenizer 和 unnormalize；
- [`_shared/pi_gemma.py`](../../policy/pi05/_shared/pi_gemma.py)：AdaRMS PiGemma 实现；
- [`_shared/rtc/`](../../policy/pi05/_shared/rtc/)：可选 Real-Time Chunking；本 baseline 的
  `rtc_config=null`，不启用 RTC。

源码基于本机 LeRobot commit `5c98e80430d4a747926b45893568e388105a2400`。模型结构、processor 和
loss 保持该版实现；本项目只对 `from_pretrained()` 加入了 fail-closed 安全修改，防止权重
加载失败时静默返回随机初始化模型。默认 checkpoint：

```text
Hugging Face ID: lerobot/pi05_base
revision: b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
model.safetensors: 14,467,165,872 bytes（约 13.5 GiB）
```

它是 LeRobot model card 标记用于下游 fine-tune 的
通用 base，不是 `lerobot/pi05_libero`，也不是 `lerobot/pi05_libero_finetuned`。后两者都不作为
本 RoboCasa baseline 的起点。路径和
revision 见 [`../../artifacts/pretrained/pi05/README.md`](../../artifacts/pretrained/pi05/README.md)。

`config.env` 显式固定 repo ID、revision 和权重 byte size；启动时 adapter 会同时校验
HF cache 目录名、revision、`config.json`、pre/postprocessor 和 `model.safetensors`。任何一项不符都会
立即停止，不会回退到 LIBERO checkpoint，也不会从头训练。正常日志必须出现：

```text
Loading pretrained Pi05 base: repo=lerobot/pi05_base revision=b211f3d...
Verified pretrained snapshot: .../models--lerobot--pi05_base/snapshots/b211f3d...
Loading pretrained weights from: ...
All keys loaded successfully!
```

项目内的 `PI05Policy.from_pretrained()` 也已改为 fail-closed：如果 safetensors 解析、key
remap 或 strict `state_dict` 加载失败，训练会直接报错终止，不再返回随机初始化模型。

## Processor：图像、state、语言和 action

当前 `batch_size=32`、`chunk_size=50`。processor 与模型之间的主要 tensor 为：

| 数据 | 外部 shape | Pi05 内部表示 |
| --- | --- | --- |
| 三路 image | 每个 `(32,3,256,256)` | resize/pad 至 224²，各自产生 256 image tokens |
| state | `(32,20)` | quantile normalize 后离散成文本 token，不直接作为连续向量入模 |
| task | 32 个字符串 | 与离散 state 拼成 prompt，tokenizer max length 200 |
| action | `(32,50,12)` | quantile normalize 后 pad 为 `(32,50,32)` |

### 1. Quantile normalization

Pi05 原始配置使用：

```text
VISUAL -> IDENTITY
STATE  -> QUANTILES
ACTION -> QUANTILES
```

本地 cache 会用前 100 episodes 重算 state20/action12 的 q01/q99：

```text
x_norm = 2 * (x - q01) / (q99 - q01) - 1
x      = (x_norm + 1) * (q99 - q01) / 2 + q01
```

分母为零时使用 `1e-8`。实现不自动 clip 超出 q01/q99 的值，因此 normalized state/action
可能越过 `[-1,1]`。输出 action 经相同 q01/q99 反归一化后再交给 RoboCasa controller。

### 2. State-to-text 与 PaliGemma tokenizer

20D normalized state 使用 256 个等宽 bins 离散化，然后和任务语言拼成：

```text
Task: Turn off the sink faucet., State: <20 integer bins>;
Action:
```

`google/paligemma-3b-pt-224` tokenizer 将 prompt pad/truncate 到最多 200 tokens。当前 Pi05
modeling 不再把 state20 作为单独 continuous token；state 信息只通过上述文本 token 进入
PaliGemma。这是修改 state encoder 或语言接口时必须注意的边界。

### 3. 图像处理

图像在 normalization processor 中保持 identity，随后 resize-with-pad 到 224×224 并从
`[0,1]` 映射到 `[-1,1]`。三路真实相机各有有效 mask；`empty_cameras=0`，不会人为补空相机。
这是有意的：RoboCasa 已经提供需要的三路视角，而通用 `pi05_base` 的 checkpoint schema 也是三路图像。

## 模型模块

### 1. SigLIP vision tower

每个 224×224 相机产生 16×16=256 个 patch tokens。三路图像共享同一套 SigLIP vision
tower，共 768 image tokens。vision tower 包含 patch/position embedding、27 个 encoder
blocks 和 post LayerNorm；multimodal projector 将 vision width 1152 投影到 VLM width 2048。

### 2. PaliGemma / Gemma 2B prefix model

图像 tokens 与最多 200 个 language/state tokens 构成 prefix。语言模型配置为：

```text
width=2048, depth=18, MLP=16384
attention heads=8, KV heads=1, head_dim=256
vocab=257152
```

prefix 负责视觉、任务语言和离散 state 的联合表示；推理时 prefix KV cache 只计算一次，
再供后续 flow denoising steps 使用。需要特别注意：标准 PaliGemma 通常 tie token embedding
与 LM head，但当前 Pi05 port 替换自定义 language model 后，runtime 中两者是两个独立参数；
checkpoint loader 会从 LM-head tensor clone 出缺少的 input embedding。

### 3. Gemma 300M action expert

action expert 配置为：

```text
width=1024, depth=18, MLP=4096
attention heads=8, KV heads=1, head_dim=256
```

32D noisy action 经 `Linear(32,1024)` 变成 50 个 suffix tokens。flow timestep 先做 1024D
sin/cos embedding，再经过两个 `Linear(1024,1024)+SiLU`；结果作为 18 层 expert 中 AdaRMS
的 conditioning。每一层把 PaliGemma prefix 与 expert suffix 放入联合 attention 计算，最后
`Linear(1024,32)` 得到 vector field。

## 模块与参数量

下表以当前配置 meta-initialize 出来的 `PI05Pytorch` 实际 module parameters 为准，并用默认
checkpoint 的 safetensors shapes 交叉核对。checkpoint 文件有 812 个 FP32 tensors、保存
3,616,757,520 个数值；加载时 `_fix_pytorch_state_dict_keys()` 会为 PaliGemma input embedding
clone 一份 526,647,296-element LM-head tensor，因此 runtime 模型有 813 个 parameter tensors、
4,143,404,816 个参数。参数量不随三相机数量、state20/action12 或 batch size 改变，因为这些
输入都投影/pad 到固定内部宽度。

| 模块 | 结构 | 参数量 |
| --- | --- | ---: |
| SigLIP embeddings | patch + position embeddings | 973,440 |
| SigLIP encoder | 27 × 15,239,504 | 411,466,608 |
| SigLIP post LayerNorm |  | 2,304 |
| **SigLIP vision tower 小计** | 27-layer vision encoder | **412,442,352** |
| multimodal projector | Linear 1152→2048 | 2,361,344 |
| PaliGemma input token embedding | vocab 257152 × width 2048；加载时由 checkpoint clone | 526,647,296 |
| PaliGemma vocabulary LM head | 独立于 input embedding | 526,647,296 |
| Gemma 2B Transformer | 18 × 110,104,576 | 1,981,882,368 |
| Gemma 2B final norm |  | 2,048 |
| **PaliGemma language/head 小计** | embedding + LM head + 18 blocks + norm | **3,035,179,008** |
| **完整 PaliGemma 小计** | vision + projector + language/head | **3,449,982,704** |
| expert vocabulary LM head | 257152 × 1024 | 263,323,648 |
| expert Transformer blocks | 18 × 23,599,104 | 424,783,872 |
| expert final AdaRMS | conditional dense | 3,148,800 |
| **Gemma action expert 小计** | head + 18 blocks + norm | **691,256,320** |
| action input projection | Linear 32→1024 | 33,792 |
| action output projection | Linear 1024→32 | 32,800 |
| timestep MLP | 2 × Linear 1024→1024 | 2,099,200 |
| **Policy 总计** | runtime module parameters | **4,143,404,816（4.143B）** |

当前 `train_expert_only=false`、`freeze_vision_encoder=false`，所以 4,143,404,816 个 runtime
parameters 均保持 `requires_grad=true`。但是 continuous-action forward 直接调用两个 Gemma
backbone，不调用 PaliGemma 或 expert 的 vocabulary output heads；这两个 head 虽合计
789,970,944 个参数并进入参数量/optimizer 枚举，正常情况下不会得到梯度。PaliGemma input
embedding 则确实用于 prompt tokens，会得到梯度。

磁盘 checkpoint 保存的是 3.617B 个 FP32 values，因此约 14.47GB；加载后多出的 input
embedding 使 runtime parameter count 上升到 4.143B。adapter 设置 `dtype=bfloat16`：
PaliGemma/action-expert 中除 vision tower、multimodal projector、input/post-attention norms
和 final norms 外的参数转为 BF16；后创建的 action input/output projections 与 timestep MLP
仍是 FP32。gradient checkpointing 打开以降低 activation 显存，但会增加重算时间。

## Flow-matching loss

令 quantile-normalized、pad 后的 action 为 `a ∈ R^(50×32)`，采样：

```text
epsilon ~ N(0,I)
t_raw ~ Beta(alpha=1.5, beta=1.0)
t = 0.999 * t_raw + 0.001

x_t = t * epsilon + (1 - t) * a
u_t = epsilon - a
v_hat = Pi05(images, language/state tokens, x_t, t)
```

网络学习从 data 向 noise 的直线路径 vector field，逐元素 loss 为：

```text
L_element = (v_hat - u_t)^2
L = mean(L_element[:, :, :12])
```

最终只对真实 12 个 action 维度、50 个 chunk timestep 和 batch 求平均；内部 pad 的后 20 个
action 维度不进入最终 loss。当前实现不读取 `action_is_pad`，所以 episode 边界复制出来的
50-step padding 仍参与 loss。这是与 ACT masked L1、Diffusion 当前 unmasked noise MSE 的重要
区别，也是后续可以实验的 loss 改动点。

optimizer 为 AdamW：peak `lr=2.5e-5`、`betas=(0.9,0.95)`、`eps=1e-8`、
`weight_decay=0.01`、global grad clip norm 1.0。cosine schedule warmup 1,000 steps，在 30,000
steps 衰减到 `2.5e-6`；训练若少于 30,000 steps，LeRobot 会按总步数自动缩放 schedule。

## Rollout sampling 与 action horizon

推理从 `x_1 ~ N(0,I)` 开始，用 10 个显式 Euler steps 积分 flow ODE：

```text
dt = -1 / 10
x_(t+dt) = x_t + dt * v_hat(x_t, t)
```

最终得到 50×32，截取前 12 维并反归一化成 50×12 action chunk。Policy queue 只执行前
`n_action_steps=10`，即 20 Hz 下约 0.5 秒，然后重新观察和规划；剩余 40 步丢弃。

修改预测 horizon：[`config.env`](config.env) 中的 `KOURO_PI05_CHUNK_SIZE`；修改每次实际
执行长度：`KOURO_PI05_N_ACTION_STEPS`；修改推理 ODE steps 目前需要在
`configuration_pi05.py` 的 `num_inference_steps` 或实验 adapter 中增加覆盖参数。

## 固定实验配置

| 参数 | 值 |
| --- | ---: |
| training steps | 20,000 |
| batch size | 32 |
| DataLoader workers | 8 |
| seed | 42 |
| dtype | bfloat16 mixed with selected FP32 modules |
| state rotation | rotation 6D |
| chunk / executed action steps | 50 / 10 |
| inference flow steps | 10 |
| max state / action dim | 32 / 32 |
| gradient checkpointing | true |
| full fine-tuning | true |
| checkpoint + inline eval interval | 2,000 steps |
| inline / final eval episodes | 5 / 50 |

当前 20k / batch 32 是第一轮设定；相比 30k / batch 64 recipe 更保守，LeRobot 会把
30k scheduler 自动缩放到实际的 20k steps。但 4.143B full fine-tuning 对单卡显存要求仍远高于
其他两个 policy。若目标机器无法容纳，可在启动命令前覆盖
`KOURO_BATCH_SIZE=4`，或者建立
`train_expert_only=true` / `freeze_vision_encoder=true` 的新实验；这些都会改变 baseline，应该
复制为新的扁平 experiment variant，而不是覆盖本目录结果。

## Docker 本地缓存

首次实际训练会自动复制：

- Pi05 base checkpoint：约 14G；
- PaliGemma tokenizer：约 17 MiB；
- task dataset、LeRobot 源码以及 RoboCasa assets。

权重和 tokenizer 复制到当前 Docker 的本地 SSD cache，训练热路径不持续读取 NAS。同一容器
再次运行会命中 fingerprint cache；新临时 Docker 会自动重建。`--dry-run` 只验证 Pi05
checkpoint 路径，不复制 14G 权重。

## 一键训练、TensorBoard 与最终评测

在已经挂载共享硬盘的 GPU Docker 内执行：

```bash
cd /mnt/data/nas/hufangchi/projects/project_kouro
KOURO_EXPERIMENT=Pi05_TurnOffSinkFaucet_baseline \
KOURO_GPU=0 bash scripts/tmux_train_turnoff_sink_faucet.sh
```

脚本不会创建 Docker。tmux 的 `train` window 自动缓存、训练、周期 eval 和最终
50-episode eval；`tensorboard` window 监听 `0.0.0.0:6008`。

检查配置但不复制大权重、不启动训练：

```bash
KOURO_EXPERIMENT=Pi05_TurnOffSinkFaucet_baseline \
bash scripts/train_turnoff_sink_faucet.sh --dry-run
```

输出位置：

- checkpoint：`checkpoints/robocasa/TurnOffSinkFaucet/pi05/Pi05_TurnOffSinkFaucet_baseline/`；
- 实际配置：`effective_config.env`；
- 冻结代码：`policy_snapshot/`、`policy_adapter_snapshot.sh`、`environment_snapshot/`；
- 训练监控：`train.log`、`status.env`、`tensorboard/`；
- RoboCasa 结果：`eval/<timestamp>/`。

## 本次实际展开命令

- 启动时间：`2026-08-17T14:28:32Z`
- Dataset cache：`/tmp/project-kouro-cache/data/TurnOffSinkFaucet/20250819/bb573179495466bd/lerobot`
- Code cache：`/tmp/project-kouro-cache/code/lerobot-d2d4f33a3555-kouro-5d71f459993b`
- Policy：`pi05` from `/mnt/data/nas/hufangchi/projects/project_kouro/policy/pi05`
- Checkpoint：`/mnt/data/nas/hufangchi/projects/project_kouro/checkpoints/robocasa/TurnOffSinkFaucet/pi05/Pi05_TurnOffSinkFaucet_baseline`

```bash
/mnt/data/nas/hufangchi/applications/conda_envs/Robotwin/bin/python3.12 -m lerobot.scripts.lerobot_train --policy.discover_packages_path=lerobot.policies.pi05 --policy.path=/tmp/project-kouro-cache/models/pi05/d64b4ab0ecf49fc3/model --policy.device=cuda --policy.use_amp=false --policy.push_to_hub=false --policy.input_features=null --policy.dtype=bfloat16 --policy.chunk_size=50 --policy.n_action_steps=10 --policy.empty_cameras=0 --policy.gradient_checkpointing=true --policy.train_expert_only=false --policy.freeze_vision_encoder=false --policy.compile_model=false --dataset.repo_id=project_kouro/robocasa_TurnOffSinkFaucet --dataset.root=/tmp/project-kouro-cache/data/TurnOffSinkFaucet/20250819/bb573179495466bd/lerobot --dataset.episodes=\[0\,1\,2\,3\,4\,5\,6\,7\,8\,9\,10\,11\,12\,13\,14\,15\,16\,17\,18\,19\,20\,21\,22\,23\,24\,25\,26\,27\,28\,29\,30\,31\,32\,33\,34\,35\,36\,37\,38\,39\,40\,41\,42\,43\,44\,45\,46\,47\,48\,49\,50\,51\,52\,53\,54\,55\,56\,57\,58\,59\,60\,61\,62\,63\,64\,65\,66\,67\,68\,69\,70\,71\,72\,73\,74\,75\,76\,77\,78\,79\,80\,81\,82\,83\,84\,85\,86\,87\,88\,89\,90\,91\,92\,93\,94\,95\,96\,97\,98\,99\] --dataset.video_backend=pyav --env.type=robocasa --env.task=TurnOffSinkFaucet --env.split=pretrain --env.obj_registries=\[objaverse\,lightwheel\] --output_dir=/mnt/data/nas/hufangchi/projects/project_kouro/checkpoints/robocasa/TurnOffSinkFaucet/pi05/Pi05_TurnOffSinkFaucet_baseline --job_name=Pi05_TurnOffSinkFaucet_baseline --steps=20000 --batch_size=32 --num_workers=8 --prefetch_factor=4 --persistent_workers=true --log_freq=100 --save_freq=2000 --eval_freq=2000 --eval.batch_size=1 --eval.n_episodes=5 --eval.use_async_envs=false --seed=42 --wandb.enable=false
```
