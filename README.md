# Project-Kouro

面向 CVPR 2027 的机器人 policy idea 与实验工作台。仓库负责统一管理可训练数据、可魔改
policy、实验定义、checkpoint 和 RoboCasa 评测，使模型结构、归一化、反归一化与 loss 都能
在当前工作区内直接查看和修改。

## 仓库结构

```text
project_kouro/
├── data/                 # LeRobot v3 训练数据
├── policy/               # 训练实际加载的 policy 核心代码
│   ├── diffusion/        # Diffusion 架构、loss、normalization、processor
│   ├── act/              # 原始 LeRobot ACT 核心代码
│   └── pi05/             # 原始 LeRobot Pi05 及其直接共享依赖
├── environment/          # RoboCasa adapter，保证训练/评测同用 rotation 6D state
├── experiments/          # 一个实验对应一个直接子文件夹
├── checkpoints/          # 训练 checkpoint
├── scripts/              # 缓存、训练、tmux、TensorBoard 与评测工具
└── artifacts/            # 共享盘上的预训练权重与兼容 wheel
```

数据大文件、checkpoint、TensorBoard event、视频和运行日志不进入 Git；实验 README、配置、
policy 快照和评测指标保留在仓库结构中。

## 实验组织原则

实验目录不按 policy、任务和 variant 拆成多层。每个实验直接使用一个能完整表达含义的目录名：

```text
experiments/<Policy>_<Task>_<Variant>/
```

已定义的三个单任务实验是
[`DP_TurnOffSinkFaucet_baseline`](experiments/DP_TurnOffSinkFaucet_baseline)、
[`ACT_TurnOffSinkFaucet_baseline`](experiments/ACT_TurnOffSinkFaucet_baseline) 和
[`Pi05_TurnOffSinkFaucet_baseline`](experiments/Pi05_TurnOffSinkFaucet_baseline)。
具体训练、监控和评测命令只记录在该实验自己的 `README.md` 中，根 README 不保存具体实验命令。

每个实验目录负责保存：

- `README.md`：实验目的、固定参数、启动命令和最终展开命令；
- `config.env`：该实验的可修改配置；
- `effective_config.env`：启动时的实际参数快照；
- `policy_snapshot/`：训练时实际使用的 policy 源码；
- `policy_adapter_snapshot.sh`：训练时 policy 专属参数与缓存逻辑的冻结快照；
- `environment_snapshot/`：训练时实际使用的 RoboCasa 观测/动作 adapter；
- `status.env`、`train.log`、`tensorboard/` 和 `eval/`：运行状态与结果。

一个目录代表一个实验。如果要更换 policy、架构、loss、action horizon、数据规模或 seed，应新建另一个
有独立名字的实验目录，而不是在同一目录堆放多个 run。

## 数据与 policy

RoboCasa365 的 12 个 Atomic 任务、episode 数和数据格式记录在 [`data/README.md`](data/README.md)。

Policy 工作区与切换约定见 [`policy/README.md`](policy/README.md)。启动训练时，脚本会把
LeRobot 源码复制到临时容器的本地 SSD，再根据实验的 `KOURO_POLICY_TYPE` 与
`KOURO_POLICY_DIR` 使用本仓库 policy 覆盖本地副本；DataLoader 同样
只读取本地数据缓存，因此 NAS 不位于训练热路径。

RoboCasa 训练/评测 adapter 见 [`environment/README.md`](environment/README.md)。当源数据的
state 包含四元数时，数据缓存和环境 adapter 会使用同一种 rotation 6D 转换；
RoboCasa 控制器要求的 12D action 保持不变。

## 输出边界

实验说明和结果归属于 `experiments/`；模型权重归属于 `checkpoints/`；可重复构建的容器本地
缓存默认归属于 `/tmp/project-kouro-cache`。仓库脚本只在已有容器中运行，不创建或启动 Docker。
