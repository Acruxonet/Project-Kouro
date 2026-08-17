# Project-Kouro

面向 CVPR 2027 的机器人 policy idea 与实验工作台。仓库负责统一管理可训练数据、可修改
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

## DSW 上的 tmux

阿里云 DSW Web Terminal 会管理默认 tmux server，并注入平台自己的 `TMUX_BINARY` 和
`TMUX_CONFIG`。系统 tmux 连接这个默认 socket 创建 detached session 时，可能在训练命令
执行前直接打印 `server exited unexpectedly`。使用 `tmux -f /dev/null -L <socket>` 建立
隔离 server 能正常工作，说明这不是 policy、训练配置或 GPU 初始化错误，而是 DSW 默认
tmux 环境的冲突。

仓库的训练和独立评测 launcher 因此不使用默认 server，而是为每个 session 建立同名的独立
socket，并忽略平台 tmux 配置。启动命令不需要额外参数；launcher 会打印正确的 attach 命令：

```bash
tmux -L <session_name> attach -t <session_name>
```

状态和停止脚本会自动查找独立 socket，同时兼容修复前已经在默认 socket 中启动的 session：

```bash
bash scripts/tmux_status.sh <session_name>
bash scripts/tmux_stop.sh <session_name>
```

若训练进程在 preflight 阶段立即失败，tmux pane 会保持 dead 状态并保留 stderr；用 status
脚本即可查看真实错误，不会再只剩 `server exited unexpectedly`。

## 输出边界

实验说明和结果归属于 `experiments/`；模型权重归属于 `checkpoints/`；可重复构建的容器本地
缓存默认归属于 `/tmp/project-kouro-cache`。仓库脚本只在已有容器中运行，不创建或启动 Docker。
