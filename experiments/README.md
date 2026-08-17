# Experiments

一个实验对应 `experiments/` 下的一个直接子文件夹：

```text
experiments/<Policy>_<Task>_<Variant>/
```

不要再拆成 `Policy/Task/Variant/runs` 多层目录。每个实验自己的 `README.md` 必须完整记录：

- 实验目的与要验证的假设；
- 环境、任务名和 train/eval split；
- 数据目录、版本、episode 选择、帧数、频率、相机、state/action 维度；
- policy、相对 baseline 的改动和关键超参数；
- 训练、tmux/TensorBoard 监控和评测命令；
- checkpoint、日志、评测结果与主要评价指标。

训练启动后，最终展开的底层命令也追加在同一个 README。

通用 launcher 通过 `KOURO_EXPERIMENT=<directory-name>` 选择实验，然后从该目录
的 `config.env` 读取 `KOURO_POLICY_TYPE` 和 `KOURO_POLICY_DIR`。没有设置
`KOURO_EXPERIMENT` 时使用当前 baseline。

当前已定义实验：

- [`DP_TurnOffSinkFaucet_baseline`](DP_TurnOffSinkFaucet_baseline)
- [`ACT_TurnOffSinkFaucet_baseline`](ACT_TurnOffSinkFaucet_baseline)
- [`Pi05_TurnOffSinkFaucet_baseline`](Pi05_TurnOffSinkFaucet_baseline)
