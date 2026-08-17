# Checkpoints

训练权重位于：

```text
checkpoints/robocasa/<Task>/<Policy>/<run_id>/
```

LeRobot 会在每个 run 下生成 `checkpoints/<step>/pretrained_model/`，并维护
`checkpoints/last` 软链接。大文件不进入 Git。
