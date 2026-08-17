# RoboCasa environment adapter

这里保存训练和评测实际使用的 RoboCasa adapter。启动脚本会把这些文件覆盖到容器本地的
LeRobot 代码缓存中，并把当次版本冻结到实验目录的 `environment_snapshot/`。

## State 表示

RoboCasa 原始 observation 和 NAS 数据集均提供两段 `xyzw` 四元数。`robocasa.py` 将它们
转换成旋转矩阵前两列组成的 continuous rotation 6D：

```text
[x, y, z, w] -> [R00, R10, R20, R01, R11, R21]
```

最终 policy state 是 20D：

```text
base_position(3) + base_rotation_6d(6)
+ ee_position_relative(3) + ee_rotation_relative_6d(6) + gripper_qpos(2)
```

数据侧由 `scripts/data/convert_state_quat_to_rotation6d.py` 做同样转换。转换只发生在容器
本地数据缓存，NAS 源数据保持 16D 不变。`patch_lerobot_config.py` 同时把 LeRobot 的
RoboCasa feature shape 修改为 20D，避免训练和 rollout 的 feature metadata 不一致。

## Action 边界

RoboCasa action 保持 12D：

```text
base_motion(4) + control_mode(1) + ee_position_delta(3)
+ ee_rotation_delta(3) + gripper(1)
```

其中 `ee_rotation_delta(3)` 是控制器定义的三维增量旋转，并非 observation 中的四元数。
因此 6D rotation 只用于 policy 的 state 输入，不能改变送给 RoboCasa 控制器的 action 接口。
