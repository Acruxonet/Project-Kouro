# RoboCasa365 training data

The unified local dataset root is:

```text
data/robocasa365/v1.0/pretrain/atomic
```

All 12 tasks are human-teleoperation RoboCasa365 datasets converted to the
LeRobot v3.0 on-disk format. They use the PandaOmron robot, run at 20 Hz, and
contain three RGB camera streams together with `observation.state` and
`action`.

## State/action semantics

### Classification used by this project

For the `EEF / joint` and `Delta / relative / absolute` experiment axes, the
current datasets are classified as:

```text
arm action space: EEF
arm action representation: Delta
observation state: absolute base pose + base-relative EEF pose
```

In other words, the existing action baseline is **`EEF + Delta`**. It is not a
joint-action dataset, an EEF relative-pose-target dataset, or an EEF
absolute-pose-target dataset. The full 12D action also includes the mobile
base, torso/control mode, and gripper; `EEF + Delta` describes the 6D arm
control part.

The words have the following precise meanings in this repository:

- **absolute pose**: the pose itself, expressed in the fixed MuJoCo world
  frame;
- **relative pose**: the pose itself, expressed in another body's coordinate
  frame (the PandaOmron mobile-base frame here), not a difference between two
  timesteps;
- **Delta action**: a requested per-control-step change from the current EEF
  pose. Its Cartesian axes are expressed in the mobile-base frame;
- **joint action**: a target/delta/command for the arm joints. No arm joint
  state or arm joint action is stored in these LeRobot datasets.

Thus, `observation.state` containing an EEF pose named `*_relative` does not
make the action a `relative` pose target. Observation and action use different
representations.

### On-disk `observation.state` (16D)

The authoritative NAS Parquet files store the following flat vector. All
poses are the measured pose at the current frame, not temporal deltas.

| Slice | Semantic field | Representation | Reference frame / meaning |
| --- | --- | --- | --- |
| `[0:3]` | `base_position` | **absolute** Cartesian position `(x, y, z)`, metres | PandaOmron mobile-base center in the MuJoCo world frame |
| `[3:7]` | `base_quaternion_xyzw` | **absolute** quaternion `(x, y, z, w)` | mobile-base orientation in the MuJoCo world frame |
| `[7:10]` | `ee_position_relative` | **relative** Cartesian position `(x, y, z)`, metres | EEF position expressed in the current mobile-base frame (`robot0_base_to_eef_pos`) |
| `[10:14]` | `ee_quaternion_relative_xyzw` | **relative** quaternion `(x, y, z, w)` | EEF orientation expressed in the current mobile-base frame (`robot0_base_to_eef_quat`) |
| `[14:16]` | `gripper_qpos` | **absolute state** | current positions of the two Panda gripper joints; not an action or delta |

For compatibility with the RoboCasa365 source data, the relative EEF position
comes from the EEF site while `robot0_base_to_eef_quat` uses the EEF body
orientation. Code that derives a new pose representation must preserve this
convention unless the dataset and evaluation observation are both changed.

### Training-time `observation.state` (20D)

The training launcher copies a selected dataset to container-local storage and
only changes the two rotation encodings from quaternion to continuous rotation
6D:

```text
[0:3]   base_position                    absolute, world frame
[3:9]   base_rotation_6d                 absolute, world frame
[9:12]  ee_position_relative             relative to mobile base
[12:18] ee_rotation_relative_6d           relative to mobile base
[18:20] gripper_qpos                      absolute joint state
```

Rotation 6D is the first two rotation-matrix columns in
`[R00, R10, R20, R01, R11, R21]` order. This conversion changes only the
numerical rotation encoding; it does not change absolute/relative semantics.
The authoritative NAS data remains 16D.

### On-disk `action` (12D)

Actions are stored in RoboCasa/robosuite's normalized controller-input space.
The arm uses PandaOmron operational-space control (`OSC_POSE`) with
`input_type=delta` and `input_ref_frame=base`.

| Slice | Semantic field | Representation | Controller meaning |
| --- | --- | --- | --- |
| `[0:4]` | `base_motion` | velocity/delta commands, not an absolute pose | `[base_x, base_y, base_yaw]` mobile-base motion plus one torso joint-position delta |
| `[4:5]` | `control_mode` | discrete mode | `-1`: arm mode/update EEF goal from achieved pose; `+1`: base mode/update from desired EEF goal |
| `[5:8]` | `ee_position_delta` | **EEF Delta** | normalized Cartesian change in the mobile-base frame; each `[-1, 1]` component is scaled to `[-0.05, 0.05]` metres by the controller |
| `[8:11]` | `ee_rotation_delta` | **EEF Delta** | normalized axis-angle rotation change in the mobile-base frame; each `[-1, 1]` component is scaled to `[-0.5, 0.5]` radians |
| `[11:12]` | `gripper` | binary command | `-1`: open, `+1`: close |

For `TurnOffSinkFaucet`, the stored statistics show that all four
`base_motion` values are `0`, `control_mode` is always `-1`, and `gripper` is
always `-1`; the varying action signal is the 6D EEF Delta command. The policy
still reads and predicts the complete 12D vector.

### Other LeRobot fields

| Field | Meaning |
| --- | --- |
| `observation.images.robot0_agentview_left` | 256x256 RGB video from the left external camera |
| `observation.images.robot0_eye_in_hand` | 256x256 RGB video from the wrist camera |
| `observation.images.robot0_agentview_right` | 256x256 RGB video from the right external camera |
| `annotation.human.task_description` | integer index for the natural-language task description |
| `annotation.human.task_name` | integer index for the task name |
| `next.reward` | sparse next-step task reward |
| `next.done` | whether the episode terminates at the next step |
| `timestamp` | frame timestamp; datasets run at 20 Hz |
| `frame_index` | zero-based frame index inside an episode |
| `episode_index` | zero-based episode index |
| `index` | global frame index in the dataset |
| `task_index` | index into the dataset task table |

The authoritative NAS datasets preserve the native 16D state layout with two
`xyzw` quaternions. Training launchers copy the selected task to the
container-local cache, convert those state rotations to a 20D rotation-6D
layout, and recompute `min/max/mean/std/q01/q10/q50/q90/q99` statistics for
state and action over the selected episodes. The source datasets in this
directory are never rewritten. RoboCasa's 12D action layout is unchanged
because its 3D end-effector rotation field is an incremental controller
command, not a quaternion.

| Task | Date | Episodes | Frames |
| --- | --- | ---: | ---: |
| `CloseDrawer` | 20250819 | 110 | 15,670 |
| `CloseFridge` | 20250819 | 100 | 25,332 |
| `CoffeeServeMug` | 20250819 | 100 | 15,597 |
| `OpenFridge` | 20250819 | 100 | 31,612 |
| `PickPlaceCounterToCabinet` | 20250819 | 100 | 22,509 |
| `PickPlaceFridgeShelfToDrawer` | 20250821 | 100 | 24,377 |
| `SlideToasterOvenRack` | 20250820 | 100 | 10,923 |
| `StartCoffeeMachine` | 20250819 | 108 | 13,722 |
| `TurnOffMicrowave` | 20250819 | 108 | 15,233 |
| `TurnOffSinkFaucet` | 20250819 | 106 | 12,309 |
| `TurnOnMicrowave` | 20250819 | 100 | 13,002 |
| `TurnOnToasterOven` | 20250820 | 100 | 16,510 |
| **Total** | | **1,232** | **216,796** |

The eight Atomic-8 baseline tasks contain exactly 100 episodes each. The four
additional tasks preserve every episode present in their official archives;
training configurations can select the first 100 episodes when an exactly
balanced 100-demo comparison is required.

The four additional datasets were converted from LeRobot v2.1 with the
`convert_dataset_v21_to_v30.py` converter from the local mu0/LeRobot checkout
at commit `68dd89de03c1b2afca8ce998c7510231d2412bd8`. Conversion was performed on
local SSD and the resulting v3.0 data was validated after copying it back to
NAS. Original `lerobot.tar` archives and temporary v2.1 copies are not included.

The complete data tree currently occupies approximately 563 MiB on NAS and is
intentionally ignored by Git.
