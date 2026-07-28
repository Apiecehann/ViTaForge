<h1 align="center">ViTaForge</h1>

<p align="center">
  A unified visuo-tactile manipulation workspace for GelSight, Xense, and
  Neote sensors, diverse demonstration collection, policy training, and
  simulation inference.
</p>

ViTaForge provides a common workflow for three tactile sensor families:

| Sensor | `sensor_type` | Ready-to-use config | Tactile observations |
|---|---|---|---|
| GelSight | `gsmini` | `task_config/gelsight.yml` | RGB, marker RGB, marker, depth, pose |
| Xense | `xensews` | `task_config/xense.yml`, `task_config/xense_8tasks.yml` | RGB, marker RGB, marker, depth, pose |
| Neote | `neote` | `task_config/neote.yml` | RGB, marker RGB, marker, depth, pose, vertex force, force field, gel particles |

The same task interface and HDF5 data layout are used across sensor types, so
sensor selection, observation modalities, collection diversity, training, and
evaluation can all be controlled through configuration files.

## Installation

Requirements are Linux, an NVIDIA GPU with a compatible driver, Conda, and
Git. The all-in-one installer creates the `UniVTAC` Conda environment and
installs the repository's local tactile simulation dependencies.

```bash
git clone --branch minnan https://github.com/Apiecehann/ViTaForge.git
cd ViTaForge
bash scripts/install.sh
conda activate UniVTAC
```

Use the modified TacEx/UIPC sources bundled under `third_party/TacEx`; do not
replace them with the public TacEx package. See
[`docs/Installation.md`](docs/Installation.md) for the manual installation
workflow.

## Task Suite

The shared Xense configuration, `task_config/xense_8tasks.yml`, supports the
following manipulation tasks:

| Task module | Objective |
|---|---|
| `grasp_half_cylinder_in_clutter` | Grasp a target half-cylinder among distractors |
| `insert_half_cylinder_into_box` | Insert a half-cylinder into the matching opening |
| `insert_usb` | Grasp and insert a USB connector |
| `place_wooden_cube_on_yellow_area` | Place a wooden cube in the target area |
| `pour_ball_to_cup` | Grasp, carry, and pour balls into a target cup |
| `pull_drawer` | Grasp and pull a drawer |
| `swap_cup_order` | Rearrange colored cups |
| `turn_gear_pair` | Grasp and rotate a gear pair |

Task assets are stored in `assets/objects/task_0724/`. Other task modules under
`envs/` can use any compatible sensor configuration.

## Diversity Data Collection

Activate the environment and run collection commands from the repository root.
The serial entry point accepts the task module, task config, GPU, start seed,
maximum seed, and number of successful episodes:

```bash
bash collect_data.sh \
  <task_name> <task_config> <gpu_id> <start_seed> <max_seed> <episode_num>
```

For example, collect one Xense episode for the clutter grasp task on GPU 0:

```bash
bash collect_data.sh \
  grasp_half_cylinder_in_clutter xense_8tasks 0 0 0 1
```

Use the multiprocessing collector for larger datasets:

```bash
python scripts/parallel_collect_data.py \
  grasp_half_cylinder_in_clutter xense_8tasks \
  --workers 2 --episodes 8 --gpu 0
```

Each worker owns a simulator application and tactile solver, so choose the
worker count according to available GPU memory. Diversity is controlled by
the seed range, randomized textures, task randomization, sensor choice, and
selected observation modalities. Successful trajectories, videos, logs, and
the success map are written to:

```text
<save_dir>/<task_name>/<task_config>/
```

See [`docs/Collection.md`](docs/Collection.md) for the HDF5 layout.

## Task Config Hyperparameters

Task configuration files live in `task_config/`. The supplied values are tuned
defaults; contact thresholds, closure targets, and trajectory offsets should
be changed conservatively because they directly affect tactile deformation and
task success.

### Collection and reset

| Parameter | Purpose |
|---|---|
| `save_dir` | Root directory for collected task data. |
| `decimation` | Number of simulation steps represented by one control step. |
| `save_frequency` | Interval between saved observation/action samples. |
| `video_frequency` | Interval between frames written to preview videos. |
| `video_size` | Output video resolution as `[width, height]`. |
| `render_frequency` | Rendering cadence; use `0` for headless collection and `1` for an interactive window. |
| `reset_time_limit` | Maximum allowed reset duration in seconds. |
| `reset_first_frame_steps` | Constrained simulation steps used to initialize the first reset frame. |
| `reset_after_actor_steps` | Settling steps after task actors are initialized. |
| `reset_final_steps` | Final unconstrained settling steps before pre-move. |
| `random_texture` | Randomize supported scene and object textures. |
| `use_seed` | Enable deterministic seed-based task randomization. |
| `episode_num` | Default number of successful episodes to collect; CLI arguments can override it. |
| `save_pre_move` | Include the robot pre-move phase in the saved trajectory. |
| `tactile_video_key` | Tactile stream used in preview videos, such as `rgb_marker` or `gel_particle`. |

### Sensor and observations

| Parameter | Purpose |
|---|---|
| `sensor_type` | Select `gsmini`, `xensews`, or `neote`. |
| `observations.camera` | Camera streams saved to each episode, for example `rgb`. |
| `observations.tactile` | Tactile modalities to save. Availability depends on the selected sensor. |
| `observations.embodiment` | Robot state streams, such as `joint` and `ee`. |
| `observations.actor` | Save task-object state when enabled. |
| `dense_gelpad` | Use the dense Neote gel-pad representation. |
| `force_field_grid` | Neote force-field resolution as `[width, height]`. |

### Xense contact and adaptive grasp

In the names below, `<object>` can be `usb`, `half_cylinder`,
`insert_half_cylinder`, `cube`, `cup`, `pour`, `pour_cup`, `drawer`, or `gear` where
that task provides an override.

| Parameter | Purpose |
|---|---|
| `xense_use_baseline_filter` | Enable Xense depth-baseline filtering before contact checks. |
| `xense_marker_reference_max_settle_steps` | Maximum steps allowed to establish a stable marker reference after reset. |
| `xense_marker_reference_stable_steps` | Consecutive stable steps required to accept the marker reference. |
| `use_adaptive_grasp` | Stop gripper closing from tactile contact instead of always using a fixed closure. |
| `adaptive_grasp_depth_threshold` | Global depth/contact threshold. A larger value stops earlier, producing lighter contact. |
| `xense_<object>_adaptive_grasp_depth_threshold` | Per-object override of the global contact threshold. |
| `xense_adaptive_grasp_require_both_contacts` | Require both tactile pads to reach the threshold. |
| `xense_<object>_adaptive_grasp_require_both_contacts` | Per-object override for one-pad or two-pad contact. |
| `xense_<object>_close_percent` | Maximum Robotiq closure target; `1.0` is open and `0.0` is fully closed. Adaptive contact may stop earlier. |
| `xense_adaptive_grasp_max_steps` | Maximum number of closing steps. |
| `xense_adaptive_grasp_tail_steps` | Extra hold/settling steps after contact is detected. |
| `xense_adaptive_grasp_check_interval` | Number of steps between tactile contact checks. |
| `xense_adaptive_grasp_qpos_step` | Gripper joint-position increment for each close update. |
| `xense_adaptive_grasp_target_tolerance` | Tolerance for considering the closure target reached. |
| `xense_adaptive_grasp_min_target_margin` | Legacy minimum margin retained before reaching the closure target. |
| `xense_adaptive_grasp_hold_margin` | Reopen margin applied when holding after contact. |
| `xense_adaptive_grasp_hold_velocity` | Gripper velocity used during the post-contact hold. |
| `xense_adaptive_grasp_min_steps_before_contact` | Ignore contact until this many closing steps have elapsed. |
| `xense_adaptive_grasp_min_travel` | Ignore contact until the gripper has moved by this minimum amount. |

### Xense task trajectory

| Parameter | Purpose |
|---|---|
| `xense_<object>_grasp_height_bias` | Per-object vertical offset applied to the planned grasp pose. |
| `xense_<object>_grasp_world_y_bias` | Per-object world-Y grasp offset. |
| `xense_pour_cup_grasp_world_x_bias` | World-X grasp offset for the pouring cup. |
| `xense_drawer_grasp_z_bias` | Drawer-specific vertical grasp offset. |
| `xense_initial_settle_steps` | Default Xense settling steps before task motion begins. |
| `xense_<object>_initial_settle_steps` | Per-task override of the initial settling duration. |
| `xense_carry_time_dilation` | Timing scale applied to Xense carry trajectories. |
| `xense_carry_segments` | Number of interpolation segments in a carry motion. |
| `xense_carry_max_step` | Maximum Cartesian displacement per carry segment. |
| `xense_post_close_settle_steps` | General settling duration after closing the gripper. |
| `xense_usb_post_close_settle_steps` | USB-specific post-close settling override. |
| `xense_cup_min_principal_ratio` | Minimum retained principal-shape ratio accepted for a cup. |
| `xense_cup_max_nonrigid_error` | Maximum non-rigid cup deformation accepted by task checks. |

### Xense pouring

| Parameter | Purpose |
|---|---|
| `xense_pour_ball_friction_ratio` | Friction scaling for the poured balls. |
| `xense_pour_grip_friction_ratio` | Contact-friction scaling between gripper and cup. |
| `xense_pour_wrist_angle_deg` | Target wrist rotation used for pouring. |
| `xense_pour_wrist_steps` | Number of interpolation steps in the wrist rotation. |
| `xense_pour_wrist_translation_x/y/z` | Cartesian translation applied during wrist rotation. |
| `xense_pour_actor_tilt_deg` | Optional additional tilt applied to the cup actor. |
| `xense_pour_actor_tilt_axis_x/y/z` | Axis of the optional actor tilt. |
| `xense_pour_carry_segments` | Interpolation segments for carrying the pouring cup. |
| `xense_pour_carry_settle_steps` | Settling steps after the carry phase. |
| `xense_pour_hold_actor_during_carry` | Temporarily constrain the cup actor during carry. |
| `xense_pour_target_y_offset`, `xense_pour_target_z_offset` | Offsets of the pour pose relative to the target cup. |
| `xense_pour_release_lift` | Vertical lift applied during release. |
| `xense_pour_release_snap_angle_deg` | Angle of each release snap motion. |
| `xense_pour_release_snap_steps` | Interpolation steps per release snap. |
| `xense_pour_release_snap_cycles` | Number of release snap cycles. |
| `xense_pour_fix_cup_during_release` | Constrain the pouring cup during the release motion. |
| `xense_pour_release_retract_x` | World-X retraction after release. |
| `xense_pour_release_carry_y` | World-Y carry motion applied during release. |

## Training

Collected episodes are first converted into the format expected by a policy,
then training is launched from that policy's directory. The preprocessors
currently read raw episodes from `../../data/<task_name>/<task_config>`.
Therefore, set `save_dir: ./data` before collecting training data, or place a
link to a run collected under another `save_dir` at that path.

### ACT

```bash
cd policy/ACT
python process_data.py <task_name> <task_config> <episode_num>
bash train.sh \
  <task_name> <task_config> <episode_num> <seed> <gpu_id> [train_config]
```

`train_config` selects a YAML file in `policy/ACT/` without its extension and
defaults to `train_config`.

### Ablation

```bash
cd policy/Ablation
python process_data.py <task_name> <task_config> <episode_num>
bash train.sh \
  <task_name> <task_config> <episode_num> <seed> <gpu_id> [train_config]
```

### ViTAL

```bash
cd policy/ViTAL
python process_data.py <task_name> <task_config> <episode_num>
bash train.sh <task_name> <task_config> <episode_num> <gpu_id>
```

## Inference

Set the checkpoint fields in `policy/ACT/deploy.yml`,
`policy/Ablation/deploy.yml`, or `policy/ViTAL/deploy.yml`, then evaluate from
the repository root. The third argument is the deployment YAML path relative
to `policy/`, without the `.yml` extension.

```bash
bash eval_policy.sh \
  <task_name> <task_config> ACT/deploy <gpu_id>

bash parallel_eval.sh \
  <task_name> <task_config> ACT/deploy <gpu_id> [workers] [total_episodes]
```

For an ACT checkpoint trained with a non-default training config, export the
matching config name during evaluation:

```bash
TRAIN_CONFIG=train_config_vision_only \
bash eval_policy.sh <task_name> <task_config> ACT/deploy <gpu_id>
```

See [`docs/Deploy.md`](docs/Deploy.md) for implementing and evaluating a custom
policy.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Acknowledgement

ViTaForge is developed on top of the
[`UniVTAC`](https://github.com/univtac/UniVTAC) framework. We thank the UniVTAC
authors for the unified simulation, data generation, learning, and benchmarking
foundation. Related resources: [paper](https://arxiv.org/abs/2602.10093),
[project website](https://univtac.github.io/), and
[dataset](https://huggingface.co/datasets/byml/UniVTAC).

```bibtex
@article{chen2026univtac,
  title={UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking},
  author={Chen, Baijun and Wan, Weijie and Chen, Tianxing and Guo, Xianda and Xu, Congsheng and Qi, Yuanyang and Zhang, Haojie and Wu, Longyan and Xu, Tianling and Li, Zixuan and others},
  journal={arXiv preprint arXiv:2602.10093},
  year={2026}
}
```
