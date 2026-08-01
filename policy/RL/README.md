# BC + Residual RL

The planner owns reset and `Pre_Move`. The learned policy only receives frames where
`phase/id == 1` and controls the action phase.

At runtime the action is:

```text
z = multimodal_encoder(qpos, policy_step, cameras, tactile)
a_BC = qpos + delta_bc_head(z)
delta = SAC_or_PPO(z)
action = clip(a_BC + residual_scale * delta_std * delta_rl)
```

The encoder also receives normalized `phase/policy_step`, allowing it to represent
the acceleration and deceleration portions of the post-grasp trajectory.

Each policy action is held for two simulator steps, matching the demonstration
`save_frequency: 2` sampling interval.

For tasks where `Pre_Move` establishes the grasp, the learned action controls the
seven arm joints and keeps the planner's gripper target unchanged. Use
`--control-gripper` only for tasks whose action phase intentionally opens or closes it.
Joint commands use physical PD targets by default so FEM contact sees continuous motion.
`--force-control` is reserved for direct-position diagnostics.

Do not pass `--headless` for GelSight/UIPC training or evaluation. The tactile/contact
pipeline requires the same rendered application mode used during demonstration
collection; forcing headless mode can leave the grasp visually aligned but unable to
retain the object during lift.

The BC and RL feature extractors start from the same multimodal encoder weights. The
encoder can stay frozen for the first comparison or be fine-tuned with
`--no-freeze-encoder`.

Train BC:

```bash
python scripts/train_bc.py \
  data_gelsight_rl_100_20260725/grasp_in_clutter/gelsight_rl_100/hdf5 \
  policy/RL/runs/grasp_half_cylinder/bc
```

Train both residual policies in separate runs:

```bash
python scripts/train_rl.py grasp_in_clutter gelsight_rl_100 \
  policy/RL/runs/grasp_half_cylinder/bc/bc_best.pt \
  policy/RL/runs/grasp_half_cylinder --algorithm sac

python scripts/train_rl.py grasp_in_clutter gelsight_rl_100 \
  policy/RL/runs/grasp_half_cylinder/bc/bc_best.pt \
  policy/RL/runs/grasp_half_cylinder --algorithm ppo
```

`act_resnet18` matches the current ACT backbone family. Alternative visual encoders
use `timm:<model_name>`, for example
`timm:vit_small_patch16_dinov3.lvd1689m` or
`timm:vit_base_patch32_clip_224.openai`. LingBot variants can be selected the same
way when their timm model identifiers are locally registered, with weights supplied
through `--visual-checkpoint`. Tactile defaults to the ACT-style ResNet18.
Use `--image-size 224` for fixed-size ViT/CLIP backbones.
