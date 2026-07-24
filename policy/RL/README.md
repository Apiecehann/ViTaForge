# BC + Residual RL

The planner owns reset and `Pre_Move`. The learned policy only receives frames where
`phase/id == 1` and controls the action phase.

At runtime the action is:

```text
z = multimodal_encoder(qpos, cameras, tactile)
a_BC = bc_head(z)
delta = SAC_or_PPO(z)
action = clip(a_BC + residual_scale * action_std * delta)
```

Each policy action is held for two simulator steps, matching the demonstration
`save_frequency: 2` sampling interval.

The BC and RL feature extractors start from the same multimodal encoder weights. The
encoder can stay frozen for the first comparison or be fine-tuned with
`--no-freeze-encoder`.

Train BC:

```bash
python scripts/train_bc.py \
  data_gelsight_rl_100_20260725/grasp_half_cylinder_in_clutter/gelsight_rl_100/hdf5 \
  policy/RL/runs/grasp_half_cylinder/bc
```

Train both residual policies in separate runs:

```bash
python scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  policy/RL/runs/grasp_half_cylinder/bc/bc_best.pt \
  policy/RL/runs/grasp_half_cylinder --algorithm sac

python scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
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
