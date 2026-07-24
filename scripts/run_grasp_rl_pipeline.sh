#!/usr/bin/env bash

set -euo pipefail

project_root=/root/gpufree-data/UniVTAC
python_bin=/opt/conda/envs/UniVTAC/bin/python
dataset_root="$project_root/data_gelsight_rl_100_20260725/grasp_half_cylinder_in_clutter/gelsight_rl_100"
run_root="$project_root/policy/RL/runs/grasp_half_cylinder_20260725"
timesteps="${1:-5000}"
evaluation_episodes="${2:-20}"

cd "$project_root"
mkdir -p "$run_root/logs" "$run_root/evaluation"

"$python_bin" scripts/validate_rl_dataset.py "$dataset_root" \
  --expected-episodes 100 --output "$run_root/dataset_validation.json" \
  2>&1 | tee "$run_root/logs/validate_dataset.log"

"$python_bin" scripts/train_bc.py "$dataset_root/hdf5" "$run_root/bc" \
  --epochs 30 --patience 5 --batch-size 32 --workers 4 --image-size 128 \
  --visual-pretrained --tactile-pretrained \
  2>&1 | tee "$run_root/logs/train_bc.log"

bc_checkpoint="$run_root/bc/bc_best.pt"

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root/evaluation" --algorithm bc \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  2>&1 | tee "$run_root/logs/eval_bc.log"

"$python_bin" scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root" --algorithm sac --total-timesteps "$timesteps" \
  2>&1 | tee "$run_root/logs/train_sac.log"

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root/evaluation" --algorithm sac \
  --model-path "$run_root/sac/final_model.zip" \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  2>&1 | tee "$run_root/logs/eval_sac.log"

"$python_bin" scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root" --algorithm ppo --total-timesteps "$timesteps" \
  2>&1 | tee "$run_root/logs/train_ppo.log"

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root/evaluation" --algorithm ppo \
  --model-path "$run_root/ppo/final_model.zip" \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  2>&1 | tee "$run_root/logs/eval_ppo.log"

"$python_bin" scripts/summarize_rl_results.py "$run_root" \
  2>&1 | tee "$run_root/logs/summarize.log"
