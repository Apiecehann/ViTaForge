#!/usr/bin/env bash

set -euo pipefail

project_root=/root/gpufree-data/UniVTAC
python_bin=/opt/conda/envs/UniVTAC/bin/python
dataset_root="$project_root/data_gelsight_rl_100_20260725/grasp_half_cylinder_in_clutter/gelsight_rl_100"
run_root="$project_root/policy/RL/runs/grasp_half_cylinder_20260725"
timesteps="${1:-5000}"
evaluation_episodes="${2:-20}"

cd "$project_root"
evaluation_root="$run_root/evaluation_final"
bc_dir="$run_root/bc_phase"
bc_checkpoint="$bc_dir/bc_best.pt"

mkdir -p "$run_root/logs" "$evaluation_root"

"$python_bin" scripts/validate_rl_dataset.py "$dataset_root" \
  --expected-episodes 100 --output "$run_root/dataset_validation.json" \
  2>&1 | tee "$run_root/logs/validate_dataset.log"

if [[ ! -f "$bc_checkpoint" ]]; then
  "$python_bin" scripts/train_bc.py "$dataset_root/hdf5" "$bc_dir" \
    --epochs 30 --patience 5 --batch-size 32 --workers 4 --image-size 128 \
    --visual-pretrained --tactile-pretrained \
    2>&1 | tee "$run_root/logs/train_bc_phase.log"
fi

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$evaluation_root" --algorithm bc \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  --action-repeat 2 --step-limit 40 \
  2>&1 | tee "$run_root/logs/eval_bc_final.log"

"$python_bin" scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root" --algorithm sac --total-timesteps "$timesteps" \
  --action-repeat 2 --step-limit 40 \
  2>&1 | tee "$run_root/logs/train_sac_final.log"

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$evaluation_root" --algorithm sac \
  --model-path "$run_root/sac/final_model.zip" \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  --action-repeat 2 --step-limit 40 \
  2>&1 | tee "$run_root/logs/eval_sac_final.log"

"$python_bin" scripts/train_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root" --algorithm ppo --total-timesteps "$timesteps" \
  --action-repeat 2 --step-limit 40 \
  2>&1 | tee "$run_root/logs/train_ppo_final.log"

"$python_bin" scripts/eval_rl.py grasp_half_cylinder_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$evaluation_root" --algorithm ppo \
  --model-path "$run_root/ppo/final_model.zip" \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  --action-repeat 2 --step-limit 40 \
  2>&1 | tee "$run_root/logs/eval_ppo_final.log"

"$python_bin" scripts/summarize_rl_results.py "$run_root" \
  --evaluation-dir "$(basename "$evaluation_root")" \
  2>&1 | tee "$run_root/logs/summarize.log"
