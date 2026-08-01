#!/usr/bin/env bash

set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON_BIN:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
python_bin="${python_bin:-$(command -v python)}"
source_dataset_root="${SOURCE_DATASET_ROOT:-$project_root/data_gelsight_rl_100/grasp_in_clutter/gelsight_rl_100}"
run_root="${RUN_ROOT:-$project_root/policy/RL/runs/grasp_half_cylinder_gelsight}"
dataset_root="$run_root/dataset_layout_b"
timesteps="${1:-20000}"
evaluation_episodes="${2:-20}"
train_evaluation_episodes="${3:-10}"
run_ppo="${RUN_PPO:-0}"
sft_train_min_success_rate="${SFT_TRAIN_MIN_SUCCESS_RATE:-0.70}"
sft_heldout_min_success_rate="${SFT_HELDOUT_MIN_SUCCESS_RATE:-0.20}"

cd "$project_root"
evaluation_root="$run_root/evaluation_final"
train_evaluation_root="$run_root/evaluation_train_seeds"
bc_dir="$run_root/bc_phase"
bc_checkpoint="$bc_dir/bc_best.pt"

mkdir -p "$run_root/logs" "$evaluation_root" "$train_evaluation_root" "$dataset_root/hdf5"

# Collection was restarted after seed 2. Seeds 3-99 share the explicit layout
# recorded in task_config/gelsight_rl_100.yml and are the reproducible SFT set.
for episode_id in $(seq 3 99); do
  source_path="$source_dataset_root/hdf5/$episode_id.hdf5"
  target_path="$dataset_root/hdf5/$episode_id.hdf5"
  if [[ ! -f "$source_path" ]]; then
    echo "Missing source episode: $source_path" >&2
    exit 1
  fi
  if [[ ! -e "$target_path" ]]; then
    ln -s "$source_path" "$target_path"
  fi
done

"$python_bin" scripts/validate_rl_dataset.py "$dataset_root" \
  --expected-episodes 97 --output "$run_root/dataset_validation.json" \
  2>&1 | tee "$run_root/logs/validate_dataset.log"

if [[ ! -f "$bc_checkpoint" ]]; then
  "$python_bin" scripts/train_bc.py "$dataset_root/hdf5" "$bc_dir" \
    --epochs 30 --patience 5 --batch-size 32 --workers 4 --image-size 128 \
    --visual-pretrained --tactile-pretrained \
    2>&1 | tee "$run_root/logs/train_bc_phase.log"
fi

if [[ ! -f "$train_evaluation_root/sft/evaluation.json" ]]; then
  "$python_bin" scripts/eval_rl.py grasp_in_clutter gelsight_rl_100 \
    "$bc_checkpoint" "$train_evaluation_root" --algorithm sft \
    --episodes "$train_evaluation_episodes" --start-seed 3 \
    --control-mode direct --action-repeat 2 --step-limit 120 \
    --control-gripper --force-control --save-traces \
    2>&1 | tee "$run_root/logs/eval_sft_train_seeds.log"
fi

"$python_bin" - "$train_evaluation_root/sft/evaluation.json" \
  "$sft_train_min_success_rate" "training-seed SFT" <<'PY'
import json
import sys

result_path, threshold, label = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(result_path, "r", encoding="utf-8") as result_file:
    success_rate = float(json.load(result_file)["success_rate"])
print(f"{label} success rate: {success_rate:.3f} (required: {threshold:.3f})")
if success_rate < threshold:
    raise SystemExit(f"{label} gate failed; refusing to start RL")
PY

if [[ ! -f "$evaluation_root/sft/evaluation.json" ]]; then
  "$python_bin" scripts/eval_rl.py grasp_in_clutter gelsight_rl_100 \
    "$bc_checkpoint" "$evaluation_root" --algorithm sft \
    --episodes "$evaluation_episodes" --start-seed 20000 \
    --control-mode direct --action-repeat 2 --step-limit 120 \
    --control-gripper --force-control --save-traces \
    2>&1 | tee "$run_root/logs/eval_sft_heldout.log"
fi

"$python_bin" - "$evaluation_root/sft/evaluation.json" \
  "$sft_heldout_min_success_rate" "held-out SFT" <<'PY'
import json
import sys

result_path, threshold, label = sys.argv[1], float(sys.argv[2]), sys.argv[3]
with open(result_path, "r", encoding="utf-8") as result_file:
    success_rate = float(json.load(result_file)["success_rate"])
print(f"{label} success rate: {success_rate:.3f} (required: {threshold:.3f})")
if success_rate < threshold:
    raise SystemExit(f"{label} gate failed; refusing to start RL")
PY

"$python_bin" scripts/train_rl.py grasp_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$run_root" --algorithm sac --total-timesteps "$timesteps" \
  --bc-dataset-root "$dataset_root" \
  --control-mode direct --action-repeat 2 --step-limit 120 \
  --control-gripper --force-control \
  2>&1 | tee "$run_root/logs/train_sac_final.log"

"$python_bin" scripts/eval_rl.py grasp_in_clutter gelsight_rl_100 \
  "$bc_checkpoint" "$evaluation_root" --algorithm sac \
  --model-path "$run_root/sac/final_model.zip" \
  --episodes "$evaluation_episodes" --start-seed 20000 \
  --control-mode direct --action-repeat 2 --step-limit 120 \
  --control-gripper --force-control --save-traces \
  2>&1 | tee "$run_root/logs/eval_sac_final.log"

if [[ "$run_ppo" == "1" ]]; then
    "$python_bin" scripts/train_rl.py grasp_in_clutter gelsight_rl_100 \
      "$bc_checkpoint" "$run_root" --algorithm ppo --total-timesteps "$timesteps" \
    --control-mode direct --action-repeat 2 --step-limit 120 \
    --control-gripper --force-control --no-initialize-actor \
    2>&1 | tee "$run_root/logs/train_ppo_final.log"

  "$python_bin" scripts/eval_rl.py grasp_in_clutter gelsight_rl_100 \
    "$bc_checkpoint" "$evaluation_root" --algorithm ppo \
    --model-path "$run_root/ppo/final_model.zip" \
    --episodes "$evaluation_episodes" --start-seed 20000 \
    --control-mode direct --action-repeat 2 --step-limit 120 \
    --control-gripper --force-control --save-traces \
    2>&1 | tee "$run_root/logs/eval_ppo_final.log"
fi

"$python_bin" scripts/summarize_rl_results.py "$run_root" \
  --evaluation-dir "$(basename "$evaluation_root")" \
  2>&1 | tee "$run_root/logs/summarize.log"

sac_video_dir="$evaluation_root/sac/video"
sac_video_count="$(find "$sac_video_dir" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
if [[ "$sac_video_count" -ne "$evaluation_episodes" ]]; then
  echo "Expected $evaluation_episodes SAC evaluation videos, found $sac_video_count" >&2
  exit 1
fi

report_dir="$run_root/report"
mkdir -p "$report_dir"
find "$sac_video_dir" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | sort -V > "$report_dir/sac_video_manifest.txt"

"$python_bin" scripts/generate_rl_report.py \
  "$dataset_root" "$run_root" "$report_dir/RL_Policy_Data_Collection_Report.pdf" \
  2>&1 | tee "$run_root/logs/generate_report.log"
