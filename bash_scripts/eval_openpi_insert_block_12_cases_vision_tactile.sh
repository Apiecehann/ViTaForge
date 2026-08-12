#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "UniVTAC" ]]; then
  if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Could not find conda. Activate UniVTAC before running this script."
    exit 2
  fi
  conda activate UniVTAC
fi

# cube, layout0: 0,1,4
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 0 \
  --max_seed 0 \
  --total_num 1

# cube, layout1: 1,3,2
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 1000 \
  --max_seed 1000 \
  --total_num 1

# cube, layout2: 2,4,0
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block cube \
  --block_base_pose_indices 2,4,0 \
  --start_seed 2000 \
  --max_seed 2000 \
  --total_num 1

# cube, layout3: 3,2,1
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block cube \
  --block_base_pose_indices 3,2,1 \
  --start_seed 3000 \
  --max_seed 3000 \
  --total_num 1

# half_cylinder, layout0: 0,1,4
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 0 \
  --max_seed 0 \
  --total_num 1

# half_cylinder, layout1: 1,3,2
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 1000 \
  --max_seed 1000 \
  --total_num 1

# half_cylinder, layout2: 2,4,0
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block half_cylinder \
  --block_base_pose_indices 2,4,0 \
  --start_seed 2000 \
  --max_seed 2000 \
  --total_num 1

# half_cylinder, layout3: 3,2,1
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block half_cylinder \
  --block_base_pose_indices 3,2,1 \
  --start_seed 3000 \
  --max_seed 3000 \
  --total_num 1

# hexagon, layout0: 0,1,4
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 0 \
  --max_seed 0 \
  --total_num 1

# hexagon, layout1: 1,3,2
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 1000 \
  --max_seed 1000 \
  --total_num 1

# hexagon, layout2: 2,4,0
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block hexagon \
  --block_base_pose_indices 2,4,0 \
  --start_seed 2000 \
  --max_seed 2000 \
  --total_num 1

# hexagon, layout3: 3,2,1
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --target_block hexagon \
  --block_base_pose_indices 3,2,1 \
  --start_seed 3000 \
  --max_seed 3000 \
  --total_num 1
