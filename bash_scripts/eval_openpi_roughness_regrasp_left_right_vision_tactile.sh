#!/usr/bin/env bash
set -euo pipefail

cd /home/fudan/Workspace/yfwu/ViTaForge

OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --openpi_host 127.0.0.1 \
  --openpi_port 8000 \
  --rough_block_side left \
  --initial_grasp_side random \
  --start_seed 0 \
  --max_seed 49 \
  --total_num 50

OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --openpi_host 127.0.0.1 \
  --openpi_port 8000 \
  --rough_block_side right \
  --initial_grasp_side random \
  --start_seed 0 \
  --max_seed 49 \
  --total_num 50
