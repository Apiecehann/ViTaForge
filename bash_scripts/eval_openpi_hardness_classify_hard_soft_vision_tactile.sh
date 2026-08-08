#!/usr/bin/env bash
set -euo pipefail

cd /home/fudan/Workspace/yfwu/ViTaForge

OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --openpi_host 10.176.42.49 \
  --openpi_port 8001 \
  --hardness_label hard \
  --start_seed 0 \
  --max_seed 49 \
  --total_num 50

OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  gelsight \
  openpi/eef_delta/deploy_vision_tactile \
  --tactile_sensor gelsight \
  --openpi_host 10.176.42.49 \
  --openpi_port 8001 \
  --hardness_label soft \
  --start_seed 0 \
  --max_seed 49 \
  --total_num 50
