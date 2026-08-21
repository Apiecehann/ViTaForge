```bash

ssh -N -L 8000:127.0.0.1:8000 -p 10383 root@42.192.34.154

ssh -N -L 8000:127.0.0.1:8000 -p 10751 root@42.192.34.154


cd /data/home/liqin/VLA/ViTaForge
conda activate UniVTAC

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_turn_gear_pair.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_regrasp.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_USB.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_classify.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

  ```