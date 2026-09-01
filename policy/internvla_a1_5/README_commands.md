```bash

ssh -N -L 8000:127.0.0.1:8000 -p 10751 root@42.192.34.154

cd /data/home/liqin/VLA/ViTaForge
conda activate UniVTAC


CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_USB.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_hardness_classify.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_hardness_classify.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50
```