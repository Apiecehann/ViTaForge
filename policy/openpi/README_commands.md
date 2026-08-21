
```bash
ssh -N -L 8000:127.0.0.1:8000 -p 10383 root@42.192.34.154

ssh -N -L 8000:127.0.0.1:8000 -p 10751 root@42.192.34.154

cd /data/home/liqin/VLA/ViTaForge
conda activate UniVTAC

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_turn_gear_pair_pi05_relative_action_vision.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100


CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_insert_USB_vision_relative.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

```