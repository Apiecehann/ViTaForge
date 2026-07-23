cd /root/gpufree-data/UniVTAC
source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC

python scripts/collect_data.py insert_usb xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py grasp_half_cylinder_in_clutter xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py place_wooden_cube_on_yellow_area xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py turn_gear_pair xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py swap_cup_order xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py insert_half_cylinder_into_box xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py pour_ball_to_cup xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0
python scripts/collect_data.py pull_drawer xense --episode_num 1 --start_seed 0 --max_seed 0 --gpu 0