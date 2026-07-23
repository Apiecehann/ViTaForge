#!/usr/bin/env bash
# Preview data collection outputs for the 3 supported tactile sensors across
# the 8 refined tasks. Override EPISODE_NUM / START_SEED / MAX_SEED / GPU when
# launching this script if needed.

set -u

source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC

cd /root/gpufree-data/UniVTAC

EPISODE_NUM="${EPISODE_NUM:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-50}"
GPU="${GPU:-0}"

status=0

run_collect() {
    local task="$1"
    local config="$2"

    echo
    echo "===== ${config} / ${task} ====="
    if ! python scripts/collect_data.py "${task}" "${config}" \
        --episode_num "${EPISODE_NUM}" \
        --start_seed "${START_SEED}" \
        --max_seed "${MAX_SEED}" \
        --gpu "${GPU}"; then
        echo "!!!!! FAILED: ${config} / ${task}"
        status=1
    fi
}

# Sensor: GelSight Mini
#run_collect insert_usb demo
#run_collect insert_half_cylinder_into_box demo
#run_collect grasp_half_cylinder_in_clutter demo
#run_collect place_wooden_cube_on_yellow_area demo
#run_collect pull_drawer demo
#run_collect pour_ball_to_cup demo
#run_collect swap_cup_order demo
#run_collect turn_gear_pair demo

# Sensor: XenseWS
#run_collect insert_usb xense
#run_collect insert_half_cylinder_into_box xense
#run_collect grasp_half_cylinder_in_clutter xense
run_collect place_wooden_cube_on_yellow_area xense
run_collect pull_drawer xense
run_collect pour_ball_to_cup xense
run_collect swap_cup_order xense
run_collect turn_gear_pair xense

# Sensor: Neote
#run_collect insert_usb neote
#run_collect insert_half_cylinder_into_box neote
#run_collect grasp_half_cylinder_in_clutter neote
#run_collect place_wooden_cube_on_yellow_area neote
#run_collect pull_drawer neote
#run_collect pour_ball_to_cup neote
#run_collect swap_cup_order neote
#run_collect turn_gear_pair neote

exit "${status}"
