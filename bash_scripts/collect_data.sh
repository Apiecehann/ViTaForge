#!/usr/bin/env bash
# Run data-collection smoke checks for the non-Xense tactile sensors.
# Defaults are intentionally light: one seed per task, video_frequency from
# each task_config, and output under ./data.
#
# Examples:
#   bash bash_scripts/collect_data.sh
#   SENSORS="gelsight neote" TASKS="insert_usb pull_drawer" bash bash_scripts/collect_data.sh
#   EPISODE_NUM=1 START_SEED=0 MAX_SEED=10 GPU=0 bash bash_scripts/collect_data.sh
#   RUN_XENSE=1 SENSORS="xense" bash bash_scripts/collect_data.sh

set -u

source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC

cd /root/gpufree-data/UniVTAC

EPISODE_NUM="${EPISODE_NUM:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-0}"
GPU="${GPU:-0}"
SENSORS="${SENSORS:-gelsight neote}"
TASKS="${TASKS:-insert_usb insert_half_cylinder_into_box grasp_half_cylinder_in_clutter place_wooden_cube_on_yellow_area pull_drawer pour_ball_to_cup swap_cup_order turn_gear_pair}"
RUN_XENSE="${RUN_XENSE:-0}"

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

run_sensor() {
    local sensor="$1"
    local config=""

    case "${sensor}" in
        gelsight|gelsight_mini|gsmini|demo)
            config="gelsight"
            ;;
        neote|xinzhi|新智)
            config="neote"
            ;;
        xense|xensews)
            if [[ "${RUN_XENSE}" != "1" ]]; then
                echo "Skip Xense by default. Set RUN_XENSE=1 to enable it."
                return 0
            fi
            config="xense"
            ;;
        *)
            echo "Unknown sensor '${sensor}'. Valid: gelsight, neote, xense."
            status=1
            return 0
            ;;
    esac

    echo
    echo "========== Sensor: ${sensor} -> config: ${config} =========="
    for task in ${TASKS}; do
        run_collect "${task}" "${config}"
    done
}

for sensor in ${SENSORS}; do
    run_sensor "${sensor}"
done

exit "${status}"
