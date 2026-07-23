#!/usr/bin/env bash
# Train ACT on collected UniVTAC data.
#
# Usage:
#   bash bash_scripts/train_task.sh <task|all> <config|all> [episodes] [gpu] [train_config] [seed]
#
# Examples:
#   bash bash_scripts/train_task.sh insert_usb xense 100 0 train_config 0
#   bash bash_scripts/train_task.sh all neote 100 0 train_config 0
#   TACTILE_KEY=rgb_marker bash bash_scripts/train_task.sh insert_usb neote 100 0 train_config 0

set -euo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
TASK_ARG="${1:-all}"
CONFIG_ARG="${2:-neote}"
EXPERT_DATA_NUM="${3:-100}"
GPU="${4:-0}"
TRAIN_CONFIG="${5:-train_config}"
SEED="${6:-0}"
# Reprocess by default so ACT data follows the current full-trajectory HDF5
# convention, including pre_move frames when save_pre_move is enabled.
REPROCESS="${REPROCESS:-1}"

TASKS=(
    insert_usb
    insert_half_cylinder_into_box
    grasp_half_cylinder_in_clutter
    place_wooden_cube_on_yellow_area
    pull_drawer
    pour_ball_to_cup
    swap_cup_order
    turn_gear_pair
)

CONFIGS=(demo xense neote)

expand_arg() {
    local arg="$1"
    shift
    if [[ "${arg}" == "all" ]]; then
        printf "%s\n" "$@"
    else
        printf "%s\n" "${arg}"
    fi
}

default_tactile_key() {
    local config="$1"
    if [[ "${config}" == "neote" ]]; then
        printf "gel_particle"
    else
        printf "rgb_marker"
    fi
}

source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC

cd "${ROOT_DIR}"

mapfile -t RUN_TASKS < <(expand_arg "${TASK_ARG}" "${TASKS[@]}")
mapfile -t RUN_CONFIGS < <(expand_arg "${CONFIG_ARG}" "${CONFIGS[@]}")

for task in "${RUN_TASKS[@]}"; do
    for config in "${RUN_CONFIGS[@]}"; do
        tactile_key="${TACTILE_KEY:-$(default_tactile_key "${config}")}"
        raw_dir="${ROOT_DIR}/data/${task}/${config}/hdf5"
        processed_dir="${ROOT_DIR}/policy/ACT/data/sim-${task}/${config}-${EXPERT_DATA_NUM}"
        ckpt_dir="${ROOT_DIR}/policy/ACT/act_ckpt/act-${task}/${config}-${EXPERT_DATA_NUM}/${TRAIN_CONFIG}"

        if [[ ! -d "${raw_dir}" ]]; then
            echo "!!!!! Missing raw data: ${raw_dir}"
            exit 1
        fi

        echo
        echo "===== ACT train: task=${task}, config=${config}, episodes=${EXPERT_DATA_NUM}, tactile=${tactile_key} ====="

        processed_count=0
        if [[ -d "${processed_dir}" ]]; then
            processed_count=$(find "${processed_dir}" -maxdepth 1 -name 'episode_*.hdf5' | wc -l | tr -d ' ')
        fi

        if [[ "${REPROCESS}" == "1" || "${processed_count}" -lt "${EXPERT_DATA_NUM}" ]]; then
            if [[ "${REPROCESS}" != "1" && "${processed_count}" -gt 0 ]]; then
                echo "Processed data incomplete (${processed_count}/${EXPERT_DATA_NUM}), reprocessing: ${processed_dir}"
            fi
            (
                cd "${ROOT_DIR}/policy/ACT"
                TACTILE_KEY="${tactile_key}" python process_data.py "${task}" "${config}" "${EXPERT_DATA_NUM}"
            )
        else
            echo "Processed data exists (${processed_count}/${EXPERT_DATA_NUM}), skip: ${processed_dir}"
        fi

        (
            cd "${ROOT_DIR}/policy/ACT"
            CUDA_VISIBLE_DEVICES="${GPU}" TACTILE_KEY="${tactile_key}" python imitate_episodes.py \
                --task_name "sim-${task}-${config}-${EXPERT_DATA_NUM}" \
                --ckpt_dir "${ckpt_dir}" \
                --config_path "./${TRAIN_CONFIG}.yml" \
                --seed "${SEED}"
        )
    done
done
