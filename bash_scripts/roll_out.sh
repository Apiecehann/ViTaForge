#!/usr/bin/env bash
# Roll out an ACT checkpoint in UniVTAC simulation.
#
# Usage:
#   bash bash_scripts/roll_out.sh <task|all> <config|all> [gpu] [train_config] [episodes] [total_num] [start_seed] [max_seed] [deploy_config]
#
# Examples:
#   bash bash_scripts/roll_out.sh insert_usb xense 0 train_config 100 20
#   bash bash_scripts/roll_out.sh all neote 0 train_config 100 20
#   TACTILE_KEY=rgb_marker bash bash_scripts/roll_out.sh insert_usb neote 0 train_config 100 20

set -euo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
TASK_ARG="${1:-insert_usb}"
CONFIG_ARG="${2:-demo}"
GPU="${3:-0}"
TRAIN_CONFIG="${4:-train_config}"
EXPERT_DATA_NUM="${5:-100}"
TOTAL_NUM="${6:-20}"
START_SEED="${7:--1}"
MAX_SEED="${8:--1}"
DEPLOY_CONFIG="${9:-ACT/deploy}"
WORKERS="${WORKERS:-1}"

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
        ckpt_dir="${ROOT_DIR}/policy/ACT/act_ckpt/act-${task}/${config}-${EXPERT_DATA_NUM}/${TRAIN_CONFIG}"

        if [[ ! -f "${ckpt_dir}/policy_last.ckpt" ]]; then
            echo "!!!!! Missing ACT checkpoint: ${ckpt_dir}/policy_last.ckpt"
            exit 1
        fi

        echo
        echo "===== ACT rollout: task=${task}, config=${config}, episodes=${EXPERT_DATA_NUM}, total=${TOTAL_NUM}, tactile=${tactile_key} ====="

        if [[ "${WORKERS}" == "1" ]]; then
            CUDA_VISIBLE_DEVICES="${GPU}" TRAIN_CONFIG="${TRAIN_CONFIG}" EP_NUM="${EXPERT_DATA_NUM}" TACTILE_KEY="${tactile_key}" \
                python scripts/eval_policy.py "${task}" "${config}" "${DEPLOY_CONFIG}" \
                    --total_num "${TOTAL_NUM}" \
                    --start_seed "${START_SEED}" \
                    --max_seed "${MAX_SEED}"
        else
            if [[ "${START_SEED}" != "-1" || "${MAX_SEED}" != "-1" ]]; then
                echo "Parallel rollout ignores explicit START_SEED/MAX_SEED; use WORKERS=1 if you need fixed seed bounds."
            fi
            TRAIN_CONFIG="${TRAIN_CONFIG}" EP_NUM="${EXPERT_DATA_NUM}" TACTILE_KEY="${tactile_key}" \
                python scripts/parallel_eval_policy.py "${task}" "${config}" "${DEPLOY_CONFIG}" \
                    --total_num "${TOTAL_NUM}" \
                    --workers "${WORKERS}" \
                    --gpu "${GPU}"
        fi
    done
done
