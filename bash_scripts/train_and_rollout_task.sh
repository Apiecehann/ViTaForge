#!/usr/bin/env bash
# Train ACT for each selected UniVTAC task/config pair, then immediately run rollout.
#
# Usage:
#   bash bash_scripts/train_and_rollout_task.sh <task|all> <config|all> [episodes] [gpu] [train_config] [seed] [rollout_total] [start_seed] [max_seed] [deploy_config]
#
# Examples:
#   bash bash_scripts/train_and_rollout_task.sh insert_usb neote 100 0 train_config 0 20
#   bash bash_scripts/train_and_rollout_task.sh all neote 100 0 train_config 0 20
#   REPROCESS=0 ROLLOUT_WORKERS=1 bash bash_scripts/train_and_rollout_task.sh all neote 100 0 train_config 0 20

set -euo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
TASK_ARG="${1:-all}"
CONFIG_ARG="${2:-neote}"
EXPERT_DATA_NUM="${3:-100}"
GPU="${4:-0}"
TRAIN_CONFIG="${5:-train_config}"
SEED="${6:-0}"
ROLLOUT_TOTAL="${7:-20}"
START_SEED="${8:--1}"
MAX_SEED="${9:--1}"
DEPLOY_CONFIG="${10:-ACT/deploy}"

# Keep train_task.sh's correctness-first default unless the caller overrides it.
REPROCESS="${REPROCESS:-1}"
# ROLLOUT_WORKERS avoids accidentally changing training behavior through WORKERS.
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-${WORKERS:-1}}"

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

cd "${ROOT_DIR}"

mapfile -t RUN_TASKS < <(expand_arg "${TASK_ARG}" "${TASKS[@]}")
mapfile -t RUN_CONFIGS < <(expand_arg "${CONFIG_ARG}" "${CONFIGS[@]}")

for task in "${RUN_TASKS[@]}"; do
    for config in "${RUN_CONFIGS[@]}"; do
        echo
        echo "========== ACT train + rollout: task=${task}, config=${config}, episodes=${EXPERT_DATA_NUM} =========="
        echo "Train config: ${TRAIN_CONFIG}; seed: ${SEED}; gpu: ${GPU}; reprocess: ${REPROCESS}"
        echo "Rollout total: ${ROLLOUT_TOTAL}; start_seed: ${START_SEED}; max_seed: ${MAX_SEED}; workers: ${ROLLOUT_WORKERS}; deploy: ${DEPLOY_CONFIG}"

        REPROCESS="${REPROCESS}" bash bash_scripts/train_task.sh \
            "${task}" "${config}" "${EXPERT_DATA_NUM}" "${GPU}" "${TRAIN_CONFIG}" "${SEED}"

        WORKERS="${ROLLOUT_WORKERS}" bash bash_scripts/roll_out.sh \
            "${task}" "${config}" "${GPU}" "${TRAIN_CONFIG}" "${EXPERT_DATA_NUM}" \
            "${ROLLOUT_TOTAL}" "${START_SEED}" "${MAX_SEED}" "${DEPLOY_CONFIG}"
    done
done
