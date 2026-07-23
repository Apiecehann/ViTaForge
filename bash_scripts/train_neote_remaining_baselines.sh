#!/usr/bin/env bash
# Run the remaining Neote ACT baseline experiments:
#   1. Vision Only for the 5 tasks that already have Neote tactile checkpoints.
#   2. Vision + Neote tactile for the 3 tasks whose main Neote tactile run is still missing.
#
# Usage:
#   bash bash_scripts/train_neote_remaining_baselines.sh [episodes] [gpu] [seed] [rollout_total] [start_seed] [max_seed] [deploy_config]
#
# Common:
#   RUN_ROLLOUT=0 bash bash_scripts/train_neote_remaining_baselines.sh 100 0 0
#   REPROCESS=0 ROLLOUT_WORKERS=1 bash bash_scripts/train_neote_remaining_baselines.sh 100 0 0 100 1000000 1000099 ACT/deploy

set -euo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
EPISODES="${1:-100}"
GPU="${2:-0}"
SEED="${3:-0}"
ROLLOUT_TOTAL="${4:-100}"
START_SEED="${5:-1000000}"
MAX_SEED="${6:-1000099}"
DEPLOY_CONFIG="${7:-ACT/deploy}"

REPROCESS="${REPROCESS:-0}"
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
DRY_RUN="${DRY_RUN:-0}"

VISION_ONLY_TRAIN_CONFIG="${VISION_ONLY_TRAIN_CONFIG:-train_config_vision_only}"
NEOTE_TACTILE_TRAIN_CONFIG="${NEOTE_TACTILE_TRAIN_CONFIG:-train_config}"
NEOTE_TACTILE_KEY="${NEOTE_TACTILE_KEY:-gel_particle}"

VISION_ONLY_TASKS=(
    insert_usb
    insert_half_cylinder_into_box
    grasp_half_cylinder_in_clutter
    place_wooden_cube_on_yellow_area
    pull_drawer
)

NEOTE_TACTILE_TASKS=(
    pour_ball_to_cup
    swap_cup_order
    turn_gear_pair
)

cd "${ROOT_DIR}"

run_cmd() {
    echo "+ $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

run_experiment() {
    local task="$1"
    local train_config="$2"
    local tactile_key="$3"
    local label="$4"

    echo
    echo "========== ${label}: task=${task}, config=neote, episodes=${EPISODES} =========="
    echo "train_config=${train_config}; tactile_key=${tactile_key}; gpu=${GPU}; seed=${SEED}; reprocess=${REPROCESS}; rollout=${RUN_ROLLOUT}"

    if [[ "${RUN_ROLLOUT}" == "1" ]]; then
        run_cmd env REPROCESS="${REPROCESS}" ROLLOUT_WORKERS="${ROLLOUT_WORKERS}" TACTILE_KEY="${tactile_key}"             bash bash_scripts/train_and_rollout_task.sh             "${task}" neote "${EPISODES}" "${GPU}" "${train_config}" "${SEED}"             "${ROLLOUT_TOTAL}" "${START_SEED}" "${MAX_SEED}" "${DEPLOY_CONFIG}"
    else
        run_cmd env REPROCESS="${REPROCESS}" TACTILE_KEY="${tactile_key}"             bash bash_scripts/train_task.sh             "${task}" neote "${EPISODES}" "${GPU}" "${train_config}" "${SEED}"
    fi
}

for task in "${VISION_ONLY_TASKS[@]}"; do
    # TACTILE_KEY is only used by preprocessing here; train_config_vision_only has tactile_names: [].
    run_experiment "${task}" "${VISION_ONLY_TRAIN_CONFIG}" "${NEOTE_TACTILE_KEY}" "Vision Only"
done

for task in "${NEOTE_TACTILE_TASKS[@]}"; do
    run_experiment "${task}" "${NEOTE_TACTILE_TRAIN_CONFIG}" "${NEOTE_TACTILE_KEY}" "Vision + Neote tactile"
done
