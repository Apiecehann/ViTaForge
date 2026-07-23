#!/usr/bin/env bash
# Train and evaluate ACT Vision Only for UniVTAC pour_ball_to_cup on the neote dataset.
#
# Default:
#   - task: pour_ball_to_cup
#   - config/sensor dataset: neote
#   - train config: train_config_vision_only.yml
#   - processed data: reuse existing ACT data by default (REPROCESS=0)
#   - rollout workers: 1, for stable/debuggable evaluation
#
# Usage from /root/gpufree-data/UniVTAC:
#   bash bash_scripts/train_pour_vision_only.sh [episodes] [gpu] [seed] [rollout_total] [start_seed] [max_seed] [deploy_config]
#
# Examples:
#   REPROCESS=0 ROLLOUT_WORKERS=1 bash bash_scripts/train_pour_vision_only.sh
#   RUN_ROLLOUT=0 REPROCESS=0 bash bash_scripts/train_pour_vision_only.sh
#   RUN_TRAIN=0 ROLLOUT_WORKERS=1 bash bash_scripts/train_pour_vision_only.sh 100 0 0 20 1000000 1000019 ACT/deploy

set -euo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
TASK="pour_ball_to_cup"
CONFIG="neote"
TRAIN_CONFIG="train_config_vision_only"

EPISODES="${1:-100}"
GPU="${2:-0}"
SEED="${3:-0}"
ROLLOUT_TOTAL="${4:-20}"
START_SEED="${5:-1000000}"
MAX_SEED="${6:-1000099}"
DEPLOY_CONFIG="${7:-ACT/deploy}"

# Vision-only training ignores tactile inputs because train_config_vision_only.yml has tactile_names: [].
# This key is still needed only when process_data.py reads the raw hdf5 structure.
TACTILE_KEY="${TACTILE_KEY:-gel_particle}"

# Reuse processed ACT data by default. Previous full preprocessing for this task was heavy and could be killed.
REPROCESS="${REPROCESS:-0}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_ROLLOUT="${RUN_ROLLOUT:-1}"

cd "${ROOT_DIR}"
mkdir -p logs
LOG="logs/pour_ball_to_cup_vision_only_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

RAW_DIR="${ROOT_DIR}/data/${TASK}/${CONFIG}/hdf5"
PROCESSED_DIR="${ROOT_DIR}/policy/ACT/data/sim-${TASK}/${CONFIG}-${EPISODES}"
CKPT_DIR="${ROOT_DIR}/policy/ACT/act_ckpt/act-${TASK}/${CONFIG}-${EPISODES}/${TRAIN_CONFIG}"
CONFIG_FILE="${ROOT_DIR}/policy/ACT/${TRAIN_CONFIG}.yml"

echo "========== Pour ball to cup / Vision Only =========="
echo "Time: $(date '+%F %T')"
echo "Root: ${ROOT_DIR}"
echo "Task: ${TASK}"
echo "Dataset config: ${CONFIG}"
echo "Episodes: ${EPISODES}"
echo "GPU: ${GPU}"
echo "Seed: ${SEED}"
echo "Train config: ${CONFIG_FILE}"
echo "Raw data: ${RAW_DIR}"
echo "Processed data: ${PROCESSED_DIR}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "REPROCESS=${REPROCESS}"
echo "TACTILE_KEY=${TACTILE_KEY}  # used only for optional preprocessing"
echo "RUN_TRAIN=${RUN_TRAIN}"
echo "RUN_ROLLOUT=${RUN_ROLLOUT}"
echo "ROLLOUT_TOTAL=${ROLLOUT_TOTAL}"
echo "START_SEED=${START_SEED}"
echo "MAX_SEED=${MAX_SEED}"
echo "ROLLOUT_WORKERS=${ROLLOUT_WORKERS}"
echo "DEPLOY_CONFIG=${DEPLOY_CONFIG}"
echo "Log: ${ROOT_DIR}/${LOG}"
echo

if [[ ! -d "${RAW_DIR}" ]]; then
    echo "!!!!! Missing raw data: ${RAW_DIR}"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "!!!!! Missing train config: ${CONFIG_FILE}"
    exit 1
fi

echo "----- Vision-only config sanity check -----"
grep -nE '^(camera_names:|tactile_names:|num_steps:|save_freq:|batch_size:|lr:)' "${CONFIG_FILE}" || true
echo

processed_count=0
if [[ -d "${PROCESSED_DIR}" ]]; then
    processed_count="$(find "${PROCESSED_DIR}" -maxdepth 1 -name 'episode_*.hdf5' | wc -l | tr -d ' ')"
fi
echo "Processed episodes currently: ${processed_count}/${EPISODES}"
echo

if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "========== Training Vision Only ACT =========="
    REPROCESS="${REPROCESS}" TACTILE_KEY="${TACTILE_KEY}" bash bash_scripts/train_task.sh \
        "${TASK}" "${CONFIG}" "${EPISODES}" "${GPU}" "${TRAIN_CONFIG}" "${SEED}"
else
    echo "Skip training because RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_ROLLOUT}" == "1" ]]; then
    echo
    echo "========== Rollout Vision Only ACT =========="
    if [[ ! -f "${CKPT_DIR}/policy_last.ckpt" ]]; then
        echo "!!!!! Missing ACT checkpoint: ${CKPT_DIR}/policy_last.ckpt"
        echo "!!!!! If training is still running or failed, check: ${ROOT_DIR}/${LOG}"
        exit 1
    fi

    WORKERS="${ROLLOUT_WORKERS}" TACTILE_KEY="${TACTILE_KEY}" bash bash_scripts/roll_out.sh \
        "${TASK}" "${CONFIG}" "${GPU}" "${TRAIN_CONFIG}" "${EPISODES}" \
        "${ROLLOUT_TOTAL}" "${START_SEED}" "${MAX_SEED}" "${DEPLOY_CONFIG}"
else
    echo "Skip rollout because RUN_ROLLOUT=${RUN_ROLLOUT}"
fi

echo
echo "========== Done =========="
echo "Log saved to: ${ROOT_DIR}/${LOG}"
