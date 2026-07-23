#!/usr/bin/env bash
# Package one successful GelSight and one successful Neote video for every UniVTAC task.
#
# It first reuses existing *_success.mp4 videos from data_compare/, data/, and data_Neto/.
# If a task/sensor is missing and RUN_COLLECT=1, it runs collect_data.py over START_SEED..MAX_SEED
# until one successful video is produced.
#
# Output layout:
#   success_task_videos/<RUN_TAG>/<task>/gelsight_success.mp4
#   success_task_videos/<RUN_TAG>/<task>/neote_success.mp4
#   success_task_videos/<RUN_TAG>/manifest.tsv
#
# Typical usage:
#   cd /root/gpufree-data/UniVTAC
#   bash bash_scripts/pack_gelsight_neote_success_videos.sh
#
# Reuse existing only, do not launch IsaacSim:
#   RUN_COLLECT=0 bash bash_scripts/pack_gelsight_neote_success_videos.sh
#
# Only try missing hard tasks:
#   TASKS="insert_usb insert_half_cylinder_into_box pour_ball_to_cup" START_SEED=1 MAX_SEED=80 bash bash_scripts/pack_gelsight_neote_success_videos.sh

set -uo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
cd "${ROOT_DIR}" || exit 1

if [[ ! -f /opt/conda/etc/profile.d/conda.sh ]]; then
    echo "!!!!! Missing conda profile: /opt/conda/etc/profile.d/conda.sh"
    exit 1
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC || exit 1

RUN_TAG="${RUN_TAG:-success_videos_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/success_task_videos/${RUN_TAG}}"
COLLECT_ROOT="${COLLECT_ROOT:-${ROOT_DIR}/success_task_videos/${RUN_TAG}/collected}"
CFG_DIR="${OUT_ROOT}/task_config"
LOG_DIR="${ROOT_DIR}/logs/gelsight_neote_success"
LOG_PATH="${LOG_DIR}/${RUN_TAG}.log"

RUN_COLLECT="${RUN_COLLECT:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-80}"
GPU="${GPU:-0}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-2}"
VIDEO_FREQUENCY="${VIDEO_FREQUENCY:-2}"
RENDER_FREQUENCY="${RENDER_FREQUENCY:-0}"
SAVE_PRE_MOVE="${SAVE_PRE_MOVE:-true}"

DEFAULT_TASKS=(
    insert_usb
    insert_half_cylinder_into_box
    grasp_half_cylinder_in_clutter
    place_wooden_cube_on_yellow_area
    pull_drawer
    pour_ball_to_cup
    swap_cup_order
    turn_gear_pair
)

if [[ -n "${TASKS:-}" ]]; then
    # shellcheck disable=SC2206
    RUN_TASKS=(${TASKS})
else
    RUN_TASKS=("${DEFAULT_TASKS[@]}")
fi

mkdir -p "${OUT_ROOT}" "${CFG_DIR}" "${LOG_DIR}" "${COLLECT_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

GELSIGHT_CFG="${CFG_DIR}/success_gelsight.yml"
NEOTE_CFG="${CFG_DIR}/success_neote.yml"
MANIFEST="${OUT_ROOT}/manifest.tsv"
MISSING="${OUT_ROOT}/missing.tsv"

cat > "${GELSIGHT_CFG}" <<EOF
save_dir: ${COLLECT_ROOT}

decimation: 1

save_frequency: ${SAVE_FREQUENCY}
video_frequency: ${VIDEO_FREQUENCY}
render_frequency: ${RENDER_FREQUENCY}

random_texture: false

use_seed: true
episode_num: 1

sensor_type: gsmini

observations:
  camera: ['rgb']
  tactile: ['rgb', 'rgb_marker', 'marker', 'depth', 'pose']
  embodiment: ['joint', 'ee']
  actor: true

save_pre_move: ${SAVE_PRE_MOVE}
EOF

cat > "${NEOTE_CFG}" <<EOF
save_dir: ${COLLECT_ROOT}

decimation: 1

save_frequency: ${SAVE_FREQUENCY}
video_frequency: ${VIDEO_FREQUENCY}
render_frequency: ${RENDER_FREQUENCY}

random_texture: false

use_seed: true
episode_num: 1

sensor_type: neote
dense_gelpad: false
force_field_grid: [64, 48]

observations:
  camera: ['rgb']
  tactile:
    - rgb
    - rgb_marker
    - marker
    - depth
    - pose
    - vertex_force
    - force_field
    - gel_particle
  embodiment: ['joint', 'ee']
  actor: true

save_pre_move: ${SAVE_PRE_MOVE}
tactile_video_key: gel_particle
EOF

printf "task\tsensor\tstatus\tpackaged_video\tsource_video\n" > "${MANIFEST}"
printf "task\tsensor\n" > "${MISSING}"

echo "========== Package GelSight/Neote successful videos =========="
echo "Time: $(date '+%F %T')"
echo "Root: ${ROOT_DIR}"
echo "Run tag: ${RUN_TAG}"
echo "Output root: ${OUT_ROOT}"
echo "Collect root: ${COLLECT_ROOT}"
echo "Log: ${LOG_PATH}"
echo "Tasks: ${RUN_TASKS[*]}"
echo "RUN_COLLECT=${RUN_COLLECT}, START_SEED=${START_SEED}, MAX_SEED=${MAX_SEED}, GPU=${GPU}"
echo "SAVE_FREQUENCY=${SAVE_FREQUENCY}, VIDEO_FREQUENCY=${VIDEO_FREQUENCY}, SAVE_PRE_MOVE=${SAVE_PRE_MOVE}"
echo

find_success_video() {
    local task="$1"
    local sensor="$2"

    if [[ "${sensor}" == "gelsight" ]]; then
        find \
            "${COLLECT_ROOT}" \
            "${ROOT_DIR}/data_compare" \
            "${ROOT_DIR}/data" \
            -path "*/${task}/*/video/*_success.mp4" -type f 2>/dev/null \
            | grep -E '/(success_gelsight|compare_gelsight|demo)/video/' \
            | sort -r \
            | head -1
    else
        find \
            "${COLLECT_ROOT}" \
            "${ROOT_DIR}/data_compare" \
            "${ROOT_DIR}/data" \
            "${ROOT_DIR}/data_Neto" \
            -path "*/${task}/*/video/*_success.mp4" -type f 2>/dev/null \
            | grep -E '/(success_neote|compare_neote|neote)/video/|/data_Neto/' \
            | sort -r \
            | head -1
    fi
}

collect_one_success() {
    local task="$1"
    local sensor="$2"
    local cfg="$3"

    echo
    echo "===== COLLECT missing ${sensor} / ${task}, seeds ${START_SEED}..${MAX_SEED} ====="
    PYTHONUNBUFFERED=1 python scripts/collect_data.py "${task}" "${cfg}" \
        --episode_num 1 \
        --start_seed "${START_SEED}" \
        --max_seed "${MAX_SEED}" \
        --gpu "${GPU}"
}

package_one() {
    local task="$1"
    local sensor="$2"
    local cfg="$3"
    local src dst task_dir

    task_dir="${OUT_ROOT}/${task}"
    mkdir -p "${task_dir}"
    dst="${task_dir}/${sensor}_success.mp4"

    src="$(find_success_video "${task}" "${sensor}")"
    if [[ -z "${src}" && "${RUN_COLLECT}" == "1" ]]; then
        collect_one_success "${task}" "${sensor}" "${cfg}"
        src="$(find_success_video "${task}" "${sensor}")"
    fi

    if [[ -n "${src}" && -f "${src}" ]]; then
        cp -f "${src}" "${dst}"
        printf "%s\t%s\tok\t%s\t%s\n" "${task}" "${sensor}" "${dst}" "${src}" >> "${MANIFEST}"
        echo "OK ${task} / ${sensor}: ${dst}"
        echo "   source: ${src}"
    else
        printf "%s\t%s\tmissing\t\t\n" "${task}" "${sensor}" >> "${MANIFEST}"
        printf "%s\t%s\n" "${task}" "${sensor}" >> "${MISSING}"
        echo "MISSING ${task} / ${sensor}"
    fi
}

for task in "${RUN_TASKS[@]}"; do
    echo
    echo "========== TASK ${task} =========="
    package_one "${task}" "gelsight" "${GELSIGHT_CFG}"
    package_one "${task}" "neote" "${NEOTE_CFG}"
done

echo
echo "========== Summary =========="
column -t -s $'\t' "${MANIFEST}" 2>/dev/null || cat "${MANIFEST}"
missing_count="$(tail -n +2 "${MISSING}" | wc -l | tr -d ' ')"
echo
echo "Missing count: ${missing_count}"
if [[ "${missing_count}" != "0" ]]; then
    echo "Missing list: ${MISSING}"
    cat "${MISSING}"
else
    echo "All requested task/sensor success videos are packaged."
fi
echo "Output root: ${OUT_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "Log: ${LOG_PATH}"

if [[ "${missing_count}" == "0" ]]; then
    exit 0
else
    exit 2
fi
