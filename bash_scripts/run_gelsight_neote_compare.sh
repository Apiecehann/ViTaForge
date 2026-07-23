#!/usr/bin/env bash
# Collect and compare GelSight Mini (demo) vs Neote tactile videos for UniVTAC tasks.
#
# This script does not modify task_config/demo.yml or task_config/neote.yml.
# It writes temporary per-run configs under data_compare/<RUN_TAG>/task_config/
# so that the resulting HDF5/video files are isolated and easy to inspect.
#
# Default smoke/visual check:
#   EPISODE_NUM=1, START_SEED=0, MAX_SEED=0, VIDEO_FREQUENCY=2, SAVE_PRE_MOVE=true
#
# Typical usage:
#   cd /root/gpufree-data/UniVTAC
#   bash bash_scripts/run_gelsight_neote_compare.sh
#
# One task only:
#   TASKS="pour_ball_to_cup" bash bash_scripts/run_gelsight_neote_compare.sh
#
# Try seeds 0..5 and keep collecting until one success per sensor/task:
#   EPISODE_NUM=1 START_SEED=0 MAX_SEED=5 bash bash_scripts/run_gelsight_neote_compare.sh
#
# Only build side-by-side videos from an existing run:
#   RUN_COLLECT=0 RUN_TAG=<existing_tag> bash bash_scripts/run_gelsight_neote_compare.sh

set -uo pipefail

ROOT_DIR="/root/gpufree-data/UniVTAC"
cd "${ROOT_DIR}" || exit 1

if [[ ! -f /opt/conda/etc/profile.d/conda.sh ]]; then
    echo "!!!!! Missing conda profile: /opt/conda/etc/profile.d/conda.sh"
    exit 1
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate UniVTAC || exit 1

RUN_TAG="${RUN_TAG:-gelsight_neote_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/data_compare/${RUN_TAG}}"
CFG_DIR="${OUT_ROOT}/task_config"
LOG_DIR="${ROOT_DIR}/logs/gelsight_neote_compare"
LOG_PATH="${LOG_DIR}/${RUN_TAG}.log"

EPISODE_NUM="${EPISODE_NUM:-1}"
START_SEED="${START_SEED:-0}"
MAX_SEED="${MAX_SEED:-0}"
GPU="${GPU:-0}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-2}"
VIDEO_FREQUENCY="${VIDEO_FREQUENCY:-2}"
RENDER_FREQUENCY="${RENDER_FREQUENCY:-0}"
SAVE_PRE_MOVE="${SAVE_PRE_MOVE:-true}"
RUN_COLLECT="${RUN_COLLECT:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
COMPARE_HEIGHT="${COMPARE_HEIGHT:-720}"
COMPARE_FPS="${COMPARE_FPS:-30}"
EXTRACT_FRAMES="${EXTRACT_FRAMES:-0}"

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

mkdir -p "${CFG_DIR}" "${LOG_DIR}" "${OUT_ROOT}/side_by_side"
exec > >(tee -a "${LOG_PATH}") 2>&1

GELSIGHT_CFG="${CFG_DIR}/compare_gelsight.yml"
NEOTE_CFG="${CFG_DIR}/compare_neote.yml"

cat > "${GELSIGHT_CFG}" <<EOF
save_dir: ${OUT_ROOT}

decimation: 1

save_frequency: ${SAVE_FREQUENCY}
video_frequency: ${VIDEO_FREQUENCY}
render_frequency: ${RENDER_FREQUENCY}

random_texture: false

use_seed: true
episode_num: ${EPISODE_NUM}

sensor_type: gsmini

observations:
  camera: ['rgb']
  tactile: ['rgb', 'rgb_marker', 'marker', 'depth', 'pose']
  embodiment: ['joint', 'ee']
  actor: true

save_pre_move: ${SAVE_PRE_MOVE}
EOF

cat > "${NEOTE_CFG}" <<EOF
save_dir: ${OUT_ROOT}

decimation: 1

save_frequency: ${SAVE_FREQUENCY}
video_frequency: ${VIDEO_FREQUENCY}
render_frequency: ${RENDER_FREQUENCY}

random_texture: false

use_seed: true
episode_num: ${EPISODE_NUM}

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

echo "========== GelSight vs Neote collection/compare =========="
echo "Time: $(date '+%F %T')"
echo "Root: ${ROOT_DIR}"
echo "Run tag: ${RUN_TAG}"
echo "Output root: ${OUT_ROOT}"
echo "Log: ${LOG_PATH}"
echo "Tasks: ${RUN_TASKS[*]}"
echo "EPISODE_NUM=${EPISODE_NUM}, START_SEED=${START_SEED}, MAX_SEED=${MAX_SEED}, GPU=${GPU}"
echo "SAVE_FREQUENCY=${SAVE_FREQUENCY}, VIDEO_FREQUENCY=${VIDEO_FREQUENCY}, SAVE_PRE_MOVE=${SAVE_PRE_MOVE}"
echo "RUN_COLLECT=${RUN_COLLECT}, RUN_COMPARE=${RUN_COMPARE}, EXTRACT_FRAMES=${EXTRACT_FRAMES}"
echo "GelSight config: ${GELSIGHT_CFG}"
echo "Neote config: ${NEOTE_CFG}"
echo

status=0

run_collect() {
    local task="$1"
    local label="$2"
    local cfg="$3"

    echo
    echo "===== COLLECT ${label} / ${task} ====="
    PYTHONUNBUFFERED=1 python scripts/collect_data.py "${task}" "${cfg}" \
        --episode_num "${EPISODE_NUM}" \
        --start_seed "${START_SEED}" \
        --max_seed "${MAX_SEED}" \
        --gpu "${GPU}"
    local rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        echo "!!!!! FAILED with exit code ${rc}: ${label} / ${task}"
        status=1
    fi
}

find_video() {
    local root="$1"
    local seed="$2"
    local video_dir="${root}/video"
    local candidate

    for suffix in success fail error ""; do
        if [[ -n "${suffix}" ]]; then
            candidate="${video_dir}/${seed}_${suffix}.mp4"
        else
            candidate="${video_dir}/${seed}.mp4"
        fi
        if [[ -f "${candidate}" ]]; then
            printf "%s" "${candidate}"
            return 0
        fi
    done

    candidate="$(find "${video_dir}" -maxdepth 1 -type f -name "${seed}*.mp4" 2>/dev/null | sort | head -1)"
    if [[ -n "${candidate}" ]]; then
        printf "%s" "${candidate}"
        return 0
    fi
    return 1
}

video_status() {
    local path="$1"
    local name
    name="$(basename "${path}")"
    case "${name}" in
        *_success.mp4) printf "success" ;;
        *_fail.mp4) printf "fail" ;;
        *_error.mp4) printf "error" ;;
        *) printf "unknown" ;;
    esac
}

build_compare_for_seed() {
    local task="$1"
    local seed="$2"
    local gelsight_root="${OUT_ROOT}/${task}/compare_gelsight"
    local neote_root="${OUT_ROOT}/${task}/compare_neote"
    local gelsight_video neote_video

    if ! gelsight_video="$(find_video "${gelsight_root}" "${seed}")"; then
        echo "----- No GelSight video for ${task} seed ${seed}"
        return 0
    fi
    if ! neote_video="$(find_video "${neote_root}" "${seed}")"; then
        echo "----- No Neote video for ${task} seed ${seed}"
        return 0
    fi

    local gs_status neote_status out_video frame_dir
    gs_status="$(video_status "${gelsight_video}")"
    neote_status="$(video_status "${neote_video}")"
    out_video="${OUT_ROOT}/side_by_side/${task}_seed${seed}_gelsight-${gs_status}_neote-${neote_status}.mp4"

    echo "===== COMPARE ${task} seed ${seed}: GelSight=${gs_status}, Neote=${neote_status} ====="
    echo "GelSight video: ${gelsight_video}"
    echo "Neote video:    ${neote_video}"
    echo "Compare video:  ${out_video}"

    if ! ffmpeg -y -loglevel warning \
        -i "${gelsight_video}" \
        -i "${neote_video}" \
        -filter_complex "[0:v]scale=-2:${COMPARE_HEIGHT},setsar=1[left];[1:v]scale=-2:${COMPARE_HEIGHT},setsar=1[right];[left][right]hstack=inputs=2:shortest=1[v]" \
        -map "[v]" \
        -r "${COMPARE_FPS}" \
        -pix_fmt yuv420p \
        "${out_video}"; then
        echo "!!!!! Failed to build comparison video: ${out_video}"
        status=1
        return 0
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${task}" "${seed}" "${gs_status}" "${neote_status}" \
        "${gelsight_video}" "${neote_video}" "${out_video}" >> "${OUT_ROOT}/side_by_side/manifest.tsv"

    if [[ "${EXTRACT_FRAMES}" == "1" ]]; then
        frame_dir="${OUT_ROOT}/side_by_side/frames/${task}_seed${seed}"
        mkdir -p "${frame_dir}"
        if ffmpeg -y -loglevel warning -i "${out_video}" "${frame_dir}/frame_%06d.png"; then
            echo "Extracted frame pairs to: ${frame_dir}"
        else
            echo "!!!!! Failed to extract frames for: ${out_video}"
            status=1
        fi
    fi
}

if [[ "${RUN_COLLECT}" == "1" ]]; then
    for task in "${RUN_TASKS[@]}"; do
        run_collect "${task}" "GelSight/demo" "${GELSIGHT_CFG}"
        run_collect "${task}" "Neote" "${NEOTE_CFG}"
    done
else
    echo "Skip collection because RUN_COLLECT=${RUN_COLLECT}"
fi

if [[ "${RUN_COMPARE}" == "1" ]]; then
    echo
    echo "========== Build side-by-side comparison videos =========="
    manifest="${OUT_ROOT}/side_by_side/manifest.tsv"
    printf "task\tseed\tgelsight_status\tneote_status\tgelsight_video\tneote_video\tcompare_video\n" > "${manifest}"
    for task in "${RUN_TASKS[@]}"; do
        for ((seed=START_SEED; seed<=MAX_SEED; seed++)); do
            build_compare_for_seed "${task}" "${seed}"
        done
    done
    echo
    echo "Comparison manifest: ${manifest}"
    echo "Comparison videos:"
    find "${OUT_ROOT}/side_by_side" -maxdepth 1 -type f -name '*.mp4' | sort || true
else
    echo "Skip comparison because RUN_COMPARE=${RUN_COMPARE}"
fi

echo
echo "========== Done =========="
echo "Output root: ${OUT_ROOT}"
echo "Log: ${LOG_PATH}"
exit "${status}"
