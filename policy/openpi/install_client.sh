#!/usr/bin/env bash
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-UniVTAC}"
OPENPI_CLIENT_PATH="${OPENPI_CLIENT_PATH:-/home/fudan/Workspace/gjzhong/openpi-client}"

if [[ -f /home/fudan/Softwares/miniconda3/etc/profile.d/conda.sh ]]; then
    source /home/fudan/Softwares/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "Could not find conda. Activate the target environment manually or install openpi-client with pip."
    exit 2
fi

conda activate "${CONDA_ENV}"
python -m pip install -e "${OPENPI_CLIENT_PATH}"
