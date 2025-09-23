#!/bin/bash
#SBATCH --job-name=extract-id        # Job name
#SBATCH --partition=small-g         # Partition
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH --time=4:00:00              # Walltime (HH:MM:SS)
#SBATCH --account=project_465001340 # Project account
#SBATCH --output=logs/%x-%j.out     # Stdout (save to logs folder)

set -euo pipefail

# ---- Positional args ----
MODEL_NAME="${1:?Need model_name}"
METHOD="${2:?Need method}"
BATCH_START="${3:?Need batch_start}"
BATCH_END="${4:?Need batch_end}"
INPUT_DIR="${5:-results}"

echo "Running with:"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  METHOD=${METHOD}"
echo "  BATCH_START=${BATCH_START}"
echo "  BATCH_END=${BATCH_END}"
echo "  INPUT_DIR=${INPUT_DIR}"

python src/extract_id.py \
  --input_dir "${INPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --method "${METHOD}" \
  --batch_start "${BATCH_START}" \
  --batch_end "${BATCH_END}"