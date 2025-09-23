#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name> [method] [input_dir]"
  echo "Example: $0 meta-llama/Meta-Llama-3-8B structured results"
  exit 1
fi

MODEL_NAME="$1"
METHOD="${2:-structured}"
INPUT_DIR="${3:-results}"

TOTAL=2244
CHUNKS=20
CHUNK_SIZE=$(( TOTAL / CHUNKS ))  # 374
REMAINDER=$(( TOTAL % CHUNKS ))   # 0 here, but we’ll still guard the last chunk

echo "Submitting ${CHUNKS} jobs for TOTAL=${TOTAL} (chunk_size=${CHUNK_SIZE})"
for i in $(seq 0 $((CHUNKS-1))); do
  START=$(( i * CHUNK_SIZE ))
  END=$(( START + CHUNK_SIZE ))
  # Make last chunk end at TOTAL to absorb any remainder
  if [[ $i -eq $((CHUNKS-1)) ]]; then
    END=${TOTAL}
  fi

  echo "Chunk $i: [${START}, ${END})"
  sbatch run_extract_id.sh "${MODEL_NAME}" "${METHOD}" "${START}" "${END}" "${INPUT_DIR}"
done
