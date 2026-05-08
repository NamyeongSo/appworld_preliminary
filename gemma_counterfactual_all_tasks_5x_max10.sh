#!/usr/bin/env bash
set -euo pipefail

# Launch counterfactual replays for every task in a dataset, repeating each task
# multiple times while capping total concurrency across the whole job.
#
# Default behavior:
#   - every task in data/datasets/test_normal.txt
#   - 5 repeated runs per task
#   - at most 10 concurrent AppWorld processes at a time
#
# This script reuses gemma_parallel_counterfactual.sh by feeding it explicit
# TASK_IDS chunks. Each chunk runs in parallel; chunks themselves run sequentially.
#
# Usage:
#   nohup ./gemma_counterfactual_all_tasks_5x_max10.sh [batch_id] \
#     > counter_all.log 2>&1 &
#
# Useful overrides:
#   DATASET=test_normal
#   REPEATS_PER_TASK=5
#   MAX_PARALLEL=10
#   BASE_EXPERIMENT=counterfactual/google/gemma-4-26b-a4b-it/test_normal
export MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://localhost:8002}"
#   TEMPERATURE=0.5
#   MAX_REPLAY_ATTEMPTS=50
#   CASE_B_MAX_ATTEMPTS=50

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BATCH_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-test_normal}"
REPEATS_PER_TASK="${REPEATS_PER_TASK:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"
LAUNCHER_SCRIPT="${LAUNCHER_SCRIPT:-$ROOT_DIR/gemma_parallel_counterfactual.sh}"
WRAPPER_LOG_DIR="logs/gemma_counterfactual_all_tasks/${BATCH_ID}"
DATASET_FILE="data/datasets/${DATASET}.txt"

if [[ ! -x "$LAUNCHER_SCRIPT" ]]; then
  echo "ERROR: launcher script is not executable: $LAUNCHER_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "ERROR: dataset file not found: $DATASET_FILE" >&2
  exit 1
fi

if ! [[ "$REPEATS_PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: REPEATS_PER_TASK must be a positive integer, got '$REPEATS_PER_TASK'." >&2
  exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_PARALLEL must be a positive integer, got '$MAX_PARALLEL'." >&2
  exit 1
fi

mkdir -p "$WRAPPER_LOG_DIR"

declare -a tasks=()
while IFS= read -r task_id || [[ -n "$task_id" ]]; do
  [[ -z "$task_id" ]] && continue
  tasks+=("$task_id")
done < "$DATASET_FILE"

if (( ${#tasks[@]} == 0 )); then
  echo "ERROR: dataset file is empty: $DATASET_FILE" >&2
  exit 1
fi

declare -a expanded_tasks=()
for task_id in "${tasks[@]}"; do
  for ((repeat_idx = 1; repeat_idx <= REPEATS_PER_TASK; repeat_idx++)); do
    expanded_tasks+=("$task_id")
  done
done

printf '%s\n' "${tasks[@]}" > "$WRAPPER_LOG_DIR/tasks_source.txt"
printf '%s\n' "${expanded_tasks[@]}" > "$WRAPPER_LOG_DIR/expanded_task_queue.txt"

echo "[$(date -Is)] DATASET_FILE=${DATASET_FILE}"
echo "[$(date -Is)] UNIQUE_TASKS=${#tasks[@]}"
echo "[$(date -Is)] REPEATS_PER_TASK=${REPEATS_PER_TASK}"
echo "[$(date -Is)] TOTAL_RUNS=${#expanded_tasks[@]}"
echo "[$(date -Is)] MAX_PARALLEL=${MAX_PARALLEL}"
echo "[$(date -Is)] WRAPPER_LOG_DIR=${WRAPPER_LOG_DIR}"

chunk_index=0
for ((offset = 0; offset < ${#expanded_tasks[@]}; offset += MAX_PARALLEL)); do
  chunk_index=$((chunk_index + 1))

  declare -a chunk_tasks=()
  for ((i = offset; i < offset + MAX_PARALLEL && i < ${#expanded_tasks[@]}; i++)); do
    chunk_tasks+=("${expanded_tasks[$i]}")
  done

  chunk_runs=${#chunk_tasks[@]}
  chunk_batch_id="$(printf '%s_chunk_%03d' "$BATCH_ID" "$chunk_index")"
  chunk_task_ids="$(printf '%s ' "${chunk_tasks[@]}")"
  chunk_task_ids="${chunk_task_ids% }"

  {
    echo "[$(date -Is)] CHUNK_START index=${chunk_index} offset=${offset} runs=${chunk_runs} batch=${chunk_batch_id}"
    printf '[%s] CHUNK_TASKS %s\n' "$(date -Is)" "$chunk_task_ids"
  } | tee -a "$WRAPPER_LOG_DIR/chunks.log"

  RUNS="$chunk_runs" \
  TASK_IDS="$chunk_task_ids" \
  "$LAUNCHER_SCRIPT" "$chunk_batch_id" | tee -a "$WRAPPER_LOG_DIR/chunks.log"

  echo "[$(date -Is)] CHUNK_END index=${chunk_index} batch=${chunk_batch_id}" | tee -a "$WRAPPER_LOG_DIR/chunks.log"
done

echo "[$(date -Is)] COMPLETE batch=${BATCH_ID}" | tee -a "$WRAPPER_LOG_DIR/chunks.log"
