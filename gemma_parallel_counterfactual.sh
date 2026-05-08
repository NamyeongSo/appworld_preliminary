#!/usr/bin/env bash
set -euo pipefail

# Launch N independent AppWorld Gemma counterfactual replay experiments in parallel.
#
# Each parallel run executes one task by default and produces one success/failure
# trajectory pair plus every intermediate replay trajectory under:
#   experiments/outputs/<experiment>/tasks/<task_id>/counterfactual/
#
# Usage:
#   nohup ./gemma_parallel_counterfactual.sh [batch_id] > gemma_counterfactual_launcher.log 2>&1 &
#
# Useful overrides:
#   RUNS=5
#   DATASET=test_normal
#   TASK_IDS="3d9a636_1 fd1f8fa_2"   # optional explicit per-run task IDs
#   TASK_OFFSET=0                       # used when TASK_IDS is not set
#   MODEL_SERVER_URL=http://localhost:8002
#   TEMPERATURE=0.5
#   MAX_REPLAY_ATTEMPTS=50
#   CASE_B_MAX_ATTEMPTS=50

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${RUNS:-5}"
BATCH_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-counterfactual/google/gemma-4-26b-a4b-it/test_normal}"
DATASET="${DATASET:-test_normal}"
TASK_OFFSET="${TASK_OFFSET:-0}"
MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://localhost:8002}"
TEMPERATURE="${TEMPERATURE:-0.5}"
MAX_REPLAY_ATTEMPTS="${MAX_REPLAY_ATTEMPTS:-50}"
CASE_B_MAX_ATTEMPTS="${CASE_B_MAX_ATTEMPTS:-50}"
APPWORLD_BIN="${APPWORLD_BIN:-$ROOT_DIR/.appworld/bin/appworld}"
CHECK_SERVER="${CHECK_SERVER:-1}"
VARY_SEED="${VARY_SEED:-1}"

export MODEL_SERVER_URL

MODEL_SERVER_PORT="${MODEL_SERVER_URL##*:}"
if ! [[ "$MODEL_SERVER_PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to parse port from MODEL_SERVER_URL='${MODEL_SERVER_URL}'." >&2
  exit 1
fi

if [[ ! -x "$APPWORLD_BIN" ]]; then
  APPWORLD_BIN="$(command -v appworld || true)"
fi
if [[ -z "$APPWORLD_BIN" ]]; then
  echo "ERROR: appworld executable not found. Set APPWORLD_BIN=/path/to/appworld." >&2
  exit 1
fi

BASE_CONFIG="experiments/configs/${BASE_EXPERIMENT}.jsonnet"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

DATASET_FILE="data/datasets/${DATASET}.txt"
if [[ ! -f "$DATASET_FILE" ]]; then
  echo "ERROR: dataset file not found: $DATASET_FILE" >&2
  exit 1
fi

if [[ "$CHECK_SERVER" == "1" ]] && command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${MODEL_SERVER_URL}/v1/models" >/dev/null; then
    cat >&2 <<EOM
ERROR: model server is not reachable at ${MODEL_SERVER_URL}/v1/models.

Start the Gemma/vLLM server first, or rerun with CHECK_SERVER=0 if you
intentionally want AppWorld to wait/fail on its own.
EOM
    exit 2
  fi
fi

# Build per-run task list. Explicit TASK_IDS can be comma or whitespace separated.
declare -a selected_tasks=()
if [[ -n "${TASK_IDS:-}" ]]; then
  normalized_task_ids="${TASK_IDS//,/ }"
  # shellcheck disable=SC2206
  selected_tasks=($normalized_task_ids)
else
  mapfile -t all_tasks < "$DATASET_FILE"
  for n in $(seq 1 "$RUNS"); do
    index=$((TASK_OFFSET + n - 1))
    if (( index >= ${#all_tasks[@]} )); then
      echo "ERROR: not enough tasks in ${DATASET_FILE} for RUNS=${RUNS} TASK_OFFSET=${TASK_OFFSET}." >&2
      exit 1
    fi
    selected_tasks+=("${all_tasks[$index]}")
  done
fi

if (( ${#selected_tasks[@]} < RUNS )); then
  echo "ERROR: got ${#selected_tasks[@]} TASK_IDS but RUNS=${RUNS}." >&2
  exit 1
fi

BASE_DIR="$(dirname "$BASE_EXPERIMENT")"
PARALLEL_EXPERIMENT_DIR="${BASE_DIR}/counterfactual_${BATCH_ID}"
CONFIG_DIR="experiments/configs/${PARALLEL_EXPERIMENT_DIR}"
LOG_DIR="logs/gemma_counterfactual/${BATCH_ID}"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"
: > "$LOG_DIR/pids.txt"
: > "$LOG_DIR/tasks.txt"

declare -a pids=()

for n in $(seq 1 "$RUNS"); do
  run_name="$(printf 'run_%02d' "$n")"
  task_id="${selected_tasks[$((n - 1))]}"
  experiment_name="${PARALLEL_EXPERIMENT_DIR}/${run_name}"
  config_path="${CONFIG_DIR}/${run_name}.jsonnet"
  log_path="${LOG_DIR}/${run_name}.log"

  seed=100
  if [[ "$VARY_SEED" == "1" ]]; then
    seed=$((100 + n))
  fi

  sed -E \
    -e "s/(\"seed\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${seed}/g" \
    -e "s/(\"random_seed\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${seed}/g" \
    -e "s/(\"port\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${MODEL_SERVER_PORT}/g" \
    -e "s/(\"temperature\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1${TEMPERATURE}/g" \
    -e "s/(\"max_replay_attempts\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${MAX_REPLAY_ATTEMPTS}/g" \
    -e "s/(\"case_b_max_attempts\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${CASE_B_MAX_ATTEMPTS}/g" \
    "$BASE_CONFIG" > "$config_path"

  echo "$run_name $task_id $experiment_name" | tee -a "$LOG_DIR/tasks.txt"

  (
    echo "[$(date -Is)] START experiment=${experiment_name} task_id=${task_id}"
    echo "[$(date -Is)] LOG=${log_path}"
    echo "[$(date -Is)] MODEL_SERVER_URL=${MODEL_SERVER_URL}"
    echo "[$(date -Is)] TEMPERATURE=${TEMPERATURE}"
    echo "[$(date -Is)] MAX_REPLAY_ATTEMPTS=${MAX_REPLAY_ATTEMPTS} CASE_B_MAX_ATTEMPTS=${CASE_B_MAX_ATTEMPTS}"
    "$APPWORLD_BIN" run "$experiment_name" --task-id "$task_id" --no-with-evaluation
    status=$?
    echo "[$(date -Is)] END experiment=${experiment_name} task_id=${task_id} status=${status}"
    exit "$status"
  ) > "$log_path" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "$pid $experiment_name $task_id $log_path" | tee -a "$LOG_DIR/pids.txt"
done

echo "Launched ${RUNS} counterfactual replay runs for batch '${BATCH_ID}'."
echo "PID list: $LOG_DIR/pids.txt"
echo "Task list: $LOG_DIR/tasks.txt"
echo "Logs:     $LOG_DIR/run_XX.log"
echo "Outputs:  experiments/outputs/${PARALLEL_EXPERIMENT_DIR}/run_XX/tasks/<task_id>/counterfactual/"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" == "0" ]]; then
  echo "All ${RUNS} counterfactual runs completed successfully."
else
  echo "At least one run failed. Check logs under $LOG_DIR." >&2
fi

exit "$failed"
