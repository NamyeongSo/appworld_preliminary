#!/usr/bin/env bash
set -euo pipefail

# Launch N independent AppWorld Gemma experiments in parallel.
#
# Usage:
#   nohup ./gemma_parallel_5.sh [batch_id] > gemma_parallel_launcher.log 2>&1 &
#
# Defaults:
#   RUNS=5
#   MODEL_SERVER_URL=http://localhost:8002
#   TEMPERATURE=0.5
#   BASE_EXPERIMENT=simplified_react_code_agent/google/gemma-4-26b-a4b-it/test_normal
#
# Each run gets a distinct experiment name, so outputs do not collide:
#   experiments/outputs/.../parallel_<batch_id>/run_01/
#   experiments/outputs/.../parallel_<batch_id>/run_02/
#   ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${RUNS:-5}"
BATCH_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-simplified_react_code_agent/google/gemma-4-26b-a4b-it/test_normal}"
MODEL_SERVER_URL="${MODEL_SERVER_URL:-http://localhost:8002}"
TEMPERATURE="${TEMPERATURE:-0.5}"
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

if [[ "$CHECK_SERVER" == "1" ]] && command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${MODEL_SERVER_URL}/v1/models" >/dev/null; then
    cat >&2 <<EOF
ERROR: model server is not reachable at ${MODEL_SERVER_URL}/v1/models.

Your Gemma config has model_server.started=true, so AppWorld expects the vLLM
server to already be running on this URL. Start the server first, or rerun with
CHECK_SERVER=0 if you intentionally want AppWorld to wait/fail on its own.
EOF
    exit 2
  fi
fi

BASE_DIR="$(dirname "$BASE_EXPERIMENT")"
PARALLEL_EXPERIMENT_DIR="${BASE_DIR}/parallel_${BATCH_ID}"
CONFIG_DIR="experiments/configs/${PARALLEL_EXPERIMENT_DIR}"
LOG_DIR="logs/gemma_parallel/${BATCH_ID}"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"
: > "$LOG_DIR/pids.txt"

declare -a pids=()

for n in $(seq 1 "$RUNS"); do
  run_name="$(printf 'run_%02d' "$n")"
  experiment_name="${PARALLEL_EXPERIMENT_DIR}/${run_name}"
  config_path="${CONFIG_DIR}/${run_name}.jsonnet"
  log_path="${LOG_DIR}/${run_name}.log"

  if [[ "$VARY_SEED" == "1" ]]; then
    seed=$((100 + n))
    sed -E \
      -e "s/(\"seed\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${seed}/" \
      -e "s/(\"random_seed\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${seed}/" \
      -e "s/(\"port\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${MODEL_SERVER_PORT}/" \
      -e "s/(\"temperature\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1${TEMPERATURE}/" \
      "$BASE_CONFIG" > "$config_path"
  else
    sed -E \
      -e "s/(\"port\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1${MODEL_SERVER_PORT}/" \
      -e "s/(\"temperature\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1${TEMPERATURE}/" \
      "$BASE_CONFIG" > "$config_path"
  fi

  (
    echo "[$(date -Is)] START experiment=${experiment_name}"
    echo "[$(date -Is)] LOG=${log_path}"
    echo "[$(date -Is)] MODEL_SERVER_URL=${MODEL_SERVER_URL}"
    echo "[$(date -Is)] TEMPERATURE=${TEMPERATURE}"
    "$APPWORLD_BIN" run "$experiment_name"
    status=$?
    echo "[$(date -Is)] END experiment=${experiment_name} status=${status}"
    exit "$status"
  ) > "$log_path" 2>&1 &

  pid=$!
  pids+=("$pid")
  echo "$pid $experiment_name $log_path" | tee -a "$LOG_DIR/pids.txt"
done

echo "Launched ${RUNS} parallel runs for batch '${BATCH_ID}'."
echo "PID list: $LOG_DIR/pids.txt"
echo "Logs:     $LOG_DIR/run_XX.log"
echo "Outputs:  experiments/outputs/${PARALLEL_EXPERIMENT_DIR}/run_XX/"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" == "0" ]]; then
  echo "All ${RUNS} runs completed successfully."
else
  echo "At least one run failed. Check logs under $LOG_DIR." >&2
fi

exit "$failed"
