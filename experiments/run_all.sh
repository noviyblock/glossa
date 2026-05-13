#!/usr/bin/env bash
# Run all 8 Glossa architecture-justification experiments sequentially.
#
# Usage:
#   bash experiments/run_all.sh [--dry-run] [--skip 03,07] [--mlflow-uri http://localhost:5000]
#
# Each experiment writes results to experiments/results/<exp>/ and logs to MLflow.
# Exit code: 0 if all succeed, 1 if any fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN=""
SKIP_LIST=""
MLFLOW_URI="http://localhost:5000"
SLOVO_ROOT="${SLOVO_ROOT:-}"
HOST="${GLOSSA_HOST:-http://localhost:8000}"
REDIS_URL="${REDIS_URL:-redis://localhost:6379}"
N_SAMPLES=50

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
ok()   { printf "${GREEN}✓ PASS${RESET}  %s\n" "$1"; }
fail() { printf "${RED}✗ FAIL${RESET}  %s\n" "$1"; }
info() { printf "${YELLOW}»${RESET} %s\n" "$1"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN="--dry-run"; shift ;;
    --skip)           SKIP_LIST="$2"; shift 2 ;;
    --mlflow-uri)     MLFLOW_URI="$2"; shift 2 ;;
    --slovo-root)     SLOVO_ROOT="$2"; shift 2 ;;
    --host)           HOST="$2"; shift 2 ;;
    --redis-url)      REDIS_URL="$2"; shift 2 ;;
    --n-samples)      N_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

should_skip() {
  [[ ",$SKIP_LIST," == *",$1,"* ]]
}

export MLFLOW_TRACKING_URI="$MLFLOW_URI"
cd "$REPO_ROOT"

PASS=0; FAIL=0
declare -A TIMES

run_exp() {
  local id="$1"; local label="$2"; shift 2
  if should_skip "$id"; then
    info "Skipping exp $id ($label)"
    return 0
  fi
  info "Exp $id — $label"
  local t0; t0=$(date +%s%3N)
  if python -m "experiments.${id}.run" "$@"; then
    local t1; t1=$(date +%s%3N)
    TIMES[$id]=$(( t1 - t0 ))
    ok "$label (${TIMES[$id]}ms)"
    (( PASS++ )) || true
  else
    local t1; t1=$(date +%s%3N)
    TIMES[$id]=$(( t1 - t0 ))
    fail "$label (${TIMES[$id]}ms)"
    (( FAIL++ )) || true
  fi
}

echo "═══════════════════════════════════════════════════════════════════"
echo "  GLOSSA Experiment Suite — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  MLflow: $MLFLOW_URI  |  dry-run: ${DRY_RUN:-no}"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

run_exp "01_gesture_backbone" "Gesture backbone comparison" \
  --n-samples "$N_SAMPLES" ${SLOVO_ROOT:+--slovo-root "$SLOVO_ROOT"} $DRY_RUN

run_exp "02_sliding_window" "Sliding window grid search" \
  --n-samples "$N_SAMPLES" ${SLOVO_ROOT:+--slovo-root "$SLOVO_ROOT"} $DRY_RUN

run_exp "03_inference_acceleration" "ONNX/OpenVINO/INT8 acceleration" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "04_asr_comparison" "Whisper ASR model comparison" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "05_nlp_llm_size" "Qwen2 LLM size comparison" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "06_rag_ablation" "RAG ablation study" \
  --n-samples "$N_SAMPLES" --nlp-url "${HOST/8000/8003}" $DRY_RUN

run_exp "07_tts_utmos" "TTS UTMOS quality evaluation" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "08_e2e_latency" "End-to-end latency breakdown" \
  --n-samples "$N_SAMPLES" --host "$HOST" --redis-url "$REDIS_URL" $DRY_RUN

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Results"
echo "  ─────────────────────────────────────────────────────────────────"
for id in 01_gesture_backbone 02_sliding_window 03_inference_acceleration \
           04_asr_comparison 05_nlp_llm_size 06_rag_ablation \
           07_tts_utmos 08_e2e_latency; do
  ms="${TIMES[$id]:-—}"
  if should_skip "${id%%_*}"; then
    printf "  %-32s  SKIPPED\n" "$id"
  elif [[ "$ms" != "—" ]]; then
    printf "  %-32s  %dms\n" "$id" "$ms"
  fi
done
echo "  ─────────────────────────────────────────────────────────────────"
printf "  PASS: %d   FAIL: %d\n" "$PASS" "$FAIL"
echo "═══════════════════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]
