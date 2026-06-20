#!/usr/bin/env bash
# Run all Glossa architecture-justification experiments sequentially.
#
# Usage:
#   bash experiments/run_all.sh            # dry-run by default
#   bash experiments/run_all.sh --no-dry-run --slovo-root data/raw/slovo
#   bash experiments/run_all.sh --skip 03,07 --n-samples 100
#   bash experiments/run_all.sh --mlflow-uri https://dagshub.com/noviyblock/glossa.mlflow
#
# Each experiment writes results to experiments/results/<exp>/ and logs to MLflow.
# Exit code: 0 if all succeed, 1 if any fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN="--dry-run"
SKIP_LIST=""
MLFLOW_URI="${MLFLOW_TRACKING_URI:-https://dagshub.com/noviyblock/glossa.mlflow}"
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
    --no-dry-run)     DRY_RUN=""; shift ;;
    --skip)           SKIP_LIST="$2"; shift 2 ;;
    --mlflow-uri)     MLFLOW_URI="$2"; shift 2 ;;
    --slovo-root)     SLOVO_ROOT="$2"; shift 2 ;;
    --host)           HOST="$2"; shift 2 ;;
    --redis-url)      REDIS_URL="$2"; shift 2 ;;
    --n-samples)      N_SAMPLES="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

should_skip() { [[ ",$SKIP_LIST," == *",$1,"* ]]; }

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

echo "═══════════════════════════════════════════════════════════════════════"
echo "  GLOSSA Experiment Suite — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  MLflow : $MLFLOW_URI"
echo "  dry-run: ${DRY_RUN:+yes (--dry-run)}${DRY_RUN:-no (REAL DATA)}"
echo "  samples: $N_SAMPLES"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

run_exp "00_compare_models" "Architecture comparison (ST-GCN vs S3D vs ResNet3D)" \
  --n-samples "$N_SAMPLES" ${SLOVO_ROOT:+--slovo-root "$SLOVO_ROOT"} $DRY_RUN

run_exp "01_gesture_backbone" "Mobile vs full ONNX backbone" \
  --n-samples "$N_SAMPLES" ${SLOVO_ROOT:+--slovo-root "$SLOVO_ROOT"} $DRY_RUN

run_exp "02_sliding_window" "Sliding window grid search (size × stride)" \
  --n-samples "$N_SAMPLES" ${SLOVO_ROOT:+--slovo-root "$SLOVO_ROOT"} $DRY_RUN

run_exp "03_inference_acceleration" "ONNX vs OpenVINO INT8 (×61 speedup)" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "04_asr_comparison" "Whisper tiny vs base (WER on Russian)" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "05_nlp_llm_size" "Qwen2-1.5B top-1 vs top-3 (BLEU 83.91)" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "06_rag_ablation" "NLP cache: with vs without (latency reduction)" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "07_tts_utmos" "Silero v4 voice quality: xenia vs eugene vs aidar" \
  --n-samples "$N_SAMPLES" $DRY_RUN

run_exp "08_e2e_latency" "Full pipeline E2E latency on target devices" \
  --n-samples "$N_SAMPLES" --host "$HOST" --redis-url "$REDIS_URL" $DRY_RUN

run_exp "09_cross_validation" "Stratified K-fold CV + learning curve" \
  --k-folds 5 $DRY_RUN

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Results"
echo "  ───────────────────────────────────────────────────────────────────"
ALL_IDS=(
  00_compare_models 01_gesture_backbone 02_sliding_window
  03_inference_acceleration 04_asr_comparison 05_nlp_llm_size
  06_rag_ablation 07_tts_utmos 08_e2e_latency 09_cross_validation
)
for id in "${ALL_IDS[@]}"; do
  ms="${TIMES[$id]:-—}"
  short="${id%%_*}"
  if should_skip "$short"; then
    printf "  %-38s  SKIPPED\n" "$id"
  elif [[ "$ms" != "—" ]]; then
    printf "  %-38s  %dms\n" "$id" "$ms"
  fi
done
echo "  ───────────────────────────────────────────────────────────────────"
printf "  PASS: %d   FAIL: %d\n" "$PASS" "$FAIL"
echo "═══════════════════════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]
