#!/usr/bin/env bash
# Run all Glossa benchmark suites and generate reports.
#
# Usage:
#   ./scripts/run_benchmarks.sh [OPTIONS]
#
# Options:
#   --host HOST         Base URL for the API gateway (default: http://localhost:8000)
#   --redis REDIS_URL   Redis connection URL (default: redis://localhost:6379)
#   --concurrency N     Peak concurrency for stress tests (default: 50)
#   --duration SEC      Duration in seconds for sustained-load tests (default: 120)
#   --locust-users N    Number of Locust users for load test (default: 30)
#   --locust-time T     Locust run time e.g. "3m" (default: 3m)
#   --skip-locust       Skip Locust scenario tests (useful in CI without services)
#   --skip-gpu          Skip GPU profiling (no GPU available)
#   --skip-flamegraph   Skip py-spy flamegraph (requires sudo / ptrace)
#   --report-dir DIR    Output directory for reports (default: benchmarks/reports)
#   --format FMT        Report format: html|markdown|both (default: both)
#   -h, --help          Show this help

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="${BENCHMARK_HOST:-http://localhost:8000}"
REDIS_URL="${BENCHMARK_REDIS_URL:-redis://localhost:6379}"
CONCURRENCY="${BENCHMARK_CONCURRENCY:-50}"
DURATION="${BENCHMARK_DURATION:-120}"
LOCUST_USERS="${BENCHMARK_LOCUST_USERS:-30}"
LOCUST_TIME="${BENCHMARK_LOCUST_TIME:-3m}"
SKIP_LOCUST=0
SKIP_GPU=0
SKIP_FLAMEGRAPH=0
REPORT_DIR="benchmarks/reports"
FORMAT="both"

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; echo -e "${BOLD}${CYAN}  $*${NC}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)           HOST="$2";          shift 2 ;;
    --redis)          REDIS_URL="$2";     shift 2 ;;
    --concurrency)    CONCURRENCY="$2";   shift 2 ;;
    --duration)       DURATION="$2";      shift 2 ;;
    --locust-users)   LOCUST_USERS="$2";  shift 2 ;;
    --locust-time)    LOCUST_TIME="$2";   shift 2 ;;
    --skip-locust)    SKIP_LOCUST=1;      shift ;;
    --skip-gpu)       SKIP_GPU=1;         shift ;;
    --skip-flamegraph)SKIP_FLAMEGRAPH=1;  shift ;;
    --report-dir)     REPORT_DIR="$2";    shift 2 ;;
    --format)         FORMAT="$2";        shift 2 ;;
    -h|--help)
      sed -n '3,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ── Validate environment ───────────────────────────────────────────────────────
RUN_TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${REPORT_DIR}/${RUN_TS}"
mkdir -p "$REPORT_DIR"

info "Benchmark run: ${RUN_TS}"
info "Host:         ${HOST}"
info "Redis:        ${REDIS_URL}"
info "Reports dir:  ${REPORT_DIR}"

# Check Python environment
if ! python -c "import benchmarks" &>/dev/null; then
  warn "benchmarks package not importable; installing in editable mode..."
  pip install -q -e "benchmarks/[all]"
fi

FAIL_COUNT=0
SUITE_RESULTS=()

run_suite() {
  local name="$1"; shift
  local start; start=$(date +%s%N)
  info "Running: ${name}"
  if "$@"; then
    local elapsed=$(( ($(date +%s%N) - start) / 1000000 ))
    ok "${name} completed in ${elapsed}ms"
    SUITE_RESULTS+=("${name}:ok:${elapsed}")
  else
    local rc=$?
    fail "${name} exited with code ${rc}"
    SUITE_RESULTS+=("${name}:fail:0")
    (( FAIL_COUNT++ )) || true
  fi
}

# ── 1. Latency benchmarks ─────────────────────────────────────────────────────
header "1/6  Latency Benchmarks"
run_suite "latency" python -m benchmarks.runners.latency_runner \
  --host "$HOST" \
  --report-dir "$REPORT_DIR"

# ── 2. Stress / concurrency tests ────────────────────────────────────────────
header "2/6  Stress & Concurrency Tests"
run_suite "stress" python -m benchmarks.runners.stress_runner \
  --host "$HOST" \
  --max-concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --report-dir "$REPORT_DIR"

# ── 3. Redis throughput benchmarks ───────────────────────────────────────────
header "3/6  Redis Load Tests"
run_suite "redis" python -m benchmarks.runners.redis_runner \
  --redis-url "$REDIS_URL" \
  --report-dir "$REPORT_DIR"

# ── 4. Queue / webhook bottleneck analysis ───────────────────────────────────
header "4/6  Queue Bottleneck Analysis"
run_suite "queue" python -m benchmarks.runners.queue_runner \
  --host "$HOST" \
  --report-dir "$REPORT_DIR"

# ── 5. Locust load scenario ───────────────────────────────────────────────────
header "5/6  Locust Load Scenarios"
if [[ $SKIP_LOCUST -eq 1 ]]; then
  warn "Skipping Locust (--skip-locust)"
else
  LOCUST_HTML="${REPORT_DIR}/locust_report.html"
  LOCUST_CSV="${REPORT_DIR}/locust"
  run_suite "locust" locust \
    -f benchmarks/locustfile.py \
    --host "$HOST" \
    --headless \
    -u "$LOCUST_USERS" \
    -r 5 \
    --run-time "$LOCUST_TIME" \
    --html "$LOCUST_HTML" \
    --csv "$LOCUST_CSV" \
    --exit-code-on-error 0
fi

# ── 6. CPU / GPU profiling ────────────────────────────────────────────────────
header "6/6  CPU & GPU Profiling"

if [[ $SKIP_FLAMEGRAPH -eq 1 ]]; then
  warn "Skipping py-spy flamegraph (--skip-flamegraph)"
else
  # Try to find running cv-service and attach py-spy
  CV_PID=$(docker inspect --format '{{.State.Pid}}' glossa-cv-service 2>/dev/null || echo "")
  if [[ -n "$CV_PID" && "$CV_PID" != "0" ]]; then
    run_suite "flamegraph-cv" py-spy record \
      --pid "$CV_PID" \
      --duration 30 \
      --format flamegraph \
      --output "${REPORT_DIR}/flamegraph_cv.svg"
  else
    warn "cv-service container not running; skipping flamegraph"
  fi
fi

if [[ $SKIP_GPU -eq 1 ]]; then
  warn "Skipping GPU profiling (--skip-gpu)"
else
  run_suite "gpu-profile" python -m benchmarks.profiling.gpu_profiler \
    --duration 30 \
    --report-dir "$REPORT_DIR"
fi

# ── Generate consolidated report ──────────────────────────────────────────────
header "Generating Report"

REPORT_ARGS=(
  --report-dir "$REPORT_DIR"
  --run-ts "$RUN_TS"
)
[[ "$FORMAT" == "html" || "$FORMAT" == "both" ]]     && REPORT_ARGS+=(--html)
[[ "$FORMAT" == "markdown" || "$FORMAT" == "both" ]] && REPORT_ARGS+=(--markdown)

run_suite "report" python -m benchmarks.reporters.report "${REPORT_ARGS[@]}"

# ── Summary ───────────────────────────────────────────────────────────────────
header "Benchmark Summary"
echo ""
printf "  %-40s  %-8s  %s\n" "Suite" "Status" "Duration"
printf "  %-40s  %-8s  %s\n" "─────────────────────────────────────" "──────" "────────"
for result in "${SUITE_RESULTS[@]}"; do
  IFS=':' read -r name status elapsed <<< "$result"
  if [[ "$status" == "ok" ]]; then
    printf "  %-40s  ${GREEN}%-8s${NC}  %sms\n" "$name" "PASS" "$elapsed"
  else
    printf "  %-40s  ${RED}%-8s${NC}  —\n" "$name" "FAIL"
  fi
done
echo ""

if [[ -f "${REPORT_DIR}/report.html" ]]; then
  ok "HTML report: ${REPORT_DIR}/report.html"
fi
if [[ -f "${REPORT_DIR}/report.md" ]]; then
  ok "Markdown report: ${REPORT_DIR}/report.md"
fi
if [[ -f "${REPORT_DIR}/locust_report.html" ]]; then
  ok "Locust report: ${REPORT_DIR}/locust_report.html"
fi

echo ""
if [[ $FAIL_COUNT -eq 0 ]]; then
  ok "All suites passed."
  exit 0
else
  fail "${FAIL_COUNT} suite(s) failed. See output above for details."
  exit 1
fi
