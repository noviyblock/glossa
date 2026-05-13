#!/usr/bin/env bash
# Comprehensive health check — exits 0 if all critical services are healthy
set -euo pipefail

COMPOSE="docker compose"
TIMEOUT="${HEALTH_TIMEOUT:-10}"
FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; (( FAIL++ )) || true; }
warn() { echo -e "  ${YELLOW}~${NC} $*"; }

check_http() {
    local name=$1 url=$2
    if curl -sf --max-time "${TIMEOUT}" "${url}" > /dev/null 2>&1; then
        ok "$name ($url)"
    else
        fail "$name — unreachable ($url)"
    fi
}

check_docker_health() {
    local name=$1
    local status
    status=$($COMPOSE ps --format "{{.Health}}" "$name" 2>/dev/null | head -1)
    case "$status" in
        healthy)   ok  "$name (Docker health: healthy)" ;;
        unhealthy) fail "$name (Docker health: unhealthy)" ;;
        *)         warn "$name (Docker health: ${status:-unknown})" ;;
    esac
}

echo "═══════════════════════════════════════"
echo " Glossa Platform Health Check"
echo " $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "═══════════════════════════════════════"

# ── Infrastructure ────────────────────────────────────────────────────────────
echo ""
echo "Infrastructure:"
check_docker_health redis
check_docker_health qdrant
check_http "Redis ping"  "http://localhost:6379" 2>/dev/null || \
    $COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1 && ok "Redis ping" || fail "Redis ping"
check_http "Qdrant"      "http://localhost:6333/healthz"

# ── Application services ──────────────────────────────────────────────────────
echo ""
echo "Application Services:"
check_http "API Gateway  /live"  "http://localhost:8000/health/live"
check_http "API Gateway  /ready" "http://localhost:8000/health/ready"
check_http "CV Service   /live"  "http://localhost:8001/health/live"
check_http "ASR Service  /live"  "http://localhost:8002/health/live"
check_http "NLP Service  /live"  "http://localhost:8003/health/live"
check_http "TTS Service  /live"  "http://localhost:8004/health/live"
check_http "MAX Adapter  /live"  "http://localhost:8005/health/live"

# ── Observability ─────────────────────────────────────────────────────────────
echo ""
echo "Observability:"
check_http "Prometheus"  "http://localhost:9090/-/healthy"
check_http "Grafana"     "http://localhost:3001/api/health"
check_http "Loki"        "http://localhost:3100/ready"
check_http "Jaeger"      "http://localhost:16686"
check_http "MLflow"      "http://localhost:5000/health"
check_http "Nginx"       "http://localhost/nginx-health"

# ── Container resource usage (informational) ──────────────────────────────────
echo ""
echo "Container Resources (top 5 by memory):"
docker stats --no-stream --format "  {{.Name}}: CPU={{.CPUPerc}} MEM={{.MemUsage}}" \
    2>/dev/null | sort -t= -k3 -hr | head -5 || true

# ── Result ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════"
if (( FAIL > 0 )); then
    echo -e "  ${RED}UNHEALTHY${NC} — $FAIL check(s) failed"
    exit 1
else
    echo -e "  ${GREEN}HEALTHY${NC} — all checks passed"
fi
echo "═══════════════════════════════════════"
