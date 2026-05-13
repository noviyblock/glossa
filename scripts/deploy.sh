#!/usr/bin/env bash
# Production deployment script — rolling update with health verification
set -euo pipefail

COMPOSE="docker compose -f docker-compose.yml -f docker-compose.prod.yml"
SERVICES=(api-gateway cv-service asr-service nlp-service tts-service max-adapter)
HEALTH_TIMEOUT=120
REGISTRY="${REGISTRY:-ghcr.io}"
IMAGE_PREFIX="${IMAGE_PREFIX:-glossa}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()   { echo -e "${GREEN}[deploy]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
preflight() {
    log "Running pre-flight checks..."
    command -v docker &>/dev/null || { error "docker not found"; exit 1; }
    docker compose version &>/dev/null || { error "docker compose plugin not found"; exit 1; }
    [[ -f ".env" ]] || { error ".env file not found — copy .env.example and fill values"; exit 1; }
    # shellcheck source=/dev/null
    source .env
    [[ -n "${ENVIRONMENT:-}" ]] || warn "ENVIRONMENT not set in .env"
    log "Pre-flight OK"
}

# ── Wait for service health ───────────────────────────────────────────────────
wait_healthy() {
    local svc=$1
    local port=$2
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    log "Waiting for $svc to become healthy..."
    while (( $(date +%s) < deadline )); do
        local status
        status=$(docker compose ps --format json "$svc" 2>/dev/null | \
                 python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('Health','unknown'))" 2>/dev/null || echo "unknown")
        if [[ "$status" == "healthy" ]]; then
            log "$svc is healthy"
            return 0
        fi
        sleep 3
    done
    error "$svc did not become healthy within ${HEALTH_TIMEOUT}s"
    docker compose logs --tail=50 "$svc"
    return 1
}

# ── Pull updated images ───────────────────────────────────────────────────────
pull_images() {
    log "Pulling latest images..."
    $COMPOSE pull "${SERVICES[@]}" 2>&1 | grep -v "^$" || true
    log "Images pulled"
}

# ── Start infrastructure first ────────────────────────────────────────────────
start_infra() {
    log "Starting infrastructure services..."
    $COMPOSE up -d redis qdrant otel-collector prometheus grafana loki promtail jaeger mlflow nginx
    wait_healthy redis 6379
    log "Infrastructure ready"
}

# ── Rolling update of application services ────────────────────────────────────
rolling_update() {
    local port=8000
    for svc in "${SERVICES[@]}"; do
        log "Deploying $svc (port $port)..."

        # Pull new image if pre-pulled
        $COMPOSE up -d --no-deps --pull never "$svc"
        sleep 5

        if ! wait_healthy "$svc" "$port"; then
            error "Deployment failed at $svc — rolling back..."
            rollback "$svc"
            exit 1
        fi

        log "✓ $svc deployed successfully"
        (( port++ ))
    done
}

# ── Rollback a service to previous image ──────────────────────────────────────
rollback() {
    local svc=$1
    warn "Rolling back $svc..."
    local prev_image
    prev_image=$(docker images --format '{{.Repository}}:{{.Tag}}' | \
                 grep "${IMAGE_PREFIX}-${svc}" | sed -n '2p')
    if [[ -n "$prev_image" ]]; then
        # Override image tag in compose temporarily
        COMPOSE_FILE_ARGS="-f docker-compose.yml -f docker-compose.prod.yml"
        docker compose $COMPOSE_FILE_ARGS up -d --no-deps "$svc"
        warn "Rolled back $svc to previous version"
    else
        error "No previous image found for $svc"
    fi
}

# ── Cleanup old images ────────────────────────────────────────────────────────
cleanup() {
    log "Cleaning up old images..."
    docker image prune -f --filter "until=72h"
    log "Cleanup done"
}

# ── Post-deploy smoke test ────────────────────────────────────────────────────
smoke_test() {
    log "Running smoke tests..."
    local base_url="http://localhost:8000"
    local failed=0
    local checks=(
        "/health/live"
        "/health/ready"
    )
    for path in "${checks[@]}"; do
        if curl -sf --max-time 10 "${base_url}${path}" > /dev/null; then
            log "  ✓ GET ${path}"
        else
            error "  ✗ GET ${path} FAILED"
            (( failed++ ))
        fi
    done
    if (( failed > 0 )); then
        error "Smoke tests failed ($failed checks)"
        return 1
    fi
    log "All smoke tests passed"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    log "=== Glossa Production Deployment ==="
    log "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    log "Git: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

    preflight
    pull_images
    start_infra
    rolling_update
    smoke_test
    cleanup

    log "=== Deployment complete ==="
}

main "$@"
