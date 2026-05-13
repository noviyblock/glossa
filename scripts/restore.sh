#!/usr/bin/env bash
# Restore from a backup created by backup.sh
set -euo pipefail

BACKUP_PATH="${1:-}"
COMPOSE="docker compose"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()   { echo -e "${GREEN}[restore]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

[[ -n "$BACKUP_PATH" ]] || error "Usage: $0 <backup-directory>"
[[ -d "$BACKUP_PATH" ]] || error "Backup directory not found: $BACKUP_PATH"
[[ -f "${BACKUP_PATH}/manifest.json" ]] || error "No manifest.json found in $BACKUP_PATH"

log "Restoring from: $BACKUP_PATH"
cat "${BACKUP_PATH}/manifest.json"
echo

read -rp "This will OVERWRITE live data. Continue? [y/N] " confirm
[[ "${confirm,,}" == "y" ]] || { log "Aborted."; exit 0; }

# ── Verify checksums ──────────────────────────────────────────────────────────
if [[ -f "${BACKUP_PATH}/SHA256SUMS" ]]; then
    log "Verifying checksums..."
    cd "$BACKUP_PATH" && sha256sum -c SHA256SUMS --ignore-missing 2>&1 | grep -v "manifest" | grep -v "SHA256SUMS"
    cd - > /dev/null
fi

# ── Redis restore ──────────────────────────────────────────────────────────────
restore_redis() {
    [[ -f "${BACKUP_PATH}/redis.tar.gz" ]] || { warn "No Redis backup found, skipping"; return; }
    log "Restoring Redis..."
    $COMPOSE stop redis
    $COMPOSE exec -T redis rm -f /data/dump.rdb /data/appendonly.aof 2>/dev/null || true
    $COMPOSE cp "${BACKUP_PATH}/redis.tar.gz" redis:/tmp/redis-backup.tar.gz
    $COMPOSE exec -T redis tar -xzf /tmp/redis-backup.tar.gz -C / --strip-components=0
    $COMPOSE start redis
    sleep 3
    $COMPOSE exec -T redis redis-cli ping
    log "Redis restored"
}

# ── Qdrant restore ────────────────────────────────────────────────────────────
restore_qdrant() {
    local snap_dir="${BACKUP_PATH}/qdrant"
    [[ -d "$snap_dir" ]] || { warn "No Qdrant backup found, skipping"; return; }
    log "Restoring Qdrant snapshots..."
    for snap_file in "${snap_dir}"/*.snapshot 2>/dev/null; do
        [[ -f "$snap_file" ]] || continue
        local filename
        filename=$(basename "$snap_file")
        local collection
        collection="${filename%%-*}"
        log "  Restoring collection $collection from $filename"
        curl -sf -XPUT \
            -F "snapshot=@${snap_file}" \
            "http://localhost:6333/collections/${collection}/snapshots/recover" || \
            warn "  Failed to restore $collection"
    done
    log "Qdrant restore complete"
}

# ── MLflow restore ────────────────────────────────────────────────────────────
restore_mlflow() {
    [[ -f "${BACKUP_PATH}/mlflow.tar.gz" ]] || { warn "No MLflow backup found, skipping"; return; }
    log "Restoring MLflow..."
    $COMPOSE stop mlflow
    $COMPOSE exec -T mlflow rm -rf /mlartifacts 2>/dev/null || true
    $COMPOSE cp "${BACKUP_PATH}/mlflow.tar.gz" mlflow:/tmp/mlflow-backup.tar.gz
    $COMPOSE exec -T mlflow tar -xzf /tmp/mlflow-backup.tar.gz -C /
    $COMPOSE start mlflow
    log "MLflow restored"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    restore_redis
    restore_qdrant
    restore_mlflow
    log "=== Restore complete ==="
    log "Restart services: docker compose restart"
}

main "$@"
