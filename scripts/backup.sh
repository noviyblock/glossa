#!/usr/bin/env bash
# Backup strategy: Redis (RDB+AOF), Qdrant snapshots, MLflow artifacts, Grafana state
# Backups are compressed and optionally uploaded to S3 / GCS.
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/opt/glossa/backups}"
TIMESTAMP=$(date -u '+%Y%m%dT%H%M%SZ')
BACKUP_PATH="${BACKUP_DIR}/${TIMESTAMP}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
S3_BUCKET="${S3_BACKUP_BUCKET:-}"
COMPOSE="docker compose"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log()   { echo -e "${GREEN}[backup $(date +%H:%M:%S)]${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }

mkdir -p "${BACKUP_PATH}"

# ── Redis ─────────────────────────────────────────────────────────────────────
backup_redis() {
    log "Backing up Redis..."
    # Trigger BGSAVE and wait for completion
    $COMPOSE exec -T redis redis-cli BGSAVE
    sleep 3
    local retries=10
    while (( retries-- > 0 )); do
        local status
        status=$($COMPOSE exec -T redis redis-cli LASTSAVE)
        if [[ "$status" != "0" ]]; then break; fi
        sleep 2
    done

    # Copy dump.rdb from volume via container
    $COMPOSE exec -T redis tar -czf - /data/dump.rdb /data/appendonly.aof \
        2>/dev/null > "${BACKUP_PATH}/redis.tar.gz" || true

    log "Redis backup: ${BACKUP_PATH}/redis.tar.gz"
}

# ── Qdrant ────────────────────────────────────────────────────────────────────
backup_qdrant() {
    log "Backing up Qdrant..."
    # Use Qdrant snapshot API
    local snapshot_url="http://localhost:6333"
    local collections
    collections=$(curl -sf "${snapshot_url}/collections" | \
                  python3 -c "import sys,json; \
                  d=json.load(sys.stdin); \
                  [print(c['name']) for c in d.get('result',{}).get('collections',[])]" \
                  2>/dev/null || echo "")

    mkdir -p "${BACKUP_PATH}/qdrant"
    for collection in $collections; do
        log "  Snapshotting collection: $collection"
        local snap_resp
        snap_resp=$(curl -sf -XPOST "${snapshot_url}/collections/${collection}/snapshots" || echo "{}")
        local snap_name
        snap_name=$(echo "$snap_resp" | python3 -c "import sys,json; \
                    d=json.load(sys.stdin); print(d.get('result',{}).get('name',''))" 2>/dev/null || echo "")
        if [[ -n "$snap_name" ]]; then
            curl -sf "${snapshot_url}/collections/${collection}/snapshots/${snap_name}" \
                -o "${BACKUP_PATH}/qdrant/${collection}-${snap_name}" || true
            log "  Saved snapshot: ${collection}-${snap_name}"
        fi
    done
    log "Qdrant backup complete"
}

# ── MLflow ────────────────────────────────────────────────────────────────────
backup_mlflow() {
    log "Backing up MLflow artifacts..."
    $COMPOSE exec -T mlflow tar -czf - /mlartifacts \
        > "${BACKUP_PATH}/mlflow.tar.gz" 2>/dev/null || true
    log "MLflow backup: ${BACKUP_PATH}/mlflow.tar.gz"
}

# ── Grafana ───────────────────────────────────────────────────────────────────
backup_grafana() {
    log "Backing up Grafana state..."
    $COMPOSE exec -T grafana tar -czf - /var/lib/grafana \
        > "${BACKUP_PATH}/grafana.tar.gz" 2>/dev/null || true
    log "Grafana backup: ${BACKUP_PATH}/grafana.tar.gz"
}

# ── Compress and checksum ─────────────────────────────────────────────────────
finalize() {
    log "Creating manifest..."
    cat > "${BACKUP_PATH}/manifest.json" <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "hostname": "$(hostname)",
    "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "files": $(ls "${BACKUP_PATH}" | grep -v manifest.json | python3 -c \
        "import sys,json; print(json.dumps(sys.stdin.read().splitlines()))")
}
EOF
    # Checksum all files
    sha256sum "${BACKUP_PATH}"/* > "${BACKUP_PATH}/SHA256SUMS" 2>/dev/null || true
    log "Manifest written"
}

# ── Upload to S3 ──────────────────────────────────────────────────────────────
upload_s3() {
    if [[ -z "$S3_BUCKET" ]]; then return 0; fi
    log "Uploading to s3://${S3_BUCKET}/backups/${TIMESTAMP}/"
    aws s3 sync "${BACKUP_PATH}" "s3://${S3_BUCKET}/backups/${TIMESTAMP}/" \
        --storage-class STANDARD_IA --quiet
    log "S3 upload complete"
}

# ── Prune old backups ──────────────────────────────────────────────────────────
prune_old() {
    log "Pruning backups older than ${RETENTION_DAYS} days..."
    find "${BACKUP_DIR}" -maxdepth 1 -type d -mtime "+${RETENTION_DAYS}" -exec rm -rf {} + 2>/dev/null || true
    log "Prune complete"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    log "=== Glossa Backup — ${TIMESTAMP} ==="
    backup_redis
    backup_qdrant
    backup_mlflow
    backup_grafana
    finalize
    upload_s3
    prune_old
    log "=== Backup complete: ${BACKUP_PATH} ==="
}

main "$@"
