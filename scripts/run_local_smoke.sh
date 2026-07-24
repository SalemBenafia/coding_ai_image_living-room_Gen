#!/usr/bin/env bash
# Local end-to-end smoke test WITHOUT docker or a GPU.
#
# Boots: MinIO (downloaded static binary) + mock ai-service + backend, then runs
# scripts/smoke_test.py against them. Tears everything down on exit.
#
# Usage:  bash scripts/run_local_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$SCRIPT_DIR/.bin"
DATA_DIR="$(mktemp -d)"
PY="${PYTHON:-python}"

# ---- test environment (matches .env.example defaults) ----
export MINIO_ROOT_USER="admin"
export MINIO_ROOT_PASSWORD="change-me-please-123"
export MINIO_ENDPOINT="localhost:9000"
export MINIO_SECURE="false"
export MINIO_BUCKET_GENERATED="generated-images"
export AI_SERVICE_URL="http://localhost:8100"
export PROMPT_ENHANCER="local"
export DATABASE_URL="sqlite:///$DATA_DIR/smoke.db"
export BACKEND_CORS_ORIGINS="http://localhost:8080"

PIDS=()
cleanup() {
  echo "-- cleanup --"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  rm -rf "$DATA_DIR"
}
trap cleanup EXIT

wait_port() {  # host port name timeout
  local port="$1" name="$2" timeout="${3:-40}" i=0
  until "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { echo "ERROR: $name (:$port) never came up"; exit 1; }
    sleep 1
  done
  echo "  $name up on :$port"
}

# ---- 1. MinIO ----
mkdir -p "$BIN_DIR"
if [ ! -x "$BIN_DIR/minio" ]; then
  echo "-- downloading MinIO server binary --"
  curl -fsSL https://dl.min.io/server/minio/release/linux-amd64/minio -o "$BIN_DIR/minio"
  chmod +x "$BIN_DIR/minio"
fi
echo "-- starting MinIO --"
MINIO_ROOT_USER="$MINIO_ROOT_USER" MINIO_ROOT_PASSWORD="$MINIO_ROOT_PASSWORD" \
  "$BIN_DIR/minio" server "$DATA_DIR/minio" --address ":9000" --console-address ":9001" \
  >"$DATA_DIR/minio.log" 2>&1 &
PIDS+=($!)
wait_port 9000 "MinIO" 40

# ---- 2. mock ai-service ----
echo "-- starting mock ai-service --"
( cd "$SCRIPT_DIR" && "$PY" -m uvicorn mock_ai_service:app --host 127.0.0.1 --port 8100 ) \
  >"$DATA_DIR/ai.log" 2>&1 &
PIDS+=($!)
wait_port 8100 "mock ai-service" 40

# ---- 3. backend ----
echo "-- starting backend --"
( cd "$REPO_ROOT/backend" && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ) \
  >"$DATA_DIR/backend.log" 2>&1 &
PIDS+=($!)
wait_port 8000 "backend" 40

# ---- 4. smoke test ----
echo "-- running smoke test --"
BACKEND="http://localhost:8000" "$PY" "$SCRIPT_DIR/smoke_test.py"
rc=$?

echo "-- backend log (tail) --"; tail -n 15 "$DATA_DIR/backend.log" || true
exit $rc
