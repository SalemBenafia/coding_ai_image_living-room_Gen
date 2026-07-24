#!/usr/bin/env bash
# =============================================================================
# Native runtime launcher — runs the EXACT same service code as docker-compose,
# but under plain processes, for hosts where Docker is unavailable (e.g. an
# unprivileged Vast.ai container: no CAP_SYS_ADMIN / user namespaces).
#
# It brings up: MinIO, PostgreSQL, llm-service (Qwen), caption-service (BLIP),
# ai-service (SDXL+LoRA), backend (FastAPI) and frontend (nginx) — wired with
# the same environment variables docker-compose.yml uses.
#
#   deploy/native/run_stack.sh up      # start everything (idempotent)
#   deploy/native/run_stack.sh down    # stop everything
#   deploy/native/run_stack.sh status  # show what's running
#
# Logs: .runtime/logs/<service>.log     PIDs: .runtime/pids/<service>.pid
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="$REPO_ROOT/.runtime"
LOGS="$RUNTIME/logs"
PIDS="$RUNTIME/pids"
BIN="$REPO_ROOT/scripts/.bin"
VENV="/venv/main"
PY="$VENV/bin/python"

mkdir -p "$LOGS" "$PIDS" "$RUNTIME/minio-data" "$RUNTIME/postgres-data"

# ---- shared environment (mirrors .env.example / docker-compose defaults) -----
export HF_HOME="${HF_HOME:-/root/hf_cache}"

export MINIO_ROOT_USER="${MINIO_ROOT_USER:-admin}"
export MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-password123}"
export MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
export MINIO_SECURE="${MINIO_SECURE:-false}"
export MINIO_BUCKET_DATA="${MINIO_BUCKET_DATA:-interior-design-data}"
export MINIO_BUCKET_IMAGES="${MINIO_BUCKET_IMAGES:-generated-images}"
export MINIO_BUCKET_MODELS="${MINIO_BUCKET_MODELS:-models}"

export POSTGRES_USER="${POSTGRES_USER:-livingroom}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-livingroom_pw}"
export POSTGRES_DB="${POSTGRES_DB:-livingroom}"
export PGPORT="${PGPORT:-5432}"
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${PGPORT}/${POSTGRES_DB}}"

export AI_SERVICE_URL="${AI_SERVICE_URL:-http://localhost:8100}"
export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://localhost:8200}"
export CAPTION_SERVICE_URL="${CAPTION_SERVICE_URL:-http://localhost:8300}"

export SDXL_MODEL_ID="${SDXL_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}"
export SDXL_VAE_ID="${SDXL_VAE_ID:-madebyollin/sdxl-vae-fp16-fix}"
export LLM_MODEL_ID="${LLM_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
export BLIP_MODEL_ID="${BLIP_MODEL_ID:-Salesforce/blip-image-captioning-large}"

export LORA_NAME="${LORA_NAME:-living-room-style-v1}"
export LORA_OBJECT_KEY="${LORA_OBJECT_KEY:-models/living-room-style-v1/pytorch_lora_weights.safetensors}"
export LORA_SCALE="${LORA_SCALE:-0.8}"
export LORA_AUTOLOAD="${LORA_AUTOLOAD:-true}"
export LORA_TRIGGER="${LORA_TRIGGER:-lvngrm living room}"

export BACKEND_CORS_ORIGINS="${BACKEND_CORS_ORIGINS:-*}"

log()  { echo -e "\033[0;36m[stack]\033[0m $*"; }
err()  { echo -e "\033[0;31m[stack]\033[0m $*" >&2; }

is_running() {  # name -> 0 if pid file process alive
  local name="$1" pidf="$PIDS/$1.pid"
  [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null
}

start_proc() {  # name  "command..."   (cwd via subshell)
  local name="$1"; shift
  if is_running "$name"; then log "$name already running (pid $(cat "$PIDS/$name.pid"))"; return 0; fi
  log "starting $name"
  nohup bash -c "$*" >"$LOGS/$name.log" 2>&1 &
  echo $! >"$PIDS/$name.pid"
}

wait_http() {  # url name timeout
  local url="$1" name="$2" timeout="${3:-60}" i=0
  until curl -fsS -o /dev/null "$url" 2>/dev/null; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { err "$name not ready after ${timeout}s ($url)"; return 1; }
    sleep 1
  done
  log "$name ready ($url)"
}

wait_tcp() {  # port name timeout
  local port="$1" name="$2" timeout="${3:-60}" i=0
  until "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)" 2>/dev/null; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { err "$name port $port not open after ${timeout}s"; return 1; }
    sleep 1
  done
  log "$name listening on :$port"
}

# ---------------------------------------------------------------- MinIO ------
start_minio() {
  start_proc minio "MINIO_ROOT_USER='$MINIO_ROOT_USER' MINIO_ROOT_PASSWORD='$MINIO_ROOT_PASSWORD' \
    '$BIN/minio' server '$RUNTIME/minio-data' --address ':9000' --console-address ':9001'"
  wait_tcp 9000 minio 40
  "$PY" "$REPO_ROOT/deploy/native/init_minio.py"
}

# ------------------------------------------------------------- PostgreSQL ----
PGBIN="$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1)"
# The postgres system user (uid 103) cannot traverse /root (mode 700), so the
# native cluster lives under /var/lib/postgresql (which postgres owns). This is
# runtime state only — never committed. Under Docker, the postgres image manages
# its own volume, so this path is native-runtime specific.
PGDATA_DIR="${PGDATA_DIR:-/var/lib/postgresql/lr-data}"
start_postgres() {
  local pgdata="$PGDATA_DIR"
  mkdir -p "$pgdata"
  if [ ! -f "$pgdata/PG_VERSION" ]; then
    log "initializing PostgreSQL cluster"
    chown -R postgres:postgres "$pgdata"
    sudo -u postgres "$PGBIN/initdb" -D "$pgdata" -U "$POSTGRES_USER" \
      --auth=trust -E UTF8 >"$LOGS/postgres-init.log" 2>&1 || {
        err "initdb failed (see $LOGS/postgres-init.log)"; return 1; }
  fi
  chown -R postgres:postgres "$pgdata"
  if ! is_running postgres; then
    log "starting PostgreSQL on :$PGPORT"
    nohup sudo -u postgres "$PGBIN/postgres" -D "$pgdata" \
      -c listen_addresses='127.0.0.1' -p "$PGPORT" >"$LOGS/postgres.log" 2>&1 &
    echo $! >"$PIDS/postgres.pid"
  fi
  wait_tcp "$PGPORT" postgres 40
  # initdb -U made $POSTGRES_USER the bootstrap superuser (connect via the
  # default 'postgres' db). Set its password and create the app database.
  local psql=(sudo -u postgres "$PGBIN/psql" -U "$POSTGRES_USER" -d postgres -p "$PGPORT" -v ON_ERROR_STOP=1)
  "${psql[@]}" -c "ALTER ROLE $POSTGRES_USER PASSWORD '$POSTGRES_PASSWORD'" >/dev/null
  if ! "${psql[@]}" -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'" | grep -q 1; then
    "${psql[@]}" -c "CREATE DATABASE $POSTGRES_DB OWNER $POSTGRES_USER" >/dev/null
    log "created database '$POSTGRES_DB'"
  fi
  log "postgres database '$POSTGRES_DB' ready"
}

# ------------------------------------------------------------- AI services ---
start_llm() {
  start_proc llm-service "cd '$REPO_ROOT/llm-service' && exec '$VENV/bin/uvicorn' app:app --host 0.0.0.0 --port 8200 --timeout-keep-alive 75"
}
start_caption() {
  start_proc caption-service "cd '$REPO_ROOT/caption-service' && exec '$VENV/bin/uvicorn' app:app --host 0.0.0.0 --port 8300 --timeout-keep-alive 75"
}
start_ai() {
  start_proc ai-service "cd '$REPO_ROOT/ai-service' && exec '$VENV/bin/uvicorn' app:app --host 0.0.0.0 --port 8100 --timeout-keep-alive 300"
}
start_backend() {
  start_proc backend "cd '$REPO_ROOT/backend' && exec '$VENV/bin/uvicorn' app.main:app --host 0.0.0.0 --port 8000"
  wait_http "http://localhost:8000/api/health/live" backend 40
}

# --------------------------------------------------------------- frontend ----
start_frontend() {
  local conf="$RUNTIME/nginx.conf" port="${FRONTEND_HOST_PORT:-10100}"
  "$PY" "$REPO_ROOT/deploy/native/render_nginx.py" "$conf" "$port"
  start_proc frontend "exec nginx -c '$conf' -g 'daemon off;'"
  wait_http "http://localhost:$port/healthz" frontend 30
}

# ------------------------------------------------------------------ verbs -----
cmd_up() {
  start_minio
  start_postgres
  start_llm
  start_caption
  start_ai
  start_backend
  start_frontend
  echo
  cmd_status
  log "AI model services load in the background; check /health for readiness."
}

cmd_down() {
  for name in frontend backend ai-service caption-service llm-service postgres minio; do
    local pidf="$PIDS/$name.pid"
    if [ -f "$pidf" ]; then
      local pid; pid="$(cat "$pidf")"
      if kill -0 "$pid" 2>/dev/null; then log "stopping $name (pid $pid)"; kill "$pid" 2>/dev/null; fi
      # postgres may fork; also stop child postmaster
      pkill -P "$pid" 2>/dev/null || true
      rm -f "$pidf"
    fi
  done
  # postgres started via sudo -u may not be tracked by the launcher pid; sweep it.
  pkill -u postgres postgres 2>/dev/null || true
  log "stack stopped"
}

cmd_status() {
  printf "%-16s %-9s %-8s %s\n" SERVICE STATE PID ENDPOINT
  declare -A ep=(
    [minio]="http://localhost:9000 (console :9001)"
    [postgres]="localhost:$PGPORT"
    [llm-service]="http://localhost:8200/health"
    [caption-service]="http://localhost:8300/health"
    [ai-service]="http://localhost:8100/health"
    [backend]="http://localhost:8000/health"
    [frontend]="http://localhost:${FRONTEND_HOST_PORT:-10100}"
  )
  for name in minio postgres llm-service caption-service ai-service backend frontend; do
    if is_running "$name"; then
      printf "%-16s \033[0;32m%-9s\033[0m %-8s %s\n" "$name" running "$(cat "$PIDS/$name.pid")" "${ep[$name]}"
    else
      printf "%-16s \033[0;31m%-9s\033[0m %-8s %s\n" "$name" stopped "-" "${ep[$name]}"
    fi
  done
}

case "${1:-up}" in
  up)      cmd_up ;;
  down)    cmd_down ;;
  restart) cmd_down; sleep 2; cmd_up ;;
  status)  cmd_status ;;
  *) err "usage: $0 {up|down|restart|status}"; exit 2 ;;
esac
