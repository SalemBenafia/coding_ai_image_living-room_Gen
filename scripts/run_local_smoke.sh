#!/usr/bin/env bash
# Bring up the native stack (no Docker) and run the end-to-end smoke test against
# the REAL services — SDXL, Qwen, MinIO, Postgres. No mocks.
#
# Usage:  bash scripts/run_local_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/venv/main/bin/python}"

echo "-- starting native stack --"
bash "$REPO_ROOT/deploy/native/run_stack.sh" up

echo "-- waiting for AI services to finish loading models --"
for svc in 8100 8200; do
  for i in $(seq 1 60); do
    ready=$(curl -fsS "http://localhost:$svc/health" 2>/dev/null | "$PY" -c "import sys,json;print(json.load(sys.stdin).get('ready'))" 2>/dev/null || echo False)
    [ "$ready" = "True" ] && { echo "  service on :$svc ready"; break; }
    sleep 3
  done
done

echo "-- running smoke test --"
BACKEND="http://localhost:8000" "$PY" "$REPO_ROOT/scripts/smoke_test.py"
