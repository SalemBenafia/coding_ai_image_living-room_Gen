#!/usr/bin/env python
"""Render an nginx config for the NATIVE runtime.

The docker frontend proxies to the `backend` service name on the compose
network; natively the backend is on localhost:8000 and the static files live in
the repo. Usage: render_nginx.py <out_conf> <listen_port>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEBROOT = REPO_ROOT / "frontend"

TEMPLATE = """
# Native runtime only: workers run as root so they can read the web root under
# /root (mode 700). In Docker the frontend is an nginx image with its own copy,
# so this does not apply there.
user root;
worker_processes 1;
error_log {logs}/nginx-error.log warn;
pid {runtime}/nginx.pid;
events {{ worker_connections 512; }}
http {{
    include {mime};
    default_type application/octet-stream;
    access_log {logs}/nginx-access.log;
    sendfile on;
    client_max_body_size 25m;

    server {{
        listen {port};
        server_name _;
        root {webroot};
        index index.html;

        location = /healthz {{ return 200 "ok\\n"; add_header Content-Type text/plain; }}

        location / {{ try_files $uri $uri/ /index.html; }}

        location /api/ {{
            proxy_pass http://127.0.0.1:8000/api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_connect_timeout 60s;
            proxy_send_timeout 360s;
            proxy_read_timeout 360s;
        }}
        location = /health {{ proxy_pass http://127.0.0.1:8000/health; }}
    }}
}}
"""


def find_mime() -> str:
    for cand in ("/etc/nginx/mime.types", "/usr/local/nginx/conf/mime.types"):
        if os.path.exists(cand):
            return cand
    return "/etc/nginx/mime.types"


def main() -> int:
    out = Path(sys.argv[1])
    port = sys.argv[2] if len(sys.argv) > 2 else "10100"
    runtime = REPO_ROOT / ".runtime"
    logs = runtime / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.format(
            logs=logs,
            runtime=runtime,
            mime=find_mime(),
            port=port,
            webroot=WEBROOT,
        )
    )
    print(f"[nginx] wrote {out} (listen :{port}, root {WEBROOT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
