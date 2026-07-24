# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# frontend — static HTML/CSS/JS served by nginx, which also reverse-proxies
# /api and /health to the backend service on the compose network.
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine

# Static site
COPY index.html /usr/share/nginx/html/index.html
COPY css /usr/share/nginx/html/css
COPY js /usr/share/nginx/html/js

# Reverse-proxy config (forwards /api and /health to the backend).
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --retries=10 \
  CMD wget -qO- http://localhost:8080/healthz || exit 1
