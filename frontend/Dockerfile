FROM nginx:1.27-alpine

# Static site
COPY index.html /usr/share/nginx/html/index.html
COPY css /usr/share/nginx/html/css
COPY js /usr/share/nginx/html/js

# Reverse-proxy config (forwards /api and /health to the backend)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
