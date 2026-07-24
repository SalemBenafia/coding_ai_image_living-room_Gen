# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# llm-service — Qwen 2.5 1.5B Instruct (prompt + caption enhancement)
#
# Uses a slim Python base: the torch cu126 wheel bundles the CUDA runtime, so
# only the host NVIDIA driver (injected by nvidia-container-runtime) is needed.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    LLM_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Model weights live in the mounted HF cache volume, not in the image.
RUN mkdir -p /hf_cache

# Run as a non-root user; the cache mount must be writable by this uid.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /hf_cache
USER appuser

EXPOSE 8200

HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8200/health/live',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200", "--timeout-keep-alive", "75"]
