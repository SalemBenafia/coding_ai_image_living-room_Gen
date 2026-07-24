# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# caption-service — BLIP image captioning for the LoRA training pipeline
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    BLIP_MODEL_ID=Salesforce/blip-image-captioning-large

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /hf_cache
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app /hf_cache
USER appuser

EXPOSE 8300

HEALTHCHECK --interval=15s --timeout=10s --start-period=90s --retries=20 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8300/health/live',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8300", "--timeout-keep-alive", "75"]
