# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# workshop — offline data + LoRA training pipeline (GPU)
#
# Not part of the always-on stack; run on demand:
#   docker compose --profile workshop run --rm workshop make all
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    WORKSHOP_DATA=/data

WORKDIR /workshop

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x *.sh

RUN mkdir -p /hf_cache /data

# Default: print the Makefile help. Override to run a pipeline target.
CMD ["make", "help"]
