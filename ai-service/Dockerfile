# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# ai-service — Stable Diffusion XL + LoRA (Diffusers, GPU)
#
# CUDA-enabled PyTorch base. torch 2.6 + cu126 covers compute capability
# 8.0-9.0 (A100/A5000/L4/H100). For Blackwell (cc >= 10.0) use a cu128+ base.
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache \
    SDXL_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 \
    SDXL_VAE_ID=madebyollin/sdxl-vae-fp16-fix

WORKDIR /srv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py app.py ./

# HF cache is a mounted volume in compose; LoRA is pulled to /tmp/lora at runtime.
RUN mkdir -p /hf_cache /tmp/lora

EXPOSE 8100

HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=40 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8100/health/live',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100", "--timeout-keep-alive", "300"]
