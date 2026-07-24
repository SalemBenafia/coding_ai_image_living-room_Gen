# CUDA-enabled PyTorch base (Ampere/Ada/Hopper compatible: cu12.x).
# torch + torchvision are preinstalled in this image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf_cache

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py app.py ./

# Model cache + LoRA download location (mounted as a volume in compose).
RUN mkdir -p /models/hf_cache /models/lora

EXPOSE 8100
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
