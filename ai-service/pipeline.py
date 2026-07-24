"""SDXL + LoRA generation pipeline wrapper.

Loads Stable Diffusion XL once at startup, optionally applies a LoRA adapter pulled
from MinIO, and exposes a thread-safe `generate()` returning PNG bytes.
"""
from __future__ import annotations

import io
import logging
import os
import threading

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

logger = logging.getLogger("pipeline")

MODELS_DIR = "/models"
LORA_DIR = os.path.join(MODELS_DIR, "lora")


def _round8(v: int, lo: int, hi: int) -> int:
    v = max(lo, min(hi, int(v)))
    return v - (v % 8)


class SDXLPipeline:
    def __init__(self) -> None:
        self.model_id = os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        self.device = os.environ.get("AI_DEVICE", "cuda")
        self.lora_object = os.environ.get("LORA_OBJECT", "").strip()
        self.lora_scale = float(os.environ.get("LORA_SCALE", "0.8"))
        self.max_steps = int(os.environ.get("MAX_STEPS", "60"))
        self.max_side = int(os.environ.get("MAX_SIDE", "1536"))

        self.pipe: StableDiffusionXLPipeline | None = None
        self.lora_loaded = False
        self._lock = threading.Lock()  # GPU: serialize generations
        self._ready = False

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        logger.info("Loading SDXL '%s' (device=%s, dtype=%s)", self.model_id,
                    "cuda" if use_cuda else "cpu", dtype)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if use_cuda else None,
        )
        self.pipe = self.pipe.to("cuda" if use_cuda else "cpu")
        # Memory-friendly settings (helpful for larger resolutions).
        self.pipe.enable_vae_tiling()
        if use_cuda:
            self.pipe.enable_vae_slicing()

        self._maybe_load_lora()
        self._ready = True
        logger.info("SDXL pipeline ready (lora_loaded=%s).", self.lora_loaded)

    def _maybe_load_lora(self) -> None:
        if not self.lora_object:
            logger.info("No LORA_OBJECT set; using base SDXL.")
            return
        try:
            from minio import Minio

            client = Minio(
                os.environ["MINIO_ENDPOINT"],
                access_key=os.environ["MINIO_ROOT_USER"],
                secret_key=os.environ["MINIO_ROOT_PASSWORD"],
                secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            )
            bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
            os.makedirs(LORA_DIR, exist_ok=True)
            weight_name = os.path.basename(self.lora_object)
            local_path = os.path.join(LORA_DIR, weight_name)
            logger.info("Downloading LoRA %s/%s -> %s", bucket, self.lora_object, local_path)
            client.fget_object(bucket, self.lora_object, local_path)
            self.pipe.load_lora_weights(LORA_DIR, weight_name=weight_name)
            self.lora_loaded = True
            logger.info("LoRA applied (scale=%.2f).", self.lora_scale)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load LoRA (%s); continuing with base SDXL.", exc)
            self.lora_loaded = False

    def is_ready(self) -> bool:
        return self._ready and self.pipe is not None

    # ---- inference -------------------------------------------------------
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> bytes:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        width = _round8(width, 512, self.max_side)
        height = _round8(height, 512, self.max_side)
        steps = max(1, min(self.max_steps, int(steps)))
        guidance_scale = float(guidance_scale)

        device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

        cross_attention_kwargs = {"scale": self.lora_scale} if self.lora_loaded else None

        with self._lock:  # one generation at a time on the GPU
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        image: Image.Image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
