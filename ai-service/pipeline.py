"""SDXL + LoRA generation pipeline.

Loads Stable Diffusion XL once at startup with the fp16-fixed VAE, optionally
applies a LoRA adapter pulled from MinIO, and exposes a thread-safe `generate()`
returning PNG bytes. The LoRA can be (re)loaded at runtime via `reload_lora()`
so a freshly trained adapter can be hot-swapped without restarting the service.
"""
from __future__ import annotations

import io
import logging
import os
import threading

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

logger = logging.getLogger("pipeline")

LOCAL_LORA_DIR = "/tmp/lora"


def _round8(v: int, lo: int, hi: int) -> int:
    v = max(lo, min(hi, int(v)))
    return v - (v % 8)


class SDXLPipeline:
    def __init__(self) -> None:
        self.model_id = os.environ.get(
            "SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.vae_id = os.environ.get("SDXL_VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
        self.device = os.environ.get("AI_DEVICE", "cuda")

        # LoRA config. The adapter is stored in MinIO under models/<...>.safetensors.
        self.lora_object_key = os.environ.get(
            "LORA_OBJECT_KEY",
            "models/living-room-style-v1/pytorch_lora_weights.safetensors",
        ).strip()
        self.lora_bucket = os.environ.get("MINIO_BUCKET_MODELS", "models")
        self.lora_scale = float(os.environ.get("LORA_SCALE", "0.8"))
        self.lora_autoload = os.environ.get("LORA_AUTOLOAD", "true").lower() == "true"
        self.lora_adapter_name = "living_room"
        # Activation phrase the LoRA was trained with. When the adapter is loaded
        # we prepend it so the learned living-room style reliably activates, even
        # though end users never type it.
        self.lora_trigger = os.environ.get("LORA_TRIGGER", "lvngrm living room").strip()

        self.max_steps = int(os.environ.get("MAX_STEPS", "60"))
        self.max_side = int(os.environ.get("MAX_SIDE", "1536"))

        self.pipe: StableDiffusionXLPipeline | None = None
        self.lora_loaded = False
        self.lora_source: str | None = None
        self._lock = threading.Lock()  # serialize GPU work
        self._ready = False
        self._load_error: str | None = None

    # ---- lifecycle -------------------------------------------------------
    def load(self) -> None:
        try:
            use_cuda = self.device == "cuda" and torch.cuda.is_available()
            dtype = torch.float16 if use_cuda else torch.float32
            logger.info(
                "Loading SDXL '%s' (device=%s, dtype=%s)",
                self.model_id, "cuda" if use_cuda else "cpu", dtype,
            )

            # fp16-fixed VAE avoids the black-image / NaN issue with SDXL in fp16.
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=dtype)
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if use_cuda else None,
            )
            self.pipe = self.pipe.to("cuda" if use_cuda else "cpu")
            self.pipe.enable_vae_tiling()
            if use_cuda:
                self.pipe.enable_vae_slicing()

            if self.lora_autoload:
                self._try_load_lora()

            self._ready = True
            logger.info("SDXL pipeline ready (lora_loaded=%s).", self.lora_loaded)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.exception("Failed to load SDXL pipeline")

    def _download_lora(self) -> str | None:
        """Fetch the LoRA safetensors from MinIO to a local path. None if absent."""
        try:
            from minio import Minio
            from minio.error import S3Error

            client = Minio(
                os.environ.get("MINIO_ENDPOINT", "minio:9000"),
                access_key=os.environ.get("MINIO_ROOT_USER", "admin"),
                secret_key=os.environ.get("MINIO_ROOT_PASSWORD", ""),
                secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            )
            os.makedirs(LOCAL_LORA_DIR, exist_ok=True)
            weight_name = os.path.basename(self.lora_object_key)
            local_path = os.path.join(LOCAL_LORA_DIR, weight_name)
            try:
                client.fget_object(self.lora_bucket, self.lora_object_key, local_path)
            except S3Error as exc:
                if exc.code in ("NoSuchKey", "NoSuchBucket"):
                    logger.info(
                        "No LoRA at %s/%s yet; using base SDXL.",
                        self.lora_bucket, self.lora_object_key,
                    )
                    return None
                raise
            logger.info("Downloaded LoRA -> %s", local_path)
            return local_path
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch LoRA (%s); using base SDXL.", exc)
            return None

    def _try_load_lora(self) -> None:
        if self.pipe is None:
            return
        local_path = self._download_lora()
        if not local_path:
            return
        weight_name = os.path.basename(local_path)
        # Unload any previous adapter so reloads are clean.
        try:
            self.pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001
            pass
        self.pipe.load_lora_weights(
            LOCAL_LORA_DIR, weight_name=weight_name, adapter_name=self.lora_adapter_name
        )
        self.lora_loaded = True
        self.lora_source = f"{self.lora_bucket}/{self.lora_object_key}"
        logger.info("LoRA applied from %s (scale=%.2f).", self.lora_source, self.lora_scale)

    def reload_lora(self) -> bool:
        """Re-download and apply the LoRA (hot-swap after training). Serialized."""
        with self._lock:
            try:
                self._try_load_lora()
                return self.lora_loaded
            except Exception as exc:  # noqa: BLE001
                logger.exception("reload_lora failed")
                self._load_error = str(exc)
                return False

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
        lora_scale: float | None = None,
    ) -> bytes:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded")

        width = _round8(width, 512, self.max_side)
        height = _round8(height, 512, self.max_side)
        steps = max(1, min(self.max_steps, int(steps)))
        guidance_scale = float(guidance_scale)
        scale = self.lora_scale if lora_scale is None else float(lora_scale)

        device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))

        kwargs: dict = {}
        if self.lora_loaded:
            kwargs["cross_attention_kwargs"] = {"scale": scale}
            # Prepend the trigger phrase (once) so the LoRA style activates.
            if self.lora_trigger and self.lora_trigger.lower() not in prompt.lower():
                prompt = f"{self.lora_trigger}, {prompt}"

        with self._lock:  # one generation at a time on the GPU
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )
        image: Image.Image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
