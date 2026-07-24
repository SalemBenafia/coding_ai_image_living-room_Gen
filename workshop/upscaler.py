"""Real-ESRGAN x4 upscaler (via spandrel, no basicsr dependency).

Loads the official RealESRGAN_x4plus weights and runs them through spandrel's
architecture auto-detection. Tiled inference keeps VRAM bounded for large inputs.
Used by stage 03 to turn Pinterest-thumbnail training images into clean 1024px
pixels so the SDXL VAE never sees hard upscaling artifacts.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader

WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
WEIGHTS_PATH = os.environ.get(
    "ESRGAN_WEIGHTS", os.path.join(os.environ.get("HF_HOME", "/tmp"), "RealESRGAN_x4plus.pth")
)


def _download_weights() -> str:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[upscaler] downloading Real-ESRGAN weights -> {WEIGHTS_PATH}")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
    return WEIGHTS_PATH


class Upscaler:
    def __init__(self, device: str | None = None, use_fp16: bool | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = (self.device == "cuda") if use_fp16 is None else use_fp16
        descriptor = ModelLoader().load_from_file(_download_weights())
        assert isinstance(descriptor, ImageModelDescriptor)
        self.scale = descriptor.scale
        model = descriptor.model.eval().to(self.device)
        if self.use_fp16:
            model = model.half()
        self.model = model

    @torch.inference_mode()
    def _upscale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).clamp(0, 1)

    @torch.inference_mode()
    def upscale(self, image: Image.Image, tile: int = 400, overlap: int = 32) -> Image.Image:
        """4x upscale a PIL image. Tiles large inputs to bound VRAM."""
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()

        _, _, h, w = x.shape
        if max(h, w) <= tile:
            y = self._upscale_tensor(x)
        else:
            y = self._tiled(x, tile, overlap)

        out = (y[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round()
        return Image.fromarray(out.astype(np.uint8))

    def _tiled(self, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
        """Simple tiled inference with overlap blending, scale-aware."""
        s = self.scale
        _, c, h, w = x.shape
        out = torch.zeros((1, c, h * s, w * s), device=self.device, dtype=x.dtype)
        weight = torch.zeros_like(out)
        step = tile - overlap
        for top in range(0, h, step):
            for left in range(0, w, step):
                bottom = min(top + tile, h)
                right = min(left + tile, w)
                patch = x[:, :, top:bottom, left:right]
                up = self._upscale_tensor(patch)
                out[:, :, top * s:bottom * s, left * s:right * s] += up
                weight[:, :, top * s:bottom * s, left * s:right * s] += 1.0
        return out / weight.clamp(min=1.0)


def resize_to(image: Image.Image, size: int, mode: str = "cover") -> Image.Image:
    """Fit an image to size x size. mode='cover' center-crops; 'contain' pads."""
    if mode == "cover":
        from PIL import ImageOps
        return ImageOps.fit(image, (size, size), Image.LANCZOS)
    image = image.copy()
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas
