"""Pillow-based cleaning helpers shared by every model stage."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def shades_of_gray(img: Image.Image, p: int = 6) -> Image.Image:
    """Colour constancy (model.md: Shades-of-Gray) to cancel sodium/LED casts."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) + 1e-6
    norm = np.power(np.mean(np.power(arr, p), axis=(0, 1)), 1.0 / p)
    gain = norm.mean() / norm
    return Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8))


def clean_vehicle_crop(img: Image.Image) -> Image.Image:
    """Light denoise + autocontrast before the feature and colour stages."""
    img = img.convert("RGB").filter(ImageFilter.MedianFilter(3))
    return ImageOps.autocontrast(img, cutoff=1)


def clean_plate_crop(img: Image.Image, min_width: int = 240) -> Image.Image:
    """Upscale small plates (cheap stand-in for Real-ESRGAN, which model.md
    applies below 110 px), then sharpen and boost contrast."""
    img = img.convert("RGB")
    if img.width < min_width:
        s = min_width / max(1, img.width)
        img = img.resize((int(img.width * s), max(1, int(img.height * s))), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))


def sharpness(bgr: np.ndarray) -> float:
    """Edge variance via Pillow, used for best-shot scoring."""
    gray = bgr_to_pil(bgr).convert("L")
    return float(np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var())
