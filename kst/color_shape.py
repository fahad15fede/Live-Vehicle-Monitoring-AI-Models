"""Colour + shape estimation without a trained model.

model.md plans a MobileNetV3 colour classifier, but no weights were delivered,
so this follows the same preprocessing (Shades-of-Gray constancy) and then
votes over HSV pixels from the vehicle body. Shape is a coarse silhouette
bucket from the box aspect ratio plus the detector class.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .imaging import shades_of_gray

# (name, hue range in degrees) for chromatic colours.
_HUES = [
    ("red", 0, 15), ("orange", 15, 40), ("yellow", 40, 70), ("green", 70, 165),
    ("blue", 165, 255), ("purple", 255, 300), ("red", 300, 361),
]


def _body_region(img: Image.Image) -> Image.Image:
    """Middle band of the crop: skips road at the bottom and glass/sky on top."""
    w, h = img.size
    return img.crop((int(w * 0.15), int(h * 0.30), int(w * 0.85), int(h * 0.80)))


def estimate_color(img: Image.Image) -> tuple[str, float]:
    body = shades_of_gray(_body_region(img)).resize((48, 32))
    hsv = np.asarray(body.convert("HSV"), dtype=np.float32)
    h = hsv[..., 0].ravel() * 360.0 / 255.0
    s = hsv[..., 1].ravel() / 255.0
    v = hsv[..., 2].ravel() / 255.0

    labels = np.empty(h.shape, dtype=object)
    achrom = s < 0.22
    labels[achrom & (v >= 0.72)] = "white"
    labels[achrom & (v < 0.72) & (v >= 0.42)] = "silver"
    labels[achrom & (v < 0.42) & (v >= 0.22)] = "gray"
    labels[v < 0.22] = "black"
    chrom = ~achrom & (v >= 0.22)
    for name, lo, hi in _HUES:
        labels[chrom & (h >= lo) & (h < hi)] = name
    # dark, low-saturation orange/red reads as brown on Karachi roads
    labels[chrom & (h >= 10) & (h < 45) & (v < 0.5)] = "brown"

    names, counts = np.unique(labels[labels != None], return_counts=True)  # noqa: E711
    if not len(names):
        return "unknown", 0.0
    i = int(counts.argmax())
    return str(names[i]), float(counts[i] / counts.sum())


def estimate_shape(class_name: str, box: tuple[int, int, int, int]) -> str:
    w, h = box[2] - box[0], box[3] - box[1]
    ratio = w / max(1, h)
    if class_name in {"motorcycle"}:
        return "two-wheeler"
    if class_name in {"auto_rickshaw", "qingqi"}:
        return "three-wheeler"
    if class_name in {"bus", "truck"}:
        return "heavy / long body" if ratio > 1.6 else "heavy / boxy"
    if class_name == "van":
        return "boxy van"
    if ratio > 1.9:
        return "long / sedan profile"
    if ratio < 1.15:
        return "tall / SUV or front view"
    return "compact / hatchback"
