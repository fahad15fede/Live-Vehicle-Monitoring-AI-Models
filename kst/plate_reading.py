"""Licence-plate locating and reading.

  * FastAlprPlateReader - fast-alpr (github.com/ankandrew/fast-alpr): YOLOv9 plate detector
                          + fast-plate-ocr recogniser, both ONNX. Primary plate locator/reader.
  * RapidOcrPlateReader - RapidOCR (PaddleOCR on ONNX), fallback reader. Its DBNet text detector
                          and a Pillow rectangle search (find_plate_blob) are fallback locators,
                          used only when fast-alpr is unavailable.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
from PIL import Image, ImageFilter

from . import config
from .image_utils import clean_plate_crop, pil_to_bgr

# Loose Pakistani plate pattern: 2-3 letters, optional year/dash, 2-4 digits.
PLATE_RE = re.compile(r"[A-Z]{1,4}[- ]?\d{0,2}[- ]?\d{2,4}")


def is_valid_plate(text: str) -> bool:
    return len(text) >= 4 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text)


class FastAlprPlateReader:
    """fast-alpr's plate detector and OCR, called separately so the detector runs on the
    vehicle crop and the OCR on the plate box we pick."""

    def __init__(self):
        from fast_alpr.default_detector import DefaultDetector
        from fast_alpr.default_ocr import DefaultOCR
        self.detector = DefaultDetector(model_name=config.ALPR_DETECTOR,
                                        conf_thresh=config.ALPR_DET_CONF)
        self.ocr = DefaultOCR(hub_ocr_model=config.ALPR_OCR, device="cpu")

    def boxes(self, crop_bgr: np.ndarray) -> list[tuple[float, int, int, int, int]]:
        h, w = crop_bgr.shape[:2]
        out = []
        for d in self.detector.predict(crop_bgr):
            b = d.bounding_box
            out.append((float(d.confidence), max(0, b.x1), max(0, b.y1), min(w, b.x2), min(h, b.y2)))
        return out

    def read(self, plate_bgr: np.ndarray) -> tuple[str | None, float]:
        if plate_bgr.size == 0:
            return None, 0.0
        r = self.ocr.predict(plate_bgr)
        if r is None or not r.text:
            return None, 0.0
        conf = r.confidence if isinstance(r.confidence, list) else [r.confidence]
        # '_' is the model's padding character; score only the real characters
        if len(conf) == len(r.text):
            conf = [c for ch, c in zip(r.text, conf) if ch != "_"]
        text = re.sub(r"[^A-Z0-9]", "", r.text.upper())
        if not is_valid_plate(text) or not conf:
            return None, 0.0
        return text, float(np.mean(conf))


class RapidOcrPlateReader:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.ocr = RapidOCR()

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"[^A-Z0-9\- ]", "", text.upper()).strip()
        m = PLATE_RE.search(text.replace(" ", ""))
        return m.group(0) if m else text.replace(" ", "")

    def find_text_boxes(self, img: Image.Image) -> list[tuple[float, int, int, int, int]]:
        """DBNet text detection only. Returns (score, x1, y1, x2, y2) in `img` coords."""
        scale = max(1.0, 320 / max(1, img.width))       # small crops: upscale so glyphs exist
        big = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS) \
            if scale > 1 else img
        res, _ = self.ocr(pil_to_bgr(big), use_det=True, use_cls=False, use_rec=False)
        out = []
        for quad in res or []:
            q = np.asarray(quad, dtype=np.float32) / scale
            out.append((1.0, int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())))
        return out

    def read(self, plate_img: Image.Image, detect_text: bool) -> tuple[str | None, float]:
        img = pil_to_bgr(clean_plate_crop(plate_img))
        res, _ = self.ocr(img) if detect_text else self.ocr(img, use_det=False, use_cls=False)
        if not res:
            return None, 0.0
        if detect_text:   # [[box, text, score], ...] - join lines top-to-bottom (stacked plates)
            res = sorted(res, key=lambda r: (r[0][0][1] // 20, r[0][0][0]))
            text = "".join(r[1] for r in res)
            score = float(np.mean([r[2] for r in res]))
        else:
            text, score = res[0][0], float(res[0][1])
        text = self._clean_text(text)
        if not is_valid_plate(text.replace("-", "")):
            return None, 0.0
        return text, score


def is_plate_shaped(b, w, h) -> bool:
    bw, bh = b[2] - b[0], b[3] - b[1]
    if bw < 6 or bh < 3:
        return False
    return 1.0 <= bw / bh <= 6.5 and bw <= 0.75 * w and (bw * bh) <= 0.12 * w * h


def has_plate_background(img: Image.Image) -> bool:
    """Pakistani plates are white, yellow (commercial) or Sindh-Ajrak white/green with dark
    glyphs: a real plate has a good share of bright, non-red pixels."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    if hsv.size == 0:
        return False
    s, v, hue = hsv[..., 1], hsv[..., 2], hsv[..., 0] * 360 // 255
    white = (s < 70) & (v > 140)
    yellow = (hue > 35) & (hue < 70) & (s > 80) & (v > 120)
    red = ((hue < 20) | (hue > 330)) & (s > 90) & (v > 90)     # CCTV overlays use red text
    return float((white | yellow).mean()) >= 0.30 and float(red.mean()) < 0.05


def plate_position_score(b, w, h) -> float:
    """Prefer horizontally centred, low (bumper height) boxes."""
    cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
    return -abs(cx - 0.5) * 2 + cy


def find_plate_blob(img: Image.Image):
    """Pillow/numpy search for a bright white or yellow plate-shaped rectangle."""
    w, h = img.size
    if w < 20 or h < 10:
        return None
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    s, v, hue = hsv[..., 1], hsv[..., 2], hsv[..., 0] * 360 // 255
    body = np.median(v)
    white = (s < 60) & (v > max(150, body + 35))
    yellow = (hue > 35) & (hue < 70) & (s > 90) & (v > 130)
    mask = Image.fromarray(((white | yellow) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    m = np.asarray(mask) > 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    best, best_rank = None, -9.0
    for x, y, bw, bh, area in stats[1:]:
        b = (x, y, x + bw, y + bh)
        if not is_plate_shaped(b, w, h * 2) or area < 0.5 * bw * bh or bw < 0.08 * w:
            continue
        r = plate_position_score(b, w, h)
        if r > best_rank:
            best, best_rank = b, r
    return best
