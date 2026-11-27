"""Stage 2: per-vehicle analysis on each crop.

  * Plates           - see plate_reading.py (fast-alpr first, RapidOCR fallback)
  * Colour / shape   - see appearance.py (called from pipeline.py)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config, device
from .image_utils import bgr_to_pil, clean_vehicle_crop
from .plate_reading import (FastAlprPlateReader, RapidOcrPlateReader, find_plate_blob,
                            has_plate_background, is_plate_shaped, plate_position_score)


@dataclass
class VehicleInfo:
    plate_text: str | None = None
    plate_conf: float | None = None
    plate_box: tuple | None = None                              # crop coords
    plate_source: str | None = None                             # which locator found it


class VehicleAnalyzer:
    """Runs every per-crop model and merges their outputs."""

    def __init__(self):
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()          # live worker + photo tab share these models
        self.alpr = self._try("fast-alpr", FastAlprPlateReader)
        self.plates = self._try("ocr", RapidOcrPlateReader)

    def providers(self) -> dict[str, str]:
        """Execution provider each plate model actually runs on (CUDA / CPU)."""
        out = {}
        for name, obj in (("plate detector", self.alpr and self.alpr.detector),
                          ("plate OCR", self.alpr and self.alpr.ocr), ("RapidOCR", self.plates)):
            if obj:
                p = device.session_providers(obj)
                out[name] = p[0].replace("ExecutionProvider", "") if p else "?"
        return out

    def _try(self, name, factory):
        try:
            return factory()
        except Exception as e:  # keep the app alive if one model is missing
            self.errors[name] = f"{type(e).__name__}: {e}"
            return None

    def analyze(self, crop_bgr: np.ndarray, plate_conf: float,
                vehicle_type: str = "car") -> VehicleInfo:
        with self._lock:
            return self._analyze(crop_bgr, plate_conf, vehicle_type)

    def _analyze(self, crop_bgr: np.ndarray, plate_conf: float, vehicle_type: str) -> VehicleInfo:
        info = VehicleInfo()
        pil = clean_vehicle_crop(bgr_to_pil(crop_bgr))
        box, source = self.locate_plate(crop_bgr, pil, vehicle_type)
        info.plate_box, info.plate_source = box, source
        if box is not None:
            text, score = self.read_plate(crop_bgr, pil, box)
            if text and score >= plate_conf:
                info.plate_text, info.plate_conf = text, score
        return info

    def read_plate(self, crop_bgr: np.ndarray, pil: Image.Image, box) -> tuple[str | None, float]:
        """fast-plate-ocr on the raw crop first, RapidOCR on the cleaned crop as fallback."""
        x1, y1, x2, y2 = box
        if self.alpr:
            text, score = self.alpr.read(crop_bgr[y1:y2, x1:x2])
            if text and score >= config.ALPR_OCR_MIN_CONF:
                return text, score
        if self.plates:
            plate = pil.crop(box)
            text, score = self.plates.read(plate, detect_text=False)
            # retry with text detection (two-line bike/rickshaw plates); ~1.2 s per crop, so
            # only without fast-alpr, whose rejects are mostly plates too blurry to read
            if text is None and not self.alpr:
                text, score = self.plates.read(plate, detect_text=True)
            if text:
                return text, score
        return None, 0.0

    def locate_plate(self, crop_bgr: np.ndarray, pil: Image.Image,
                     vehicle_type: str = "car") -> tuple[tuple | None, str | None]:
        """Plate box in crop coords. fast-alpr's plate detector when it is loaded; otherwise
        OCR text detection, or a Pillow bright-rectangle search."""
        w, h = pil.size
        pad = lambda b: (max(0, b[0] - 3), max(0, b[1] - 3), min(w, b[2] + 3), min(h, b[3] + 3))
        if self.alpr:
            good = [b for b in self.alpr.boxes(crop_bgr) if is_plate_shaped(b[1:], w, h)]
            if good:   # boxes are tight already; padding lowers OCR confidence
                return tuple(max(good)[1:]), "fast-alpr"
            return None, None
        # --- fallbacks, only without fast-alpr ---
        # plate search starts this far down the vehicle box (bike boxes include the rider)
        top = int(h * config.PLATE_SEARCH_FROM.get(vehicle_type, 0.45))
        lower = pil.crop((0, top, w, h))
        if self.plates:
            texts = [(s, x1, y1 + top, x2, y2 + top) for s, x1, y1, x2, y2 in self.plates.find_text_boxes(lower)]
            texts = [t for t in texts if is_plate_shaped(t[1:], w, h)
                     and has_plate_background(pil.crop(t[1:]))]   # rejects CCTV overlay text
            if texts:
                return pad(max(texts, key=lambda t: plate_position_score(t[1:], w, h))[1:]), "ocr-detector"
        blob = find_plate_blob(lower)
        if blob:
            x1, y1, x2, y2 = blob
            return pad((x1, y1 + top, x2, y2 + top)), "pillow"
        return None, None
