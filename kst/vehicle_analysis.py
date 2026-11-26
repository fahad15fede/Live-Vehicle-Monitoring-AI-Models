"""Stage 2: per-vehicle analysis on each crop.

  * FeatureDetector  - best.pt (YOLOv8n): tinted_windshield, police_revolving_light, front_plate
  * Plates           - see plate_reading.py (fast-alpr first, RapidOCR fallback)
  * Colour / shape   - see appearance.py (called from pipeline.py)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config
from .image_utils import bgr_to_pil, clean_vehicle_crop
from .plate_reading import (FastAlprPlateReader, RapidOcrPlateReader, find_plate_blob,
                            has_plate_background, is_plate_shaped, plate_position_score)


def _ensure_feature_weights():
    if not config.FEATURE_MODEL.exists() and config.FEATURE_MODEL_ZIP.exists():
        shutil.copyfile(config.FEATURE_MODEL_ZIP, config.FEATURE_MODEL)
    return config.FEATURE_MODEL


@dataclass
class VehicleInfo:
    features: list[str] = field(default_factory=list)
    plate_text: str | None = None
    plate_conf: float | None = None
    plate_box: tuple | None = None                              # crop coords
    plate_source: str | None = None                             # which locator found it


class FeatureDetector:
    def __init__(self):
        from ultralytics import YOLO
        self.model = YOLO(str(_ensure_feature_weights()))
        self.names = self.model.names

    def __call__(self, crop: Image.Image, conf: float):
        imgsz = 320 if max(crop.size) <= 400 else 640   # crops are small; 320 is ~4x cheaper
        r = self.model.predict(crop, conf=conf, imgsz=imgsz, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append((self.names[int(b.cls)], float(b.conf), x1, y1, x2, y2))
        return out


class VehicleAnalyzer:
    """Runs every per-crop model and merges their outputs."""

    def __init__(self):
        self.errors: dict[str, str] = {}
        self.features = self._try("features", FeatureDetector)
        self.alpr = self._try("fast-alpr", FastAlprPlateReader)
        self.plates = self._try("ocr", RapidOcrPlateReader)

    def _try(self, name, factory):
        try:
            return factory()
        except Exception as e:  # keep the app alive if one model is missing
            self.errors[name] = f"{type(e).__name__}: {e}"
            return None

    def analyze(self, crop_bgr: np.ndarray, feat_conf: float, plate_conf: float,
                vehicle_type: str = "car") -> VehicleInfo:
        info = VehicleInfo()
        pil = clean_vehicle_crop(bgr_to_pil(crop_bgr))
        front_plates = []
        if self.features:
            for label, c, *xyxy in self.features(pil, feat_conf):
                if label == "front_plate":
                    front_plates.append((c, *xyxy))
                else:
                    info.features.append(label)

        box, source = self.locate_plate(crop_bgr, pil, front_plates, vehicle_type)
        info.plate_box, info.plate_source = box, source
        if box is not None:
            text, score = self.read_plate(crop_bgr, pil, box)
            if text and score >= plate_conf:
                info.plate_text, info.plate_conf = text, score
        info.features.append("front_plate" if box is not None else "no_front_plate_seen")
        info.features = sorted(set(info.features))
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

    def locate_plate(self, crop_bgr: np.ndarray, pil: Image.Image, front_plates=(),
                     vehicle_type: str = "car") -> tuple[tuple | None, str | None]:
        """Plate box in crop coords. fast-alpr's plate detector when it is loaded; otherwise
        best.pt's front_plate box, OCR text detection, or a Pillow bright-rectangle search."""
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
        good = [b for b in front_plates if b[0] >= config.FRONT_PLATE_MIN_CONF
                and is_plate_shaped(b[1:], w, h) and (b[2] + b[4]) / 2 >= top]
        if good:
            return pad(max(good)[1:]), "best.pt"
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
