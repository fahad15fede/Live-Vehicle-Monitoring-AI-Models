"""Stage 2: per-vehicle analysis on each crop.

  * FeatureDetector  - best.pt (YOLOv8n): tinted_windshield, police_revolving_light, front_plate
  * PlateReader      - finds the plate (best.pt `front_plate` box, else the lower part
                       of the crop), cleans it with Pillow, reads it with RapidOCR
                       (PaddleOCR models on ONNX Runtime).
  * Colour / shape   - see color_shape.py

plateReader.pt as delivered is a stock COCO YOLOv8n (80 generic classes) and cannot
read plates. CustomPlateModel only switches on if the file is replaced by weights
whose classes look like plates or characters.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config
from .imaging import bgr_to_pil, clean_plate_crop, clean_vehicle_crop, pil_to_bgr

# Loose Pakistani plate pattern: 2-3 letters, optional year/dash, 2-4 digits.
PLATE_RE = re.compile(r"[A-Z]{1,4}[- ]?\d{0,2}[- ]?\d{2,4}")


def _ensure_feature_weights():
    if not config.FEATURE_MODEL.exists() and config.FEATURE_MODEL_ZIP.exists():
        shutil.copyfile(config.FEATURE_MODEL_ZIP, config.FEATURE_MODEL)
    return config.FEATURE_MODEL


@dataclass
class VehicleInfo:
    features: list[str] = field(default_factory=list)
    feature_boxes: list[tuple] = field(default_factory=list)   # (label, conf, x1,y1,x2,y2) crop coords
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


class CustomPlateModel:
    """Optional slot for a real plate detector in plateReader.pt."""

    def __init__(self):
        self.model = None
        self.status = "missing"
        if not config.PLATE_MODEL.exists():
            return
        from ultralytics import YOLO
        m = YOLO(str(config.PLATE_MODEL))
        names = {str(n).lower() for n in m.names.values()}
        if any("plate" in n for n in names) or len(names) <= 40:
            self.model, self.status = m, "active"
        else:
            self.status = f"disabled: generic {len(names)}-class model (COCO), not a plate reader"

    def boxes(self, crop: Image.Image, conf: float):
        if self.model is None:
            return []
        r = self.model.predict(crop, conf=conf, verbose=False)[0]
        return [(float(b.conf), *(int(v) for v in b.xyxy[0].tolist())) for b in r.boxes]


class PlateReader:
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
        has_digit = any(c.isdigit() for c in text)
        if len(text) < 4 or not has_digit:
            return None, 0.0
        return text, score


class VehicleAnalyzer:
    """Runs every per-crop model and merges their outputs."""

    def __init__(self):
        self.errors: dict[str, str] = {}
        self.features = self._try("features", FeatureDetector)
        self.plates = self._try("ocr", PlateReader)
        self.custom_plate = self._try("plateReader.pt", CustomPlateModel)

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
        plate_candidates = []
        if self.features:
            for label, c, *xyxy in self.features(pil, feat_conf):
                if label == "front_plate":
                    plate_candidates.append((c, *xyxy))
                else:
                    info.features.append(label)
                    info.feature_boxes.append((label, c, *xyxy))
        if self.custom_plate:
            plate_candidates += self.custom_plate.boxes(pil, plate_conf)

        box, source = self.locate_plate(pil, plate_candidates, vehicle_type)
        info.plate_box, info.plate_source = box, source
        if box is not None and self.plates:
            plate = pil.crop(box)
            text, score = self.plates.read(plate, detect_text=False)
            if text is None:            # retry with detection (two-line bike/rickshaw plates)
                text, score = self.plates.read(plate, detect_text=True)
            if text and score >= plate_conf:
                info.plate_text, info.plate_conf = text, score
        info.features.append("front_plate" if box is not None else "no_front_plate_seen")
        info.features = sorted(set(info.features))
        return info

    def locate_plate(self, pil: Image.Image, model_boxes,
                     vehicle_type: str = "car") -> tuple[tuple | None, str | None]:
        """Plate box in crop coords, from (in order): a confident model box, OCR text
        detection on the lower part of the vehicle, or a Pillow bright-rectangle search."""
        w, h = pil.size
        pad = lambda b: (max(0, b[0] - 3), max(0, b[1] - 3), min(w, b[2] + 3), min(h, b[3] + 3))
        # plate search starts this far down the vehicle box (bike boxes include the rider)
        top = int(h * config.PLATE_SEARCH_FROM.get(vehicle_type, 0.45))
        good = [b for b in model_boxes if b[0] >= config.PLATE_MODEL_MIN_CONF
                and _plate_shaped(b[1:], w, h) and (b[2] + b[4]) / 2 >= top]
        if good:
            return pad(max(good)[1:]), "best.pt"
        lower = pil.crop((0, top, w, h))
        if self.plates:
            texts = [(s, x1, y1 + top, x2, y2 + top) for s, x1, y1, x2, y2 in self.plates.find_text_boxes(lower)]
            texts = [t for t in texts if _plate_shaped(t[1:], w, h)
                     and _plate_background(pil.crop(t[1:]))]   # rejects CCTV overlay text
            if texts:
                return pad(max(texts, key=lambda t: _plate_rank(t[1:], w, h))[1:]), "ocr-detector"
        blob = find_plate_blob(lower)
        if blob:
            x1, y1, x2, y2 = blob
            return pad((x1, y1 + top, x2, y2 + top)), "pillow"
        return None, None


def _plate_shaped(b, w, h) -> bool:
    bw, bh = b[2] - b[0], b[3] - b[1]
    if bw < 6 or bh < 3:
        return False
    return 1.0 <= bw / bh <= 6.5 and bw <= 0.75 * w and (bw * bh) <= 0.12 * w * h


def _plate_background(img: Image.Image) -> bool:
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


def _plate_rank(b, w, h) -> float:
    """Prefer horizontally centred, low (bumper height) boxes."""
    cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
    return -abs(cx - 0.5) * 2 + cy


def find_plate_blob(img: Image.Image):
    """Pillow/numpy search for a bright white or yellow plate-shaped rectangle."""
    from PIL import ImageFilter
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
    import cv2
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    best, best_rank = None, -9.0
    for x, y, bw, bh, area in stats[1:]:
        b = (x, y, x + bw, y + bh)
        if not _plate_shaped(b, w, h * 2) or area < 0.5 * bw * bh or bw < 0.08 * w:
            continue
        r = _plate_rank(b, w, h)
        if r > best_rank:
            best, best_rank = b, r
    return best
