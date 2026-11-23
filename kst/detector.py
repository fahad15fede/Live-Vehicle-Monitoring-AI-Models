"""Stage 1: YOLO11m vehicle detector (FP16 ONNX via onnxruntime)."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from . import config


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    class_name: str
    confidence: float
    track_id: int = -1
    info: dict[str, Any] = field(default_factory=dict)
    class_scores: dict | None = None   # {class: share}, used for track soft-voting


class VehicleDetector:
    def __init__(self, path=config.VEHICLE_MODEL):
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.fp16 = "float16" in inp.type
        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = [names[i] for i in sorted(names)] or [
            "auto_rickshaw", "bus", "car", "motorcycle", "qingqi", "truck", "van"]

    def _letterbox(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        s = min(self.size / w, self.size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        return canvas, s, px, py

    def __call__(self, frame: np.ndarray, conf: float = config.VEHICLE_CONF) -> list[Detection]:
        img, s, px, py = self._letterbox(frame)
        t = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255.0
        if self.fp16:
            t = t.astype(np.float16)
        pred = np.squeeze(self.session.run(None, {self.input_name: t})[0]).astype(np.float32)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        weights = np.array([config.CLASS_WEIGHT.get(n, 1.0) for n in self.names], np.float32)
        raw = pred[:, 4:]
        cls = (raw * weights).argmax(1)                 # class choice uses the prior weights
        scores = raw[np.arange(len(cls)), cls]          # reported confidence stays raw
        thr = np.array([min(conf, config.CLASS_CONF_FLOOR.get(n, conf)) for n in self.names])
        keep = scores >= thr[cls]
        if not keep.any():
            return []
        xywh, cls, scores, raw = pred[keep, :4], cls[keep], scores[keep], raw[keep]
        H, W = frame.shape[:2]
        x1 = np.clip((xywh[:, 0] - xywh[:, 2] / 2 - px) / s, 0, W)
        y1 = np.clip((xywh[:, 1] - xywh[:, 3] / 2 - py) / s, 0, H)
        x2 = np.clip((xywh[:, 0] + xywh[:, 2] / 2 - px) / s, 0, W)
        y2 = np.clip((xywh[:, 1] + xywh[:, 3] / 2 - py) / s, 0, H)
        boxes = np.stack([x1, y1, x2, y2], 1)
        rects = [[int(a), int(b), int(c - a), int(d - b)] for a, b, c, d in boxes]
        # Class-agnostic NMS: one box per object...
        idx = np.asarray(cv2.dnn.NMSBoxes(rects, scores.tolist(), 0.0, config.NMS_IOU)).reshape(-1)
        out = []
        for i in idx:
            i = int(i)
            x, y, w, h = rects[i]
            if w < 4 or h < 4:
                continue
            # ...but keep the class evidence of the anchors it suppressed: a rickshaw whose
            # best anchor said "car" usually has neighbouring anchors that say "rickshaw".
            same = _iou_many(boxes[i], boxes) >= config.NMS_IOU
            vec = raw[same].max(0) * weights
            c = int(vec.argmax())
            out.append(Detection((x, y, x + w, y + h), self.names[c], float(raw[same][:, c].max()),
                                 class_scores=dict(zip(self.names, (vec / max(1e-6, vec.sum())).tolist()))))
        return out


def _iou_many(b: np.ndarray, bs: np.ndarray) -> np.ndarray:
    ix1, iy1 = np.maximum(b[0], bs[:, 0]), np.maximum(b[1], bs[:, 1])
    ix2, iy2 = np.minimum(b[2], bs[:, 2]), np.minimum(b[3], bs[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (b[2] - b[0]) * (b[3] - b[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])
    return inter / np.maximum(area - inter, 1e-6)


class BikeHelper:
    """Uses the COCO YOLOv8n in plateReader.pt to catch two-wheelers the main model misses."""

    COCO_TWO_WHEELERS = (1, 3)   # bicycle, motorcycle

    def __init__(self, path=config.PLATE_MODEL):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        names = {int(k): str(v) for k, v in self.model.names.items()}
        if names.get(3) != "motorcycle":
            raise ValueError("plateReader.pt is no longer the COCO model; bike helper disabled")

    def __call__(self, frame: np.ndarray, conf: float = config.BIKE_HELPER_CONF) -> list[Detection]:
        """Two-wheelers, with rider fusion: a weak bike box (>= 0.08) that touches a person
        box becomes one 'motorcycle + rider' box. Lone bikes need `conf`; lone people are ignored."""
        r = self.model.predict(frame, conf=0.08, classes=[0, *self.COCO_TWO_WHEELERS],
                               imgsz=640, verbose=False)[0]
        people, bikes = [], []
        for b in r.boxes:
            box, c = tuple(int(v) for v in b.xyxy[0].tolist()), float(b.conf)
            (people if int(b.cls) == 0 else bikes).append((box, c))
        out = []
        for box, c in bikes:
            rider = max(((p, pc) for p, pc in people if _touches(box, p) and pc >= 0.25),
                        key=lambda t: t[1], default=None)
            if rider:
                p = rider[0]
                box = (min(box[0], p[0]), min(box[1], p[1]), max(box[2], p[2]), max(box[3], p[3]))
                c = max(c, min(rider[1], 0.6))
            elif c < conf:
                continue
            out.append(Detection(box, "motorcycle", c))
        return out


def _touches(bike, person) -> bool:
    """Person overlaps the bike horizontally and sits on / above it (a rider)."""
    ox = min(bike[2], person[2]) - max(bike[0], person[0])
    return ox > 0.3 * min(bike[2] - bike[0], person[2] - person[0]) and \
        person[1] <= bike[1] + 0.5 * (bike[3] - bike[1]) and person[3] >= bike[1]


def merge_bikes(dets: list[Detection], bikes: list[Detection]) -> list[Detection]:
    from .tracker import iou
    for b in bikes:
        best, best_iou = None, 0.0
        for d in dets:
            o = iou(b.box, d.box)
            if o > best_iou:
                best, best_iou = d, o
        if best is None or best_iou < 0.3:
            dets.append(b)                       # missed by the main model
        elif best.class_name == "car" and best_iou > 0.5 and b.confidence > best.confidence:
            best.class_name = "motorcycle"       # main model mislabelled a bike as a car
            best.class_scores = None
    return dets
