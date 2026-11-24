"""Orchestrates the cascade for one camera stream:

  frame -> VehicleDetector (ONNX) + BikeHelper (COCO two-wheelers) -> tracker
        -> reading zone (READING_ZONE, bottom 60% of frame height):
             crossing = count + analyse (best.pt features/plate box -> Pillow clean ->
             RapidOCR plate, Pillow colour, box-geometry shape), re-analysed a few
             times while near the line
        -> per-track majority votes -> SQLite (only vehicles that entered the zone)
"""
from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, storage
from .analyzers import VehicleAnalyzer
from .color_shape import estimate_color, estimate_shape
from .detector import Detection, VehicleDetector, merge_bikes
from .imaging import bgr_to_pil, sharpness
from .tracker import IoUTracker


@dataclass
class TrackState:
    plates: Counter = field(default_factory=Counter)
    plate_best: dict = field(default_factory=dict)     # text -> best conf
    colors: Counter = field(default_factory=Counter)
    features: Counter = field(default_factory=Counter)
    shape: str | None = None
    best_sharp: float = -1.0
    analyses: int = 0
    last_cy: float | None = None
    in_zone: bool = False
    crossed: bool = False                  # has entered the reading zone
    entered_this_frame: bool = False
    plate_rel: tuple | None = None         # plate box relative to the vehicle box (0..1)
    plate_src: str | None = None

    def summary(self) -> dict:
        plate = self.plates.most_common(1)[0][0] if self.plates else None
        color = self.colors.most_common(1)[0][0] if self.colors else None
        total_c = sum(self.colors.values()) or 1
        need = max(1, self.analyses // 3)      # feature must recur to be reported
        feats = sorted(f for f, n in self.features.items() if n >= need)
        if "front_plate" in feats:
            feats = [f for f in feats if f != "no_front_plate_seen"]
        return {
            "plate_text": plate,
            "plate_conf": self.plate_best.get(plate),
            "color": color,
            "color_conf": (self.colors[color] / total_c) if color else None,
            "shape": self.shape,
            "features": feats,
            "in_zone": self.in_zone,
            "crossed": self.crossed,
            "entered_this_frame": self.entered_this_frame,
            "plate_rel": self.plate_rel,
            "plate_src": self.plate_src,
        }


class Pipeline:
    def __init__(self, detector: VehicleDetector, analyzer: VehicleAnalyzer, con,
                 camera_id: str = "KHI-CAM-01", bike_helper=None,
                 zone: tuple[float, float] = config.READING_ZONE):
        self.detector, self.analyzer, self.con = detector, analyzer, con
        self.bike_helper = bike_helper
        self.camera_id = camera_id
        self.zone = zone
        self.run_id = uuid.uuid4().hex[:12]
        self.tracker = IoUTracker()
        self.states: dict[int, TrackState] = defaultdict(TrackState)
        self.counts: Counter = Counter()          # line crossings per class
        self.seen: dict[int, dict] = {}           # unique tracks that entered the zone
        self.current_dets: list[Detection] = []
        self.frame_no = 0
        config.CROPS_DIR.mkdir(parents=True, exist_ok=True)

    def detect(self, frame: np.ndarray, vehicle_conf: float) -> list[Detection]:
        dets = self.detector(frame, vehicle_conf)
        if self.bike_helper is not None:
            dets = merge_bikes(dets, self.bike_helper(frame))
        return dets

    def process(self, frame: np.ndarray, vehicle_conf: float, feat_conf: float,
                plate_conf: float) -> list[Detection]:
        self.frame_no += 1
        H = frame.shape[0]
        top, bottom = self.zone[0] * H, self.zone[1] * H
        dets = self.tracker.update(self.detect(frame, vehicle_conf), self.frame_no)

        due = []
        for d in dets:
            st = self.states[d.track_id]
            x1, y1, x2, y2 = d.box
            cy = (y1 + y2) / 2
            # In the zone: centre inside it, or the box overlaps it by >= 1/3 of the zone.
            overlap = max(0.0, min(y2, bottom) - max(y1, top))
            in_zone = top <= cy <= bottom or overlap >= (bottom - top) / 3
            st.in_zone = in_zone
            # Also count vehicles that jumped over the zone between analysed frames.
            jumped = st.last_cy is not None and (
                (st.last_cy < top and cy > bottom) or (st.last_cy > bottom and cy < top))
            st.entered_this_frame = not st.crossed and (in_zone or jumped)
            if not st.crossed and (in_zone or jumped):
                st.crossed = True
                self.counts[d.class_name] += 1
            st.last_cy = cy
            big = min(x2 - x1, y2 - y1) >= config.MIN_CROP_SIDE
            if in_zone and big and st.analyses < config.MAX_ANALYSES_PER_TRACK:
                due.append((st.analyses, -(x2 - x1) * (y2 - y1), d))
        due.sort(key=lambda t: (t[0], t[1]))      # never-analysed and biggest first
        for _, _, d in due[:config.ANALYSES_PER_FRAME]:
            x1, y1, x2, y2 = d.box
            self._analyze(d, self.states[d.track_id], frame[y1:y2, x1:x2], feat_conf, plate_conf,
                          frame.shape)
        for d in dets:
            d.info = self.states[d.track_id].summary()
        self._update_seen(dets)
        self.current_dets = dets
        return dets

    def _update_seen(self, dets: list[Detection]) -> None:
        """Update rows only for tracks that have entered the reading zone."""
        clock = datetime.now().strftime("%H:%M:%S")
        for row in self.seen.values():
            row["In zone"] = False
        for d in dets:
            i = d.info
            if not i.get("crossed"):
                continue
            row = self.seen.get(d.track_id)
            if row is None:
                row = self.seen[d.track_id] = {
                    "ID": d.track_id, "First seen": clock, "In zone": False,
                }
            row.update({
                "Type": d.class_name,
                "Conf": round(max(d.confidence, row.get("Conf", 0.0)), 2),
                "Plate": i.get("plate_text") or "",
                "Color": i.get("color") or "",
                "Shape": i.get("shape") or "",
                "Features": ", ".join(f for f in i.get("features", []) if f != "no_front_plate_seen"),
                "Entered zone": bool(i.get("crossed")),
                "In zone": bool(i.get("in_zone")),
                "Frame": self.frame_no,
                "Last seen": clock,
            })

    def seen_table(self):
        import pandas as pd
        cols = ["ID", "Type", "Conf", "Plate", "Color", "Shape", "Features",
                "Entered zone", "In zone", "Frame", "First seen", "Last seen"]
        return pd.DataFrame(list(self.seen.values()), columns=cols).sort_values(
            "ID", ascending=False)

    def frame_table(self):
        import pandas as pd
        cols = ["ID", "Type", "Conf", "Plate", "Color", "Shape", "Features",
                "Entered zone"]
        rows = []
        for d in self.current_dets:
            info = d.info or {}
            if not (info.get("in_zone") or info.get("entered_this_frame")):
                continue
            rows.append({
                "ID": d.track_id,
                "Type": d.class_name,
                "Conf": round(d.confidence, 2),
                "Plate": info.get("plate_text") or "",
                "Color": info.get("color") or "",
                "Shape": info.get("shape") or "",
                "Features": ", ".join(
                    f for f in info.get("features", []) if f != "no_front_plate_seen"),
                "Entered zone": bool(info.get("crossed")),
            })
        return pd.DataFrame(rows, columns=cols).sort_values("ID", ascending=False)

    def process_photo(self, frame: np.ndarray, vehicle_conf: float, feat_conf: float,
                      plate_conf: float) -> list[Detection]:
        """Single image: no line, analyse every detected vehicle once."""
        dets = self.tracker.update(self.detect(frame, vehicle_conf), 1)
        for d in dets:
            x1, y1, x2, y2 = d.box
            if min(x2 - x1, y2 - y1) >= 8:
                self._analyze(d, self.states[d.track_id], frame[y1:y2, x1:x2], feat_conf, plate_conf)
            d.info = self.states[d.track_id].summary()
        return dets

    def analyze_whole_image(self, frame: np.ndarray, feat_conf: float, plate_conf: float) -> Detection:
        """Treat the full image as one vehicle (close-up photos the detector misses)."""
        h, w = frame.shape[:2]
        d = Detection((0, 0, w, h), "car", 0.0, track_id=self.tracker.next_id)
        self.tracker.next_id += 1
        self._analyze(d, self.states[d.track_id], frame, feat_conf, plate_conf)
        d.info = self.states[d.track_id].summary()
        return d

    def _analyze(self, d: Detection, st: TrackState, crop, feat_conf, plate_conf, frame_shape=None):
        info = self.analyzer.analyze(crop, feat_conf, plate_conf, d.class_name)
        if d.class_name == "motorcycle":   # windshield tint is meaningless on a bike
            info.features = [f for f in info.features if f != "tinted_windshield"]
        if info.plate_box is not None and frame_shape is not None:
            # drop plate boxes that sit on burned-in camera text
            px = d.box[0] + (info.plate_box[0] + info.plate_box[2]) / 2
            py = d.box[1] + (info.plate_box[1] + info.plate_box[3]) / 2
            H, W = frame_shape[:2]
            if any(mx1 * W <= px <= mx2 * W and my1 * H <= py <= my2 * H
                   for mx1, my1, mx2, my2 in config.OSD_MASKS):
                info.plate_box, info.plate_text = None, None
                info.features = [f for f in info.features if f != "front_plate"]
        if info.plate_box is not None:
            ch, cw = crop.shape[:2]
            b = info.plate_box
            st.plate_rel = (b[0] / cw, b[1] / ch, b[2] / cw, b[3] / ch)
            st.plate_src = info.plate_source
        st.analyses += 1
        for f in info.features:
            st.features[f] += 1
        if info.plate_text:
            st.plates[info.plate_text] += 1
            st.plate_best[info.plate_text] = max(st.plate_best.get(info.plate_text, 0), info.plate_conf)
        color, _ = estimate_color(bgr_to_pil(crop))
        st.colors[color] += 1
        st.shape = estimate_shape(d.class_name, d.box)

        crop_path = None
        sharp = sharpness(crop)
        if sharp > st.best_sharp:   # keep only the sharpest shot per track
            st.best_sharp = sharp
            path = config.CROPS_DIR / f"{self.run_id}_{d.track_id:04d}.jpg"
            cv2.imwrite(str(path), crop)
            crop_path = str(path)

        summary = st.summary()
        summary.pop("crossed", None)
        rec = {"track_id": d.track_id, "vehicle_type": d.class_name,
               "det_conf": d.confidence, "crop_path": crop_path, **summary}
        storage.upsert(self.con, self.run_id, self.camera_id, rec)


PALETTE = {"car": (60, 200, 60), "motorcycle": (255, 170, 0), "bus": (0, 140, 255),
           "truck": (0, 80, 230), "van": (200, 90, 220), "auto_rickshaw": (0, 220, 220),
           "qingqi": (180, 200, 0)}


PLATE_COLOR = (255, 0, 255)


def draw(frame: np.ndarray, dets: list[Detection], zone: tuple[float, float] | None = None,
         counts: Counter | None = None) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    if zone is not None:
        zt, zb = int(zone[0] * H), int(zone[1] * H)
        band = out.copy()
        cv2.rectangle(band, (0, zt), (W, zb), (0, 200, 255), -1)
        out = cv2.addWeighted(band, 0.15, out, 0.85, 0)
        for y in (zt, zb):
            cv2.line(out, (0, y), (W, y), (0, 200, 255), 2)
        cv2.putText(out, "READING ZONE", (8, zt + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
    for d in dets:
        x1, y1, x2, y2 = d.box
        col = PALETTE.get(d.class_name, (255, 255, 255))
        info = d.info or {}
        highlighted = info.get("in_zone") or info.get("entered_this_frame")
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if highlighted else 1)
        rel = info.get("plate_rel")
        clipped = y1 <= 1 or x1 <= 1 or y2 >= H - 2 or x2 >= W - 2   # relative pos. unreliable
        if rel and not clipped:   # plate box follows the vehicle: stored relative to the vehicle box
            bw, bh = x2 - x1, y2 - y1
            p1 = (int(x1 + rel[0] * bw), int(y1 + rel[1] * bh))
            p2 = (int(x1 + rel[2] * bw), int(y1 + rel[3] * bh))
            cv2.rectangle(out, p1, p2, PLATE_COLOR, 2)
            label = info.get("plate_text") or "plate"
            cv2.putText(out, label, (p1[0], p2[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        PLATE_COLOR, 1, cv2.LINE_AA)
        lines = [f"#{d.track_id} {d.class_name} {d.confidence:.0%}"]
        if info.get("plate_text"):
            lines.append(f"PLATE {info['plate_text']}")
        tail = info.get("color") or ""
        feats = [f for f in info.get("features", []) if f not in ("front_plate", "no_front_plate_seen")]
        if feats:
            tail += " | " + ",".join(f.replace("_", " ") for f in feats)
        if tail.strip():
            lines.append(tail.strip())
        y = max(14, y1 - 6 - 16 * (len(lines) - 1))
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y - th - 4), (x1 + tw + 4, y + 3), col, -1)
            cv2.putText(out, line, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            y += 16
    if counts is not None:
        text = f"Read in zone: {sum(counts.values())}  " + "  ".join(f"{k}:{v}" for k, v in counts.items())
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
        cv2.putText(out, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out
