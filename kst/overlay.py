"""Draws detections, plate boxes, the reading zone and counts onto a frame."""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .vehicle_detection import Detection

CLASS_COLORS = {"car": (60, 200, 60), "motorcycle": (255, 170, 0), "bus": (0, 140, 255),
                "truck": (0, 80, 230), "van": (200, 90, 220), "auto_rickshaw": (0, 220, 220),
                "qingqi": (180, 200, 0)}
PLATE_BOX_COLOR = (255, 0, 255)
ZONE_COLOR = (0, 200, 255)


def draw_detections(frame: np.ndarray, dets: list[Detection], zone: tuple[float, float] | None = None,
                    counts: Counter | None = None) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    if zone is not None:
        zt, zb = int(zone[0] * H), int(zone[1] * H)
        band = out.copy()
        cv2.rectangle(band, (0, zt), (W, zb), ZONE_COLOR, -1)
        out = cv2.addWeighted(band, 0.15, out, 0.85, 0)
        for y in (zt, zb):
            cv2.line(out, (0, y), (W, y), ZONE_COLOR, 2)
        cv2.putText(out, "READING ZONE", (8, zt + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ZONE_COLOR, 1, cv2.LINE_AA)
    for d in dets:
        x1, y1, x2, y2 = d.box
        col = CLASS_COLORS.get(d.class_name, (255, 255, 255))
        info = d.info or {}
        highlighted = info.get("in_zone") or info.get("entered_this_frame")
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if highlighted else 1)
        rel = info.get("plate_rel")
        clipped = y1 <= 1 or x1 <= 1 or y2 >= H - 2 or x2 >= W - 2   # relative pos. unreliable
        if rel and not clipped:   # plate box follows the vehicle: stored relative to the vehicle box
            bw, bh = x2 - x1, y2 - y1
            p1 = (int(x1 + rel[0] * bw), int(y1 + rel[1] * bh))
            p2 = (int(x1 + rel[2] * bw), int(y1 + rel[3] * bh))
            cv2.rectangle(out, p1, p2, PLATE_BOX_COLOR, 2)
            label = info.get("plate_text") or "plate"
            cv2.putText(out, label, (p1[0], p2[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        PLATE_BOX_COLOR, 1, cv2.LINE_AA)
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
