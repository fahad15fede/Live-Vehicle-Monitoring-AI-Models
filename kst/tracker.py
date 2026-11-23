"""Lightweight tracker (stand-in for BoT-SORT): IoU matching with a centre-distance
fallback, so tracks survive when only a few frames per second are processed."""
from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def match_score(det_box, track_box) -> float:
    o = iou(det_box, track_box)
    if o > 0:
        return 0.2 + o                     # any overlap beats distance-only matches
    (dx, dy), (tx, ty) = _center(det_box), _center(track_box)
    dist = ((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5
    limit = 0.8 * max(track_box[2] - track_box[0], track_box[3] - track_box[1])
    return 0.2 * (1 - dist / limit) if dist < limit else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple
    first_frame: int
    last_frame: int = 0
    hits: int = 0
    class_votes: dict = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        return max(self.class_votes, key=self.class_votes.get)


class IoUTracker:
    def __init__(self, max_age: int = 8):
        self.max_age = max_age             # in processed frames
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, dets: list[Detection], frame_no: int) -> list[Detection]:
        live = {k: t for k, t in self.tracks.items() if frame_no - t.last_frame <= self.max_age}
        pairs = sorted(((match_score(d.box, t.box), di, tid)
                        for di, d in enumerate(dets) for tid, t in live.items()), reverse=True)
        used_d, used_t = set(), set()
        for score, di, tid in pairs:
            if score <= 0.02:
                break
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            dets[di].track_id = tid
        for di, d in enumerate(dets):
            if di not in used_d:
                d.track_id = self.next_id
                live[self.next_id] = Track(self.next_id, d.box, first_frame=frame_no)
                self.next_id += 1
            t = live[d.track_id]
            t.box, t.last_frame = d.box, frame_no
            t.hits += 1
            # Soft voting: add the full class evidence, weighted by detection confidence,
            # so a rickshaw that is "car 0.40 / rickshaw 0.35" every frame still accumulates.
            votes = d.class_scores or {d.class_name: 1.0}
            for name, share in votes.items():
                t.class_votes[name] = t.class_votes.get(name, 0.0) + share * d.confidence
            d.class_name = t.class_name    # smooth class flicker across frames
        self.tracks = live
        return dets
