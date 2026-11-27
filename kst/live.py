"""Background inference worker for the live view.

Three stages run in parallel, so none waits for another:
  capture thread (stream.LiveCapture)  -> newest decoded frame
  LiveRunner thread (this file)        -> detection + tracking + plates on the GPU, overlay
  Streamlit UI (app.py)                -> only displays the latest published Snapshot

The runner loops back-to-back with no sleeps: as soon as one frame is done it takes the
next, which keeps the GPU busy as long as frames keep arriving.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config
from .overlay import draw_detections
from .pipeline import Pipeline


@dataclass
class Snapshot:
    jpeg: bytes | None = None             # annotated frame, downscaled + JPEG-encoded
    frame_no: int = 0
    in_frame: int = 0
    zone_vehicles: int = 0
    read_in_zone: int = 0
    tracks: int = 0
    type_counts: Counter = field(default_factory=Counter)
    counts: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)  # reading-zone table rows (copies)
    infer_ms: float = 0.0
    fps: float = 0.0
    progress: tuple | None = None
    status: str = "Starting..."


def _encode(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    if w > config.DISPLAY_MAX_WIDTH:
        s = config.DISPLAY_MAX_WIDTH / w
        frame = cv2.resize(frame, (config.DISPLAY_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


class LiveRunner(threading.Thread):
    def __init__(self, source, pipeline: Pipeline, vehicle_conf: float, plate_conf: float):
        super().__init__(daemon=True, name="inference")
        self.source, self.pipeline = source, pipeline
        # The UI updates these while running (slider changes apply on the next frame).
        self.vehicle_conf, self.plate_conf = vehicle_conf, plate_conf
        self.done_reason: str | None = None
        self.error: str | None = None
        self.last_poll = time.time()          # UI heartbeat; see LIVE_IDLE_TIMEOUT_S
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._snap = Snapshot()

    # ---------------------------------------------------------------- UI side
    def snapshot(self) -> Snapshot:
        self.last_poll = time.time()
        with self._lock:
            return self._snap

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout)

    @property
    def finished(self) -> bool:
        return self.done_reason is not None

    # ---------------------------------------------------------------- worker side
    def run(self) -> None:
        try:
            self._loop()
        except Exception as e:                # surface it in the UI instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            self.done_reason = "Stopped by an error"

    def _loop(self) -> None:
        pl, src = self.pipeline, self.source
        fps = 0.0
        while not self._stop_evt.is_set():
            if time.time() - self.last_poll > config.LIVE_IDLE_TIMEOUT_S:
                self.done_reason = "Stopped: the page was closed"
                return
            frame = src.next_frame(0.5)
            if frame is None:
                if src.ended:
                    # short label; the detailed reason is src.error (shown once by the UI)
                    self.done_reason = "Video finished" if src.is_file else \
                        "Camera signal lost" if getattr(src, "connected", True) else \
                        "Could not connect to the camera"
                    return
                with self._lock:                  # still connected, no new frame yet
                    self._snap = Snapshot(**{**self._snap.__dict__, "status": src.status})
                continue
            t0 = time.perf_counter()
            dets = pl.process(frame, self.vehicle_conf, self.plate_conf)
            infer = time.perf_counter() - t0
            jpeg = _encode(draw_detections(frame, dets, pl.zone, pl.counts))
            total = time.perf_counter() - t0
            fps = 1 / total if fps == 0 else 0.9 * fps + 0.1 / total
            rows = pl.seen_table().to_dict("records")
            snap = Snapshot(
                jpeg=jpeg, frame_no=pl.frame_no, in_frame=len(dets), zone_vehicles=len(pl.seen),
                read_in_zone=sum(pl.counts.values()), tracks=pl.tracker.next_id - 1,
                type_counts=Counter(r["Type"] for r in pl.seen.values()),
                counts=dict(pl.counts), rows=rows, infer_ms=infer * 1000, fps=fps,
                progress=src.progress(), status="Running")
            with self._lock:
                self._snap = snap
        self.done_reason = self.done_reason or "Stopped by operator"
