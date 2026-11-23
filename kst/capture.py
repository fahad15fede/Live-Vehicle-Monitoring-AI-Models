"""Live capture for RTSP / IP cameras and webcams.

Analysis takes 1.5-4.5 s per frame on CPU. Reading a live stream directly from
that loop lets the decoder buffer fill and the picture drifts further and
further behind real time (then the camera usually drops the connection). A
background thread therefore drains the stream continuously and keeps only the
newest frame; the UI analyses whatever is newest when it is ready.

RTSP is opened through FFmpeg over TCP (no smeared/grey frames from UDP packet
loss) with open/read timeouts, falls back to UDP for cameras that refuse TCP,
and reconnects automatically if the signal drops.
"""
from __future__ import annotations

import os
import re
import threading
import time

import cv2
import numpy as np

OPEN_TIMEOUT_MS = 10_000
READ_TIMEOUT_MS = 10_000
RECONNECT_ATTEMPTS = 5          # consecutive failed reconnects before giving up
RECONNECT_BACKOFF_S = (1, 2, 3, 5, 8)

_env_lock = threading.Lock()    # OPENCV_FFMPEG_CAPTURE_OPTIONS is process-global


def mask_url(url: str) -> str:
    """rtsp://user:secret@host/... -> rtsp://user:***@host/... (safe to display)."""
    return re.sub(r"(://[^:/@\s]+:)[^@/\s]+@", r"\1***@", url)


def _open_network(url: str):
    """-> (capture, transport, first_frame) or (None, "", None)."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
    transports = ["tcp", "udp"] if url.lower().startswith(("rtsp://", "rtsps://")) else [None]
    for transport in transports:
        with _env_lock:
            old = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            if transport:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
            finally:
                if old is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old
        if cap.isOpened():
            ok, frame = cap.read()        # some servers accept the session but send nothing
            if ok and frame is not None:
                return cap, (transport or "ffmpeg"), frame
        cap.release()
    return None, "", None


def _open_webcam(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)     # fall back to the default backend
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, "webcam", frame
    cap.release()
    return None, "", None


class LiveCapture:
    """Thread-backed capture that always serves the newest frame."""

    def __init__(self, source: str | int):
        self.source = source
        self.label = mask_url(source) if isinstance(source, str) else f"webcam {source}"
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._seq = 0                 # increments with every decoded frame
        self._served = 0              # seq of the last frame handed out
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.transport = ""
        self.status = "connecting"    # connecting | live | reconnecting | lost | stopped
        self.reconnects = 0
        self.fps = 0.0

    # ---- lifecycle
    def _connect(self) -> bool:
        if isinstance(self.source, int):
            cap, transport, frame = _open_webcam(self.source)
        else:
            cap, transport, frame = _open_network(self.source)
        if cap is None:
            return False
        self._cap, self.transport = cap, transport
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (0 < self.fps < 121):
            self.fps = 25.0
        self._publish(frame)
        return True

    def open(self) -> bool:
        """Connect synchronously (so the UI can report failure), then start the reader."""
        if not self._connect():
            self.status = "lost"
            return False
        self.status = "live"
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.label}",
                                        daemon=True)
        self._thread.start()
        return True

    def isOpened(self) -> bool:      # cv2.VideoCapture-compatible
        return self.status in ("live", "reconnecting", "connecting")

    def release(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)       # a blocked read ends at READ_TIMEOUT_MS; thread releases cap
        if t is None or not t.is_alive():
            self._release_cap()
        self.status = "stopped"

    def _release_cap(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0.0                # live: no length
        cap = self._cap
        return cap.get(prop) if cap is not None else 0.0

    # ---- reader thread
    def _publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                cap = self._cap
                ok, frame = (cap.read() if cap is not None else (False, None))
                if ok and frame is not None:
                    self._publish(frame)
                    continue
                if self._stop.is_set():
                    break
                if not self._reconnect():
                    self.status = "lost"
                    with self._cond:
                        self._cond.notify_all()
                    break
        finally:
            self._release_cap()

    def _reconnect(self) -> bool:
        self.status = "reconnecting"
        self._release_cap()
        for attempt in range(RECONNECT_ATTEMPTS):
            if self._stop.wait(RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S) - 1)]):
                return False
            if self._connect():
                self.reconnects += 1
                self.status = "live"
                return True
        return False

    # ---- consumer side
    def read_latest(self, timeout: float = 2.0) -> np.ndarray | None:
        """Newest frame not handed out before. Waits up to `timeout` for one.
        Returns None if no new frame arrived in time (check `status`)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._seq == self._served and self.status in ("live", "reconnecting") \
                    and not self._stop.is_set():
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
            if self._seq == self._served or self._frame is None:
                return None
            self._served = self._seq
            return self._frame

    def read(self):                    # cv2.VideoCapture-compatible
        frame = self.read_latest()
        return frame is not None, frame
