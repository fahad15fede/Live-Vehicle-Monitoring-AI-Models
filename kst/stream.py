"""Video sources with one interface: next_frame() / ended / status / progress() / release().

  * LiveCapture - RTSP / IP camera (URL) or webcam (index). A background thread pulls frames
                  continuously and keeps only the newest, so inference never lags behind real
                  time and the camera never drops us for not reading. FFmpeg over TCP with
                  timeouts, hardware (GPU) video decoding when available, auto-reconnect.
  * FileCapture - video file, read in order (no frames dropped except the chosen skip).
"""
from __future__ import annotations

import os
import socket
import threading
from urllib.parse import urlsplit

import cv2

from . import config

# FFmpeg defaults to UDP for RTSP, which is often blocked / lossy (grey or no frames).
# Read by OpenCV at open time; options unknown to a protocol (e.g. local files) are ignored.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

OPEN_TIMEOUT_MS = 30_000
READ_TIMEOUT_MS = 5_000


def _ffmpeg_open(url: str, timeouts: bool, hw: bool = False) -> cv2.VideoCapture:
    """FFmpeg capture; `hw` asks for hardware video decoding (see config.HW_VIDEO_DECODE)."""
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
              cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS] if timeouts else []
    params += [cv2.CAP_PROP_HW_ACCELERATION,
               cv2.VIDEO_ACCELERATION_ANY if hw else cv2.VIDEO_ACCELERATION_NONE]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


def open_source(src: str | int, hw: bool = False) -> cv2.VideoCapture:
    if isinstance(src, int):                      # webcam
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    else:
        cap = _ffmpeg_open(src, timeouts=True, hw=hw)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def diagnose(src: str | int) -> str:
    """Explain why a source failed to open: unreachable host/port vs. stream-level failure."""
    if isinstance(src, int):
        return (f"Could not open webcam {src}. Try another index, or close other apps "
                "using the camera.")
    u = urlsplit(src)
    if u.scheme.lower() not in ("rtsp", "rtsps", "http", "https", "rtmp"):
        return f"'{src}' is not a stream URL. It should look like rtsp://ip:port/path."
    host, port = u.hostname, u.port or {"rtsp": 554, "rtsps": 322, "http": 80,
                                       "https": 443, "rtmp": 1935}[u.scheme.lower()]
    try:
        socket.create_connection((host, port), timeout=5).close()
    except OSError as e:
        return (f"Nothing is reachable at {host}:{port} ({e.__class__.__name__}). The stream "
                "server is not running, or the IP / port is wrong, or a firewall blocks it. "
                "If the server runs on this PC use 127.0.0.1; RTSP servers usually use port 8554 "
                "or 554. Test the URL in VLC (Media > Open Network Stream).")
    return (f"{host}:{port} is reachable, but the stream did not open. Check the path "
            f"('{u.path or '/'}'), that something is actually publishing to it, and the "
            "username / password (URL-encode special characters, e.g. @ -> %40).")


def hw_decoding(cap: cv2.VideoCapture) -> bool:
    try:
        return int(cap.get(cv2.CAP_PROP_HW_ACCELERATION)) not in (0, -1)
    except Exception:
        return False


class LiveCapture:
    """Newest-frame reader for RTSP URLs and webcams."""

    is_file = False

    def __init__(self, src: str | int, max_reconnects: int = 5):
        """Returns immediately: connecting happens on the background thread, so the UI never
        blocks for OPEN_TIMEOUT_MS. Watch `status` / `ended` / `error`."""
        self.src = src
        self.max_reconnects = max_reconnects
        self.lost = False                 # True once connecting / reconnecting has given up
        self.connecting = True
        self.connected = False            # at least one frame received
        self.reconnecting = False
        self.error: str | None = None
        self.fps, self.hw = 0.0, False
        self._use_hw = config.HW_VIDEO_DECODE and not isinstance(src, int)
        self._cap = None
        self._frame = None
        self._new = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="capture")
        self._thread.start()

    def isOpened(self) -> bool:
        return not self.lost

    @property
    def ended(self) -> bool:
        return self.lost

    @property
    def status(self) -> str:
        if self.connecting:
            return f"Connecting to camera (up to {OPEN_TIMEOUT_MS // 1000} s)..."
        return "Reconnecting to camera..." if self.reconnecting else "Waiting for camera frames..."

    def progress(self):
        return None

    def next_frame(self, timeout: float = 0.5):
        """Newest frame not returned before, waiting up to `timeout` s; None if none arrived."""
        with self._new:
            if self._frame is None and not self.lost and not self._stop.is_set():
                self._new.wait(timeout)
            frame, self._frame = self._frame, None
        return frame

    def release(self) -> None:
        self._stop.set()
        with self._new:
            self._new.notify_all()
        # the thread releases the capture on exit (it may be mid-connect: don't wait long)
        self._thread.join(timeout=1.0)

    def _give_up(self, msg: str) -> None:
        self.error, self.lost = msg, True
        with self._new:
            self._new.notify_all()

    def _run(self) -> None:
        try:
            self._cap = open_source(self.src, self._use_hw)
            self.connecting = False
            if self._stop.is_set():
                return
            if not self._cap.isOpened():
                self._give_up(diagnose(self.src))
                return
            self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.hw = hw_decoding(self._cap)
            self._loop()
        except Exception as e:
            self._give_up(f"Camera error: {type(e).__name__}: {e}")
        finally:
            self.connecting = False
            if self._cap is not None:
                self._cap.release()

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                failures = 0
                self.reconnecting = False
                self.connected = True
                with self._new:
                    self._frame = frame
                    self._new.notify_all()
                continue
            if self.hw:                   # HW decoder failing: switch to CPU decoding for good
                self._use_hw = False
                failures = 0
            failures += 1                 # read failed: reconnect with backoff
            if failures > self.max_reconnects:
                self._give_up("Camera signal lost (reconnecting failed).")
                return
            self.reconnecting = True
            self._cap.release()
            if self._stop.wait(min(2 ** failures, 10)):
                return
            self._cap = open_source(self.src, self._use_hw)
            self.hw = hw_decoding(self._cap)


class FileCapture:
    """Sequential reader for video files: analyse one frame, drop `skip`, repeat."""

    is_file = True
    status = "Reading video..."

    def __init__(self, path: str, skip: int = 0):
        self.skip = skip
        self._cap = _ffmpeg_open(path, timeouts=False, hw=config.HW_VIDEO_DECODE)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.hw = hw_decoding(self._cap)
        self.ended = False
        self._first = True

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def next_frame(self, timeout: float = 0.0):
        if not self._first:
            for _ in range(self.skip):
                if not self._cap.grab():
                    self.ended = True
                    return None
        self._first = False
        ok, frame = self._cap.read()
        if not ok:
            self.ended = True
            return None
        return frame

    def progress(self):
        if not self.total:
            return None
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)), self.total

    def release(self) -> None:
        self._cap.release()


# Backwards-compatible name.
RtspStream = LiveCapture
