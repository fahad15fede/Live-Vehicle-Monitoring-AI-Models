from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "vehicle_det_fp16.onnx"
OUTPUTS_ARCHIVE = ROOT / "member1_outputs.zip"
DATABASE_PATH = ROOT / "data" / "traffic_records.sqlite3"
DEMO_VIDEO_MEMBER = (
    "content/member1_detection_tracking/outputs/annotated_videos/"
    "tracked_traffic_scene.mp4"
)
DEMO_CSV_MEMBER = (
    "content/member1_detection_tracking/outputs/tracking_logs/"
    "vehicle_trajectory_logs.csv"
)
CLASS_NAMES = [
    "auto_rickshaw",
    "bus",
    "car",
    "motorcycle",
    "chinchi",
    "truck",
    "van",
]
INPUT_SIZE = 640


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database() -> Any:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            vehicle_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            plate_text TEXT,
            plate_category TEXT,
            plate_confidence REAL,
            make TEXT,
            model TEXT,
            color TEXT,
            detected_attributes TEXT NOT NULL DEFAULT '[]',
            forensic_caption TEXT,
            watchlist_flagged INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            UNIQUE(run_id, track_id)
        )
        """
    )
    connection.commit()
    return connection


@st.cache_resource(show_spinner="Loading the vehicle detector...")
def load_detector() -> ort.InferenceSession:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Vehicle detector not found: {MODEL_PATH}")
    return ort.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )


def letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height))
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def detect_vehicles(
    session: ort.InferenceSession,
    frame: np.ndarray,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    padded, scale, pad_x, pad_y = letterbox(frame)
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor[None], dtype=np.float32) / 255.0
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected detector output shape: {output.shape}")
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    if predictions.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(f"Unexpected detector output shape: {output.shape}")

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    candidates: list[dict[str, Any]] = []

    for class_id in np.unique(class_ids[scores >= confidence_threshold]):
        indices = np.flatnonzero(
            (class_ids == class_id) & (scores >= confidence_threshold)
        )
        xywh = boxes[indices]
        left = (xywh[:, 0] - xywh[:, 2] / 2 - pad_x) / scale
        top = (xywh[:, 1] - xywh[:, 3] / 2 - pad_y) / scale
        right = (xywh[:, 0] + xywh[:, 2] / 2 - pad_x) / scale
        bottom = (xywh[:, 1] + xywh[:, 3] / 2 - pad_y) / scale
        image_height, image_width = frame.shape[:2]
        boxes_for_nms = [
            [
                max(0, int(x1)),
                max(0, int(y1)),
                max(1, int(x2 - x1)),
                max(1, int(y2 - y1)),
            ]
            for x1, y1, x2, y2 in zip(left, top, right, bottom)
        ]
        kept = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores[indices].astype(float).tolist(),
            confidence_threshold,
            0.45,
        )
        for kept_index in np.asarray(kept).reshape(-1):
            candidate_index = int(kept_index)
            x1, y1, x2, y2 = boxes_for_nms[candidate_index]
            candidates.append(
                {
                    "box": (
                        x1,
                        y1,
                        min(image_width, x1 + boxes_for_nms[candidate_index][2]),
                        min(image_height, y1 + boxes_for_nms[candidate_index][3]),
                    ),
                    "class_id": int(class_id),
                    "class_name": CLASS_NAMES[int(class_id)],
                    "confidence": float(scores[indices[candidate_index]]),
                }
            )
    return candidates


def intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def assign_track_ids(
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    frame_number: int,
    next_track_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    remaining = [
        track for track in tracks if frame_number - track["last_frame"] <= 45
    ]
    matches: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for track_index, track in enumerate(remaining):
            if track["class_name"] != detection["class_name"]:
                continue
            overlap = intersection_over_union(detection["box"], track["box"])
            if overlap >= 0.2:
                matches.append((overlap, detection_index, track_index))
    matches.sort(reverse=True)

    matched_detections: set[int] = set()
    matched_tracks: set[int] = set()
    updated_tracks: list[dict[str, Any]] = []
    for _, detection_index, track_index in matches:
        if detection_index in matched_detections or track_index in matched_tracks:
            continue
        detection = detections[detection_index]
        track = remaining[track_index]
        detection["track_id"] = track["track_id"]
        updated_tracks.append(
            {
                "track_id": track["track_id"],
                "box": detection["box"],
                "class_name": detection["class_name"],
                "last_frame": frame_number,
            }
        )
        matched_detections.add(detection_index)
        matched_tracks.add(track_index)

    for detection_index, detection in enumerate(detections):
        if detection_index in matched_detections:
            continue
        detection["track_id"] = next_track_id
        updated_tracks.append(
            {
                "track_id": next_track_id,
                "box": detection["box"],
                "class_name": detection["class_name"],
                "last_frame": frame_number,
            }
        )
        next_track_id += 1

    for track_index, track in enumerate(remaining):
        if track_index not in matched_tracks:
            updated_tracks.append(track)
    return detections, updated_tracks, next_track_id


def store_live_detections(
    detections: list[dict[str, Any]], run_id: str, camera_id: str, source: str
) -> None:
    now = utc_now()
    with connect_database() as connection:
        for detection in detections:
            connection.execute(
                """
                INSERT INTO vehicle_records (
                    run_id, camera_id, timestamp, track_id, vehicle_type,
                    confidence, source, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, track_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    confidence = MAX(vehicle_records.confidence, excluded.confidence),
                    last_seen = excluded.last_seen
                """,
                (
                    run_id,
                    camera_id,
                    now,
                    detection["track_id"],
                    detection["class_name"],
                    detection["confidence"],
                    source,
                    now,
                    now,
                ),
            )


def create_capture_temp_file(suffix: str, content: bytes) -> str:
    descriptor, path = tempfile.mkstemp(prefix="karachi_traffic_", suffix=suffix)
    with os.fdopen(descriptor, "wb") as file:
        file.write(content)
    return path


def remove_capture_temp_file() -> None:
    path = st.session_state.get("capture_temp_path")
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        st.session_state.capture_temp_path = None


def stop_stream() -> None:
    capture = st.session_state.get("capture")
    if capture is not None:
        capture.release()
    st.session_state.capture = None
    st.session_state.stream_running = False
    remove_capture_temp_file()


def open_capture(
    source_type: str,
    webcam_index: int,
    rtsp_url: str,
    uploaded_file: Any,
) -> tuple[cv2.VideoCapture, str, str]:
    if source_type == "Webcam":
        return cv2.VideoCapture(webcam_index), f"CAM-{webcam_index}", "webcam"
    if source_type == "RTSP stream":
        if not rtsp_url.strip():
            raise ValueError("Enter an RTSP URL before starting the stream.")
        return cv2.VideoCapture(rtsp_url.strip()), "KHI-RTSP-01", "rtsp"
    if source_type == "Upload video":
        if uploaded_file is None:
            raise ValueError("Choose a video file before starting playback.")
        path = create_capture_temp_file(
            Path(uploaded_file.name).suffix or ".mp4", uploaded_file.getvalue()
        )
        st.session_state.capture_temp_path = path
        return cv2.VideoCapture(path), "KHI-UPLOAD-01", "upload"
    if source_type == "Included demo video":
        if not OUTPUTS_ARCHIVE.is_file():
            raise FileNotFoundError(f"Demo archive not found: {OUTPUTS_ARCHIVE}")
        with zipfile.ZipFile(OUTPUTS_ARCHIVE) as archive:
            video_bytes = archive.read(DEMO_VIDEO_MEMBER)
        path = create_capture_temp_file(".mp4", video_bytes)
        st.session_state.capture_temp_path = path
        return cv2.VideoCapture(path), "KHI-DEMO-01", "demo"
    raise ValueError(f"Unsupported video source: {source_type}")


def draw_detections(
    frame: np.ndarray, detections: list[dict[str, Any]]
) -> np.ndarray:
    rendered = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = (42, 212, 153)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label = (
            f"#{detection['track_id']} {detection['class_name']} "
            f"{detection['confidence']:.0%}"
        )
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(0, y1 - text_height - 10)
        cv2.rectangle(
            rendered,
            (x1, label_top),
            (x1 + text_width + 10, y1),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (x1 + 5, max(text_height, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (9, 18, 19),
            2,
            cv2.LINE_AA,
        )
    return rendered


def get_records() -> pd.DataFrame:
    with connect_database() as connection:
        return pd.read_sql_query(
            """
            SELECT run_id, camera_id, timestamp, track_id, vehicle_type, confidence,
                   plate_text, plate_category, plate_confidence, make, model,
                   color, detected_attributes, forensic_caption,
                   watchlist_flagged, source, first_seen, last_seen
            FROM vehicle_records
            ORDER BY last_seen DESC
            """,
            connection,
        )


def import_demo_records() -> int:
    if not OUTPUTS_ARCHIVE.is_file():
        raise FileNotFoundError(f"Demo archive not found: {OUTPUTS_ARCHIVE}")
    with zipfile.ZipFile(OUTPUTS_ARCHIVE) as archive:
        frame = pd.read_csv(archive.open(DEMO_CSV_MEMBER))
    required = {"track_id", "class_name", "confidence"}
    if not required.issubset(frame.columns):
        raise ValueError("The included trajectory CSV is missing required columns.")

    run_id = "included-trajectory-demo"
    now = utc_now()
    summaries = (
        frame.sort_values("confidence", ascending=False)
        .drop_duplicates("track_id")
        .to_dict("records")
    )
    with connect_database() as connection:
        for record in summaries:
            connection.execute(
                """
                INSERT INTO vehicle_records (
                    run_id, camera_id, timestamp, track_id, vehicle_type,
                    confidence, source, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, track_id) DO UPDATE SET
                    timestamp = excluded.timestamp,
                    vehicle_type = excluded.vehicle_type,
                    confidence = excluded.confidence,
                    last_seen = excluded.last_seen
                """,
                (
                    run_id,
                    "KHI-DEMO-01",
                    now,
                    int(record["track_id"]),
                    str(record["class_name"]),
                    float(record["confidence"]),
                    "included trajectory CSV",
                    now,
                    now,
                ),
            )
    return len(summaries)


def render_live_monitor() -> None:
    if not st.session_state.get("stream_running"):
        st.info("Start a camera or video source to see live detections.")
        return

    @st.fragment(run_every=0.15)
    def live_frame() -> None:
        capture = st.session_state.get("capture")
        if capture is None or not capture.isOpened():
            st.error("The video source is no longer available. Stop and reconnect.")
            return
        ok, frame = capture.read()
        if not ok:
            if st.session_state.get("source_kind") in {"upload", "demo"}:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                st.error("No frame received. Check the camera or stream connection.")
                return

        try:
            session = load_detector()
            detections = detect_vehicles(
                session, frame, st.session_state.confidence_threshold
            )
        except (RuntimeError, ValueError, cv2.error) as error:
            st.error(f"Vehicle detection failed: {error}")
            return

        frame_number = st.session_state.get("frame_number", 0) + 1
        detections, tracks, next_track_id = assign_track_ids(
            detections,
            st.session_state.get("tracks", []),
            frame_number,
            st.session_state.get("next_track_id", 1),
        )
        st.session_state.frame_number = frame_number
        st.session_state.tracks = tracks
        st.session_state.next_track_id = next_track_id

        store_live_detections(
            detections,
            st.session_state.run_id,
            st.session_state.camera_id,
            st.session_state.source_kind,
        )
        annotated = draw_detections(frame, detections)
        video_col, stats_col = st.columns([3.2, 1])
        with video_col:
            st.image(
                annotated,
                channels="BGR",
                use_container_width=True,
                caption=(
                    f"{st.session_state.camera_id} · "
                    f"{datetime.now().strftime('%H:%M:%S')}"
                ),
            )
        with stats_col:
            st.metric("Vehicles in frame", len(detections))
            st.metric(
                "Tracks this run",
                max(0, st.session_state.next_track_id - 1),
            )
            st.caption(
                "Vehicle detector: YOLO11m ONNX · local CPU inference"
            )
            st.caption(
                "Track IDs use frame-to-frame box overlap; "
                "they reset when a new run starts."
            )
            if detections:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Track": item["track_id"],
                                "Class": item["class_name"],
                                "Confidence": f"{item['confidence']:.0%}",
                            }
                            for item in detections
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

    live_frame()


def initialize_state() -> None:
    defaults = {
        "capture": None,
        "capture_temp_path": None,
        "stream_running": False,
        "source_kind": None,
        "camera_id": None,
        "run_id": None,
        "frame_number": 0,
        "next_track_id": 1,
        "tracks": [],
        "confidence_threshold": 0.35,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    st.set_page_config(
        page_title="Karachi Traffic Intelligence",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
        :root { --ink: #e8eee9; --muted: #8d9d97; --line: #263630; --panel: #111a17; --green: #2ad499; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: radial-gradient(ellipse at 75% -15%, #16352c 0, #0b100f 43%); color: var(--ink); }
        [data-testid="stSidebar"] { background: #0c1311; border-right: 1px solid var(--line); }
        [data-testid="stMetric"] { background: #111a17; border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
        [data-testid="stMetricLabel"] { color: #9aaba4; }
        [data-testid="stMetricValue"] { color: #f0f7f2; font-family: 'DM Mono', monospace; }
        .hero { border: 1px solid #29453a; border-radius: 16px; padding: 23px 26px; margin: 3px 0 22px; background: linear-gradient(110deg, #13251e, #111a17 65%, #15231d); }
        .hero-kicker { color: var(--green); font: 500 11px 'DM Mono', monospace; letter-spacing: .16em; text-transform: uppercase; }
        .hero h1 { margin: 7px 0 4px; font-size: 30px; color: #f0f5f1; }
        .hero p { margin: 0; color: #9aaba4; }
        .mono { font-family: 'DM Mono', monospace; color: #b7c8bf; }
        div.stButton > button { border-radius: 8px; }
        div.stButton > button[kind="primary"] { background: #20a879; border-color: #20a879; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🚦 KHI / TRAFFIC OPS")
        st.caption("LOCAL SURVEILLANCE CONSOLE")
        st.divider()
        st.markdown("**Detector**")
        if MODEL_PATH.is_file():
            st.success("YOLO11m ONNX · ready")
        else:
            st.error("vehicle_det_fp16.onnx not found")
        st.caption("Live vehicle detection runs locally on the Streamlit host.")
        st.markdown("**Plate reader**")
        st.caption(
            "plateReader.pt is present but is not loaded: its inference "
            "interface has not been verified. Plate fields stay blank."
        )
        if st.session_state.stream_running:
            st.divider()
            st.markdown("🟢 **Stream running**")
            if st.button("■  Stop stream", use_container_width=True):
                stop_stream()
                st.rerun()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Karachi · City traffic intelligence</div>
          <h1>Traffic surveillance</h1>
          <p>Live vehicle detection, camera operations, and searchable vehicle records.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    records = get_records()
    distinct_tracks = (
        int(
            records[["run_id", "track_id"]]
            .drop_duplicates()
            .shape[0]
        )
        if not records.empty
        else 0
    )
    active_tracks = 0
    if st.session_state.stream_running:
        active_tracks = max(0, st.session_state.next_track_id - 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracked vehicles", f"{distinct_tracks:,}")
    col2.metric("Current run", f"{active_tracks:,}")
    col3.metric("Records stored", f"{len(records):,}")
    col4.metric(
        "Detector",
        "READY" if MODEL_PATH.is_file() else "MISSING",
    )

    dashboard_tab, live_tab, records_tab, models_tab = st.tabs(
        ["Overview", "Live monitor", "Vehicle records", "Models & data"]
    )

    with dashboard_tab:
        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("Operations overview")
            st.markdown(
                "Connect a local camera, RTSP feed, uploaded clip, or the "
                "included annotated demo. Detections are saved to a local "
                "SQLite database."
            )
            if st.session_state.stream_running:
                st.success(
                    f"Receiving frames from {st.session_state.camera_id}."
                )
            else:
                st.info("No source playing. Open Live monitor to connect.")
        with right:
            st.subheader("Detection mix")
            if records.empty:
                st.caption("Vehicle classes appear here after the first detections.")
            else:
                mix = (
                    records.groupby("vehicle_type")
                    .size()
                    .rename("Records")
                    .sort_values(ascending=False)
                )
                st.bar_chart(mix, color="#2ad499", height=210)
        st.caption(
            "Plate OCR, make/model, color, attributes, and watchlist matches "
            "are shown only when a configured model provides them."
        )

    with live_tab:
        st.subheader("Live camera")
        st.caption(
            "Webcam devices are opened on the computer running Streamlit. "
            "RTSP credentials stay in this local session."
        )
        if st.session_state.stream_running:
            st.success(
                f"PLAYING · {st.session_state.camera_id} · "
                f"{st.session_state.source_kind.upper()}"
            )
        source_col, setting_col = st.columns([1, 1.4])
        with source_col:
            source_type = st.selectbox(
                "Video source",
                ["Webcam", "RTSP stream", "Upload video", "Included demo video"],
                disabled=st.session_state.stream_running,
            )
            webcam_index = st.number_input(
                "Camera index",
                min_value=0,
                max_value=10,
                value=0,
                disabled=st.session_state.stream_running
                or source_type != "Webcam",
            )
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "mov", "avi", "mkv"],
                disabled=st.session_state.stream_running
                or source_type != "Upload video",
            )
        with setting_col:
            rtsp_url = st.text_input(
                "RTSP URL",
                placeholder="rtsp://camera-address/stream",
                type="password",
                disabled=st.session_state.stream_running
                or source_type != "RTSP stream",
            )
            confidence = st.slider(
                "Detection confidence",
                min_value=0.10,
                max_value=0.90,
                value=0.35,
                step=0.05,
                disabled=st.session_state.stream_running,
            )
            st.caption(
                "Included demo playback loops automatically. Webcam/RTSP "
                "capture is server-side; run this app on the camera host."
            )

        if not st.session_state.stream_running:
            if st.button("▶  Play stream", type="primary"):
                try:
                    capture, camera_id, source_kind = open_capture(
                        source_type,
                        int(webcam_index),
                        rtsp_url,
                        uploaded_file,
                    )
                except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                    st.error(f"Could not start the selected source: {error}")
                else:
                    if not capture.isOpened():
                        capture.release()
                        remove_capture_temp_file()
                        st.error(
                            "Could not open this source. Check the camera "
                            "index, RTSP address, or video file."
                        )
                    else:
                        st.session_state.capture = capture
                        st.session_state.camera_id = camera_id
                        st.session_state.source_kind = source_kind
                        st.session_state.run_id = str(uuid.uuid4())
                        st.session_state.frame_number = 0
                        st.session_state.next_track_id = 1
                        st.session_state.tracks = []
                        st.session_state.confidence_threshold = confidence
                        st.session_state.stream_running = True
                        st.rerun()
        else:
            if st.button("■  Stop stream", type="secondary"):
                stop_stream()
                st.rerun()
        render_live_monitor()

    with records_tab:
        st.subheader("Vehicle records")
        st.caption(
            "Searchable vehicle observations follow the master record fields "
            "in model.md. Unknown predictions are stored as null."
        )
        search_col, class_col, source_col = st.columns([2, 1, 1])
        search_text = search_col.text_input(
            "Search", placeholder="Plate, camera, or track ID"
        )
        class_values = (
            ["All"] + sorted(records["vehicle_type"].dropna().unique().tolist())
            if not records.empty
            else ["All"]
        )
        selected_class = class_col.selectbox("Vehicle class", class_values)
        source_values = (
            ["All"] + sorted(records["source"].dropna().unique().tolist())
            if not records.empty
            else ["All"]
        )
        selected_source = source_col.selectbox("Source", source_values)

        filtered = records.copy()
        if selected_class != "All":
            filtered = filtered[filtered["vehicle_type"] == selected_class]
        if selected_source != "All":
            filtered = filtered[filtered["source"] == selected_source]
        if search_text.strip():
            query = search_text.strip().lower()
            searchable = (
                filtered["camera_id"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(query)
                | filtered["track_id"].astype(str).str.contains(query)
                | filtered["plate_text"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(query)
            )
            filtered = filtered[searchable]
        if filtered.empty:
            st.info("No records match this filter yet.")
        else:
            st.dataframe(
                filtered[
                    [
                        "timestamp",
                        "camera_id",
                        "track_id",
                        "vehicle_type",
                        "confidence",
                        "plate_text",
                        "make",
                        "model",
                        "color",
                        "source",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                    "Detection confidence",
                    min_value=0,
                    max_value=1,
                    format="%.0%%",
                    ),
                },
            )
            choices = filtered["track_id"].astype(int).tolist()
            selected_track = st.selectbox("Record details · track ID", choices)
            detail = filtered[filtered["track_id"] == selected_track].iloc[0].to_dict()
            detail["watchlist_flagged"] = bool(detail["watchlist_flagged"])
            detail["detected_attributes"] = json.loads(detail["detected_attributes"])
            with st.expander("Master record details", expanded=True):
                st.json(detail)

        st.divider()
        st.markdown("#### Delete stored records")
        st.warning(
            "This permanently removes all vehicle records from the local database. "
            "It does not delete model files or the supplied demo archive."
        )
        confirm_delete = st.checkbox(
            "I understand this will delete all stored vehicle records."
        )
        if st.button(
            "Delete all records",
            disabled=not confirm_delete or st.session_state.stream_running,
        ):
            with connect_database() as connection:
                connection.execute("DELETE FROM vehicle_records")
            st.success("All stored vehicle records have been deleted.")
            st.rerun()

    with models_tab:
        st.subheader("Models & local data")
        st.markdown(
            """
            | Asset | Role | Status |
            |---|---|---|
            | `vehicle_det_fp16.onnx` | YOLO11m vehicle detection | Used for live inference |
            | `plateReader.pt` | Plate-reading model | Present; not loaded until its inference contract is verified |
            | `best.pt.zip` | Model checkpoint archive | Available in workspace |
            """
        )
        st.markdown("#### Included trajectory data")
        st.write(
            "Import the trajectory CSV bundled in `member1_outputs.zip` to "
            "browse existing tracked vehicles without starting a camera."
        )
        if st.button(
            "Import included vehicle records",
            disabled=st.session_state.stream_running,
        ):
            try:
                imported = import_demo_records()
            except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
                st.error(f"Could not import demo records: {error}")
            else:
                st.success(f"Imported or refreshed {imported} tracked vehicles.")
                st.rerun()
        st.caption(f"Local database: {DATABASE_PATH}")
        st.caption(
            "Camera frames and uploaded videos are not saved. Vehicle metadata "
            "is kept in SQLite until you delete it."
        )


if __name__ == "__main__":
    main()
