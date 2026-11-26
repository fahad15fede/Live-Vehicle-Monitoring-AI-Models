"""Karachi Smart Traffic AI - Streamlit front end.

All model logic lives in the `kst` package; this file is UI only.
Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from kst import config, database
from kst.appearance import COLOR_NAMES
from kst.overlay import draw_detections
from kst.pipeline import ZONE_TABLE_COLUMNS, Pipeline
from kst.vehicle_analysis import VehicleAnalyzer
from kst.vehicle_detection import TwoWheelerDetector, VehicleDetector

st.set_page_config(page_title="Karachi Smart Traffic AI", page_icon="🚦", layout="wide")

ZONE_TABLE_CONFIG = {
    "Conf": st.column_config.ProgressColumn("Conf", min_value=0.0, max_value=1.0, format="%.2f"),
    "Entered zone": st.column_config.CheckboxColumn("Entered zone"),
    "In zone": st.column_config.CheckboxColumn("In zone"),
}


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading vehicle detector (ONNX)...")
def get_detector() -> VehicleDetector:
    return VehicleDetector()


@st.cache_resource(show_spinner="Loading feature model, plate OCR...")
def get_analyzer() -> VehicleAnalyzer:
    return VehicleAnalyzer()


@st.cache_resource(show_spinner="Loading two-wheeler detector (plateReader.pt / COCO)...")
def get_two_wheeler_detector():
    try:
        return TwoWheelerDetector()
    except Exception:   # optional: missing / replaced weights just disable it
        return None


@st.cache_resource
def get_db():
    return database.connect()


def new_pipeline(cam_id: str) -> Pipeline:
    return Pipeline(get_detector(), get_analyzer(), get_db(), cam_id,
                    two_wheeler_detector=get_two_wheeler_detector())


# ---------------------------------------------------------------- stream control
def stop_stream() -> None:
    cap = st.session_state.pop("cap", None)
    if cap is not None:
        cap.release()
    tmp = st.session_state.pop("tmp_path", None)
    if tmp and os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass
    st.session_state.running = False


def _tmp_video(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    st.session_state.tmp_path = path
    return path


def start_stream(kind: str, cam_index: int, rtsp: str, upload) -> None:
    stop_stream()
    if kind == "Webcam":
        cap, cam_id = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW), f"CAM-{cam_index}"
    elif kind == "RTSP / IP camera":
        if not rtsp.strip():
            st.error("Enter an RTSP URL first.")
            return
        cap, cam_id = cv2.VideoCapture(rtsp.strip()), "KHI-RTSP-01"
    elif kind == "Upload video":
        if upload is None:
            st.error("Choose a video file first.")
            return
        cap = cv2.VideoCapture(_tmp_video(upload.getvalue(), Path(upload.name).suffix or ".mp4"))
        cam_id = "KHI-UPLOAD-01"
    else:
        with zipfile.ZipFile(config.DEMO_ARCHIVE) as z:
            cap = cv2.VideoCapture(_tmp_video(z.read(config.DEMO_VIDEO_MEMBER), ".mp4"))
        cam_id = "KHI-DEMO-01"
    if not cap.isOpened():
        st.error("Could not open the source. Check the camera index / URL / file.")
        cap.release()
        return
    if kind == "Webcam":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # always show the newest camera frame
    st.session_state.cap = cap
    st.session_state.is_file = kind in ("Upload video", "Included demo video")
    st.session_state.src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st.session_state.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    st.session_state.pipeline = new_pipeline(cam_id)
    st.session_state.finished = None
    st.session_state.analysed_frames = 0
    st.session_state.running = True


def read_frame(skip: int):
    """Analyse one frame, then drop the next `skip` frames (any source).
    The first frame of a run is read without skipping. Files are played once."""
    cap = st.session_state.cap
    if st.session_state.get("analysed_frames", 0) > 0:
        for _ in range(skip):
            if not cap.grab():
                return None
    ok, frame = cap.read()
    if not ok:
        return None
    st.session_state.analysed_frames = st.session_state.get("analysed_frames", 0) + 1
    return frame


def finish_stream(reason: str) -> None:
    pl = st.session_state.pipeline
    st.session_state.finished = {"reason": reason, "counts": dict(pl.counts),
                                 "tracks": pl.tracker.next_id - 1, "camera": pl.camera_id,
                                 "rows": pl.seen_table().to_dict("records")}
    stop_stream()


# ---------------------------------------------------------------- UI pieces
def sidebar() -> dict:
    with st.sidebar:
        st.header("🚦 Karachi Smart Traffic AI")
        kind = st.radio("Source", ["Webcam", "RTSP / IP camera", "Upload video", "Included demo video"])
        cam_index = st.number_input("Webcam index", 0, 10, 0) if kind == "Webcam" else 0
        rtsp = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:554/stream") \
            if kind == "RTSP / IP camera" else ""
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"]) \
            if kind == "Upload video" else None
        skip = st.slider("Frames to skip after each analysed frame", 0, 30, config.DEFAULT_SKIP_FRAMES,
                         help="Analyse 1 frame, skip this many, analyse the next. "
                              "5 = frames 1, 7, 13, 19 ... are analysed.")
        c1, c2 = st.columns(2)
        if c1.button("▶ Start", type="primary", width="stretch"):
            start_stream(kind, int(cam_index), rtsp, upload)
        if c2.button("■ Stop", width="stretch") and st.session_state.get("running"):
            finish_stream("Stopped by operator")

        st.subheader("Confidence thresholds")
        st.caption("Kept low on purpose: recall over precision. Raise to cut noise.")
        p = {
            "skip": skip,
            "vehicle_conf": st.slider("Vehicle", 0.05, 0.9, config.VEHICLE_CONF, 0.05),
            "feat_conf": st.slider("Features (best.pt)", 0.05, 0.9, config.FEATURE_CONF, 0.05),
            "plate_conf": st.slider("Plate OCR", 0.05, 0.9, config.PLATE_CONF, 0.05),
        }
    return p


def live_tab(p: dict) -> None:
    if not st.session_state.get("running"):
        done = st.session_state.get("finished")
        if done:
            st.success(f"**{done['reason']}** · {done['camera']} · "
                       f"{sum(done['counts'].values())} vehicles passed the reading zone "
                       f"({done['tracks']} tracked in total). Results are in **Search** / **Records**.")
            if done["counts"]:
                st.bar_chart(pd.Series(done["counts"], name="vehicles"))
            all_frames_table(done["rows"])
        else:
            st.info("Choose a source in the sidebar and press **Start**. Vehicles are counted, "
                    "and their plate, features and colour are read, inside the reading zone "
                    f"({config.READING_ZONE[0]:.0%}-{config.READING_ZONE[1]:.0%} of the frame height).")
        return

    @st.fragment(run_every=0.1)
    def loop():
        if not st.session_state.get("running"):
            return
        frame = read_frame(p["skip"])
        if frame is None:
            finish_stream("Video finished" if st.session_state.is_file else "Camera signal lost")
            st.rerun()
        pl = st.session_state.pipeline
        t0 = time.time()
        dets = pl.process(frame, p["vehicle_conf"], p["feat_conf"], p["plate_conf"])
        dt = time.time() - t0
        left, right = st.columns([3, 1.3])
        caption = (f"{pl.camera_id} · analysed frame #{st.session_state.analysed_frames} "
                   f"(1 analysed, {p['skip']} skipped) · {dt*1000:.0f} ms")
        if st.session_state.is_file and st.session_state.total_frames:
            pos = int(st.session_state.cap.get(cv2.CAP_PROP_POS_FRAMES))
            left.progress(min(1.0, pos / st.session_state.total_frames))
            caption += f" · frame {pos}/{st.session_state.total_frames}"
        left.image(draw_detections(frame, dets, pl.zone, pl.counts), channels="BGR",
                   width="stretch", caption=caption)
        with right:
            a, b, c = st.columns(3)
            a.metric("In frame", len(dets))
            b.metric("Zone vehicles", len(pl.seen))
            c.metric("Read in zone", sum(pl.counts.values()))
            type_counts = Counter(r["Type"] for r in pl.seen.values())
            if type_counts:
                st.caption(" · ".join(f"**{k}** {v}" for k, v in type_counts.most_common()))
    loop()
    zone_entries_table()


@st.fragment(run_every=0.2)
def zone_entries_table() -> None:
    if not st.session_state.get("running"):
        return
    pl = st.session_state.pipeline
    df = pl.seen_table().reset_index(drop=True)
    st.subheader(f"Reading-zone vehicle entries ({len(df)})")
    st.caption(f"Updated through analyzed frame {pl.frame_no}.")
    st.dataframe(df, hide_index=True, width="stretch", height=360,
                 key=f"live_zone_entries_{pl.run_id}", column_config=ZONE_TABLE_CONFIG)


def all_frames_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=ZONE_TABLE_COLUMNS)
    st.subheader(f"Vehicles that entered the reading zone ({len(df)})")
    if df.empty:
        st.caption("No vehicles have entered the reading zone.")
        return
    st.dataframe(df, hide_index=True, width="stretch", height=360,
                 key="final_zone_entries", column_config=ZONE_TABLE_CONFIG)


def image_tab(p: dict) -> None:
    st.caption("Analyse a single photo: best for testing plate reading on a close-up car.")
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"], key="img")
    if up is None:
        return
    frame = cv2.cvtColor(np.asarray(Image.open(up).convert("RGB")), cv2.COLOR_RGB2BGR)
    pipe = new_pipeline("KHI-PHOTO")
    dets = pipe.process_photo(frame, p["vehicle_conf"], p["feat_conf"], p["plate_conf"])
    if not dets:   # close-up photo the detector misses: analyse the whole image as one vehicle
        dets = [pipe.analyze_whole_image(frame, p["feat_conf"], p["plate_conf"])]
        st.warning("No vehicle detected, so the whole image was analysed as one vehicle.")
    st.image(draw_detections(frame, dets), channels="BGR", width="stretch")
    st.json([{"id": d.track_id, "type": d.class_name, "conf": round(d.confidence, 3), **d.info}
             for d in dets])


def search_tab() -> None:
    con = get_db()
    c1, c2 = st.columns([2, 1])
    plate = c1.text_input("Plate number (partial OK, e.g. BKA 492, KHI, 492)")
    fuzz = c2.slider("Plate match strictness", 0.3, 1.0, 0.6, 0.05,
                     help="Lower finds more OCR-garbled plates (O/0, B/8, S/5 are treated alike).")
    c3, c4, c5 = st.columns(3)
    types = c3.multiselect("Vehicle type", get_detector().names)
    colors = c4.multiselect("Color", COLOR_NAMES)
    features = get_analyzer().features
    feat_names = (list(features.names.values()) if features else []) + ["no_front_plate_seen"]
    feats = c5.multiselect("Has features", feat_names)
    df = database.search(con, plate, types, colors, feats, fuzz)
    st.write(f"**{len(df)}** matching sightings")
    if df.empty:
        return
    for _, r in df.head(60).iterrows():
        with st.container(border=True):
            a, b = st.columns([1, 3])
            if r.crop_path and os.path.exists(r.crop_path):
                a.image(r.crop_path, width="stretch")
            feats_txt = ", ".join(json.loads(r.features or "[]")) or "—"
            match = f" · match {r['match']:.0%}" if "match" in r and pd.notna(r.get("match")) else ""
            b.markdown(
                f"**Plate:** `{r.plate_text or 'unread'}`{match}  \n"
                f"**Type:** {r.vehicle_type} ({r.det_conf:.0%}) · **Color:** {r.color or '—'} · "
                f"**Shape:** {r.shape or '—'}  \n**Features:** {feats_txt}  \n"
                f"**Camera:** {r.camera_id} · track #{r.track_id} · last seen {r.last_seen}")


def records_tab() -> None:
    con = get_db()
    df = database.all_records(con)
    st.dataframe(df.drop(columns=["plate_norm"], errors="ignore"), width="stretch",
                 hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "sightings.csv")
    with st.expander("Danger zone"):
        if st.checkbox("I understand this deletes every stored sighting") and \
                st.button("Delete all records"):
            database.clear(con)
            st.rerun()


def models_tab() -> None:
    det, an = get_detector(), get_analyzer()
    st.markdown(f"""
| Stage | Model | Status |
|---|---|---|
| Vehicle detection | `vehicle_det_fp16.onnx` (YOLO11m) | ✅ {', '.join(det.names)} · {det.session.get_providers()[0]} |
| Features | `best.pt` (YOLOv8n) | {'✅ ' + ', '.join(an.features.names.values()) if an.features else '❌ ' + an.errors.get('features', '')} |
| Plate detection + OCR | fast-alpr: `{config.ALPR_DETECTOR}` + `{config.ALPR_OCR}` (ONNX) | {'✅' if an.alpr else '❌ ' + an.errors.get('fast-alpr', '') + ' (falling back to best.pt / RapidOCR)'} |
| Plate OCR fallback | RapidOCR (PaddleOCR on ONNX) + Pillow cleaning | {'✅' if an.plates else '❌ ' + an.errors.get('ocr', '')} |
| Two-wheeler detector | `plateReader.pt` (COCO YOLOv8n: bicycle + motorcycle) | {'✅ active, adds bikes the main model misses' if get_two_wheeler_detector() else '❌ unavailable'} |
| Color | Pillow Shades-of-Gray + HSV voting | ✅ heuristic (no colour model delivered) |
| Shape | Box geometry + vehicle class | ✅ heuristic |
| Tracking | IoU tracker + per-track majority votes | ✅ |
""")


def main() -> None:
    st.session_state.setdefault("running", False)
    p = sidebar()
    tabs = st.tabs(["📹 Live monitor", "🖼️ Photo check", "🔎 Search", "🗂️ Records", "🧠 Models"])
    with tabs[0]:
        live_tab(p)
    with tabs[1]:
        image_tab(p)
    with tabs[2]:
        search_tab()
    with tabs[3]:
        records_tab()
    with tabs[4]:
        models_tab()


main()
