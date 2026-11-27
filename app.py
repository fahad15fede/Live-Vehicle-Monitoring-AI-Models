"""Karachi Smart Traffic AI - Streamlit front end.

All model logic lives in the `kst` package; this file is UI only.
Run:  python -m streamlit run app.py

Live runs: capture thread -> GPU inference thread (kst.live.LiveRunner) -> this UI, which
only displays the newest result, so the page never blocks on inference.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from kst import config, database, device
from kst.appearance import COLOR_NAMES
from kst.live import LiveRunner
from kst.overlay import draw_detections
from kst.pipeline import ZONE_TABLE_COLUMNS, Pipeline
from kst.stream import FileCapture, LiveCapture
from kst.vehicle_analysis import VehicleAnalyzer
from kst.vehicle_detection import TwoWheelerDetector, VehicleDetector

st.set_page_config(page_title="Karachi Smart Traffic AI", page_icon="🚦", layout="wide")

ZONE_TABLE_CONFIG = {
    "Conf": st.column_config.ProgressColumn("Conf", min_value=0.0, max_value=1.0, format="%.2f"),
    "Entered zone": st.column_config.CheckboxColumn("Entered zone"),
    "In zone": st.column_config.CheckboxColumn("In zone"),
}


# ---------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading vehicle detector (yollo-world.pt)...")
def get_detector() -> VehicleDetector:
    return VehicleDetector()


@st.cache_resource(show_spinner="Loading plate detector + OCR...")
def get_analyzer() -> VehicleAnalyzer:
    return VehicleAnalyzer()


@st.cache_resource(show_spinner="Loading two-wheeler detector (plateReader.pt / COCO)...")
def get_two_wheeler_detector():
    if not config.USE_TWO_WHEELER_DETECTOR:
        return None
    try:
        return TwoWheelerDetector()
    except Exception:   # optional: missing / replaced weights just disable it
        return None


@st.cache_resource
def get_db():
    return database.connect()


def new_pipeline(cam_id: str, con=None) -> Pipeline:
    return Pipeline(get_detector(), get_analyzer(), con or get_db(), cam_id,
                    two_wheeler_detector=get_two_wheeler_detector())


# ---------------------------------------------------------------- stream control
def stop_stream() -> None:
    runner = st.session_state.pop("runner", None)
    if runner is not None:
        runner.stop()
        runner.source.release()
        runner.pipeline.con.close()
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


def start_stream(kind: str, cam_index: int, rtsp: str, upload, p: dict) -> None:
    stop_stream()
    get_detector(), get_analyzer()                # load + warm up models before the clock starts
    if kind == "Webcam":
        src, cam_id = LiveCapture(cam_index), f"CAM-{cam_index}"
    elif kind == "RTSP / IP camera":
        if not rtsp.strip():
            st.error("Enter an RTSP URL first.")
            return
        # returns at once; connecting runs in the background (status shown in the live view)
        src, cam_id = LiveCapture(rtsp.strip()), "KHI-RTSP-01"
    elif kind == "Upload video":
        if upload is None:
            st.error("Choose a video file first.")
            return
        src = FileCapture(_tmp_video(upload.getvalue(), Path(upload.name).suffix or ".mp4"), p["skip"])
        cam_id = "KHI-UPLOAD-01"
    else:
        with zipfile.ZipFile(config.DEMO_ARCHIVE) as z:
            src = FileCapture(_tmp_video(z.read(config.DEMO_VIDEO_MEMBER), ".mp4"), p["skip"])
        cam_id = "KHI-DEMO-01"
    if not src.isOpened():
        st.error("Could not open the video file.")
        src.release()
        return
    # The worker gets its own DB connection: sqlite connections are not safe across threads.
    runner = LiveRunner(src, new_pipeline(cam_id, database.connect()),
                        p["vehicle_conf"], p["plate_conf"])
    runner.start()
    st.session_state.runner = runner
    st.session_state.finished = None
    st.session_state.running = True


def finish_stream(reason: str) -> None:
    runner = st.session_state.runner
    snap = runner.snapshot()
    st.session_state.finished = {"reason": reason, "counts": snap.counts, "tracks": snap.tracks,
                                 "camera": runner.pipeline.camera_id, "rows": snap.rows,
                                 "error": runner.error or getattr(runner.source, "error", None)}
    stop_stream()


# ---------------------------------------------------------------- UI pieces
def sidebar() -> dict:
    with st.sidebar:
        st.header("🚦 Karachi Smart Traffic AI")
        st.caption(f"Compute: **{device.summary()}**")
        kind = st.radio("Source", ["Webcam", "RTSP / IP camera", "Upload video", "Included demo video"])
        cam_index = st.number_input("Webcam index", 0, 10, 0) if kind == "Webcam" else 0
        rtsp = st.text_input("RTSP URL", placeholder="rtsp://user:pass@ip:554/stream") \
            if kind == "RTSP / IP camera" else ""
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"]) \
            if kind == "Upload video" else None
        default_skip = config.DEFAULT_SKIP_FRAMES_GPU if device.USE_CUDA else config.DEFAULT_SKIP_FRAMES
        skip = st.slider("Video files: frames to skip after each analysed frame", 0, 30, default_skip,
                         help="Analyse 1 frame, skip this many, analyse the next. "
                              "Live cameras always analyse the newest frame instead.") \
            if kind in ("Upload video", "Included demo video") else 0

        st.subheader("Confidence thresholds")
        st.caption("Kept low on purpose: recall over precision. Raise to cut noise.")
        p = {
            "skip": skip,
            "vehicle_conf": st.slider("Vehicle", 0.05, 0.9, config.VEHICLE_CONF, 0.05),
            "plate_conf": st.slider("Plate OCR", 0.05, 0.9, config.PLATE_CONF, 0.05),
        }
        c1, c2 = st.columns(2)
        if c1.button("▶ Start", type="primary", width="stretch"):
            start_stream(kind, int(cam_index), rtsp, upload, p)
        if c2.button("■ Stop", width="stretch") and st.session_state.get("running"):
            finish_stream("Stopped by operator")
    runner = st.session_state.get("runner")
    if runner is not None:                        # slider changes apply to the running worker
        runner.vehicle_conf, runner.plate_conf = p["vehicle_conf"], p["plate_conf"]
    return p


def live_tab() -> None:
    # The auto-refreshing fragments are module-level and rendered on EVERY run (they draw
    # nothing when idle). Defining them conditionally made Streamlit log
    # "The fragment with id ... does not exist anymore" after each full rerun.
    if not st.session_state.get("running"):
        done = st.session_state.get("finished")
        if done and done.get("error") and not done["tracks"]:
            # failed before anything was processed: one error box, no empty results
            st.error(f"**{done['reason']}** · {done['camera']}  \n{done['error']}")
        elif done:
            if done.get("error"):
                st.error(f"**{done['reason']}** · {done['camera']}  \n{done['error']}")
            st.success(f"**{done['reason']}** · {done['camera']} · "
                       f"{sum(done['counts'].values())} vehicles passed the reading zone "
                       f"({done['tracks']} tracked in total). Results are in **Search** / **Records**.")
            if done["counts"]:
                st.bar_chart(pd.Series(done["counts"], name="vehicles"))
            all_frames_table(done["rows"])
        else:
            st.info("Choose a source in the sidebar and press **Start**. Vehicles are counted, "
                    "and their plate and colour are read, inside the reading zone "
                    f"({config.READING_ZONE[0]:.0%}-{config.READING_ZONE[1]:.0%} of the frame height).")
    live_view()
    zone_entries_table()


@st.fragment(run_every=1 / config.DISPLAY_FPS)
def live_view() -> None:
    runner = st.session_state.get("runner")
    if runner is None:
        return
    if runner.finished:
        finish_stream(runner.done_reason)
        st.rerun()                                # full rerun: show the results
    snap = runner.snapshot()
    if snap.jpeg is None:
        st.info(snap.status)
        return
    left, right = st.columns([3, 1.3])
    src = runner.source
    caption = (f"{runner.pipeline.camera_id} · analysed frame #{snap.frame_no} · "
               f"{snap.fps:.1f} FPS · inference {snap.infer_ms:.0f} ms")
    caption += f" · 1 analysed, {src.skip} skipped" if src.is_file else " · live: newest frame"
    if snap.status != "Running":
        caption += f" · {snap.status}"
    if snap.progress:
        pos, total = snap.progress
        left.progress(min(1.0, pos / total))
        caption += f" · frame {pos}/{total}"
    left.image(snap.jpeg, width="stretch", caption=caption)
    with right:
        a, b, c = st.columns(3)
        a.metric("In frame", snap.in_frame)
        b.metric("Zone vehicles", snap.zone_vehicles)
        c.metric("Read in zone", snap.read_in_zone)
        if snap.type_counts:
            st.caption(" · ".join(f"**{k}** {v}" for k, v in snap.type_counts.most_common()))
        st.caption(f"Compute: {device.summary()}"
                   + (" · HW video decode" if getattr(src, "hw", False) else ""))


@st.fragment(run_every=0.5)
def zone_entries_table() -> None:
    runner = st.session_state.get("runner")
    if runner is None:
        return
    snap = runner.snapshot()
    df = pd.DataFrame(snap.rows, columns=ZONE_TABLE_COLUMNS)
    st.subheader(f"Reading-zone vehicle entries ({len(df)})")
    st.caption(f"Updated through analysed frame {snap.frame_no}.")
    st.dataframe(df, hide_index=True, width="stretch", height=360,
                 key=f"live_zone_entries_{runner.pipeline.run_id}", column_config=ZONE_TABLE_CONFIG)


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
    dets = pipe.process_photo(frame, p["vehicle_conf"], p["plate_conf"])
    if not dets:   # close-up photo the detector misses: analyse the whole image as one vehicle
        dets = [pipe.analyze_whole_image(frame, p["plate_conf"])]
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
    c3, c4 = st.columns(2)
    types = c3.multiselect("Vehicle type", get_detector().names)
    colors = c4.multiselect("Color", COLOR_NAMES)
    df = database.search(con, plate, types, colors, fuzz)
    st.write(f"**{len(df)}** matching sightings")
    if df.empty:
        return
    for _, r in df.head(60).iterrows():
        with st.container(border=True):
            a, b = st.columns([1, 3])
            if r.crop_path and os.path.exists(r.crop_path):
                a.image(r.crop_path, width="stretch")
            match = f" · match {r['match']:.0%}" if "match" in r and pd.notna(r.get("match")) else ""
            b.markdown(
                f"**Plate:** `{r.plate_text or 'unread'}`{match}  \n"
                f"**Type:** {r.vehicle_type} ({r.det_conf:.0%}) · **Color:** {r.color or '—'} · "
                f"**Shape:** {r.shape or '—'}  \n"
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
    prov = an.providers()
    st.caption(f"Compute device: **{device.summary()}** · ONNX Runtime providers: "
               f"`{[p[0] if isinstance(p, tuple) else p for p in device.ort_providers()]}` "
               "(set `KST_DEVICE=cpu` to force CPU)")
    st.markdown(f"""
| Stage | Model | Status |
|---|---|---|
| Vehicle detection | `{det.model_file}` | ✅ {', '.join(det.names)} · {det.backend} |
| Plate detection + OCR | fast-alpr: `{config.ALPR_DETECTOR}` + `{config.ALPR_OCR}` (ONNX) | {'✅ ' + prov.get('plate detector', '?') + ' / ' + prov.get('plate OCR', '?') if an.alpr else '❌ ' + an.errors.get('fast-alpr', '') + ' (falling back to RapidOCR)'} |
| Plate OCR fallback | RapidOCR (PaddleOCR on ONNX) + Pillow cleaning | {'✅ ' + prov.get('RapidOCR', '?') if an.plates else '❌ ' + an.errors.get('ocr', '')} |
| Two-wheeler detector | `plateReader.pt` (COCO YOLOv8n: bicycle + motorcycle) | {'✅ active, adds bikes the main model misses' if get_two_wheeler_detector() else ('⏸ off: all vehicles come from ' + det.model_file if not config.USE_TWO_WHEELER_DETECTOR else '❌ unavailable')} |
| Color | Pillow Shades-of-Gray + HSV voting | ✅ heuristic (no colour model delivered) |
| Shape | Box geometry + vehicle class | ✅ heuristic |
| Tracking | IoU tracker + per-track majority votes | ✅ |
""")


def main() -> None:
    st.session_state.setdefault("running", False)
    p = sidebar()
    tabs = st.tabs(["📹 Live monitor", "🖼️ Photo check", "🔎 Search", "🗂️ Records", "🧠 Models"])
    with tabs[0]:
        live_tab()
    with tabs[1]:
        image_tab(p)
    with tabs[2]:
        search_tab()
    with tabs[3]:
        records_tab()
    with tabs[4]:
        models_tab()


main()
