"""Central paths and thresholds.

Confidences are intentionally low (recall over precision); the UI sliders let
operators raise them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CROPS_DIR = DATA_DIR / "crops"
DATABASE_PATH = DATA_DIR / "traffic_records.sqlite3"

# ---------------------------------------------------------------- model files
VEHICLE_MODEL = ROOT / "vehicle_det_fp16.onnx"
# plateReader.pt is a stock COCO YOLOv8n (it cannot read plates). It is used only to find
# two-wheelers the vehicle model misses (COCO 1=bicycle, 3=motorcycle).
TWO_WHEELER_MODEL = ROOT / "plateReader.pt"
# best.pt was delivered as best.pt.zip. A torch checkpoint is itself a zip,
# so it is copied (not extracted) to a .pt file on first use.
FEATURE_MODEL_ZIP = ROOT / "best.pt.zip"
FEATURE_MODEL = ROOT / "best.pt"

# ---------------------------------------------------------------- demo video
DEMO_ARCHIVE = ROOT / "member1_outputs.zip"
DEMO_VIDEO_MEMBER = (
    "content/member1_detection_tracking/outputs/annotated_videos/"
    "tracked_traffic_scene.mp4"
)

# ---------------------------------------------------------------- vehicle detection
VEHICLE_CONF = 0.25
PLATE_CONF = 0.15
FEATURE_CONF = 0.15
NMS_IOU = 0.45

# Small vehicles score low on the YOLO11m model; give them their own, lower floor.
CLASS_CONF_FLOOR = {"motorcycle": 0.08, "qingqi": 0.12, "auto_rickshaw": 0.12}
# Prior weights for choosing the class: the model under-scores three-wheelers and
# tends to call them "car". Raise auto_rickshaw if rickshaws still show as cars.
CLASS_WEIGHT = {"auto_rickshaw": 1.6, "qingqi": 1.3}
TWO_WHEELER_CONF = 0.20

# ---------------------------------------------------------------- reading zone
# Reading zone, as fractions of frame height (top, bottom): the bottom 60% of the video.
# Vehicles are counted when they enter it and analysed (plate / features / colour) while inside it.
READING_ZONE = (0.40, 1.00)
MAX_ANALYSES_PER_TRACK = 4
ANALYSES_PER_FRAME = 3          # CPU cap, internal only
# Burned-in camera on-screen text (timestamp / MAC address), as frame fractions
# (x1, y1, x2, y2). Plate boxes found here are ignored. Default = demo camera's OSD.
OSD_MASKS = [(0.0, 0.85, 0.5, 1.0)]
DEFAULT_SKIP_FRAMES = 5         # analyse 1 frame, skip 5, analyse the next
MIN_CROP_SIDE = 40

# ---------------------------------------------------------------- plates
# fast-alpr (github.com/ankandrew/fast-alpr). Models download to the user cache on first run.
ALPR_DETECTOR = "yolo-v9-t-384-license-plate-end2end"
ALPR_DET_CONF = 0.35
ALPR_OCR = "cct-s-v2-global-model"
# Below this mean character confidence a fast-plate-ocr read is dropped and RapidOCR is tried.
ALPR_OCR_MIN_CONF = 0.55
# best.pt's front_plate head is weak (3 epochs); only trust its box above this.
FRONT_PLATE_MIN_CONF = 0.35
# Where plate search starts, as a fraction down the vehicle box (bike boxes include the rider).
PLATE_SEARCH_FROM = {"car": 0.45, "van": 0.45, "motorcycle": 0.60, "qingqi": 0.50,
                     "auto_rickshaw": 0.50, "bus": 0.55, "truck": 0.55}
