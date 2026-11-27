"""SQLite record store + search (plate fuzzy match and type / colour filters)."""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    vehicle_type TEXT NOT NULL,
    det_conf REAL NOT NULL,
    plate_text TEXT,
    plate_norm TEXT,
    plate_conf REAL,
    color TEXT,
    color_conf REAL,
    shape TEXT,
    features TEXT NOT NULL DEFAULT '[]',
    crop_path TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    UNIQUE(run_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_plate ON sightings(plate_norm);
"""

# Common OCR confusions on Pakistani plates (model.md correction table).
_CONFUSE = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                          "Z": "2", "S": "5", "B": "8", "G": "6"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_plate(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def _fuzzy_key(text: str) -> str:
    return normalize_plate(text).translate(_CONFUSE)


def connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DATABASE_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert(con: sqlite3.Connection, run_id: str, camera_id: str, rec: dict) -> None:
    ts = now()
    con.execute(
        """
        INSERT INTO sightings (run_id, camera_id, track_id, vehicle_type, det_conf,
            plate_text, plate_norm, plate_conf, color, color_conf, shape,
            crop_path, first_seen, last_seen)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id, track_id) DO UPDATE SET
            vehicle_type=excluded.vehicle_type,
            det_conf=MAX(det_conf, excluded.det_conf),
            plate_text=COALESCE(excluded.plate_text, plate_text),
            plate_norm=COALESCE(excluded.plate_norm, plate_norm),
            plate_conf=COALESCE(excluded.plate_conf, plate_conf),
            color=COALESCE(excluded.color, color),
            color_conf=COALESCE(excluded.color_conf, color_conf),
            shape=COALESCE(excluded.shape, shape),
            crop_path=COALESCE(excluded.crop_path, crop_path),
            last_seen=excluded.last_seen
        """,
        (run_id, camera_id, rec["track_id"], rec["vehicle_type"], rec["det_conf"],
         rec.get("plate_text"), normalize_plate(rec.get("plate_text")) or None,
         rec.get("plate_conf"), rec.get("color"), rec.get("color_conf"), rec.get("shape"),
         rec.get("crop_path"), ts, ts),
    )
    con.commit()


def all_records(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM sightings ORDER BY last_seen DESC", con)


def search(con: sqlite3.Connection, plate: str = "", vehicle_types=(), colors=(),
           min_score: float = 0.6) -> pd.DataFrame:
    df = all_records(con)
    if df.empty:
        return df
    if vehicle_types:
        df = df[df.vehicle_type.isin(vehicle_types)]
    if colors:
        df = df[df.color.isin(colors)]
    if plate.strip():
        q = _fuzzy_key(plate)

        def score(p):
            if not p:
                return 0.0
            k = _fuzzy_key(p)
            return 1.0 if q in k else SequenceMatcher(None, q, k).ratio()

        df = df.assign(match=df.plate_norm.apply(score))
        df = df[df.match >= min_score].sort_values("match", ascending=False)
    return df


def clear(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM sightings")
    con.commit()
