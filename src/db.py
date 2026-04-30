"""SQLite database module for Lane Detection media library and results."""

import sqlite3
import os
import json
import cv2
import numpy as np
from datetime import datetime

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lane_detection.db"
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS media_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            media_type  TEXT    NOT NULL,
            file_ext    TEXT    NOT NULL,
            file_data   BLOB    NOT NULL,
            thumbnail   BLOB,
            uploaded_at TEXT    DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS detection_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id        INTEGER REFERENCES media_items(id) ON DELETE CASCADE,
            processed_at    TEXT    DEFAULT (datetime('now', 'localtime')),
            params_json     TEXT,
            avg_curvature_m REAL,
            avg_offset_m    REAL,
            lane_detected   INTEGER,
            frames_processed INTEGER,
            result_thumb    BLOB
        );
    """)
    conn.commit()
    conn.close()


def _make_thumbnail(file_data: bytes, file_ext: str, size=(320, 180)) -> bytes:
    """Generate a small JPEG thumbnail from image or first video frame."""
    arr = np.frombuffer(file_data, dtype=np.uint8)
    ext = file_ext.lower()

    if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)
        if not ret:
            return b""
    else:
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return b""

    thumb = cv2.resize(frame, size)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


def save_media(name: str, media_type: str, file_data: bytes, file_ext: str) -> int:
    """Save uploaded media to DB. Returns new row id."""
    thumb = _make_thumbnail(file_data, file_ext)
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO media_items (name, media_type, file_ext, file_data, thumbnail) VALUES (?,?,?,?,?)",
        (name, media_type, file_ext, file_data, thumb),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_media():
    """Return list of all media items (without file_data blob for speed)."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, name, media_type, file_ext, thumbnail, uploaded_at FROM media_items ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return rows


def get_media(media_id: int):
    """Return full media row including file_data."""
    conn = _connect()
    row = conn.execute("SELECT * FROM media_items WHERE id=?", (media_id,)).fetchone()
    conn.close()
    return row


def delete_media(media_id: int):
    conn = _connect()
    conn.execute("DELETE FROM media_items WHERE id=?", (media_id,))
    conn.commit()
    conn.close()


def save_result(
    media_id: int,
    params: dict,
    avg_curvature: float,
    avg_offset: float,
    lane_detected: bool,
    frames_processed: int,
    result_thumb: bytes,
) -> int:
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO detection_results
           (media_id, params_json, avg_curvature_m, avg_offset_m,
            lane_detected, frames_processed, result_thumb)
           VALUES (?,?,?,?,?,?,?)""",
        (
            media_id,
            json.dumps(params),
            avg_curvature,
            avg_offset,
            int(lane_detected),
            frames_processed,
            result_thumb,
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_results_for_media(media_id: int):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM detection_results WHERE media_id=? ORDER BY processed_at DESC",
        (media_id,),
    ).fetchall()
    conn.close()
    return rows


def get_all_results():
    conn = _connect()
    rows = conn.execute("""
        SELECT r.*, m.name as media_name, m.media_type
        FROM detection_results r
        JOIN media_items m ON r.media_id = m.id
        ORDER BY r.processed_at DESC
    """).fetchall()
    conn.close()
    return rows


def db_stats() -> dict:
    conn = _connect()
    media_count  = conn.execute("SELECT COUNT(*) FROM media_items").fetchone()[0]
    result_count = conn.execute("SELECT COUNT(*) FROM detection_results").fetchone()[0]
    conn.close()
    return {"media": media_count, "results": result_count}
