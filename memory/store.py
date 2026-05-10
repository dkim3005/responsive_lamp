from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from config import DB_PATH, OBJECT_IOU_DUP_THRESHOLD


class MemoryStore:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True) if Path(db_path).parent != Path(".") else None
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                bbox TEXT NOT NULL,
                zone TEXT NOT NULL,
                confidence REAL NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                count INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_label ON memory(label);
            CREATE INDEX IF NOT EXISTS idx_last_seen ON memory(last_seen);

            CREATE TABLE IF NOT EXISTS latency_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT NOT NULL,
                value_ms REAL NOT NULL,
                ts REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    def upsert_observation(self, label: str, bbox: list[float], zone: str, conf: float, ts: float | None = None) -> str:
        ts = ts or time.time()
        rows = self.conn.execute(
            "SELECT * FROM memory WHERE label = ? AND last_seen >= ? ORDER BY last_seen DESC",
            (label, ts - 60.0),
        ).fetchall()
        for row in rows:
            old_bbox = json.loads(row["bbox"])
            if _iou(old_bbox, bbox) >= OBJECT_IOU_DUP_THRESHOLD:
                merged = [0.7 * old + 0.3 * new for old, new in zip(old_bbox, bbox)]
                self.conn.execute(
                    """
                    UPDATE memory
                    SET bbox = ?, zone = ?, confidence = ?, last_seen = ?, count = count + 1
                    WHERE id = ?
                    """,
                    (json.dumps(merged), zone, conf, ts, row["id"]),
                )
                self.conn.commit()
                return "update"

        self.conn.execute(
            """
            INSERT INTO memory(label, bbox, zone, confidence, first_seen, last_seen, count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (label, json.dumps(bbox), zone, conf, ts, ts),
        )
        self.conn.commit()
        return "insert"

    def query(self, object_name: str, limit: int = 5) -> list[dict]:
        q = f"%{object_name.lower().strip()}%"
        rows = self.conn.execute(
            """
            SELECT * FROM memory
            WHERE lower(label) LIKE ?
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (q, limit),
        ).fetchall()
        return [dict(row) | {"bbox": json.loads(row["bbox"])} for row in rows]

    def log_latency(self, metric: str, value_ms: float) -> None:
        self.conn.execute(
            "INSERT INTO latency_log(metric, value_ms, ts) VALUES (?, ?, ?)",
            (metric, value_ms, time.time()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def zone_for_bbox(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    col = "left" if cx < 1 / 3 else "center" if cx < 2 / 3 else "right"
    row = "top" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "bottom"
    return f"{row}-{col}"


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0

