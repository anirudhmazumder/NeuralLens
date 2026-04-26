"""Persistent experience buffer backed by SQLite.

Stores every (state, action, reward) tuple across runs and pod restarts,
so the agent can learn from past successes and failures. A module-level
singleton (`get_buffer()`) lets both optimizer.py and main.py share one DB
connection without passing the instance around.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# /workspace/ persists across pod restarts in most cloud envs; fall back locally
_DEFAULT_DB = Path(os.getenv("MEMORY_DB_PATH", "./neurallens_memory.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    url           TEXT    NOT NULL,
    action_type   TEXT    NOT NULL,
    action_detail TEXT    NOT NULL,
    before_score  REAL    NOT NULL,
    after_score   REAL    NOT NULL,
    reward        REAL    NOT NULL,
    accepted      BOOLEAN NOT NULL,
    page_text_hash TEXT   NOT NULL,
    roi_deltas    TEXT    NOT NULL
)
"""


class ExperienceBuffer:
    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def store(
        self,
        url: str,
        action: dict,
        before_score: float,
        after_score: float,
        reward: float,
        accepted: bool,
        roi_deltas: dict,
        page_text: str = "",
    ) -> None:
        text_hash = hashlib.md5(page_text.encode()).hexdigest() if page_text else ""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO experiences
                   (timestamp, url, action_type, action_detail, before_score,
                    after_score, reward, accepted, page_text_hash, roi_deltas)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    url,
                    action.get("action_type", "unknown"),
                    json.dumps(action),
                    float(before_score),
                    float(after_score),
                    float(reward),
                    int(accepted),
                    text_hash,
                    json.dumps(roi_deltas),
                ),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_negative_experiences(self, action_type: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM experiences WHERE reward < 0"
        params: list = []
        if action_type:
            sql += " AND action_type = ?"
            params.append(action_type)
        sql += " ORDER BY timestamp DESC LIMIT 20"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_positive_experiences(self, action_type: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM experiences WHERE reward > 0"
        params: list = []
        if action_type:
            sql += " AND action_type = ?"
            params.append(action_type)
        sql += " ORDER BY reward DESC LIMIT 20"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def get_action_stats(self) -> dict:
        """Return per-action-type aggregate stats across all stored experiences."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                       action_type,
                       AVG(reward)  AS avg_reward,
                       SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) * 1.0
                           / COUNT(*) AS success_rate,
                       COUNT(*)     AS count
                   FROM experiences
                   GROUP BY action_type
                   ORDER BY avg_reward DESC"""
            ).fetchall()
        return {
            row["action_type"]: {
                "avg_reward": round(float(row["avg_reward"]), 4),
                "success_rate": round(float(row["success_rate"]), 3),
                "count": int(row["count"]),
            }
            for row in rows
        }

    def get_similar_page_experiences(self, page_text_hash: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences WHERE page_text_hash = ? "
                "ORDER BY timestamp DESC LIMIT 10",
                (page_text_hash,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ── Module-level singleton ────────────────────────────────────────────────────

_buffer: Optional[ExperienceBuffer] = None


def get_buffer() -> ExperienceBuffer:
    """Return the shared ExperienceBuffer instance (lazy-init, one per process)."""
    global _buffer
    if _buffer is None:
        _buffer = ExperienceBuffer()
    return _buffer
