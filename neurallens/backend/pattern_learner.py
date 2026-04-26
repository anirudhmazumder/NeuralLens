"""Pattern learning engine — discovers TRANSFERABLE patterns about what drives neural engagement.

Stores every experience in SQLite, then periodically mines correlations between
edit features (lexical complexity, urgency, social proof, etc.) and ROI deltas.
The top patterns are injected into the agent's system prompt so behaviour
genuinely improves across sessions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DB = Path(os.getenv("MEMORY_DB_PATH", "./neurallens_memory.db"))

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS pattern_experiences (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT    NOT NULL,
        url                 TEXT    NOT NULL,
        url_hash            TEXT    NOT NULL,
        component_type      TEXT    NOT NULL,
        action_type         TEXT    NOT NULL,
        before_content      TEXT,
        after_content       TEXT,
        before_scores       TEXT    NOT NULL,
        after_scores        TEXT    NOT NULL,
        reward              REAL    NOT NULL,
        accepted            BOOLEAN NOT NULL,
        extracted_features  TEXT    NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS patterns (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type            TEXT    NOT NULL,
        pattern_description     TEXT    NOT NULL,
        pattern_features        TEXT    NOT NULL,
        avg_language_roi_delta  REAL    DEFAULT 0,
        avg_attention_roi_delta REAL    DEFAULT 0,
        avg_visual_roi_delta    REAL    DEFAULT 0,
        avg_overall_delta       REAL    DEFAULT 0,
        confidence              REAL    DEFAULT 0,
        sample_count            INTEGER DEFAULT 0,
        last_updated            TEXT    NOT NULL
    )""",
]

# ── Feature extractors ────────────────────────────────────────────────────────

_URGENCY_WORDS = {
    "now", "free", "instant", "instantly", "proven", "guaranteed", "limited",
    "exclusive", "today", "immediately", "fast", "quick", "easy", "simple",
    "boost", "double", "triple", "save", "get", "start",
}
_POSITIVE_WORDS = {
    "great", "amazing", "excellent", "best", "trusted", "loved", "award",
    "winner", "top", "leading", "premium", "powerful", "effective", "results",
    "success", "transform", "discover", "unlock",
}
_NEGATIVE_WORDS = {
    "hard", "difficult", "complex", "confusing", "expensive", "costly",
    "risk", "problem", "issue", "struggle", "fail", "lost", "miss", "waste",
}
_PASSIVE_HELPERS = {"was", "were", "is", "are", "be", "been", "being"}
# Common past-participle endings. A be-verb + word-ending-in-these = passive bigram.
_PP_ENDINGS = ("ed", "en", "ied", "wn", "ult", "ilt", "ept", "ent", "ung", "unk")


def _flesch_ease(text: str) -> float:
    try:
        import textstat  # type: ignore
        return float(textstat.flesch_reading_ease(text))
    except ImportError:
        pass
    sentences = max(1, len(re.split(r"[.!?]+", text)))
    words = text.split()
    if not words:
        return 50.0
    syllables = sum(_syllables(w) for w in words)
    asl = len(words) / sentences
    asw = syllables / len(words)
    return max(0.0, min(100.0, 206.835 - 1.015 * asl - 84.6 * asw))


def _syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:\"'")
    if len(word) <= 3:
        return 1
    count, prev_vowel = 0, False
    for ch in word:
        is_v = ch in "aeiouy"
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e"):
        count -= 1
    return max(1, count)


def _urgency_score(text: str) -> float:
    words = set(text.lower().split())
    return len(words & _URGENCY_WORDS) / max(len(words), 1)


def _social_proof(text: str) -> float:
    numbers = len(re.findall(r"\d[\d,]*[%+]?", text))
    trust_words = sum(
        text.lower().count(w)
        for w in ("customer", "review", "star", "rating", "clients", "users")
    )
    return min(1.0, numbers * 0.1 + trust_words * 0.15)


def _active_voice_ratio(text: str) -> float:
    """Estimate active-voice ratio using be-verb + past-participle bigrams.

    The naive approach of counting 'is', 'are', etc. misclassifies active-voice
    copula constructions like "This is simple" as passive. We instead require a
    be-verb immediately followed by a past-participle-form word (e.g. 'was designed',
    'is shown', 'were launched').
    """
    words = text.lower().split()
    if len(words) < 3:
        return 1.0
    passive_count = 0
    for i in range(len(words) - 1):
        w = words[i].strip(".,!?;:\"'")
        nxt = words[i + 1].strip(".,!?;:\"'")
        if w in _PASSIVE_HELPERS and any(nxt.endswith(sfx) for sfx in _PP_ENDINGS):
            passive_count += 1
    sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
    passive_ratio = min(1.0, passive_count / sentence_count)
    return 1.0 - passive_ratio


def _emotional_valence(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    return pos / total if total else 0.5


def extract_features(before: str, after: str, component_type: str) -> dict:
    bw, aw = before.split(), after.split()
    return {
        "component_type": component_type,
        "length_delta": len(aw) - len(bw),
        "flesch_delta": _flesch_ease(after) - _flesch_ease(before),
        "urgency_delta": _urgency_score(after) - _urgency_score(before),
        "social_proof_delta": _social_proof(after) - _social_proof(before),
        "active_voice_delta": _active_voice_ratio(after) - _active_voice_ratio(before),
        "valence_delta": _emotional_valence(after) - _emotional_valence(before),
        "became_question": int(after.strip().endswith("?") and not before.strip().endswith("?")),
        "after_urgency": _urgency_score(after),
        "after_social_proof": _social_proof(after),
        "after_reading_ease": _flesch_ease(after),
        "after_word_count": len(aw),
    }


# ── Correlation helper ────────────────────────────────────────────────────────

def _pearson_r(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / denom if denom else 0.0


# ── Main class ────────────────────────────────────────────────────────────────

class PatternLibrary:
    def __init__(self, db_path: Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            for stmt in _SCHEMA:
                conn.execute(stmt)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def store_experience(
        self,
        url: str,
        component_type: str,
        action_type: str,
        before_content: str,
        after_content: str,
        before_scores: dict,
        after_scores: dict,
        reward: float,
        accepted: bool,
    ) -> None:
        features = extract_features(before_content, after_content, component_type)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO pattern_experiences
                   (timestamp, url, url_hash, component_type, action_type,
                    before_content, after_content, before_scores, after_scores,
                    reward, accepted, extracted_features)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    url, url_hash, component_type, action_type,
                    before_content[:500], after_content[:500],
                    json.dumps(before_scores), json.dumps(after_scores),
                    float(reward), int(accepted), json.dumps(features),
                ),
            )
        with self._conn() as conn:
            n = conn.execute("SELECT COUNT(*) FROM pattern_experiences").fetchone()[0]
        if n > 0 and n % 5 == 0:
            self.discover_patterns()

    # ── Pattern discovery ─────────────────────────────────────────────────────

    def discover_patterns(self) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM pattern_experiences ORDER BY timestamp DESC LIMIT 300"
            ).fetchall()
        if len(rows) < 3:
            return

        # Group by (action_type, component_type)
        groups: dict[tuple, list] = {}
        for row in rows:
            key = (row["action_type"], row["component_type"])
            groups.setdefault(key, []).append(row)

        now = datetime.now(timezone.utc).isoformat()
        for (action_type, comp_type), grp in groups.items():
            if len(grp) < 3:
                continue

            rewards = [float(r["reward"]) for r in grp]
            feats = [json.loads(r["extracted_features"]) for r in grp]
            after_s = [json.loads(r["after_scores"]) for r in grp]
            before_s = [json.loads(r["before_scores"]) for r in grp]

            lang_d = [a.get("language_roi", 0) - b.get("language_roi", 0) for a, b in zip(after_s, before_s)]
            attn_d = [a.get("attention_roi", 0) - b.get("attention_roi", 0) for a, b in zip(after_s, before_s)]
            vis_d = [a.get("visual_roi", 0) - b.get("visual_roi", 0) for a, b in zip(after_s, before_s)]

            feat_keys = [k for k in feats[0] if k not in ("component_type",)]
            best_feat, best_r = None, 0.0
            for fk in feat_keys:
                vals = [f.get(fk, 0.0) for f in feats]
                r = _pearson_r(vals, rewards)
                if abs(r) > abs(best_r):
                    best_r, best_feat = r, fk

            if best_feat is None or abs(best_r) < 0.25:
                continue

            n = len(grp)
            confidence = min(1.0, n / 10.0) * min(1.0, abs(best_r))
            direction = "increasing" if best_r > 0 else "decreasing"
            avg_lang = sum(lang_d) / n
            avg_attn = sum(attn_d) / n
            avg_vis = sum(vis_d) / n
            avg_reward = sum(rewards) / n
            feat_label = best_feat.replace("_", " ")

            description = (
                f"When applying '{action_type}' to {comp_type} blocks, "
                f"{direction} {feat_label} correlates with reward "
                f"(r={best_r:+.2f}, n={n}, avg_Δoverall={avg_reward:+.4f})"
            )

            if "urgency" in best_feat or "valence" in best_feat:
                ptype = "LEXICAL"
            elif "social_proof" in best_feat:
                ptype = "SOCIAL_PROOF"
            elif "flesch" in best_feat or "length" in best_feat or "active_voice" in best_feat:
                ptype = "COGNITIVE_LOAD"
            elif "question" in best_feat:
                ptype = "STRUCTURAL"
            else:
                ptype = "LEXICAL"

            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id FROM patterns WHERE pattern_description LIKE ?",
                    (f"%{action_type}%{comp_type}%",),
                ).fetchone()
                if existing:
                    conn.execute(
                        """UPDATE patterns SET pattern_description=?,
                           avg_language_roi_delta=?, avg_attention_roi_delta=?,
                           avg_visual_roi_delta=?, avg_overall_delta=?,
                           confidence=?, sample_count=?, last_updated=?
                           WHERE id=?""",
                        (description, avg_lang, avg_attn, avg_vis, avg_reward,
                         confidence, n, now, existing["id"]),
                    )
                else:
                    conn.execute(
                        """INSERT INTO patterns
                           (pattern_type, pattern_description, pattern_features,
                            avg_language_roi_delta, avg_attention_roi_delta,
                            avg_visual_roi_delta, avg_overall_delta,
                            confidence, sample_count, last_updated)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (ptype, description, json.dumps({best_feat: best_r}),
                         avg_lang, avg_attn, avg_vis, avg_reward, confidence, n, now),
                    )

        with self._conn() as conn:
            conn.execute("DELETE FROM patterns WHERE confidence < 0.2 AND sample_count >= 20")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_relevant_patterns(
        self, component_type: str, action_type: str, top_k: int = 3
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM patterns
                   WHERE pattern_description LIKE ? OR pattern_description LIKE ?
                   ORDER BY (confidence * ABS(avg_overall_delta)) DESC LIMIT ?""",
                (f"%{action_type}%", f"%{component_type}%", top_k),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_patterns(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM patterns ORDER BY confidence DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_experience_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM pattern_experiences").fetchone()[0]


# ── Module singleton ──────────────────────────────────────────────────────────

_library: Optional[PatternLibrary] = None


def get_library() -> PatternLibrary:
    global _library
    if _library is None:
        _library = PatternLibrary()
    return _library
