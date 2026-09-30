"""Persistent datastore for collected behavioural sessions.

A single-file SQLite database keeps the collected corpus portable, queryable and
citable in a paper ("sessions were stored in a SQLite datastore"). Each row is
one labelled trial: the full raw multimodal session (JSON) plus indexed metadata
for fast stats and filtering.

The raw JSON is exactly the schema the collector posts and the trainer consumes,
so a collected corpus can be fed straight back into `bcg.train` by pointing the
dataset loader at the export.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .config import ARTIFACTS
from .features.aggregate import session_to_features

DB_PATH = ARTIFACTS / "dataset" / "sessions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    participant   TEXT,
    label         TEXT NOT NULL,          -- 'human' or a bot family name
    is_bot        INTEGER NOT NULL,       -- 0 human, 1 bot
    task          TEXT,                   -- which collection task produced it
    consent       INTEGER NOT NULL DEFAULT 0,
    user_agent    TEXT,
    n_mouse       INTEGER,
    n_keys        INTEGER,
    n_clicks      INTEGER,
    duration_ms   REAL,
    straightness  REAL,
    speed_cv      REAL,
    raw_json      TEXT NOT NULL           -- full multimodal session
);
CREATE INDEX IF NOT EXISTS idx_label ON sessions(label);
CREATE INDEX IF NOT EXISTS idx_ts ON sessions(ts);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def insert_session(raw: dict, meta: dict) -> str:
    """Store one labelled trial. `meta` carries participant/label/task/consent."""
    sid = str(uuid.uuid4())
    feats = session_to_features(raw)
    m = raw.get("mouse", {})
    label = meta.get("label", "human")
    row = {
        "id": sid,
        "ts": time.time(),
        "participant": meta.get("participant"),
        "label": label,
        "is_bot": 0 if label == "human" else 1,
        "task": meta.get("task"),
        "consent": 1 if meta.get("consent") else 0,
        "user_agent": meta.get("user_agent"),
        "n_mouse": len(m.get("x", [])),
        "n_keys": len(raw.get("keystroke", {}).get("dwell", [])),
        "n_clicks": raw.get("clicks", {}).get("n_clicks", 0),
        "duration_ms": feats.get("duration_ms", 0.0),
        "straightness": feats.get("straightness", 0.0),
        "speed_cv": feats.get("speed_cv", 0.0),
        "raw_json": json.dumps(raw),
    }
    with _conn() as c:
        c.execute(
            f"INSERT INTO sessions ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
            list(row.values()),
        )
    return sid


def stats() -> dict[str, Any]:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM sessions").fetchone()["n"]
        by_label = {r["label"]: r["n"] for r in
                    c.execute("SELECT label, COUNT(*) n FROM sessions GROUP BY label")}
        participants = c.execute(
            "SELECT COUNT(DISTINCT participant) n FROM sessions").fetchone()["n"]
        agg = c.execute(
            "SELECT AVG(duration_ms) d, AVG(n_mouse) m, AVG(straightness) s, "
            "AVG(speed_cv) v FROM sessions").fetchone()
        # small distributions for the presentation charts
        straight = [r["straightness"] for r in
                    c.execute("SELECT straightness FROM sessions")]
        cv = [r["speed_cv"] for r in c.execute("SELECT speed_cv FROM sessions")]
    return {
        "total": total,
        "by_label": by_label,
        "n_human": by_label.get("human", 0),
        "n_bot": total - by_label.get("human", 0),
        "participants": participants,
        "avg_duration_ms": round(agg["d"] or 0, 1),
        "avg_mouse_points": round(agg["m"] or 0, 1),
        "avg_straightness": round(agg["s"] or 0, 4),
        "avg_speed_cv": round(agg["v"] or 0, 4),
        "dist_straightness": straight,
        "dist_speed_cv": cv,
    }


def list_sessions(limit: int = 60, label: str | None = None) -> list[dict]:
    q = ("SELECT id, ts, label, is_bot, task, participant, n_mouse, n_keys, "
         "n_clicks, duration_ms, straightness, speed_cv FROM sessions")
    args: list = []
    if label:
        q += " WHERE label = ?"
        args.append(label)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args)]


def get_session(sid: str) -> dict | None:
    with _conn() as c:
        r = c.execute("SELECT raw_json, label, task FROM sessions WHERE id = ?",
                      (sid,)).fetchone()
    if not r:
        return None
    return {"label": r["label"], "task": r["task"], **json.loads(r["raw_json"])}


def export_jsonl() -> str:
    with _conn() as c:
        rows = c.execute("SELECT label, raw_json FROM sessions ORDER BY ts").fetchall()
    out = []
    for r in rows:
        rec = json.loads(r["raw_json"])
        rec["label"] = r["label"]
        out.append(json.dumps(rec))
    return "\n".join(out)


def export_features_csv() -> str:
    """Flat feature CSV — the analysis-ready table for the paper / classic ML."""
    with _conn() as c:
        rows = c.execute("SELECT label, is_bot, task, participant, raw_json "
                         "FROM sessions ORDER BY ts").fetchall()
    if not rows:
        return ""
    lines, header = [], None
    for r in rows:
        raw = json.loads(r["raw_json"])
        feats = session_to_features(raw)
        base = {"label": r["label"], "is_bot": r["is_bot"],
                "task": r["task"], "participant": r["participant"]}
        record = {**base, **feats}
        if header is None:
            header = list(record.keys())
            lines.append(",".join(header))
        lines.append(",".join(str(record[k]) for k in header))
    return "\n".join(lines)
