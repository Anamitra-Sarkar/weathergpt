"""User context store — SQLite per user_id, fact with provenance."""
from __future__ import annotations
import sqlite3, json, time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

DB_PATH = Path("weathergpt.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS user_context (
        user_id TEXT, fact TEXT, value TEXT, confidence REAL, source TEXT, created_at TEXT, updated_at TEXT, confirmed INTEGER, expiry TEXT, PRIMARY KEY (user_id, fact)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, decision TEXT, forecast TEXT, actual TEXT, feedback TEXT, timestamp TEXT
    )""")
    conn.commit()
    conn.close()

init_db()

def upsert_fact(user_id: str, fact: str, value: Any, confidence: float = 0.9, source: str = "user", confirmed: bool = True, expiry: Optional[str]=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    # Check conflict
    c.execute("SELECT value FROM user_context WHERE user_id=? AND fact=?", (user_id, fact))
    row = c.fetchone()
    if row and row[0] != str(value):
        # Conflict — keep newer with higher confidence
        pass
    c.execute("INSERT OR REPLACE INTO user_context (user_id,fact,value,confidence,source,created_at,updated_at,confirmed,expiry) VALUES (?,?,?,?,?,?,?,?,?)",
              (user_id, fact, str(value), confidence, source, now, now, int(confirmed), expiry))
    conn.commit()
    conn.close()

def get_context(user_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT fact,value,confidence,source,confirmed,expiry FROM user_context WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    out={}
    for fact,value,conf,source,confirmed,expiry in rows:
        # Check expiry
        if expiry and expiry < datetime.utcnow().isoformat():
            continue
        out[fact] = {"value": value, "confidence": conf, "source": source, "confirmed": bool(confirmed)}
    return out

def add_feedback(user_id: str, decision: str, forecast: str, actual: str, feedback: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (user_id,decision,forecast,actual,feedback,timestamp) VALUES (?,?,?,?,?,?)",
              (user_id, decision, forecast, actual, feedback, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
