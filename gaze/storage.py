"""SQLite behavior logging + aggregation for the dashboard.

Three tables:
  sessions  — one row per monitoring run
  samples   — periodic focus snapshots (focused_ratio + dominant state)
  events    — discrete distraction episodes (drowsy / looking_away / head_turned)

All timestamps are UNIX epoch seconds. The DB lives next to the package
under ~/.gazenailong/history.db so it survives across runs.
"""

import os
import time
import sqlite3
import threading
from datetime import datetime, timedelta

DB_DIR = os.path.join(os.path.expanduser("~"), ".gazenailong")
DB_PATH = os.path.join(DB_DIR, "history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts   REAL NOT NULL,
    end_ts     REAL
);
CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    ts          REAL NOT NULL,
    focused     REAL NOT NULL,      -- 0..1 fraction focused over the window
    state       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    ts          REAL NOT NULL,      -- when the distraction started
    end_ts      REAL,               -- when it ended
    type        TEXT NOT NULL       -- FocusState.value
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class Store:
    """Thread-safe wrapper. The monitor writes from its worker thread while
    the Flask server reads from request threads, so we serialise with a lock
    and open the connection with check_same_thread=False."""

    def __init__(self, path=DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self.session_id = None
        self._open_event_id = None

    # ---------- session lifecycle ----------
    def start_session(self):
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO sessions(start_ts) VALUES (?)", (time.time(),))
            self._db.commit()
            self.session_id = cur.lastrowid
        return self.session_id

    def end_session(self):
        if self.session_id is None:
            return
        self.close_event()  # close any dangling distraction
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET end_ts=? WHERE id=?",
                (time.time(), self.session_id))
            self._db.commit()
        self.session_id = None

    # ---------- writes ----------
    def log_sample(self, focused_ratio, state):
        if self.session_id is None:
            return
        with self._lock:
            self._db.execute(
                "INSERT INTO samples(session_id, ts, focused, state) VALUES (?,?,?,?)",
                (self.session_id, time.time(), float(focused_ratio), str(state)))
            self._db.commit()

    def open_event(self, state):
        """Record the start of a distraction episode."""
        if self.session_id is None:
            return
        self.close_event()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events(session_id, ts, type) VALUES (?,?,?)",
                (self.session_id, time.time(), str(state)))
            self._db.commit()
            self._open_event_id = cur.lastrowid

    def close_event(self):
        if self._open_event_id is None:
            return
        with self._lock:
            self._db.execute(
                "UPDATE events SET end_ts=? WHERE id=?",
                (time.time(), self._open_event_id))
            self._db.commit()
        self._open_event_id = None

    # ---------- reads / aggregation for charts ----------
    def _epoch_for_day_start(self, days_ago=0):
        d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        d -= timedelta(days=days_ago)
        return d.timestamp()

    def recent_samples(self, since_seconds=900):
        """Focus timeline for the live chart (default last 15 min)."""
        cutoff = time.time() - since_seconds
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, focused, state FROM samples WHERE ts>=? ORDER BY ts",
                (cutoff,)).fetchall()
        return [{"ts": r["ts"], "focused": r["focused"], "state": r["state"]} for r in rows]

    def distraction_breakdown(self, since_seconds=86400):
        """Total seconds spent in each distraction type (default last 24h)."""
        cutoff = time.time() - since_seconds
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT type, ts, COALESCE(end_ts, ?) AS e FROM events WHERE ts>=?",
                (now, cutoff)).fetchall()
        totals = {}
        counts = {}
        for r in rows:
            dur = max(0.0, r["e"] - r["ts"])
            totals[r["type"]] = totals.get(r["type"], 0.0) + dur
            counts[r["type"]] = counts.get(r["type"], 0) + 1
        return {"seconds": totals, "counts": counts}

    def daily_focus(self, days=7):
        """Average focused fraction per day for the last `days` days."""
        out = []
        with self._lock:
            for i in range(days - 1, -1, -1):
                start = self._epoch_for_day_start(i)
                end = start + 86400
                row = self._db.execute(
                    "SELECT AVG(focused) AS avg_f, COUNT(*) AS n "
                    "FROM samples WHERE ts>=? AND ts<?", (start, end)).fetchone()
                label = datetime.fromtimestamp(start).strftime("%a %m/%d")
                out.append({
                    "day": label,
                    "focused": round((row["avg_f"] or 0.0) * 100, 1),
                    "samples": row["n"],
                })
        return out

    def today_summary(self):
        start = self._epoch_for_day_start(0)
        with self._lock:
            srow = self._db.execute(
                "SELECT AVG(focused) AS avg_f, COUNT(*) AS n FROM samples WHERE ts>=?",
                (start,)).fetchone()
            erow = self._db.execute(
                "SELECT COUNT(*) AS c FROM events WHERE ts>=?", (start,)).fetchone()
        n = srow["n"] or 0
        # samples are spaced ~SAMPLE_LOG_INTERVAL secs apart
        from .config import SAMPLE_LOG_INTERVAL
        focused_min = round(n * SAMPLE_LOG_INTERVAL * (srow["avg_f"] or 0.0) / 60.0, 1)
        return {
            "avg_focused": round((srow["avg_f"] or 0.0) * 100, 1),
            "focused_minutes": focused_min,
            "distractions": erow["c"] or 0,
        }

    def close(self):
        with self._lock:
            self._db.close()
