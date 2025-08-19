from __future__ import annotations
import json, sqlite3, os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from .types import StepRecord, RunConfig

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL,
  cfg_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id   INTEGER NOT NULL,
  t        INTEGER NOT NULL,

  novelty  REAL, threat REAL, support REAL, control REAL,

  b_energy REAL, b_safety REAL, b_belong REAL,
  s_valence REAL, s_arousal REAL, s_dominance REAL,
  m_da REAL, m_5ht REAL, m_ne REAL,

  action INTEGER,
  reward REAL,
  loss REAL,

  FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_steps_run_t ON steps(run_id, t);
"""

INSERT_RUN = "INSERT INTO runs(started_at, cfg_json) VALUES (?, ?);"
INSERT_STEP = """
INSERT INTO steps(
  run_id, t,
  novelty, threat, support, control,
  b_energy, b_safety, b_belong,
  s_valence, s_arousal, s_dominance,
  m_da, m_5ht, m_ne,
  action, reward, loss
) VALUES (?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?,?);
"""

def _ensure_db(db_path: str) -> sqlite3.Connection:
    Path(os.path.dirname(db_path) or ".").mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    # esegui DDL in transazione
    with conn:
        for stmt in DDL.strip().split(";\n\n"):
            s = stmt.strip()
            if s:
                conn.executescript(s + ";")
    return conn

def _step_tuple(run_id: int, rec: StepRecord) -> Tuple:
    # Ordine dei campi deve combaciare con INSERT_STEP
    return (
        run_id, rec.get("t", 0),
        rec.get("novelty"), rec.get("threat"), rec.get("support"), rec.get("control"),
        rec.get("b_energy"), rec.get("b_safety"), rec.get("b_belong"),
        rec.get("s_valence"), rec.get("s_arousal"), rec.get("s_dominance"),
        rec.get("m_da"), rec.get("m_5ht"), rec.get("m_ne"),
        rec.get("action"), rec.get("reward"), rec.get("loss"),
    )

class SQLiteLogger:
    def __init__(self, db_path: str = "data/runs.sqlite", flush_every: int = 100):
        self.db_path = db_path
        self.flush_every = flush_every
        self.conn = _ensure_db(db_path)
        self.buffer: List[Tuple] = []

    def start_run(self, cfg: RunConfig) -> int:
        cfg_json = json.dumps(cfg.__dict__, ensure_ascii=False)
        started_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self.conn:
            cur = self.conn.execute(INSERT_RUN, (started_at, cfg_json))
            run_id = cur.lastrowid
        return int(run_id)

    def append(self, run_id: int, rec: StepRecord) -> None:
        self.buffer.append(_step_tuple(run_id, rec))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.conn:
            self.conn.executemany(INSERT_STEP, self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.conn.close()
