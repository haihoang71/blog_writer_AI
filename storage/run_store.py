"""SQLite persistence for generation runs, events, and local observations.

Survives process restart. Does not store Langfuse secrets or ground-truth
labels that production diagnosis should never see — ground truth lives in
``faults/ground_truth.py`` under a separate table/dir.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_LOCK = threading.Lock()
_DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "runs.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    from config.settings import get_settings

    configured = getattr(get_settings(), "data_dir", None)
    if configured:
        path = Path(str(configured)) / "runs.sqlite3"
    else:
        path = _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    langfuse_trace_id TEXT,
                    topic TEXT,
                    fault_scenario TEXT NOT NULL DEFAULT 'none',
                    execution_mode TEXT NOT NULL DEFAULT 'mock',
                    provider TEXT,
                    model TEXT,
                    status TEXT NOT NULL,
                    enable_hitl INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS observations (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_obs_run ON observations(run_id);
                """
            )
            conn.commit()
        finally:
            conn.close()


_RUN_FIELDS = (
    "run_id",
    "task_id",
    "langfuse_trace_id",
    "topic",
    "fault_scenario",
    "execution_mode",
    "provider",
    "model",
    "status",
    "enable_hitl",
    "error",
    "started_at",
    "ended_at",
    "metadata_json",
)


def _prepare_run_values(record: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in _RUN_FIELDS:
        if key not in record:
            continue
        values[key] = record[key]
    if "enable_hitl" in values:
        values["enable_hitl"] = 1 if record.get("enable_hitl") else 0
    if "metadata_json" in values and isinstance(values["metadata_json"], dict):
        values["metadata_json"] = json.dumps(values["metadata_json"])
    return values


def upsert_run(record: dict[str, Any]) -> None:
    """Insert a run or patch only the fields present on ``record``."""
    init_db()
    if not record.get("run_id"):
        raise ValueError("run_id is required")
    values = _prepare_run_values(record)
    with _LOCK:
        conn = _connect()
        try:
            existing = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (record["run_id"],)
            ).fetchone()
            if existing:
                patch = {k: v for k, v in values.items() if k != "run_id"}
                if not patch:
                    return
                assignments = ", ".join(f"{k} = :{k}" for k in patch)
                patch["run_id"] = record["run_id"]
                conn.execute(
                    f"UPDATE runs SET {assignments} WHERE run_id = :run_id",
                    patch,
                )
            else:
                insert = {key: None for key in _RUN_FIELDS}
                insert.update(
                    {
                        "run_id": record["run_id"],
                        "task_id": record.get("task_id") or record["run_id"],
                        "fault_scenario": record.get("fault_scenario") or "none",
                        "execution_mode": record.get("execution_mode") or "mock",
                        "status": record.get("status") or "queued",
                        "enable_hitl": 1 if record.get("enable_hitl") else 0,
                        "metadata_json": "{}",
                    }
                )
                insert.update(values)
                if insert.get("metadata_json") is None:
                    insert["metadata_json"] = "{}"
                conn.execute(
                    f"INSERT INTO runs ({', '.join(_RUN_FIELDS)}) "
                    f"VALUES ({', '.join(':' + f for f in _RUN_FIELDS)})",
                    insert,
                )
            conn.commit()
        finally:
            conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    init_db()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            row = conn.execute("SELECT * FROM runs WHERE task_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["enable_hitl"] = bool(data.get("enable_hitl"))
        try:
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
        return data
    finally:
        conn.close()


def list_runs(*, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """Newest runs first."""
    init_db()
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["enable_hitl"] = bool(data.get("enable_hitl"))
            try:
                data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
            except json.JSONDecodeError:
                data["metadata"] = {}
            out.append(data)
        return out
    finally:
        conn.close()


def append_event(run_id: str, kind: str, payload: dict[str, Any] | None = None) -> None:
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO events (run_id, ts, kind, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, _now(), kind, json.dumps(payload or {})),
            )
            conn.commit()
        finally:
            conn.close()


def list_events(run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, ts, kind, payload_json FROM events WHERE run_id = ? AND id > ? ORDER BY id",
            (run_id, after_id),
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out
    finally:
        conn.close()


def replace_observations(run_id: str, observations: list[dict[str, Any]]) -> None:
    init_db()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute("DELETE FROM observations WHERE run_id = ?", (run_id,))
            for obs in observations:
                oid = str(obs.get("id") or uuid4())
                conn.execute(
                    "INSERT INTO observations (id, run_id, payload_json) VALUES (?, ?, ?)",
                    (oid, run_id, json.dumps(obs)),
                )
            conn.commit()
        finally:
            conn.close()


def add_observation(run_id: str, observation: dict[str, Any]) -> None:
    init_db()
    oid = str(observation.get("id") or uuid4())
    observation = {**observation, "id": oid}
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO observations (id, run_id, payload_json) VALUES (?, ?, ?)",
                (oid, run_id, json.dumps(observation)),
            )
            conn.commit()
        finally:
            conn.close()


def list_observations(run_id: str) -> list[dict[str, Any]]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT payload_json FROM observations WHERE run_id = ?", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            try:
                out.append(json.loads(row["payload_json"]))
            except json.JSONDecodeError:
                continue
        return out
    finally:
        conn.close()


def new_ids() -> tuple[str, str]:
    """Return (task_id, run_id)."""
    task_id = str(uuid4())
    run_id = str(uuid4())
    return task_id, run_id
