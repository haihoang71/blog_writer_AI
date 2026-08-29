"""Eval-only ground truth. Production traces must never contain expected_detector."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
# Tests/deployments can override this path. By default artifacts live next to
# the configured DATA_DIR so a Docker volume persists them with SQLite.
GROUND_TRUTH_ROOT: Path | None = None


def _json_root() -> Path:
    if GROUND_TRUTH_ROOT is not None:
        return Path(GROUND_TRUTH_ROOT)
    from config.settings import get_settings

    configured = getattr(get_settings(), "data_dir", None)
    root = Path(str(configured)) if configured else Path("data")
    return root / "fault_ground_truth"


def write_ground_truth(run_id: str, record: dict[str, Any]) -> Path:
    """Write a portable eval artifact, outside SQLite and Langfuse traces."""
    root = _json_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}.json"
    temporary = root / f".{run_id}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)
    return target


def _db_path() -> Path:
    from config.settings import get_settings

    configured = getattr(get_settings(), "data_dir", None)
    root = Path(str(configured)) if configured else Path(__file__).resolve().parent.parent / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ground_truth.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ground_truth (
                    run_id TEXT PRIMARY KEY,
                    eval_run_id TEXT,
                    task_id TEXT,
                    langfuse_trace_id TEXT,
                    scenario TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def record(
    *,
    run_id: str,
    scenario: str,
    payload: dict[str, Any],
    task_id: str | None = None,
    langfuse_trace_id: str | None = None,
    eval_run_id: str | None = None,
) -> None:
    """Persist labels next to the run, never on the Langfuse trace."""
    init()
    with _LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO ground_truth
                (run_id, eval_run_id, task_id, langfuse_trace_id, scenario, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    eval_run_id,
                    task_id,
                    langfuse_trace_id,
                    scenario,
                    json.dumps(payload),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    write_ground_truth(
        run_id,
        {
            "run_id": run_id,
            "eval_run_id": eval_run_id,
            "task_id": task_id,
            "langfuse_trace_id": langfuse_trace_id,
            "scenario": scenario,
            "payload": payload,
        },
    )


def get(run_id: str) -> dict[str, Any] | None:
    init()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM ground_truth WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return data
    finally:
        conn.close()
