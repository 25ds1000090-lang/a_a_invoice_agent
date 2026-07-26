from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.getenv("DATABASE_PATH", "/tmp/a2a_invoice_agent.sqlite3")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level="IMMEDIATE")
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _initialise(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    principal_hash TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_principal_created
                ON tasks(principal_hash, created_at);

                CREATE TABLE IF NOT EXISTS idempotency (
                    principal_hash TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    PRIMARY KEY(principal_hash, message_id)
                );

                CREATE TABLE IF NOT EXISTS decision_cache (
                    package_hash TEXT PRIMARY KEY,
                    decision_json TEXT NOT NULL
                );
                """
            )

    def get_idempotency(
        self, conn: sqlite3.Connection, principal_hash: str, message_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT principal_hash, message_id, message_hash, task_id
            FROM idempotency
            WHERE principal_hash = ? AND message_id = ?
            """,
            (principal_hash, message_id),
        ).fetchone()

    def insert_idempotency(
        self,
        conn: sqlite3.Connection,
        principal_hash: str,
        message_id: str,
        message_hash: str,
        task_id: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO idempotency(principal_hash, message_id, message_hash, task_id)
            VALUES (?, ?, ?, ?)
            """,
            (principal_hash, message_id, message_hash, task_id),
        )

    def insert_task(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        context_id: str,
        principal_hash: str,
        batch_id: str,
        state: str,
        task_json: dict[str, Any],
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO tasks(
                task_id, context_id, principal_hash, batch_id, state,
                task_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                context_id,
                principal_hash,
                batch_id,
                state,
                canonical_json(task_json),
                now,
                now,
            ),
        )

    def update_task(
        self,
        conn: sqlite3.Connection,
        *,
        task_id: str,
        principal_hash: str,
        expected_states: tuple[str, ...],
        state: str,
        task_json: dict[str, Any],
        now: str,
    ) -> bool:
        placeholders = ",".join("?" for _ in expected_states)
        cur = conn.execute(
            f"""
            UPDATE tasks
            SET state = ?, task_json = ?, updated_at = ?
            WHERE task_id = ?
              AND principal_hash = ?
              AND state IN ({placeholders})
            """,
            (
                state,
                canonical_json(task_json),
                now,
                task_id,
                principal_hash,
                *expected_states,
            ),
        )
        return cur.rowcount == 1

    def get_task_for_owner(
        self, conn: sqlite3.Connection, task_id: str, principal_hash: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT *
            FROM tasks
            WHERE task_id = ? AND principal_hash = ?
            """,
            (task_id, principal_hash),
        ).fetchone()

    def task_exists(self, conn: sqlite3.Connection, task_id: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            is not None
        )

    def list_tasks(self, principal_hash: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT task_json
                FROM tasks
                WHERE principal_hash = ?
                ORDER BY created_at ASC
                """,
                (principal_hash,),
            ).fetchall()
        return [json.loads(row["task_json"]) for row in rows]

    def get_cached_decision(
        self, conn: sqlite3.Connection, package_hash: str
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT decision_json FROM decision_cache WHERE package_hash = ?",
            (package_hash,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["decision_json"])

    def put_cached_decision(
        self,
        conn: sqlite3.Connection,
        package_hash: str,
        decision: dict[str, Any],
    ) -> None:
        conn.execute(
            """
            INSERT INTO decision_cache(package_hash, decision_json)
            VALUES (?, ?)
            ON CONFLICT(package_hash) DO NOTHING
            """,
            (package_hash, canonical_json(decision)),
        )
