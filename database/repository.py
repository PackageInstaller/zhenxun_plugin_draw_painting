from __future__ import annotations

import atexit
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import os
import queue
import sqlite3
from threading import Lock
import time
from typing import Any, ParamSpec, TypeVar

try:
    import ujson as json
except ImportError:
    import json

from zhenxun.services.log import logger

P = ParamSpec("P")
R = TypeVar("R")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "record.db")


class DatabaseError(Exception):
    """Base exception for plugin persistence errors."""

    def __init__(
        self,
        message: str,
        error_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or 5000
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        message = f"数据库错误 [{self.error_code}]: {self.message}"
        if self.details:
            message += f"\n详细信息: {self.details!s}"
        return message


class ConnectionError(DatabaseError):
    """Raised when a pooled SQLite connection cannot be acquired."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, 5001, details)


class QueryError(DatabaseError):
    """Raised when a SQLite statement or transaction fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, 5002, details)


class TransactionError(DatabaseError):
    """Reserved for explicit multi-step transaction failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, 5003, details)


class ConnectionPool:
    """Small thread-safe SQLite connection pool."""

    def __init__(self, db_path: str, max_connections: int = 5) -> None:
        self.max_connections = max_connections
        self.connections: queue.Queue[sqlite3.Connection] = queue.Queue(max_connections)
        self.lock = Lock()
        for _ in range(max_connections):
            connection = sqlite3.connect(db_path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            self.connections.put(connection)

    def get_connection(self) -> sqlite3.Connection:
        try:
            return self.connections.get(timeout=5)
        except queue.Empty as exc:
            raise ConnectionError(
                "无法获取数据库连接，连接池已满",
                {"最大连接数": self.max_connections},
            ) from exc

    def return_connection(self, connection: sqlite3.Connection) -> None:
        self.connections.put(connection)

    def close_all(self) -> None:
        with self.lock:
            while not self.connections.empty():
                self.connections.get_nowait().close()


def retry_on_error(
    max_retries: int = 3, delay: float = 0.1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry synchronous repository operations after a database error."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: DatabaseError | None = None
            for attempt in range(max_retries):
                try:
                    return function(*args, **kwargs)
                except DatabaseError as exc:
                    last_error = exc
                    if attempt + 1 < max_retries:
                        time.sleep(delay)
            if last_error is None:
                raise RuntimeError("max_retries must be greater than zero")
            raise last_error

        return wrapper

    return decorator


class DatabaseHandler:
    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.pool = ConnectionPool(db_path)
        self.setup_database()

    @contextmanager
    def get_db_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        connection = self.pool.get_connection()
        cursor = connection.cursor()
        try:
            yield cursor
            connection.commit()
        except Exception as exc:
            connection.rollback()
            logger.error(f"Database error: {exc}")
            raise QueryError("数据库操作失败", {"错误信息": str(exc)}) from exc
        finally:
            cursor.close()
            self.pool.return_connection(connection)

    def setup_database(self) -> None:
        """Create the plugin schema and supporting indexes."""

        with self.get_db_cursor() as cursor:
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS draw_record (
                    user_id TEXT,
                    card_name TEXT,
                    times TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    card_type TEXT,
                    PRIMARY KEY (user_id, card_type)
                );

                CREATE TABLE IF NOT EXISTS renamed_record (
                    user_id TEXT,
                    old_image_name TEXT,
                    new_image_name TEXT,
                    rename_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, old_image_name)
                );

                CREATE TABLE IF NOT EXISTS draw_history_record (
                    user_id TEXT,
                    card_type TEXT,
                    history TEXT,
                    total_count INTEGER DEFAULT 0,
                    last_draw_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, card_type)
                );

                CREATE TABLE IF NOT EXISTS user_info (
                    user_id TEXT PRIMARY KEY,
                    read_help INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_card_name
                    ON draw_record (card_name, card_type);
                """
            )

            # `CREATE TABLE IF NOT EXISTS` does not upgrade databases created by
            # older plugin versions. Add the timestamp column before recreating
            # the history index so existing installations migrate in place.
            history_columns = {
                str(row["name"])
                for row in cursor.execute(
                    "PRAGMA table_info(draw_history_record)"
                ).fetchall()
            }
            if "last_draw_time" not in history_columns:
                cursor.execute(
                    "ALTER TABLE draw_history_record "
                    "ADD COLUMN last_draw_time TIMESTAMP"
                )
                cursor.execute(
                    "UPDATE draw_history_record "
                    "SET last_draw_time = CURRENT_TIMESTAMP "
                    "WHERE last_draw_time IS NULL"
                )
                logger.info("已迁移抽取历史数据库：新增 last_draw_time 字段")

            history_index_columns = tuple(
                str(row["name"])
                for row in cursor.execute(
                    "PRAGMA index_info(idx_draw_history)"
                ).fetchall()
            )
            expected_index_columns = ("user_id", "card_type", "last_draw_time")
            if history_index_columns != expected_index_columns:
                cursor.execute("DROP INDEX IF EXISTS idx_draw_history")
                cursor.execute(
                    "CREATE INDEX idx_draw_history ON draw_history_record "
                    "(user_id, card_type, last_draw_time)"
                )

    @retry_on_error()
    def update_draw_record(
        self, user_id: str, card_name: str, card_type: str = "Wife"
    ) -> None:
        """Update the user's current draw without changing draw history."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO draw_record
                    (user_id, card_name, times, card_type)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, card_name, timestamp, card_type),
            )

    @retry_on_error()
    def get_user_info(self, user_id: str) -> dict[str, Any]:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO user_info (user_id, read_help) VALUES (?, 0)",
                (user_id,),
            )
            cursor.execute(
                "SELECT user_id, read_help FROM user_info WHERE user_id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else {"user_id": user_id, "read_help": 0}

    @retry_on_error()
    def mark_help_as_read(self, user_id: str) -> None:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE user_info SET read_help = 1 WHERE user_id = ?",
                (user_id,),
            )

    @retry_on_error()
    def update_renamed_record(
        self, user_id: str, old_image_name: str, new_image_name: str
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR IGNORE INTO renamed_record
                    (user_id, old_image_name, new_image_name, rename_time)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, old_image_name, new_image_name, timestamp),
            )

    @retry_on_error()
    def get_renamed_images(self) -> list[tuple[str, str]]:
        with self.get_db_cursor() as cursor:
            cursor.execute("SELECT old_image_name, new_image_name FROM renamed_record")
            return [
                (row["old_image_name"], row["new_image_name"])
                for row in cursor.fetchall()
            ]

    @retry_on_error()
    def get_last_trigger_time(
        self, user_id: str, card_type: str = "Wife"
    ) -> datetime | None:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "SELECT times FROM draw_record WHERE user_id = ? AND card_type = ?",
                (user_id, card_type),
            )
            row = cursor.fetchone()
            if not row or not row["times"]:
                return None
            return datetime.strptime(row["times"], "%Y-%m-%d %H:%M:%S")

    @retry_on_error()
    def get_draw_history_record(
        self, user_id: str, card_type: str = "Wife"
    ) -> tuple[list[dict[str, str]], int]:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT history, total_count FROM draw_history_record
                WHERE user_id = ? AND card_type = ?
                """,
                (user_id, card_type),
            )
            row = cursor.fetchone()
            if not row:
                return [], 0
            history = json.loads(row["history"]) if row["history"] else []
            return history, row["total_count"]

    @retry_on_error()
    def get_card_name(self, user_id: str, card_type: str = "Wife") -> str | None:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "SELECT card_name FROM draw_record WHERE user_id = ? AND card_type = ?",
                (user_id, card_type),
            )
            row = cursor.fetchone()
            return row["card_name"] if row else None

    @retry_on_error()
    def get_all_selected_wives_or_husbands(self, card_type: str = "Wife") -> list[str]:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "SELECT card_name FROM draw_record WHERE card_type = ?",
                (card_type,),
            )
            return [row["card_name"] for row in cursor.fetchall()]

    @retry_on_error()
    def get_selected_wives_or_husbands_by_game(
        self, game_name: str, card_type: str = "Wife"
    ) -> list[str]:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT card_name FROM draw_record
                WHERE LOWER(card_name) LIKE ? AND card_type = ?
                """,
                (f"{game_name.lower()}_%", card_type),
            )
            return [row["card_name"] for row in cursor.fetchall()]

    @retry_on_error()
    def delete_draw_record(self, user_id: str, card_type: str = "Wife") -> None:
        with self.get_db_cursor() as cursor:
            cursor.execute(
                "DELETE FROM draw_record WHERE user_id = ? AND card_type = ?",
                (user_id, card_type),
            )

    @retry_on_error()
    def log_draw_history_record(
        self, user_id: str, card_name: str, card_type: str = "Wife"
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_db_cursor() as cursor:
            cursor.execute(
                """
                SELECT history, total_count FROM draw_history_record
                WHERE user_id = ? AND card_type = ?
                """,
                (user_id, card_type),
            )
            row = cursor.fetchone()
            history = json.loads(row["history"]) if row and row["history"] else []
            total_count = row["total_count"] + 1 if row else 1
            history.append({"timestamp": timestamp, "card_name": card_name})
            cursor.execute(
                """
                INSERT OR REPLACE INTO draw_history_record
                    (user_id, card_type, history, total_count, last_draw_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    card_type,
                    json.dumps(history, ensure_ascii=False),
                    total_count,
                    timestamp,
                ),
            )

    def close(self) -> None:
        self.pool.close_all()


db_handler = DatabaseHandler()
atexit.register(db_handler.close)
