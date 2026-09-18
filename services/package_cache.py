from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time


@dataclass(frozen=True)
class CachedPackage:
    message_id: int
    reply_seq: int | None
    object_key: str
    expires_at: float


@dataclass(frozen=True)
class SharedPackage:
    game_name: str
    object_key: str
    husbands_count: int
    wives_count: int
    expires_at: float

    @property
    def total_count(self) -> int:
        return self.husbands_count + self.wives_count


class PaintingPackageStore:
    """Persistent package/message cache plus an OSS object cleanup queue."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path, timeout=30)

    def initialize(self) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS painting_package_messages (
                    scope_key TEXT NOT NULL,
                    game_key TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    reply_seq INTEGER,
                    object_key TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (scope_key, game_key)
                )
                """
            )
            columns = {
                str(row[1])
                for row in db.execute(
                    "PRAGMA table_info(painting_package_messages)"
                ).fetchall()
            }
            if "reply_seq" not in columns:
                db.execute(
                    """
                    ALTER TABLE painting_package_messages
                    ADD COLUMN reply_seq INTEGER
                    """
                )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS painting_package_objects (
                    object_key TEXT PRIMARY KEY,
                    expires_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS painting_package_shared (
                    game_key TEXT PRIMARY KEY,
                    game_name TEXT NOT NULL,
                    object_key TEXT NOT NULL,
                    husbands_count INTEGER NOT NULL,
                    wives_count INTEGER NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_painting_package_expiry
                ON painting_package_objects(expires_at)
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_painting_package_shared_expiry
                ON painting_package_shared(expires_at)
                """
            )

    def get_valid(
        self,
        scope_key: str,
        game_key: str,
        now: float | None = None,
    ) -> CachedPackage | None:
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            row = db.execute(
                """
                SELECT message_id, reply_seq, object_key, expires_at
                FROM painting_package_messages
                WHERE scope_key=? AND game_key=? AND expires_at>?
                """,
                (scope_key, game_key, now),
            ).fetchone()
        if row is None:
            return None
        return CachedPackage(
            message_id=int(row[0]),
            reply_seq=int(row[1]) if row[1] is not None else None,
            object_key=str(row[2]),
            expires_at=float(row[3]),
        )

    def get_shared_valid(
        self,
        game_key: str,
        now: float | None = None,
    ) -> SharedPackage | None:
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            row = db.execute(
                """
                SELECT game_name, object_key, husbands_count,
                       wives_count, expires_at
                FROM painting_package_shared
                WHERE game_key=? AND expires_at>?
                """,
                (game_key, now),
            ).fetchone()
        if row is None:
            return None
        return SharedPackage(
            game_name=str(row[0]),
            object_key=str(row[1]),
            husbands_count=int(row[2]),
            wives_count=int(row[3]),
            expires_at=float(row[4]),
        )

    def save_shared_package(
        self,
        game_key: str,
        game_name: str,
        object_key: str,
        husbands_count: int,
        wives_count: int,
        expires_at: float,
    ) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO painting_package_objects(object_key, expires_at)
                VALUES (?, ?)
                ON CONFLICT(object_key) DO UPDATE SET
                    expires_at=excluded.expires_at
                """,
                (object_key, expires_at),
            )
            db.execute(
                """
                INSERT INTO painting_package_shared(
                    game_key, game_name, object_key,
                    husbands_count, wives_count, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_key) DO UPDATE SET
                    game_name=excluded.game_name,
                    object_key=excluded.object_key,
                    husbands_count=excluded.husbands_count,
                    wives_count=excluded.wives_count,
                    expires_at=excluded.expires_at
                """,
                (
                    game_key,
                    game_name,
                    object_key,
                    husbands_count,
                    wives_count,
                    expires_at,
                ),
            )

    def promote_legacy_package(
        self,
        game_key: str,
        game_name: str,
        husbands_count: int,
        wives_count: int,
        now: float | None = None,
    ) -> SharedPackage | None:
        """Promote a pre-shared-cache group entry without repacking it."""
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            row = db.execute(
                """
                SELECT object_key, expires_at
                FROM painting_package_messages
                WHERE game_key=? AND expires_at>?
                ORDER BY expires_at DESC
                LIMIT 1
                """,
                (game_key, now),
            ).fetchone()
        if row is None:
            return None
        package = SharedPackage(
            game_name=game_name,
            object_key=str(row[0]),
            husbands_count=husbands_count,
            wives_count=wives_count,
            expires_at=float(row[1]),
        )
        self.save_shared_package(
            game_key,
            package.game_name,
            package.object_key,
            package.husbands_count,
            package.wives_count,
            package.expires_at,
        )
        return package

    def track_object(self, object_key: str, expires_at: float) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO painting_package_objects(object_key, expires_at)
                VALUES (?, ?)
                ON CONFLICT(object_key) DO UPDATE SET
                    expires_at=excluded.expires_at
                """,
                (object_key, expires_at),
            )

    def save_message(
        self,
        scope_key: str,
        game_key: str,
        message_id: int,
        reply_seq: int | None,
        object_key: str,
        expires_at: float,
    ) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO painting_package_objects(object_key, expires_at)
                VALUES (?, ?)
                ON CONFLICT(object_key) DO UPDATE SET
                    expires_at=excluded.expires_at
                """,
                (object_key, expires_at),
            )
            db.execute(
                """
                INSERT INTO painting_package_messages(
                    scope_key, game_key, message_id, reply_seq,
                    object_key, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_key, game_key) DO UPDATE SET
                    message_id=excluded.message_id,
                    reply_seq=excluded.reply_seq,
                    object_key=excluded.object_key,
                    expires_at=excluded.expires_at
                """,
                (
                    scope_key,
                    game_key,
                    message_id,
                    reply_seq,
                    object_key,
                    expires_at,
                ),
            )

    def update_reply_seq(
        self,
        scope_key: str,
        game_key: str,
        reply_seq: int,
    ) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                UPDATE painting_package_messages
                SET reply_seq=?
                WHERE scope_key=? AND game_key=?
                """,
                (reply_seq, scope_key, game_key),
            )

    def invalidate_message(self, scope_key: str, game_key: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                """
                DELETE FROM painting_package_messages
                WHERE scope_key=? AND game_key=?
                """,
                (scope_key, game_key),
            )

    def list_expired_objects(self, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            rows = db.execute(
                """
                SELECT object_key FROM painting_package_objects
                WHERE expires_at<=?
                """,
                (now,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def mark_object_deleted(self, object_key: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "DELETE FROM painting_package_messages WHERE object_key=?",
                (object_key,),
            )
            db.execute(
                "DELETE FROM painting_package_shared WHERE object_key=?",
                (object_key,),
            )
            db.execute(
                "DELETE FROM painting_package_objects WHERE object_key=?",
                (object_key,),
            )

    def purge_expired_messages(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            db.execute(
                "DELETE FROM painting_package_messages WHERE expires_at<=?",
                (now,),
            )
