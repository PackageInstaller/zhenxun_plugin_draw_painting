from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time
from typing import Any

from zhenxun.services.log import logger

from .archive_policy import target_library
from .feature_paths import is_absolute_key, normalize_image_path
from .model import GENERAL_THRESHOLD, MODEL_VERSION, TagPrediction
from .vision.resources import is_memory_error
from .vision.types import MULTIPLE_TAGS

ERROR_RETRY_SECONDS = 6 * 60 * 60
MISSING_RETENTION_SECONDS = 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class StoredImageFeature:
    path: str
    content_sha256: str
    library: str
    file_size: int
    mtime_ns: int
    model_version: str
    width: int
    height: int
    male_probability: float
    female_probability: float
    general_tags: dict[str, float]
    character_tags: dict[str, float]
    rating_tags: dict[str, float]
    processed_at: float
    subject_tags: dict[str, float]
    subject_status: str
    subject_gender: str
    analysis: dict[str, Any]

    def to_prediction(self) -> TagPrediction:
        return TagPrediction(
            width=self.width,
            height=self.height,
            male_probability=self.male_probability,
            female_probability=self.female_probability,
            general_tags=self.general_tags,
            character_tags=self.character_tags,
            rating_tags=self.rating_tags,
            model_version=self.model_version,
            subject_tags=self.subject_tags,
            subject_status=self.subject_status,
            subject_gender=self.subject_gender,
            analysis=self.analysis,
        )


class ImageFeatureStore:
    """Persistent, versioned image traits without modifying image pixels."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._initialized = False
        self._initialize_lock = Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._initialize_lock:
            if not self._initialized:
                self._initialize_schema()
                self._initialized = True

    def _initialize_schema(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_absolute_paths()
        migrated = False
        with self._connect() as connection:
            migrated = self._migrate_legacy_schema(connection)
            self._create_schema(connection)
            self._ensure_subject_schema(connection)
            self._migrate_portable_paths(connection)
            self._refresh_subject_selection(connection)
            self._refresh_sliced_detection(connection)
            self._backfill_tag_index(connection)
        if migrated or self._needs_compaction():
            try:
                self._compact_database()
            except sqlite3.Error as exc:
                # The compact schema is already committed. A concurrent worker may
                # temporarily prevent VACUUM; a later startup will retry it.
                logger.warning(
                    "立绘特征数据库空间回收暂未完成，" f"将在下次启动重试: {exc}"
                )

    def _backup_absolute_paths(self) -> None:
        if not self.database_path.exists():
            return
        backup = self.database_path.with_suffix(".pre_relative_paths.bak")
        if backup.exists():
            return
        with closing(sqlite3.connect(self.database_path)) as source:
            if not source.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'image_features'"
            ).fetchone():
                return
            if not any(
                is_absolute_key(row[0])
                for row in source.execute("SELECT path FROM image_features")
            ):
                return
            temporary = backup.with_suffix(".bak.tmp")
            with closing(sqlite3.connect(temporary)) as destination:
                source.backup(destination)
            temporary.replace(backup)
        logger.info(f"相对路径迁移前已备份特征数据库: {backup}")

    @staticmethod
    def _migrate_portable_paths(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS feature_metadata "
            "(key TEXT PRIMARY KEY, value TEXT)"
        )
        if connection.execute(
            "SELECT 1 FROM feature_metadata WHERE key = 'relative_paths_v1'"
        ).fetchone():
            return
        # Prefer a current, successfully tagged result over missing/legacy rows
        # from the previous installation; retain the winning row's integer ID.
        rows = connection.execute(
            "SELECT id, path, moved_from FROM image_features ORDER BY "
            "(model_version = ?) DESC, (status = 'tagged') DESC, "
            "processed_at DESC, id DESC",
            (MODEL_VERSION,),
        ).fetchall()
        groups: dict[str, list] = {}
        for row in rows:
            groups.setdefault(normalize_image_path(row["path"]), []).append(row)
        for key, group in groups.items():
            winner = group[0]
            for duplicate in group[1:]:
                connection.execute(
                    "DELETE FROM image_feature_tags WHERE image_id = ?",
                    (duplicate["id"],),
                )
                connection.execute(
                    "DELETE FROM image_features WHERE id = ?", (duplicate["id"],)
                )
            previous = winner["moved_from"]
            connection.execute(
                "UPDATE image_features SET path = ?, moved_from = ? WHERE id = ?",
                (
                    key,
                    normalize_image_path(previous) if previous else None,
                    winner["id"],
                ),
            )
        connection.execute(
            "INSERT INTO feature_metadata VALUES ('relative_paths_v1', '1')"
        )

    @staticmethod
    def _refresh_subject_selection(connection: sqlite3.Connection) -> None:
        if connection.execute(
            "SELECT 1 FROM feature_metadata WHERE key = 'subject_selection_v3'"
        ).fetchone():
            return
        # Only previous weak/overlapping crop decisions need the new geometry
        # pass. Keep all other same-model cache entries, including relocated ones.
        connection.execute(
            """
            UPDATE image_features SET
                status = CASE WHEN status = 'tagged' THEN 'pending' ELSE status END,
                subject_status = 'legacy'
            WHERE model_version = ? AND library IN ('wives', 'husbands')
              AND status IN ('tagged', 'missing', 'pending') AND (
                subject_status = 'weak_detection' OR
                CASE WHEN json_valid(analysis_json)
                    THEN json_extract(analysis_json, '$.selection_status')
                END IN ('tentative_detection', 'overlapping_people')
            )
            """,
            (MODEL_VERSION,),
        )
        connection.execute(
            "DELETE FROM image_feature_tags WHERE image_id IN "
            "(SELECT id FROM image_features WHERE subject_status = 'legacy')"
        )
        connection.execute(
            "INSERT INTO feature_metadata VALUES ('subject_selection_v3', '1')"
        )

    @staticmethod
    def _refresh_sliced_detection(connection: sqlite3.Connection) -> None:
        if connection.execute(
            "SELECT 1 FROM feature_metadata WHERE key = 'sliced_detection_v1'"
        ).fetchone():
            return
        connection.execute(
            """
            UPDATE image_features SET
                status = CASE WHEN status = 'tagged' THEN 'pending' ELSE status END,
                subject_status = 'legacy'
            WHERE model_version = ? AND library IN ('wives', 'husbands')
              AND status IN ('tagged', 'pending', 'missing')
              AND (width >= 1024 OR height >= 1024)
              AND (subject_status IN ('no_person', 'weak_detection') OR
                CASE WHEN json_valid(analysis_json)
                    THEN json_extract(analysis_json, '$.selection_status')
                END IN ('no_person', 'weak_detection', 'tentative_detection'))
            """,
            (MODEL_VERSION,),
        )
        connection.execute(
            "DELETE FROM image_feature_tags WHERE image_id IN "
            "(SELECT id FROM image_features WHERE subject_status = 'legacy')"
        )
        connection.execute(
            "INSERT INTO feature_metadata VALUES ('sliced_detection_v1', '1')"
        )

    @staticmethod
    def _ensure_subject_schema(connection: sqlite3.Connection) -> None:
        columns = ImageFeatureStore._table_columns(connection, "image_features")
        additions = {
            "subject_tags_json": "TEXT NOT NULL DEFAULT '{}'",
            "subject_status": "TEXT NOT NULL DEFAULT 'legacy'",
            "subject_gender": "TEXT NOT NULL DEFAULT 'unknown'",
            "analysis_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE image_features ADD COLUMN {name} {definition}"
                )
        if "subject_status" not in columns:
            # Existing rows describe entire scenes, never reinterpret them as
            # subject tags. Their JSON remains available during re-indexing.
            connection.execute("DELETE FROM image_feature_tags")

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _table_columns(
        connection: sqlite3.Connection,
        table: str,
    ) -> set[str]:
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _migrate_legacy_schema(self, connection: sqlite3.Connection) -> bool:
        """Replace path-heavy tag rows with compact integer image IDs once."""

        if not self._table_exists(connection, "image_features"):
            return False
        feature_columns = self._table_columns(connection, "image_features")
        if "id" in feature_columns:
            return self._migrate_legacy_tag_table(connection)

        logger.warning(
            "检测到旧版立绘特征数据库，正在迁移为整数 ID 索引；"
            "首次启动可能需要几分钟，请勿强制结束进程..."
        )
        connection.execute("DROP TABLE IF EXISTS image_features_migrating")
        connection.execute("DROP TABLE IF EXISTS image_feature_tags_migrating")
        connection.execute(
            """
            CREATE TABLE image_features_migrating (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                content_sha256 TEXT,
                library TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                male_probability REAL,
                female_probability REAL,
                general_tags_json TEXT NOT NULL DEFAULT '{}',
                character_tags_json TEXT NOT NULL DEFAULT '{}',
                rating_tags_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                error TEXT,
                processed_at REAL NOT NULL,
                moved_from TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO image_features_migrating (
                path, content_sha256, library, file_size, mtime_ns,
                model_version, width, height, male_probability,
                female_probability, general_tags_json,
                character_tags_json, rating_tags_json, status, error,
                processed_at, moved_from
            )
            SELECT
                path, content_sha256, library, file_size, mtime_ns,
                model_version, width, height, male_probability,
                female_probability, general_tags_json,
                character_tags_json, rating_tags_json, status, error,
                processed_at, moved_from
            FROM image_features
            """
        )
        connection.execute(
            """
            CREATE TABLE image_feature_tags_migrating (
                image_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (image_id, tag)
            ) WITHOUT ROWID
            """
        )
        if self._table_exists(connection, "image_feature_tags"):
            tag_columns = self._table_columns(connection, "image_feature_tags")
            if "path" in tag_columns:
                connection.execute(
                    """
                    INSERT INTO image_feature_tags_migrating(image_id, tag, score)
                    SELECT features.id, tags.tag, tags.score
                    FROM image_feature_tags AS tags
                    JOIN image_features_migrating AS features
                      ON features.path = tags.path
                    """
                )
            elif "image_id" in tag_columns:
                connection.execute(
                    """
                    INSERT INTO image_feature_tags_migrating(image_id, tag, score)
                    SELECT image_id, tag, score FROM image_feature_tags
                    """
                )

        connection.execute("DROP TABLE IF EXISTS image_feature_tags")
        connection.execute("DROP TABLE image_features")
        connection.execute(
            "ALTER TABLE image_features_migrating RENAME TO image_features"
        )
        connection.execute(
            "ALTER TABLE image_feature_tags_migrating RENAME TO image_feature_tags"
        )
        logger.info("立绘特征数据库表迁移完成，正在建立查询索引...")
        return True

    def _migrate_legacy_tag_table(self, connection: sqlite3.Connection) -> bool:
        if not self._table_exists(connection, "image_feature_tags"):
            return False
        tag_columns = self._table_columns(connection, "image_feature_tags")
        if "image_id" in tag_columns:
            return False
        if "path" not in tag_columns:
            raise RuntimeError("无法识别 image_feature_tags 的数据库结构。")

        connection.execute("DROP TABLE IF EXISTS image_feature_tags_migrating")
        connection.execute(
            """
            CREATE TABLE image_feature_tags_migrating (
                image_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (image_id, tag)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            INSERT INTO image_feature_tags_migrating(image_id, tag, score)
            SELECT features.id, tags.tag, tags.score
            FROM image_feature_tags AS tags
            JOIN image_features AS features ON features.path = tags.path
            """
        )
        connection.execute("DROP TABLE image_feature_tags")
        connection.execute(
            "ALTER TABLE image_feature_tags_migrating RENAME TO image_feature_tags"
        )
        return True

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_features (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                content_sha256 TEXT,
                library TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                male_probability REAL,
                female_probability REAL,
                general_tags_json TEXT NOT NULL DEFAULT '{}',
                character_tags_json TEXT NOT NULL DEFAULT '{}',
                rating_tags_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                error TEXT,
                processed_at REAL NOT NULL,
                moved_from TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_feature_tags (
                image_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (image_id, tag)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_features_hash_model
            ON image_features(content_sha256, model_version, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_features_filter
            ON image_features(library, status, model_version, id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_feature_tags_lookup
            ON image_feature_tags(tag, score, image_id)
            """
        )

    def _compact_database(self) -> None:
        before = self.database_path.stat().st_size
        logger.info("正在压缩立绘特征数据库并回收旧路径索引空间...")
        connection = sqlite3.connect(self.database_path, timeout=300)
        try:
            connection.execute("PRAGMA busy_timeout = 300000")
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")
            connection.execute("PRAGMA optimize")
        finally:
            connection.close()
        after = self.database_path.stat().st_size
        logger.info(
            f"立绘特征数据库优化完成: {before / 1024**2:.1f} MiB -> "
            f"{after / 1024**2:.1f} MiB"
        )

    def _needs_compaction(self) -> bool:
        connection = sqlite3.connect(self.database_path, timeout=30)
        try:
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            free_pages = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
        finally:
            connection.close()
        return free_pages >= 10_000 and free_pages * 4 >= page_count

    def _backfill_tag_index(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT features.id, features.subject_tags_json, features.general_tags_json
            FROM image_features AS features
            WHERE features.status = 'tagged'
              AND features.model_version = ?
              AND features.subject_status = 'confident'
              AND NOT EXISTS (
                  SELECT 1 FROM image_feature_tags AS tags
                  WHERE tags.image_id = features.id
              )
            """,
            (MODEL_VERSION,),
        )
        pending: list[tuple[int, str, float]] = []
        for row in rows:
            try:
                tags = self._index_tags(
                    self._decode_tags(str(row["subject_tags_json"])),
                    self._decode_tags(str(row["general_tags_json"])),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            pending.extend((int(row["id"]), tag, score) for tag, score in tags.items())
            if len(pending) >= 10_000:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO image_feature_tags(image_id, tag, score)
                    VALUES (?, ?, ?)
                    """,
                    pending,
                )
                pending.clear()
        if pending:
            connection.executemany(
                """
                INSERT OR REPLACE INTO image_feature_tags(image_id, tag, score)
                VALUES (?, ?, ?)
                """,
                pending,
            )

    @staticmethod
    def _index_tags(
        subject_tags: dict[str, float], scene_tags: dict[str, float]
    ) -> dict[str, float]:
        # Only explicit person-count filters refer to the whole scene. Hair,
        # clothing, anatomy, etc. must always come from the same selected crop.
        return {
            **subject_tags,
            **{
                tag: score
                for tag, score in scene_tags.items()
                if tag in MULTIPLE_TAGS and score >= 0.6
            },
        }

    @staticmethod
    def _decode_tags(raw: str) -> dict[str, float]:
        values = json.loads(raw)
        return {str(name): float(score) for name, score in values.items()}

    @classmethod
    def _row_to_feature(cls, row: sqlite3.Row) -> StoredImageFeature:
        return StoredImageFeature(
            path=str(row["path"]),
            content_sha256=str(row["content_sha256"]),
            library=str(row["library"]),
            file_size=int(row["file_size"]),
            mtime_ns=int(row["mtime_ns"]),
            model_version=str(row["model_version"]),
            width=int(row["width"]),
            height=int(row["height"]),
            male_probability=float(row["male_probability"]),
            female_probability=float(row["female_probability"]),
            general_tags=cls._decode_tags(row["general_tags_json"]),
            character_tags=cls._decode_tags(row["character_tags_json"]),
            rating_tags=cls._decode_tags(row["rating_tags_json"]),
            processed_at=float(row["processed_at"]),
            subject_tags=cls._decode_tags(row["subject_tags_json"]),
            subject_status=str(row["subject_status"]),
            subject_gender=str(row["subject_gender"]),
            analysis=json.loads(row["analysis_json"]),
        )

    def get_cached(
        self,
        path: str | Path,
        file_size: int,
        mtime_ns: int,
    ) -> StoredImageFeature | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM image_features
                WHERE path = ? AND file_size = ? AND mtime_ns = ?
                  AND model_version = ? AND status = 'tagged'
                """,
                (normalize_image_path(path), file_size, mtime_ns, MODEL_VERSION),
            ).fetchone()
        return self._row_to_feature(row) if row else None

    def needs_processing(
        self,
        path: str | Path,
        file_size: int,
        mtime_ns: int,
        now: float | None = None,
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT file_size, mtime_ns, model_version, status, processed_at
                FROM image_features WHERE path = ?
                """,
                (normalize_image_path(path),),
            ).fetchone()
        if row is None:
            return True
        if (
            int(row["file_size"]) != file_size
            or int(row["mtime_ns"]) != mtime_ns
            or str(row["model_version"]) != MODEL_VERSION
        ):
            return True
        if row["status"] == "tagged":
            return False
        if row["status"] == "error":
            current_time = time.time() if now is None else now
            return current_time - float(row["processed_at"]) >= ERROR_RETRY_SECONDS
        return True

    def archive_candidates(self) -> set[str]:
        """Stream current library rows once at startup to review archival policy."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT path, library, subject_status, subject_gender,
                       subject_tags_json, general_tags_json, analysis_json
                FROM image_features
                WHERE model_version = ? AND status = 'tagged'
                  AND library IN ('wives', 'husbands')
                """,
                (MODEL_VERSION,),
            )
            return {
                str(row["path"])
                for row in rows
                if target_library(
                    str(row["library"]),
                    str(row["subject_status"]),
                    str(row["subject_gender"]),
                    self._decode_tags(row["subject_tags_json"]),
                    json.loads(row["analysis_json"]).get("scene_gender"),
                    json.loads(row["analysis_json"]).get("subject_gender"),
                    json.loads(row["analysis_json"]).get("detection"),
                    self._decode_tags(row["general_tags_json"]),
                )
                is not None
            }

    def processing_index(
        self,
    ) -> dict[str, tuple[int, int, str, str, float]]:
        """Load the lightweight index once for a full directory scan."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT path, file_size, mtime_ns, model_version, status,
                       processed_at
                FROM image_features
                """
            ).fetchall()
        return {
            str(row["path"]): (
                int(row["file_size"]),
                int(row["mtime_ns"]),
                str(row["model_version"]),
                str(row["status"]),
                float(row["processed_at"]),
            )
            for row in rows
        }

    def find_by_hash(self, content_sha256: str) -> StoredImageFeature | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM image_features
                WHERE content_sha256 = ? AND model_version = ?
                  AND status IN ('tagged', 'missing', 'pending') AND error IS NULL
                  AND width IS NOT NULL AND subject_status != 'legacy'
                ORDER BY processed_at DESC LIMIT 1
                """,
                (content_sha256, MODEL_VERSION),
            ).fetchone()
        return self._row_to_feature(row) if row else None

    def find_paths_by_tags(
        self,
        library: str,
        required_tags: tuple[str, ...],
        minimum_score: float = GENERAL_THRESHOLD,
    ) -> set[str]:
        if not required_tags:
            return set()
        unique_tags = tuple(dict.fromkeys(required_tags))
        tag_joins = "\n".join(
            f"""
            JOIN image_feature_tags AS tag_{index}
              ON tag_{index}.image_id = features.id
             AND tag_{index}.tag = ?
             AND tag_{index}.score >= ?
            """
            for index, _tag in enumerate(unique_tags)
        )
        tag_values: list[str | float] = []
        for tag in unique_tags:
            tag_values.extend((tag, minimum_score))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT features.path
                FROM image_features AS features
                {tag_joins}
                WHERE features.library = ?
                  AND features.status = 'tagged'
                  AND features.model_version = ?
                  AND features.subject_status = 'confident'
                  AND features.subject_gender = ?
                """,
                (
                    *tag_values,
                    library,
                    MODEL_VERSION,
                    "female" if library == "wives" else "male",
                ),
            ).fetchall()
        return {str(row["path"]) for row in rows}

    @staticmethod
    def _encode_tags(tags: dict[str, float]) -> str:
        return json.dumps(
            {tag: round(score, 4) for tag, score in tags.items()},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def save_prediction(
        self,
        path: str | Path,
        library: str,
        file_size: int,
        mtime_ns: int,
        content_sha256: str,
        prediction: TagPrediction,
        *,
        moved_from: str | Path | None = None,
    ) -> None:
        normalized_path = normalize_image_path(path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_features (
                    path, content_sha256, library, file_size, mtime_ns,
                    model_version, width, height, male_probability,
                    female_probability, general_tags_json,
                    character_tags_json, rating_tags_json, status, error,
                    processed_at, moved_from, subject_tags_json,
                    subject_status, subject_gender, analysis_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tagged', NULL,
                    ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(path) DO UPDATE SET
                    content_sha256 = excluded.content_sha256,
                    library = excluded.library,
                    file_size = excluded.file_size,
                    mtime_ns = excluded.mtime_ns,
                    model_version = excluded.model_version,
                    width = excluded.width,
                    height = excluded.height,
                    male_probability = excluded.male_probability,
                    female_probability = excluded.female_probability,
                    general_tags_json = excluded.general_tags_json,
                    character_tags_json = excluded.character_tags_json,
                    rating_tags_json = excluded.rating_tags_json,
                    status = 'tagged',
                    error = NULL,
                    processed_at = excluded.processed_at,
                    moved_from = excluded.moved_from,
                    subject_tags_json = excluded.subject_tags_json,
                    subject_status = excluded.subject_status,
                    subject_gender = excluded.subject_gender,
                    analysis_json = excluded.analysis_json
                """,
                (
                    normalized_path,
                    content_sha256,
                    library,
                    file_size,
                    mtime_ns,
                    prediction.model_version,
                    prediction.width,
                    prediction.height,
                    prediction.male_probability,
                    prediction.female_probability,
                    self._encode_tags(prediction.general_tags),
                    self._encode_tags(prediction.character_tags),
                    self._encode_tags(prediction.rating_tags),
                    time.time(),
                    normalize_image_path(moved_from) if moved_from else None,
                    self._encode_tags(prediction.subject_tags),
                    prediction.subject_status,
                    prediction.subject_gender,
                    json.dumps(
                        prediction.analysis, ensure_ascii=False, separators=(",", ":")
                    ),
                ),
            )
            connection.execute(
                """
                DELETE FROM image_feature_tags
                WHERE image_id = (SELECT id FROM image_features WHERE path = ?)
                """,
                (normalized_path,),
            )
            image_row = connection.execute(
                "SELECT id FROM image_features WHERE path = ?",
                (normalized_path,),
            ).fetchone()
            if image_row is None:
                raise RuntimeError("保存图片特征后未能取得图片 ID。")
            image_id = int(image_row["id"])
            connection.executemany(
                """
                INSERT INTO image_feature_tags(image_id, tag, score)
                VALUES (?, ?, ?)
                """,
                (
                    (image_id, tag, score)
                    for tag, score in self._index_tags(
                        prediction.subject_tags, prediction.general_tags
                    ).items()
                    if prediction.subject_status == "confident"
                ),
            )

    def retry_resource_errors(self) -> int:
        """Startup repair for legacy OOM errors; never invalidate successful tags."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, error FROM image_features WHERE status = 'error'"
            ).fetchall()
            affected = [(row["id"],) for row in rows if is_memory_error(row["error"])]
            connection.executemany(
                "UPDATE image_features SET status = 'pending' WHERE id = ?",
                affected,
            )
        return len(affected)

    def mark_error(
        self,
        path: str | Path,
        library: str,
        file_size: int,
        mtime_ns: int,
        error: str,
        content_sha256: str | None = None,
    ) -> None:
        normalized_path = normalize_image_path(path)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_features (
                    path, content_sha256, library, file_size, mtime_ns,
                    model_version, status, error, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'error', ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    content_sha256 = excluded.content_sha256,
                    library = excluded.library,
                    file_size = excluded.file_size,
                    mtime_ns = excluded.mtime_ns,
                    model_version = excluded.model_version,
                    status = 'error',
                    error = excluded.error,
                    processed_at = excluded.processed_at
                """,
                (
                    normalized_path,
                    content_sha256,
                    library,
                    file_size,
                    mtime_ns,
                    MODEL_VERSION,
                    error[:2000],
                    time.time(),
                ),
            )
            connection.execute(
                """
                DELETE FROM image_feature_tags
                WHERE image_id = (SELECT id FROM image_features WHERE path = ?)
                """,
                (normalized_path,),
            )

    def mark_missing(self, path: str | Path) -> None:
        normalized_path = normalize_image_path(path)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_features
                SET status = 'missing', processed_at = ?
                WHERE path = ? AND status != 'missing'
                """,
                (time.time(), normalized_path),
            )
            connection.execute(
                """
                DELETE FROM image_feature_tags
                WHERE image_id = (SELECT id FROM image_features WHERE path = ?)
                """,
                (normalized_path,),
            )

    def reconcile_library(
        self,
        library: str,
        signatures: dict[str, tuple[int, int]],
        *,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Called only after a complete successful directory scan.

        Invalidate changed/deleted paths immediately, retain deleted features for
        one day to reuse a same-content rename, then delete both rows and tags.
        User draw/history records live in a different database and are untouched.
        """
        now = time.time() if now is None else now
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TEMP TABLE scan_files (
                    path TEXT PRIMARY KEY, file_size INTEGER, mtime_ns INTEGER
                ) WITHOUT ROWID
                """
            )
            connection.executemany(
                "INSERT INTO scan_files VALUES (?, ?, ?)",
                ((path, *signature) for path, signature in signatures.items()),
            )
            connection.execute(
                """
                UPDATE image_features SET status = 'pending'
                WHERE library = ? AND status = 'tagged' AND EXISTS (
                    SELECT 1 FROM scan_files AS scan
                    WHERE scan.path = image_features.path AND (
                        scan.file_size != image_features.file_size
                        OR scan.mtime_ns != image_features.mtime_ns
                    )
                )
                """,
                (library,),
            )
            missing = connection.execute(
                """
                UPDATE image_features SET status = 'missing', processed_at = ?
                WHERE library = ? AND status != 'missing' AND NOT EXISTS (
                    SELECT 1 FROM scan_files WHERE path = image_features.path
                )
                """,
                (now, library),
            ).rowcount
            connection.execute(
                """
                DELETE FROM image_feature_tags WHERE image_id IN (
                    SELECT id FROM image_features
                    WHERE library = ? AND status != 'tagged'
                )
                """,
                (library,),
            )
            purged = connection.execute(
                """
                DELETE FROM image_features
                WHERE library = ? AND status = 'missing' AND processed_at < ?
                  AND NOT EXISTS (
                    SELECT 1 FROM scan_files WHERE path = image_features.path
                  )
                """,
                (library, now - MISSING_RETENTION_SECONDS),
            ).rowcount
        return missing, purged

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS amount
                FROM image_features GROUP BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["amount"]) for row in rows}


__all__ = [
    "ImageFeatureStore",
    "StoredImageFeature",
    "normalize_image_path",
]
