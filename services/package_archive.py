from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata
import uuid
import zipfile

from fuzzywuzzy import fuzz
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


@dataclass(frozen=True)
class PaintingArchive:
    path: Path
    game_name: str
    husbands_count: int
    wives_count: int

    @property
    def total_count(self) -> int:
        return self.husbands_count + self.wives_count


def _new_progress() -> Progress:
    """Match the Rich progress style already used by the model downloader."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TextColumn("已用时间:"),
        TimeElapsedColumn(),
        TextColumn("预计剩余:"),
        TimeRemainingColumn(),
    )


def normalize_game_name(name: str) -> str:
    """Normalize punctuation/case while retaining Chinese and Latin letters."""
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return "".join(char for char in normalized if char.isalnum())


def list_game_prefixes(*folders: Path) -> list[str]:
    prefixes: dict[str, str] = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file() or "_" not in path.name:
                continue
            prefix = path.name.partition("_")[0]
            prefixes.setdefault(prefix.casefold(), prefix)
    return sorted(prefixes.values(), key=str.casefold)


def resolve_game_prefix(
    requested_name: str,
    prefixes: list[str],
    *,
    allow_fuzzy: bool = False,
) -> str | None:
    """Resolve an alias-normalized game name to the actual filename prefix."""
    requested_casefold = requested_name.casefold()
    for prefix in prefixes:
        if prefix.casefold() == requested_casefold:
            return prefix

    requested_normalized = normalize_game_name(requested_name)
    normalized_matches = [
        prefix
        for prefix in prefixes
        if normalize_game_name(prefix) == requested_normalized
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    # A few configured canonical names differ slightly from the on-disk prefix,
    # such as "崩坏2" and "崩坏学园2". Only enable this for known aliases/games.
    if allow_fuzzy and prefixes:
        match, score = max(
            (
                (
                    prefix,
                    max(
                        fuzz.ratio(requested_name, prefix),
                        fuzz.partial_ratio(requested_name, prefix),
                    ),
                )
                for prefix in prefixes
            ),
            key=lambda item: item[1],
        )
        if score >= 75:
            return match
    return None


def _files_for_game(folder: Path, game_prefix: str) -> list[Path]:
    if not folder.is_dir():
        return []
    prefix_casefold = game_prefix.casefold()
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file()
            and "_" in path.name
            and path.name.partition("_")[0].casefold() == prefix_casefold
        ),
        key=lambda path: path.name.casefold(),
    )


def count_game_paintings(
    game_prefix: str,
    husbands_folder: Path,
    wives_folder: Path,
) -> tuple[int, int]:
    return (
        len(_files_for_game(husbands_folder, game_prefix)),
        len(_files_for_game(wives_folder, game_prefix)),
    )


def safe_archive_name(game_name: str) -> str:
    invalid = '<>:"/\\|?*'
    cleaned = "".join(
        "_" if char in invalid or ord(char) < 32 else char for char in game_name
    ).strip(" ._")
    return f"{cleaned or 'game'}_立绘.zip"


def build_painting_archive(
    game_prefix: str,
    husbands_folder: Path,
    wives_folder: Path,
    work_dir: Path,
) -> PaintingArchive:
    husbands = _files_for_game(husbands_folder, game_prefix)
    wives = _files_for_game(wives_folder, game_prefix)
    if not husbands and not wives:
        raise FileNotFoundError(f"没有找到 {game_prefix} 的立绘")

    work_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / f"{uuid.uuid4().hex}_{safe_archive_name(game_prefix)}"
    source_size = sum(path.stat().st_size for path in (*husbands, *wives))
    try:
        # PNG/JPEG data is already compressed. ZIP_STORED is much faster and avoids
        # blocking a worker thread for little or no size reduction.
        with _new_progress() as progress:
            task_id = progress.add_task(
                f"打包 {escape(game_prefix)} 立绘",
                total=max(1, source_size),
            )
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                for folder_name, files in (
                    ("husbands", husbands),
                    ("wives", wives),
                ):
                    for path in files:
                        archive.write(path, arcname=f"{folder_name}/{path.name}")
                        progress.update(task_id, advance=path.stat().st_size)
            progress.update(task_id, completed=max(1, source_size))
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    return PaintingArchive(
        path=archive_path,
        game_name=game_prefix,
        husbands_count=len(husbands),
        wives_count=len(wives),
    )
