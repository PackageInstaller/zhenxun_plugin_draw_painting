from __future__ import annotations

from dataclasses import dataclass

import yaml

from zhenxun.services.log import logger

from ..config import paths


@dataclass
class GameInfo:
    """游戏信息类"""

    name: str
    aliases: list[str]
    short_name: str | None = None
    en_name: str | None = None


class GameAliasManager:
    """游戏别名管理器"""

    def __init__(self):
        self.games_config: list[GameInfo] = []
        self.game_aliases: dict[str, list[str]] = {}
        self._name_mapping: dict[str, str] = {}
        self._load_config()

    def _load_config(self):
        """加载游戏配置"""

        try:
            with open(paths.GAME_ALIASES_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.games_config = [GameInfo(**game_data) for game_data in data["games"]]

            self.game_aliases = {game.name: game.aliases for game in self.games_config}

            for game in self.games_config:
                self._name_mapping[game.name.lower()] = game.name
                for alias in game.aliases:
                    self._name_mapping[alias.lower()] = game.name
                if game.en_name:
                    self._name_mapping[game.en_name.lower()] = game.name
                if game.short_name:
                    self._name_mapping[game.short_name.lower()] = game.name

        except Exception as e:
            logger.error(f"加载游戏别名配置失败: {e}")
            self.games_config = []
            self.game_aliases = {}
            self._name_mapping = {}

    async def get_game_name_from_alias(self, alias: str) -> str:
        """从别名获取标准游戏名称"""
        return self._name_mapping.get(alias.lower(), alias)

    def reload_config(self):
        """重新加载配置"""
        self.games_config.clear()
        self.game_aliases.clear()
        self._name_mapping.clear()
        self._load_config()

    def get_game_info(self, name: str) -> GameInfo | None:
        """获取游戏完整信息"""
        std_name = self._name_mapping.get(name.lower())
        if std_name:
            return next(
                (game for game in self.games_config if game.name == std_name), None
            )
        return None


game_alias_manager = GameAliasManager()


async def get_game_name_from_alias(alias: str) -> str:
    """返回游戏名"""
    return await game_alias_manager.get_game_name_from_alias(alias)
