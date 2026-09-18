"""Backward-compatible exports for the plugin's former utility module.

New code should import from the focused modules under :mod:`..services`.
"""

from ..services.alias_registry import (
    GameAliasManager,
    GameInfo,
    game_alias_manager,
    get_game_name_from_alias,
)
from ..services.command_guard import CommandHandler
from ..services.draw_support import (
    get_random_choice,
    improved_partial_word_match,
    is_exact_match,
)
from ..services.help_confirmation import (
    HelpConfirmationManager,
    HelpConfirmationState,
    help_manager,
)
from ..services.messaging import (
    get_original_sender,
    send_forward_msg_handler,
    send_image_message,
)
from ..services.model import (
    ModelManager,
    TagPrediction,
    WDTaggerModel,
    determine_gender,
)
from ..services.numbers import format_time, parse_chinese_numeral
from ..services.rename import perform_husband_rename, perform_wife_rename
from ..services.statistics import calculate_game_stats, generate_and_send_stats

__all__ = [
    "CommandHandler",
    "GameAliasManager",
    "GameInfo",
    "HelpConfirmationManager",
    "HelpConfirmationState",
    "ModelManager",
    "TagPrediction",
    "WDTaggerModel",
    "calculate_game_stats",
    "determine_gender",
    "format_time",
    "game_alias_manager",
    "generate_and_send_stats",
    "get_game_name_from_alias",
    "get_original_sender",
    "get_random_choice",
    "help_manager",
    "improved_partial_word_match",
    "is_exact_match",
    "parse_chinese_numeral",
    "perform_husband_rename",
    "perform_wife_rename",
    "send_forward_msg_handler",
    "send_image_message",
]
