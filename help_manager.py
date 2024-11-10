from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HelpConfirmationState:
    message_id: int
    confirmed: bool
    processing: bool
    start_time: datetime

class HelpConfirmationManager:
    _instance = None
    _help_confirmations: Dict[str, HelpConfirmationState] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HelpConfirmationManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def add_confirmation(cls, user_id: str, message_id: int) -> None:
        cls._help_confirmations[user_id] = HelpConfirmationState(
            message_id=message_id,
            confirmed=False,
            processing=True,
            start_time=datetime.now()
        )

    @classmethod
    def get_confirmation(cls, user_id: str) -> Optional[HelpConfirmationState]:
        return cls._help_confirmations.get(user_id)

    @classmethod
    def remove_confirmation(cls, user_id: str) -> None:
        cls._help_confirmations.pop(user_id, None)

    @classmethod
    def set_confirmed(cls, user_id: str, confirmed: bool = True) -> None:
        if user_id in cls._help_confirmations:
            cls._help_confirmations[user_id].confirmed = confirmed
            cls._help_confirmations[user_id].processing = False

    @classmethod
    def is_processing(cls, user_id: str) -> bool:
        confirmation = cls._help_confirmations.get(user_id)
        if not confirmation:
            return False
        if (datetime.now() - confirmation.start_time).total_seconds() > 60:
            cls.remove_confirmation(user_id)
            return False
        return confirmation.processing

help_manager = HelpConfirmationManager()