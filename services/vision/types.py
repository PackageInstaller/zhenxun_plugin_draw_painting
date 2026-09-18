from dataclasses import dataclass, field
from typing import Any

from .assets import MODEL_VERSION

GENERAL_THRESHOLD = 0.35
MALE_TAGS = frozenset(("1boy", "multiple_boys", "male_focus", "2boys", "3boys"))
FEMALE_TAGS = frozenset(("1girl", "multiple_girls", "2girls", "3girls"))
MULTIPLE_TAGS = frozenset(
    (
        "multiple_boys",
        "multiple_girls",
        "2boys",
        "2girls",
        "3boys",
        "3girls",
        "4boys",
        "4girls",
        "5boys",
        "5girls",
        "6+boys",
        "6+girls",
    )
)


@dataclass(frozen=True, slots=True)
class TagPrediction:
    width: int
    height: int
    male_probability: float
    female_probability: float
    general_tags: dict[str, float]
    character_tags: dict[str, float]
    rating_tags: dict[str, float]
    model_version: str = MODEL_VERSION
    subject_tags: dict[str, float] = field(default_factory=dict)
    subject_status: str = "legacy"
    subject_gender: str = "unknown"
    analysis: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelTags:
    general: dict[str, float]
    characters: dict[str, float]
    ratings: dict[str, float]

    def gender_scores(self) -> tuple[float, float]:
        return (
            max((self.general.get(tag, 0.0) for tag in MALE_TAGS), default=0.0),
            max((self.general.get(tag, 0.0) for tag in FEMALE_TAGS), default=0.0),
        )


PredictionOutcome = TagPrediction | Exception
