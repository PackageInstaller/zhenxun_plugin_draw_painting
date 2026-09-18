"""Gender-library moves use subject tags; Others uses raw scene gender scores."""

import math

from .vision.types import FEMALE_TAGS, MALE_TAGS

AUTO_MOVE_THRESHOLD = 0.70
AUTO_MOVE_MARGIN = 0.10
OTHERS_THRESHOLD = 0.30
# When the scene itself admits no humans, WD's near-zero gender logits are
# corroborated evidence and a solo camie read on an androgynous mascot is a
# phantom.  Below this the models may simply disagree on art style, which is
# not grounds for archiving a clearly gendered character.
NO_HUMANS_SCENE_SUPPORT = 0.50
# In a scene that admits no humans, gender logits describe a creature:
# weak reads are phantom and even a moderate "certain lead" is unreliable,
# so archiving needs a much weaker signal and rescuing a mislabelled human
# needs near-certain agreement.
NON_HUMAN_WEAK_GENDER = 0.60
STRONG_SUBJECT_LEAD = 0.80
NO_HUMANS_THRESHOLD = 0.65
STRONG_NO_HUMANS_THRESHOLD = 0.80
NON_HUMAN_CORROBORATION_THRESHOLD = 0.50
NON_HUMAN_TAGS = frozenset(
    (
        "animal_focus",
        "creature",
        "digimon_(creature)",
        "food_focus",
        "landscape",
        "mecha_focus",
        "non-humanoid_robot",
        "pikmin_(creature)",
        "pokemon_(creature)",
        "robot_animal",
        "scenery",
        "slime_(creature)",
        "vehicle_focus",
    )
)


def fused_scene_gender_scores(scores: object) -> tuple[float, float] | None:
    """Use unfiltered model output: absent tags are not zero probabilities."""
    values = _model_pairs(scores)
    if values is None:
        return None
    return tuple(
        round(0.4 * values[0][index] + 0.6 * values[1][index], 8) for index in (0, 1)
    )


def consensus_gender_scores(scores: object) -> tuple[float, float] | None:
    """A gender signal exists only when BOTH taggers see it (min consensus).

    Existence decisions (is this image genderless?) must not be vetoed by one
    model's phantom read: an androgynous mascot can score "1boy 0.54" in a
    single tagger while the other sees no humans at all.  The weighted fusion
    stays reserved for the move-confidence thresholds.
    """
    values = _model_pairs(scores)
    if values is None:
        return None
    return tuple(
        round(min(values[0][index], values[1][index]), 8) for index in (0, 1)
    )


def _model_pairs(scores: object) -> list[tuple[float, float]] | None:
    if not isinstance(scores, dict):
        return None
    values = []
    for model in ("wd", "camie"):
        pair = scores.get(model)
        if not isinstance(pair, tuple | list) or len(pair) != 2:
            return None
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in pair
        ):
            return None
        values.append((float(pair[0]), float(pair[1])))
    return values


def fused_gender_scores(tags: dict[str, float]) -> tuple[float, float]:
    def score(names: frozenset[str]) -> float:
        return max(
            (tags[tag] for tag in names if tag in tags and math.isfinite(tags[tag])),
            default=0.0,
        )

    return score(MALE_TAGS), score(FEMALE_TAGS)


def explicitly_non_human(tags: object) -> bool:
    """Prefer direct scene semantics over incidental independent gender logits."""
    if not isinstance(tags, dict):
        return False

    def score(name: str) -> float:
        value = tags.get(name, 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            return 0.0
        return float(value)

    no_humans = score("no_humans")
    if no_humans >= STRONG_NO_HUMANS_THRESHOLD:
        return True
    return no_humans >= NO_HUMANS_THRESHOLD and any(
        score(tag) >= NON_HUMAN_CORROBORATION_THRESHOLD for tag in NON_HUMAN_TAGS
    )


def target_library(
    library: str,
    status: str,
    gender: str,
    tags: dict[str, float],
    scene_gender: object = None,
    subject_scores: object = None,
    detection: object = None,
    scene_tags: object = None,
) -> str | None:
    if isinstance(detection, dict) and detection.get("budget_exhausted"):
        return None
    if library not in ("wives", "husbands"):
        return None
    # 1girl/1boy are independent image-tag logits, not probabilities that the
    # depicted subject is human. Animal/mascot silhouettes can activate them.
    # Strong, corroborated `no_humans` evidence is therefore authoritative.
    if explicitly_non_human(scene_tags):
        return "Others"
    scene_scores = fused_scene_gender_scores(scene_gender)
    raw_subject = fused_scene_gender_scores(subject_scores)
    if scene_scores is not None:
        effective_scores = scene_scores
        effective_subject = raw_subject
        threshold = OTHERS_THRESHOLD
        escape = OTHERS_THRESHOLD
        no_humans = scene_tags.get("no_humans", 0.0) if isinstance(
            scene_tags, dict
        ) else 0.0
        if isinstance(no_humans, int | float) and no_humans >= (
            NO_HUMANS_SCENE_SUPPORT
        ):
            # The scene itself says no humans: gender logits on this image
            # describe a creature, so weak reads count as phantom.  Only an
            # overwhelmingly certain subject lead (both taggers near-certain)
            # may still rescue a mislabelled human.
            consensus_scene = consensus_gender_scores(scene_gender)
            if consensus_scene is not None:
                effective_scores = consensus_scene
                effective_subject = consensus_gender_scores(subject_scores)
                threshold = NON_HUMAN_WEAK_GENDER
                escape = STRONG_SUBJECT_LEAD
        if all(score < threshold for score in effective_scores):
            # A small but certain lead can disappear in whole-scene logits.
            # Do not archive it as genderless merely because the background
            # dominates; the lead must survive the same consensus rule.
            if effective_subject is not None and (
                max(effective_subject) >= escape
            ):
                return None
            return "Others"
    if status != "confident":
        return None
    if subject_scores is not None and raw_subject is None:
        return None
    male, female = raw_subject or fused_gender_scores(tags)
    if library == "wives" and gender == "male":
        target, opposite, destination = male, female, "husbands"
    elif library == "husbands" and gender == "female":
        target, opposite, destination = female, male, "wives"
    else:
        return None
    if (
        target > AUTO_MOVE_THRESHOLD
        and opposite <= AUTO_MOVE_THRESHOLD
        and target - opposite >= AUTO_MOVE_MARGIN
    ):
        return destination
    return None
