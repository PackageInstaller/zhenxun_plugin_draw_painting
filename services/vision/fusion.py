"""Pure fusion rules, kept separate from heavyweight model sessions."""

from .types import MULTIPLE_TAGS, ModelTags


def fuse_general(wd: ModelTags, camie: ModelTags, *, strict: bool) -> dict[str, float]:
    fused: dict[str, float] = {}
    for tag in wd.general.keys() | camie.general.keys():
        left, right = wd.general.get(tag), camie.general.get(tag)
        if left is not None and right is not None:
            score = 0.4 * left + 0.6 * right
            accepted = (
                left >= 0.35 and right >= 0.492
                if strict
                else score >= 0.4 and max(left, right) >= 0.5
            )
        else:
            score = left if right is None else right
            accepted = score is not None and score >= (0.7 if strict else 0.6)
        if accepted and score is not None:
            fused[tag] = round(score, 4)
    return dict(sorted(fused.items(), key=lambda pair: pair[1], reverse=True))


def subject_gender(wd: ModelTags, camie: ModelTags) -> tuple[str, str, float, float]:
    wm, wf = wd.gender_scores()
    cm, cf = camie.gender_scores()
    male, female = min(wm, cm), min(wf, cf)
    if any(
        result.general.get(tag, 0.0) >= 0.6
        for result in (wd, camie)
        for tag in MULTIPLE_TAGS
    ):
        return "unknown", "multiple_in_crop", male, female
    if male >= 0.65 and max(wf, cf) < 0.35:
        return "male", "confident", male, female
    if female >= 0.65 and max(wm, cm) < 0.35:
        return "female", "confident", male, female
    # Independent multilabel logits are not complementary probabilities. A
    # moderately high opposite tag in one model need not veto two very strong
    # winners; require a large within-model margin in BOTH models instead.
    # 0.80 (not 0.85): desaturated "gray-toned" leads read ~0.82 in the weaker
    # tagger with a large internal margin - still a clear winner.
    if male >= 0.80 and min(wm - wf, cm - cf) >= 0.3:
        return "male", "confident", male, female
    if female >= 0.80 and min(wf - wm, cf - cm) >= 0.3:
        return "female", "confident", female, male
    return "unknown", "gender_uncertain", male, female
