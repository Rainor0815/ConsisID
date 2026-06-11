from dataclasses import dataclass
from typing import Any, Dict, List, Optional


DEFAULT_FACE_PROMPT_SUFFIX = (
    "The subject's face remains clearly visible, centered, and front-facing in every frame, "
    "with both eyes visible and no occlusion."
)

DEFAULT_FACE_NEGATIVE_PROMPT = (
    "face out of frame, back of head, hidden face, cropped face, occluded face, no face, "
    "blurred face, side turned away, duplicate faces"
)

DEFAULT_FACE_FALLBACK_PROMPT = (
    "A steady realistic medium close-up portrait of the same person facing the camera. "
    "The subject's full face is centered, clearly visible, front-facing, well lit, and unobstructed. "
    "Both eyes are visible, the expression is natural, and the background is simple and softly blurred."
)


@dataclass
class RecoveryProfile:
    attempt_index: int
    name: str
    prompt: str
    negative_prompt: Optional[str]
    seed_offset: int
    episodic_top_k: Optional[int]
    episodic_base_weight: Optional[float]
    local_face_scale: Optional[float]

    def to_row(self) -> Dict[str, Any]:
        return {
            "recovery_profile": self.name,
            "attempt_seed_offset": self.seed_offset,
            "attempt_prompt": self.prompt,
            "attempt_negative_prompt": self.negative_prompt or "",
            "local_face_scale": self.local_face_scale if self.local_face_scale is not None else "",
        }


def _split_values(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_values(value: Optional[str], cast):
    parsed = []
    for item in _split_values(value):
        if item.lower() in {"none", "default"}:
            parsed.append(None)
        else:
            parsed.append(cast(item))
    return parsed


def _value_at(values: List[Any], index: int, default=None):
    if not values:
        return default
    if index < len(values):
        return values[index]
    return values[-1]


def _append_text(base: Optional[str], suffix: Optional[str]) -> Optional[str]:
    suffix = (suffix or "").strip()
    if not suffix:
        return base
    base = (base or "").strip()
    if not base:
        return suffix
    return f"{base} {suffix}"


def build_recovery_profiles(base_prompt: str, negative_prompt: Optional[str], segment_index: int, args) -> List[RecoveryProfile]:
    max_attempts = max(int(getattr(args, "face_recovery_max_attempts", 1)), 1)
    seed_stride = int(getattr(args, "face_recovery_seed_stride", 101))
    prompt_suffix = getattr(args, "face_recovery_prompt_suffix", None) or DEFAULT_FACE_PROMPT_SUFFIX
    negative_suffix = getattr(args, "face_recovery_negative_prompt", None) or DEFAULT_FACE_NEGATIVE_PROMPT
    fallback_prompt = getattr(args, "face_recovery_fallback_prompt", None) or DEFAULT_FACE_FALLBACK_PROMPT
    local_face_scales = _parse_values(getattr(args, "face_recovery_local_face_scales", None), float)
    top_ks = _parse_values(getattr(args, "face_recovery_top_ks", None), int)
    base_weights = _parse_values(getattr(args, "face_recovery_base_weights", None), float)

    profiles = [
        RecoveryProfile(
            attempt_index=0,
            name="initial",
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            seed_offset=0,
            episodic_top_k=None,
            episodic_base_weight=None,
            local_face_scale=_value_at(local_face_scales, 0, None),
        )
    ]

    for attempt_index in range(1, max_attempts):
        prompt = _append_text(base_prompt, prompt_suffix)
        if attempt_index == max_attempts - 1 and fallback_prompt:
            prompt = fallback_prompt
        profile_seed_offset = attempt_index * seed_stride
        profiles.append(
            RecoveryProfile(
                attempt_index=attempt_index,
                name=f"face_recovery_{attempt_index:02d}",
                prompt=prompt,
                negative_prompt=_append_text(negative_prompt, negative_suffix),
                seed_offset=profile_seed_offset,
                episodic_top_k=_value_at(top_ks, attempt_index - 1, None),
                episodic_base_weight=_value_at(base_weights, attempt_index - 1, None),
                local_face_scale=_value_at(local_face_scales, attempt_index, _value_at(local_face_scales, -1, None)),
            )
        )
    return profiles


def _metric(summary: Dict[str, Any], key: str, fallback: float = -1.0) -> float:
    value = summary.get(key)
    if value is None:
        return fallback
    return float(value)


def segment_passes_face_guard(
    summary: Dict[str, Any],
    min_detection_rate: float,
    min_mean_similarity: float,
    min_frame_similarity: Optional[float] = None,
) -> bool:
    if _metric(summary, "detection_rate", 0.0) < min_detection_rate:
        return False
    if _metric(summary, "mean_similarity") < min_mean_similarity:
        return False
    if min_frame_similarity is not None and _metric(summary, "min_similarity") < min_frame_similarity:
        return False
    return True


def rank_segment_attempt(summary: Dict[str, Any]):
    return (
        _metric(summary, "detection_rate", 0.0),
        -float(summary.get("zero_face_chunks") or 0),
        -float(summary.get("max_consecutive_missing_frames") or 0),
        _metric(summary, "mean_similarity"),
        _metric(summary, "min_similarity"),
        _metric(summary, "mean_face_area_ratio", 0.0),
    )


def format_recovery_reason(
    summary: Dict[str, Any],
    min_detection_rate: float,
    min_mean_similarity: float,
    min_frame_similarity: Optional[float] = None,
) -> str:
    reasons = []
    detection_rate = _metric(summary, "detection_rate", 0.0)
    if detection_rate < min_detection_rate:
        reasons.append(f"detection_rate {detection_rate:.4f} < {min_detection_rate:.4f}")
    mean_similarity = summary.get("mean_similarity")
    if mean_similarity is None or float(mean_similarity) < min_mean_similarity:
        reasons.append(f"mean_similarity {mean_similarity} < {min_mean_similarity:.4f}")
    if min_frame_similarity is not None:
        min_similarity = summary.get("min_similarity")
        if min_similarity is None or float(min_similarity) < min_frame_similarity:
            reasons.append(f"min_similarity {min_similarity} < {min_frame_similarity:.4f}")
    return "; ".join(reasons)
