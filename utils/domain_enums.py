"""
Canonical enums for domain annotation.

Single source of truth for domain classification categories. Used by
`models/domain_classifier.py` via `pipeline/global_var.py` to constrain
LLM output to a known vocabulary.

To change/add a category, edit this file. The 86 values below are loaded
into the Qwen3OmniDomainClassifier prompt and used to validate LLM output
(non-matching values fall back to "unknown").

Override (optional):
    A config file may provide `domain_annotation.{group}.{field}_enum` to
    override the defaults below for a specific dataset. If absent or empty,
    the values here apply.
"""
from typing import Dict, List


# =============================================================================
# 1. Text Domain - what the audio is about and how it's performed
# =============================================================================

TEXT_DOMAIN_ENUMS: Dict[str, List[str]] = {
    # 大类: high-level content category
    "domain": [
        "podcast",
        "audiobook",
        "news",
        "interview",
        "lecture",
        "drama",
        "documentary",
        "conversation",
        "monologue",
        "education",
        "entertainment",
        "religion",
        "advertisement",
        "unknown",
    ],
    # 细化场景: more specific topical scenario
    "scenario": [
        "tech",
        "business",
        "science",
        "history",
        "politics",
        "lifestyle",
        "true_crime",
        "comedy",
        "fiction",
        "non_fiction",
        "self_help",
        "sports",
        "music_review",
        "health",
        "finance",
        "religion",
        "kids",
        "language_learning",
        "philosophy",
        "general",
        "unknown",
    ],
    # 演绎风格: how content is delivered
    "style": [
        "neutral",
        "narrative",
        "dialogue",
        "debate",
        "scripted",
        "spontaneous",
        "expressive",
        "calm",
        "energetic",
        "formal",
        "casual",
        "humorous",
        "serious",
        "unknown",
    ],
}


# =============================================================================
# 2. Acoustic Domain - physical recording environment and quality
# =============================================================================

ACOUSTIC_DOMAIN_ENUMS: Dict[str, List[str]] = {
    # 物理环境
    "environment": [
        "studio",
        "indoor_quiet",
        "indoor_noisy",
        "outdoor",
        "vehicle",
        "phone_call",
        "online_meeting",
        "live_event",
        "unknown",
    ],
    # 背景音情况
    "background": [
        "clean",
        "music",
        "noise",
        "applause",
        "ambient",
        "speech_overlap",
        "echo_reverb",
        "unknown",
    ],
    # 录音质量等级
    "quality": [
        "professional",
        "semi_pro",
        "mobile",
        "low",
        "very_low",
        "unknown",
    ],
}


# =============================================================================
# 3. Speaker Domain - biological / identity attributes of the speaker
# =============================================================================

SPEAKER_DOMAIN_ENUMS: Dict[str, List[str]] = {
    "gender": [
        "male",
        "female",
        "unknown",
    ],
    "age_group": [
        "child",
        "teen",
        "young_adult",
        "adult",
        "senior",
        "unknown",
    ],
    "accent": [
        "native_standard",
        "native_regional",
        "foreign",
        "non_standard",
        "unknown",
    ],
}


# =============================================================================
# Helpers
# =============================================================================

def resolve_text_enums(cfg_section: dict) -> Dict[str, List[str]]:
    """Read enums from config section with fallback to canonical defaults.

    Allows per-dataset override. If `cfg_section` provides a non-empty list
    for a given key (e.g. "domain_enum"), use it; otherwise use the default.
    """
    out = {}
    for key in ("domain", "scenario", "style"):
        cfg_key = f"{key}_enum"
        cfg_val = cfg_section.get(cfg_key)
        out[key] = cfg_val if cfg_val else TEXT_DOMAIN_ENUMS[key]
    return out


def resolve_acoustic_enums(cfg_section: dict) -> Dict[str, List[str]]:
    out = {}
    for key in ("environment", "background", "quality"):
        cfg_key = f"{key}_enum"
        cfg_val = cfg_section.get(cfg_key)
        out[key] = cfg_val if cfg_val else ACOUSTIC_DOMAIN_ENUMS[key]
    return out


def resolve_speaker_enums(cfg_section: dict) -> Dict[str, List[str]]:
    out = {}
    for key in ("gender", "age_group", "accent"):
        cfg_key = f"{key}_enum"
        cfg_val = cfg_section.get(cfg_key)
        out[key] = cfg_val if cfg_val else SPEAKER_DOMAIN_ENUMS[key]
    return out
