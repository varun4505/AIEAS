"""Scoring helpers for the candidate evaluation pipeline."""
from __future__ import annotations

from statistics import mean
from typing import Iterable, Sequence


EMOTION_BASELINE_SCORES = {
    "happy": 90.0,
    "neutral": 80.0,
    "calm": 85.0,
    "surprise": 65.0,
    "sad": 40.0,
    "fear": 35.0,
    "disgust": 30.0,
    "angry": 25.0,
}


def emotion_to_score(emotion: str) -> float:
    """Map an emotion label to a heuristic score between 0 and 100."""
    key = emotion.lower().strip()
    return EMOTION_BASELINE_SCORES.get(key, 60.0)


def compute_emotion_stability(timeline: Sequence[str]) -> float:
    """Return a stability score based on how consistent the detected emotions were."""
    if not timeline:
        return 50.0

    dominant = max(set(timeline), key=timeline.count)
    dominant_ratio = timeline.count(dominant) / len(timeline)
    baseline = emotion_to_score(dominant)
    return min(100.0, baseline * dominant_ratio)


def normalize_audio_confidence(confidence: float | None) -> float:
    """Ensure audio emotion confidence stays within 0-100."""
    if confidence is None:
        return 60.0
    return max(0.0, min(100.0, confidence))


def normalize_stress_index(stress_index: float | None) -> float:
    """Clamp stress index to 0-100, where lower is better."""
    if stress_index is None:
        return 70.0
    return max(0.0, min(100.0, stress_index))


def compute_final_suitability(
    job_fit: float,
    face_emotion_score: float,
    audio_emotion_score: float,
    stress_index: float,
) -> float:
    """Blend subscores into a holistic 0-100 suitability value."""
    weights = {
        "job_fit": 0.5,
        "face": 0.2,
        "audio": 0.15,
        "stress": 0.15,
    }

    stress_component = 100.0 - stress_index
    weighted_sum = (
        job_fit * weights["job_fit"]
        + face_emotion_score * weights["face"]
        + audio_emotion_score * weights["audio"]
        + stress_component * weights["stress"]
    )

    return round(weighted_sum, 2)


def average_score(values: Iterable[float]) -> float:
    """Convenience helper to average numeric scores."""
    values = list(values)
    if not values:
        return 0.0
    return round(mean(values), 2)
