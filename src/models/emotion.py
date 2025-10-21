"""Emotion and stress analysis helpers."""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from src.utils.scoring import (
    emotion_to_score,
    normalize_audio_confidence,
    normalize_stress_index,
    compute_emotion_stability,
)


@dataclass
class FaceEmotionReport:
    dominant_emotion: str
    timeline: List[str]
    confidence: float
    note: str = ""


@dataclass
class AudioEmotionReport:
    label: str
    confidence: float
    note: str = ""


@dataclass
class StressReport:
    stress_index: float
    note: str = ""


def detect_emotions_from_camera(image_file) -> FaceEmotionReport:
    """Analyze a snapshot captured from the webcam and infer emotion."""
    if image_file is None:
        return FaceEmotionReport("neutral", [], 50.0, note="No image provided; using neutral baseline.")

    try:
        from deepface import DeepFace  # type: ignore
    except ImportError:
        return FaceEmotionReport(
            "neutral",
            ["neutral"],
            55.0,
            note="DeepFace not installed; returning mock inference.",
        )

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        tmp.write(image_file.getvalue())
        tmp.flush()
        try:
            analysis = DeepFace.analyze(img_path=tmp.name, actions=["emotion"], enforce_detection=False)
        except Exception as exc:  # pragma: no cover - external dependency/runtime specific
            return FaceEmotionReport("neutral", [], 50.0, note=f"Emotion detection failed: {exc}")

    if isinstance(analysis, list):
        analysis = analysis[0]

    emotions = analysis.get("emotion", {})
    if not emotions:
        return FaceEmotionReport("neutral", [], 50.0, note="No emotions detected; returning neutral.")

    dominant = analysis.get("dominant_emotion", max(emotions, key=emotions.get))
    confidence = float(emotions.get(dominant, 0.0))
    timeline = sorted(emotions, key=emotions.get, reverse=True)[:5]
    return FaceEmotionReport(dominant, timeline, confidence, note="DeepFace analysis completed.")


def analyze_audio_emotion(audio_file) -> AudioEmotionReport:
    """Classify the dominant emotion of an uploaded audio sample."""
    if audio_file is None:
        return AudioEmotionReport("calm", 60.0, note="No audio provided; defaulting to calm.")

    try:
        import librosa  # type: ignore
    except ImportError:
        return AudioEmotionReport("calm", 62.0, note="librosa not installed; returning mock inference.")

    try:
        data, sr = librosa.load(io.BytesIO(audio_file.getvalue()), sr=None)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        energy = float(np.mean(np.abs(data)))
        tempo, _ = librosa.beat.beat_track(y=data, sr=sr)
    except Exception as exc:  # pragma: no cover - audio parsing errors
        return AudioEmotionReport("calm", 55.0, note=f"Audio analysis failed: {exc}")

    # Simple heuristics: higher energy + tempo -> happy, low energy -> sad, otherwise calm.
    if energy > 0.2 and tempo > 120:
        label = "happy"
        confidence = 85.0
    elif energy < 0.05:
        label = "sad"
        confidence = 65.0
    else:
        label = "calm"
        confidence = 75.0

    return AudioEmotionReport(label, confidence, note="Heuristic audio classification applied.")


def analyze_eeg_signal(file_obj) -> StressReport:
    """Estimate stress from a simulated EEG CSV upload."""
    if file_obj is None:
        return StressReport(70.0, note="No EEG data provided; using neutral stress index.")

    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:  # pragma: no cover - inconsistent CSV
        return StressReport(65.0, note=f"EEG parsing failed: {exc}")

    signal_cols = [col for col in df.columns if col.lower().startswith("channel")]
    if not signal_cols:
        return StressReport(60.0, note="EEG CSV missing channel columns; default stress applied.")

    variability = float(df[signal_cols].std().mean())
    stress_index = normalize_stress_index(70.0 + variability * 15)
    return StressReport(stress_index, note="Stress index derived from EEG variance.")


def summarize_face_emotion(report: FaceEmotionReport) -> Dict[str, Union[str, float]]:
    """Convert the face emotion report into numeric indicators for dashboards."""
    stability = compute_emotion_stability(report.timeline or [report.dominant_emotion])
    return {
        "dominant": report.dominant_emotion,
        "confidence": normalize_audio_confidence(report.confidence),
        "stability": stability,
    }
