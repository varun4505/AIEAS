"""Typed dictionary definitions for interview session storage."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class InterviewParticipant(TypedDict, total=False):
    """Represents a participant taking part in the interview session."""

    role: Literal["candidate", "ai", "observer"]
    display_name: str
    metadata: Dict[str, Any]


class StressSnapshot(TypedDict, total=False):
    """Captures stress analysis results tied to an event or time range."""

    source: Literal["video", "audio", "text"]
    score: float
    level: Literal["calm", "elevated", "high"]
    note: Optional[str]


class InterviewEvent(TypedDict, total=False):
    """Atomic unit captured along the interview timeline."""

    event_id: str
    session_id: str
    type: Literal["question", "answer", "analysis", "system", "marker"]
    actor: str
    created_at: str
    payload: Dict[str, Any]
    stress: Optional[StressSnapshot]
    duration_seconds: Optional[float]


class InterviewSession(TypedDict, total=False):
    """Top level container stored in the interview_sessions collection."""

    id: str
    candidate_id: Optional[str]
    candidate_name: Optional[str]
    position: Optional[str]
    job_description_id: Optional[str]
    status: Literal["draft", "scheduled", "active", "completed", "aborted"]
    created_at: str
    updated_at: Optional[str]
    started_at: Optional[str]
    ended_at: Optional[str]
    settings: Dict[str, Any]
    participants: List[InterviewParticipant]
    events: List[InterviewEvent]
    metrics: Dict[str, Any]
    recordings: Dict[str, Any]