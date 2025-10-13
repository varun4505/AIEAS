"""Local JSON storage helpers for prototypes."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

from src.config.settings import CANDIDATES_PATH, JOB_DESCRIPTIONS_PATH


def _load_json(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_json(path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_job_descriptions() -> list[dict[str, Any]]:
    return _load_json(JOB_DESCRIPTIONS_PATH)


def save_job_description(title: str, description: str) -> dict[str, Any]:
    entries = load_job_descriptions()
    entry = {
        "id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    entries.append(entry)
    _save_json(JOB_DESCRIPTIONS_PATH, entries)
    return entry


def load_candidates() -> list[dict[str, Any]]:
    return _load_json(CANDIDATES_PATH)


def save_candidate(candidate_payload: Dict[str, Any]) -> Dict[str, Any]:
    entries = load_candidates()
    payload = {
        "id": candidate_payload.get("id", str(uuid.uuid4())),
        "created_at": candidate_payload.get("created_at", datetime.utcnow().isoformat() + "Z"),
        **candidate_payload,
    }
    entries.append(payload)
    _save_json(CANDIDATES_PATH, entries)
    return payload
