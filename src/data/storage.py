"""Local JSON storage helpers for prototypes."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

from bson import ObjectId

from src.config.settings import CANDIDATES_PATH, INTERVIEW_SESSIONS_PATH, JOB_DESCRIPTIONS_PATH
from src.data.mongo import insert_one, find_all, get_db, get_collection


def _load_json(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_json(path, payload: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _delete_mongo_document(collection_name: str, entry_id: str) -> bool:
    coll = get_collection(collection_name)
    if coll is None:
        return False

    deleted = 0
    try:
        oid = ObjectId(entry_id)
    except Exception:
        oid = None

    if oid:
        deleted = coll.delete_one({"_id": oid}).deleted_count
        if deleted:
            return True

    deleted = coll.delete_one({"id": entry_id}).deleted_count
    return deleted > 0


def load_job_descriptions() -> list[dict[str, Any]]:
    # Prefer MongoDB when available
    if get_db():
        docs = find_all("job_descriptions")
        # Ensure created_at exists for older records
        for d in docs:
            d.setdefault("created_at", d.get("created_at") or datetime.utcnow().isoformat() + "Z")
        return docs
    return _load_json(JOB_DESCRIPTIONS_PATH)


def save_job_description(title: str, description: str) -> dict[str, Any]:
    entry = {
        "title": title,
        "description": description,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    if get_db():
        saved = insert_one("job_descriptions", entry)
        if saved.get("id"):
            entry["id"] = saved["id"]
        return entry
    entries = load_job_descriptions()
    entry["id"] = str(uuid.uuid4())
    entries.append(entry)
    _save_json(JOB_DESCRIPTIONS_PATH, entries)
    return entry


def load_candidates() -> list[dict[str, Any]]:
    if get_db():
        docs = find_all("candidates")
        for d in docs:
            d.setdefault("created_at", d.get("created_at") or datetime.utcnow().isoformat() + "Z")
        return docs
    return _load_json(CANDIDATES_PATH)


def save_candidate(candidate_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "created_at": candidate_payload.get("created_at", datetime.utcnow().isoformat() + "Z"),
        **candidate_payload,
    }
    if get_db():
        saved = insert_one("candidates", payload)
        if saved.get("id"):
            payload["id"] = saved["id"]
        return payload
    payload["id"] = candidate_payload.get("id", str(uuid.uuid4()))
    entries = load_candidates()
    entries.append(payload)
    _save_json(CANDIDATES_PATH, entries)
    return payload


def load_interview_sessions() -> list[dict[str, Any]]:
    if get_db():
        docs = find_all("interview_sessions")
        for d in docs:
            now_iso = datetime.utcnow().isoformat() + "Z"
            d.setdefault("created_at", now_iso)
            d.setdefault("updated_at", d.get("updated_at") or d.get("created_at") or now_iso)
            d.setdefault("status", "draft")
            d.setdefault("participants", [])
            d.setdefault("events", [])
            d.setdefault("settings", {})
            d.setdefault("metrics", {})
            d.setdefault("recordings", {})
        return docs
    return _load_json(INTERVIEW_SESSIONS_PATH)


def save_interview_session(session_payload: Dict[str, Any]) -> Dict[str, Any]:
    now_iso = datetime.utcnow().isoformat() + "Z"
    session = dict(session_payload)
    session.setdefault("id", session_payload.get("id", str(uuid.uuid4())))
    session.setdefault("created_at", now_iso)
    session.setdefault("updated_at", session["created_at"])
    session.setdefault("status", "draft")
    session.setdefault("settings", {})
    session.setdefault("participants", [])
    session.setdefault("events", [])
    session.setdefault("metrics", {})
    session.setdefault("recordings", {})
    if get_db():
        saved = insert_one("interview_sessions", session)
        return saved
    entries = load_interview_sessions()
    entries.append(session)
    _save_json(INTERVIEW_SESSIONS_PATH, entries)
    return session


def append_interview_event(session_id: str, event_payload: Dict[str, Any]) -> Dict[str, Any] | None:
    now_iso = datetime.utcnow().isoformat() + "Z"
    event = dict(event_payload)
    event.setdefault("event_id", event.get("id", str(uuid.uuid4())))
    event.setdefault("session_id", session_id)
    event.setdefault("created_at", now_iso)
    event.setdefault("payload", {})
    event.pop("id", None)

    if get_db():
        coll = get_collection("interview_sessions")
        if coll is None:
            return None

        try:
            oid = ObjectId(session_id)
        except Exception:
            oid = None

        filter_q = {"_id": oid} if oid else {"id": session_id}
        update_doc = {"$push": {"events": event}, "$set": {"updated_at": now_iso}}
        result = coll.update_one(filter_q, update_doc)
        if result.matched_count == 0:
            return None
        doc = coll.find_one(filter_q)
        if not doc:
            return None
        if "_id" in doc:
            doc["id"] = str(doc["_id"])
            doc.pop("_id", None)
        return doc

    sessions = load_interview_sessions()
    updated = None
    for idx, session in enumerate(sessions):
        if session.get("id") == session_id:
            session.setdefault("events", [])
            session["events"].append(event)
            session["updated_at"] = now_iso
            sessions[idx] = session
            updated = session
            break
    if updated is None:
        return None
    _save_json(INTERVIEW_SESSIONS_PATH, sessions)
    return updated


def update_interview_session(session_id: str, patch: Dict[str, Any]) -> Dict[str, Any] | None:
    """Update an interview session document with the provided fields."""

    patch = dict(patch)
    patch.setdefault("updated_at", datetime.utcnow().isoformat() + "Z")

    if get_db():
        coll = get_collection("interview_sessions")
        if coll is None:
            return None

        try:
            oid = ObjectId(session_id)
        except Exception:
            oid = None

        filter_q = {"_id": oid} if oid else {"id": session_id}
        result = coll.find_one_and_update(filter_q, {"$set": patch}, return_document=True)
        if not result and oid:
            # Fallback to string id if lookup by ObjectId failed
            result = coll.find_one_and_update({"id": session_id}, {"$set": patch}, return_document=True)
        if not result:
            return None
        doc = dict(result)
        if "_id" in doc:
            doc["id"] = str(doc["_id"])
            doc.pop("_id", None)
        return doc

    sessions = load_interview_sessions()
    updated = None
    for idx, session in enumerate(sessions):
        if session.get("id") == session_id:
            sessions[idx] = {**session, **patch}
            updated = sessions[idx]
            break
    if updated is None:
        return None
    _save_json(INTERVIEW_SESSIONS_PATH, sessions)
    return updated


def update_candidate(entry_id: str, patch: Dict[str, Any]) -> Dict[str, Any] | None:
    """Update fields on an existing candidate record and return the updated doc.

    Supports MongoDB (preferred) and falls back to JSON storage.
    """
    if get_db():
        coll = get_collection("candidates")
        if coll is None:
            return None

        try:
            from bson import ObjectId

            oid = ObjectId(entry_id)
        except Exception:
            oid = None

        filter_q = {"_id": oid} if oid else {"id": entry_id}
        result = coll.find_one_and_update(filter_q, {"$set": patch}, return_document=True)
        if not result:
            # Try by id string fallback
            result = coll.find_one_and_update({"id": entry_id}, {"$set": patch}, return_document=True)
        if not result:
            return None
        doc = dict(result)
        if "_id" in doc:
            doc["id"] = str(doc["_id"])
            doc.pop("_id", None)
        return doc

    # JSON fallback
    entries = load_candidates()
    updated = None
    for idx, item in enumerate(entries):
        if item.get("id") == entry_id:
            entries[idx] = {**item, **patch}
            updated = entries[idx]
            break
    if updated is None:
        return None
    _save_json(CANDIDATES_PATH, entries)
    return updated


def delete_job_description(entry_id: str) -> bool:
    if get_db():
        return _delete_mongo_document("job_descriptions", entry_id)
    entries = load_job_descriptions()
    filtered = [item for item in entries if item.get("id") != entry_id]
    _save_json(JOB_DESCRIPTIONS_PATH, filtered)
    return len(entries) != len(filtered)


def delete_candidate(entry_id: str) -> bool:
    if get_db():
        return _delete_mongo_document("candidates", entry_id)
    entries = load_candidates()
    filtered = [item for item in entries if item.get("id") != entry_id]
    _save_json(CANDIDATES_PATH, filtered)
    return len(entries) != len(filtered)
