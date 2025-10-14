from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from src.config.settings import get_mongodb_uri
from urllib.parse import urlparse


def get_db() -> Optional[MongoClient]:
    uri = get_mongodb_uri()
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception:
        return None


def get_collection(name: str) -> Optional[Collection]:
    client = get_db()
    if not client:
        return None
    uri = get_mongodb_uri()
    if not uri:
        return None
    parsed = urlparse(uri)
    dbname = parsed.path.lstrip("/") if parsed.path and parsed.path != "/" else None
    if not dbname:
        dbname = "aieas"
    db = client.get_database(dbname)
    return db[name]


def insert_one(collection_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    coll = get_collection(collection_name)
    if coll is None:
        raise RuntimeError("MongoDB is not configured or reachable")
    result = coll.insert_one(payload)
    payload = dict(payload)
    payload["id"] = str(result.inserted_id)
    payload.pop("_id", None)
    return payload


def find_all(collection_name: str) -> List[Dict[str, Any]]:
    coll = get_collection(collection_name)
    if coll is None:
        return []
    docs = list(coll.find())
    normalized: List[Dict[str, Any]] = []
    for d in docs:
        doc = dict(d)
        if "_id" in doc:
            doc["id"] = str(doc["_id"])
            doc.pop("_id", None)
        normalized.append(doc)
    return normalized
