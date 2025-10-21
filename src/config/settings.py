"""Application settings and environment helpers."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from a local .env file if present.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = PROJECT_ROOT / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

JOB_DESCRIPTIONS_PATH = STORAGE_DIR / "job_descriptions.json"
CANDIDATES_PATH = STORAGE_DIR / "candidates.json"
INTERVIEW_SESSIONS_PATH = STORAGE_DIR / "interview_sessions.json"


def get_groq_api_key() -> str | None:
    """Return the Groq API key from the environment if configured."""
    return os.getenv("GROQ_API_KEY")


def get_mongodb_uri() -> str | None:
    """Return the MongoDB connection URI from the environment if configured."""
    return os.getenv("MONGODB_URI")
