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


def get_groq_api_key() -> str | None:
    """Return the Groq API key from the environment if configured."""
    return os.getenv("GROQ_API_KEY")
