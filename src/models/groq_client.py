"""Thin wrapper around the Groq chat completion endpoint."""
from __future__ import annotations

from typing import Any, Dict, List

import requests

from src.config.settings import get_groq_api_key

GROQ_API_BASE = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqClientError(RuntimeError):
    """Raised when the Groq API returns an error."""


def call_groq_chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Send a chat completion request to Groq or return a mock response."""
    api_key = get_groq_api_key()
    if not api_key:
        # Mocked fallback for local demos without credentials.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Fit Score: 65\n"
                            "Strengths: Strong analytical experience, solid communication skills.\n"
                            "Gaps: Limited leadership track record, lacks cloud certifications.\n"
                            "Summary: Candidate is a moderate fit; consider for technical rounds."
                        ),
                    }
                }
            ]
        }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}

    response = requests.post(GROQ_API_BASE, json=payload, headers=headers, timeout=60)
    if response.status_code >= 400:
        raise GroqClientError(f"Groq API error ({response.status_code}): {response.text}")

    return response.json()
