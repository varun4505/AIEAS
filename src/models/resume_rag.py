"""RAG pipeline that scores resumes against job descriptions using the Groq API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import json
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.groq_client import call_groq_chat
from src.utils.embedding import compute_embeddings


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        # remove first fence label (e.g., ```json)
        trimmed = trimmed.split("\n", 1)[-1]
        trimmed = trimmed.rsplit("```", 1)[0]
    return trimmed.strip()


def _extract_json_block(text: str) -> str:
    cleaned = _strip_code_fences(text)
    # If the cleaned text already looks like JSON, return it
    if cleaned.startswith("{") and cleaned.rstrip().endswith("}"):
        return cleaned
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return match.group(0)
    return cleaned


@dataclass
class RagResult:
    fit_score: float
    strengths: str
    gaps: str
    summary: str
    retrieved_chunks: List[str]
    raw_response: str


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for idx in range(0, len(words), step):
        chunk_words = words[idx : idx + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks


def _build_prompt(job_description: str, resume_chunks: List[str]) -> List[dict[str, str]]:
    context = "\n\n".join(f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(resume_chunks))
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant helping a recruiter evaluate a resume. "
            "Return the result as a JSON object with the following keys: "
            "fit_score (number 0-100), strengths (array of short strings), gaps (array of short strings), summary (string). "
            "If you cannot compute a value, return an empty array or an empty string, not null."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Job Description:\n{job_description}\n\n"
            f"Relevant Resume Excerpts:\n{context}\n\n"
            "Return a fit score (0-100), bullet strengths, bullet gaps, and a short summary."
            "Output must be valid JSON. Example output:\n"
            "{"
            "\"fit_score\": 78,\n"
            "\"strengths\": [\"Strong analytical skills\", \"Good communicator\"],\n"
            "\"gaps\": [\"Limited cloud experience\"],\n"
            "\"summary\": \"Short human-readable summary here.\"\n"
            "}"
        ),
    }
    return [system_msg, user_msg]


def _parse_groq_payload(payload: str) -> RagResult:
    # Normalize payload and use regex to extract components more robustly.
    text = _strip_code_fences(payload)
    fit_score = 0.0
    strengths: List[str] = []
    gaps: List[str] = []
    summary = ""

    # Extract Fit Score via regex (0-100)
    m = re.search(r"fit\s*score\s*[:\-]?\s*(\d{1,3})", text, re.IGNORECASE)
    if m:
        try:
            fit_score = float(m.group(1))
        except Exception:
            fit_score = 0.0

    # Extract sections by headings
    def extract_section(name: str) -> List[str]:
        pattern = rf"{name}[:\-]?\s*(.*?)(?:\n\n|$)"
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        items: List[str] = []
        for block in matches:
            # split on lines or bullets
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                # remove leading bullets
                items.append(line.lstrip("-* "))
        return items

    strengths = extract_section("strengths") or extract_section("strength")
    gaps = extract_section("gaps") or extract_section("gaps and improvements") or extract_section("gaps/")

    # Summary - prefer a block after 'summary' heading, otherwise fallback to the last paragraph
    sum_matches = re.findall(r"summary[:\-]?\s*(.*?)(?:\n\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if sum_matches:
        summary = " ".join(p.strip() for p in sum_matches)
    else:
        # fallback: last 2 lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        summary = " ".join(lines[-2:]) if len(lines) >= 2 else (lines[0] if lines else "No summary provided.")

    return RagResult(
        fit_score=max(0.0, min(100.0, fit_score)),
        strengths="\n".join(f"- {item}" for item in strengths) or "- Not specified",
        gaps="\n".join(f"- {item}" for item in gaps) or "- Not specified",
        summary=summary or "No summary provided.",
        retrieved_chunks=[],
        raw_response=payload,
    )


def rag_resume_score(job_description: str, resume_text: str, *, top_k: int = 3) -> RagResult:
    """Return the Groq-evaluated fit between a job description and resume."""
    if not job_description.strip():
        raise ValueError("Job description is required for RAG scoring")
    if not resume_text.strip():
        raise ValueError("Resume text is empty; upload a valid PDF")

    chunks = _chunk_text(resume_text)
    if not chunks:
        raise ValueError("Resume parsing produced no text")

    # Embed job description and resume chunks.
    jd_embedding = compute_embeddings(job_description)
    chunk_embeddings = compute_embeddings(chunks)
    scores = cosine_similarity(jd_embedding, chunk_embeddings)[0]

    top_indices = np.argsort(scores)[::-1][: max(1, min(top_k, len(chunks)))]
    top_chunks = [chunks[idx] for idx in top_indices]

    messages = _build_prompt(job_description, top_chunks)
    response = call_groq_chat(messages)

    content = response["choices"][0]["message"]["content"]
    # Try JSON parse first (we instruct the model to return JSON). If that
    # fails, fall back to the existing regex parser.
    parsed_result = None
    try:
        json_text = _extract_json_block(content)
        obj = json.loads(json_text)
        fit_score = float(obj.get("fit_score", 0) or 0)

        def _ensure_list(value):
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                return [value]
            return []

        strengths = _ensure_list(obj.get("strengths", []))
        gaps = _ensure_list(obj.get("gaps", []))
        summary = obj.get("summary", "") or ""
        parsed_result = RagResult(
            fit_score=max(0.0, min(100.0, fit_score)),
            strengths="\n".join(f"- {s}" for s in strengths) or "- Not specified",
            gaps="\n".join(f"- {g}" for g in gaps) or "- Not specified",
            summary=summary or "No summary provided.",
            retrieved_chunks=[],
            raw_response=content,
        )
    except Exception:
        parsed_result = _parse_groq_payload(content)

    result = parsed_result
    result.retrieved_chunks = top_chunks
    return result
