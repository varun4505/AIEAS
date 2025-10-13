"""RAG pipeline that scores resumes against job descriptions using the Groq API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.groq_client import call_groq_chat
from src.utils.embedding import compute_embeddings


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
            "Respond in plain text with headings for Fit Score, Strengths, Gaps, and Summary."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Job Description:\n{job_description}\n\n"
            f"Relevant Resume Excerpts:\n{context}\n\n"
            "Return a fit score (0-100), bullet strengths, bullet gaps, and a short summary."
        ),
    }
    return [system_msg, user_msg]


def _parse_groq_payload(payload: str) -> RagResult:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    fit_score = 0.0
    strengths: List[str] = []
    gaps: List[str] = []
    summary_lines: List[str] = []

    current_section = None
    for line in lines:
        lower = line.lower()
        if "fit" in lower and "score" in lower:
            current_section = "fit"
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                fit_score = float(digits)
            continue
        if "strength" in lower:
            current_section = "strengths"
            continue
        if "gap" in lower or "improve" in lower:
            current_section = "gaps"
            continue
        if "summary" in lower:
            current_section = "summary"
            continue

        if current_section == "strengths":
            strengths.append(line.lstrip("- "))
        elif current_section == "gaps":
            gaps.append(line.lstrip("- "))
        elif current_section == "summary":
            summary_lines.append(line)

    summary = " ".join(summary_lines) if summary_lines else "No summary provided."
    return RagResult(
        fit_score=max(0.0, min(100.0, fit_score)),
        strengths="\n".join(f"- {item}" for item in strengths) or "- Not specified",
        gaps="\n".join(f"- {item}" for item in gaps) or "- Not specified",
        summary=summary,
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
    result = _parse_groq_payload(content)
    result.retrieved_chunks = top_chunks
    return result
