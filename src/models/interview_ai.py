"""LLM-backed helpers that drive the real-time interview experience."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from src.config.settings import get_groq_api_key
from src.models.groq_client import call_groq_chat

DEFAULT_QUESTION_BANK = [
    "Walk me through a recent project that you're proud of and why it mattered.",
    "Describe a time you had to troubleshoot a difficult issue. What was your approach?",
    "How do you ensure you stay current with new tools or technologies relevant to this role?",
    "Tell me about a moment when you had to push back on a stakeholder decision.",
    "Explain a complex concept from your field to someone unfamiliar with it.",
    "How do you prioritize tasks when everything feels urgent?",
]


def _conversation_log(events: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for event in events:
        e_type = event.get("type")
        payload = event.get("payload", {})
        if e_type == "question":
            question = payload.get("text") or payload.get("message")
            if question:
                lines.append(f"AI: {question.strip()}")
        elif e_type == "answer":
            answer = payload.get("transcript") or payload.get("text")
            if answer:
                lines.append(f"Candidate: {answer.strip()}")
    return "\n".join(lines)


def _extract_json_block(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in response")


def _fallback_question(asked: Iterable[str]) -> str:
    asked_set = {q.strip().lower() for q in asked if q}
    for question in DEFAULT_QUESTION_BANK:
        if question.strip().lower() not in asked_set:
            return question
    return "Could you share more detail about a challenge you overcame recently?"


def generate_next_question(
    *,
    candidate_name: str,
    position: str,
    job_description: str,
    persona: str,
    events: List[Dict[str, Any]],
    candidate_retrieved_chunks: List[str] | None = None,
) -> str:
    history = _conversation_log(events)
    api_key = get_groq_api_key()
    if not api_key:
        asked = [event.get("payload", {}).get("text", "") for event in events if event.get("type") == "question"]
        return _fallback_question(asked)

    system_prompt = (
        "You are an AI interviewer running a structured, competency-based interview. "
        "Ask concise, open-ended questions one at a time. Do not provide commentary or multiple questions in a single turn."
    )
    history_block = history or "No questions have been asked yet. Start with an opening question."
    
    # Build candidate context block from retrieved chunks if available
    candidate_context = ""
    if candidate_retrieved_chunks:
        candidate_context = "Candidate profile (relevant excerpts from resume analysis):\n"
        for idx, chunk in enumerate(candidate_retrieved_chunks, start=1):
            candidate_context += f"[{idx}] {chunk.strip()}\n"
        candidate_context += "\n"
    
    user_prompt = (
        f"Candidate name: {candidate_name or 'Candidate'}\n"
        f"Target role: {position or 'Unknown role'}\n"
        f"Interview persona: {persona}.\n"
        "Conduct the interview based on the following job description:\n"
        f"{job_description.strip() or 'Job description not provided.'}\n\n"
        + candidate_context
        + "Conversation so far:\n"
        + f"{history_block}\n\n"
        + "Provide the next question only."
    )
    response = call_groq_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
    )
    content = response["choices"][0]["message"]["content"].strip()
    # Some models prepend labels like "AI:"; strip them when present.
    if content.lower().startswith("ai:"):
        content = content[3:].strip()
    return content


def analyze_answer(
    *,
    question: str,
    answer: str,
    job_description: str,
    persona: str,
) -> Dict[str, Any]:
    answer = answer.strip()
    if not answer:
        return {
            "overall_score": 0.0,
            "summary": "No answer captured.",
            "strengths": [],
            "risks": ["Candidate did not respond."],
            "follow_up": "Could you please provide an answer?",
        }

    api_key = get_groq_api_key()
    if not api_key:
        # Lightweight heuristic fallback when Groq key is absent.
        word_count = len(answer.split())
        score = min(1.0, word_count / 120.0)
        strengths = []
        if word_count > 80:
            strengths.append("Provides detailed, thorough response.")
        if "team" in answer.lower():
            strengths.append("Highlights collaboration.")
        risks = []
        if word_count < 40:
            risks.append("Answer is brief; may lack depth.")
        return {
            "overall_score": round(score * 100, 1),
            "summary": answer[:160] + ("..." if len(answer) > 160 else ""),
            "strengths": strengths,
            "risks": risks or ["Needs elaboration."],
            "follow_up": "What specific contribution did you make, and what was the outcome?",
        }

    system_prompt = (
        "You are assisting a recruiter with real-time interview evaluation."
        "Return feedback as JSON with keys: overall_score (0-100 number), summary (string <= 3 sentences),"
        " strengths (array of short bullet strings), risks (array), follow_up (string)."
    )
    user_prompt = (
        f"Job description context:\n{job_description.strip() or 'N/A'}\n\n"
        f"Interview persona: {persona}.\n"
        f"Question asked: {question}\n"
        f"Candidate answer: {answer}\n\n"
        "Respond with JSON only."
    )
    response = call_groq_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response["choices"][0]["message"]["content"]
    try:
        payload = _extract_json_block(content)
    except Exception:
        # As a last resort, wrap free-form text.
        summary = content.strip()
        return {
            "overall_score": 0.0,
            "summary": summary,
            "strengths": [],
            "risks": ["Model returned unstructured text."],
            "follow_up": "Can you clarify the main impact you delivered?",
        }

    overall_score = float(payload.get("overall_score", 0) or 0)
    strengths = payload.get("strengths") or []
    risks = payload.get("risks") or []
    follow_up = payload.get("follow_up") or ""
    summary = payload.get("summary") or ""

    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    if not isinstance(risks, list):
        risks = [str(risks)]

    return {
        "overall_score": max(0.0, min(100.0, overall_score)),
        "summary": str(summary),
        "strengths": [str(item) for item in strengths],
        "risks": [str(item) for item in risks],
        "follow_up": str(follow_up) if follow_up else "Could you expand on the steps you took?",
    }