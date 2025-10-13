"""Streamlit entry point for the AIEAS prototype."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.exporters import candidate_to_pdf, candidates_to_csv
from src.data.storage import (
    load_candidates,
    load_job_descriptions,
    save_candidate,
    save_job_description,
)
from src.models.emotion import (
    analyze_audio_emotion,
    analyze_eeg_signal,
    detect_emotions_from_camera,
    summarize_face_emotion,
)
from src.models.resume_rag import rag_resume_score
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.scoring import (
    compute_final_suitability,
    emotion_to_score,
    normalize_audio_confidence,
    normalize_stress_index,
)

st.set_page_config(page_title="AIEAS - Interview & Emotion Analysis", layout="wide")


def candidate_sidebar() -> Dict[str, str]:
    st.sidebar.header("Candidate Info")
    name = st.sidebar.text_input("Full Name")
    email = st.sidebar.text_input("Email")
    position = st.sidebar.text_input("Job Position")
    return {"name": name, "email": email, "position": position}


def candidate_view():
    info = candidate_sidebar()
    st.header("Candidate Module")
    st.caption(
        "Upload a resume, capture a webcam snapshot, and optionally provide audio or EEG data "
        "to generate a multi-modal suitability profile."
    )

    job_descriptions = load_job_descriptions()
    jd_options = {jd["title"]: jd for jd in job_descriptions}

    col_left, col_right = st.columns(2)
    with col_left:
        selected_title = st.selectbox(
            "Select Stored Job Description",
            options=["-- None --", *jd_options.keys()],
            index=0,
        )
        manual_jd = st.text_area(
            "Or Paste Job Description",
            help="Provide the job description if it is not stored yet.",
        )
        resume_file = st.file_uploader("Resume (PDF)", type=["pdf"], help="Only PDF resumes are supported.")
    with col_right:
        captured_image = st.camera_input("Webcam Snapshot (optional)" )
        audio_file = st.file_uploader(
            "Audio Sample (optional)",
            type=["wav", "mp3", "ogg", "m4a"],
            help="Provide a short voice sample for audio emotion analysis.",
        )
        eeg_file = st.file_uploader(
            "EEG CSV (optional)",
            type=["csv"],
            help="Simulated EEG readings to approximate stress levels.",
        )

    job_description = ""
    job_descriptor = None
    if selected_title != "-- None --":
        job_descriptor = jd_options.get(selected_title)
        if job_descriptor:
            job_description = job_descriptor["description"]
    if manual_jd.strip():
        job_description = manual_jd.strip()

    run_analysis = st.button(
        "Analyze Candidate",
        type="primary",
        disabled=not resume_file or not job_description or not info["name"],
    )

    if run_analysis:
        with st.spinner("Processing candidate data..."):
            try:
                resume_text = extract_text_from_pdf(resume_file)
                rag_result = rag_resume_score(job_description, resume_text)
            except Exception as exc:
                st.error(f"Resume scoring failed: {exc}")
                return

            face_report = detect_emotions_from_camera(captured_image)
            audio_report = analyze_audio_emotion(audio_file)
            stress_report = analyze_eeg_signal(eeg_file)

            face_score = emotion_to_score(face_report.dominant_emotion)
            audio_score = normalize_audio_confidence(audio_report.confidence)
            stress_index = normalize_stress_index(stress_report.stress_index)
            final_score = compute_final_suitability(
                rag_result.fit_score, face_score, audio_score, stress_index
            )

            face_summary = summarize_face_emotion(face_report)

            st.subheader("Job Fit Assessment")
            st.progress(min(rag_result.fit_score / 100.0, 1.0) if rag_result.fit_score else 0.0)
            st.metric("Fit Score", f"{rag_result.fit_score:.1f}")
            st.text_area("Summary", rag_result.summary, height=120)
            col_strength, col_gaps = st.columns(2)
            with col_strength:
                st.markdown("**Strengths**")
                st.text(rag_result.strengths)
            with col_gaps:
                st.markdown("**Gaps**")
                st.text(rag_result.gaps)

            st.subheader("Emotion Insights")
            emotion_cols = st.columns(3)
            emotion_cols[0].metric(
                "Face Emotion",
                face_report.dominant_emotion.capitalize(),
                f"Confidence {face_summary['confidence']:.0f}%",
            )
            emotion_cols[1].metric(
                "Audio Emotion",
                audio_report.label.capitalize(),
                f"Confidence {audio_report.confidence:.0f}%",
            )
            emotion_cols[2].metric(
                "Stress Index",
                f"{stress_index:.0f}",
                "Lower is better",
            )
            if face_report.note:
                st.caption(f"Face analysis: {face_report.note}")
            if audio_report.note:
                st.caption(f"Audio analysis: {audio_report.note}")
            if stress_report.note:
                st.caption(f"EEG analysis: {stress_report.note}")

            st.subheader("Final Suitability")
            st.metric("Suitability Score", f"{final_score:.1f}")
            st.progress(min(final_score / 100, 1.0))

            if rag_result.retrieved_chunks:
                with st.expander("Retrieved Resume Chunks"):
                    for idx, chunk in enumerate(rag_result.retrieved_chunks, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.write(chunk)
                        st.divider()

            candidate_payload = {
                "name": info["name"],
                "email": info["email"],
                "position": info["position"],
                "job_description_title": job_descriptor["title"] if job_descriptor else "Manual Entry",
                "job_fit_score": round(rag_result.fit_score, 2),
                "strengths": rag_result.strengths,
                "gaps": rag_result.gaps,
                "summary": rag_result.summary,
                "face_emotion": face_report.dominant_emotion,
                "face_confidence": round(face_summary["confidence"], 2),
                "emotion_stability": round(face_summary["stability"], 2),
                "face_note": face_report.note,
                "audio_emotion": audio_report.label,
                "audio_confidence": round(audio_report.confidence, 2),
                "audio_note": audio_report.note,
                "stress_index": round(stress_index, 2),
                "stress_note": stress_report.note,
                "final_suitability": final_score,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            save_candidate(candidate_payload)
            st.success("Candidate analysis saved.")

            st.download_button(
                "Download Candidate JSON",
                data=json.dumps(candidate_payload, indent=2).encode("utf-8"),
                file_name=f"{info['name'].replace(' ', '_').lower()}_analysis.json",
                mime="application/json",
            )


def recruiter_view():
    st.header("Recruiter Module")
    st.caption("Manage job descriptions, review analyzed candidates, and export reports.")

    with st.expander("Add Job Description"):
        jd_title = st.text_input("Job Title", key="jd_title")
        jd_description = st.text_area("Job Description", key="jd_description")
        if st.button("Save Job Description", key="save_jd"):
            if not jd_title or not jd_description:
                st.warning("Provide both title and description before saving.")
            else:
                entry = save_job_description(jd_title, jd_description)
                st.success(f"Saved job description '{entry['title']}'.")

    job_descriptions = load_job_descriptions()
    if job_descriptions:
        jd_df = pd.DataFrame(job_descriptions)[["title", "created_at"]]
        st.subheader("Stored Job Descriptions")
        st.dataframe(jd_df, use_container_width=True)
    else:
        st.info("No job descriptions saved yet.")

    candidates = load_candidates()
    if candidates:
        st.subheader("Candidate Overview")
        candidates_df = pd.DataFrame(candidates)
        display_cols = [
            "name",
            "position",
            "job_fit_score",
            "emotion_stability",
            "audio_confidence",
            "stress_index",
            "final_suitability",
            "created_at",
        ]
        existing_cols = [col for col in display_cols if col in candidates_df.columns]
        st.dataframe(candidates_df[existing_cols], use_container_width=True)

        csv_bytes = candidates_to_csv(candidates)
        st.download_button(
            "Download Candidates CSV",
            data=csv_bytes,
            file_name="candidates_report.csv",
            mime="text/csv",
        )

        candidate_names = [cand.get("name", "Candidate") for cand in candidates]
        selected_idx = st.selectbox("Choose candidate for PDF report", range(len(candidate_names)), format_func=lambda idx: candidate_names[idx])
        pdf_bytes = candidate_to_pdf(candidates[selected_idx])
        st.download_button(
            "Download Candidate PDF",
            data=pdf_bytes,
            file_name=f"{candidate_names[selected_idx].replace(' ', '_').lower()}_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No candidate analyses stored yet.")


candidate_tab, recruiter_tab = st.tabs(["Candidate", "Recruiter"])
with candidate_tab:
    candidate_view()
with recruiter_tab:
    recruiter_view()
