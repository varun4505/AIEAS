"""Streamlit entry point for the AIEAS prototype."""
from __future__ import annotations

import json
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.exporters import candidate_to_pdf, candidates_to_csv
from src.data.storage import (
    append_interview_event,
    delete_candidate,
    delete_job_description,
    load_candidates,
    load_interview_sessions,
    load_job_descriptions,
    save_candidate,
    save_interview_session,
    save_job_description,
    update_interview_session,
)
from src.models.emotion import (
    analyze_audio_emotion,
    analyze_eeg_signal,
    detect_emotions_from_camera,
    summarize_face_emotion,
)
from src.models.interview_ai import analyze_answer, generate_next_question
from src.models.resume_rag import rag_resume_score
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.scoring import (
    compute_final_suitability,
    emotion_to_score,
    normalize_audio_confidence,
    normalize_stress_index,
)

st.set_page_config(page_title="AIEAS - Interview & Emotion Analysis", layout="wide")


def compute_interview_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize interview progression for dashboard metrics."""

    question_count = sum(1 for event in events if event.get("type") == "question")
    answer_count = sum(1 for event in events if event.get("type") == "answer")
    analysis_events = [event for event in events if event.get("type") == "analysis"]

    scores: List[float] = []
    last_summary = ""
    last_follow_up = ""
    for entry in analysis_events:
        payload = entry.get("payload", {}) or {}
        score = payload.get("overall_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        last_summary = payload.get("summary", last_summary)
        last_follow_up = payload.get("follow_up", last_follow_up)

    average_score = round(sum(scores) / len(scores), 1) if scores else 0.0
    return {
        "questions_asked": question_count,
        "answers_recorded": answer_count,
        "analysis_count": len(analysis_events),
        "average_answer_score": average_score,
        "last_summary": last_summary,
        "last_follow_up": last_follow_up,
    }


def _submit_answer_callback(
    session_id: str,
    open_question_text: str,
    job_description_text: str,
    persona: str,
    answer_key: str,
    auto_followups: bool,
) -> None:
    """Callback triggered by the answer form submit button.

    Reads the answer from st.session_state[answer_key], persists the answer,
    runs analysis, appends events and clears the state for the widget safely.
    """
    try:
        answer_text = st.session_state.get(answer_key, "").strip()
        if not answer_text:
            st.warning("Capture an answer before submitting.")
            return

        answer_event = {
            "type": "answer",
            "actor": "candidate",
            "payload": {"transcript": answer_text, "word_count": len(answer_text.split())},
        }
        session_after_answer = append_interview_event(session_id, answer_event)
        if not session_after_answer:
            st.warning("Unable to persist answer. Check storage configuration.")
            return

        analysis_result = analyze_answer(
            question=open_question_text or "",
            answer=answer_text,
            job_description=job_description_text,
            persona=persona,
        )
        session_after_analysis = append_interview_event(
            session_id, {"type": "analysis", "actor": "ai", "payload": analysis_result}
        )
        if session_after_analysis:
            metrics = compute_interview_metrics(session_after_analysis.get("events", []))
            update_interview_session(session_id, {"metrics": metrics})
            follow_up_text = (analysis_result.get("follow_up") or "").strip()
            if follow_up_text and auto_followups:
                appended = append_interview_event(
                    session_id,
                    {
                        "type": "question",
                        "actor": "ai",
                        "payload": {"text": follow_up_text, "origin": "auto_follow_up", "persona": persona},
                    },
                )
                if not appended:
                    st.warning("Follow-up question could not be stored.")

            # Clear the widget-backed state safely
            st.session_state[answer_key] = ""
            st.toast("Answer logged and evaluated.")
            # Trigger a rerun so UI updates consistently
            st.experimental_rerun()
        else:
            st.warning("Answer saved but analysis failed to persist.")
    except Exception as exc:
        st.error(f"Answer handling failed: {exc}")


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
        github_url = st.text_input("GitHub username or URL (optional)")
        linkedin_url = st.text_input("LinkedIn profile URL (optional)")
        resume_file = st.file_uploader("Resume (PDF)", type=["pdf"], help="Only PDF resumes are supported.")
    with col_right:
        captured_image = st.camera_input("Webcam Snapshot (optional)" )
        # Temporarily hidden: Audio and EEG inputs
        audio_file = None  # st.file_uploader("Audio Sample (optional)", type=["wav", "mp3", "ogg", "m4a"])
        eeg_file = None  # st.file_uploader("EEG CSV (optional)", type=["csv"])

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
                # Import profile helpers defensively: some environments may not
                # have BeautifulSoup/requests or the helper may fail to import.
                try:
                    from src.utils.online_profiles import (
                        fetch_github_profile,
                        fetch_linkedin_profile,
                        extract_profile_links,
                    )
                except Exception:
                    # Fall back to minimal no-op implementations so resume
                    # scoring can continue even if profile helpers are missing.
                    def fetch_github_profile(_identifier: str) -> str:  # type: ignore
                        return ""

                    def fetch_linkedin_profile(_url: str) -> str:  # type: ignore
                        return ""

                    def extract_profile_links(_text: str) -> dict:  # type: ignore
                        return {}

                extracted = extract_profile_links(resume_text)
                if not github_url and extracted.get("github"):
                    github_url = extracted["github"]
                if not linkedin_url and extracted.get("linkedin"):
                    linkedin_url = extracted["linkedin"]

                gh_text = fetch_github_profile(github_url) if github_url else ""
                li_text = fetch_linkedin_profile(linkedin_url) if linkedin_url else ""
                combined_text = "\n\n".join(p for p in [resume_text, gh_text, li_text] if p)
                rag_result = rag_resume_score(job_description, combined_text)
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

            normalized_github_link = github_url.strip() if github_url else ""
            if normalized_github_link and not normalized_github_link.lower().startswith("http"):
                normalized_github_link = f"https://github.com/{normalized_github_link.lstrip('@')}"
            normalized_linkedin_link = linkedin_url.strip() if linkedin_url else ""
            if normalized_linkedin_link and not normalized_linkedin_link.lower().startswith("http"):
                normalized_linkedin_link = f"https://{normalized_linkedin_link.lstrip('@')}"

                candidate_payload = {
                "name": info["name"],
                "email": info["email"],
                "position": info["position"],
                    "resume_text": resume_text,
                    "extracted_links": extracted,
                "job_description_title": job_descriptor["title"] if job_descriptor else "Manual Entry",
                "job_fit_score": round(rag_result.fit_score, 2),
                "strengths": rag_result.strengths,
                "gaps": rag_result.gaps,
                "summary": rag_result.summary,
                "retrieved_chunks": rag_result.retrieved_chunks,
                "github_url": normalized_github_link,
                "linkedin_url": normalized_linkedin_link,
                "github_profile_text": gh_text,
                "linkedin_profile_text": li_text,
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
            st.success("Thanks! Your submission has been sent to the recruiting team.")
            st.info("You will hear from us once the review is complete.")


def interview_view():
    st.header("AI-Powered Interview Module")
    st.caption("Run and review real-time AI-led interviews with stress and scoring overlays.")

    job_descriptions = load_job_descriptions()
    candidates = load_candidates()
    candidate_options = {cand.get("name", "Unnamed"): cand for cand in candidates}
    jd_by_title = {jd.get("title", "Untitled"): jd for jd in job_descriptions}
    jd_by_id = {jd.get("id"): jd for jd in job_descriptions}
    title_options = ["-- None --", *jd_by_title.keys()]

    with st.form("interview_setup_form"):
        setup_cols = st.columns([2, 1])
        with setup_cols[0]:
            # Allow selecting an existing candidate to reuse the RAG/profile
            selected_candidate_name = st.selectbox(
                "Reuse Existing Candidate",
                options=["-- None --", *candidate_options.keys()],
                index=0,
            )
            candidate_name = st.text_input("Candidate Name", help="Display name for the interview room.")
            candidate_identifier = st.text_input("Candidate ID (optional)")
            target_role = st.text_input("Role / Position (optional)", help="Optional: useful for tailoring technical questions but not required to start.")
            selected_jd_title = st.selectbox(
                "Attach Job Description",
                options=title_options,
                index=0,
            )
        with setup_cols[1]:
            ai_persona = st.selectbox(
                "AI Interview Persona",
                options=[
                    "Structured Behavioral Coach",
                    "Deep Technical Challenger",
                    "Friendly Culture Evaluator",
                ],
            )
            duration_minutes = st.slider("Planned Duration (minutes)", min_value=10, max_value=90, value=30)
            auto_followups = st.checkbox("Auto ask suggested follow-ups", value=True)
            stress_sources = st.multiselect(
                "Stress Signals",
                options=["video", "audio", "text"],
                default=["video", "audio"],
            )

        create_session = st.form_submit_button(
            "Create Interview Session",
            type="primary",
        )

        if create_session:
        # Validate required fields server-side to avoid Streamlit form disabled UI edge cases
            if not candidate_name or not candidate_name.strip():
                # If user selected an existing candidate, populate name from that record
                if selected_candidate_name and selected_candidate_name != "-- None --":
                    candidate_obj = candidate_options.get(selected_candidate_name)
                    candidate_name = candidate_obj.get("name") if candidate_obj else candidate_name
                else:
                    st.warning("Candidate Name is required to create a session. Please enter the candidate's name.")
                    st.stop()

            # If a candidate was selected, attach their retrieved chunks (more efficient than full text)
            candidate_retrieved_chunks = None
            if selected_candidate_name and selected_candidate_name != "-- None --":
                candidate_obj = candidate_options.get(selected_candidate_name)
                if candidate_obj:
                    candidate_retrieved_chunks = candidate_obj.get("retrieved_chunks", [])
            now_iso = datetime.utcnow().isoformat() + "Z"
            job_descriptor = jd_by_title.get(selected_jd_title)
            job_description_text = job_descriptor.get("description", "") if job_descriptor else ""
            session_payload = {
                "id": str(uuid.uuid4()),
                "candidate_id": candidate_identifier or None,
                "candidate_name": candidate_name,
                "position": target_role,
                "job_description_id": job_descriptor.get("id") if job_descriptor else None,
                "status": "scheduled",
                "created_at": now_iso,
                "updated_at": now_iso,
                "settings": {
                    "ai_persona": ai_persona,
                    "duration_minutes": duration_minutes,
                    "auto_followups": auto_followups,
                    "stress_sources": stress_sources,
                    "job_description": job_description_text,
                    "job_description_title": job_descriptor.get("title") if job_descriptor else None,
                    "candidate_retrieved_chunks": candidate_retrieved_chunks,
                },
                "participants": [
                    {"role": "candidate", "display_name": candidate_name, "metadata": {}},
                    {
                        "role": "ai",
                        "display_name": ai_persona,
                        "metadata": {"model": "groq/llama-3.1-8b-instant"},
                    },
                ],
                "events": [],
                "metrics": {},
                "recordings": {
                    "video_path": None,
                    "audio_path": None,
                    "transcript_path": None,
                },
            }
            saved = save_interview_session(session_payload)
            if saved.get("id"):
                st.session_state["active_interview_session_id"] = saved["id"]
            st.success("Interview session created.")
            st.rerun()

    sessions = load_interview_sessions()
    if not sessions:
        st.info("No interview sessions stored yet. Configure the form above to create one.")
        return

    session_labels: List[str] = []
    preselected_idx = 0
    active_id = st.session_state.get("active_interview_session_id")
    for idx, session in enumerate(sessions):
        metrics_preview = session.get("metrics") or compute_interview_metrics(session.get("events", []))
        label = (
            f"{session.get('candidate_name', 'Unnamed')} • "
            f"{session.get('status', 'draft').title()} • "
            f"{metrics_preview.get('questions_asked', 0)} Q"
        )
        session_labels.append(label)
        if active_id and session.get("id") == active_id:
            preselected_idx = idx

    selected_idx = st.selectbox(
        "Select interview session",
        options=list(range(len(sessions))),
        format_func=lambda idx: session_labels[idx],
        index=preselected_idx,
    )

    selected_session = sessions[selected_idx]
    session_id = selected_session.get("id")
    if not session_id:
        st.warning("Selected session is missing an identifier. Please recreate the session.")
        return
    settings = selected_session.get("settings", {}) or {}
    events = selected_session.get("events", [])
    status = selected_session.get("status", "draft")
    persona = settings.get("ai_persona", "Structured Behavioral Coach")
    job_description_text = settings.get("job_description", "")
    if not job_description_text and selected_session.get("job_description_id"):
        descriptor = jd_by_id.get(selected_session["job_description_id"])
        if descriptor:
            job_description_text = descriptor.get("description", "")

    live_metrics = compute_interview_metrics(events)

    meta_cols = st.columns(4)
    meta_cols[0].metric("Status", status.title())
    meta_cols[1].metric("Questions", live_metrics["questions_asked"])
    meta_cols[2].metric("Answers", live_metrics["answers_recorded"])
    meta_cols[3].metric("Avg Score", f"{live_metrics['average_answer_score']:.1f}")

    control_cols = st.columns(3)
    now_iso = datetime.utcnow().isoformat() + "Z"
    if status in {"draft", "scheduled"}:
        if control_cols[0].button("Start Interview", key=f"start_{session_id}"):
            update_interview_session(session_id, {"status": "active", "started_at": now_iso})
            st.session_state["active_interview_session_id"] = session_id
            st.rerun()
    elif status == "active":
        if control_cols[0].button("End Interview", type="secondary", key=f"end_{session_id}"):
            update_interview_session(session_id, {"status": "completed", "ended_at": now_iso})
            st.toast("Interview marked as completed.")
            st.rerun()
        control_cols[1].button("Pause (coming soon)", disabled=True)
    else:
        if control_cols[0].button("Reopen Interview", key=f"reopen_{session_id}"):
            update_interview_session(session_id, {"status": "active"})
            st.session_state["active_interview_session_id"] = session_id
            st.rerun()

    with st.expander("Job Description Context", expanded=False):
        st.write(job_description_text or "No job description attached.")

    open_question = events[-1] if events and events[-1].get("type") == "question" else None
    latest_analysis = next((event for event in reversed(events) if event.get("type") == "analysis"), None)

    st.subheader("Live Interview Canvas")
    canvas_cols = st.columns([2, 1])
    with canvas_cols[0]:
        st.markdown("**Current Question**")
        if open_question:
            st.write(open_question.get("payload", {}).get("text", "Awaiting question text."))
        else:
            st.info("No active question. Generate the next one when you're ready.")

        if status == "active":
            if not open_question:
                if st.button("Generate Next Question", key=f"generate_{session_id}", use_container_width=True):
                    try:
                        question_text = generate_next_question(
                            candidate_name=selected_session.get("candidate_name", ""),
                            position=selected_session.get("position", ""),
                            job_description=job_description_text,
                            persona=persona,
                            events=events,
                            candidate_retrieved_chunks=selected_session.get("settings", {}).get("candidate_retrieved_chunks") or None,
                        )
                        question_event = {
                            "type": "question",
                            "actor": "ai",
                            "payload": {
                                "text": question_text,
                                "persona": persona,
                            },
                        }
                        updated = append_interview_event(session_id, question_event)
                        if updated:
                            st.toast("Question delivered to the candidate.")
                            st.rerun()
                        else:
                            st.warning("Unable to persist the question. Check storage configuration.")
                    except Exception as exc:
                        st.error(f"Question generation failed: {exc}")
            else:
                answer_key = f"answer_input_{session_id}"
                if answer_key not in st.session_state:
                    st.session_state[answer_key] = ""
                st.divider()
                st.markdown("**Capture Candidate Answer**")
                
                # Answer input tabs: Manual entry vs. Automatic capture
                answer_tabs = st.tabs(["Manual Entry", "Auto Capture (ASR + Video)"])
                
                with answer_tabs[0]:
                    with st.form(f"answer_form_{session_id}"):
                        st.text_area("Type or paste the candidate's answer", key=answer_key, height=180)
                        # Use the safe callback to handle submission and state clearing
                        st.form_submit_button(
                            "Submit Answer",
                            use_container_width=True,
                            on_click=_submit_answer_callback,
                            args=(
                                session_id,
                                (open_question.get("payload", {}).get("text", "") if open_question else ""),
                                job_description_text,
                                persona,
                                answer_key,
                                settings.get("auto_followups", True),
                            ),
                        )
                
                with answer_tabs[1]:
                    st.info("✨ **Automatic Interview Capture** using Groq Whisper API for speech-to-text")
                    
                    # Camera input for video stress analysis
                    st.markdown("**📹 Video Capture**")
                    camera_image = st.camera_input("Enable camera for stress monitoring", key=f"camera_{session_id}")
                    if camera_image:
                        st.caption("✅ Camera feed active. Stress analysis will process frames in real-time.")
                        # TODO: Wire this to stress detection model
                        # Example workflow:
                        # 1. Convert camera_image to numpy array using PIL/OpenCV
                        # 2. Extract face using MediaPipe or cv2.CascadeClassifier
                        # 3. Run stress model inference (CNN/ViT) on face crops
                        # 4. Append StressSnapshot to interview event with source="video"
                        # 5. Display stress level indicator in UI
                    
                    st.markdown("**🎤 Audio Transcription (Groq Whisper API)**")
                    st.caption("Upload an audio recording to transcribe using Groq Whisper API")
                    
                    # Audio file uploader for ASR
                    audio_file = st.file_uploader(
                        "Upload Audio Recording",
                        type=["wav", "mp3", "m4a", "ogg", "flac"],
                        key=f"audio_upload_{session_id}",
                        help="Record your answer using any audio recorder and upload here for automatic transcription"
                    )
                    
                    if audio_file is not None:
                        st.audio(audio_file, format=f"audio/{audio_file.type.split('/')[-1]}")
                        
                        if st.button("🔤 Transcribe with Groq Whisper", key=f"transcribe_{session_id}", use_container_width=True):
                            with st.spinner("Transcribing audio using Groq Whisper API..."):
                                try:
                                    from src.models.asr import transcribe_audio
                                    
                                    # Read audio bytes
                                    audio_bytes = audio_file.read()
                                    audio_file.seek(0)  # Reset file pointer for audio player
                                    
                                    # Transcribe using Groq Whisper
                                    result = transcribe_audio(audio_bytes, audio_file.name)
                                    
                                    if result["success"]:
                                        transcribed_text = result["transcript"]
                                        
                                        # Store transcription in session state with a unique key
                                        transcription_key = f"transcription_{session_id}"
                                        st.session_state[transcription_key] = transcribed_text
                                        
                                        st.success("✅ Transcription complete!")
                                        st.text_area(
                                            "Transcribed Text (edit if needed)",
                                            value=transcribed_text,
                                            height=150,
                                            key=f"transcript_display_{session_id}"
                                        )
                                        
                                        # Show detected language and duration if available
                                        info_cols = st.columns(2)
                                        if result.get("language"):
                                            info_cols[0].caption(f"🌐 Language: {result['language']}")
                                        if result.get("duration"):
                                            info_cols[1].caption(f"⏱️ Duration: {result['duration']:.1f}s")
                                        
                                        # Button to use this transcription as the answer
                                        if st.button("✓ Use This Transcription as Answer", key=f"use_transcript_{session_id}", type="primary"):
                                            # Copy transcription to answer field and trigger submission
                                            st.session_state[answer_key] = transcribed_text
                                            st.toast("Transcription copied to answer field. Please submit manually from the Manual Entry tab.")
                                            st.rerun()
                                    else:
                                        st.error(f"❌ Transcription failed: {result.get('error', 'Unknown error')}")
                                        st.info("Make sure your GROQ_API_KEY is set correctly in .env file")
                                        
                                except Exception as exc:
                                    st.error(f"❌ Error during transcription: {str(exc)}")
                                    import traceback
                                    st.code(traceback.format_exc())
                    
                    # Display current answer if transcription was used
                    if st.session_state.get(answer_key):
                        st.divider()
                        st.success("✅ Answer ready for submission. Switch to 'Manual Entry' tab to review and submit.")
                    
                    st.divider()
                    st.caption(
                        "**✅ Active Features:**\n"
                        "• Audio transcription via Groq Whisper API (upload .wav, .mp3, .m4a files)\n"
                        "• Camera feed capture for future stress analysis\n\n"
                        "**🚧 Coming Soon:**\n"
                        "• Real-time audio recording with `streamlit-webrtc` or `streamlit-audio-recorder`\n"
                        "• Video stress detection model (see `docs/ASR_VIDEO_SETUP.md` for training guide)\n"
                        "• Prosody analysis for confidence scoring\n"
                        "5. Wire audio stream to `process_audio_stream()` helper\n"
                        "6. Enable HTTPS for WebRTC (required for camera/mic access in production)"
                    )
        else:
            st.info("Start the interview to begin asking questions.")

    with canvas_cols[1]:
        st.metric("Average Score", f"{live_metrics['average_answer_score']:.1f}")
        st.metric("Questions Asked", live_metrics["questions_asked"])
        st.metric("Analyses", live_metrics["analysis_count"])
        if latest_analysis:
            payload = latest_analysis.get("payload", {}) or {}
            st.markdown("**Latest Evaluation**")
            st.metric("Latest Score", f"{payload.get('overall_score', 0):.1f}")
            st.write(payload.get("summary", "No summary provided."))
            strengths = [item for item in payload.get("strengths", []) if item]
            if strengths:
                st.markdown("**Strengths**")
                st.markdown("\n".join(f"- {item}" for item in strengths))
            risks = [item for item in payload.get("risks", []) if item]
            if risks:
                st.markdown("**Risks / Watch-outs**")
                st.markdown("\n".join(f"- {item}" for item in risks))
            if payload.get("follow_up"):
                st.caption(f"Suggested follow-up: {payload['follow_up']}")
        else:
            st.caption("Answer evaluations will display here after the first response.")

    timeline_rows = []
    for event in events:
        payload = event.get("payload", {}) or {}
        summary = payload.get("text") or payload.get("summary") or payload.get("message") or ""
        score = payload.get("overall_score") if event.get("type") == "analysis" else None
        timeline_rows.append(
            {
                "Time": event.get("created_at"),
                "Actor": event.get("actor"),
                "Type": event.get("type"),
                "Summary": summary,
                "Score": score,
            }
        )

    st.subheader("Interview Timeline")
    if timeline_rows:
        timeline_df = pd.DataFrame(timeline_rows)
        st.dataframe(timeline_df, use_container_width=True)
    else:
        st.caption("Timeline will populate as the interview progresses.")


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

        jd_ids = [jd.get("id") for jd in job_descriptions]
        delete_idx = st.selectbox(
            "Select job description to delete",
            options=list(range(len(job_descriptions))),
            format_func=lambda idx: job_descriptions[idx].get("title", "Untitled"),
        )
        if st.button("Delete Job Description", type="secondary", key="delete_jd"):
            entry_id = jd_ids[delete_idx]
            if entry_id and delete_job_description(entry_id):
                st.success("Job description removed.")
                st.rerun()
            else:
                st.warning("Unable to delete job description. Please try again.")
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

        candidate_ids = [cand.get("id") for cand in candidates]
        if st.button("Delete Candidate", type="secondary", key="delete_candidate"):
            entry_id = candidate_ids[selected_idx]
            if entry_id and delete_candidate(entry_id):
                st.success("Candidate removed.")
                st.rerun()
            else:
                st.warning("Unable to delete candidate. Please try again.")

        # Allow recruiters to fetch or refresh online profile summaries for a candidate
        if st.button("Fetch/Refresh Online Profiles", key="refresh_profiles"):
            from src.utils.online_profiles import fetch_github_profile, fetch_linkedin_profile
            from src.data.storage import update_candidate

            sc = candidates[selected_idx]
            gh = ""
            li = ""
            if sc.get("github_url"):
                gh = fetch_github_profile(sc.get("github_url"))
            if sc.get("linkedin_url"):
                li = fetch_linkedin_profile(sc.get("linkedin_url"))

            patched = {}
            if gh:
                patched["github_profile_text"] = gh
            if li:
                patched["linkedin_profile_text"] = li

            if patched and sc.get("id"):
                updated = update_candidate(sc.get("id"), patched)
                if updated:
                    st.success("Profiles fetched and saved.")
                    st.experimental_rerun()
                else:
                    st.warning("Failed to persist profiles to storage.")
            else:
                st.info("No profile URLs present or nothing new to fetch.")

        selected_candidate = candidates[selected_idx]
        st.subheader(f"Detailed Analytics: {selected_candidate.get('name', 'Candidate')}")

        job_fit = selected_candidate.get("job_fit_score", 0.0)
        final_score = selected_candidate.get("final_suitability", 0.0)
        emotion_cols = st.columns(3)
        with emotion_cols[0]:
            st.metric("Job Fit", f"{job_fit:.1f}")
            st.progress(min(job_fit / 100.0, 1.0))
        with emotion_cols[1]:
            st.metric("Suitability", f"{final_score:.1f}")
            st.progress(min(final_score / 100.0, 1.0))
        with emotion_cols[2]:
            st.metric(
                "Stress Index",
                f"{selected_candidate.get('stress_index', 0):.0f}",
                "Lower is better",
            )

        st.markdown("**Summary**")
        st.write(selected_candidate.get("summary", "No summary provided."))

        link_parts = []
        if (gh_url := selected_candidate.get("github_url")):
            link_parts.append(f"[GitHub Profile]({gh_url})")
        if (li_url := selected_candidate.get("linkedin_url")):
            link_parts.append(f"[LinkedIn Profile]({li_url})")
        if link_parts:
            st.markdown(" | ".join(link_parts))

        strengths_gaps_cols = st.columns(2)
        with strengths_gaps_cols[0]:
            st.markdown("**Strengths**")
            st.text(selected_candidate.get("strengths", "- Not specified"))
        with strengths_gaps_cols[1]:
            st.markdown("**Gaps**")
            st.text(selected_candidate.get("gaps", "- Not specified"))

        emotion_detail_cols = st.columns(3)
        emotion_detail_cols[0].metric(
            "Face Emotion",
            str(selected_candidate.get("face_emotion", "unknown")).capitalize(),
            f"Confidence {selected_candidate.get('face_confidence', 0):.0f}%",
        )
        emotion_detail_cols[1].metric(
            "Audio Emotion",
            str(selected_candidate.get("audio_emotion", "unknown")).capitalize(),
            f"Confidence {selected_candidate.get('audio_confidence', 0):.0f}%",
        )
        emotion_detail_cols[2].metric(
            "Emotion Stability",
            f"{selected_candidate.get('emotion_stability', 0):.0f}",
        )

        if note := selected_candidate.get("face_note"):
            st.caption(f"Face analysis: {note}")
        if note := selected_candidate.get("audio_note"):
            st.caption(f"Audio analysis: {note}")
        if note := selected_candidate.get("stress_note"):
            st.caption(f"EEG analysis: {note}")

        if selected_candidate.get("github_profile_text"):
            with st.expander("GitHub Profile Highlights"):
                st.write(selected_candidate["github_profile_text"])
        if selected_candidate.get("linkedin_profile_text"):
            with st.expander("LinkedIn Profile Highlights"):
                st.write(selected_candidate["linkedin_profile_text"])

        if selected_candidate.get("resume_text"):
            with st.expander("Full Resume Text"):
                st.text_area(
                    "Resume content",
                    value=selected_candidate["resume_text"],
                    height=300,
                    disabled=True,
                    key=f"resume_text_{selected_idx}",
                )
                extracted = selected_candidate.get("extracted_links") or {}
                if extracted:
                    st.markdown("**Extracted Links**")
                    st.json(extracted)

        if selected_candidate.get("retrieved_chunks"):
            with st.expander("Retrieved Resume Chunks"):
                for idx, chunk in enumerate(selected_candidate["retrieved_chunks"], start=1):
                    st.markdown(f"**Chunk {idx}**")
                    st.write(chunk)
                    st.divider()

        st.download_button(
            "Download Candidate JSON",
            data=json.dumps(selected_candidate, indent=2).encode("utf-8"),
            file_name=f"{candidate_names[selected_idx].replace(' ', '_').lower()}_analysis.json",
            mime="application/json",
            key="json_download_recruiter",
        )
    else:
        st.info("No candidate analyses stored yet.")

    st.subheader("Interview Sessions")
    sessions = load_interview_sessions()
    if not sessions:
        st.caption("No AI-led interviews have been recorded yet.")
        return

    session_rows: List[Dict[str, Any]] = []
    computed_metrics: Dict[str, Dict[str, Any]] = {}
    for session in sessions:
        session_id = session.get("id", "")
        events = session.get("events", [])
        metrics = session.get("metrics") or compute_interview_metrics(events)
        computed_metrics[session_id] = metrics
        session_rows.append(
            {
                "Candidate": session.get("candidate_name", "Unnamed"),
                "Status": session.get("status", "draft").title(),
                "Questions": metrics.get("questions_asked", 0),
                "Average Score": metrics.get("average_answer_score", 0.0),
                "Last Update": session.get("updated_at") or session.get("created_at"),
                "Session ID": session_id,
            }
        )

    overview_df = pd.DataFrame(session_rows)
    st.dataframe(overview_df.drop(columns=["Session ID"]), use_container_width=True)

    selected_session_id = st.selectbox(
        "Select session to review",
        options=[row["Session ID"] for row in session_rows],
        format_func=lambda sid: next(
            (f"{row['Candidate']} • {row['Status']}" for row in session_rows if row["Session ID"] == sid),
            sid or "Unnamed Session",
        ),
    )

    session_detail = next((session for session in sessions if session.get("id") == selected_session_id), None)
    if not session_detail:
        st.warning("Unable to load session details; it may have been deleted.")
        return

    detail_metrics = computed_metrics.get(selected_session_id, compute_interview_metrics(session_detail.get("events", [])))
    detail_cols = st.columns(4)
    detail_cols[0].metric("Status", session_detail.get("status", "draft").title())
    detail_cols[1].metric("Questions", detail_metrics.get("questions_asked", 0))
    detail_cols[2].metric("Analyses", detail_metrics.get("analysis_count", 0))
    detail_cols[3].metric("Avg Score", f"{detail_metrics.get('average_answer_score', 0.0):.1f}")

    st.markdown("**Latest Summary**")
    st.write(detail_metrics.get("last_summary") or "No summary recorded yet.")
    if detail_metrics.get("last_follow_up"):
        st.caption(f"Suggested follow-up: {detail_metrics['last_follow_up']}")

    events = session_detail.get("events", [])
    if events:
        timeline = []
        for event in events:
            payload = event.get("payload", {}) or {}
            timeline.append(
                {
                    "Time": event.get("created_at"),
                    "Actor": event.get("actor"),
                    "Type": event.get("type"),
                    "Summary": payload.get("text") or payload.get("summary") or payload.get("message"),
                    "Score": payload.get("overall_score") if event.get("type") == "analysis" else None,
                }
            )
        timeline_df = pd.DataFrame(timeline)
        st.markdown("**Timeline**")
        st.dataframe(timeline_df, use_container_width=True)
    else:
        st.caption("No events recorded for this session yet.")

    st.download_button(
        "Download Session JSON",
        data=json.dumps(session_detail, indent=2).encode("utf-8"),
        file_name=f"interview_session_{selected_session_id}.json",
        mime="application/json",
        key=f"download_session_{selected_session_id}",
    )


# Temporarily hidden: AI Interview tab
candidate_tab, recruiter_tab = st.tabs(["Candidate", "Recruiter"])
with candidate_tab:
    candidate_view()
# with interview_tab:
#     interview_view()
with recruiter_tab:
    recruiter_view()
