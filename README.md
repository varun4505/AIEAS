# AIEAS - Automated Interview & Emotion Analysis System

Prototype Streamlit application that assists recruiters with automated resume screening and multimodal emotion-aware analysis during interviews.

## Features

- Candidate workflow with resume upload, Groq-powered RAG scoring, and multimodal emotion analysis (video, audio, optional EEG).
- Recruiter workflow for managing job descriptions, reviewing candidate scores, and exporting reports.
- Modular Python package layout (`src/`) to simplify extension and testing.

## Getting Started

1. Create a virtual environment and install dependencies from `requirements.txt`.
2. Copy `.env.example` to `.env` and add your Groq API key.
3. Run the Streamlit app:
   ```bash
   streamlit run src/app/main.py
   ```

## Notes

- Emotion detection functions provide graceful fallbacks when camera/audio hardware or optional libraries are not available.
- Candidate and job description data is persisted locally under `storage/` in JSON format for easy prototyping.
- The Groq API is used for large language model inference; ensure your API key has access to LLaMA 3 or Mistral models.
