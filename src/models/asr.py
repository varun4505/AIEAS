"""Audio transcription using Groq Whisper API.

This module provides speech-to-text functionality for the AI interview system
using Groq's Whisper API endpoint.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

from groq import Groq

from src.config.settings import get_groq_api_key


def transcribe_audio(audio_data: bytes, filename: str = "audio.wav") -> Dict[str, Any]:
    """Transcribe audio using Groq Whisper API.
    
    Args:
        audio_data: Raw audio bytes (WAV, MP3, M4A, or other supported format)
        filename: Name for the temporary audio file (must have correct extension)
        
    Returns:
        Dictionary with:
        - transcript: str of transcribed text
        - success: bool indicating if transcription succeeded
        - error: str error message if success is False
        
    Example:
        >>> with open("recording.wav", "rb") as f:
        ...     audio_bytes = f.read()
        >>> result = transcribe_audio(audio_bytes, "recording.wav")
        >>> print(result["transcript"])
    """
    api_key = get_groq_api_key()
    if not api_key:
        return {
            "transcript": "",
            "success": False,
            "error": "Groq API key not configured. Please set GROQ_API_KEY environment variable.",
        }
    
    try:
        # Create a temporary file to store the audio
        # Groq API requires a file path, not raw bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Initialize Groq client and transcribe
            client = Groq(api_key=api_key)
            
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(filename, audio_file.read()),
                    model="whisper-large-v3",  # Groq's Whisper model
                    response_format="verbose_json",  # Get detailed response with timestamps
                    language="en",  # Set to None for auto-detection, or specify language code
                    temperature=0.0,  # Lower temperature for more deterministic output
                )
            
            # Extract text from response
            if hasattr(transcription, 'text'):
                transcript_text = transcription.text.strip()
            else:
                transcript_text = str(transcription).strip()
            
            return {
                "transcript": transcript_text,
                "success": True,
                "error": None,
                "language": getattr(transcription, 'language', 'en'),
                "duration": getattr(transcription, 'duration', None),
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as exc:
        return {
            "transcript": "",
            "success": False,
            "error": f"Transcription failed: {str(exc)}",
        }


def transcribe_audio_file(file_path: str) -> Dict[str, Any]:
    """Transcribe audio from a file path.
    
    Args:
        file_path: Path to audio file (WAV, MP3, M4A, etc.)
        
    Returns:
        Dictionary with transcript, success, and error fields
        
    Example:
        >>> result = transcribe_audio_file("interview_audio.mp3")
        >>> if result["success"]:
        ...     print(result["transcript"])
    """
    try:
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        filename = os.path.basename(file_path)
        return transcribe_audio(audio_data, filename)
        
    except Exception as exc:
        return {
            "transcript": "",
            "success": False,
            "error": f"Failed to read audio file: {str(exc)}",
        }


def transcribe_with_fallback(audio_data: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio with automatic fallback to empty string on error.
    
    This is a convenience wrapper that returns just the transcript text,
    or an empty string if transcription fails.
    
    Args:
        audio_data: Raw audio bytes
        filename: Name for temporary audio file
        
    Returns:
        Transcribed text, or empty string if transcription failed
    """
    result = transcribe_audio(audio_data, filename)
    if result["success"]:
        return result["transcript"]
    else:
        # Log error but return empty string
        print(f"ASR Warning: {result.get('error', 'Unknown error')}")
        return ""
