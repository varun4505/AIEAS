"""Utilities to export candidate summaries in various formats.

This module attempts to write PDFs using a Unicode TrueType font when
available. If a TTF cannot be registered, the exporter will fall back to
replacing common Unicode characters (bullets, smart quotes, dashes, ellipses)
with ASCII equivalents to avoid FPDFUnicodeEncodingException raised by the
core PDF fonts.
"""
from __future__ import annotations

import io
import os
import re
from typing import Iterable

import pandas as pd


_REPLACEMENTS = {
    "\u2022": "-",  # bullet
    "\u2023": "-",
    "\u2013": "-",  # en dash
    "\u2014": "--",  # em dash
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201C": '"',  # left double quote
    "\u201D": '"',  # right double quote
    "\u2026": "...",  # ellipsis
    "\u00A0": " ",  # non-breaking space
}


def _sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # Replace common Unicode characters with ASCII equivalents
    for k, v in _REPLACEMENTS.items():
        s = s.replace(k, v)
    # Remove other control characters
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    return s


def _find_windows_font() -> str | None:
    """Return a path to a likely TTF on Windows, or None if not found."""
    # Try a short list of common font file names
    candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "DejaVuSans.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "seguiemj.ttf"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def candidates_to_csv(candidates: Iterable[dict]) -> bytes:
    df = pd.DataFrame(list(candidates))
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def candidate_to_pdf(candidate: dict) -> bytes:
    try:
        from fpdf import FPDF
    except Exception:
        # Provide a plain text fallback if FPDF is unavailable.
        buffer = io.StringIO()
        for key, value in candidate.items():
            buffer.write(f"{key}: {value}\n")
        return buffer.getvalue().encode("utf-8")

    pdf = FPDF()
    pdf.add_page()

    # Try to register a Unicode TTF font (if available on the system). This
    # lets us keep Unicode characters intact. If registration fails, we'll
    # fall back to core fonts and sanitize text.
    font_registered = False
    try:
        font_path = _find_windows_font()
        if font_path:
            # add_font requires uni=True for unicode support
            pdf.add_font("LocalUnicode", "", font_path, uni=True)
            pdf.set_font("LocalUnicode", size=12)
            font_registered = True
    except Exception:
        font_registered = False

    if not font_registered:
        # Use a core font but sanitize text to avoid encoding errors
        pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, txt="Candidate Report", ln=True, align="C")
    pdf.ln(4)

    for key, value in candidate.items():
        text = f"{key}: {value}"
        # Always coerce to string first
        text = str(text)

        # Try to write the original text. If the selected font doesn't support
        # some characters, catch the unicode encoding exception and retry with
        # a sanitized version that replaces problematic characters.
        try:
            pdf.multi_cell(0, 8, txt=text)
        except Exception as exc:  # defensive: catch FPDFUnicodeEncodingException
            try:
                # Lazy import of the specific exception class (if available)
                from fpdf.errors import FPDFUnicodeEncodingException

                is_unicode_exc = isinstance(exc, FPDFUnicodeEncodingException)
            except Exception:
                # If we can't import the class, fall back to name check
                is_unicode_exc = "FPDFUnicodeEncodingException" in repr(exc)

            if is_unicode_exc:
                safe = _sanitize_text(text)
                pdf.multi_cell(0, 8, txt=safe)
            else:
                # Re-raise if it's an unrelated exception
                raise

        pdf.ln(1)

    return bytes(pdf.output(dest="S"))
