"""Utilities for working with PDF files."""
from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    import fitz  # type: ignore
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore


def extract_text_from_pdf(file_obj: BinaryIO | bytes | Path) -> str:
    """Extract raw text content from a PDF upload.

    Parameters
    ----------
    file_obj:
        A file-like object, raw bytes, or path representing the resume PDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError("PyMuPDF is required for PDF text extraction") from exc

    if isinstance(file_obj, Path):
        data_bytes = file_obj.read_bytes()
    elif isinstance(file_obj, (bytes, bytearray)):
        data_bytes = bytes(file_obj)
    else:
        data_bytes = file_obj.read()

    with fitz.open(stream=data_bytes, filetype="pdf") as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    if text.strip():
        return text

    # Fallback to pdfminer if PyMuPDF returns empty output.
    try:
        from pdfminer.high_level import extract_text
    except ImportError as exc:  # pragma: no cover - fallback when pdfminer absent
        raise RuntimeError("PDF extraction failed and pdfminer.six is unavailable") from exc

    buffer = io.BytesIO(data_bytes)
    return extract_text(buffer)
