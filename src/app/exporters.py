"""Utilities to export candidate summaries in various formats."""
from __future__ import annotations

import io
from typing import Iterable, List

import pandas as pd


def candidates_to_csv(candidates: Iterable[dict]) -> bytes:
    df = pd.DataFrame(list(candidates))
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def candidate_to_pdf(candidate: dict) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        # Provide a plain text fallback if FPDF is unavailable.
        buffer = io.StringIO()
        for key, value in candidate.items():
            buffer.write(f"{key}: {value}\n")
        return buffer.getvalue().encode("utf-8")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Candidate Report", ln=True, align="C")
    pdf.ln(4)

    for key, value in candidate.items():
        pdf.multi_cell(0, 8, txt=f"{key}: {value}")
        pdf.ln(1)

    return bytes(pdf.output(dest="S"))
