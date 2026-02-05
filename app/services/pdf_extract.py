from __future__ import annotations

import io
import logging


def extract_pdf_text(file_bytes: bytes, logger: logging.Logger) -> str:
    try:
        import pdfplumber  # type: ignore

        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as exc:
        logger.warning("pdfplumber failed: %s", exc)
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception as fallback_exc:
            logger.warning("PyPDF2 failed: %s", fallback_exc)
            return ""
