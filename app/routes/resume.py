from __future__ import annotations

from flask import Blueprint, current_app, render_template, request

from services.pdf_extract import extract_pdf_text
from services.store import get_store

resume_bp = Blueprint("resume", __name__)


@resume_bp.route("/resume", methods=["GET", "POST"])
def resume():
    error = ""
    results = []
    keywords = []
    fallback_results = []

    if request.method == "POST":
        file = request.files.get("cv")
        if not file or not file.filename:
            error = "Veuillez ajouter un CV au format .pdf."
        elif not file.filename.lower().endswith(".pdf"):
            error = "Format non supporté. Utilisez un fichier .pdf."
        else:
            file_bytes = file.read()
            text = extract_pdf_text(file_bytes, current_app.logger)
            if not text.strip():
                error = "Impossible de lire ce PDF. Essayez un autre fichier."
            else:
                store = get_store(current_app)
                query_by_cat = store.parse_resume_text(text)
                results, keywords, fallback_results = store.search_with_fallback(query_by_cat)

    return render_template(
        "resume.html",
        results=results,
        fallback_results=fallback_results,
        keywords=keywords,
        error=error,
        done=bool(results or fallback_results or error),
        total_steps=6,
    )
