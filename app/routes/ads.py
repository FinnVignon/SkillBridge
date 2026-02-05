from __future__ import annotations

from flask import Blueprint, current_app, send_from_directory

ads_bp = Blueprint("ads", __name__)


@ads_bp.route("/ads/<path:filename>")
def ads(filename: str):
    return send_from_directory(current_app.config["ADS_DIR"], filename)
