from __future__ import annotations

import logging

from flask import Flask, render_template

from config import load_config
from routes.about import about_bp
from routes.ads import ads_bp
from routes.chat import chat_bp
from routes.home import home_bp
from routes.resume import resume_bp
from services.store import register_template_filters


def configure_logging(app: Flask) -> None:
    level = logging.DEBUG if app.config.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(error):
        app.logger.warning("404 not found: %s", error)
        return render_template("error.html", code=404, message="Page introuvable"), 404

    @app.errorhandler(500)
    def server_error(error):
        app.logger.exception("500 server error: %s", error)
        return (
            render_template(
                "error.html",
                code=500,
                message="Une erreur est survenue. Réessayez plus tard.",
            ),
            500,
        )


config = load_config()
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(config)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

configure_logging(app)
register_template_filters(app)

app.register_blueprint(home_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(resume_bp)
app.register_blueprint(about_bp)
app.register_blueprint(ads_bp)

register_error_handlers(app)

app.logger.info("Preloading RNCP data and indices...")
try:
    from services.store import get_store

    get_store(app)
    app.logger.info("Preload complete.")
except Exception as exc:
    app.logger.exception("Preload failed: %s", exc)


if __name__ == "__main__":
    app.run(debug=app.config.get("DEBUG", False))
