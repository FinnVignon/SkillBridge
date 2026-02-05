from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    SECRET_KEY: str
    DATA_DIR: str
    ADS_DIR: str
    DEBUG: bool = False
    MAX_CONTENT_LENGTH: int = 8 * 1024 * 1024


@dataclass(frozen=True)
class DevConfig(Config):
    DEBUG: bool = True


def load_config() -> Config:
    secret = os.environ.get("FLASK_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "FLASK_SECRET_KEY is required. Set a secure secret key before running the app."
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "export-fiches-csv-2026-02-02"))
    ads_dir = os.path.abspath(os.path.join(base_dir, "..", "ads"))

    env = os.environ.get("FLASK_ENV", "production").lower()
    if env == "development":
        return DevConfig(SECRET_KEY=secret, DATA_DIR=data_dir, ADS_DIR=ads_dir)

    return Config(SECRET_KEY=secret, DATA_DIR=data_dir, ADS_DIR=ads_dir)
