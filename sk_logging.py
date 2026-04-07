from __future__ import annotations

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os


def project_root() -> Path:
    env = os.environ.get("SAGE_KAIZEN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


def get_logger(name: str, *, file_name: str = "sage_kaizen.log") -> logging.Logger:
    """
    Idempotent rotating-file logger.
    Safe to call repeatedly across Streamlit reruns.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / file_name

    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(handler)
    return logger
