"""Logging utilities for safe API key handling and error tracking."""

import logging
import os


def get_logger() -> logging.Logger:
    """Get or create the application logger."""
    logger = logging.getLogger("tap_ex")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), "tap_ex.log")

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Also emit to console (shows up in Streamlit logs)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def safe_key_fingerprint(key: str) -> str:
    """Return a safe fingerprint of an API key for logging (never the full key)."""
    if not isinstance(key, str) or not key:
        return "<empty>"
    tail = key[-4:] if len(key) >= 4 else key
    return f"len={len(key)} tail=***{tail}"


def looks_like_auth_error(message: str) -> bool:
    """Check if an error message suggests an authentication/API key problem."""
    m = (message or "").lower()
    return any(
        s in m
        for s in [
            "api key",
            "invalid api key",
            "invalid key",
            "unauthorized",
            "permission denied",
            "forbidden",
            "401",
            "403",
        ]
    )
