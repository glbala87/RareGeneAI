"""Structured logging for RareGeneAI."""

import sys

from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure structured logging."""
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
        )

    logger.info(f"RareGeneAI logging initialized at {level} level")
