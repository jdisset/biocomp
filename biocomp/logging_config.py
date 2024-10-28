import logging
from rich.logging import RichHandler
from typing import Optional
from pathlib import Path

DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    default_level: int = logging.INFO, log_file: Optional[Path] = None, rich_logging: bool = True
) -> None:
    """Configure logging for the biocomp project.

    Args:
        default_level: Default logging level for all loggers
        log_file: Optional file path to write logs to
        rich_logging: Whether to use Rich formatting for console output
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup handlers
    handlers = []
    if rich_logging:
        console_handler = RichHandler(
            show_path=True, omit_repeated_times=False, log_time_format=DEFAULT_DATE_FORMAT
        )
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT))
        handlers.append(file_handler)

    # Configure root logger
    root_logger.setLevel(default_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # Set default levels for external libraries
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the specified name and optional level.

    Args:
        name: Logger name (usually __name__)
        level: Optional specific level for this logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
