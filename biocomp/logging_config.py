import logging
import matplotlib
from rich.logging import RichHandler
from typing import Optional, Dict
from pathlib import Path

# Disable matplotlib's font manager debug logs before it's even imported
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Prevent matplotlib from setting up handlers on its own
matplotlib.set_loglevel("WARNING")

# Default format includes timestamp, logger name, and message
DEFAULT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure default levels for various loggers
DEFAULT_LOGGER_LEVELS: Dict[str, int] = {
    # External libraries
    "matplotlib": logging.WARNING,
    "matplotlib.font_manager": logging.ERROR,  # Belt and suspenders approach
    "PIL": logging.WARNING,
    "jax": logging.WARNING,
    "ray": logging.WARNING,
    "ray._private.worker": logging.WARNING,
    "fontTools": logging.WARNING,
    "h5py": logging.WARNING,
    "numba": logging.WARNING,
    "parso": logging.WARNING,
    # Project-specific default levels
    "biocomp": logging.INFO,
    "biocomp.plotting": logging.INFO,
}

# Set critical loggers immediately
for logger_name in ["matplotlib.font_manager", "ray._private.worker"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def setup_logging(
    default_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    rich_logging: bool = True,
    logger_levels: Optional[Dict[str, int]] = None,
) -> None:
    """Configure logging for the biocomp project."""
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

    # Apply logger-specific levels
    levels_to_apply = DEFAULT_LOGGER_LEVELS.copy()
    if logger_levels:
        levels_to_apply.update(logger_levels)

    for logger_name, level in levels_to_apply.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with the specified name and optional level."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
