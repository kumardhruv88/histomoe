"""
Structured logger with Rich formatting for HistoMoE.

Usage
-----
    from histomoe.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def get_logger(
    name: str,
    level: int = logging.INFO,
    rich_format: bool = True,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create and return a named logger with optional Rich formatting.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    level : int
        Logging level (e.g., ``logging.DEBUG``, ``logging.INFO``).
    rich_format : bool
        If True and Rich is installed, use Rich's coloured handler.
    log_file : str, optional
        If provided, also write logs to this file path.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    fmt = "%(message)s"

    if rich_format and _RICH_AVAILABLE:
        console = Console(stderr=True)
        handler: logging.Handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(logging.Formatter(fmt, datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    logger.addHandler(handler)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    logger.propagate = False
    return logger
