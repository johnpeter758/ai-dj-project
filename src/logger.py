"""
Centralized logging configuration for AI DJ Project.

Provides:
- File and console logging
- Log rotation
- Module-level loggers
- Structured logging utilities
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Default log file
DEFAULT_LOG_FILE = LOG_DIR / "ai_dj.log"

# Log levels
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default configuration
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5


def setup_logger(
    name: str = "ai_dj",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    rotation: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: logs/ai_dj.log)
        console: Enable console output
        rotation: Enable log rotation
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
    
    # File handler
    if rotation:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file or DEFAULT_LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_file or DEFAULT_LOG_FILE, encoding="utf-8")
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a module-level logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Add handlers if none exist (lazy initialization)
    if not logger.handlers:
        formatter = logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATE_FORMAT)
        
        # File handler
        file_handler = logging.handlers.RotatingFileHandler(
            DEFAULT_LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def set_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Change log level for a logger or all loggers.
    
    Args:
        level: New log level
        logger_name: Specific logger name, or None for root
    """
    level_val = LOG_LEVELS.get(level.upper(), logging.INFO)
    if logger_name:
        logging.getLogger(logger_name).setLevel(level_val)
    else:
        logging.root.setLevel(level_val)


def log_execution(logger: logging.Logger):
    """Decorator to log function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.exception(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


# Default logger for quick imports
default_logger = setup_logger()

__all__ = [
    "setup_logger",
    "get_logger",
    "set_level",
    "log_execution",
    "default_logger",
    "LOG_DIR",
    "LOG_LEVELS",
]
