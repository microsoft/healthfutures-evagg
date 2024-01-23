import logging
import logging.config
from typing import Any, Dict, Optional

LOGGING_CONFIG: Dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color_log_formatter": {
            "()": "lib.evagg.svc.logging.ColorConsoleFormatter",
        },
    },
    "handlers": {
        "console_color": {
            "class": "logging.StreamHandler",
            "formatter": "color_log_formatter",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console_color"],
    },
}


class ColorConsoleFormatter(logging.Formatter):
    dim_grey = "\x1b[38;5;8m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[38;5;11m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    # verbose "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    formatting = "%(levelname)s:%(name)s:%(message)s"

    FORMATS = {
        logging.DEBUG: dim_grey + formatting + reset,
        logging.INFO: grey + formatting + reset,
        logging.WARNING: yellow + formatting + reset,
        logging.ERROR: red + formatting + reset,
        logging.CRITICAL: red + formatting + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        # Strip "lib." prefix off of the record name.
        record.name = record.name.replace("lib.", "")
        return formatter.format(record)


class LogProvider:
    def __init__(self, level: str = "WARNING") -> None:
        # Get the base log level from the config (default to WARNING).
        level_number = getattr(logging, level, None)
        if not isinstance(level_number, int):
            raise ValueError(f"Invalid log.level: {level}")

        LOGGING_CONFIG["root"]["level"] = level_number
        logging.config.dictConfig(LOGGING_CONFIG)

        logger = logging.getLogger(__name__)
        level_name = logging.getLevelName(logger.getEffectiveLevel())
        logger.info(f"Level:{level_name}")


_log_provider: Optional[LogProvider] = None


def init_logger(**kwargs: Any) -> LogProvider:
    global _log_provider
    if not _log_provider:
        _log_provider = LogProvider(**kwargs)
    else:
        logger = logging.getLogger(__name__)
        logger.warning("Logging service already initialized - ignoring new initialization.")
    return _log_provider
