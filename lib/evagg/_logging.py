import logging
import logging.config
from typing import Dict, Optional

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color_log_formatter": {
            "()": "lib.evagg._logging.ColorConsoleFormatter",
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

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def configure_logging(log_config: Optional[Dict[str, str]]) -> None:
    log_config = log_config or {}

    # Get the base log level from the config (default to WARNING).
    level = getattr(logging, log_config.get("log_level", "WARNING"), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log.log_level: {log_config['log_level']}")

    LOGGING_CONFIG["root"]["level"] = level
    logging.config.dictConfig(LOGGING_CONFIG)

    logger = logging.getLogger(__name__)
    level_name = logging.getLevelName(logger.getEffectiveLevel())
    logger.info(f"Level:{level_name}")
