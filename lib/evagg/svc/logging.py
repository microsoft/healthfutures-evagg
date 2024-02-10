import logging
import logging.config
import os
from typing import Any, Callable, Dict, List, Optional

PROMPT = logging.CRITICAL + 5

LogFilter = Callable[[logging.LogRecord], bool]

LOGGING_CONFIG: Dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color_log_formatter": {"()": "lib.evagg.svc.logging.ColorConsoleFormatter"},
        "prompt_formatter": {"()": "lib.evagg.svc.logging.PromptFormatter"},
    },
    "filters": {
        "console_filter": {"()": "lib.evagg.svc.logging.init_console_filter"},
        "prompt_filter": {"()": "lib.evagg.svc.logging.init_prompt_filter"},
    },
    "handlers": {
        "console_color": {
            "class": "logging.StreamHandler",
            "formatter": "color_log_formatter",
            "stream": "ext://sys.stdout",
            "filters": ["console_filter"],
        },
        "prompt_files": {
            "()": "lib.evagg.svc.logging.PromptHandler",
            "filters": ["prompt_filter"],
            "formatter": "prompt_formatter",
        },
    },
    "root": {
        "handlers": ["console_color", "prompt_files"],
    },
}


DEFAULT_EXCLUSIONS = [
    "azure.core.pipeline.policies.*",
    "openai._base_client",
    "httpcore.*",
    "httpx",
]


def init_prompt_filter(is_enabled: bool = False) -> LogFilter:

    # Only handle prompt logs.
    def filter(record: logging.LogRecord) -> bool:
        return is_enabled and record.levelno == PROMPT

    return filter


def init_console_filter(exclude_modules: List[str], include_modules: List[str], exclude_defaults: bool) -> LogFilter:
    exclusions = set(DEFAULT_EXCLUSIONS if exclude_defaults else [])
    exclusions.update(exclude_modules or [])
    inclusions = set(include_modules or [])

    def filter(record: logging.LogRecord) -> bool:
        def match_module(record: logging.LogRecord, module: str) -> bool:
            return record.name.startswith(module[:-1]) if module.endswith("*") else record.name == module

        # Prompt logs are handled separately.
        # if record.levelno == PROMPT:
        #     ddd return False
        # Don't filter out warnings and above.
        if record.levelno >= logging.WARNING:
            return True
        if any(match_module(record, module) for module in inclusions):
            return True
        return all(not match_module(record, module) for module in exclusions)

    return filter


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
        if record.levelno == PROMPT:
            return PromptFormatter.format_prompt(record)
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        # Strip "lib." prefix off of the record name.
        record.name = record.name.replace("lib.", "")
        return formatter.format(record)


class PromptFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return self.format_prompt(record)

    @classmethod
    def format_prompt(cls, record: logging.LogRecord) -> str:
        settings = record.__dict__.get("prompt_settings", {})
        prompt = record.__dict__.get("prompt_text", "")
        result = record.__dict__.get("prompt_result", "")
        return (
            f"#{' settings '.ljust(80, '#')}\n"
            + f"{'\n'.join(f'{k}: {v}' for k, v in settings.items())}\n"
            + f"#{' prompt '.ljust(80, '#')}\n"
            + f"{prompt}\n"
            + f"#{' result '.ljust(80, '#')}\n"
            + f"{result}\n"
        )


class PromptHandler(logging.Handler):
    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.log_dir = log_dir
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        with open(f"{self.log_dir}/{record.name}.log", "a") as f:
            f.write(self.format(record) + "\n")


class LogProvider:
    def __init__(
        self,
        level: str = "WARNING",
        exclude_modules: Optional[List[str]] = None,
        include_modules: Optional[List[str]] = None,
        exclude_defaults: bool = True,
        prompts_enabled: bool = False,
        prompt_output_dir: str = ".out",
        prompt_dir_prefix: str = "prompts",
    ) -> None:
        # Add custom log level for prompts.
        logging.addLevelName(PROMPT, "PROMPT")

        # Set up the base log level (default to WARNING).
        level_number = getattr(logging, level, None)
        if not isinstance(level_number, int):
            raise ValueError(f"Invalid log.level: {level}")
        LOGGING_CONFIG["root"]["level"] = level_number

        # Set up the prompt handler's logging arguments.
        if prompts_enabled:
            LOGGING_CONFIG["filters"]["prompt_filter"]["is_enabled"] = True
            log_dir = f"{prompt_output_dir}/{prompt_dir_prefix}"
            LOGGING_CONFIG["handlers"]["prompt_files"]["log_dir"] = log_dir
            os.makedirs(log_dir, exist_ok=True)

        # Set up the console filter's module logging arguments.
        LOGGING_CONFIG["filters"]["console_filter"].update(
            {
                "exclude_modules": exclude_modules or [],
                "include_modules": include_modules or [],
                "exclude_defaults": exclude_defaults,
            }
        )

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
