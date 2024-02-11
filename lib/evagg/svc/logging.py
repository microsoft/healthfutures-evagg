import logging
import logging.config
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

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
            "formatter": "prompt_formatter",
            "filters": ["prompt_filter"],
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


def init_prompt_filter(prompts_enabled: bool = False) -> LogFilter:
    # Only handle prompt logs.
    def filter(record: logging.LogRecord) -> bool:
        return prompts_enabled and record.levelno == PROMPT

    return filter


def init_console_filter(exclude_modules: Set[str], include_modules: Set[str]) -> LogFilter:
    def filter(record: logging.LogRecord) -> bool:
        def match_module(record: logging.LogRecord, module: str) -> bool:
            return record.name.startswith(module[:-1]) if module.endswith("*") else record.name == module

        # Prompt logs are handled by the prompt handler.
        if record.levelno == PROMPT:
            return False
        # Don't filter out warnings and above.
        if record.levelno >= logging.WARNING:
            return True
        if any(match_module(record, module) for module in include_modules):
            return True
        return all(not match_module(record, module) for module in exclude_modules)

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
        return self.format_console(record.levelno, record)

    @classmethod
    def format_console(cls, formatno: int, record: logging.LogRecord) -> str:
        log_fmt = cls.FORMATS.get(formatno)
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
    _log_dir: Optional[str] = None

    def __init__(self, to_console: bool, to_file: bool, log_root: str, msg_level: Optional[int] = None) -> None:
        self._to_file = to_file
        self._to_console = to_console
        self._msg_level = msg_level
        super().__init__()

        if to_file:
            self._log_dir = f"{log_root}/prompt_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(self._log_dir, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        if 
        with open(f"{self.log_dir}/{record.name}.log", "a") as f:
            f.write(self.format(record) + "\n")


class LogProvider:
    def __init__(
        self,
        level: str = "WARNING",
        exclude_modules: Optional[List[str]] = None,
        include_modules: Optional[List[str]] = None,
        exclude_defaults: bool = True,
        prompts_to_file: bool = False,
        prompts_to_console: bool = False,
        prompt_output_root: str = ".out",
        prompt_msg_level: str = "INFO",
    ) -> None:
        # Add custom log level for prompts.
        logging.addLevelName(PROMPT, "PROMPT")

        # Set up the base log level (defaults to WARNING).
        level_number = self._level_to_number(level)
        LOGGING_CONFIG["root"]["level"] = level_number

        # Set up the prompt handler's logging arguments.
        LOGGING_CONFIG["handlers"]["prompt_files"]["to_console"] = prompts_to_console
        LOGGING_CONFIG["handlers"]["prompt_files"]["to_file"] = prompts_to_file
        LOGGING_CONFIG["handlers"]["prompt_files"]["log_root"] = prompt_output_root
        # Allow prompt records through to the handler if either console or file logging is enabled.
        LOGGING_CONFIG["filters"]["prompt_filter"]["prompts_enabled"] = prompts_to_console or prompts_to_file

        prompt_msg_level_number = self._level_to_number(prompt_msg_level)
        if prompt_msg_level_number <= level_number:


        # Set up the console filter's module logging arguments.
        exclusions = set(DEFAULT_EXCLUSIONS if exclude_defaults else [])
        exclusions.update(exclude_modules or [])
        inclusions = set(include_modules or [])
        LOGGING_CONFIG["filters"]["console_filter"].update(
            {
                "exclude_modules": exclusions,
                "include_modules": inclusions,
            }
        )

        logging.config.dictConfig(LOGGING_CONFIG)

        logger = logging.getLogger(__name__)
        level_name = logging.getLevelName(logger.getEffectiveLevel())
        logger.info(f"Level:{level_name}")

    @classmethod
    def _level_to_number(cls, level: str) -> int:
        level_number = getattr(logging, level, None)
        if not isinstance(level_number, int):
            raise ValueError(f"Invalid log level: {level}")
        return level_number


_log_provider: Optional[LogProvider] = None


def init_logger(**kwargs: Any) -> LogProvider:
    global _log_provider
    if not _log_provider:
        _log_provider = LogProvider(**kwargs)
    else:
        logger = logging.getLogger(__name__)
        logger.warning("Logging service already initialized - ignoring new initialization.")
    return _log_provider
