import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

LogFilter = Callable[[logging.LogRecord], bool]
PROMPT = logging.CRITICAL + 5

LOGGING_CONFIG: Dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"module_filter": {"()": "lib.evagg.svc.logging.init_module_filter"}},
    "handlers": {
        "console_handler": {
            "class": "lib.evagg.svc.logging.ConsoleHandler",
            "filters": ["module_filter"],
        },
        "file_handler": {
            "()": "lib.evagg.svc.logging.FileHandler",
            "filters": ["module_filter"],
        },
    },
    "root": {
        "handlers": ["console_handler", "file_handler"],
    },
}


DEFAULT_EXCLUSIONS = [
    "azure.core.pipeline.policies.*",
    "openai._base_client",
    "httpcore.*",
    "httpx",
]


def init_module_filter(exclude_modules: Set[str], include_modules: Set[str]) -> LogFilter:
    def filter(record: logging.LogRecord) -> bool:
        def match_module(record: logging.LogRecord, module: str) -> bool:
            return record.name.startswith(module[:-1]) if module.endswith("*") else record.name == module

        # Don't filter out warnings and above.
        if record.levelno >= logging.WARNING:
            return True
        if any(match_module(record, module) for module in include_modules):
            return True
        return all(not match_module(record, module) for module in exclude_modules)

    return filter


class LoggingFormatter(logging.Formatter):
    dim_grey = "\x1b[38;5;8m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[38;5;11m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"
    BASE_FMT = "%(levelname)s:%(name)s:%(message)s"
    DEFAULT_FMT = "%(asctime)s.%(msecs)03d\t" + BASE_FMT

    FORMATS = {
        logging.DEBUG: dim_grey + BASE_FMT + reset,
        logging.INFO: grey + BASE_FMT + reset,
        logging.WARNING: yellow + BASE_FMT + reset,
        logging.ERROR: red + BASE_FMT + reset,
        logging.CRITICAL: red + BASE_FMT + reset,
    }

    @classmethod
    def format_default(cls, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(cls.DEFAULT_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

    @classmethod
    def format_color(cls, formatno: int, record: logging.LogRecord) -> str:
        log_fmt = cls.FORMATS.get(formatno)
        formatter = logging.Formatter(log_fmt)
        # Strip "lib." prefix off of the record name.
        record.name = record.name.replace("lib.", "")
        return formatter.format(record)

    @classmethod
    def format_prompt(cls, record: logging.LogRecord) -> str:
        prompt_tag = record.__dict__.get("prompt_tag", "(no tag)")
        prompt_model = record.__dict__.get("prompt_model", "(no model)")
        settings = record.__dict__.get("prompt_settings", {})
        prompt = record.__dict__.get("prompt_text", "")
        response = record.__dict__.get("prompt_response", "")
        header = f"{prompt_tag} {prompt_model} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return (
            f"#{'/' * 80}\n#{(' ' + header + ' ').ljust(80, '/')}\n"
            + f"#{' SETTINGS '.ljust(80, '#')}\n"
            + f"{'\n'.join(f'{k}: {v}' for k, v in settings.items())}\n"
            + f"#{' PROMPT '.ljust(80, '#')}\n"
            + f"{prompt}\n"
            + f"#{' RESPONSE '.ljust(80, '#')}\n"
            + f"{response}\n"
        )


class FileHandler(logging.Handler):
    def __init__(self, log_root: str, prompts_enabled: bool, console_enabled: bool) -> None:
        super().__init__()
        self._prompts_enabled = prompts_enabled
        self._console_enabled = console_enabled
        # If prompts and console output are both disabled, filter out all records.
        if not prompts_enabled and not console_enabled:
            self.addFilter(lambda _: False)
            return

        self._log_dir = f"{log_root}/logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self._log_dir, exist_ok=True)

        # If console output is enabled, write to file the
        # command-line arguments used to start the program.
        if console_enabled:
            with open(f"{self._log_dir}/console.log", "w") as f:
                f.write("ARGS:" + " ".join(sys.argv) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        if self._prompts_enabled and record.levelno == PROMPT:
            file_name = record.__dict__.get("prompt_tag", "prompt")
            with open(f"{self._log_dir}/{file_name}.log", "a") as f:
                f.write(LoggingFormatter.format_prompt(record) + "\n")
        if self._console_enabled and record.levelno != PROMPT:
            with open(f"{self._log_dir}/console.log", "a") as f:
                f.write(LoggingFormatter.format_default(record) + "\n")


class ConsoleHandler(logging.StreamHandler):
    def __init__(self, prompts_enabled: bool, prompt_msgs_enabled: bool) -> None:
        super().__init__(stream=sys.stdout)
        self._prompts_enabled = prompts_enabled
        self._prompt_msgs_enabled = prompt_msgs_enabled
        # If prompts and prompt messages are both disabled, filter out prompt records.
        if not prompts_enabled and not prompt_msgs_enabled:
            self.addFilter(lambda record: record.levelno != PROMPT)

    def format(self, record: logging.LogRecord) -> str:
        # Handle non-prompt records.
        if record.levelno != PROMPT:
            return LoggingFormatter.format_color(record.levelno, record)

        outputs = []
        if self._prompts_enabled:
            outputs.append(LoggingFormatter.format_prompt(record))
        if self._prompt_msgs_enabled:
            outputs.append(LoggingFormatter.format_color(logging.INFO, record))
        return "\n".join(outputs)


class LogProvider:
    def __init__(
        self,
        level: Optional[str] = None,
        exclude_modules: Optional[List[str]] = None,
        include_modules: Optional[List[str]] = None,
        exclude_defaults: Optional[bool] = True,
        prompts_to_file: Optional[bool] = False,
        prompts_to_console: Optional[bool] = False,
        console_to_file: Optional[bool] = False,
        output_path: Optional[str] = ".out",
    ) -> None:
        # Set up the base log level (defaults to WARNING).
        level_number = self._level_to_number(level or "WARNING")
        LOGGING_CONFIG["root"]["level"] = level_number
        # Add custom log level for prompts.
        logging.addLevelName(PROMPT, "PROMPT")

        # Set up the file handler logging arguments.
        LOGGING_CONFIG["handlers"]["file_handler"]["log_root"] = output_path
        LOGGING_CONFIG["handlers"]["file_handler"]["prompts_enabled"] = prompts_to_file or False
        LOGGING_CONFIG["handlers"]["file_handler"]["console_enabled"] = console_to_file or False

        # Set up the console handler logging arguments.
        LOGGING_CONFIG["handlers"]["console_handler"]["prompts_enabled"] = prompts_to_console or False
        LOGGING_CONFIG["handlers"]["console_handler"]["prompt_msgs_enabled"] = level_number <= logging.INFO

        # Set up the module filter.
        exclusions = set(DEFAULT_EXCLUSIONS if exclude_defaults else [])
        exclusions.update(exclude_modules or [])
        inclusions = set(include_modules or [])
        LOGGING_CONFIG["filters"]["module_filter"]["exclude_modules"] = exclusions
        LOGGING_CONFIG["filters"]["module_filter"]["include_modules"] = inclusions

        # Set up the global logging configuration.
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
