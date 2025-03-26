import logging
import logging.config
import os
import sys
from typing import Callable, Dict, List, Optional, Set

from .run import get_run_path, set_output_root

LogFilter = Callable[[logging.LogRecord], bool]
PROMPT = logging.CRITICAL + 5
_log_initialized = False


LOGGING_CONFIG: Dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"module_filter": {"()": "lib.evagg.utils.logging.init_module_filter"}},
    "handlers": {
        "console_handler": {
            "class": "lib.evagg.utils.logging.ConsoleHandler",
            "filters": ["module_filter"],
        },
        "file_handler": {
            "()": "lib.evagg.utils.logging.FileHandler",
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


def _format_prompt(record: logging.LogRecord) -> str:
    prompt_tag = record.__dict__.get("prompt_tag", "(no tag)")
    metadata = record.__dict__.get("prompt_metadata", {})
    settings = record.__dict__.get("prompt_settings", {})
    prompt = record.__dict__.get("prompt_text", "")
    thought = record.__dict__.get("prompt_thought", "")
    response = record.__dict__.get("prompt_response", "")
    return (
        f"#{'#' * 80}\n"
        + f"#{' METADATA '.ljust(80, '#')}\n"
        + f"{'prompt_tag':<16}: {prompt_tag}\n"
        + f"{'\n'.join(f'{k:<16}: {v}' for k, v in metadata.items())}\n"
        + f"#{' SETTINGS '.ljust(80, '#')}\n"
        + f"{'\n'.join(f'{k}: {v}' for k, v in settings.items())}\n"
        + f"#{' PROMPT '.ljust(80, '#')}\n"
        + f"{prompt}\n"
        + f"#{' THOUGHT '.ljust(80, '#')}\n"
        + f"{thought}\n"
        + f"#{' RESPONSE '.ljust(80, '#')}\n"
        + f"{response}\n"
    )


class FileHandler(logging.Handler):
    FORMAT = "%(asctime)s.%(msecs)03d\t%(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, logs_enabled: bool, prompt_msgs_enabled: bool) -> None:
        super().__init__()
        self._logs_enabled = logs_enabled
        self._prompt_msgs_enabled = prompt_msgs_enabled
        self._formatter = logging.Formatter(fmt=self.FORMAT, datefmt=self.DATE_FORMAT)

        # If file logs are disabled, filter out all records.
        if not logs_enabled:
            self.addFilter(lambda _: False)
            return

        # If console output is enabled, write to file the
        # command-line arguments used to start the program.
        self._console_log = os.path.join(get_run_path(), "console.log")
        with open(self._console_log, "a") as f:
            f.write("ARGS:" + " ".join(sys.argv) + "\n")

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == PROMPT:
            file_name = record.__dict__.get("prompt_tag", "prompt")
            with open(os.path.join(get_run_path(), f"{file_name}.log"), "a") as f:
                f.write(_format_prompt(record) + "\n")
        if record.levelno != PROMPT or self._prompt_msgs_enabled:
            with open(self._console_log, "a") as f:
                f.write(self._formatter.format(record) + "\n")


class ConsoleHandler(logging.StreamHandler):
    BASE_FMT = "%(levelname)s:%(name)s:%(message)s"
    dim_grey = "\x1b[38;5;8m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[1;33m"
    red = "\x1b[31;20m"
    reset = "\x1b[0m"

    COLOR_FORMATTERS = {
        logging.DEBUG: logging.Formatter(dim_grey + BASE_FMT + reset),
        logging.INFO: logging.Formatter(grey + BASE_FMT + reset),
        logging.WARNING: logging.Formatter(yellow + BASE_FMT + reset),
        logging.ERROR: logging.Formatter(red + BASE_FMT + reset),
        logging.CRITICAL: logging.Formatter(red + BASE_FMT + reset),
        PROMPT: logging.Formatter(grey + BASE_FMT + reset),  # equivalent to INFO
    }

    def __init__(self, prompts_enabled: bool, prompt_msgs_enabled: bool) -> None:
        super().__init__(stream=sys.stdout)
        self._prompts_enabled = prompts_enabled
        self._prompt_msgs_enabled = prompt_msgs_enabled
        # If prompts and prompt messages are both disabled, filter out prompt records.
        if not prompts_enabled and not prompt_msgs_enabled:
            self.addFilter(lambda record: record.levelno != PROMPT)

    def format(self, record: logging.LogRecord) -> str:
        outputs = []
        if record.levelno == PROMPT and self._prompts_enabled:
            outputs.append(_format_prompt(record))
        if record.levelno != PROMPT or self._prompt_msgs_enabled:
            # Strip "lib." prefix off of the record name.
            record.name = record.name.replace("lib.", "")
            outputs.append(self.COLOR_FORMATTERS[record.levelno].format(record))
        return "\n".join(outputs)


def init_logger(
    level: Optional[str] = None,
    exclude_modules: Optional[List[str]] = None,
    include_modules: Optional[List[str]] = None,
    exclude_defaults: Optional[bool] = True,
    prompts_to_console: Optional[bool] = False,
    to_file: Optional[bool] = False,
    root: Optional[str] = ".out",
) -> None:
    global _log_initialized
    if _log_initialized:
        logger = logging.getLogger(__name__)
        logger.warning("Logging service already initialized - ignoring new initialization.")
        return

    if root:
        set_output_root(root)

    _log_initialized = True
    # Set up the base log level (defaults to WARNING).
    level_number = getattr(logging, level or "WARNING", None)
    if not isinstance(level_number, int):
        raise ValueError(f"Invalid log level: {level}")
    LOGGING_CONFIG["root"]["level"] = level_number
    # Add custom log level for prompts.
    logging.addLevelName(PROMPT, "PROMPT")

    # Set up the file handler logging arguments.
    LOGGING_CONFIG["handlers"]["file_handler"]["logs_enabled"] = to_file or False
    LOGGING_CONFIG["handlers"]["file_handler"]["prompt_msgs_enabled"] = level_number <= logging.INFO

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
