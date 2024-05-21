import logging
import traceback
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Sequence

from lib.di import DiContainer
from lib.evagg import IEvAggApp

logger = logging.getLogger(__name__)


def _parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    # Accept a path to a config file (required).
    parser.add_argument("config", help="Path to config file for an IEvAggApp object.")

    # Accept an optional list of key/value pairs to override or add to config dictionary.
    parser.add_argument(
        "-o",
        "--override",
        nargs="*",
        help="Override config values. Specify as key1.key2.keyN:value. Can be specified multiple times.",
    )

    return parser.parse_args()


def _parse_override_args(overrides: Sequence[str] | None) -> Dict[str, Any]:
    """Parse the override arguments into a nested dictionary."""
    if overrides is None:
        return {}

    def _parse_override_value(val: str) -> Any:
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    override_dict: Dict[str, Any] = {}
    for override in overrides:
        key_path, _, value = override.partition(":")
        keys = key_path.split(".")
        current_dict = override_dict
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[keys[-1]] = _parse_override_value(value)

    return override_dict


def run_sync() -> None:
    args = _parse_args()
    # Make a spec dictionary out of the factory yaml and the override args and instantiate it.
    spec = {"di_factory": args.config, **_parse_override_args(args.override)}
    app: IEvAggApp = DiContainer().create_instance(spec, {})

    try:
        app.execute()
    except KeyboardInterrupt as e:
        # Less verbose KeyboardInterrupt handling.
        if tb := e.__traceback__:
            while tb.tb_next:
                if "site-packages" in tb.tb_next.tb_frame.f_code.co_filename:
                    break
                tb = tb.tb_next

            # Print only the stack frame immediately before the site-packages level.
            file_ref = f"{tb.tb_frame.f_code.co_filename}::{tb.tb_frame.f_code.co_name}"
            print(f" KeyboardInterrupt in {file_ref} at line {tb.tb_lineno}")
            # And then the innermost traceback.
            traceback.print_tb(tb, limit=-1)
    except Exception as e:
        logger.error(f"Error executing app: {e}")
        # log the stack trace using logger.error
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    run_sync()
