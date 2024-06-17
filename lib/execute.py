import logging
import os
import traceback
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Sequence

from lib.evagg import IEvAggApp

from .di import DiContainer

logger = logging.getLogger(__name__)


def _parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    # Accept a path to a config file (required).
    parser.add_argument("config", help="Path to config file for an IEvAggApp object.")

    # Accept an optional count of how many times to retry the app if it fails. Default is 0.
    parser.add_argument(
        "-r",
        "--retries",
        type=int,
        default=0,
        help="Number of times to retry the app if it throws an exception. -1 for infinite retries. Default is 0.",
    )

    # Accept an optional list of key/value pairs to override or add to config dictionary.
    parser.add_argument(
        "-o",
        "--override",
        nargs="*",
        help="Override config values. Specify as key.subkey.subkey:value. Can be specified multiple times.",
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


def run_evagg_app() -> None:
    args = _parse_args()

    config_yaml = args.config
    if not os.path.exists(config_yaml):
        if not config_yaml.endswith(".yaml"):
            config_yaml += ".yaml"
        if not os.path.exists(config_yaml):
            config_yaml = os.path.join("lib", "config", config_yaml)
            if not os.path.exists(config_yaml):
                raise FileNotFoundError(f"EvAgg app config file not found: {args.config}")

    # Make a spec dictionary out of the factory yaml and the override args and instantiate it.
    spec = {"di_factory": config_yaml, **_parse_override_args(args.override)}
    app: IEvAggApp = DiContainer().create_instance(spec, {})

    while True:

        try:
            app.execute()
            break
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
                break
        except Exception as e:
            logger.error(f"Error executing app: {e}")
            # log the stack trace using logger.error
            logger.error(traceback.format_exc())
            # Check if we should retry the app run.
            if args.retries == 0:
                exit(1)
            if args.retries > 0:
                args.retries -= 1


if __name__ == "__main__":
    run_evagg_app()
