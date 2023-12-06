from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Sequence

from lib.di import DiContainer
from lib.evagg import IEvAggApp


def _parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    # Accept a path to a config file (required).
    parser.add_argument("config", help="Path to config file.")

    # Accept an optional list of key/value pairs to override or add to config dictionary.
    parser.add_argument(
        "-o",
        "--override",
        nargs="*",
        help="Override config values. Specify as key1.key2.keyN:value. Can be specified multiple times.",
    )

    return parser.parse_args()


def _parse_override_args(overrides: Sequence[str] | None) -> Dict:
    """Parse the override arguments into a nested dictionary."""
    if overrides is None:
        return {}

    override_dict = {}
    for override in overrides:
        key_path, _, value = override.partition(":")
        keys = key_path.split(".")
        current_dict = override_dict
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[keys[-1]] = value

    return override_dict


def main() -> None:
    args = _parse_args()

    config_override = _parse_override_args(args.override)
    di = DiContainer(config=Path(args.config), overrides=config_override)
    app: IEvAggApp = di.build()
    app.execute()


if __name__ == "__main__":
    main()
