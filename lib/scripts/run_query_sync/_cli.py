from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml

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

    override_dict: Dict[Any, Any] = {}
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


def _nested_update(d: Dict, u: Dict) -> Dict:
    """Recursively update a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main() -> None:
    args = _parse_args()
    config: Dict[str, Any]

    # Read in the config dictionary.
    with open(Path(args.config), "r") as f:
        config = yaml.safe_load(f)

    # Merge in any overrides.
    config = _nested_update(config, _parse_override_args(args.override))

    # Instantiate and run the app.
    app: IEvAggApp = DiContainer().build(config)
    app.execute()


if __name__ == "__main__":
    main()
