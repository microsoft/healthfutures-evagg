from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence

from lib.di import DiContainer
from lib.evagg import IEvAggApp


def _parse_args(args: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("-c", "--config", help="YAML config file specifying execution configuration.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    di = DiContainer(config=Path(args.config))
    app: IEvAggApp = di.build()
    app.execute()


if __name__ == "__main__":
    main()
