import json

from ._interfaces import IWriteOutput

class ConsoleOutputWriter(IWriteOutput):
    def __init__(self) -> None:
        pass

    def write(self, fields: dict[str, dict[str, str]]) -> None:
        print(json.dumps(fields, indent=4))

class FileOutputWriter(IWriteOutput):
    def __init__(self, path: str) -> None:
        self._path = path

    def write(self, fields: dict[str, dict[str, str]]) -> None:
        print(f"Writing output to: {self._path}")
        
        # TODO create parent if doesn't exist.

        with open(self._path, "w") as f:
            json.dump(fields, f, indent=4)
