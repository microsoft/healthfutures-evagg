import json
import os
from typing import Dict, Sequence

from lib.evagg import ConsoleOutputWriter, FileOutputWriter


def test_console_output_writer(capsys):
    # Arrange
    writer = ConsoleOutputWriter()
    fields: Dict[str, Sequence[Dict[str, str]]] = {
        "field1": [{"key1": "value1"}, {"key2": "value2"}],
        "field2": [{"key3": "value3"}, {"key4": "value4"}],
    }
    expected_output = json.dumps(fields, indent=4) + "\n"

    # Act
    writer.write(fields)

    # Assert
    captured = capsys.readouterr()
    assert captured.out == expected_output


def test_file_output_writer(tmp_path):
    # Arrange
    path = os.path.join(tmp_path, "output.json")
    writer = FileOutputWriter(path)
    fields: Dict[str, Sequence[Dict[str, str]]] = {
        "field1": [{"key1": "value1"}, {"key2": "value2"}],
        "field2": [{"key3": "value3"}, {"key4": "value4"}],
    }

    # Act
    writer.write(fields)

    # Assert
    with open(path, "r") as f:
        assert json.load(f) == fields
