import os
from unittest.mock import patch

from lib.evagg import TableOutputWriter


def test_stdout_output():
    table_output_writer = TableOutputWriter(None)
    output_path = table_output_writer.write([])
    assert output_path is None


@patch("lib.evagg.io.get_run_path")
def test_csv_output_no_results(mock_get_run_path, tmp_path):
    mock_get_run_path.return_value = str(tmp_path / "run")
    table_output_writer = TableOutputWriter("test")
    output_path = table_output_writer.write([])
    assert output_path is None


@patch("lib.evagg.io.get_run_path")
def test_csv_output(mock_get_run_path, tmp_path):
    mock_get_run_path.return_value = str(tmp_path / "run")
    output = [
        {"column1": "value1", "column2": "value2"},
        {"column1": "value3", "column2": "value4"},
    ]

    table_output_writer = TableOutputWriter("test")
    output_path = table_output_writer.write(output)
    assert output_path == str(tmp_path / "run" / "test.tsv")

    with open(output_path or "", "r") as f:
        lines = f.readlines()
        assert lines[0].strip().startswith("# Generated ")
        assert lines[1].strip() == "column1\tcolumn2"
        assert lines[2].strip() == "value1\tvalue2"
        assert lines[3].strip() == "value3\tvalue4"


@patch("lib.evagg.io.get_run_path")
def test_csv_output_warning(mock_get_run_path, tmp_path):
    mock_get_run_path.return_value = str(tmp_path / "run")
    os.makedirs(tmp_path / "run")

    # Create blank file to be overwritten - should generate logger warning
    with open(tmp_path / "run" / "test.tsv", "w") as f:
        f.write("")

    with patch("lib.evagg.io.logger") as mock_logger:
        TableOutputWriter("test")
    mock_logger.warning.assert_called_once()
