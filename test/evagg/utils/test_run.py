from datetime import datetime
from pathlib import Path

import pytest

from lib.evagg.utils.run import (
    RunRecord,
    _current_run,
    get_previous_run,
    get_run_path,
    set_output_root,
    set_run_complete,
)


def test_run_record():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_record = RunRecord(
        name="test",
        timestamp=ts,
        args=[],
        git={
            "branch": "main",
            "commit": "1234567890",
            "modified_files": ["test.py"],
        },
    )
    assert run_record.dir_name == f"run_test_{ts}"
    assert run_record.start_datetime == datetime.strptime(ts, "%Y%m%d_%H%M%S")
    assert run_record.start == datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")


def test_output_path(tmpdir):
    set_output_root(str(tmpdir))
    path = get_run_path()
    assert Path(path).is_dir()
    assert Path(f"{path}/run.json").is_file()
    with pytest.raises(ValueError):
        set_output_root(str(tmpdir))

    _current_run.elapsed_secs = None
    set_run_complete(output_file="test.txt")
    with pytest.raises(ValueError):
        set_run_complete(output_file=None)
    # Undo run completion to avoid polluting other tests
    _current_run.elapsed_secs = None
    _current_run.path = None


def test_previous_run(test_resources_path):
    assert get_previous_run() is None
    _current_run.path = None
    set_output_root(str(test_resources_path))
    run = get_previous_run("test")
    assert run.name == "evagg_pipeline_truthset"  # type: ignore
