from unittest.mock import patch

import pytest

from lib.evagg.utils.git import GitError, ModifiedFile, RepoStatus


@pytest.mark.parametrize(
    "status, type, name",
    [
        ("1 .M N... 100644 100644 100644 hash1 hash2 lib/evagg/changed.py", "changed", "lib/evagg/changed.py"),
        ("2 .M N... 100644 100644 100644 hash1 hash2 R100 lib/evagg/to.py\tlib/evagg/from.py", "renamed", "lib/evagg/to.py"),
        ("u .M N... 100644 100644 100644 hash1 hash2 hash3 hash4 lib/evagg/u.py", "unmerged", "lib/evagg/u.py"),
        ("? lib/evagg/new.py", "untracked", "lib/evagg/new.py"),
        ("! lib/evagg/i.py", "ignored", "lib/evagg/i.py"),
    ]
)
def test_parse_modified_file(status, type, name):
    assert ModifiedFile(status).type == type
    assert ModifiedFile(status).name == name
    assert ModifiedFile(status).type == type
    assert ModifiedFile(status).name == name


def test_parse_failed():
    with pytest.raises(ValueError):
        ModifiedFile("invalid status line")


@patch("os.getcwd", return_value="/")
def test_repo_status_no_repo(mock_getcwd):
    with pytest.raises(GitError):
        RepoStatus()
    assert mock_getcwd.called


@patch("subprocess.check_output", side_effect=Exception)
def test_repo_status_error(mock_check_output):
    with pytest.raises(GitError):
        RepoStatus()
    assert mock_check_output.called
