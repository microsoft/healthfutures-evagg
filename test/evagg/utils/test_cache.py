from unittest.mock import patch

from lib.evagg.utils.cache import ObjectCache, ObjectFileCache
from lib.evagg.utils.run import RunRecord


@patch("lib.evagg.utils.cache.get_previous_run")
@patch("lib.evagg.utils.cache.get_run_path")
@patch("os.makedirs")
@patch("subprocess.run")
def test_previous_run(mock_subprocess_run, mock_dirs, mock_run_path, mock_previous_run):
    mock_run_path.return_value = "root"
    mock_previous_run.return_value = RunRecord(
        name="cache_test_run",
        timestamp="20240101_000000",
        args=[],
        git={},
        path="other_root",
    )
    ObjectFileCache("me", use_previous_cache=True)
    mock_dirs.assert_called_with("root/results_cache/me", exist_ok=True)
    mock_subprocess_run.assert_called_with(["cp", "-r", "other_root/results_cache/me", "root/results_cache"], check=True)

def test_object_cache():
    cache = ObjectCache(lambda x: x > 0)
    cache._cache[0] = 1
    assert cache.get(0) == 1
    assert cache.get(1) is None
    cache._cache[0] = -1
    assert cache.get(0) is None
    assert cache.get(1) is None
