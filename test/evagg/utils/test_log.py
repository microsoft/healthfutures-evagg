import logging
from unittest.mock import patch

from lib.evagg.utils.logging import PROMPT, init_logger
from lib.evagg.utils.run import _current_run
from lib.execute import run_evagg_app


def test_log(capfd, tmpdir, test_file_contents):
    # sample_config triggers log setup and exercises SimpleFileLibrary.
    with patch("sys.argv", ["test", "sample_config", "--override", "log.level:DEBUG", f"log.root:{tmpdir}"]):
        run_evagg_app()

    logger = logging.getLogger(__name__)
    logger.warning("warning test log message")
    logger.info("info test log message")
    logger.debug("debug test log message")
    # Should trigger warning in log.
    init_logger()

    prompt_log = {
        "prompt_tag": "test",
        "prompt_metadata": {"name": "test", "type": "test"},
        "prompt_settings": {"model": "test", "max_tokens": 10},
        "prompt_text": "\n".join(["line1", "line2"]),
        "prompt_response": "test response",
    }
    logger.log(PROMPT, "Test chat complete in 0.0 seconds.", extra=prompt_log)

    # Check logging output to stdout
    out, _ = capfd.readouterr()
    assert "WARNING:test_log:warning test log message" in out
    assert "INFO:test_log:info test log message" in out
    assert "DEBUG:test_log:debug test log message" in out
    assert "PROMPT:test_log:Test chat complete in 0.0 seconds." in out
    assert "WARNING:evagg.utils.logging:Logging service already initialized" in out
    # One output dir with 3 files: run.json, test.log, console.log
    assert len(tmpdir.listdir()) == 1
    log_dir = tmpdir.listdir()[0]
    assert len(log_dir.listdir()) == 3
    assert log_dir.join("test.log").read() in out
    assert test_file_contents("test_prompt_log.txt") == log_dir.join("test.log").read()
    assert str(log_dir.join("console.log").read()).startswith("ARGS:test sample_config --override log.level:DEBUG")

    # Undo run completion to avoid polluting other tests
    _current_run.elapsed_secs = None
    _current_run.path = None
