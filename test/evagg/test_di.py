import logging
from typing import Any
from unittest.mock import patch

import pytest

from lib.di import DiContainer
from lib.evagg import IExtractFields, IGetPapers, IWriteOutput
from lib.evagg.types import Paper
from lib.evagg.utils.run import _current_run
from lib.execute import run_evagg_app

LOGGER = logging.getLogger(__name__)


def test_di_container():
    container = DiContainer()
    assert container

    # Test _try_import
    assert container._try_import("lib.di") == container._modules["lib.di"]
    assert container._try_import("lib.di") == container._modules["lib.di"]

    # Test _nested_update
    d = {"a": 1, "b": {"c": 2}}
    u = {"b": {"d": 3}}
    assert container._nested_update(d, u) == {"a": 1, "b": {"c": 2, "d": 3}}


def test_missing_file_execute():
    with patch("sys.argv", ["test", "config.yaml", "--override", "a.b:1", "c:2"]):
        with pytest.raises(FileNotFoundError):
            run_evagg_app()
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value.di_factory:missing.yaml"]):
        with pytest.raises(FileNotFoundError):
            run_evagg_app()


def test_default_path():
    with patch(
        "sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value.di_factory:queries/EXOC2.yaml"]
    ):
        run_evagg_app()


def test_missing_entrypoint():
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "di_factory:lib.evagg.app.missing"]):
        with pytest.raises(TypeError):
            run_evagg_app()


def test_missing_resource():
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:{{missing}}"]):
        with pytest.raises(ValueError):
            run_evagg_app()


def test_duplicate_resource():
    with pytest.raises(ValueError):
        spec = {"di_factory": "test/resources/di.yaml"}
        DiContainer().create_instance(spec, {"resource": "duplicate"})


def test_run_evagg_app(caplog):
    caplog.set_level(logging.INFO)
    with patch("sys.argv", ["test", "test/resources/di.yaml"]):
        run_evagg_app()
    assert "Test app executed with value: {'a': 'b:2 c:3'}" in caplog.text


def test_run_evagg_app_override(caplog):
    caplog.set_level(logging.INFO)
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:overridden_arg"]):
        run_evagg_app()
    assert "Test app executed with value: overridden_arg" in caplog.text


def test_run_evagg_app_override_true(caplog):
    caplog.set_level(logging.INFO)
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:true"]):
        run_evagg_app()
    assert "Test app executed with value: True" in caplog.text


def test_run_evagg_app_override_false(caplog):
    caplog.set_level(logging.INFO)
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:false"]):
        run_evagg_app()
    assert "Test app executed with value: False" in caplog.text


def test_run_evagg_app_interrupt(capfd):
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:KeyboardInterrupt"]):
        run_evagg_app()
    out, err = capfd.readouterr()
    assert "KeyboardInterrupt in" in out


def test_run_evagg_app_exception(caplog):
    with patch("sys.argv", ["test", "test/resources/di.yaml", "--override", "test_value:Exception", "--retries", "1"]):
        with pytest.raises(SystemExit):
            run_evagg_app()
    assert "Error executing app: " in caplog.text


def test_sample_config():
    with patch("sys.argv", ["test", "sample_config"]):
        run_evagg_app()


@pytest.fixture
def mock_library(mock_client: type) -> IGetPapers:
    return mock_client(IGetPapers)


@pytest.fixture
def mock_extractor(mock_client: type) -> IExtractFields:
    return mock_client(IExtractFields)


@pytest.fixture
def mock_writer(mock_client: type) -> IWriteOutput:
    return mock_client(IWriteOutput)


def test_evagg_paper_query_app(json_load, mock_library: Any, mock_extractor: Any, mock_writer: Any):
    spec = {
        "di_factory": "lib.evagg.app.PaperQueryApp",
        "queries": [{"gene_symbol": "test"}],
        "library": "{{mock_library}}",
        "extractor": "{{mock_extractor}}",
        "writer": "{{mock_writer}}",
    }

    resources = {
        "mock_library": mock_library([Paper(**json_load("rare_disease_paper.json"))]),
        "mock_extractor": mock_extractor([{"evidence": "value"}]),
        "mock_writer": mock_writer(None),
    }
    _current_run.elapsed_secs = None  # Reset elapsed_secs to avoid error on multiple runs in the same test.
    DiContainer().create_instance(spec, resources).execute()

    # Test missing query gene_symbol.
    with pytest.raises(ValueError):
        spec["queries"] = [{}]
        DiContainer().create_instance(spec, resources).execute()
