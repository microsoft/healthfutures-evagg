import os
import tempfile
from typing import NamedTuple

import pytest

from lib.evagg.sk import SemanticKernelClient, SemanticKernelConfig


class MockSKResult(NamedTuple):
    error_occurred: bool
    result: str


@pytest.fixture
def fake_config():
    return SemanticKernelConfig(
        deployment="my_deployment", endpoint="https://my_endpoint.com", api_key="my_api_key", skill_dir=None
    )


def test_run_completion_function(mocker, fake_config):
    mocker.patch("lib.evagg.sk._client.SemanticKernelClient._invoke_function", return_value="my_expected_result")

    client = SemanticKernelClient(fake_config)
    context_variables = {"input": "my_input", "gene": "gene", "variant": "variant"}

    # Test all functions in the content skill/plugin
    for function_name in ["inheritance", "phenotype", "zygosity"]:
        result = client.run_completion_function("content", function_name, context_variables)
        assert result == "my_expected_result"


def test_cached_function(mocker, fake_config):
    mocker.patch("lib.evagg.sk._client.SemanticKernelClient._invoke_function", return_value="my_expected_result")
    mocker.spy(SemanticKernelClient, "_import_function")

    client = SemanticKernelClient(fake_config)
    context_variables = {"input": "my_input", "gene": "gene", "variant": "variant"}

    # This should load the function from disk
    result = client.run_completion_function("content", "inheritance", context_variables)
    assert result == "my_expected_result"

    # This should use the cached function
    result = client.run_completion_function("content", "inheritance", context_variables)
    assert result == "my_expected_result"

    assert SemanticKernelClient._import_function.call_count == 1


def test_nonexistent_function(fake_config):
    client = SemanticKernelClient(fake_config)
    context_variables = {"input": "my_input", "gene": "gene", "variant": "variant"}

    with pytest.raises(ValueError):
        client.run_completion_function("content", "fake_function", context_variables)


def test_failed_call(fake_config):
    client = SemanticKernelClient(fake_config)
    context_variables = {"input": "my_input", "gene": "gene", "variant": "variant"}

    # This will fail because we provided a bunk config
    with pytest.raises(ValueError):
        client.run_completion_function("content", "moi", context_variables)


def test_custom_skill_dir(mocker, fake_config):
    mocker.patch("lib.evagg.sk._client.SemanticKernelClient._invoke_function", return_value="my_expected_result")

    # Create a fake skill directory with a fake function
    skill = "fake_skill"
    func = "fake_func"
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = os.path.join(tmpdir, skill)
        os.mkdir(skill_dir)
        func_dir = os.path.join(skill_dir, func)
        os.mkdir(func_dir)
        with open(os.path.join(func_dir, "config.json"), "w") as f:
            f.write(
                """
{
    "type": "completion",
    "completion": {
        "max_tokens": 1000,
        "temperature": 0.2,
        "top_p": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
}
"""
            )
        with open(os.path.join(func_dir, "skprompt.txt"), "w") as f:
            f.write("Fake prompt: {{$input}}")
        fake_config.skill_dir = tmpdir

        client = SemanticKernelClient(fake_config)
        context_variables = {"input": "my_input"}
        result = client.run_completion_function(skill, func, context_variables)
        assert result == "my_expected_result"
