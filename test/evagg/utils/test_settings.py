from unittest.mock import patch

import pytest

from lib.evagg.utils import get_azure_credential, get_dotenv_settings, get_env_settings


@patch.dict("os.environ", {"PREFIX1_SETTING1": "testval1", "PREFIX1_SETTING2": "testval2"}, clear=True)
def test_env_settings():
    settings = get_env_settings("PREFIX1_")
    assert settings == {"setting1": "testval1", "setting2": "testval2"}


def test_dotenv_settings(tmpdir):
    dotenv = tmpdir.mkdir("sub").join(".env")
    dotenv.write("PREFIX1_SETTING1=testval1\nPREFIX1_SETTING2=testval2\n")
    settings = get_dotenv_settings(str(dotenv), "PREFIX1_")
    assert settings == {"setting1": "testval1", "setting2": "testval2"}


@patch("lib.evagg.utils.settings.DefaultAzureCredential")
def test_azure_credential_default(mock_credential):
    get_azure_credential()
    mock_credential.assert_called_once_with()


@patch("lib.evagg.utils.settings.AzureCliCredential")
def test_azure_credential_cli(mock_credential):
    get_azure_credential("AzureCli")
    mock_credential.assert_called_once_with()


def test_azure_credential_invalid():
    with pytest.raises(ValueError):
        get_azure_credential("Invalid")
