import json
import os
import xml.etree.ElementTree as Et

import pytest


@pytest.fixture
def test_resources_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))


@pytest.fixture
def json_load(test_resources_path):
    def _loader(file_name):
        with open(os.path.join(test_resources_path, file_name), "r") as json_file:
            return json.load(json_file)

    return _loader


@pytest.fixture
def xml_load(test_resources_path):
    def _loader(file_name):
        with open(os.path.join(test_resources_path, file_name), "r") as xml_file:
            return Et.fromstring(xml_file.read())

    return _loader
