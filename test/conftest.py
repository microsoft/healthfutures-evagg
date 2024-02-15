import json
import os
import xml.etree.ElementTree as Et
from typing import Optional

import pytest


@pytest.fixture
def test_resources_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "resources"))


@pytest.fixture
def test_file_contents(test_resources_path):
    def _loader(file_name):
        with open(os.path.join(test_resources_path, file_name), "r") as file:
            return file.read()

    return _loader


@pytest.fixture
def json_load(test_file_contents):
    def _loader(file_name):
        return json.loads(test_file_contents(file_name))

    return _loader


@pytest.fixture
def xml_parse():
    def _parser(content):
        return Et.fromstring(content)

    return _parser


@pytest.fixture
def xml_load(test_file_contents, xml_parse):
    def _loader(file_name):
        return xml_parse(test_file_contents(file_name))

    return _loader


@pytest.fixture
def arg_loader(json_load, xml_load):
    def _loader(arg):
        if isinstance(arg, str) and arg.endswith(".xml"):
            return xml_load(arg)
        if isinstance(arg, str) and arg.endswith(".json"):
            return json_load(arg)
        return arg

    return _loader


@pytest.fixture
def mock_client(arg_loader):
    def client_creator(*interfaces: type):
        class MockClient:
            def __init__(self, *args) -> None:
                self._responses = iter(arg_loader(response) for response in args)
                self._calls = []

                # Get all the method names from the interfaces.
                methods = [
                    name
                    for interface in interfaces
                    for name, fn in vars(interface).items()
                    if callable(fn) and not name.startswith("_")
                ]

                # For each method name, create a method that returns the next response.
                for method_name in methods:

                    def create_method(name):
                        def method(*args, **kwargs):
                            arg_list = list(map(str, args)) + ["=".join(map(str, item)) for item in kwargs.items()]
                            call_string = f"{name}(\n\t{',\n\t'.join(arg_list)}\n)"
                            print(call_string)

                            self._calls.append((name, *args, *[{k: v} for k, v in kwargs.items()]))
                            return next(self._responses)

                        return method

                    setattr(self, method_name, create_method(method_name))

            def last_call(self, method_name: Optional[str] = None):
                return (
                    self._calls[-1]
                    if method_name is None
                    else next(call for call in reversed(self._calls) if call[0] == method_name)[1:]
                )

            def call_count(self, method_name: Optional[str] = None):
                return len([call for call in self._calls if method_name is None or call[0] == method_name])

        return MockClient

    return client_creator
