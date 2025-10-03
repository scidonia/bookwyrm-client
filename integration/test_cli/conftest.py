"""Configuration for CLI integration tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically add 'cli' mark to all tests in the test_cli directory."""
    for item in items:
        item.add_marker(pytest.mark.cli)
