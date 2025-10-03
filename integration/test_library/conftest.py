"""Configuration for library integration tests."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Automatically add marks to library tests based on their location."""
    for item in items:
        # Add library mark to all tests in test_library directory
        item.add_marker(pytest.mark.library)

        # Add sync/async marks based on subdirectory
        test_path = str(item.fspath)
        if "/sync/" in test_path:
            item.add_marker(pytest.mark.sync)
        elif "/async/" in test_path:
            item.add_marker(getattr(pytest.mark, "async"))
