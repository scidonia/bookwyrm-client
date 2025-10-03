"""Shared fixtures and configuration for integration tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from bookwyrm.client import BookWyrmClient
from bookwyrm.async_client import AsyncBookWyrmClient


@pytest.fixture
def cli_runner():
    """Provide a CLI runner for testing commands."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def api_key():
    """Provide a test API key."""
    return "test-api-key-12345"


@pytest.fixture
def client(api_key):
    """Provide a BookWyrm client for testing."""
    return BookWyrmClient(base_url="https://api.bookwyrm.ai:443", api_key=api_key)


@pytest.fixture
async def async_client(api_key):
    """Provide an async BookWyrm client for testing."""
    client = AsyncBookWyrmClient(
        base_url="https://api.bookwyrm.ai:443", api_key=api_key
    )
    yield client
    await client.close()


@pytest.fixture
def env_vars(api_key):
    """Set up environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "BOOKWYRM_API_KEY": api_key,
            "BOOKWYRM_API_URL": "https://api.bookwyrm.ai:443",
        },
    ):
        yield


# Pytest markers for organizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "live_api: marks tests that require live API access"
    )
    config.addinivalue_line("markers", "cli: marks CLI-specific tests")
    config.addinivalue_line("markers", "library: marks library-specific tests")
    config.addinivalue_line("markers", "integration: marks full integration tests")


@pytest.fixture(autouse=True)
def skip_live_api_without_key(request):
    """Skip live API tests if no API key is provided."""
    if request.node.get_closest_marker("liveonly"):
        if not os.getenv("BOOKWYRM_API_KEY"):
            pytest.skip("Live API tests require BOOKWYRM_API_KEY environment variable")
