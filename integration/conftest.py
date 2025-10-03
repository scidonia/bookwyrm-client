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
def api_key():
    """Provide a test API key."""
    return os.environ["BOOKWYRM_API_KEY"]


@pytest.fixture
def api_url():
    """Provide a URL for the endpoints."""
    return os.environ["BOOKWYRM_API_URL"]


@pytest.fixture
def client(api_key, api_url):
    """Provide a BookWyrm client for testing."""
    return BookWyrmClient(base_url=api_url, api_key=api_key)


@pytest.fixture
async def async_client(api_key, api_url):
    """Provide an async BookWyrm client for testing."""
    client = AsyncBookWyrmClient(base_url=api_url, api_key=api_key)
    yield client
    await client.close()
