"""Integration tests for synchronous PDF endpoints."""

import pytest


@pytest.mark.pdf
def test_extract_pdf_mock():
    """Test extract_pdf with mocked API."""
    # TODO: Implement mocked PDF extraction test
    pass


@pytest.mark.pdf
def test_stream_extract_pdf_mock():
    """Test stream_extract_pdf with mocked API."""
    # TODO: Implement mocked streaming PDF extraction test
    pass


@pytest.mark.pdf
@pytest.mark.liveonly
def test_extract_pdf_live_api():
    """Test extract_pdf against live API."""
    # TODO: Implement live API PDF extraction test
    pass
