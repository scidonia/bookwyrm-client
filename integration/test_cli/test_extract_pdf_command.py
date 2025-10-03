"""Integration tests for the extract-pdf CLI command."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

pytestmark = pytest.mark.pdf


def create_test_pdf_file() -> Path:
    """Create a minimal test PDF file."""
    # Minimal PDF content (valid but simple)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_file.write(pdf_content)
    temp_file.close()
    return Path(temp_file.name)


def run_bookwyrm_command(
    args: List[str], input_data: str = None
) -> subprocess.CompletedProcess:
    """Run a bookwyrm CLI command and return the result."""
    cmd = ["python", "-m", "bookwyrm.cli"] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        input=input_data,
        timeout=60,  # PDF processing can take longer
    )
    return result


def test_extract_pdf_command_basic_help():
    """Test that the extract-pdf command shows help information."""
    result = run_bookwyrm_command(["extract-pdf", "--help"])

    assert result.returncode == 0
    assert "extract-pdf" in result.stdout.lower()
    assert "pdf" in result.stdout.lower()


def test_extract_pdf_command_missing_args():
    """Test extract-pdf command with missing required arguments."""
    result = run_bookwyrm_command(["extract-pdf"])

    # Should fail due to missing PDF source argument
    assert result.returncode != 0


def test_extract_pdf_command_with_local_file():
    """Test basic extract-pdf command functionality with local PDF file."""
    # Create temporary PDF file
    test_file = create_test_pdf_file()

    try:
        # Run extract-pdf command
        result = run_bookwyrm_command(["extract-pdf", str(test_file)])

        # Check that command executed (may fail due to API key, but should parse args correctly)
        if result.returncode != 0:
            # If it fails due to API key or network, that's expected in test environment
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )
        else:
            # If it succeeds, should have some output
            assert len(result.stdout) > 0

    finally:
        # Clean up
        test_file.unlink()


def test_extract_pdf_command_with_file_option():
    """Test extract-pdf command using --file option instead of positional argument."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(["extract-pdf", "--file", str(test_file)])

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        test_file.unlink()


def test_extract_pdf_command_with_output_option():
    """Test extract-pdf command with --output option."""
    test_file = create_test_pdf_file()
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            ["extract-pdf", "--file", str(test_file), "--output", str(output_path)]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )

    finally:
        test_file.unlink()
        if output_path.exists():
            output_path.unlink()


def test_extract_pdf_command_with_page_range():
    """Test extract-pdf command with --start-page and --num-pages options."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--start-page",
                "1",
                "--num-pages",
                "1",
            ]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )

    finally:
        test_file.unlink()


def test_extract_pdf_command_with_start_page_only():
    """Test extract-pdf command with only --start-page option."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            ["extract-pdf", "--file", str(test_file), "--start-page", "1"]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )

    finally:
        test_file.unlink()




def test_extract_pdf_command_with_verbose_option():
    """Test extract-pdf command with --verbose option."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            ["extract-pdf", "--file", str(test_file), "--verbose"]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )

    finally:
        test_file.unlink()


def test_extract_pdf_command_with_url_option():
    """Test extract-pdf command with --url option."""
    result = run_bookwyrm_command(
        ["extract-pdf", "--url", "https://example.com/document.pdf"]
    )

    # Should fail due to network/API issues in test environment, but args should parse
    assert result.returncode != 0
    # Should fail on network/API, not argument parsing
    assert not ("usage:" in result.stderr.lower() and "error:" in result.stderr.lower())


def test_extract_pdf_command_with_api_options():
    """Test extract-pdf command with --api-key and --base-url options."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--api-key",
                "test-key",
                "--base-url",
                "https://test.example.com",
            ]
        )

        # Should fail on API call but args should parse correctly
        if result.returncode != 0:
            # Should not be an argument parsing error
            assert not (
                "usage:" in result.stderr.lower() and "error:" in result.stderr.lower()
            )

    finally:
        test_file.unlink()


def test_extract_pdf_command_invalid_file():
    """Test extract-pdf command with non-existent file."""
    result = run_bookwyrm_command(["extract-pdf", "--file", "/nonexistent/file.pdf"])

    assert result.returncode != 0
    assert (
        "file" in result.stderr.lower()
        or "not found" in result.stderr.lower()
        or "no such file" in result.stderr.lower()
    )


def test_extract_pdf_command_invalid_page_range():
    """Test extract-pdf command with invalid page range."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--start-page",
                "0",  # Invalid: pages are 1-based
            ]
        )

        # Should fail due to invalid page number or pass validation to API
        assert result.returncode != 0

    finally:
        test_file.unlink()


def test_extract_pdf_command_negative_num_pages():
    """Test extract-pdf command with negative num-pages."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--start-page",
                "1",
                "--num-pages",
                "-1",  # Invalid: negative pages
            ]
        )

        # Should fail due to invalid page count
        assert result.returncode != 0

    finally:
        test_file.unlink()


def test_extract_pdf_command_multiple_input_sources():
    """Test extract-pdf command with multiple input sources (should fail)."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                str(test_file),  # Positional argument
                "--file",
                str(test_file),  # Also --file option
            ]
        )

        # Should fail due to multiple input sources
        assert result.returncode != 0
        assert (
            "multiple" in result.stderr.lower()
            or "one" in result.stderr.lower()
            or "either" in result.stderr.lower()
        )

    finally:
        test_file.unlink()


def test_extract_pdf_command_url_and_file():
    """Test extract-pdf command with both --url and --file (should fail)."""
    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--url",
                "https://example.com/doc.pdf",
            ]
        )

        # Should fail due to multiple input sources
        assert result.returncode != 0
        assert (
            "multiple" in result.stderr.lower()
            or "one" in result.stderr.lower()
            or "either" in result.stderr.lower()
        )

    finally:
        test_file.unlink()


def test_extract_pdf_command_with_complex_options():
    """Test extract-pdf command with multiple options combined."""
    test_file = create_test_pdf_file()
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--start-page",
                "1",
                "--num-pages",
                "1",
                "--output",
                str(output_path),
                "--verbose",
            ]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
            )

    finally:
        test_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_extract_pdf_command_live_api_local_file(api_key, api_url):
    """Test extract-pdf command against live API with local file."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_pdf_file()
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--output",
                str(output_path),
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Check that output file was created and contains valid JSON
        if output_path.exists():
            with open(output_path, "r") as f:
                output_data = json.load(f)
            assert isinstance(output_data, dict)
            assert "pages" in output_data or "total_pages" in output_data

    finally:
        test_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_extract_pdf_command_live_api_streaming(api_key, api_url):
    """Test extract-pdf command streaming functionality with live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--verbose",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain progress or extraction information
        assert (
            "page" in result.stdout.lower()
            or "progress" in result.stdout.lower()
            or "extracted" in result.stdout.lower()
            or "processing" in result.stdout.lower()
        )

    finally:
        test_file.unlink()


@pytest.mark.liveonly
def test_extract_pdf_command_live_api_with_page_range(api_key, api_url):
    """Test extract-pdf command against live API with page range."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_pdf_file()

    try:
        result = run_bookwyrm_command(
            [
                "extract-pdf",
                "--file",
                str(test_file),
                "--start-page",
                "1",
                "--num-pages",
                "1",
                "--api-key",
                api_key,
                "--base-url",
                api_url,
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should process only the specified page range
        assert "page" in result.stdout.lower()

    finally:
        test_file.unlink()




@pytest.mark.liveonly
def test_extract_pdf_command_live_api_url_source(api_key, api_url):
    """Test extract-pdf command against live API with URL source."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Use a known PDF URL (this might fail if the URL is not accessible)
    result = run_bookwyrm_command(
        [
            "extract-pdf",
            "--url",
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "--api-key",
            api_key,
            "--base-url",
            api_url,
        ]
    )

    # This test might fail due to network issues, so we're more lenient
    if result.returncode == 0:
        assert len(result.stdout) > 0
        assert "page" in result.stdout.lower() or "extracted" in result.stdout.lower()
    else:
        # If it fails, it should be due to network/URL issues, not argument parsing
        assert not (
            "usage:" in result.stderr.lower() and "error:" in result.stderr.lower()
        )
