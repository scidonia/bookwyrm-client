"""Integration tests for the classify CLI command."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

pytestmark = pytest.mark.classify


def create_test_file(content: str, suffix: str = ".txt") -> Path:
    """Create a temporary file with test content."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def create_test_binary_file(content: bytes, suffix: str = ".bin") -> Path:
    """Create a temporary binary file with test content."""
    temp_file = tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False)
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def run_bookwyrm_command(
    args: List[str], input_data: str = None
) -> subprocess.CompletedProcess:
    """Run a bookwyrm CLI command and return the result."""
    cmd = ["python", "-m", "bookwyrm.cli"] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, input=input_data, timeout=30
    )
    return result


@pytest.fixture
def sample_json_content():
    """Sample JSON content for testing."""
    return json.dumps(
        {"name": "test", "data": [1, 2, 3], "nested": {"key": "value"}}, indent=2
    )


@pytest.fixture
def sample_python_content():
    """Sample Python code content for testing."""
    return '''#!/usr/bin/env python3
"""Sample Python script for testing."""

import os
import sys

def main():
    print("Hello, world!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Chicago
"""


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test HTML page.</p>
</body>
</html>
"""


def test_classify_command_basic_help():
    """Test that the classify command shows help information."""
    result = run_bookwyrm_command(["classify", "--help"])

    assert result.returncode == 0
    assert "classify" in result.stdout.lower()
    assert "content" in result.stdout.lower() or "file" in result.stdout.lower()


def test_classify_command_with_empty_stdin():
    """Test classify command with empty stdin."""
    result = run_bookwyrm_command(["classify"], input_data="")

    # Should fail due to empty stdin
    assert result.returncode != 0
    assert "no content" in result.stderr.lower()


def test_classify_command_with_text_file(sample_python_content):
    """Test basic classify command functionality with text file."""
    # Create temporary file
    test_file = create_test_file(sample_python_content, ".py")

    try:
        # Run classify command
        result = run_bookwyrm_command(["classify", "--file", str(test_file)])

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


def test_classify_command_with_output_option(sample_csv_content):
    """Test classify command with --output option."""
    test_file = create_test_file(sample_csv_content, ".csv")
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            ["classify", "--file", str(test_file), "--output", str(output_path)]
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


def test_classify_command_with_filename_hint(sample_python_content):
    """Test classify command with --filename option for better classification."""
    test_file = create_test_file(sample_python_content, ".txt")  # Wrong extension

    try:
        result = run_bookwyrm_command(
            [
                "classify",
                "--file",
                str(test_file),
                "--filename",
                "script.py",  # Correct hint
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


def test_classify_command_with_stdin_content():
    """Test classify command with stdin content input."""
    stdin_content = "import pandas as pd\ndf = pd.DataFrame()"
    
    result = run_bookwyrm_command(
        [
            "classify",
            "--filename",
            "script.py",
        ],
        input_data=stdin_content
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


def test_classify_command_with_url_option():
    """Test classify command with --url option."""
    result = run_bookwyrm_command(
        [
            "classify",
            "--url",
            "https://www.gutenberg.org/ebooks/18857.epub3.images",
            "--filename",
            "alice.epub",
        ]
    )

    # Should fail due to network/API issues in test environment, but args should parse
    assert result.returncode != 0
    # Should fail on network/API, not argument parsing
    assert not ("usage:" in result.stderr.lower() and "error:" in result.stderr.lower())


def test_classify_command_with_api_options(sample_html_content):
    """Test classify command with --api-key and --base-url options."""
    test_file = create_test_file(sample_html_content, ".html")

    try:
        result = run_bookwyrm_command(
            [
                "classify",
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


def test_classify_command_invalid_file():
    """Test classify command with non-existent file."""
    result = run_bookwyrm_command(["classify", "--file", "/nonexistent/file.txt"])

    assert result.returncode != 0
    assert (
        "file" in result.stderr.lower()
        or "not found" in result.stderr.lower()
        or "no such file" in result.stderr.lower()
    )


def test_classify_command_binary_file():
    """Test classify command with binary file content."""
    # Create a simple binary file (fake image header)
    binary_content = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    )
    test_file = create_test_binary_file(binary_content, ".png")

    try:
        result = run_bookwyrm_command(["classify", "--file", str(test_file)])

        # Check command parsing (binary files should be handled)
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


def test_classify_command_multiple_input_sources():
    """Test classify command with multiple input sources (should fail)."""
    test_file = create_test_file("test content", ".txt")

    try:
        result = run_bookwyrm_command(
            ["classify", "--file", str(test_file), "--url", "https://example.com/file.txt"]
        )

        # Should fail due to multiple input sources
        assert result.returncode != 0
        assert (
            "only one" in result.stderr.lower()
            or "one of" in result.stderr.lower()
        )

    finally:
        test_file.unlink()


def test_classify_command_with_verbose_option(sample_json_content):
    """Test classify command with --verbose option."""
    test_file = create_test_file(sample_json_content, ".json")

    try:
        result = run_bookwyrm_command(
            ["classify", "--file", str(test_file), "--verbose"]
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


@pytest.mark.liveonly
def test_classify_command_live_api_text_file(sample_python_content, api_key, api_url):
    """Test classify command against live API with text file."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_file(sample_python_content, ".py")
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "classify",
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
            assert "classification" in output_data

    finally:
        test_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_classify_command_live_api_stdin_content(api_key, api_url):
    """Test classify command against live API with stdin content."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    stdin_content = '{"name": "test", "data": [1, 2, 3]}'
    
    result = run_bookwyrm_command(
        [
            "classify",
            "--filename",
            "data.json",
            "--api-key",
            api_key,
            "--base-url",
            api_url,
        ],
        input_data=stdin_content
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert len(result.stdout) > 0

    # Should contain classification information
    assert (
        "classification" in result.stdout.lower()
        or "format" in result.stdout.lower()
        or "type" in result.stdout.lower()
    )


@pytest.mark.liveonly
def test_classify_command_live_api_binary_file(api_key, api_url):
    """Test classify command against live API with binary file."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Create a simple binary file
    binary_content = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10"
    )
    test_file = create_test_binary_file(binary_content, ".png")

    try:
        result = run_bookwyrm_command(
            [
                "classify",
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

        # Should detect as image or binary
        assert (
            "image" in result.stdout.lower()
            or "png" in result.stdout.lower()
            or "binary" in result.stdout.lower()
        )

    finally:
        test_file.unlink()


@pytest.mark.liveonly
def test_classify_command_live_api_with_filename_hint(
    sample_csv_content, api_key, api_url
):
    """Test classify command against live API with filename hint."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Create file with wrong extension
    test_file = create_test_file(sample_csv_content, ".txt")

    try:
        result = run_bookwyrm_command(
            [
                "classify",
                "--file",
                str(test_file),
                "--filename",
                "data.csv",  # Correct hint
                "--api-key",
                api_key,
                "--base-url",
                api_url,
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should detect as CSV due to filename hint
        assert "csv" in result.stdout.lower() or "comma" in result.stdout.lower()

    finally:
        test_file.unlink()
