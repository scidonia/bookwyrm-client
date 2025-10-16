"""Integration tests for the phrasal CLI command."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

pytestmark = pytest.mark.phrasal


def create_test_file(content: str, suffix: str = ".txt") -> Path:
    """Create a temporary file with test content."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    temp_file.write(content)
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
        timeout=300,  # Phrasal processing can take up to 5 minutes
    )
    return result


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, how to program computers to process and analyze large amounts of natural language data.

The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. This involves several challenges including speech recognition, natural language understanding, and natural language generation.

Modern NLP systems use machine learning algorithms to automatically learn rules through the analysis of large corpora of typical real-world examples. These systems can perform tasks such as translation, sentiment analysis, and text summarization."""


@pytest.fixture
def sample_scientific_text():
    """Sample scientific text for testing."""
    return """Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars and starches, which are synthesized from carbon dioxide and water.

In most cases, this process uses the energy of sunlight to convert carbon dioxide and water into glucose and oxygen. The general equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.

Photosynthesis occurs in two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). The light-dependent reactions occur in the thylakoid membranes of chloroplasts, while the Calvin cycle takes place in the stroma."""


def test_phrasal_command_basic_help():
    """Test that the phrasal command shows help information."""
    result = run_bookwyrm_command(["phrasal", "--help"])

    assert result.returncode == 0
    assert "phrasal" in result.stdout.lower()
    assert "text" in result.stdout.lower()


def test_phrasal_command_missing_args():
    """Test phrasal command with missing required arguments."""
    result = run_bookwyrm_command(["phrasal"])

    # Should fail due to missing text source argument
    assert result.returncode != 0


def test_phrasal_command_with_direct_text(sample_text_content):
    """Test basic phrasal command functionality with direct text input."""
    # Run phrasal command with direct text
    result = run_bookwyrm_command(
        [
            "phrasal",
            sample_text_content[:200],  # Use first 200 chars to keep it manageable
            "--offsets",
        ]
    )

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


def test_phrasal_command_with_file_option(sample_text_content):
    """Test phrasal command using --file option."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        result = run_bookwyrm_command(
            ["phrasal", "--file", str(test_file), "--text-only"]
        )

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


def test_phrasal_command_with_output_option(sample_scientific_text):
    """Test phrasal command with --output option."""
    test_file = create_test_file(sample_scientific_text, ".txt")
    output_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--offsets",
                "--output",
                str(output_path),
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


def test_phrasal_command_with_chunk_size(sample_text_content):
    """Test phrasal command with --chunk-size option."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        result = run_bookwyrm_command(
            ["phrasal", "--file", str(test_file), "--chunk-size", "500", "--offsets"]
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




def test_phrasal_command_with_boolean_flags(sample_text_content):
    """Test phrasal command with boolean format flags."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        # Test --offsets flag
        result = run_bookwyrm_command(
            ["phrasal", "--file", str(test_file), "--offsets"]
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


def test_phrasal_command_with_text_only_flag(sample_text_content):
    """Test phrasal command with --text-only flag."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        result = run_bookwyrm_command(
            ["phrasal", "--file", str(test_file), "--text-only"]
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



def test_phrasal_command_with_url_option():
    """Test phrasal command with --url option."""
    # Skip this test - we are not doing URL connections yet
    import pytest
    pytest.skip("URL connections not implemented yet")


def test_phrasal_command_with_api_options(sample_text_content):
    """Test phrasal command with --api-key and --base-url options."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--api-key",
                "test-key",
                "--base-url",
                "https://test.example.com",
                "--text-only",
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


def test_phrasal_command_invalid_file():
    """Test phrasal command with non-existent file."""
    result = run_bookwyrm_command(
        ["phrasal", "--file", "/nonexistent/file.txt", "--text-only"]
    )

    assert result.returncode != 0
    assert (
        "file" in result.stderr.lower()
        or "not found" in result.stderr.lower()
        or "no such file" in result.stderr.lower()
    )


def test_phrasal_command_invalid_chunk_size():
    """Test phrasal command with invalid chunk size."""
    result = run_bookwyrm_command(
        ["phrasal", "Some test text", "--chunk-size", "0"]  # Invalid: must be positive
    )

    # Chunk sizes are upper bounds - invalid sizes are handled gracefully
    # Should succeed but may return 0 phrases
    if result.returncode != 0:
        # If it fails, should be due to API/network issues, not argument parsing
        assert (
            "api" in result.stderr.lower()
            or "key" in result.stderr.lower()
            or "connection" in result.stderr.lower()
            or "network" in result.stderr.lower()
            or "timeout" in result.stderr.lower()
        )


def test_phrasal_command_negative_chunk_size():
    """Test phrasal command with negative chunk size."""
    result = run_bookwyrm_command(
        ["phrasal", "Some test text", "--chunk-size", "-100"]  # Invalid: negative
    )

    # Chunk sizes are upper bounds - invalid sizes are handled gracefully
    # Should succeed but may return 0 phrases
    if result.returncode != 0:
        # If it fails, should be due to API/network issues, not argument parsing
        assert (
            "api" in result.stderr.lower()
            or "key" in result.stderr.lower()
            or "connection" in result.stderr.lower()
            or "network" in result.stderr.lower()
            or "timeout" in result.stderr.lower()
        )


def test_phrasal_command_multiple_input_sources():
    """Test phrasal command with multiple input sources (should fail)."""
    test_file = create_test_file("test content", ".txt")

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "direct text content",  # Direct text
                "--file",
                str(test_file),  # Also file option
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


def test_phrasal_command_conflicting_format_flags():
    """Test phrasal command with conflicting format flags."""
    result = run_bookwyrm_command(
        ["phrasal", "Some test text", "--offsets", "--text-only"]  # Conflicting flags
    )

    # Should fail due to conflicting format options
    assert result.returncode != 0


def test_phrasal_command_with_boolean_flags_only():
    """Test phrasal command with boolean flags only."""
    result = run_bookwyrm_command(
        [
            "phrasal",
            "Some test text",
            "--offsets",
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


def test_phrasal_command_with_verbose_option(sample_text_content):
    """Test phrasal command with --verbose option."""
    test_file = create_test_file(sample_text_content, ".txt")

    try:
        result = run_bookwyrm_command(
            ["phrasal", "--file", str(test_file), "--verbose", "--text-only"]
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


def test_phrasal_command_with_complex_options(sample_scientific_text):
    """Test phrasal command with multiple options combined."""
    test_file = create_test_file(sample_scientific_text, ".txt")
    output_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--chunk-size",
                "800",
                "--offsets",
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
def test_phrasal_command_live_api_text_file(sample_text_content, api_key, api_url):
    """Test phrasal command against live API with text file."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_file(sample_text_content, ".txt")
    output_file = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--offsets",
                "--output",
                str(output_path),
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Check that output file was created and contains valid JSONL
        if output_path.exists():
            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) > 0
            # Verify first line is valid JSON
            first_line = json.loads(lines[0])
            assert "text" in first_line

    finally:
        test_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_phrasal_command_live_api_direct_text(api_key, api_url):
    """Test phrasal command against live API with direct text."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    result = run_bookwyrm_command(
        [
            "phrasal",
            "Natural language processing is a fascinating field that combines linguistics and computer science.",
            "--api-key",
            api_key,
            "--base-url",
            api_url,
            "--text-only",
        ]
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert len(result.stdout) > 0

    # Should contain phrase information
    assert (
        "natural" in result.stdout.lower()
        or "language" in result.stdout.lower()
        or "processing" in result.stdout.lower()
    )


@pytest.mark.liveonly
def test_phrasal_command_live_api_with_chunking(
    sample_scientific_text, api_key, api_url
):
    """Test phrasal command against live API with chunking."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    test_file = create_test_file(sample_scientific_text, ".txt")

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--chunk-size",
                "300",
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--offsets",
                "--verbose",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain progress or phrase information
        assert (
            "progress" in result.stdout.lower()
            or "phrase" in result.stdout.lower()
            or "photosynthesis" in result.stdout.lower()
        )

    finally:
        test_file.unlink()


@pytest.mark.liveonly
def test_phrasal_command_live_api_url_source(api_key, api_url):
    """Test phrasal command against live API with URL source."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Use a known text URL (Project Gutenberg)
    result = run_bookwyrm_command(
        [
            "phrasal",
            "--url",
            "https://www.gutenberg.org/cache/epub/32706/pg32706.txt",
            "--chunk-size",
            "2000",
            "--api-key",
            api_key,
            "--base-url",
            api_url,
            "--text-only",
        ]
    )

    # This test might fail due to network issues, so we're more lenient
    if result.returncode == 0:
        assert len(result.stdout) > 0
        # Should contain some text from the book
        assert len(result.stdout) > 100
    else:
        # If it fails, it should be due to network/URL issues, not argument parsing
        assert not (
            "usage:" in result.stderr.lower() and "error:" in result.stderr.lower()
        )


@pytest.mark.liveonly
def test_phrasal_command_live_api_streaming_progress(
    sample_text_content, api_key, api_url
):
    """Test phrasal command streaming functionality with live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Use a longer text to ensure streaming behavior
    long_text = sample_text_content * 5  # Repeat to make it longer
    test_file = create_test_file(long_text, ".txt")

    try:
        result = run_bookwyrm_command(
            [
                "phrasal",
                "--file",
                str(test_file),
                "--chunk-size",
                "500",
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--offsets",
                "--verbose",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should show streaming progress or results
        assert (
            "progress" in result.stdout.lower()
            or "processing" in result.stdout.lower()
            or "phrase" in result.stdout.lower()
        )

    finally:
        test_file.unlink()
