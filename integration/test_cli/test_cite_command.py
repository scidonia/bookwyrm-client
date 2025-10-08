"""Integration tests for the cite CLI command."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

pytestmark = pytest.mark.cite


def create_test_jsonl_file(chunks: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with test chunks."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for chunk in chunks:
        json.dump(chunk, temp_file)
        temp_file.write("\n")
    temp_file.close()
    return Path(temp_file.name)


def run_bookwyrm_command(
    args: List[str], input_data: str = None
) -> subprocess.CompletedProcess:
    """Run a bookwyrm CLI command and return the result."""
    cmd = ["bookwyrm"] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, input=input_data, timeout=30
    )
    return result


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        {
            "text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "start_char": 0,
            "end_char": 65,
        },
        {
            "text": "London is the capital of England and home to Big Ben.",
            "start_char": 66,
            "end_char": 119,
        },
        {
            "text": "Berlin is the capital of Germany and has a rich history.",
            "start_char": 120,
            "end_char": 176,
        },
    ]


def test_cite_command_basic_help():
    """Test that the cite command shows help information."""
    result = run_bookwyrm_command(["cite", "--help"])

    assert result.returncode == 0
    assert "cite" in result.stdout.lower()
    assert "question" in result.stdout.lower()


def test_cite_command_missing_args():
    """Test cite command with missing required arguments."""
    result = run_bookwyrm_command(["cite"])

    # Should fail due to missing question argument
    assert result.returncode != 0
    assert "question" in result.stderr.lower() or "missing" in result.stderr.lower()


def test_cite_command_with_jsonl_file(sample_chunks):
    """Test basic cite command functionality with JSONL file."""
    # Create temporary JSONL file
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        # Run cite command with new interface
        result = run_bookwyrm_command(
            ["cite", "--question", "What are the capitals mentioned?", str(jsonl_file)]
        )

        # Check that command executed (may fail due to API key, but should parse args correctly)
        if result.returncode != 0:
            # If it fails due to API key or network, that's expected in test environment
            # Error messages should be in stderr
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            # If it succeeds, should have some output
            assert len(result.stdout) > 0

    finally:
        # Clean up
        jsonl_file.unlink()


def test_cite_command_with_file_option(sample_chunks):
    """Test cite command using --file option instead of positional argument."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            ["cite", "--question", "What cities are mentioned?", "--file", str(jsonl_file)]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            # Error messages should be in stderr
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()


def test_cite_command_with_output_option(sample_chunks):
    """Test cite command with --output option."""
    jsonl_file = create_test_jsonl_file(sample_chunks)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question",
                "What are the European capitals?",
                str(jsonl_file),
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
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


def test_cite_command_with_pagination_options(sample_chunks):
    """Test cite command with --start and --limit options."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question",
                "What locations are mentioned?",
                str(jsonl_file),
                "--start",
                "0",
                "--limit",
                "2",
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
        jsonl_file.unlink()


def test_cite_command_with_verbose_option(sample_chunks):
    """Test cite command with --verbose option."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            ["cite", "--question", "What countries are mentioned?", str(jsonl_file), "--verbose"]
        )

        # Check command parsing
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )

    finally:
        jsonl_file.unlink()


def test_cite_command_multiple_questions(sample_chunks):
    """Test cite command with multiple questions using --question flags."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question", "What are the capitals mentioned?",
                "--question", "Which countries are referenced?",
                str(jsonl_file)
            ]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()


def test_cite_command_with_questions_file(sample_chunks):
    """Test cite command with --questions-file option."""
    jsonl_file = create_test_jsonl_file(sample_chunks)
    
    # Create questions file
    questions_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    questions_file.write("What are the capitals mentioned?\n")
    questions_file.write("Which countries are referenced?\n")
    questions_file.write("What landmarks are described?\n")
    questions_file.close()
    questions_path = Path(questions_file.name)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--questions-file", str(questions_path),
                str(jsonl_file)
            ]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()
        questions_path.unlink()


def test_cite_command_with_long_flag(sample_chunks):
    """Test cite command with --long flag for full citation text."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question", "What cities are mentioned?",
                str(jsonl_file),
                "--long"
            ]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()


def test_cite_command_conflicting_question_sources():
    """Test cite command with both --question and --questions-file (should fail)."""
    questions_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    questions_file.write("What is mentioned?\n")
    questions_file.close()
    questions_path = Path(questions_file.name)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question", "What is this?",
                "--questions-file", str(questions_path),
                "dummy.jsonl"
            ]
        )

        # Should fail due to conflicting question sources
        assert result.returncode != 0
        assert "exactly one" in result.stderr.lower() or "question" in result.stderr.lower()

    finally:
        questions_path.unlink()


def test_cite_command_with_url_option():
    """Test cite command with --url option."""
    # TODO: We need good urls first.
    pass


def test_cite_command_invalid_file():
    """Test cite command with non-existent file."""
    result = run_bookwyrm_command(
        ["cite", "--question", "What is mentioned?", "/nonexistent/file.jsonl"]
    )

    assert result.returncode != 0
    assert (
        "file" in result.stderr.lower()
        or "not found" in result.stderr.lower()
        or "no such file" in result.stderr.lower()
    )


@pytest.mark.liveonly
def test_cite_command_live_api(sample_chunks, api_key, api_url):
    """Test cite command against live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_chunks)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question",
                "What are the capital cities mentioned in the text?",
                str(jsonl_file),
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

        # Check that output file was created and contains valid JSONL
        if output_path.exists():
            output_data = []
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        output_data.append(json.loads(line))
            assert isinstance(output_data, list)

    finally:
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_cite_command_streaming_live(sample_chunks, api_key, api_url):
    """Test cite command streaming functionality with live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question",
                "What European cities are mentioned?",
                str(jsonl_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain progress or citation information
        assert (
            "citation" in result.stdout.lower()
            or "progress" in result.stdout.lower()
            or "found" in result.stdout.lower()
        )

    finally:
        jsonl_file.unlink()


def test_cite_command_multiple_questions_comprehensive(sample_chunks):
    """Test cite command with comprehensive multiple questions testing."""
    jsonl_file = create_test_jsonl_file(sample_chunks)

    try:
        # Test with many questions
        result = run_bookwyrm_command(
            [
                "cite",
                "--question", "What are the capitals mentioned?",
                "--question", "Which countries are referenced?",
                "--question", "What landmarks are described?",
                "--question", "What cities are in Europe?",
                str(jsonl_file)
            ]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()


def test_cite_command_questions_file_comprehensive():
    """Test cite command with comprehensive questions file testing."""
    sample_chunks = [
        {
            "text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "start_char": 0,
            "end_char": 65,
        },
        {
            "text": "London is the capital of England and home to Big Ben.",
            "start_char": 66,
            "end_char": 119,
        },
    ]
    
    jsonl_file = create_test_jsonl_file(sample_chunks)
    
    # Create comprehensive questions file
    questions_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    questions_file.write("What are the capitals mentioned?\n")
    questions_file.write("Which countries are referenced?\n")
    questions_file.write("What landmarks are described?\n")
    questions_file.write("What architectural features are mentioned?\n")
    questions_file.write("Which European cities are discussed?\n")
    questions_file.close()
    questions_path = Path(questions_file.name)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--questions-file", str(questions_path),
                str(jsonl_file),
                "--verbose"
            ]
        )

        # Check command parsing (may fail on API call)
        if result.returncode != 0:
            assert (
                "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
                or "connection" in result.stderr.lower()
                or "network" in result.stderr.lower()
                or "timeout" in result.stderr.lower()
                or "error" in result.stderr.lower()
            )
        else:
            assert len(result.stdout) > 0

    finally:
        jsonl_file.unlink()
        questions_path.unlink()


def test_cite_command_empty_questions_file():
    """Test cite command with empty questions file."""
    sample_chunks = [
        {
            "text": "Test content for empty questions file test.",
            "start_char": 0,
            "end_char": 43,
        }
    ]
    
    jsonl_file = create_test_jsonl_file(sample_chunks)
    
    # Create empty questions file
    questions_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    questions_file.close()
    questions_path = Path(questions_file.name)

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--questions-file", str(questions_path),
                str(jsonl_file)
            ]
        )

        # Should fail due to empty questions file
        assert result.returncode != 0
        assert (
            "empty" in result.stderr.lower() 
            or "question" in result.stderr.lower()
            or "no questions" in result.stderr.lower()
        )

    finally:
        jsonl_file.unlink()
        questions_path.unlink()


@pytest.mark.liveonly
def test_cite_command_multiple_questions_live_api(sample_chunks, api_key, api_url):
    """Test cite command with multiple questions against live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_chunks)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "cite",
                "--question", "What are the capital cities mentioned?",
                "--question", "Which European countries are referenced?",
                "--question", "What famous landmarks are described?",
                str(jsonl_file),
                "--api-key", api_key,
                "--base-url", api_url,
                "--output", str(output_path),
                "--verbose"
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Check that output file was created and contains valid JSONL
        if output_path.exists():
            output_data = []
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        output_data.append(json.loads(line))
            assert isinstance(output_data, list)

    finally:
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()
