"""Integration tests for the summarize CLI command."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest

pytestmark = pytest.mark.summarize


def create_test_jsonl_file(phrases: List[Dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with test phrases."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for phrase in phrases:
        json.dump(phrase, temp_file)
        temp_file.write("\n")
    temp_file.close()
    return Path(temp_file.name)


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
        timeout=300,  # Summarization can take up to 5 minutes
    )
    return result


@pytest.fixture
def sample_phrases():
    """Sample text phrases for testing."""
    return [
        {
            "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.",
            "start_char": 0,
            "end_char": 109,
        },
        {
            "text": "The goal of NLP is to enable computers to understand, interpret, and generate human language.",
            "start_char": 110,
            "end_char": 202,
        },
        {
            "text": "Modern NLP systems use machine learning algorithms to automatically learn rules.",
            "start_char": 203,
            "end_char": 282,
        },
        {
            "text": "These systems can perform tasks such as translation, sentiment analysis, and text summarization.",
            "start_char": 283,
            "end_char": 378,
        },
    ]


@pytest.fixture
def sample_scientific_phrases():
    """Sample scientific text phrases for testing."""
    return [
        {
            "text": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
            "start_char": 0,
            "end_char": 87,
        },
        {
            "text": "This chemical energy is stored in carbohydrate molecules, such as sugars and starches.",
            "start_char": 88,
            "end_char": 174,
        },
        {
            "text": "The general equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
            "start_char": 175,
            "end_char": 264,
        },
        {
            "text": "Photosynthesis occurs in two main stages: light-dependent and light-independent reactions.",
            "start_char": 265,
            "end_char": 355,
        },
    ]


@pytest.fixture
def sample_scientific_phrases():
    """Sample scientific text phrases for testing."""
    return [
        {
            "text": "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
            "start_char": 0,
            "end_char": 87,
        },
        {
            "text": "This chemical energy is stored in carbohydrate molecules, such as sugars and starches.",
            "start_char": 88,
            "end_char": 174,
        },
        {
            "text": "The general equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.",
            "start_char": 175,
            "end_char": 264,
        },
        {
            "text": "Photosynthesis occurs in two main stages: light-dependent and light-independent reactions.",
            "start_char": 265,
            "end_char": 355,
        },
    ]


@pytest.fixture
def sample_long_content():
    """Sample long content for testing chunking."""
    return """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do."""


def test_summarize_command_basic_help():
    """Test that the summarize command shows help information."""
    result = run_bookwyrm_command(["summarize", "--help"])

    assert result.returncode == 0
    assert "summarize" in result.stdout.lower()
    assert "jsonl" in result.stdout.lower() or "file" in result.stdout.lower()


def test_summarize_command_missing_args():
    """Test summarize command with missing required arguments."""
    result = run_bookwyrm_command(["summarize"])

    # Should fail due to missing JSONL file argument
    assert result.returncode != 0


def test_summarize_command_basic(sample_phrases):
    """Test basic summarize command functionality with JSONL file."""
    # Create temporary JSONL file
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        # Run summarize command
        result = run_bookwyrm_command(["summarize", str(jsonl_file)])

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
        jsonl_file.unlink()


def test_summarize_command_with_output_option(sample_scientific_phrases):
    """Test summarize command with --output option."""
    jsonl_file = create_test_jsonl_file(sample_scientific_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            ["summarize", str(jsonl_file), "--output", str(output_path)]
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


def test_summarize_command_with_max_tokens_option(sample_phrases):
    """Test summarize command with --max-tokens option."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(
            ["summarize", str(jsonl_file), "--max-tokens", "5000"]
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


def test_summarize_command_debug(sample_phrases):
    """Test summarize command with --include-debug option."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_file), "--include-debug"])

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


def test_summarize_command_streaming(sample_phrases):
    """Test summarize command (always streaming by default)."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_file)])

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


def test_summarize_command_without_streaming_options(sample_phrases):
    """Test summarize command without any streaming options (always streams by default)."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_file)])

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


def test_summarize_command_with_api_options(sample_phrases):
    """Test summarize command with --api-key and --base-url options."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
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
        jsonl_file.unlink()


def test_summarize_command_invalid_file():
    """Test summarize command with non-existent file."""
    result = run_bookwyrm_command(["summarize", "/nonexistent/file.jsonl"])

    assert result.returncode != 0
    assert (
        "file" in result.stderr.lower()
        or "not found" in result.stderr.lower()
        or "no such file" in result.stderr.lower()
    )


def test_summarize_command_invalid_max_tokens():
    """Test summarize command with invalid max-tokens value."""
    result = run_bookwyrm_command(
        [
            "summarize",
            "test.jsonl",  # File doesn't need to exist for this validation test
            "--max-tokens",
            "0",  # Invalid: must be positive
        ]
    )

    # Should fail due to invalid max-tokens value or file not found
    assert result.returncode != 0


def test_summarize_command_negative_max_tokens():
    """Test summarize command with negative max-tokens value."""
    result = run_bookwyrm_command(
        [
            "summarize",
            "test.jsonl",  # File doesn't need to exist for this validation test
            "--max-tokens",
            "-1000",  # Invalid: negative
        ]
    )

    # Should fail due to invalid max-tokens value
    assert result.returncode != 0


def test_summarize_command_with_verbose_option(sample_phrases):
    """Test summarize command with --verbose option."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_file), "--verbose"])

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


def test_summarize_command_with_complex_options(sample_scientific_phrases):
    """Test summarize command with multiple options combined."""
    jsonl_file = create_test_jsonl_file(sample_scientific_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
                "--max-tokens",
                "8000",
                "--include-debug",
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
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


def test_summarize_command_empty_jsonl_file():
    """Test summarize command with empty JSONL file."""
    # Create empty JSONL file
    jsonl_file = create_test_jsonl_file([])

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_file)])

        # Should fail due to empty content or pass to API for validation
        if result.returncode != 0:
            # Could fail on empty content validation or API call
            assert (
                "empty" in result.stderr.lower()
                or "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
            )

    finally:
        jsonl_file.unlink()


def test_summarize_command_malformed_jsonl():
    """Test summarize command with malformed JSONL file."""
    # Create file with invalid JSON
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    temp_file.write('{"text": "valid json"}\n')
    temp_file.write("invalid json line\n")  # This will cause parsing issues
    temp_file.write('{"text": "another valid line"}\n')
    temp_file.close()
    jsonl_path = Path(temp_file.name)

    try:
        result = run_bookwyrm_command(["summarize", str(jsonl_path)])

        # Should fail due to malformed JSON or pass to API for validation
        if result.returncode != 0:
            # Could fail on JSON parsing or API call
            assert (
                "json" in result.stderr.lower()
                or "parse" in result.stderr.lower()
                or "api" in result.stderr.lower()
                or "key" in result.stderr.lower()
            )

    finally:
        jsonl_path.unlink()


def test_summarize_command_large_max_tokens(sample_phrases):
    """Test summarize command with very large max-tokens value."""
    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(
            ["summarize", str(jsonl_file), "--max-tokens", "100000"]  # Very large value
        )

        # Check command parsing (should accept large values)
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


@pytest.mark.liveonly
def test_summarize_command_live_api(sample_phrases, api_key, api_url):
    """Test summarize command against live API with basic functionality."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
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

        # Check that output file was created and contains valid JSON
        if output_path.exists():
            with open(output_path, "r") as f:
                output_data = json.load(f)
            assert isinstance(output_data, dict)
            assert "summary" in output_data

    finally:
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_summarize_command_live_api_streaming(
    sample_scientific_phrases, api_key, api_url
):
    """Test summarize command streaming functionality with live API."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_scientific_phrases)

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--verbose",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain progress or summary information
        assert (
            "progress" in result.stdout.lower()
            or "summary" in result.stdout.lower()
            or "processing" in result.stdout.lower()
        )

    finally:
        jsonl_file.unlink()


@pytest.mark.liveonly
def test_summarize_command_live_api_with_debug(sample_phrases, api_key, api_url):
    """Test summarize command against live API with debug information."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_phrases)

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--include-debug",
                "--max-tokens",
                "5000",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain debug information
        assert (
            "debug" in result.stdout.lower()
            or "level" in result.stdout.lower()
            or "summary" in result.stdout.lower()
        )

    finally:
        jsonl_file.unlink()


@pytest.mark.liveonly
def test_summarize_command_live_api_no_stream(
    sample_scientific_phrases, api_key, api_url
):
    """Test summarize command against live API (always streams by default)."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_scientific_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
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

        # Check that output file was created
        if output_path.exists():
            with open(output_path, "r") as f:
                output_data = json.load(f)
            assert isinstance(output_data, dict)
            assert "summary" in output_data

    finally:
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


@pytest.mark.liveonly
def test_summarize_command_live_api_large_content(
    sample_long_content, api_key, api_url
):
    """Test summarize command against live API with larger content requiring chunking."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    # Create phrases from the long content
    phrases = []
    sentences = sample_long_content.split(". ")
    start_char = 0
    for sentence in sentences:
        if sentence.strip():
            end_char = start_char + len(sentence) + 1  # +1 for the period
            phrases.append(
                {
                    "text": sentence.strip()
                    + ("." if not sentence.endswith(".") else ""),
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )
            start_char = end_char + 1

    jsonl_file = create_test_jsonl_file(phrases)

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--max-tokens",
                "3000",  # Smaller chunks to test hierarchical summarization
                "--include-debug",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Should contain summary of AI/ML content
        assert (
            "artificial" in result.stdout.lower()
            or "intelligence" in result.stdout.lower()
            or "machine" in result.stdout.lower()
            or "learning" in result.stdout.lower()
        )

    finally:
        jsonl_file.unlink()


@pytest.mark.liveonly
def test_summarize_command_live_api_comprehensive_options(
    sample_phrases, api_key, api_url
):
    """Test summarize command against live API with all options combined."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()

    try:
        result = run_bookwyrm_command(
            [
                "summarize",
                str(jsonl_file),
                "--api-key",
                api_key,
                "--base-url",
                api_url,
                "--max-tokens",
                "6000",
                "--include-debug",
                "--output",
                str(output_path),
                "--verbose",
            ]
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0

        # Check that output file was created and contains comprehensive data
        if output_path.exists():
            with open(output_path, "r") as f:
                output_data = json.load(f)
            assert isinstance(output_data, dict)
            assert "summary" in output_data

            # Debug mode should include additional information
            if "debug_info" in output_data or "levels_used" in output_data:
                assert output_data.get("levels_used", 0) >= 1

    finally:
        jsonl_file.unlink()
        if output_path.exists():
            output_path.unlink()


def test_summarize_command_model_strength_levels(sample_phrases, api_key, api_url):
    """Test summarize command with different model strength levels."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_phrases)
    
    # Test different model strength levels
    model_strengths = ["swift", "smart", "clever", "wise", "brainiac"]
    
    for strength in model_strengths:
        output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output_path = Path(output_file.name)
        output_file.close()
        
        try:
            result = run_bookwyrm_command([
                "summarize", str(jsonl_file),
                "--output", str(output_path),
                "--model-strength", strength,
                "--max-tokens", "1000",
                "--verbose",
                "--api-key", api_key,
                "--base-url", api_url,
            ])

            assert result.returncode == 0, f"Failed for model strength: {strength}. Error: {result.stderr}"
            assert len(result.stdout) > 0
            assert output_path.exists()

            # Verify output file structure
            with open(output_path, "r") as f:
                summary_data = json.load(f)

            assert "summary" in summary_data
            assert isinstance(summary_data["summary"], str)
            assert len(summary_data["summary"]) > 0

        finally:
            # Cleanup
            if output_path.exists():
                output_path.unlink()
    
    # Cleanup phrases file
    jsonl_file.unlink()


def test_summarize_command_with_pydantic_model(sample_scientific_phrases, api_key, api_url):
    """Test summarize command with structured Pydantic model output."""
    if not api_key:
        pytest.skip("No API key provided for live test")

    jsonl_file = create_test_jsonl_file(sample_scientific_phrases)
    output_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    output_path = Path(output_file.name)
    output_file.close()
    
    # Create a temporary Pydantic model file
    model_file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    model_content = '''from pydantic import BaseModel, Field
from typing import List, Optional

class ScientificSummary(BaseModel):
    """Structured summary of scientific content."""
    
    title: Optional[str] = Field(None, description="Main topic or title")
    key_concepts: List[str] = Field(default_factory=list, description="Key scientific concepts mentioned")
    main_findings: List[str] = Field(default_factory=list, description="Main findings or conclusions")
    methodology: Optional[str] = Field(None, description="Research methodology if mentioned")
    implications: Optional[str] = Field(None, description="Implications or significance")
    summary: str = Field(..., description="Overall summary of the content")
'''
    
    model_file.write(model_content)
    model_file.close()
    model_path = Path(model_file.name)
    
    try:
        result = run_bookwyrm_command([
            "summarize", str(jsonl_file),
            "--output", str(output_path),
            "--model-class-file", str(model_path),
            "--model-class-name", "ScientificSummary",
            "--model-strength", "wise",
            "--max-tokens", "2000",
            "--verbose",
            "--api-key", api_key,
            "--base-url", api_url,
        ])

        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert len(result.stdout) > 0
        assert output_path.exists()

        # Verify structured output
        with open(output_path, "r") as f:
            summary_data = json.load(f)

        assert "summary" in summary_data
        
        # The summary field should contain structured JSON data
        if isinstance(summary_data["summary"], str):
            # Try to parse as JSON
            try:
                structured_summary = json.loads(summary_data["summary"])
                assert isinstance(structured_summary, dict)
                # Check for expected Pydantic model fields
                assert "summary" in structured_summary  # Required field
                # Optional fields may or may not be present
                possible_fields = ["title", "key_concepts", "main_findings", "methodology", "implications"]
                assert any(field in structured_summary for field in possible_fields)
            except json.JSONDecodeError:
                # If it's not JSON, it should still be a valid summary string
                assert len(summary_data["summary"]) > 0
        elif isinstance(summary_data["summary"], dict):
            # Already parsed as structured data
            assert "summary" in summary_data["summary"]

    finally:
        # Cleanup
        for file_path in [jsonl_file, output_path, model_path]:
            if file_path.exists():
                file_path.unlink()
