"""Integration tests for asynchronous phrasal text processing."""

import pytest
from bookwyrm.models import (
    ResponseFormat,
    TextResult,
    TextSpanResult,
    PhraseProgressUpdate,
)


pytestmark = pytest.mark.phrasal


@pytest.mark.asyncio
async def test_stream_process_text_with_offsets_flag(async_client):
    """Test phrasal processing with offsets boolean flag."""
    text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

    phrases = []
    progress_messages = []

    async for response in async_client.stream_process_text(text=text, offsets=True):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)
        elif isinstance(response, PhraseProgressUpdate):
            progress_messages.append(response.message)

    # Verify we got results
    assert len(phrases) > 0, "Should have extracted phrases"

    # Verify at least some phrases have position information
    text_span_results = [p for p in phrases if isinstance(p, TextSpanResult)]
    assert len(text_span_results) > 0, "Should have phrases with position offsets"

    # Verify position information is valid
    for phrase in text_span_results:
        assert hasattr(phrase, "start_char"), "Should have start_char"
        assert hasattr(phrase, "end_char"), "Should have end_char"
        assert phrase.start_char >= 0, "start_char should be non-negative"
        assert (
            phrase.end_char > phrase.start_char
        ), "end_char should be greater than start_char"
        assert (
            phrase.text == text[phrase.start_char : phrase.end_char]
        ), "Text should match slice"


@pytest.mark.asyncio
async def test_stream_process_text_with_text_only_flag(async_client):
    """Test phrasal processing with text_only boolean flag."""
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence."

    phrases = []

    async for response in async_client.stream_process_text(text=text, text_only=True):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)

    # Verify we got results
    assert len(phrases) > 0, "Should have extracted phrases"

    # All results should be TextResult (no position info)
    for phrase in phrases:
        assert isinstance(
            phrase, TextResult
        ), "Should be TextResult without position info"
        assert hasattr(phrase, "text"), "Should have text"
        assert phrase.text.strip(), "Text should not be empty"


@pytest.mark.asyncio
async def test_stream_process_text_with_chunk_size_and_offsets(async_client):
    """Test phrasal processing with chunk_size and offsets."""
    long_text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves."""

    chunks = []

    async for response in async_client.stream_process_text(
        text=long_text, chunk_size=1000, offsets=True
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            chunks.append(response)

    # Verify we got chunks
    assert len(chunks) > 0, "Should have created chunks"

    # Verify chunks are reasonably sized (not exceeding chunk_size by too much)
    for chunk in chunks:
        assert (
            len(chunk.text) <= 1200
        ), f"Chunk too large: {len(chunk.text)} chars"  # Allow some flexibility

    # Verify at least some chunks have position information
    text_span_chunks = [c for c in chunks if isinstance(c, TextSpanResult)]
    assert len(text_span_chunks) > 0, "Should have chunks with position offsets"


@pytest.mark.asyncio
async def test_stream_process_text_from_url_with_chunk_size(async_client):
    """Test phrasal processing from URL with chunk_size."""
    # TODO: We need to add a URL in a bucket for fetch here.
    pass


@pytest.mark.asyncio
async def test_stream_process_text_url_text_only(async_client):
    """Test phrasal processing from URL with text_only format."""
    # TODO: We need to add a URL in a bucket for fetch here.
    pass


@pytest.mark.asyncio
async def test_stream_process_text_response_format_enum(async_client):
    """Test phrasal processing using ResponseFormat enum."""
    text = "This is a test sentence for response format testing."

    # Test WITH_OFFSETS format
    phrases_with_offsets = []
    async for response in async_client.stream_process_text(
        text=text, response_format=ResponseFormat.WITH_OFFSETS
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_with_offsets.append(response)

    # Test TEXT_ONLY format
    phrases_text_only = []
    async for response in async_client.stream_process_text(
        text=text, response_format=ResponseFormat.TEXT_ONLY
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_text_only.append(response)

    # Verify both formats work
    assert len(phrases_with_offsets) > 0, "Should have phrases with offsets"
    assert len(phrases_text_only) > 0, "Should have text-only phrases"

    # Verify format differences
    text_span_results = [
        p for p in phrases_with_offsets if isinstance(p, TextSpanResult)
    ]
    assert len(text_span_results) > 0, "WITH_OFFSETS should include position info"

    for phrase in phrases_text_only:
        assert isinstance(
            phrase, TextResult
        ), "TEXT_ONLY should not include position info"


@pytest.mark.asyncio
async def test_stream_process_text_response_format_string(async_client):
    """Test phrasal processing using string response format."""
    text = "Testing string format parameters for phrasal processing."

    # Test "with_offsets" string format
    phrases_offsets = []
    async for response in async_client.stream_process_text(
        text=text, response_format="with_offsets"
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_offsets.append(response)

    # Test "text_only" string format
    phrases_text = []
    async for response in async_client.stream_process_text(
        text=text, response_format="text_only"
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_text.append(response)

    # Verify both string formats work
    assert len(phrases_offsets) > 0, "Should work with 'with_offsets' string"
    assert len(phrases_text) > 0, "Should work with 'text_only' string"


@pytest.mark.asyncio
async def test_stream_process_text_alternative_string_formats(async_client):
    """Test phrasal processing using alternative string formats."""
    text = "Testing alternative string format parameters."

    # Test "offsets" string format
    phrases_offsets = []
    async for response in async_client.stream_process_text(
        text=text, response_format="offsets"
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_offsets.append(response)

    # Test "text" string format
    phrases_text = []
    async for response in async_client.stream_process_text(
        text=text, response_format="text"
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases_text.append(response)

    # Verify alternative string formats work
    assert len(phrases_offsets) > 0, "Should work with 'offsets' string"
    assert len(phrases_text) > 0, "Should work with 'text' string"


@pytest.mark.asyncio
async def test_stream_process_text_boolean_flag_precedence(async_client):
    """Test that boolean flags take precedence over response_format parameter."""
    text = "Testing boolean flag precedence over response_format."

    # Boolean flag should override response_format
    phrases = []
    async for response in async_client.stream_process_text(
        text=text,
        response_format=ResponseFormat.TEXT_ONLY,  # This should be overridden
        offsets=True,  # This should take precedence
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)

    # Should have position info despite TEXT_ONLY in response_format
    text_span_results = [p for p in phrases if isinstance(p, TextSpanResult)]
    assert len(text_span_results) > 0, "Boolean flag should override response_format"


@pytest.mark.asyncio
async def test_stream_process_text_error_multiple_boolean_flags(async_client):
    """Test that multiple boolean flags raise an error."""
    text = "Testing multiple boolean flags error."

    with pytest.raises(ValueError, match="Only one response format flag can be True"):
        async for _ in async_client.stream_process_text(
            text=text, offsets=True, text_only=True
        ):
            pass


@pytest.mark.asyncio
async def test_stream_process_text_error_no_input(async_client):
    """Test that missing text input raises an error."""
    with pytest.raises(ValueError, match="Either text or text_url is required"):
        async for _ in async_client.stream_process_text():
            pass


@pytest.mark.asyncio
async def test_stream_process_text_error_invalid_string_format(async_client):
    """Test that invalid string format raises an error."""
    text = "Testing invalid format string."

    with pytest.raises(ValueError, match="Invalid response_format"):
        async for _ in async_client.stream_process_text(
            text=text, response_format="invalid_format"
        ):
            pass
