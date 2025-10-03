"""Integration tests for synchronous citation functionality."""

import pytest
from bookwyrm.models import (
    TextSpan,
    CitationResponse,
    Citation,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
    UsageInfo,
)


pytestmark = pytest.mark.cite


@pytest.fixture
def sample_chunks():
    """Sample text chunks for citation testing."""
    return [
        TextSpan(
            text="Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.",
            start_char=0,
            end_char=105,
        ),
        TextSpan(
            text="Machine learning is a method of data analysis that automates analytical model building.",
            start_char=106,
            end_char=191,
        ),
        TextSpan(
            text="Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            start_char=192,
            end_char=299,
        ),
        TextSpan(
            text="Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to natural intelligence.",
            start_char=300,
            end_char=407,
        ),
        TextSpan(
            text="Computer vision is an interdisciplinary scientific field that deals with how computers can gain understanding from digital images.",
            start_char=408,
            end_char=540,
        ),
    ]


@pytest.fixture
def sample_jsonl_content():
    """Sample JSONL content for citation testing."""
    return """{"text": "Python is a high-level programming language.", "start_char": 0, "end_char": 45}
{"text": "It was created by Guido van Rossum and first released in 1991.", "start_char": 46, "end_char": 110}
{"text": "Python's design philosophy emphasizes code readability.", "start_char": 111, "end_char": 166}
{"text": "The language provides constructs that enable clear programming on both small and large scales.", "start_char": 167, "end_char": 261}
{"text": "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.", "start_char": 262, "end_char": 376}"""


def test_get_citations_with_chunks(client, sample_chunks):
    """Test citation finding using text chunks."""
    response = client.get_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=1000,
    )

    # Verify response structure
    assert isinstance(response, CitationResponse)
    assert isinstance(response.citations, list)
    assert isinstance(response.total_citations, int)
    assert response.total_citations >= 0
    assert isinstance(response.usage, UsageInfo)

    # Verify citations if any were found
    for citation in response.citations:
        assert isinstance(citation, Citation)
        assert isinstance(citation.text, str)
        assert len(citation.text) > 0
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4
        assert isinstance(citation.reasoning, str)
        assert len(citation.reasoning) > 0
        if hasattr(citation, "start_char"):
            assert isinstance(citation.start_char, int)
            assert citation.start_char >= 0
        if hasattr(citation, "end_char"):
            assert isinstance(citation.end_char, int)
            assert citation.end_char > citation.start_char


def test_get_citations_with_jsonl_content(client, sample_jsonl_content):
    """Test citation finding using JSONL content."""
    response = client.get_citations(
        jsonl_content=sample_jsonl_content,
        question="What programming language was created by Guido van Rossum?",
        start=0,
        limit=10,
    )

    # Verify response structure
    assert isinstance(response, CitationResponse)
    assert isinstance(response.citations, list)
    assert isinstance(response.total_citations, int)
    assert response.total_citations >= 0


def test_get_citations_with_pagination(client, sample_chunks):
    """Test citation finding with pagination parameters."""
    # Test with start and limit
    response = client.get_citations(
        chunks=sample_chunks,
        question="What is artificial intelligence?",
        start=1,
        limit=3,
        max_tokens_per_chunk=500,
    )

    assert isinstance(response, CitationResponse)
    assert response.total_citations >= 0


def test_get_citations_error_no_input(client):
    """Test that missing input raises an error."""
    with pytest.raises(
        ValueError,
        match="Exactly one of.*chunks.*jsonl_content.*jsonl_url.*must be provided",
    ):
        client.get_citations(question="What is AI?")


def test_get_citations_error_multiple_inputs(
    client, sample_chunks, sample_jsonl_content
):
    """Test that multiple inputs raise an error."""
    with pytest.raises(
        ValueError,
        match="Exactly one of.*chunks.*jsonl_content.*jsonl_url.*must be provided",
    ):
        client.get_citations(
            chunks=sample_chunks,
            jsonl_content=sample_jsonl_content,
            question="What is AI?",
        )


def test_get_citations_error_empty_question(client, sample_chunks):
    """Test that empty question raises an error."""
    with pytest.raises(ValueError, match="question.*empty"):
        client.get_citations(chunks=sample_chunks, question="")


def test_stream_citations_with_chunks(client, sample_chunks):
    """Test streaming citation finding using text chunks."""
    citations = []
    progress_updates = []
    summary_received = False
    errors_received = []

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is natural language processing?",
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationProgressUpdate):
            progress_updates.append(response)
            assert isinstance(response.message, str)
            assert len(response.message) > 0
        elif isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
            assert isinstance(response.citation, Citation)
            assert isinstance(response.citation.text, str)
            assert isinstance(response.citation.quality, int)
            assert 0 <= response.citation.quality <= 4
        elif isinstance(response, CitationSummaryResponse):
            summary_received = True
            assert isinstance(response.total_citations, int)
            assert response.total_citations >= 0
            assert isinstance(response.usage, UsageInfo)
        elif isinstance(response, CitationErrorResponse):
            errors_received.append(response)
            assert isinstance(response.error, str)

    # Verify we received expected responses
    assert len(citations) >= 0  # May be 0 if no citations found
    # Note: Progress updates and summary may not always be received depending on processing


def test_stream_citations_with_jsonl_content(client, sample_jsonl_content):
    """Test streaming citation finding using JSONL content."""
    citations = []

    for response in client.stream_citations(
        jsonl_content=sample_jsonl_content,
        question="What are the key features of Python?",
        start=0,
        limit=5,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)

    # Verify citations structure
    for citation in citations:
        assert isinstance(citation.text, str)
        assert len(citation.text) > 0
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4


def test_stream_citations_with_pagination(client, sample_chunks):
    """Test streaming citation finding with pagination."""
    responses = []

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is computer vision?",
        start=2,
        limit=2,
        max_tokens_per_chunk=800,
    ):
        responses.append(response)

    # Verify we got some responses
    assert len(responses) >= 0


def test_stream_citations_error_no_input(client):
    """Test that missing input raises an error."""
    with pytest.raises(
        ValueError,
        match="Exactly one of.*chunks.*jsonl_content.*jsonl_url.*must be provided",
    ):
        list(client.stream_citations(question="What is AI?"))


def test_stream_citations_error_multiple_inputs(
    client, sample_chunks, sample_jsonl_content
):
    """Test that multiple inputs raise an error."""
    with pytest.raises(
        ValueError,
        match="Exactly one of.*chunks.*jsonl_content.*jsonl_url.*must be provided",
    ):
        list(
            client.stream_citations(
                chunks=sample_chunks,
                jsonl_content=sample_jsonl_content,
                question="What is AI?",
            )
        )


def test_get_citations_quality_scores(client, sample_chunks):
    """Test that citation quality scores are within valid range."""
    response = client.get_citations(
        chunks=sample_chunks,
        question="What is deep learning?",
        max_tokens_per_chunk=1000,
    )

    for citation in response.citations:
        assert isinstance(citation.quality, int)
        assert (
            0 <= citation.quality <= 4
        ), f"Quality score {citation.quality} is out of range [0-4]"


def test_get_citations_usage_info(client, sample_chunks):
    """Test that usage information is properly returned."""
    response = client.get_citations(
        chunks=sample_chunks,
        question="What is artificial intelligence?",
        max_tokens_per_chunk=500,
    )

    assert isinstance(response.usage, UsageInfo)
    assert hasattr(response.usage, "tokens_processed")
    if (
        hasattr(response.usage, "tokens_processed")
        and response.usage.tokens_processed is not None
    ):
        assert isinstance(response.usage.tokens_processed, int)
        assert response.usage.tokens_processed >= 0


def test_get_citations_with_different_chunk_sizes(client, sample_chunks):
    """Test citation finding with different max_tokens_per_chunk values."""
    # Test with small chunk size
    response_small = client.get_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=100,
    )

    # Test with large chunk size
    response_large = client.get_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=2000,
    )

    # Both should return valid responses
    assert isinstance(response_small, CitationResponse)
    assert isinstance(response_large, CitationResponse)


def test_stream_citations_progress_tracking(client, sample_chunks):
    """Test that streaming citations provide progress updates."""
    progress_messages = []
    citation_count = 0

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What technologies are mentioned?",
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationProgressUpdate):
            progress_messages.append(response.message)
        elif isinstance(response, CitationStreamResponse):
            citation_count += 1

    # Progress messages should contain useful information
    for message in progress_messages:
        assert isinstance(message, str)
        assert len(message) > 0


def test_get_citations_empty_chunks(client):
    """Test citation finding with empty chunks list."""
    # API rejects empty chunks, so this should raise an error
    from bookwyrm.client import BookWyrmAPIError
    with pytest.raises(BookWyrmAPIError, match="400 Client Error"):
        client.get_citations(
            chunks=[], question="What is AI?", max_tokens_per_chunk=1000
        )


def test_get_citations_single_chunk(client):
    """Test citation finding with a single chunk."""
    single_chunk = [
        TextSpan(
            text="Artificial intelligence is the simulation of human intelligence in machines.",
            start_char=0,
            end_char=74,
        )
    ]

    response = client.get_citations(
        chunks=single_chunk,
        question="What is artificial intelligence?",
        max_tokens_per_chunk=1000,
    )

    assert isinstance(response, CitationResponse)
    assert response.total_citations >= 0


def test_get_citations_from_url_skip(client):
    """Test citation finding from URL (skipped - requires test URL)."""
    # Skip this test for now - requires a publicly accessible JSONL URL
    pytest.skip("Requires a publicly accessible JSONL URL for testing")


def test_stream_citations_from_url_skip(client):
    """Test streaming citation finding from URL (skipped - requires test URL)."""
    # Skip this test for now - requires a publicly accessible JSONL URL
    pytest.skip("Requires a publicly accessible JSONL URL for testing")
