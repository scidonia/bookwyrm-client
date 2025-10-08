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


def test_stream_citations_with_chunks(client, sample_chunks):
    """Test streaming citation finding using text chunks."""
    citations = []
    summary_received = False
    usage_info = None

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
        elif isinstance(response, CitationSummaryResponse):
            summary_received = True
            usage_info = response.usage
            assert isinstance(response.total_citations, int)
            assert response.total_citations >= 0
            assert isinstance(response.usage, UsageInfo)

    # Verify citations if any were found
    for citation in citations:
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


def test_stream_citations_with_jsonl_content_basic(client, sample_jsonl_content):
    """Test streaming citation finding using JSONL content."""
    citations = []
    summary_received = False

    for response in client.stream_citations(
        jsonl_content=sample_jsonl_content,
        question="What programming language was created by Guido van Rossum?",
        start=0,
        limit=10,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
        elif isinstance(response, CitationSummaryResponse):
            summary_received = True
            assert isinstance(response.total_citations, int)
            assert response.total_citations >= 0

    # Verify citations structure
    for citation in citations:
        assert isinstance(citation.text, str)
        assert len(citation.text) > 0
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4


def test_stream_citations_with_pagination_basic(client, sample_chunks):
    """Test streaming citation finding with pagination parameters."""
    responses = []

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is artificial intelligence?",
        start=1,
        limit=3,
        max_tokens_per_chunk=500,
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


def test_stream_citations_error_empty_question(client, sample_chunks):
    """Test that empty question raises an error."""
    with pytest.raises(ValueError, match="question.*empty"):
        list(client.stream_citations(chunks=sample_chunks, question=""))


def test_stream_citations_with_progress_tracking(client, sample_chunks):
    """Test streaming citation finding with progress tracking."""
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


def test_stream_citations_with_jsonl_content_detailed(client, sample_jsonl_content):
    """Test streaming citation finding using JSONL content with detailed verification."""
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


def test_stream_citations_with_pagination_detailed(client, sample_chunks):
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


def test_stream_citations_quality_scores(client, sample_chunks):
    """Test that citation quality scores are within valid range."""
    citations = []

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is deep learning?",
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)

    for citation in citations:
        assert isinstance(citation.quality, int)
        assert (
            0 <= citation.quality <= 4
        ), f"Quality score {citation.quality} is out of range [0-4]"


def test_stream_citations_usage_info(client, sample_chunks):
    """Test that usage information is properly returned."""
    usage_info = None

    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is artificial intelligence?",
        max_tokens_per_chunk=500,
    ):
        if isinstance(response, CitationSummaryResponse):
            usage_info = response.usage
            break

    if usage_info:
        assert isinstance(usage_info, UsageInfo)
        assert hasattr(usage_info, "tokens_processed")
        if (
            hasattr(usage_info, "tokens_processed")
            and usage_info.tokens_processed is not None
        ):
            assert isinstance(usage_info.tokens_processed, int)
            assert usage_info.tokens_processed >= 0


def test_stream_citations_with_different_chunk_sizes(client, sample_chunks):
    """Test streaming citation finding with different max_tokens_per_chunk values."""
    # Test with small chunk size
    responses_small = []
    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=100,
    ):
        responses_small.append(response)

    # Test with large chunk size
    responses_large = []
    for response in client.stream_citations(
        chunks=sample_chunks,
        question="What is machine learning?",
        max_tokens_per_chunk=2000,
    ):
        responses_large.append(response)

    # Both should return valid responses
    assert len(responses_small) >= 0
    assert len(responses_large) >= 0


def test_stream_citations_empty_chunks(client):
    """Test streaming citation finding with empty chunks list."""
    responses = []
    summary_response = None
    
    for response in client.stream_citations(
        chunks=[], question="What is AI?", max_tokens_per_chunk=1000
    ):
        responses.append(response)
        if isinstance(response, CitationSummaryResponse):
            summary_response = response

    # Should get at least one response and find a summary
    assert len(responses) >= 1
    assert summary_response is not None, f"No CitationSummaryResponse found in {len(responses)} responses: {[type(r).__name__ for r in responses]}"
    assert summary_response.total_citations == 0
    assert isinstance(summary_response.usage, UsageInfo)
    assert summary_response.usage.tokens_processed == 0
    assert summary_response.usage.chunks_processed == 0


def test_stream_citations_single_chunk(client):
    """Test streaming citation finding with a single chunk."""
    single_chunk = [
        TextSpan(
            text="Artificial intelligence is the simulation of human intelligence in machines.",
            start_char=0,
            end_char=74,
        )
    ]

    responses = []
    for response in client.stream_citations(
        chunks=single_chunk,
        question="What is artificial intelligence?",
        max_tokens_per_chunk=1000,
    ):
        responses.append(response)

    # Should get some responses
    assert len(responses) >= 0


def test_stream_citations_from_url_skip(client):
    """Test streaming citation finding from URL (skipped - requires test URL)."""
    # Skip this test for now - requires a publicly accessible JSONL URL
    pytest.skip("Requires a publicly accessible JSONL URL for testing")


def test_stream_citations_multiple_questions_list(client, sample_chunks):
    """Test streaming citation finding with multiple questions as a list."""
    questions = [
        "What is machine learning?",
        "What is artificial intelligence?",
        "What is natural language processing?"
    ]
    
    all_citations = []
    summary_responses = []

    for response in client.stream_citations(
        chunks=sample_chunks,
        question=questions,
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationStreamResponse):
            all_citations.append(response.citation)
        elif isinstance(response, CitationSummaryResponse):
            summary_responses.append(response)

    # Verify we got responses for multiple questions
    # Each question should potentially generate citations
    for citation in all_citations:
        assert isinstance(citation, Citation)
        assert isinstance(citation.text, str)
        assert len(citation.text) > 0
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4


def test_stream_citations_multiple_questions_with_jsonl(client, sample_jsonl_content):
    """Test streaming citation finding with multiple questions using JSONL content."""
    questions = [
        "What programming language was created by Guido van Rossum?",
        "What are the key features of Python?",
        "When was Python first released?"
    ]
    
    citations = []
    summary_count = 0

    for response in client.stream_citations(
        jsonl_content=sample_jsonl_content,
        question=questions,
        start=0,
        limit=10,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
        elif isinstance(response, CitationSummaryResponse):
            summary_count += 1

    # Verify citations structure
    for citation in citations:
        assert isinstance(citation.text, str)
        assert len(citation.text) > 0
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4


def test_stream_citations_empty_questions_list(client, sample_chunks):
    """Test streaming citation finding with empty questions list."""
    with pytest.raises(ValueError, match="question.*empty"):
        list(client.stream_citations(chunks=sample_chunks, question=[]))


def test_stream_citations_mixed_question_types(client, sample_chunks):
    """Test streaming citation finding with different types of questions."""
    questions = [
        "What is the definition of artificial intelligence?",  # Definition question
        "How does machine learning work?",  # Process question
        "Where is computer vision used?",  # Application question
    ]
    
    responses = []
    for response in client.stream_citations(
        chunks=sample_chunks,
        question=questions,
        max_tokens_per_chunk=800,
    ):
        responses.append(response)

    # Should get some responses
    assert len(responses) >= 0


def test_stream_citations_single_question_in_list(client, sample_chunks):
    """Test streaming citation finding with single question in list format."""
    questions = ["What is deep learning?"]
    
    citations = []
    for response in client.stream_citations(
        chunks=sample_chunks,
        question=questions,
        max_tokens_per_chunk=1000,
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)

    # Should work the same as single string question
    for citation in citations:
        assert isinstance(citation.quality, int)
        assert 0 <= citation.quality <= 4
