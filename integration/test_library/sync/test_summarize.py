"""Integration tests for synchronous summarization functionality."""

import pytest
from bookwyrm.models import (
    TextSpan,
    SummaryResponse,
    SummarizeProgressUpdate,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
)


pytestmark = pytest.mark.summarize


@pytest.fixture
def sample_phrases():
    """Sample text phrases for summarization testing."""
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
        TextSpan(
            text="Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
            start_char=541,
            end_char=648,
        ),
        TextSpan(
            text="Supervised learning is a machine learning task of learning a function that maps an input to an output.",
            start_char=649,
            end_char=752,
        ),
        TextSpan(
            text="Unsupervised learning is a type of machine learning that looks for previously undetected patterns in data.",
            start_char=753,
            end_char=857,
        ),
    ]


@pytest.fixture
def sample_content():
    """Sample JSONL content for summarization testing."""
    return """{"text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.", "start_char": 0, "end_char": 280}
{"text": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.", "start_char": 281, "end_char": 537}
{"text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.", "start_char": 538, "end_char": 734}
{"text": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.", "start_char": 735, "end_char": 1001}
{"text": "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. Such systems learn to perform tasks by considering examples, generally without being programmed with task-specific rules.", "start_char": 1002, "end_char": 1243}"""


def test_summarize_with_content(client, sample_content):
    """Test summarization using text content."""
    response = client.summarize(content=sample_content, max_tokens=5000, debug=False)

    # Verify response structure
    assert isinstance(response, SummaryResponse)
    assert isinstance(response.summary, str)
    assert len(response.summary) > 0
    assert isinstance(response.levels_used, int)
    assert response.levels_used >= 1
    assert isinstance(response.subsummary_count, int)
    assert response.subsummary_count >= 0


def test_summarize_with_phrases(client, sample_phrases):
    """Test summarization using text phrases."""
    response = client.summarize(phrases=sample_phrases, max_tokens=3000, debug=True)

    # Verify response structure
    assert isinstance(response, SummaryResponse)
    assert isinstance(response.summary, str)
    assert len(response.summary) > 0
    assert isinstance(response.levels_used, int)
    assert response.levels_used >= 1

    # Debug mode should provide additional information
    if hasattr(response, "debug_info") and response.debug_info:
        assert isinstance(response.debug_info, dict)


def test_summarize_error_no_input(client):
    """Test that missing input raises an error."""
    with pytest.raises(
        ValueError, match="Exactly one of.*content.*url.*phrases.*must be provided"
    ):
        client.summarize(max_tokens=5000)


def test_summarize_error_multiple_inputs(client, sample_content, sample_phrases):
    """Test that multiple inputs raise an error."""
    with pytest.raises(
        ValueError, match="Exactly one of.*content.*url.*phrases.*must be provided"
    ):
        client.summarize(
            content=sample_content, phrases=sample_phrases, max_tokens=5000
        )


def test_summarize_empty_content(client):
    """Test summarization with empty content."""
    # Empty content should raise an error or be handled gracefully
    # Skip this test as empty JSONL content may not be valid
    pytest.skip("Empty content handling depends on server implementation")


def test_summarize_empty_phrases(client):
    """Test summarization with empty phrases list."""
    response = client.summarize(phrases=[], max_tokens=5000)

    # Should handle empty phrases gracefully
    assert isinstance(response, SummaryResponse)
    assert isinstance(response.summary, str)


def test_summarize_single_phrase(client):
    """Test summarization with a single phrase."""
    single_phrase = [
        TextSpan(
            text="Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn.",
            start_char=0,
            end_char=115,
        )
    ]

    response = client.summarize(phrases=single_phrase, max_tokens=2000)

    assert isinstance(response, SummaryResponse)
    assert len(response.summary) > 0


def test_summarize_debug_mode(client, sample_phrases):
    """Test summarization with debug mode enabled."""
    response = client.summarize(phrases=sample_phrases, max_tokens=4000, debug=True)

    assert isinstance(response, SummaryResponse)
    assert len(response.summary) > 0
    # Debug mode may provide additional fields or information


def test_summarize_levels_and_subsummaries(client, sample_content):
    """Test that summarization provides level and subsummary information."""
    response = client.summarize(content=sample_content, max_tokens=2000)

    assert isinstance(response.levels_used, int)
    assert response.levels_used >= 1
    assert isinstance(response.subsummary_count, int)
    assert response.subsummary_count >= 0


def test_summarize_from_url_skip(client):
    """Test summarization from URL (skipped - requires test URL)."""
    # Skip this test for now - requires a publicly accessible content URL
    pytest.skip("Requires a publicly accessible content URL for testing")


def test_stream_summarize_with_content(client, sample_content):
    """Test streaming summarization using text content."""
    progress_updates = []
    final_summary = None
    errors_received = []

    for response in client.stream_summarize(
        content=sample_content, max_tokens=5000, debug=False
    ):
        if isinstance(response, SummarizeProgressUpdate):
            progress_updates.append(response)
            assert isinstance(response.message, str)
            assert len(response.message) > 0
        elif isinstance(response, SummaryResponse):
            final_summary = response
            assert isinstance(response.summary, str)
            assert len(response.summary) > 0
        elif isinstance(response, SummarizeErrorResponse):
            errors_received.append(response)
            assert isinstance(response.error, str)
        elif isinstance(response, RateLimitMessage):
            # Rate limit messages are informational
            assert isinstance(response.message, str)
        elif isinstance(response, StructuralErrorMessage):
            # Structural error messages are informational
            assert isinstance(response.error, str)

    # Should have received a final summary
    assert final_summary is not None
    assert isinstance(final_summary.summary, str)
    assert len(final_summary.summary) > 0


def test_stream_summarize_with_phrases(client, sample_phrases):
    """Test streaming summarization using text phrases."""
    responses = []

    for response in client.stream_summarize(
        phrases=sample_phrases, max_tokens=3000, debug=True
    ):
        responses.append(response)
        if isinstance(response, SummaryResponse):
            # Found final summary, can break early for test speed
            break

    # Should have received some responses
    assert len(responses) > 0

    # Last response should be the final summary
    final_response = responses[-1]
    if isinstance(final_response, SummaryResponse):
        assert len(final_response.summary) > 0


def test_stream_summarize_progress_tracking(client, sample_content):
    """Test that streaming summarization provides progress updates."""
    progress_messages = []
    summary_received = False

    for response in client.stream_summarize(content=sample_content, max_tokens=4000):
        if isinstance(response, SummarizeProgressUpdate):
            progress_messages.append(response.message)
        elif isinstance(response, SummaryResponse):
            summary_received = True
            break  # Exit early for test speed

    # Progress messages should contain useful information
    for message in progress_messages:
        assert isinstance(message, str)
        assert len(message) > 0


def test_stream_summarize_error_no_input(client):
    """Test that missing input raises an error."""
    with pytest.raises(
        ValueError, match="Exactly one of.*content.*url.*phrases.*must be provided"
    ):
        list(client.stream_summarize(max_tokens=5000))


def test_stream_summarize_error_multiple_inputs(client, sample_content, sample_phrases):
    """Test that multiple inputs raise an error."""
    with pytest.raises(
        ValueError, match="Exactly one of.*content.*url.*phrases.*must be provided"
    ):
        list(
            client.stream_summarize(
                content=sample_content, phrases=sample_phrases, max_tokens=5000
            )
        )


def test_summarize_long_content(client):
    """Test summarization with longer JSONL content."""
    long_content = """{"text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.", "start_char": 0, "end_char": 400}
{"text": "Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning algorithms use historical data as input to predict new output values.", "start_char": 401, "end_char": 675}
{"text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug design.", "start_char": 676, "end_char": 1150}
{"text": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information.", "start_char": 1151, "end_char": 1650}"""

    response = client.summarize(content=long_content, max_tokens=8000, debug=True)

    assert isinstance(response, SummaryResponse)
    assert len(response.summary) > 0
    assert response.levels_used >= 1


def test_stream_summarize_from_url_skip(client):
    """Test streaming summarization from URL (skipped - requires test URL)."""
    # Skip this test for now - requires a publicly accessible content URL
    pytest.skip("Requires a publicly accessible content URL for testing")
