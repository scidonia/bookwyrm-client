"""Pydantic models for the BookWyrm client."""

from pydantic import BaseModel, model_validator
from typing import List, Optional, Union, Literal
from enum import Enum


class TextChunk(BaseModel):
    """Model for a single text chunk with phrasal offsets."""

    text: str
    start_char: int
    end_char: int


class CitationRequest(BaseModel):
    """Request model for citation processing."""

    chunks: Optional[List[TextChunk]] = None
    jsonl_content: Optional[str] = None
    jsonl_url: Optional[str] = None
    question: str
    start: Optional[int] = 0
    limit: Optional[int] = None
    max_tokens_per_chunk: Optional[int] = 1000

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one input source is provided."""
        sources = [self.chunks, self.jsonl_content, self.jsonl_url]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of 'chunks', 'jsonl_content', or 'jsonl_url' must be provided"
            )

        if self.start is not None and self.start < 0:
            raise ValueError("start must be >= 0")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be > 0")

        return self


class Citation(BaseModel):
    """Model for a single citation with phrasal information."""

    start_chunk: int
    end_chunk: int
    text: str
    reasoning: str
    quality: int  # 0-4 scale, 0=unrelated, 4=perfectly answers


class UsageInfo(BaseModel):
    """Usage tracking information."""

    tokens_processed: int
    chunks_processed: int
    estimated_cost: float
    remaining_credits: Optional[float] = None


class CitationResponse(BaseModel):
    """Response model for citation results."""

    citations: List[Citation]
    total_citations: int
    usage: Optional[UsageInfo] = None


class CitationProgressUpdate(BaseModel):
    """Progress update during citation processing."""

    type: Literal["progress"] = "progress"
    chunks_processed: int
    total_chunks: int
    citations_found: int
    current_chunk_range: str
    message: str


class CitationStreamResponse(BaseModel):
    """Individual citation found during streaming."""

    type: Literal["citation"] = "citation"
    citation: Citation


class CitationSummaryResponse(BaseModel):
    """Final summary of citation processing."""

    type: Literal["summary"] = "summary"
    total_citations: int
    chunks_processed: int
    token_chunks_processed: int
    start_offset: int
    usage: UsageInfo


class CitationErrorResponse(BaseModel):
    """Error during citation processing."""

    type: Literal["error"] = "error"
    error: str


StreamingCitationResponse = Union[
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
]


class Phrase(BaseModel):
    """Model for a phrase with text and optional position information."""

    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class SummarizeRequest(BaseModel):
    """Request model for summarization processing."""

    content: Optional[str] = None
    url: Optional[str] = None
    phrases: Optional[List[Phrase]] = None
    max_tokens: int = 10000  # Default max tokens for chunking
    debug: bool = False  # Include intermediate summaries in response

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one input source is provided."""
        sources = [self.content, self.url, self.phrases]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of 'content', 'url', or 'phrases' must be provided"
            )

        if self.max_tokens > 131072:
            raise ValueError(
                f"max_tokens cannot exceed 131,072 (got {self.max_tokens})"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1 (got {self.max_tokens})")

        return self


class SummarizeProgressUpdate(BaseModel):
    """Progress update during summarization processing."""

    type: Literal["progress"] = "progress"
    current_level: int
    total_levels: int
    chunks_processed: int
    total_chunks: int
    summaries_created: int
    message: str


class SummaryResponse(BaseModel):
    """Response model for summarization results."""

    type: Literal["summary"] = "summary"
    summary: str
    subsummary_count: int
    levels_used: int
    total_tokens: int
    intermediate_summaries: Optional[List[List[str]]] = (
        None  # Debug: summaries by level
    )


class SummarizeErrorResponse(BaseModel):
    """Error during summarization processing."""

    type: Literal["error"] = "error"
    error: str


StreamingSummarizeResponse = Union[
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
]


class ResponseFormat(str, Enum):
    """Response format options for phrasal processing."""

    TEXT_ONLY = "text_only"
    WITH_OFFSETS = "with_offsets"


class ProcessTextRequest(BaseModel):
    """Request model for phrasal text processing.

    Example usage with URL:
        request = ProcessTextRequest(
            text_url="https://www.gutenberg.org/cache/epub/32706/pg32706.txt",
            chunk_size=1000,
            response_format=ResponseFormat.WITH_OFFSETS
        )
    """

    text: Optional[str] = None
    text_url: Optional[str] = None
    chunk_size: Optional[int] = None
    response_format: ResponseFormat = ResponseFormat.WITH_OFFSETS
    spacy_model: str = "en_core_web_sm"

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one of text or text_url is provided."""
        if not self.text and not self.text_url:
            raise ValueError("Either 'text' or 'text_url' must be provided")
        if self.text and self.text_url:
            raise ValueError("Only one of 'text' or 'text_url' should be provided")
        return self


class PhraseProgressUpdate(BaseModel):
    """Progress update for phrasal processing."""

    type: Literal["progress"] = "progress"
    phrases_processed: int
    chunks_created: int
    bytes_processed: int
    message: str


class PhraseResult(BaseModel):
    """Result containing a phrase or chunk."""

    type: Literal["phrase"] = "phrase"
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


StreamingPhrasalResponse = Union[
    PhraseProgressUpdate,
    PhraseResult,
]


class ClassifyRequest(BaseModel):
    """Request model for file classification."""

    content: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None  # Optional hint for classification
    content_encoding: Optional[str] = (
        None  # "base64" if content is base64-encoded binary
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one of content or url is provided."""
        sources = [self.content, self.url]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError("Exactly one of 'content' or 'url' must be provided")

        return self


class FileClassification(BaseModel):
    """Model for file classification results."""

    format_type: str  # General file format (e.g., "text", "image", "binary", "archive")
    content_type: (
        str  # Specific content type (e.g., "python_code", "json_data", "jpeg_image")
    )
    mime_type: str  # MIME type detected
    confidence: float  # Confidence score 0.0-1.0
    details: dict  # Additional classification details (encoding, pygments_lexer, etc.)
    classification_methods: Optional[List[str]] = (
        None  # Methods used for classification
    )


class ClassifyResponse(BaseModel):
    """Response model for classification results."""

    classification: FileClassification
    file_size: int
    sample_preview: Optional[str] = None  # First few characters if text-based
