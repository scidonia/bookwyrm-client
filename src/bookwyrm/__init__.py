"""BookWyrm client library."""

try:
    from importlib.metadata import version

    __version__ = version("bookwyrm")
except ImportError:
    # Fallback for Python < 3.8 or if package not installed
    __version__ = "unknown"

from .client import BookWyrmClient, BookWyrmClientError, BookWyrmAPIError
from .async_client import AsyncBookWyrmClient
from .models import (
    Text,
    Span,
    TextSpan,
    CitationRequest,
    Citation,
    CitationResponse,
    UsageInfo,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
    StreamingCitationResponse,
    SummarizeRequest,
    SummaryResponse,
    SummarizeProgressUpdate,
    SummarizeErrorResponse,
    StreamingSummarizeResponse,
    ProcessTextRequest,
    ResponseFormat,
    PhraseProgressUpdate,
    TextResult,
    TextSpanResult,
    StreamingPhrasalResponse,
    ClassifyRequest,
    ClassifyResponse,
    FileClassification,
    PDFExtractRequest,
    PDFExtractResponse,
    PDFTextElement,
    PDFPage,
    PDFStructuredData,
)

__all__ = [
    "BookWyrmClient",
    "AsyncBookWyrmClient",
    "BookWyrmClientError",
    "BookWyrmAPIError",
    "Text",
    "Span",
    "TextSpan",
    "CitationRequest",
    "Citation",
    "CitationResponse",
    "UsageInfo",
    "CitationProgressUpdate",
    "CitationStreamResponse",
    "CitationSummaryResponse",
    "CitationErrorResponse",
    "StreamingCitationResponse",
    "SummarizeRequest",
    "SummaryResponse",
    "SummarizeProgressUpdate",
    "SummarizeErrorResponse",
    "StreamingSummarizeResponse",
    "ProcessTextRequest",
    "ResponseFormat",
    "PhraseProgressUpdate",
    "TextResult",
    "TextSpanResult",
    "StreamingPhrasalResponse",
    "ClassifyRequest",
    "ClassifyResponse",
    "FileClassification",
    "PDFExtractRequest",
    "PDFExtractResponse",
    "PDFTextElement",
    "PDFPage",
    "PDFStructuredData",
]
