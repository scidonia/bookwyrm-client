"""Pydantic models for the BookWyrm client."""

import base64
import json
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Literal, Dict, Type
from enum import Enum


class ModelStrength(str, Enum):
    """Model strength levels for processing quality vs speed trade-offs."""

    SWIFT = "swift"  # Fast processing
    SMART = "smart"  # Intelligent analysis
    CLEVER = "clever"  # Advanced reasoning
    WISE = "wise"  # High-quality analysis
    BRAINIAC = "brainiac"  # Maximum sophistication


class Text(BaseModel):
    """Base text model containing just text content."""

    text: str = Field(..., description="The text content")


class Span(BaseModel):
    """Base span model with position information."""

    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")


class TextSpan(Text, Span):
    """Text content with character position information."""

    pass


class CitationRequest(BaseModel):
    """Request model for citation processing.

    Use this model to request citations for a question from text chunks.
    Provide exactly one of: chunks, jsonl_content, or jsonl_url.
    """

    chunks: Optional[List[TextSpan]] = Field(
        None, description="List of text chunks to search"
    )
    jsonl_content: Optional[str] = Field(
        None, description="Raw JSONL content as string"
    )
    jsonl_url: Optional[str] = Field(
        None, description="URL to fetch JSONL content from"
    )
    question: Union[str, List[str]] = Field(
        ..., description="The question(s) to find citations for"
    )
    start: Optional[int] = Field(0, description="Starting chunk index (0-based)")
    limit: Optional[int] = Field(
        None, description="Maximum number of chunks to process"
    )
    max_tokens_per_chunk: Optional[int] = Field(
        1000, description="Maximum tokens per chunk"
    )
    model_strength: ModelStrength = Field(
        ModelStrength.SWIFT,
        description="Model strength level for processing quality vs speed trade-offs",
    )

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one input source is provided and question is not empty."""
        sources = [self.chunks, self.jsonl_content, self.jsonl_url]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of 'chunks', 'jsonl_content', or 'jsonl_url' must be provided"
            )

        # Validate question(s)
        if isinstance(self.question, str):
            if not self.question or not self.question.strip():
                raise ValueError("question cannot be empty")
        elif isinstance(self.question, list):
            if not self.question:
                raise ValueError("question list cannot be empty")
            if len(self.question) > 20:
                raise ValueError("question list cannot contain more than 20 questions")
            for i, q in enumerate(self.question):
                if not q or not q.strip():
                    raise ValueError(f"question at index {i} cannot be empty")
        else:
            raise ValueError("question must be a string or list of strings")

        if self.start is not None and self.start < 0:
            raise ValueError("start must be >= 0")

        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be > 0")

        return self


class Citation(BaseModel):
    """A citation found in response to a question.

    Citations include the relevant text, reasoning for why it's relevant,
    and a quality score indicating how well it answers the question.
    """

    start_chunk: int = Field(..., description="Starting chunk index (inclusive)")
    end_chunk: int = Field(..., description="Ending chunk index (inclusive)")
    text: str = Field(..., description="The citation text content")
    reasoning: str = Field(
        ..., description="Explanation of why this citation is relevant"
    )
    quality: int = Field(
        ..., description="Quality score (0-4): 0=unrelated, 4=perfectly answers"
    )
    question_index: Optional[int] = Field(
        None,
        description="1-based index of the question this citation answers (only present for multi-question requests)",
    )


class UsageInfo(BaseModel):
    """Usage and billing information for API requests.

    Tracks token usage, processing statistics, and cost estimates.
    """

    tokens_processed: int = Field(
        ..., description="Total tokens processed in the request"
    )
    chunks_processed: int = Field(..., description="Number of text chunks processed")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")
    remaining_credits: Optional[float] = Field(
        None, description="Remaining account credits"
    )


class CitationResponse(BaseModel):
    """Response containing citation results and usage information.

    This is the response from non-streaming citation requests.
    """

    citations: List[Citation] = Field(..., description="List of found citations")
    total_citations: int = Field(..., description="Total number of citations found")
    usage: Optional[UsageInfo] = Field(
        None, description="Usage and billing information"
    )


class CitationProgressUpdate(BaseModel):
    """Progress update during citation processing.

    Sent during streaming citation requests to show processing progress.
    """

    type: Literal["progress"] = Field("progress", description="Message type identifier")
    chunks_processed: int = Field(..., description="Number of chunks processed so far")
    total_chunks: int = Field(..., description="Total number of chunks to process")
    citations_found: int = Field(..., description="Number of citations found so far")
    current_chunk_range: str = Field(
        ..., description="Range of chunks currently being processed"
    )
    message: str = Field(..., description="Human-readable progress message")


class CitationStreamResponse(BaseModel):
    """Individual citation found during streaming.

    Sent when a citation is found during streaming citation requests.
    """

    type: Literal["citation"] = Field("citation", description="Message type identifier")
    citation: Citation = Field(..., description="The found citation")


class CitationSummaryResponse(BaseModel):
    """Final summary of citation processing.

    Sent at the end of streaming citation requests with final statistics.
    """

    type: Literal["summary"] = Field("summary", description="Message type identifier")
    total_citations: int = Field(..., description="Total number of citations found")
    chunks_processed: int = Field(..., description="Total number of chunks processed")
    token_chunks_processed: int = Field(
        ..., description="Number of token chunks processed"
    )
    start_offset: int = Field(..., description="Starting offset used for processing")
    usage: UsageInfo = Field(..., description="Usage and billing information")


class CitationErrorResponse(BaseModel):
    """Error during citation processing.

    Sent when an error occurs during streaming citation requests.
    """

    type: Literal["error"] = Field("error", description="Message type identifier")
    error_message: str = Field(
        ..., description="Error message describing what went wrong"
    )
    recoverable: bool = Field(True, description="Whether the error is recoverable")


StreamingCitationResponse = Union[
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
]


class SummarizeRequest(BaseModel):
    """Request model for summarization processing."""

    content: Optional[str] = None
    url: Optional[str] = None
    phrases: Optional[List[TextSpan]] = None
    max_tokens: int = 10000  # Default max tokens for chunking
    debug: bool = False  # Include intermediate summaries in response
    model_strength: ModelStrength = ModelStrength.SWIFT  # Default to swift
    # Pydantic model option for structured output
    model_name: Optional[str] = None
    model_schema_json: Optional[str] = None
    summary_class: Optional[Type[BaseModel]] = Field(
        None, exclude=True
    )  # Direct Pydantic class - excluded from serialization
    # Custom prompt option
    chunk_prompt: Optional[str] = None
    summary_of_summaries_prompt: Optional[str] = None

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

        # Handle direct Pydantic model conversion
        if self.summary_class is not None:
            if self.model_name or self.model_schema_json:
                raise ValueError(
                    "Cannot specify both 'summary_class' and 'model_name'/'model_schema_json'. Use either the direct class or the name/schema pair."
                )

            # Convert Pydantic class to name and schema
            self.model_name = self.summary_class.__name__
            self.model_schema_json = json.dumps(self.summary_class.model_json_schema())
            # Clear the summary_class since it's now converted and excluded from serialization
            self.summary_class = None

        # Structured output validation
        # Check if both pydantic model and custom prompts are specified
        has_pydantic_model = bool(
            self.model_name or self.model_schema_json or self.summary_class
        )
        has_custom_prompts = bool(self.chunk_prompt or self.summary_of_summaries_prompt)

        if has_pydantic_model and has_custom_prompts:
            raise ValueError(
                "Cannot specify both pydantic model options (summary_class/model_name/model_schema_json) and custom prompt options (chunk_prompt/summary_of_summaries_prompt). These are mutually exclusive."
            )

        # Validate pydantic model fields are complete
        if self.model_name and not self.model_schema_json:
            raise ValueError(
                "model_schema_json is required when model_name is provided"
            )
        if self.model_schema_json and not self.model_name:
            raise ValueError(
                "model_name is required when model_schema_json is provided"
            )

        # Validate custom prompts are complete
        if self.chunk_prompt and not self.summary_of_summaries_prompt:
            raise ValueError(
                "summary_of_summaries_prompt is required when chunk_prompt is provided"
            )
        if self.summary_of_summaries_prompt and not self.chunk_prompt:
            raise ValueError(
                "chunk_prompt is required when summary_of_summaries_prompt is provided"
            )

        return self


class SummarizeProgressUpdate(BaseModel):
    """Progress update during summarization processing.

    Sent during streaming summarization to show hierarchical processing progress.
    """

    type: Literal["progress"] = Field("progress", description="Message type identifier")
    current_level: int = Field(
        ..., description="Current hierarchical level being processed"
    )
    total_levels: int = Field(..., description="Total number of hierarchical levels")
    chunks_processed: int = Field(
        ..., description="Number of chunks processed at current level"
    )
    total_chunks: int = Field(
        ..., description="Total number of chunks at current level"
    )
    summaries_created: int = Field(
        ..., description="Number of summaries created so far"
    )
    message: str = Field(..., description="Human-readable progress message")


class SummaryResponse(BaseModel):
    """Response model for summarization results.

    Contains the final summary and metadata about the summarization process.
    """

    type: Literal["summary"] = Field("summary", description="Message type identifier")
    summary: str = Field(..., description="The final summary text or structured JSON")
    subsummary_count: int = Field(
        ..., description="Number of intermediate summaries created"
    )
    levels_used: int = Field(..., description="Number of hierarchical levels used")
    total_tokens: int = Field(..., description="Total tokens processed")
    intermediate_summaries: Optional[List[List[str]]] = Field(
        None, description="Debug information with summaries by level"
    )


class SummarizeErrorResponse(BaseModel):
    """Error during summarization processing.

    Sent when an error occurs during streaming summarization requests.
    """

    type: Literal["error"] = Field("error", description="Message type identifier")
    error: Optional[str] = Field(
        None, description="Error message describing what went wrong"
    )
    recoverable: bool = Field(True, description="Whether the error is recoverable")


class RateLimitMessage(BaseModel):
    """Rate limit retry message during summarization.

    Sent when rate limits are encountered and retries are being attempted.
    """

    type: Literal["rate_limit"] = Field(
        "rate_limit", description="Message type identifier"
    )
    message: str = Field(..., description="Human-readable message about the rate limit")
    attempt: int = Field(..., description="Current retry attempt number")
    max_attempts: int = Field(..., description="Maximum number of retry attempts")
    wait_time: float = Field(..., description="Time to wait before retry (seconds)")
    error_details: Optional[str] = Field(
        None, description="Additional error details if available"
    )


class StructuralErrorMessage(BaseModel):
    """Structural output error message during summarization.

    Sent when structured output parsing fails and retries are being attempted.
    """

    type: Literal["structural_error"] = Field(
        "structural_error", description="Message type identifier"
    )
    message: str = Field(
        ..., description="Human-readable message about the structural error"
    )
    attempt: int = Field(..., description="Current retry attempt number")
    max_attempts: int = Field(..., description="Maximum number of retry attempts")
    error_type: str = Field(..., description="Type of structural error encountered")
    error_details: Optional[str] = Field(
        None, description="Additional error details if available"
    )


StreamingSummarizeResponse = Union[
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
]


class ResponseFormat(str, Enum):
    """Response format options for phrasal processing.

    Determines whether position information is included in phrasal responses.
    """

    TEXT_ONLY = "text_only"
    WITH_OFFSETS = "with_offsets"


class ContentEncoding(str, Enum):
    """Content encoding format for file classification."""

    RAW = "raw"
    UTF8 = "utf-8"
    BASE64 = "base64"


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

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one of text or text_url is provided."""
        if not self.text and not self.text_url:
            raise ValueError("Either 'text' or 'text_url' must be provided")
        if self.text and self.text_url:
            raise ValueError("Only one of 'text' or 'text_url' should be provided")
        return self


class PhraseProgressUpdate(BaseModel):
    """Progress update for phrasal processing.

    Sent during streaming phrasal processing to show progress.
    """

    type: Literal["progress"] = Field("progress", description="Message type identifier")
    phrases_processed: int = Field(
        ..., description="Number of phrases processed so far"
    )
    chunks_created: int = Field(..., description="Number of chunks created so far")
    bytes_processed: int = Field(..., description="Number of bytes processed")
    message: str = Field(..., description="Human-readable progress message")


class TextResult(Text):
    """A simple text result without position information.

    Used when ResponseFormat.TEXT_ONLY is specified in phrasal processing.
    """

    type: Literal["text"] = Field("text", description="Message type identifier")


class TextSpanResult(TextSpan):
    """A text span result with position information.

    Used when ResponseFormat.WITH_OFFSETS is specified in phrasal processing.
    Inherits from TextSpan to include position data.
    """

    type: Literal["text_span"] = Field(
        "text_span", description="Message type identifier"
    )


StreamingPhrasalResponse = Union[
    PhraseProgressUpdate,
    TextResult,
    TextSpanResult,
]


class ClassifyRequest(BaseModel):
    """Request model for file classification."""

    content: Optional[str] = None  # Text or encoded file content
    content_bytes: Optional[bytes] = None  # Raw file bytes
    filename: Optional[str] = None  # Optional hint for classification
    content_encoding: ContentEncoding = ContentEncoding.RAW  # Default to raw bytes

    @model_validator(mode="after")
    def validate_input_source(self):
        """Validate that exactly one of content or content_bytes is provided."""
        sources = [self.content, self.content_bytes]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of 'content' or 'content_bytes' must be provided"
            )

        return self


class FileClassification(BaseModel):
    """Classification results for a file.

    Contains detailed information about the file's format, content type,
    and confidence in the classification.
    """

    format_type: str = Field(
        ...,
        description="General file format (e.g., 'text', 'image', 'binary', 'archive')",
    )
    content_type: str = Field(
        ...,
        description="Specific content type (e.g., 'python_code', 'json_data', 'jpeg_image')",
    )
    mime_type: str = Field(..., description="Detected MIME type")
    confidence: float = Field(
        ..., description="Classification confidence score (0.0-1.0)"
    )
    details: dict = Field(
        ..., description="Additional classification details (encoding, language, etc.)"
    )
    classification_methods: Optional[List[str]] = Field(
        None, description="Methods used for classification"
    )


class ClassifyResponse(BaseModel):
    """Response model for classification results.

    Contains the classification results along with file metadata.
    """

    classification: FileClassification = Field(
        ..., description="The file classification results"
    )
    file_size: int = Field(..., description="Size of the file in bytes")
    sample_preview: Optional[str] = Field(
        None, description="First few characters if text-based file"
    )


class PDFExtractRequest(BaseModel):
    """Request model for PDF structure extraction."""

    pdf_url: Optional[str] = None
    pdf_content: Optional[str] = None  # Base64 encoded PDF content
    pdf_bytes: Optional[bytes] = None  # Raw PDF bytes
    filename: Optional[str] = None  # Optional filename hint
    start_page: Optional[int] = None  # 1-based page number to start from
    num_pages: Optional[int] = None  # Number of pages to process from start_page
    lang: str = "en"  # Language code for OCR processing

    # Feature flags
    enable_layout_detection: bool = False
    force_ocr: bool = False  # Force OCR even for native text PDFs

    @model_validator(mode="after")
    def validate_input_source(self) -> "PDFExtractRequest":
        """Validate that exactly one of pdf_url, pdf_content, or pdf_bytes is provided."""
        sources = [self.pdf_url, self.pdf_content, self.pdf_bytes]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of 'pdf_url', 'pdf_content', or 'pdf_bytes' must be provided"
            )

        if self.start_page is not None and self.start_page < 1:
            raise ValueError("start_page must be >= 1")

        if self.num_pages is not None and self.num_pages < 1:
            raise ValueError("num_pages must be >= 1")

        # Auto-enable force_ocr when layout detection is enabled
        if self.enable_layout_detection and not self.force_ocr:
            self.force_ocr = True

        return self


class PDFBoundingBox(BaseModel):
    """Bounding box coordinates for PDF elements.

    Represents a rectangular bounding box with top-left and bottom-right coordinates.
    """

    x1: float = Field(..., description="Left edge x-coordinate")
    y1: float = Field(..., description="Top edge y-coordinate")
    x2: float = Field(..., description="Right edge x-coordinate")
    y2: float = Field(..., description="Bottom edge y-coordinate")


# New unified layout models to match runpod-pdf schema
class ContentType(str, Enum):
    """Content type enumeration for layout regions."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    SEAL = "seal"


class TextLabelType(str, Enum):
    """Specific text label types from layout detection."""

    TEXT = "text"
    PARAGRAPH_TITLE = "paragraph_title"
    NUMBER = "number"
    TITLE = "title"
    PARAGRAPH = "paragraph"
    HEADER = "header"
    FOOTER = "footer"
    REFERENCE = "reference"
    FIGURE_TITLE = "figure_title"
    REFERENCE_CONTENT = "reference_content"


class ImageLabelType(str, Enum):
    """Specific image label types from layout detection."""

    IMAGE = "image"
    FIGURE = "figure"
    CHART = "chart"
    PHOTO = "photo"
    DIAGRAM = "diagram"


class FormulaLabelType(str, Enum):
    """Specific formula label types from layout detection."""

    FORMULA = "formula"
    EQUATION = "equation"


class TableLabelType(str, Enum):
    """Specific table label types from layout detection."""

    TABLE = "table"
    TABLE_CAPTION = "table caption"


class TextContent(BaseModel):
    """Text content with OCR results and layout classification."""

    content_type: Literal[ContentType.TEXT] = Field(
        default=ContentType.TEXT,
        description="Content type discriminator, always 'text' for text content",
    )
    label: TextLabelType = Field(
        description="Specific text label from layout detection"
    )
    text: Optional[str] = Field(
        default=None, description="OCR extracted text content, if available"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCR confidence score between 0.0 and 1.0",
    )
    reading_order: Optional[int] = Field(
        default=None,
        ge=0,
        description="Reading order index for this text block within the page",
    )
    block_id: Optional[int] = Field(
        default=None,
        ge=0,
        description="Unique identifier for this text block within the page",
    )


class SimpleTable(BaseModel):
    """Simple table representation with rows as lists of cells."""

    rows: List[List[str]] = Field(
        default_factory=list,
        description="Table data as list of rows, each row is a list of cell values",
    )


class TableContent(BaseModel):
    """Table content with structure and cell data.

    Includes both a simplified table representation (simple field) and
    detailed cell-level data for advanced processing.
    """

    content_type: Literal[ContentType.TABLE] = Field(
        default=ContentType.TABLE,
        description="Content type discriminator, always 'table' for table content",
    )
    label: TableLabelType = Field(
        description="Specific table label from layout detection"
    )
    html: Optional[str] = Field(
        default=None, description="HTML representation of the table structure"
    )
    simple: Optional[SimpleTable] = Field(
        default=None,
        description="Simple table representation with rows as lists of cells",
    )
    cells: List[Dict[str, Union[str, int, float]]] = Field(
        default_factory=list,
        description="Structured cell data with positions and content",
    )
    raw_texts: List[str] = Field(
        default_factory=list,
        description="Raw cell texts extracted by OCR, in reading order",
    )
    rows: int = Field(default=0, ge=0, description="Number of rows in the table")
    cols: int = Field(default=0, ge=0, description="Number of columns in the table")
    has_header: bool = Field(
        default=False, description="Whether the table has a header row"
    )
    table_type: Optional[str] = Field(
        default=None,
        description="Classification of table type",
    )


class ImageContent(BaseModel):
    """Image content with metadata and optional extracted data."""

    content_type: Literal[ContentType.IMAGE] = Field(
        default=ContentType.IMAGE,
        description="Content type discriminator, always 'image' for image content",
    )
    label: ImageLabelType = Field(
        description="Specific image label from layout detection"
    )
    image_data: Optional[str] = Field(
        default=None,
        description="Base64 encoded image data extracted from the document",
    )
    file_format: Optional[str] = Field(
        default=None, description="Image file format (e.g., 'png', 'jpg', 'gif')"
    )
    width: Optional[int] = Field(
        default=None, gt=0, description="Image width in pixels"
    )
    height: Optional[int] = Field(
        default=None, gt=0, description="Image height in pixels"
    )
    caption: Optional[str] = Field(
        default=None, description="Image caption text, if detected nearby"
    )


class FormulaContent(BaseModel):
    """Mathematical formula content with LaTeX representation."""

    content_type: Literal[ContentType.FORMULA] = Field(
        default=ContentType.FORMULA,
        description="Content type discriminator, always 'formula' for formula content",
    )
    label: FormulaLabelType = Field(
        description="Specific formula label from layout detection"
    )
    latex: str = Field(
        description="LaTeX representation of the mathematical formula or equation"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Formula recognition confidence score between 0.0 and 1.0",
    )


class SealContent(BaseModel):
    """Circular seal text content."""

    content_type: Literal[ContentType.SEAL] = Field(
        default=ContentType.SEAL,
        description="Content type discriminator, always 'seal' for seal content",
    )
    text: str = Field(description="Text content extracted from the circular seal")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Seal text recognition confidence score between 0.0 and 1.0",
    )


# Tagged union using discriminated union
LayoutContent = Union[
    TextContent, TableContent, ImageContent, FormulaContent, SealContent
]


class UnifiedLayoutRegion(BaseModel):
    """Unified layout region with bounding box and typed content."""

    bbox: List[List[float]] = Field(
        description="Bounding box polygon as list of [x, y] coordinate pairs"
    )
    coordinates: PDFBoundingBox = Field(
        description="Simplified rectangular bounding box coordinates"
    )
    content: LayoutContent = Field(
        discriminator="content_type",
        description="Typed content of the layout region",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Layout detection confidence score between 0.0 and 1.0",
    )


class PDFTextElement(BaseModel):
    """Legacy text element model for backward compatibility."""

    text: str = Field(..., description="The extracted text content")
    confidence: float = Field(..., description="OCR confidence score (0.0-1.0)")
    bbox: List[List[float]] = Field(
        ..., description="Raw bounding box polygon coordinates"
    )
    coordinates: PDFBoundingBox = Field(
        ..., description="Simplified rectangular bounding box"
    )


class PDFPage(BaseModel):
    """Data for a single PDF page with unified layout regions."""

    page_number: int = Field(..., description="The page number (1-based)")

    # New unified layout structure
    layout_regions: List[UnifiedLayoutRegion] = Field(
        default_factory=list,
        description="Unified list of all detected layout regions with typed content",
    )

    # Optional fields
    reading_order: Optional[List[int]] = Field(
        default=None,
        description="Global reading order indices for all content elements",
    )

    @classmethod
    def from_runpod_page_data(cls, page_data: dict) -> "PDFPage":
        """Create PDFPage from runpod-pdf PageData format."""
        return cls(**page_data)

    def get_text_content(self) -> List[TextContent]:
        """Extract all text content from layout regions."""
        return [
            region.content
            for region in self.layout_regions
            if region.content.content_type == ContentType.TEXT
        ]

    def get_table_content(self) -> List[TableContent]:
        """Extract all table content from layout regions."""
        return [
            region.content
            for region in self.layout_regions
            if region.content.content_type == ContentType.TABLE
        ]

    def get_image_content(self) -> List[ImageContent]:
        """Extract all image content from layout regions."""
        return [
            region.content
            for region in self.layout_regions
            if region.content.content_type == ContentType.IMAGE
        ]

    def get_formula_content(self) -> List[FormulaContent]:
        """Extract all formula content from layout regions."""
        return [
            region.content
            for region in self.layout_regions
            if region.content.content_type == ContentType.FORMULA
        ]

    def get_seal_content(self) -> List[SealContent]:
        """Extract all seal content from layout regions."""
        return [
            region.content
            for region in self.layout_regions
            if region.content.content_type == ContentType.SEAL
        ]

    def to_legacy_text_blocks(self) -> List[PDFTextElement]:
        """Convert layout regions to legacy text blocks format for backward compatibility."""
        legacy_blocks = []
        for region in self.layout_regions:
            if region.content.content_type == ContentType.TEXT:
                text_content = region.content
                legacy_block = PDFTextElement(
                    text=text_content.text or "",
                    confidence=text_content.confidence or 1.0,
                    bbox=region.bbox,
                    coordinates=region.coordinates,
                )
                legacy_blocks.append(legacy_block)
        return legacy_blocks


class PDFStructuredData(BaseModel):
    """Complete structured data from PDF extraction."""

    pages: List[PDFPage] = Field(..., description="List of extracted page data")
    total_pages: int = Field(..., description="Total number of pages processed")

    def get_all_text_content(self) -> List[TextContent]:
        """Get all text content from all pages."""
        all_text = []
        for page in self.pages:
            all_text.extend(page.get_text_content())
        return all_text

    def get_all_table_content(self) -> List[TableContent]:
        """Get all table content from all pages."""
        all_tables = []
        for page in self.pages:
            all_tables.extend(page.get_table_content())
        return all_tables


class PDFExtractResponse(BaseModel):
    """Response model for PDF extraction results.

    Contains the extracted PDF data and processing metadata.
    """

    pages: List[PDFPage] = Field(..., description="List of extracted page data")
    total_pages: int = Field(..., description="Total number of pages processed")
    processing_time: Optional[float] = Field(
        None, description="Time taken for processing (seconds)"
    )


class PDFStreamMetadata(BaseModel):
    """Metadata for PDF streaming extraction."""

    type: Literal["metadata"] = "metadata"
    total_pages: int
    total_pages_in_document: int
    start_page: int
    current_page: int


class PDFStreamPageResponse(BaseModel):
    """Individual page response during streaming."""

    type: Literal["page"] = "page"
    total_pages: int
    total_pages_in_document: int
    current_page: int
    document_page: int
    page_data: PDFPage


class PDFStreamPageError(BaseModel):
    """Error processing a specific page during streaming."""

    type: Literal["page_error"] = "page_error"
    total_pages: int
    total_pages_in_document: int
    current_page: int
    document_page: int
    error: str


class PDFStreamComplete(BaseModel):
    """Completion message for streaming extraction."""

    type: Literal["complete"] = "complete"
    total_pages: int
    total_pages_in_document: int
    current_page: int


class PDFStreamError(BaseModel):
    """Error during PDF streaming extraction."""

    model_config = {"extra": "allow"}

    type: Literal["error"] = "error"
    error_type: str
    message: str
    recoverable: bool = False


# PDF Progress and result models for SSE streaming
class PDFProgressUpdate(BaseModel):
    """Progress update for streaming responses."""

    model_config = {"extra": "allow"}

    type: Literal["progress"] = "progress"
    pages_processed: int
    total_pages: int
    message: str


StreamingPDFResponse = Union[
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    PDFProgressUpdate,
]


class ClassifyProgressUpdate(BaseModel):
    """Progress update during file classification.

    Sent during streaming classification to show processing progress.
    """

    type: Literal["progress"] = Field("progress", description="Message type identifier")
    bytes_processed: int = Field(..., description="Number of bytes processed")
    chunks_processed: int = Field(..., description="Number of chunks processed")
    message: str = Field(..., description="Human-readable progress message")


class ClassifyStreamResponse(BaseModel):
    """Classification result for streaming classification.

    Sent when classification is complete during streaming requests.
    """

    type: Literal["classification"] = Field(
        "classification", description="Message type identifier"
    )
    classification: FileClassification = Field(
        ..., description="The file classification results"
    )
    file_size: int = Field(..., description="Size of the file in bytes")
    sample_preview: Optional[str] = Field(
        None, description="First few characters if text-based file"
    )


class ClassifyErrorResponse(BaseModel):
    """Error during classification processing.

    Sent when an error occurs during streaming classification requests.
    """

    type: Literal["error"] = Field("error", description="Message type identifier")
    error_type: str = Field(..., description="Type of error that occurred")
    message: str = Field(..., description="Error message describing what went wrong")
    recoverable: bool = Field(True, description="Whether the error is recoverable")


StreamingClassifyResponse = Union[
    ClassifyProgressUpdate,
    ClassifyStreamResponse,
    ClassifyErrorResponse,
]


class CharacterMapping(BaseModel):
    """Mapping from character position in raw text to bounding box coordinates."""

    char_index: int = Field(..., description="Character index in the raw text")
    page_number: int = Field(..., description="PDF page number (1-based)")
    x1: float = Field(..., description="Left edge x-coordinate")
    y1: float = Field(..., description="Top edge y-coordinate")
    x2: float = Field(..., description="Right edge x-coordinate")
    y2: float = Field(..., description="Bottom edge y-coordinate")
    confidence: float = Field(..., description="OCR confidence score (0.0-1.0)")
    original_text_element_index: int = Field(
        ..., description="Index of the original text element on the page"
    )


class PDFTextMapping(BaseModel):
    """Complete mapping from PDF extraction to raw text with character positions."""

    raw_text: str = Field(..., description="The complete raw text with newlines")
    character_mappings: List[CharacterMapping] = Field(
        ..., description="Character position to bounding box mappings"
    )
    total_pages: int = Field(..., description="Total number of pages processed")
    total_characters: int = Field(
        ..., description="Total number of characters in raw text"
    )
    source_file: Optional[str] = Field(
        None, description="Source PDF extraction JSON file"
    )

    def get_bounding_boxes_for_range(
        self, start_char: int, end_char: int
    ) -> Dict[int, List[Dict[str, Union[int, float]]]]:
        """Get bounding boxes for a character range, grouped by page.

        Args:
            start_char: Starting character index (inclusive)
            end_char: Ending character index (exclusive)

        Returns:
            Dictionary mapping page numbers to lists of bounding box info.
            Each bounding box dict contains: char_index, x1, y1, x2, y2, confidence, original_text_element_index

        Examples:
            ```python
            # Get bounding boxes for characters 100-200
            boxes = mapping.get_bounding_boxes_for_range(100, 200)

            # boxes = {
            #     1: [{'char_index': 100, 'x1': 50.0, 'y1': 100.0, ...}, ...],
            #     2: [{'char_index': 180, 'x1': 25.0, 'y1': 50.0, ...}, ...]
            # }

            for page_num, page_boxes in boxes.items():
                print(f"Page {page_num}: {len(page_boxes)} characters")
            ```
        """
        if start_char < 0:
            start_char = 0
        if end_char > len(self.character_mappings):
            end_char = len(self.character_mappings)
        if start_char >= end_char:
            return {}

        result: Dict[int, List[Dict[str, Union[int, float]]]] = {}

        for mapping in self.character_mappings[start_char:end_char]:
            page_num = mapping.page_number
            if page_num not in result:
                result[page_num] = []

            result[page_num].append(
                {
                    "char_index": mapping.char_index,
                    "x1": mapping.x1,
                    "y1": mapping.y1,
                    "x2": mapping.x2,
                    "y2": mapping.y2,
                    "confidence": mapping.confidence,
                    "original_text_element_index": mapping.original_text_element_index,
                }
            )

        return result

    def get_pages_for_range(self, start_char: int, end_char: int) -> List[int]:
        """Get list of page numbers that contain characters in the given range.

        Args:
            start_char: Starting character index (inclusive)
            end_char: Ending character index (exclusive)

        Returns:
            Sorted list of page numbers containing characters in the range

        Examples:
            ```python
            pages = mapping.get_pages_for_range(100, 200)
            # pages = [1, 2, 3]  # Characters 100-200 span pages 1, 2, and 3
            ```
        """
        boxes_by_page = self.get_bounding_boxes_for_range(start_char, end_char)
        return sorted(boxes_by_page.keys())
