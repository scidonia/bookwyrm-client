# Models

Pydantic models used by the BookWyrm client library.

## Citation Models

### TextChunk

Model for a single text chunk with phrasal offsets.

```python
class TextChunk(BaseModel):
    text: str
    start_char: int
    end_char: int
```

**Example:**
```python
chunk = TextChunk(
    text="The sky is blue due to Rayleigh scattering.",
    start_char=0,
    end_char=44
)
```

### CitationRequest

Request model for citation processing.

```python
class CitationRequest(BaseModel):
    chunks: Optional[List[TextChunk]] = None
    jsonl_content: Optional[str] = None
    jsonl_url: Optional[str] = None
    question: str
    start: Optional[int] = 0
    limit: Optional[int] = None
    max_tokens_per_chunk: Optional[int] = 1000
```

**Validation:** Exactly one of `chunks`, `jsonl_content`, or `jsonl_url` must be provided.

**Example:**
```python
request = CitationRequest(
    chunks=[chunk1, chunk2],
    question="Why is the sky blue?",
    start=0,
    limit=100
)
```

### Citation

Model for a single citation with phrasal information.

```python
class Citation(BaseModel):
    start_chunk: int
    end_chunk: int
    text: str
    reasoning: str
    quality: int  # 0-4 scale, 0=unrelated, 4=perfectly answers
```

### CitationResponse

Response model for citation results.

```python
class CitationResponse(BaseModel):
    citations: List[Citation]
    total_citations: int
    usage: Optional[UsageInfo] = None
```

### Streaming Citation Models

- `CitationProgressUpdate`: Progress updates during processing
- `CitationStreamResponse`: Individual citation found
- `CitationSummaryResponse`: Final summary
- `CitationErrorResponse`: Error during processing

## Summarization Models

### Phrase

Model for a phrase with text and optional position information.

```python
class Phrase(BaseModel):
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
```

### SummarizeRequest

Request model for summarization processing.

```python
class SummarizeRequest(BaseModel):
    content: Optional[str] = None
    url: Optional[str] = None
    phrases: Optional[List[Phrase]] = None
    max_tokens: int = 10000
    debug: bool = False
    # Pydantic model option for structured output
    model_name: Optional[str] = None
    model_schema_json: Optional[str] = None
    # Custom prompt option
    chunk_prompt: Optional[str] = None
    summary_of_summaries_prompt: Optional[str] = None
```

**Validation Rules:**
- Exactly one of `content`, `url`, or `phrases` must be provided
- `max_tokens` must be between 1 and 131,072
- Pydantic model options and custom prompts are mutually exclusive
- If using Pydantic model, both `model_name` and `model_schema_json` are required
- If using custom prompts, both `chunk_prompt` and `summary_of_summaries_prompt` are required

**Example:**
```python
# Basic summarization
request = SummarizeRequest(
    content=jsonl_content,
    max_tokens=10000,
    debug=True
)

# Structured output with Pydantic model
request = SummarizeRequest(
    content=jsonl_content,
    model_name="BookSummary",
    model_schema_json=json.dumps(BookSummary.model_json_schema()),
    max_tokens=8000
)

# Custom prompts
request = SummarizeRequest(
    content=jsonl_content,
    chunk_prompt="Extract key concepts...",
    summary_of_summaries_prompt="Create comprehensive overview...",
    max_tokens=12000
)
```

### SummaryResponse

Response model for summarization results.

```python
class SummaryResponse(BaseModel):
    type: Literal["summary"] = "summary"
    summary: str
    subsummary_count: int
    levels_used: int
    total_tokens: int
    intermediate_summaries: Optional[List[List[str]]] = None
```

### Streaming Summarization Models

- `SummarizeProgressUpdate`: Progress updates during processing
- `SummarizeErrorResponse`: Error during processing
- `RateLimitMessage`: Rate limit retry messages
- `StructuralErrorMessage`: Structural output error messages

## Phrasal Analysis Models

### ProcessTextRequest

Request model for phrasal text processing.

```python
class ProcessTextRequest(BaseModel):
    text: Optional[str] = None
    text_url: Optional[str] = None
    chunk_size: Optional[int] = None
    response_format: ResponseFormat = ResponseFormat.WITH_OFFSETS
    spacy_model: str = "en_core_web_sm"
```

**Validation:** Exactly one of `text` or `text_url` must be provided.

### ResponseFormat

Response format options for phrasal processing.

```python
class ResponseFormat(str, Enum):
    TEXT_ONLY = "text_only"
    WITH_OFFSETS = "with_offsets"
```

### Streaming Phrasal Models

- `PhraseProgressUpdate`: Progress updates
- `PhraseResult`: Individual phrase or chunk result

## Classification Models

### ClassifyRequest

Request model for file classification.

```python
class ClassifyRequest(BaseModel):
    content: str  # Base64-encoded file content
    filename: Optional[str] = None
    content_encoding: str = "base64"
```

### FileClassification

Model for file classification results.

```python
class FileClassification(BaseModel):
    format_type: str  # e.g., "text", "image", "binary"
    content_type: str  # e.g., "python_code", "json_data"
    mime_type: str
    confidence: float  # 0.0-1.0
    details: dict
    classification_methods: Optional[List[str]] = None
```

### ClassifyResponse

Response model for classification results.

```python
class ClassifyResponse(BaseModel):
    classification: FileClassification
    file_size: int
    sample_preview: Optional[str] = None
```

## PDF Extraction Models

### PDFExtractRequest

Request model for PDF structure extraction.

```python
class PDFExtractRequest(BaseModel):
    pdf_url: Optional[str] = None
    pdf_content: Optional[str] = None  # Base64 encoded
    filename: Optional[str] = None
    start_page: Optional[int] = None  # 1-based
    num_pages: Optional[int] = None
```

**Validation:** Exactly one of `pdf_url` or `pdf_content` must be provided.

### PDFBoundingBox

Bounding box coordinates.

```python
class PDFBoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
```

### PDFTextElement

A text element extracted from PDF.

```python
class PDFTextElement(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]]  # Raw polygon
    coordinates: PDFBoundingBox  # Simplified rectangle
```

### PDFPage

Data for a single PDF page.

```python
class PDFPage(BaseModel):
    page_number: int
    text_blocks: List[PDFTextElement]
    tables: List[dict] = []  # Future feature
    images: List[dict] = []  # Future feature
```

### PDFExtractResponse

Response model for PDF extraction results.

```python
class PDFExtractResponse(BaseModel):
    pages: List[PDFPage]
    total_pages: int
    extraction_method: str = "paddleocr"
    processing_time: Optional[float] = None
```

### Streaming PDF Models

- `PDFStreamMetadata`: Metadata about the extraction
- `PDFStreamPageResponse`: Individual page response
- `PDFStreamPageError`: Error processing a page
- `PDFStreamComplete`: Completion message
- `PDFStreamError`: General error

## Utility Models

### UsageInfo

Usage tracking information.

```python
class UsageInfo(BaseModel):
    tokens_processed: int
    chunks_processed: int
    estimated_cost: Optional[float] = None
    remaining_credits: Optional[float] = None
```

## Union Types

The library defines several union types for streaming responses:

```python
StreamingCitationResponse = Union[
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
]

StreamingSummarizeResponse = Union[
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
]

StreamingPhrasalResponse = Union[
    PhraseProgressUpdate,
    PhraseResult,
]

StreamingPDFResponse = Union[
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
]
```

## Model Validation

All models use Pydantic validation. Common validation patterns:

### Input Source Validation

Many models validate that exactly one input source is provided:

```python
@model_validator(mode="after")
def validate_input_source(self):
    sources = [self.content, self.url, self.phrases]
    provided_sources = [s for s in sources if s is not None]
    
    if len(provided_sources) != 1:
        raise ValueError("Exactly one input source must be provided")
    
    return self
```

### Range Validation

Numeric fields often have range validation:

```python
if self.max_tokens > 131072:
    raise ValueError(f"max_tokens cannot exceed 131,072")
if self.max_tokens < 1:
    raise ValueError(f"max_tokens must be at least 1")
```

### Mutual Exclusion

Some models validate mutually exclusive options:

```python
has_pydantic_model = bool(self.model_name or self.model_schema_json)
has_custom_prompts = bool(self.chunk_prompt or self.summary_of_summaries_prompt)

if has_pydantic_model and has_custom_prompts:
    raise ValueError("Cannot specify both pydantic model and custom prompts")
```

## Creating Custom Models

For structured summarization, you can create custom Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class CustomSummary(BaseModel):
    title: Optional[str] = Field(None, description="Document title")
    key_points: Optional[List[str]] = Field(None, description="Main points")
    conclusion: Optional[str] = Field(None, description="Final conclusion")

# Use with SummarizeRequest
import json

request = SummarizeRequest(
    content=content,
    model_name="CustomSummary",
    model_schema_json=json.dumps(CustomSummary.model_json_schema())
)
```

The model schema is serialized to JSON and sent to the API, where it's used to guide the LLM's structured output generation.
