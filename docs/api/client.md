# BookWyrmClient

The synchronous client for the BookWyrm API.

## Class: BookWyrmClient

```python
class BookWyrmClient:
    def __init__(
        self,
        base_url: str = "https://api.bookwyrm.ai:443",
        api_key: Optional[str] = None,
    )
```

### Parameters

- `base_url` (str): Base URL of the BookWyrm API
- `api_key` (Optional[str]): API key for authentication

### Example

```python
from bookwyrm import BookWyrmClient

# Using environment variable for API key
client = BookWyrmClient()

# Explicit API key
client = BookWyrmClient(api_key="your-api-key")

# Custom base URL
client = BookWyrmClient(
    base_url="https://custom-api.example.com",
    api_key="your-api-key"
)
```

## Methods

### get_citations

Find citations for a question from text chunks.

```python
def get_citations(self, request: CitationRequest) -> CitationResponse
```

**Parameters:**
- `request` (CitationRequest): Citation request with chunks and question

**Returns:**
- `CitationResponse`: Citation response with found citations

**Raises:**
- `BookWyrmAPIError`: If the API request fails

**Example:**

```python
from bookwyrm.models import CitationRequest, TextChunk

chunks = [
    TextChunk(text="The sky is blue.", start_char=0, end_char=16),
    TextChunk(text="Water is wet.", start_char=17, end_char=30)
]

request = CitationRequest(
    chunks=chunks,
    question="Why is the sky blue?"
)

response = client.get_citations(request)
print(f"Found {response.total_citations} citations")
```

### stream_citations

Stream citations as they are found.

```python
def stream_citations(
    self, request: CitationRequest
) -> Iterator[StreamingCitationResponse]
```

**Parameters:**
- `request` (CitationRequest): Citation request with chunks and question

**Yields:**
- `StreamingCitationResponse`: Progress updates, citations, summary, or errors

**Example:**

```python
for response in client.stream_citations(request):
    if hasattr(response, 'message'):  # Progress update
        print(f"Progress: {response.message}")
    elif hasattr(response, 'citation'):  # Citation found
        print(f"Citation: {response.citation.text}")
    elif hasattr(response, 'total_citations'):  # Summary
        print(f"Complete: {response.total_citations} citations found")
```

### summarize

Get a summary of the provided content.

```python
def summarize(self, request: SummarizeRequest) -> SummaryResponse
```

**Parameters:**
- `request` (SummarizeRequest): Summarization request with content and options

**Returns:**
- `SummaryResponse`: Summary response with the generated summary

**Example:**

```python
from bookwyrm.models import SummarizeRequest

request = SummarizeRequest(
    content=jsonl_content,
    max_tokens=10000,
    debug=True
)

response = client.summarize(request)
print(response.summary)
```

### stream_summarize

Stream summarization progress and results.

```python
def stream_summarize(
    self, request: SummarizeRequest
) -> Iterator[StreamingSummarizeResponse]
```

**Parameters:**
- `request` (SummarizeRequest): Summarization request with content and options

**Yields:**
- `StreamingSummarizeResponse`: Progress updates, summary, or errors

**Example:**

```python
for response in client.stream_summarize(request):
    if hasattr(response, 'message'):  # Progress update
        print(f"Progress: {response.message}")
    elif hasattr(response, 'summary'):  # Final summary
        print(f"Summary: {response.summary}")
```

### process_text

Process text using phrasal analysis.

```python
def process_text(
    self, request: ProcessTextRequest
) -> Iterator[StreamingPhrasalResponse]
```

**Parameters:**
- `request` (ProcessTextRequest): Text processing request

**Yields:**
- `StreamingPhrasalResponse`: Progress updates and phrase results

**Example:**

```python
from bookwyrm.models import ProcessTextRequest, ResponseFormat

request = ProcessTextRequest(
    text="Your text here",
    response_format=ResponseFormat.WITH_OFFSETS
)

phrases = []
for response in client.process_text(request):
    if hasattr(response, 'text'):  # Phrase result
        phrases.append(response)
```

### classify

Classify file content to determine file type and format.

```python
def classify(self, request: ClassifyRequest) -> ClassifyResponse
```

**Parameters:**
- `request` (ClassifyRequest): Classification request with base64-encoded content

**Returns:**
- `ClassifyResponse`: Classification response with detected file type

**Example:**

```python
import base64
from bookwyrm.models import ClassifyRequest

# Read file as binary and encode
with open("document.pdf", "rb") as f:
    content = base64.b64encode(f.read()).decode('ascii')

request = ClassifyRequest(
    content=content,
    filename="document.pdf"
)

response = client.classify(request)
print(f"File type: {response.classification.format_type}")
```

### extract_pdf

Extract structured data from a PDF file.

```python
def extract_pdf(self, request: PDFExtractRequest) -> PDFExtractResponse
```

**Parameters:**
- `request` (PDFExtractRequest): PDF extraction request

**Returns:**
- `PDFExtractResponse`: PDF extraction response with structured data

**Example:**

```python
from bookwyrm.models import PDFExtractRequest

request = PDFExtractRequest(
    pdf_url="https://example.com/document.pdf",
    start_page=1,
    num_pages=5
)

response = client.extract_pdf(request)
print(f"Extracted {response.total_pages} pages")
```

### stream_extract_pdf

Stream PDF extraction with progress updates.

```python
def stream_extract_pdf(
    self, request: PDFExtractRequest
) -> Iterator[StreamingPDFResponse]
```

**Parameters:**
- `request` (PDFExtractRequest): PDF extraction request

**Yields:**
- `StreamingPDFResponse`: Metadata, pages, completion, or errors

**Example:**

```python
pages = []
for response in client.stream_extract_pdf(request):
    if hasattr(response, 'page_data'):  # Page extracted
        pages.append(response.page_data)
    elif hasattr(response, 'total_pages'):  # Metadata
        print(f"Processing {response.total_pages} pages")
```

## Context Manager Support

The client supports context manager usage for automatic cleanup:

```python
with BookWyrmClient() as client:
    response = client.get_citations(request)
    # Client is automatically closed when exiting the context
```

## Error Handling

The client raises specific exceptions for different error conditions:

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

try:
    response = client.get_citations(request)
except BookWyrmAPIError as e:
    print(f"API Error: {e}")
    if e.status_code:
        print(f"Status Code: {e.status_code}")
except BookWyrmClientError as e:
    print(f"Client Error: {e}")
```

## Session Management

The client uses a `requests.Session` internally for connection pooling and cookie persistence. You can close the session manually:

```python
client.close()
```

Or use the context manager for automatic cleanup:

```python
with BookWyrmClient() as client:
    # Use client
    pass
# Session is automatically closed
```
