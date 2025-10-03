# Examples

This page contains practical examples of using the BookWyrm client library.

## Phrasal Analysis

### Extract Phrases from Text

```python
from typing import List, Union
from bookwyrm import BookWyrmClient
from bookwyrm.models import ProcessTextRequest, ResponseFormat, TextResult, TextSpanResult, PhraseProgressUpdate

# Create client
client: BookWyrmClient = BookWyrmClient()

text: str = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language.
"""

request: ProcessTextRequest = ProcessTextRequest(
    text=text,
    response_format=ResponseFormat.WITH_OFFSETS,
    spacy_model="en_core_web_sm"
)

phrases: List[TextSpanResult] = []
for response in client.process_text(request):
    if isinstance(response, TextSpanResult):
        phrases.append(response)
        print(f"Phrase: {response.text}")
        print(f"Position: {response.start_char}-{response.end_char}")
    elif isinstance(response, PhraseProgressUpdate):
        print(f"Progress: {response.message}")

# phrases is now List[TextSpanResult] where each TextSpanResult has:
# - type: Literal["phrase"]
# - text: str (the phrase content)
# - start_char: int (starting character position)
# - end_char: int (ending character position)
```

### Create Text Chunks

```python
from typing import List

long_text: str = "Your long text content here..."

# Create chunks of specific size
request: ProcessTextRequest = ProcessTextRequest(
    text=long_text,
    chunk_size=1000,  # ~1000 characters per chunk
    response_format=ResponseFormat.WITH_OFFSETS
)

chunks: List[TextSpanResult] = []
for response in client.process_text(request):
    if isinstance(response, TextSpanResult):
        chunks.append(response)

print(f"Created {len(chunks)} chunks")

# chunks is now List[TextSpanResult] where each chunk has:
# - type: Literal["phrase"]
# - text: str (the chunk content)
# - start_char: int (starting character position)
# - end_char: int (ending character position)
```

### Process Text from URL

```python
from typing import TextIO

request: ProcessTextRequest = ProcessTextRequest(
    text_url="https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    chunk_size=2000,
    response_format=ResponseFormat.WITH_OFFSETS
)

# Save to JSONL file
with open("alice_phrases.jsonl", "w") as f:
    f: TextIO
    for response in client.process_text(request):
        if isinstance(response, TextSpanResult):
            f.write(response.model_dump_json() + "\n")
```

## Citation Finding

### Basic Citation Finding

```python
from typing import List
from bookwyrm.models import CitationRequest, TextChunk, CitationResponse, Citation

# Prepare text chunks (you can get these from phrasal analysis above)
chunks: List[TextChunk] = [
    TextChunk(
        text="Climate change refers to long-term shifts in global temperatures and weather patterns.",
        start_char=0,
        end_char=89
    ),
    TextChunk(
        text="The primary cause is human activities, particularly fossil fuel burning.",
        start_char=90,
        end_char=161
    ),
    TextChunk(
        text="This releases greenhouse gases like CO2 into the atmosphere.",
        start_char=162,
        end_char=222
    )
]

# Find citations
request: CitationRequest = CitationRequest(
    chunks=chunks,
    question="What causes climate change?"
)

response: CitationResponse = client.get_citations(request)

print(f"Found {response.total_citations} citations:")
citation: Citation
for citation in response.citations:
    print(f"- Quality {citation.quality}/4: {citation.text}")

# response is CitationResponse with:
# - citations: List[Citation] 
# - total_citations: int
# - usage: Optional[UsageInfo]
#
# Each Citation has:
# - start_chunk: int (inclusive)
# - end_chunk: int (inclusive) 
# - text: str (the citation content)
# - reasoning: str (why it's relevant)
# - quality: int (0-4 scale, 4=best)
```

### Streaming Citations with Progress

```python
from typing import List
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskID
from bookwyrm.models import (
    CitationProgressUpdate, 
    CitationStreamResponse, 
    CitationSummaryResponse, 
    CitationErrorResponse,
    Citation
)

with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
    task: TaskID = progress.add_task("Finding citations...", total=None)
    
    citations: List[Citation] = []
    for update in client.stream_citations(request):
        if isinstance(update, CitationProgressUpdate):
            progress.update(task, description=update.message)
        elif isinstance(update, CitationStreamResponse):
            citations.append(update.citation)
            print(f"Found: {update.citation.text[:50]}...")
        elif isinstance(update, CitationSummaryResponse):
            print(f"Complete: {update.total_citations} citations found")
        elif isinstance(update, CitationErrorResponse):
            print(f"Error: {update.error}")

print(f"Total citations found: {len(citations)}")

# update can be one of:
# - CitationProgressUpdate: type="progress", message: str, chunks_processed: int, etc.
# - CitationStreamResponse: type="citation", citation: Citation
# - CitationSummaryResponse: type="summary", total_citations: int, usage: UsageInfo
# - CitationErrorResponse: type="error", error: str
#
# citations is List[Citation] (same structure as non-streaming)
```

### Using JSONL Files

```python
# Load from JSONL file
request: CitationRequest = CitationRequest(
    jsonl_url="https://example.com/chunks.jsonl",
    question="What is machine learning?",
    start=0,
    limit=100
)

response: CitationResponse = client.get_citations(request)
```

## Text Summarization

### Basic Summarization

```python
from typing import TextIO
from bookwyrm.models import SummarizeRequest, SummaryResponse

# Load JSONL content
with open("book_phrases.jsonl", "r") as f:
    f: TextIO
    content: str = f.read()

request: SummarizeRequest = SummarizeRequest(
    content=content,
    max_tokens=5000,
    debug=True  # Include intermediate summaries
)

response: SummaryResponse = client.summarize(request)

print("Summary:")
print(response.summary)
print(f"\nUsed {response.levels_used} levels")
print(f"Created {response.subsummary_count} subsummaries")

# response is SummaryResponse with:
# - type: Literal["summary"]
# - summary: str (the final summary text)
# - subsummary_count: int
# - levels_used: int (hierarchical levels)
# - total_tokens: int
# - intermediate_summaries: Optional[List[List[str]]] (if debug=True)
```



## PDF Extraction

### Extract Text from PDF

```python
from typing import BinaryIO
import base64
from bookwyrm.models import PDFExtractRequest, PDFExtractResponse, PDFPage, PDFTextElement

# Load PDF file
with open("document.pdf", "rb") as f:
    f: BinaryIO
    pdf_bytes: bytes = f.read()
    pdf_content: str = base64.b64encode(pdf_bytes).decode('ascii')

request: PDFExtractRequest = PDFExtractRequest(
    pdf_content=pdf_content,
    filename="document.pdf"
)

response: PDFExtractResponse = client.extract_pdf(request)

print(f"Extracted {response.total_pages} pages")
page: PDFPage
for page in response.pages:
    print(f"Page {page.page_number}: {len(page.text_blocks)} text elements")
    element: PDFTextElement
    for element in page.text_blocks[:3]:  # Show first 3 elements
        print(f"  - {element.text[:50]}...")

# response is PDFExtractResponse with:
# - pages: List[PDFPage]
# - total_pages: int
# - extraction_method: str (e.g., "paddleocr")
# - processing_time: Optional[float]
#
# Each PDFPage has:
# - page_number: int (1-based)
# - text_blocks: List[PDFTextElement]
# - tables: List[dict] (placeholder)
# - images: List[dict] (placeholder)
#
# Each PDFTextElement has:
# - text: str (extracted text)
# - confidence: float (0.0-1.0 OCR confidence)
# - bbox: List[List[float]] (raw polygon coordinates)
# - coordinates: PDFBoundingBox (x1, y1, x2, y2 rectangle)
```

### Stream PDF Extraction with Progress

```python
from typing import List
from rich.progress import Progress, BarColumn, TaskProgressColumn, TaskID
from bookwyrm.models import (
    PDFExtractRequest, 
    PDFStreamMetadata, 
    PDFStreamPageResponse, 
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    PDFPage
)

request: PDFExtractRequest = PDFExtractRequest(
    pdf_url="https://example.com/document.pdf",
    start_page=1,
    num_pages=10
)

pages: List[PDFPage] = []
with Progress(BarColumn(), TaskProgressColumn()) as progress:
    task: TaskID = progress.add_task("Extracting PDF...", total=100)
    
    for response in client.stream_extract_pdf(request):
        if isinstance(response, PDFStreamMetadata):
            progress.update(task, total=response.total_pages)
        elif isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
            progress.update(task, completed=response.current_page)
        elif isinstance(response, PDFStreamPageError):
            print(f"Error on page {response.document_page}: {response.error}")
        elif isinstance(response, PDFStreamComplete):
            print("PDF extraction completed")
        elif isinstance(response, PDFStreamError):
            print(f"Extraction error: {response.error}")

print(f"Extracted {len(pages)} pages")

# response can be one of:
# - PDFStreamMetadata: type="metadata", total_pages: int, start_page: int, etc.
# - PDFStreamPageResponse: type="page", page_data: PDFPage, document_page: int
# - PDFStreamPageError: type="page_error", error: str, document_page: int
# - PDFStreamComplete: type="complete", current_page: int
# - PDFStreamError: type="error", error: str
#
# pages is List[PDFPage] (same structure as non-streaming)
```

## File Classification

### Classify File Content

```python
from typing import BinaryIO, Any
import base64
from bookwyrm.models import ClassifyRequest, ClassifyResponse

# Read file as binary
with open("unknown_file.dat", "rb") as f:
    f: BinaryIO
    file_bytes: bytes = f.read()
    content: str = base64.b64encode(file_bytes).decode('ascii')

request: ClassifyRequest = ClassifyRequest(
    content=content,
    filename="unknown_file.dat"
)

response: ClassifyResponse = client.classify(request)

print(f"Format: {response.classification.format_type}")
print(f"Content Type: {response.classification.content_type}")
print(f"MIME Type: {response.classification.mime_type}")
print(f"Confidence: {response.classification.confidence:.2%}")

if response.classification.details:
    print("Details:")
    key: str
    value: Any
    for key, value in response.classification.details.items():
        print(f"  {key}: {value}")

# response is ClassifyResponse with:
# - classification: FileClassification
# - file_size: int (bytes)
# - sample_preview: Optional[str] (first few chars if text)
#
# FileClassification has:
# - format_type: str (e.g., "text", "image", "binary")
# - content_type: str (e.g., "python_code", "jpeg_image")
# - mime_type: str (e.g., "text/plain", "image/jpeg")
# - confidence: float (0.0-1.0)
# - details: dict (encoding, language, etc.)
# - classification_methods: Optional[List[str]]
```

## Async Usage

### Using AsyncBookWyrmClient

```python
import asyncio
from bookwyrm import AsyncBookWyrmClient
from bookwyrm.models import CitationResponse, CitationStreamResponse

async def main() -> None:
    client: AsyncBookWyrmClient
    async with AsyncBookWyrmClient() as client:
        # Async citation finding
        response: CitationResponse = await client.get_citations(request)
        print(f"Found {response.total_citations} citations")
        
        # Async streaming
        async for update in client.stream_citations(request):
            if isinstance(update, CitationStreamResponse):
                print(f"Citation: {update.citation.text}")

# Run async code
asyncio.run(main())

# All async methods return the same types as their sync counterparts:
# - get_citations() -> CitationResponse
# - stream_citations() -> AsyncIterator[StreamingCitationResponse]
# - classify() -> ClassifyResponse
# - extract_pdf() -> PDFExtractResponse
# - stream_extract_pdf() -> AsyncIterator[StreamingPDFResponse]
# - etc.
```

## Error Handling

### Comprehensive Error Handling

```python
from typing import Optional
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError
from bookwyrm.models import CitationResponse

try:
    response: CitationResponse = client.get_citations(request)
except BookWyrmAPIError as e:
    status_code: Optional[int] = e.status_code
    if status_code == 401:
        print("Authentication failed - check your API key")
    elif status_code == 429:
        print("Rate limit exceeded - please wait")
    elif status_code == 500:
        print("Server error - please try again later")
    else:
        print(f"API error: {e}")
except BookWyrmClientError as e:
    print(f"Client error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
