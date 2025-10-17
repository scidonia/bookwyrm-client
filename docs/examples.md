# Examples

This page contains practical examples of using the BookWyrm client library.

## Phrasal Analysis

### Extract Phrases from Text

```python
from typing import List, Union
from bookwyrm import BookWyrmClient
from bookwyrm.models import ResponseFormat, TextResult, TextSpanResult, PhraseProgressUpdate

# Create client
client: BookWyrmClient = BookWyrmClient()

text: str = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language.
"""

# Using function arguments (recommended)
phrases: List[TextSpanResult] = []
for response in client.stream_process_text(
    text=text,
    offsets=True,  # or response_format="with_offsets" or ResponseFormat.WITH_OFFSETS
):
    if isinstance(response, TextSpanResult):
        phrases.append(response)
        print(f"Phrase: {response.text}")
        print(f"Position: {response.start_char}-{response.end_char}")
    elif isinstance(response, PhraseProgressUpdate):
        print(f"Progress: {response.message}")

# phrases is now List[TextSpanResult] where each TextSpanResult has:
# - type: Literal["text_span"]
# - text: str (the phrase content)
# - start_char: int (starting character position)
# - end_char: int (ending character position)
```

### Create Phrasal Text Chunks

```python
from typing import List

# Example text with multiple sentences
text: str = """Natural language processing enables computers to understand human language. 
Machine learning algorithms power these systems. Deep learning has revolutionized the field. 
Modern NLP applications include chatbots, translation, and sentiment analysis."""

# Create phrasal chunks bounded by size - fit as many complete phrases as possible
chunks: List[TextSpanResult] = []
for response in client.stream_process_text(
    text=text,
    chunk_size=125,  # Bounded by 125 characters per chunk (smaller for demo)
    offsets=True  # boolean flag for WITH_OFFSETS
):
    if isinstance(response, TextSpanResult):
        chunks.append(response)

print(f"Created {len(chunks)} phrasal chunks")

# Example output for the above text:
# Chunk 1: "Natural language processing enables computers to understand human language. Machine learning algorithms power these systems."
# Chunk 2: "Deep learning has revolutionized the field. Modern NLP applications include chatbots, translation, and sentiment analysis."

# Each chunk is bounded by the chunk size but fits as many complete phrases/sentences as possible up to that limit
# chunks is now List[TextSpanResult] where each phrasal chunk has:
# - type: Literal["text_span"]
# - text: str (the chunk content containing multiple phrases)
# - start_char: int (starting character position)
# - end_char: int (ending character position)
```

### Process Text from URL

```python
from typing import TextIO

# Save to JSONL file
with open("alice_phrases.jsonl", "w") as f:
    f: TextIO
    for response in client.stream_process_text(
        text_url="https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        chunk_size=2000,
        text_only=True  # boolean flag for TEXT_ONLY
    ):
        if isinstance(response, TextSpanResult):
            f.write(response.model_dump_json() + "\n")
```

## Citation Finding

### Basic Citation Finding

```python
from typing import List
from bookwyrm.models import TextSpan, CitationResponse, Citation

# Prepare text chunks (you can get these from phrasal analysis above)
chunks: List[TextSpan] = [
    TextSpan(
        text="Climate change refers to long-term shifts in global temperatures and weather patterns.",
        start_char=0,
        end_char=89
    ),
    TextSpan(
        text="The primary cause is human activities, particularly fossil fuel burning.",
        start_char=90,
        end_char=161
    ),
    TextSpan(
        text="This releases greenhouse gases like CO2 into the atmosphere.",
        start_char=162,
        end_char=222
    )
]

# Find citations using streaming (the only available method)
citations: List[Citation] = []
for stream_response in client.stream_citations(
    chunks=chunks,
    question="What causes climate change?"
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)
    elif hasattr(stream_response, 'total_citations'):
        print(f"Found {stream_response.total_citations} citations total")

print(f"Found {len(citations)} citations:")
citation: Citation
for citation in citations:
    print(f"- Quality {citation.quality}/4: {citation.text}")

# citations is List[Citation] where each Citation has:
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
    for update in client.stream_citations(
        chunks=chunks,
        question="What causes climate change?"
    ):
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
citations: List[Citation] = []
for stream_response in client.stream_citations(
    jsonl_url="https://example.com/chunks.jsonl",
    question="What is machine learning?",
    start=0,
    limit=100
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)
```

## PDF Extraction

### Extract Text from PDF

```python
from typing import BinaryIO
from bookwyrm.models import PDFPage, PDFTextElement

# Load PDF file using raw bytes (recommended)
with open("document.pdf", "rb") as f:
    f: BinaryIO
    pdf_bytes: bytes = f.read()

pages: List[PDFPage] = []
for response in client.stream_extract_pdf(
    pdf_bytes=pdf_bytes,
    filename="document.pdf"
):
    if hasattr(response, 'page_data'):
        pages.append(response.page_data)
    elif hasattr(response, 'total_pages') and hasattr(response, 'type') and response.type == "metadata":
        print(f"Starting extraction of {response.total_pages} pages")

print(f"Extracted {len(pages)} pages")
page: PDFPage
for page in pages:
    print(f"Page {page.page_number}: {len(page.text_blocks)} text elements")
    element: PDFTextElement
    for element in page.text_blocks[:3]:  # Show first 3 elements
        print(f"  - {element.text[:50]}...")

# pages is List[PDFPage] where each PDFPage has:
# - page_number: int (1-based)
# - text_blocks: List[PDFTextElement]
# - tables: List[dict] (placeholder)
# - images: List[dict] (placeholder)
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
    PDFStreamMetadata, 
    PDFStreamPageResponse, 
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    PDFPage
)

pages: List[PDFPage] = []
with Progress(BarColumn(), TaskProgressColumn()) as progress:
    task: TaskID = progress.add_task("Extracting PDF...", total=100)
    
    for response in client.stream_extract_pdf(
        pdf_url="https://example.com/document.pdf",
        start_page=1,
        num_pages=10
    ):
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
from bookwyrm.models import ClassifyResponse

# Read file as binary
with open("unknown_file.dat", "rb") as f:
    f: BinaryIO
    file_bytes: bytes = f.read()

# Classify using raw bytes (recommended)
response: ClassifyResponse = client.classify(
    content_bytes=file_bytes,
    filename="unknown_file.dat"
)

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

## Text Summarization

### Basic Text Summarization

```python
from typing import List
from bookwyrm.models import SummaryResponse, TextSpan

# Summarize from plain text
text: str = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language.
In particular, how to program computers to process and analyze large amounts of natural language data.
The goal is a computer capable of understanding the contents of documents, including 
the contextual nuances of the language within them. The technology can then accurately 
extract information and insights contained in the documents as well as categorize and 
organize the documents themselves.
"""

# Summarize using streaming (the only available method)
final_summary: SummaryResponse = None
for response in client.stream_summarize(
    content=text,
    max_tokens=5000
):
    if hasattr(response, 'summary'):
        final_summary = response
        break
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

if final_summary:
    print("Summary:")
    print(final_summary.summary)
    print(f"\nProcessed {final_summary.total_tokens} tokens across {final_summary.levels_used} levels")

# final_summary is SummaryResponse with:
# - type: Literal["summary"]
# - summary: str (the final summary text)
# - subsummary_count: int (number of intermediate summaries)
# - levels_used: int (hierarchical levels used)
# - total_tokens: int (total tokens processed)
# - intermediate_summaries: Optional[List[List[str]]] (debug info if requested)
```

### Summarize from URL

```python
final_summary: SummaryResponse = None
for response in client.stream_summarize(
    url="https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    max_tokens=10000,
    debug=True  # Include intermediate summaries
):
    if hasattr(response, 'summary'):
        final_summary = response
        break
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

if final_summary:
    print("Final Summary:")
    print(final_summary.summary)
    
    if final_summary.intermediate_summaries:
        print(f"\nDebug: {len(final_summary.intermediate_summaries)} levels of summaries")
        for level, summaries in enumerate(final_summary.intermediate_summaries):
            print(f"Level {level + 1}: {len(summaries)} summaries")
```

### Summarize from Phrases

```python
# Use phrases from previous phrasal analysis
phrases: List[TextSpan] = [
    TextSpan(text="Machine learning is a subset of AI.", start_char=0, end_char=38),
    TextSpan(text="It uses algorithms to learn from data.", start_char=39, end_char=77),
    TextSpan(text="Deep learning uses neural networks.", start_char=78, end_char=113),
    # ... more phrases
]

final_summary: SummaryResponse = None
for response in client.stream_summarize(
    phrases=phrases,
    max_tokens=2000
):
    if hasattr(response, 'summary'):
        final_summary = response
        break
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

if final_summary:
    print(final_summary.summary)
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
        # Async citation finding (streaming only)
        citations: List[Citation] = []
        async for stream_response in client.stream_citations(
            chunks=chunks,
            question="What causes climate change?"
        ):
            if hasattr(stream_response, 'citation'):
                citations.append(stream_response.citation)
            elif hasattr(stream_response, 'total_citations'):
                print(f"Found {stream_response.total_citations} citations")
        
        print(f"Found {len(citations)} citations")
        
        # Async streaming
        async for update in client.stream_citations(
            chunks=chunks,
            question="What causes climate change?"
        ):
            if isinstance(update, CitationStreamResponse):
                print(f"Citation: {update.citation.text}")

# Run async code
asyncio.run(main())

# All async methods return the same types as their sync counterparts:
# - stream_citations() -> AsyncIterator[StreamingCitationResponse]
# - stream_summarize() -> AsyncIterator[StreamingSummarizeResponse]
# - stream_process_text() -> AsyncIterator[StreamingPhrasalResponse]
# - classify() -> ClassifyResponse
# - extract_pdf() -> PDFExtractResponse
# - stream_extract_pdf() -> AsyncIterator[StreamingPDFResponse]
```

## Error Handling

### Comprehensive Error Handling

```python
from typing import Optional
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError
from bookwyrm.models import CitationResponse

try:
    citations: List[Citation] = []
    for stream_response in client.stream_citations(
        chunks=chunks,
        question="What causes climate change?"
    ):
        if hasattr(stream_response, 'citation'):
            citations.append(stream_response.citation)
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
