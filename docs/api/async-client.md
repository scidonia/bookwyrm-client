# AsyncBookWyrmClient

The asynchronous client for the BookWyrm API, providing full async/await support.

## Class: AsyncBookWyrmClient

```python
class AsyncBookWyrmClient:
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
from bookwyrm import AsyncBookWyrmClient
import asyncio

async def main():
    # Using environment variable for API key
    client = AsyncBookWyrmClient()
    
    # Explicit API key
    client = AsyncBookWyrmClient(api_key="your-api-key")
    
    # Custom base URL
    client = AsyncBookWyrmClient(
        base_url="https://custom-api.example.com",
        api_key="your-api-key"
    )

asyncio.run(main())
```

## Methods

All methods are async and must be awaited.

### get_citations

Find citations for a question from text chunks.

```python
async def get_citations(self, request: CitationRequest) -> CitationResponse
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

async def find_citations():
    chunks = [
        TextChunk(text="The sky is blue.", start_char=0, end_char=16),
        TextChunk(text="Water is wet.", start_char=17, end_char=30)
    ]

    request = CitationRequest(
        chunks=chunks,
        question="Why is the sky blue?"
    )

    async with AsyncBookWyrmClient() as client:
        response = await client.get_citations(request)
        print(f"Found {response.total_citations} citations")

asyncio.run(find_citations())
```

### stream_citations

Stream citations as they are found.

```python
async def stream_citations(
    self, request: CitationRequest
) -> AsyncIterator[StreamingCitationResponse]
```

**Parameters:**
- `request` (CitationRequest): Citation request with chunks and question

**Yields:**
- `StreamingCitationResponse`: Progress updates, citations, summary, or errors

**Example:**

```python
async def stream_citations_example():
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_citations(request):
            if hasattr(response, 'message'):  # Progress update
                print(f"Progress: {response.message}")
            elif hasattr(response, 'citation'):  # Citation found
                print(f"Citation: {response.citation.text}")
            elif hasattr(response, 'total_citations'):  # Summary
                print(f"Complete: {response.total_citations} citations found")

asyncio.run(stream_citations_example())
```

### summarize

Get a summary of the provided content.

```python
async def summarize(self, request: SummarizeRequest) -> SummaryResponse
```

**Parameters:**
- `request` (SummarizeRequest): Summarization request with content and options

**Returns:**
- `SummaryResponse`: Summary response with the generated summary

**Example:**

```python
from bookwyrm.models import SummarizeRequest

async def summarize_text():
    request = SummarizeRequest(
        content=jsonl_content,
        max_tokens=10000,
        debug=True
    )

    async with AsyncBookWyrmClient() as client:
        response = await client.summarize(request)
        print(response.summary)

asyncio.run(summarize_text())
```

### stream_summarize

Stream summarization progress and results.

```python
async def stream_summarize(
    self, request: SummarizeRequest
) -> AsyncIterator[StreamingSummarizeResponse]
```

**Parameters:**
- `request` (SummarizeRequest): Summarization request with content and options

**Yields:**
- `StreamingSummarizeResponse`: Progress updates, summary, or errors

**Example:**

```python
async def stream_summarize_example():
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_summarize(request):
            if hasattr(response, 'message'):  # Progress update
                print(f"Progress: {response.message}")
            elif hasattr(response, 'summary'):  # Final summary
                print(f"Summary: {response.summary}")

asyncio.run(stream_summarize_example())
```

### process_text

Process text using phrasal analysis.

```python
async def process_text(
    self, request: ProcessTextRequest
) -> AsyncIterator[StreamingPhrasalResponse]
```

**Parameters:**
- `request` (ProcessTextRequest): Text processing request

**Yields:**
- `StreamingPhrasalResponse`: Progress updates and phrase results

**Example:**

```python
from bookwyrm.models import ProcessTextRequest, ResponseFormat

async def process_text_example():
    request = ProcessTextRequest(
        text="Your text here",
        response_format=ResponseFormat.WITH_OFFSETS
    )

    phrases = []
    async with AsyncBookWyrmClient() as client:
        async for response in client.process_text(request):
            if hasattr(response, 'text'):  # Phrase result
                phrases.append(response)

asyncio.run(process_text_example())
```

### classify

Classify file content to determine file type and format.

```python
async def classify(self, request: ClassifyRequest) -> ClassifyResponse
```

**Parameters:**
- `request` (ClassifyRequest): Classification request with base64-encoded content

**Returns:**
- `ClassifyResponse`: Classification response with detected file type

**Example:**

```python
import base64
from bookwyrm.models import ClassifyRequest

async def classify_file():
    # Read file as binary and encode
    with open("document.pdf", "rb") as f:
        content = base64.b64encode(f.read()).decode('ascii')

    request = ClassifyRequest(
        content=content,
        filename="document.pdf"
    )

    async with AsyncBookWyrmClient() as client:
        response = await client.classify(request)
        print(f"File type: {response.classification.format_type}")

asyncio.run(classify_file())
```

### extract_pdf

Extract structured data from a PDF file.

```python
async def extract_pdf(self, request: PDFExtractRequest) -> PDFExtractResponse
```

**Parameters:**
- `request` (PDFExtractRequest): PDF extraction request

**Returns:**
- `PDFExtractResponse`: PDF extraction response with structured data

**Example:**

```python
from bookwyrm.models import PDFExtractRequest

async def extract_pdf_example():
    request = PDFExtractRequest(
        pdf_url="https://example.com/document.pdf",
        start_page=1,
        num_pages=5
    )

    async with AsyncBookWyrmClient() as client:
        response = await client.extract_pdf(request)
        print(f"Extracted {response.total_pages} pages")

asyncio.run(extract_pdf_example())
```

### stream_extract_pdf

Stream PDF extraction with progress updates.

```python
async def stream_extract_pdf(
    self, request: PDFExtractRequest
) -> AsyncIterator[StreamingPDFResponse]
```

**Parameters:**
- `request` (PDFExtractRequest): PDF extraction request

**Yields:**
- `StreamingPDFResponse`: Metadata, pages, completion, or errors

**Example:**

```python
async def stream_extract_pdf_example():
    pages = []
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_extract_pdf(request):
            if hasattr(response, 'page_data'):  # Page extracted
                pages.append(response.page_data)
            elif hasattr(response, 'total_pages'):  # Metadata
                print(f"Processing {response.total_pages} pages")

asyncio.run(stream_extract_pdf_example())
```

## Async Context Manager Support

The async client supports async context manager usage for automatic cleanup:

```python
async def example():
    async with AsyncBookWyrmClient() as client:
        response = await client.get_citations(request)
        # Client is automatically closed when exiting the context
```

## Error Handling

The async client raises the same exceptions as the synchronous client:

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

async def example_with_error_handling():
    try:
        async with AsyncBookWyrmClient() as client:
            response = await client.get_citations(request)
    except BookWyrmAPIError as e:
        print(f"API Error: {e}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
    except BookWyrmClientError as e:
        print(f"Client Error: {e}")
```

## Session Management

The async client uses `httpx.AsyncClient` internally. You can close the client manually:

```python
await client.close()
```

Or use the async context manager for automatic cleanup:

```python
async with AsyncBookWyrmClient() as client:
    # Use client
    pass
# Client is automatically closed
```

## Concurrent Operations

You can run multiple async operations concurrently:

```python
import asyncio

async def concurrent_operations():
    async with AsyncBookWyrmClient() as client:
        # Run multiple operations concurrently
        tasks = [
            client.get_citations(request1),
            client.get_citations(request2),
            client.summarize(summarize_request)
        ]
        
        results = await asyncio.gather(*tasks)
        return results

results = asyncio.run(concurrent_operations())
```

## Streaming with asyncio

Handle multiple streams concurrently:

```python
async def handle_multiple_streams():
    async with AsyncBookWyrmClient() as client:
        async def handle_citations():
            async for response in client.stream_citations(citation_request):
                print(f"Citation: {response}")
        
        async def handle_summarization():
            async for response in client.stream_summarize(summarize_request):
                print(f"Summary progress: {response}")
        
        # Run both streams concurrently
        await asyncio.gather(
            handle_citations(),
            handle_summarization()
        )

asyncio.run(handle_multiple_streams())
```
