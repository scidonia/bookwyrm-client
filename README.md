# bookwyrm

A Python client library and CLI designed to accelerate the development of RAG (Retrieval Augmented Generation) systems and AI agents. BookWyrm provides powerful text processing capabilities through a simple API, making it easy to build sophisticated document analysis and citation systems.

## Documentation

📖 **[Full Documentation](https://bookwyrm-client.readthedocs.io)**

## Key Capabilities

BookWyrm simplifies RAG and agent development by providing these core endpoints:

- **Citation Finding** - Automatically find and extract relevant citations from text chunks based on questions or queries
- **Text Processing** - Break down large documents into meaningful phrases and chunks with configurable sizing
- **Document Classification** - Intelligently classify files and content by format, type, and structure
- **PDF Structure Extraction** - Extract structured text data from PDF files using OCR with bounding box coordinates
- **Summarization** - Generate concise summaries from collections of text phrases or documents
- **Streaming Support** - Real-time processing with progress updates for all major operations

These capabilities work together to provide a complete pipeline for document ingestion, processing, and retrieval - the foundation of any RAG system.

## Installation

```bash
pip install bookwyrm
```

## Getting an API Key

To use the BookWyrm client, you'll need an API key from bookwyrm.ai:

1. Visit [bookwyrm.ai](https://bookwyrm.ai)
1. Click on "Sign up for beta" to create an account
1. Once registered, you can create an API key in the dashboard.
1. Set your API key as an environment variable or pass it directly to the client

```bash
export BOOKWYRM_API_KEY="your-api-key-here"
```

## Development Installation

### Using uv (recommended for development)

```bash
# Clone the repository
git clone https://github.com/scidonia/bookwyrm-client.git
cd bookwyrm-client

# Install dependencies and create virtual environment
uv sync

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
# Install from PyPI (when published)
pip install bookwyrm
```

## Usage

### Python Library

The BookWyrm client provides both synchronous and asynchronous interfaces for text processing, citation finding, summarization, and phrasal analysis.

#### Synchronous Client

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import TextSpan

# Initialize client
client = BookWyrmClient(base_url="https://api.bookwyrm.ai:443", api_key="your-key")

# Citation finding using function interface
chunks = [
    TextSpan(text="This is the first chunk.", start_char=0, end_char=25),
    TextSpan(text="This is the second chunk.", start_char=26, end_char=52),
]

# Stream citations (real-time results) - function interface
citations = []
for stream_response in client.stream_citations(
    chunks=chunks,
    question="What are the chunks about?"
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)
        print(f"New citation: {stream_response.citation.text}")
    elif hasattr(stream_response, 'message'):
        print(f"Progress: {stream_response.message}")
    elif hasattr(stream_response, 'total_citations'):
        print(f"Found {stream_response.total_citations} citations total")

for citation in citations:
    print(f"Quality: {citation.quality}/4")
    print(f"Text: {citation.text}")
    print(f"Reasoning: {citation.reasoning}")

# Phrasal text processing with boolean flags
for response in client.stream_process_text(
    text_url="https://www.gutenberg.org/cache/epub/32706/pg32706.txt",  # Triplanetary by E. E. Smith
    chunk_size=1000,
    offsets=True  # Boolean flag for WITH_OFFSETS
):
    if hasattr(response, 'text'):
        print(f"Phrase: {response.text[:100]}...")
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

# File classification using function interface
classification_response = client.classify(
    content_bytes=open("alice_wonderland.epub", "rb").read(),
    filename="alice_wonderland.epub"  # Optional hint
)
print(f"Format: {classification_response.classification.format_type}")
print(f"Content Type: {classification_response.classification.content_type}")
print(f"MIME Type: {classification_response.classification.mime_type}")
print(f"Confidence: {classification_response.classification.confidence:.2%}")
print(f"File Size: {classification_response.file_size:,} bytes")

# Classify local text content
with open("document.txt", "r") as f:
    content = f.read()

local_response = client.classify(
    content=content,
    filename="document.txt"
)
print(f"Local file classified as: {local_response.classification.content_type}")

# Classify binary content using raw bytes
with open("image.jpg", "rb") as f:
    binary_content = f.read()

binary_response = client.classify(
    content_bytes=binary_content,
    filename="image.jpg"
)
print(f"Binary file classified as: {binary_response.classification.content_type}")

# Streaming PDF extraction with progress
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamPageError

pages = []
for stream_response in client.stream_extract_pdf(
    pdf_url="https://example.com/document.pdf",
    start_page=1,
    num_pages=5
):
    if isinstance(stream_response, PDFStreamMetadata):
        print(f"Starting extraction of {stream_response.total_pages} pages")
    elif isinstance(stream_response, PDFStreamPageResponse):
        pages.append(stream_response.page_data)
        print(f"Processed page {stream_response.document_page}: {len(stream_response.page_data.text_blocks)} elements")
    elif isinstance(stream_response, PDFStreamPageError):
        print(f"Error on page {stream_response.document_page}: {stream_response.error}")

print(f"Extracted {len(pages)} pages")
print(f"Found {sum(len(page.text_blocks) for page in pages)} text elements")

# Extract from local PDF file using raw bytes
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

local_pages = []
for stream_response in client.stream_extract_pdf(
    pdf_bytes=pdf_bytes,
    filename="document.pdf",
    start_page=10,
    num_pages=5
):
    if isinstance(stream_response, PDFStreamPageResponse):
        local_pages.append(stream_response.page_data)

print(f"Extracted pages 10-14: {len(local_pages)} pages processed")

# Streaming summarization
final_summary = None
for response in client.stream_summarize(
    content="Long text content to summarize...",
    max_tokens=5000,
    debug=True
):
    if hasattr(response, 'summary'):
        final_summary = response
        break
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

if final_summary:
    print(f"Summary: {final_summary.summary}")
    print(f"Used {final_summary.levels_used} levels")

client.close()
```

#### Asynchronous Client

```python
import asyncio
from bookwyrm import AsyncBookWyrmClient, CitationRequest, ProcessTextRequest, ResponseFormat, ClassifyRequest, SummarizeRequest

async def main():
    # Initialize async client
    async with AsyncBookWyrmClient(base_url="https://api.bookwyrm.ai:443", api_key="your-key") as client:
        
        # Stream citations
        citations = []
        async for stream_response in client.stream_citations(
            jsonl_url="https://example.com/chunks.jsonl",
            question="What is the main topic?"
        ):
            if hasattr(stream_response, 'citation'):
                citations.append(stream_response.citation)
                print(f"New citation: {stream_response.citation.text}")
            elif hasattr(stream_response, 'total_citations'):
                print(f"Found {stream_response.total_citations} citations")

        # Phrasal text processing with boolean flags
        async for response in client.stream_process_text(
            text_url="https://www.gutenberg.org/cache/epub/32706/pg32706.txt",  # Triplanetary by E. E. Smith
            chunk_size=500,
            text_only=True  # Boolean flag for TEXT_ONLY
        ):
            if hasattr(response, 'text'):
                print(f"Phrase: {response.text[:100]}...")
            elif hasattr(response, 'message'):
                print(f"Progress: {response.message}")

        # File classification using function interface
        classification = await client.classify(
            content_bytes=open("alice_wonderland.epub", "rb").read()
        )
        print(f"Classified as: {classification.classification.content_type}")
        print(f"Confidence: {classification.classification.confidence:.2%}")

        # Streaming PDF extraction
        from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamPageError
        
        pages = []
        async for stream_response in client.stream_extract_pdf(
            pdf_url="https://example.com/document.pdf",
            start_page=1,
            num_pages=10
        ):
            if isinstance(stream_response, PDFStreamMetadata):
                print(f"Processing {stream_response.total_pages} pages...")
            elif isinstance(stream_response, PDFStreamPageResponse):
                pages.append(stream_response.page_data)
                print(f"Page {stream_response.document_page}: {len(stream_response.page_data.text_blocks)} elements")
            elif isinstance(stream_response, PDFStreamPageError):
                print(f"Error on page {stream_response.document_page}: {stream_response.error}")

asyncio.run(main())
```

### Command Line Interface

The CLI provides a rich, interactive interface for text processing operations:

#### Citation Finding

```bash
# Single question with JSONL file
bookwyrm cite --question "What is the main theme?" chunks.jsonl

# Multiple questions using flags
bookwyrm cite --question "What is AI?" --question "How does ML work?" chunks.jsonl

# Questions from file (one per line)
bookwyrm cite --questions-file questions.txt chunks.jsonl

# Save results to JSON
bookwyrm cite --question "What is the main theme?" chunks.jsonl --output results.json

# Use a URL as source
bookwyrm cite --question "What is the main theme?" --url https://example.com/chunks.jsonl

# Use --file option instead of positional argument
bookwyrm cite --question "What is the main theme?" --file chunks.jsonl

# Process only a subset of chunks
bookwyrm cite --question "What is the main theme?" chunks.jsonl --start 10 --limit 100

# Show full citation text without truncation
bookwyrm cite --question "What is the main theme?" chunks.jsonl --long

# Verbose output with detailed citation information
bookwyrm cite --question "What is the main theme?" chunks.jsonl --verbose --long
```

#### Phrasal Text Processing

```bash
# Process text from a URL (Triplanetary by E. E. Smith from Project Gutenberg)
bookwyrm phrasal --url "https://www.gutenberg.org/cache/epub/32706/pg32706.txt" --chunk-size 1000 --offsets --output triplanetary_phrases.jsonl

# Process text from a file using boolean flags
bookwyrm phrasal --file document.txt --offsets --output phrases.jsonl

# Process text directly with text-only output
bookwyrm phrasal "This is some text to analyze for phrases." --text-only

# Traditional format option still works
bookwyrm phrasal --file document.txt --format with_offsets --output phrases.jsonl

# Use different SpaCy models
bookwyrm phrasal --file document.txt --spacy-model en_core_web_lg --offsets
```

#### File Classification

```bash
# Classify a URL resource (EPUB from Project Gutenberg)
bookwyrm classify --url "https://www.gutenberg.org/ebooks/18857.epub3.images" --output classification.json

# Classify a local file
bookwyrm classify --file document.pdf --output results.json

# Classify text content directly
bookwyrm classify "import pandas as pd\ndf = pd.DataFrame()" --filename "script.py"

# Classify with filename hint for better accuracy
bookwyrm classify --url "https://example.com/data" --filename "data.json"

# Note: Binary files are automatically detected and base64-encoded when using --file option
```

#### PDF Structure Extraction

```bash
# Extract structured data from a local PDF file (with streaming progress)
bookwyrm extract-pdf document.pdf --output extracted_data.json

# Extract from a PDF URL with streaming progress
bookwyrm extract-pdf --url "https://example.com/document.pdf" --output results.json

# Use --file option instead of positional argument
bookwyrm extract-pdf --file document.pdf --output data.json

# Extract specific page ranges
bookwyrm extract-pdf document.pdf --start-page 5 --num-pages 10 --output pages_5_to_14.json

# Extract from page 10 to end of document
bookwyrm extract-pdf document.pdf --start-page 10 --output from_page_10.json

# Use non-streaming mode (no progress bar)
bookwyrm extract-pdf document.pdf --no-stream --output results.json

# Show detailed extraction results with verbose output
bookwyrm extract-pdf document.pdf --verbose --output detailed_results.json

# Use custom PDF extraction API endpoint
bookwyrm extract-pdf document.pdf --base-url "http://localhost:8000" --output results.json

# Auto-save with generated filename (no --output needed)
bookwyrm extract-pdf my_document.pdf --start-page 5 --num-pages 3
# Saves to: my_document_pages_5-7_extracted.json
```

#### Summarization

```bash
# Summarize a JSONL file of phrases
bookwyrm summarize phrases.jsonl --output summary.json

# Include debug information
bookwyrm summarize phrases.jsonl --debug --max-tokens 5000
```

#### Global Options

All commands support these options:

```bash
# Set API key and base URL for individual commands
bookwyrm phrasal --api-key YOUR_KEY --base-url https://api.bookwyrm.ai:443 --url "https://example.com/text.txt"

# Enable verbose output (per command)
bookwyrm cite --verbose --question "Question?" chunks.jsonl

# Use environment variables (recommended)
export BOOKWYRM_API_URL="https://api.bookwyrm.ai:443"
export BOOKWYRM_API_KEY="your-api-key"
export BOOKWYRM_PDF_API_URL="https://pdf-api.bookwyrm.ai:443"  # Optional: separate PDF API endpoint
bookwyrm phrasal --url "https://example.com/text.txt"
```

**Note:** API key and base URL options are available on each command individually, not as global app-level options. Using environment variables is the recommended approach for setting these values across all commands.

### Environment Variables

Set these environment variables for convenience:

```bash
export BOOKWYRM_API_KEY="your-api-key"
export BOOKWYRM_API_URL="https://api.bookwyrm.ai:443"
export BOOKWYRM_PDF_API_URL="https://pdf-api.bookwyrm.ai:443"  # Optional: separate PDF API endpoint
```

## Development

This project supports both uv and pip for development:

```bash
# With uv
uv sync
uv run pytest integration/
uv run bookwyrm --help

# With pip
pip install -r requirements-integration.txt
pytest integration/
bookwyrm --help
```

### Running Tests

```bash
# Run all integration tests
pytest integration/

# Run specific test suites
pytest integration/ -k test_cli
pytest integration/ -k test_library

# Run specific features
pytest integration/ -m cite
pytest integration/ -m summarize

# Run with tox (recommended)
tox -e dev-local
tox -e dev-local-cli-cite
```

## API Reference

### Models

- `TextSpan`: Represents a text span with start/end character positions
- `CitationRequest`: Request model for citation processing
- `Citation`: A found citation with quality score and reasoning
- `CitationResponse`: Response containing multiple citations
- `UsageInfo`: Token usage and cost information
- `ClassifyRequest`: Request model for file classification
- `ClassifyResponse`: Response containing classification results
- `FileClassification`: Detailed classification information
- `PDFExtractRequest`: Request model for PDF structure extraction
- `PDFExtractResponse`: Response containing extracted PDF data
- `PDFPage`: Individual page data with text elements
- `PDFTextElement`: Text element with position and confidence
- `StreamingPDFResponse`: Union type for streaming PDF responses

### Clients

- `BookWyrmClient`: Synchronous client with `get_citations()`, `stream_citations()`, `classify()`, `stream_extract_pdf()`, and other methods
- `AsyncBookWyrmClient`: Asynchronous client with async versions of the same methods

### Exceptions

- `BookWyrmClientError`: Base exception class
- `BookWyrmAPIError`: API-specific errors with status codes

## License

See LICENSE file for details.
