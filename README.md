# bookwyrm-client

A Python client library for interacting with BookWyrm instances, featuring both synchronous and asynchronous clients plus a rich command-line interface.

## Installation

### Using uv (recommended for development)

```bash
# Clone the repository
git clone https://github.com/yourusername/bookwyrm-client.git
cd bookwyrm-client

# Install dependencies and create virtual environment
uv sync

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
# Install from PyPI (when published)
pip install bookwyrm-client

# Or install from source
git clone https://github.com/yourusername/bookwyrm-client.git
cd bookwyrm-client
pip install -e .

# For development
pip install -r requirements-dev.txt
```

## Usage

### Python Library

The BookWyrm client provides both synchronous and asynchronous interfaces for text processing, citation finding, summarization, and phrasal analysis.

#### Synchronous Client

```python
from bookwyrm_client import BookWyrmClient, CitationRequest, TextChunk, ProcessTextRequest, ResponseFormat, ClassifyRequest

# Initialize client
client = BookWyrmClient(base_url="http://localhost:8000", api_key="your-key")

# Citation finding
chunks = [
    TextChunk(text="This is the first chunk.", start_char=0, end_char=25),
    TextChunk(text="This is the second chunk.", start_char=26, end_char=52),
]

request = CitationRequest(
    chunks=chunks,
    question="What are the chunks about?",
    max_tokens_per_chunk=1000
)

# Get citations (non-streaming)
response = client.get_citations(request)
print(f"Found {response.total_citations} citations")
for citation in response.citations:
    print(f"Quality: {citation.quality}/4")
    print(f"Text: {citation.text}")
    print(f"Reasoning: {citation.reasoning}")

# Stream citations (real-time results)
for stream_response in client.stream_citations(request):
    if hasattr(stream_response, 'citation'):
        print(f"New citation: {stream_response.citation.text}")
    elif hasattr(stream_response, 'message'):
        print(f"Progress: {stream_response.message}")

# Phrasal text processing
phrasal_request = ProcessTextRequest(
    text_url="https://www.gutenberg.org/cache/epub/32706/pg32706.txt",  # Triplanetary by E. E. Smith
    chunk_size=1000,
    response_format=ResponseFormat.WITH_OFFSETS
)

for response in client.process_text(phrasal_request):
    if hasattr(response, 'text'):
        print(f"Phrase: {response.text[:100]}...")
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

# File classification
classify_request = ClassifyRequest(
    url="https://www.gutenberg.org/ebooks/18857.epub3.images",
    filename="alice_wonderland.epub"  # Optional hint
)

classification_response = client.classify(classify_request)
print(f"Format: {classification_response.classification.format_type}")
print(f"Content Type: {classification_response.classification.content_type}")
print(f"MIME Type: {classification_response.classification.mime_type}")
print(f"Confidence: {classification_response.classification.confidence:.2%}")
print(f"File Size: {classification_response.file_size:,} bytes")

# Classify local content
with open("document.txt", "r") as f:
    content = f.read()

local_classify_request = ClassifyRequest(
    content=content,
    filename="document.txt"
)

local_response = client.classify(local_classify_request)
print(f"Local file classified as: {local_response.classification.content_type}")

client.close()
```

#### Asynchronous Client

```python
import asyncio
from bookwyrm_client import AsyncBookWyrmClient, CitationRequest, ProcessTextRequest, ResponseFormat, ClassifyRequest

async def main():
    # Initialize async client
    async with AsyncBookWyrmClient(base_url="http://localhost:8000", api_key="your-key") as client:
        
        # Citation finding
        request = CitationRequest(
            jsonl_url="https://example.com/chunks.jsonl",
            question="What is the main topic?",
        )
        
        response = await client.get_citations(request)
        print(f"Found {response.total_citations} citations")
        
        # Stream citations
        async for stream_response in client.stream_citations(request):
            if hasattr(stream_response, 'citation'):
                print(f"New citation: {stream_response.citation.text}")

        # Phrasal text processing
        phrasal_request = ProcessTextRequest(
            text_url="https://www.gutenberg.org/cache/epub/32706/pg32706.txt",  # Triplanetary by E. E. Smith
            chunk_size=500,
            response_format=ResponseFormat.TEXT_ONLY
        )

        async for response in client.process_text(phrasal_request):
            if hasattr(response, 'text'):
                print(f"Phrase: {response.text[:100]}...")
            elif hasattr(response, 'message'):
                print(f"Progress: {response.message}")

        # File classification
        classify_request = ClassifyRequest(
            url="https://www.gutenberg.org/ebooks/18857.epub3.images"
        )
        
        classification = await client.classify(classify_request)
        print(f"Classified as: {classification.classification.content_type}")
        print(f"Confidence: {classification.classification.confidence:.2%}")

asyncio.run(main())
```

### Command Line Interface

The CLI provides a rich, interactive interface for text processing operations:

#### Citation Finding

```bash
# Find citations in a JSONL file
bookwyrm-client cite chunks.jsonl "What is the main theme?"

# Save results to JSON
bookwyrm-client cite chunks.jsonl "What is the main theme?" --output results.json

# Use a URL as source
bookwyrm-client cite-url https://example.com/chunks.jsonl "What is the main theme?"

# Process only a subset of chunks
bookwyrm-client cite chunks.jsonl "What is the main theme?" --start 10 --limit 100

# Use non-streaming mode
bookwyrm-client cite chunks.jsonl "What is the main theme?" --no-stream
```

#### Phrasal Text Processing

```bash
# Process text from a URL (Triplanetary by E. E. Smith from Project Gutenberg)
bookwyrm-client phrasal --url "https://www.gutenberg.org/cache/epub/32706/pg32706.txt" --chunk-size 1000 --output triplanetary_phrases.jsonl

# Process text from a file
bookwyrm-client phrasal --file document.txt --format with_offsets --output phrases.jsonl

# Process text directly
bookwyrm-client phrasal "This is some text to analyze for phrases." --format text_only

# Use different SpaCy models
bookwyrm-client phrasal --file document.txt --spacy-model en_core_web_lg
```

#### File Classification

```bash
# Classify a URL resource (EPUB from Project Gutenberg)
bookwyrm-client classify --url "https://www.gutenberg.org/ebooks/18857.epub3.images" --output classification.json

# Classify a local file
bookwyrm-client classify --file document.pdf --output results.json

# Classify text content directly
bookwyrm-client classify "import pandas as pd\ndf = pd.DataFrame()" --filename "script.py"

# Classify with filename hint for better accuracy
bookwyrm-client classify --url "https://example.com/data" --filename "data.json"
```

#### Summarization

```bash
# Summarize a JSONL file of phrases
bookwyrm-client summarize phrases.jsonl --output summary.json

# Include debug information
bookwyrm-client summarize phrases.jsonl --debug --max-tokens 5000
```

#### Global Options

```bash
# Set API key and base URL for all commands
bookwyrm-client --api-key YOUR_KEY --base-url http://localhost:8000 phrasal --url "https://example.com/text.txt"

# Enable verbose output
bookwyrm-client --verbose cite chunks.jsonl "Question?"
```

### Environment Variables

Set these environment variables for convenience:

```bash
export BOOKWYRM_API_KEY="your-api-key"
export BOOKWYRM_BASE_URL="http://localhost:8000"
```

## Development

This project supports both uv and pip for development:

```bash
# With uv
uv sync
uv run pytest
uv run bookwyrm-client --help

# With pip
pip install -r requirements-dev.txt
pytest
bookwyrm-client --help
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bookwyrm_client

# Run async tests specifically
pytest -k "async"
```

## API Reference

### Models

- `TextChunk`: Represents a text chunk with start/end character positions
- `CitationRequest`: Request model for citation processing
- `Citation`: A found citation with quality score and reasoning
- `CitationResponse`: Response containing multiple citations
- `UsageInfo`: Token usage and cost information
- `ClassifyRequest`: Request model for file classification
- `ClassifyResponse`: Response containing classification results
- `FileClassification`: Detailed classification information

### Clients

- `BookWyrmClient`: Synchronous client with `get_citations()`, `stream_citations()`, `classify()`, and other methods
- `AsyncBookWyrmClient`: Asynchronous client with async versions of the same methods

### Exceptions

- `BookWyrmClientError`: Base exception class
- `BookWyrmAPIError`: API-specific errors with status codes

## License

See LICENSE file for details.
