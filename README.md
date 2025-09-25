# bookwyrm

A Python client library for interacting with BookWyrm instances, featuring both synchronous and asynchronous clients plus a rich command-line interface.

## Installation

### Using uv (recommended for development)

```bash
# Clone the repository
git clone https://github.com/yourusername/bookwyrm.git
cd bookwyrm

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

## Getting an API Key

To use the BookWyrm client, you'll need an API key from bookwyrm.ai:

1. Visit [bookwyrm.ai](https://bookwyrm.ai)
2. Click on "Sign up for beta" to create an account
3. Once registered, you'll receive your API key
4. Set your API key as an environment variable or pass it directly to the client

```bash
export BOOKWYRM_API_KEY="your-api-key-here"
```

## Usage

### Python Library

The BookWyrm client provides both synchronous and asynchronous interfaces for text processing, citation finding, summarization, and phrasal analysis.

#### Synchronous Client

```python
from bookwyrm import BookWyrmClient, CitationRequest, TextChunk, ProcessTextRequest, ResponseFormat, ClassifyRequest, SummarizeRequest

# Initialize client
client = BookWyrmClient(base_url="https://api.bookwyrm.ai:443", api_key="your-key")

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

# Classify binary content (automatically base64 encoded)
with open("image.jpg", "rb") as f:
    binary_content = f.read()
    import base64
    encoded_content = base64.b64encode(binary_content).decode("ascii")

binary_classify_request = ClassifyRequest(
    content=encoded_content,
    content_encoding="base64",
    filename="image.jpg"
)

binary_response = client.classify(binary_classify_request)
print(f"Binary file classified as: {binary_response.classification.content_type}")

client.close()
```

#### Asynchronous Client

```python
import asyncio
from bookwyrm import AsyncBookWyrmClient, CitationRequest, ProcessTextRequest, ResponseFormat, ClassifyRequest, SummarizeRequest

async def main():
    # Initialize async client
    async with AsyncBookWyrmClient(base_url="https://api.bookwyrm.ai:443", api_key="your-key") as client:
        
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
bookwyrm cite "What is the main theme?" chunks.jsonl

# Save results to JSON
bookwyrm cite "What is the main theme?" chunks.jsonl --output results.json

# Use a URL as source
bookwyrm cite "What is the main theme?" --url https://example.com/chunks.jsonl

# Use --file option instead of positional argument
bookwyrm cite "What is the main theme?" --file chunks.jsonl

# Process only a subset of chunks
bookwyrm cite "What is the main theme?" chunks.jsonl --start 10 --limit 100

# Use non-streaming mode
bookwyrm cite "What is the main theme?" chunks.jsonl --no-stream
```

#### Phrasal Text Processing

```bash
# Process text from a URL (Triplanetary by E. E. Smith from Project Gutenberg)
bookwyrm phrasal --url "https://www.gutenberg.org/cache/epub/32706/pg32706.txt" --chunk-size 1000 --output triplanetary_phrases.jsonl

# Process text from a file
bookwyrm phrasal --file document.txt --format with_offsets --output phrases.jsonl

# Process text directly
bookwyrm phrasal "This is some text to analyze for phrases." --format text_only

# Use different SpaCy models
bookwyrm phrasal --file document.txt --spacy-model en_core_web_lg
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
bookwyrm cite --verbose "Question?" chunks.jsonl

# Use environment variables (recommended)
export BOOKWYRM_API_URL="https://api.bookwyrm.ai:443"
export BOOKWYRM_API_KEY="your-api-key"
bookwyrm phrasal --url "https://example.com/text.txt"
```

**Note:** API key and base URL options are available on each command individually, not as global app-level options. Using environment variables is the recommended approach for setting these values across all commands.

### Environment Variables

Set these environment variables for convenience:

```bash
export BOOKWYRM_API_KEY="your-api-key"
export BOOKWYRM_API_URL="https://api.bookwyrm.ai:443"
```

## Development

This project supports both uv and pip for development:

```bash
# With uv
uv sync
uv run pytest
uv run bookwyrm --help

# With pip
pip install -r requirements-dev.txt
pytest
bookwyrm --help
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bookwyrm

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
