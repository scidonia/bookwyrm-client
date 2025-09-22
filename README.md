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

### Command Line Interface

The CLI provides a rich, interactive interface for finding citations:

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

# Set API key and base URL
bookwyrm-client --api-key YOUR_KEY --base-url http://localhost:8000 cite chunks.jsonl "Question?"
```

### Python Library - Synchronous Client

```python
from bookwyrm_client import BookWyrmClient, CitationRequest, TextChunk

# Initialize client
client = BookWyrmClient(base_url="http://localhost:8000", api_key="your-key")

# Prepare text chunks
chunks = [
    TextChunk(text="This is the first chunk.", start_char=0, end_char=25),
    TextChunk(text="This is the second chunk.", start_char=26, end_char=52),
]

# Create request
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

client.close()
```

### Python Library - Asynchronous Client

```python
import asyncio
from bookwyrm_client import AsyncBookWyrmClient, CitationRequest, TextChunk

async def main():
    # Initialize async client
    async with AsyncBookWyrmClient(base_url="http://localhost:8000") as client:
        
        # Prepare request
        request = CitationRequest(
            jsonl_url="https://example.com/chunks.jsonl",
            question="What is the main topic?",
        )
        
        # Get citations
        response = await client.get_citations(request)
        print(f"Found {response.total_citations} citations")
        
        # Stream citations
        async for stream_response in client.stream_citations(request):
            if hasattr(stream_response, 'citation'):
                print(f"New citation: {stream_response.citation.text}")

asyncio.run(main())
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

### Clients

- `BookWyrmClient`: Synchronous client with `get_citations()` and `stream_citations()` methods
- `AsyncBookWyrmClient`: Asynchronous client with async versions of the same methods

### Exceptions

- `BookWyrmClientError`: Base exception class
- `BookWyrmAPIError`: API-specific errors with status codes

## License

See LICENSE file for details.
