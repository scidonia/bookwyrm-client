# BookWyrm Client

A Python client library for the BookWyrm API, providing citation finding, text summarization, phrasal analysis, and PDF extraction capabilities.

## Features

- **Citation Finding**: Find relevant citations for questions in text chunks
- **Text Summarization**: Hierarchical summarization with support for custom Pydantic models
- **Phrasal Analysis**: Extract phrases and chunks from text using NLP
- **PDF Extraction**: Extract structured text data from PDF files
- **File Classification**: Automatically detect file types and formats
- **Streaming Support**: Real-time progress updates for long-running operations
- **Async Support**: Full async/await support with `AsyncBookWyrmClient`

## Installation

```bash
pip install bookwyrm
```

## Quick Start

```python
from bookwyrm import BookWyrmClient

# Initialize client
client = BookWyrmClient(api_key="your-api-key")

# Find citations
from bookwyrm.models import TextSpan

chunks = [TextSpan(text="Your text here", start_char=0, end_char=14)]

citations = []
for stream_response in client.stream_citations(
    chunks=chunks,
    question="What is this about?"
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)
    elif hasattr(stream_response, 'total_citations'):
        print(f"Found {stream_response.total_citations} citations")
```

## Documentation

- [Getting Started](getting-started.md) - Installation and basic usage
- [Examples](examples.md) - Code examples for common use cases
- [CLI Reference](cli.md) - Command-line interface documentation
- [API Reference](api/index.md) - Detailed API documentation
- [Contributing](contributing.md) - How to contribute to the project

## API Key

Get your API key from [https://api.bookwyrm.ai](https://api.bookwyrm.ai) and set it as an environment variable:

```bash
export BOOKWYRM_API_KEY="your-api-key"
```

Or pass it directly to the client:

```python
client = BookWyrmClient(api_key="your-api-key")
```
