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

## Quick Start

```python
from bookwyrm import BookWyrmClient

# Initialize client
client = BookWyrmClient(api_key="your-api-key")

# Find citations
from bookwyrm.models import CitationRequest, TextChunk

chunks = [TextChunk(text="Your text here", start_char=0, end_char=14)]
request = CitationRequest(chunks=chunks, question="What is this about?")
response = client.get_citations(request)

print(f"Found {response.total_citations} citations")
```

## Installation

```bash
pip install bookwyrm
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
