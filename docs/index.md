# BookWyrm Client

A Python client library for the BookWyrm API, providing citation finding, text summarization, phrasal analysis, and PDF extraction capabilities.

## Features

- **Citation Finding**: Find relevant citations for questions in text chunks
- **Text Summarization**: Hierarchical summarization with support for custom Pydantic models
- **Phrasal Analysis**: Extract phrases and chunks from text using NLP
- **PDF Extraction**: Extract structured text data from PDF files with advanced table processing
- **Simple Table Format**: Easy-to-use table data as arrays (no complex parsing needed)
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

# Extract PDF with table detection
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

pages = []
tables = []
for response in client.stream_extract_pdf(
    pdf_bytes=pdf_bytes,
    filename="document.pdf",
    enable_layout_detection=True  # Enables table detection
):
    if hasattr(response, 'page_data'):
        pages.append(response.page_data)
        
        # Process tables with simple format
        for region in response.page_data.layout_regions:
            if region.content.content_type == "table" and region.content.simple:
                table_data = {
                    "headers": region.content.simple.rows[0],
                    "data": region.content.simple.rows[1:]
                }
                tables.append(table_data)
                print(f"Found table: {len(table_data['data'])} rows")

print(f"Extracted {len(pages)} pages and {len(tables)} tables")
```

## Documentation

- [Getting Started](getting-started.md) - Installation and basic usage
- [Client Guide](client-guide.md) - Python client library tutorial with examples
- [CLI Guide](cli-guide.md) - Comprehensive CLI tutorial with examples
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
