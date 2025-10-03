# API Reference

The BookWyrm client library provides both synchronous and asynchronous clients for interacting with the BookWyrm API.

## Clients

- **[BookWyrmClient](client.md)** - Synchronous client using `requests`
- **[AsyncBookWyrmClient](async-client.md)** - Asynchronous client using `httpx`

## Models

- **[Models](models.md)** - Pydantic models for requests and responses

## Quick Reference

### Import the clients

```python
from bookwyrm import BookWyrmClient, AsyncBookWyrmClient
```

### Basic usage

```python
from bookwyrm import BookWyrmClient, AsyncBookWyrmClient
from bookwyrm.models import TextSpan

# Create some example chunks
chunks = [
    TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
    TextSpan(text="Water molecules are polar.", start_char=43, end_char=69),
    TextSpan(text="Plants appear green due to chlorophyll.", start_char=70, end_char=109)
]

# Synchronous
client = BookWyrmClient(api_key="your-key")
citations = []
for stream_response in client.stream_citations(
    chunks=chunks,
    question="Why is the sky blue?"
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)

# Asynchronous
async with AsyncBookWyrmClient(api_key="your-key") as client:
    citations = []
    async for stream_response in client.stream_citations(
        chunks=chunks,
        question="Why is the sky blue?"
    ):
        if hasattr(stream_response, 'citation'):
            citations.append(stream_response.citation)
```

### Available methods

Both clients provide the same methods:

- `stream_citations()` - Find citations in text
- `stream_summarize()` - Summarize text content
- `stream_process_text()` - Extract phrases from text
- `classify()` - Classify file content
- `extract_pdf()` / `stream_extract_pdf()` - Extract text from PDFs

### Error handling

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

try:
    citations = []
    for stream_response in client.stream_citations(
        chunks=chunks,
        question="Your question here"
    ):
        if hasattr(stream_response, 'citation'):
            citations.append(stream_response.citation)
except BookWyrmAPIError as e:
    print(f"API Error: {e}")
    if e.status_code:
        print(f"Status Code: {e.status_code}")
except BookWyrmClientError as e:
    print(f"Client Error: {e}")
```

## Environment Variables

Set these for automatic configuration:

```bash
export BOOKWYRM_API_KEY="your-api-key"
export BOOKWYRM_API_URL="https://api.bookwyrm.ai:443"
```
