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
# Synchronous
client = BookWyrmClient(api_key="your-key")
response = client.get_citations(
    chunks=chunks,
    question="Your question here?"
)

# Asynchronous
async with AsyncBookWyrmClient(api_key="your-key") as client:
    response = await client.get_citations(
        chunks=chunks,
        question="Your question here?"
    )
```

### Available methods

Both clients provide the same methods:

- `get_citations()` / `stream_citations()` - Find citations in text
- `summarize()` / `stream_summarize()` - Summarize text content
- `process_text()` - Extract phrases from text
- `classify()` - Classify file content
- `extract_pdf()` / `stream_extract_pdf()` - Extract text from PDFs

### Error handling

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

try:
    response = client.get_citations(request)
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
