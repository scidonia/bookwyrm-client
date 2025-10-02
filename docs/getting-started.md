# Getting Started

This guide will help you get up and running with the BookWyrm client library.

## Installation

Install the BookWyrm client using pip:

```bash
pip install bookwyrm
```

## Authentication

You'll need an API key to use the BookWyrm API. Get one from [https://api.bookwyrm.ai](https://api.bookwyrm.ai).

Set your API key as an environment variable:

```bash
export BOOKWYRM_API_KEY="your-api-key"
```

Or pass it directly when creating a client:

```python
from bookwyrm import BookWyrmClient

client = BookWyrmClient(api_key="your-api-key")
```

## Basic Usage

### Citation Finding

Find citations that answer a specific question:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import CitationRequest, TextChunk

# Create text chunks
chunks = [
    TextChunk(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
    TextChunk(text="Water boils at 100°C at sea level.", start_char=43, end_char=78)
]

# Create request
request = CitationRequest(
    chunks=chunks,
    question="Why is the sky blue?"
)

# Get citations
client = BookWyrmClient()
response = client.get_citations(request)

for citation in response.citations:
    print(f"Quality: {citation.quality}/4")
    print(f"Text: {citation.text}")
    print(f"Reasoning: {citation.reasoning}")
```

### Text Summarization

Summarize text content:

```python
from bookwyrm.models import SummarizeRequest

# Load your JSONL content
with open("phrases.jsonl", "r") as f:
    content = f.read()

request = SummarizeRequest(
    content=content,
    max_tokens=10000
)

response = client.summarize(request)
print(response.summary)
```

### Streaming Operations

Use streaming for real-time progress updates:

```python
# Stream citations
for update in client.stream_citations(request):
    if hasattr(update, 'message'):
        print(f"Progress: {update.message}")
    elif hasattr(update, 'citation'):
        print(f"Found citation: {update.citation.text}")
```

## Command Line Interface

The library includes a CLI for common operations:

```bash
# Find citations
bookwyrm cite "Why is the sky blue?" data/chunks.jsonl

# Summarize text
bookwyrm summarize data/phrases.jsonl --output summary.json

# Process text into phrases
bookwyrm phrasal "Your text here" --output phrases.jsonl

# Extract PDF content
bookwyrm extract-pdf document.pdf --output extracted.json
```

## Error Handling

Handle API errors gracefully:

```python
from bookwyrm.client import BookWyrmAPIError

try:
    response = client.get_citations(request)
except BookWyrmAPIError as e:
    print(f"API Error: {e}")
    if e.status_code:
        print(f"Status Code: {e.status_code}")
```

## Next Steps

- Check out [Examples](examples.md) for more detailed use cases
- Read the [CLI Reference](cli.md) for command-line usage
- Explore the [API Reference](api/) for detailed documentation
