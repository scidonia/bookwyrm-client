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

### Step 1: Process Text into Chunks

First, process your text into meaningful chunks using `stream_process_text`:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import TextResult, TextSpanResult

# Initialize client
client = BookWyrmClient(api_key="your-api-key")

# Process text into chunks
text = """
The sky appears blue due to a phenomenon called Rayleigh scattering. 
When sunlight enters Earth's atmosphere, it collides with gas molecules. 
Blue light waves are shorter than red light waves, so they get scattered 
more in all directions by the tiny gas molecules in the atmosphere.

Water boils at 100°C (212°F) at sea level. This temperature is called 
the boiling point and occurs when the vapor pressure of the liquid equals 
the atmospheric pressure surrounding it.
"""

chunks = []
for response in client.stream_process_text(
    text=text,
    chunk_size=1000,  # Create bounded phrasal chunks
    offsets=True  # Include position information
):
    if isinstance(response, (TextResult, TextSpanResult)):
        chunks.append(response)
        print(f"Chunk: {response.text[:50]}...")

print(f"Created {len(chunks)} chunks")
```

### Step 2: Find Citations

Use the generated chunks to find citations that answer specific questions:

```python
# Use the chunks from Step 1 to find citations
response = client.get_citations(
    chunks=chunks,
    question="Why is the sky blue?"
)

for citation in response.citations:
    print(f"Quality: {citation.quality}/4")
    print(f"Text: {citation.text}")
    print(f"Reasoning: {citation.reasoning}")
```

### Step 3: Text Summarization

Summarize your processed chunks or phrases:

```python
# Use chunks from Step 1 for summarization
response = client.summarize(
    phrases=chunks,  # Use the chunks we created
    max_tokens=10000
)

print(response.summary)

# Or load from JSONL file if you saved chunks
# with open("phrases.jsonl", "r") as f:
#     jsonl_content = f.read()
# 
# response = client.summarize(
#     jsonl_content=jsonl_content,
#     max_tokens=10000
# )
```

### Step 4: Streaming Operations

Use streaming for real-time progress updates on any operation:

```python
# Stream text processing (always streaming)
for response in client.stream_process_text(
    text_url="https://www.gutenberg.org/files/11/11-0.txt",
    chunk_size=2000
):
    if isinstance(response, (TextResult, TextSpanResult)):
        print(f"Processed chunk: {response.text[:50]}...")
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

# Stream citations
for update in client.stream_citations(
    chunks=chunks,
    question="Why is the sky blue?"
):
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

# Process text into phrases/chunks
bookwyrm phrasal "Your text here" --output phrases.jsonl --chunk-size 1000

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
- Explore the [API Reference](api/index.md) for detailed documentation
