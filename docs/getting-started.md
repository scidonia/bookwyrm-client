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

### Step 1: Classify PDF Files

Start by classifying your PDF documents to understand their content type:

```python
from bookwyrm import BookWyrmClient

# Initialize client
client = BookWyrmClient(api_key="your-api-key")

# Classify a PDF file
with open("research_paper.pdf", "rb") as f:
    pdf_bytes = f.read()

response = client.classify(
    content_bytes=pdf_bytes,
    filename="research_paper.pdf"
)

print(f"Format type: {response.classification.format_type}")
print(f"Content type: {response.classification.content_type}")
print(f"Confidence: {response.classification.confidence}")
print(f"MIME type: {response.classification.mime_type}")
```

### Step 2: Extract Content from PDFs

Extract structured content from PDF documents:

```python
# Extract content from the same PDF file
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamPageError

extracted_text = ""
pages = []
for response in client.stream_extract_pdf(
    pdf_bytes=pdf_bytes,
    filename="research_paper.pdf"
):
    if isinstance(response, PDFStreamMetadata):
        print(f"Processing {response.total_pages} pages")
    elif isinstance(response, PDFStreamPageResponse):
        pages.append(response.page_data)
        for text_block in response.page_data.text_blocks:
            extracted_text += text_block.text + "\n"
        print(f"Page {response.document_page}: {len(response.page_data.text_blocks)} elements")
    elif isinstance(response, PDFStreamPageError):
        print(f"Error on page {response.document_page}: {response.error}")

print(f"Extracted {len(extracted_text)} characters from PDF")
```

### Step 3: Process Text into Chunks

Process the extracted text (or any text) into meaningful chunks using `stream_process_text`:

```python
from bookwyrm.models import TextResult, TextSpanResult

# Use the extracted text from Step 1, or provide your own text
text = extracted_text or """
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

### Step 4: Find Citations

Use the generated chunks to find citations that answer specific questions:

```python
# Use the chunks from Step 1 to find citations
citations = []
for stream_response in client.stream_citations(
    chunks=chunks,
    question="Why is the sky blue?"
):
    if hasattr(stream_response, 'citation'):
        citations.append(stream_response.citation)
    elif hasattr(stream_response, 'total_citations'):
        print(f"Found {stream_response.total_citations} total citations")

response = type('obj', (object,), {'citations': citations})()

for citation in response.citations:
    print(f"Quality: {citation.quality}/4")
    print(f"Text: {citation.text}")
    print(f"Reasoning: {citation.reasoning}")
```

### Step 5: Text Summarization

Summarize your processed chunks or phrases:

```python
# Use chunks from Step 1 for summarization
final_summary = None
for response in client.stream_summarize(
    phrases=chunks,  # Use the chunks we created
    max_tokens=10000
):
    if hasattr(response, 'summary'):
        final_summary = response
        break
    elif hasattr(response, 'message'):
        print(f"Progress: {response.message}")

if final_summary:
    print(final_summary.summary)

# Or load from JSONL file if you saved chunks
# with open("phrases.jsonl", "r") as f:
#     jsonl_content = f.read()
# 
# final_summary = None
# for response in client.stream_summarize(
#     jsonl_content=jsonl_content,
#     max_tokens=10000
# ):
#     if hasattr(response, 'summary'):
#         final_summary = response
#         break
```

### Step 6: Streaming Operations

Use streaming for real-time progress updates on any operation:

```python
# Stream PDF extraction
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamPageError

for response in client.stream_extract_pdf(
    pdf_bytes=pdf_bytes,
    filename="document.pdf"
):
    if isinstance(response, PDFStreamMetadata):
        print(f"Processing {response.total_pages} pages")
    elif isinstance(response, PDFStreamPageResponse):
        print(f"Processing page {response.document_page}")
    elif isinstance(response, PDFStreamPageError):
        print(f"Error on page {response.document_page}: {response.error}")

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

# Classify PDF files first
bookwyrm classify document.pdf

# Extract PDF content
bookwyrm extract-pdf document.pdf --output extracted.json

# Process text into phrases/chunks
bookwyrm phrasal "Your text here" --output phrases.jsonl --chunk-size 1000
```

## Error Handling

Handle API errors gracefully:

```python
from bookwyrm.client import BookWyrmAPIError

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
```

## Next Steps

- Check out [Examples](examples.md) for more detailed use cases
- Read the [CLI Reference](cli.md) for command-line usage
- Explore the [API Reference](api/index.md) for detailed documentation
