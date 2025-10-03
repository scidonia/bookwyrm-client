# Examples

This page contains practical examples of using the BookWyrm client library.

## Phrasal Analysis

### Extract Phrases from Text

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import ProcessTextRequest, ResponseFormat

# Create client
client = BookWyrmClient()

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and human language.
"""

request = ProcessTextRequest(
    text=text,
    response_format=ResponseFormat.WITH_OFFSETS,
    spacy_model="en_core_web_sm"
)

phrases = []
for response in client.process_text(request):
    if hasattr(response, 'text'):  # PhraseResult
        phrases.append(response)
        print(f"Phrase: {response.text}")
        if response.start_char is not None:
            print(f"Position: {response.start_char}-{response.end_char}")
```

### Create Text Chunks

```python
# Create chunks of specific size
request = ProcessTextRequest(
    text=long_text,
    chunk_size=1000,  # ~1000 characters per chunk
    response_format=ResponseFormat.WITH_OFFSETS
)

chunks = []
for response in client.process_text(request):
    if hasattr(response, 'text'):
        chunks.append(response)

print(f"Created {len(chunks)} chunks")
```

### Process Text from URL

```python
request = ProcessTextRequest(
    text_url="https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    chunk_size=2000,
    response_format=ResponseFormat.WITH_OFFSETS
)

# Save to JSONL file
with open("alice_phrases.jsonl", "w") as f:
    for response in client.process_text(request):
        if hasattr(response, 'text'):
            f.write(response.model_dump_json() + "\n")
```

## Citation Finding

### Basic Citation Finding

```python
from bookwyrm.models import CitationRequest, TextChunk

# Prepare text chunks (you can get these from phrasal analysis above)
chunks = [
    TextChunk(
        text="Climate change refers to long-term shifts in global temperatures and weather patterns.",
        start_char=0,
        end_char=89
    ),
    TextChunk(
        text="The primary cause is human activities, particularly fossil fuel burning.",
        start_char=90,
        end_char=161
    ),
    TextChunk(
        text="This releases greenhouse gases like CO2 into the atmosphere.",
        start_char=162,
        end_char=222
    )
]

# Find citations
request = CitationRequest(
    chunks=chunks,
    question="What causes climate change?"
)

response = client.get_citations(request)

print(f"Found {response.total_citations} citations:")
for citation in response.citations:
    print(f"- Quality {citation.quality}/4: {citation.text}")
```

### Streaming Citations with Progress

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

with Progress(SpinnerColumn(), TextColumn("{task.description}")) as progress:
    task = progress.add_task("Finding citations...", total=None)
    
    citations = []
    for update in client.stream_citations(request):
        if hasattr(update, 'message'):
            progress.update(task, description=update.message)
        elif hasattr(update, 'citation'):
            citations.append(update.citation)
            print(f"Found: {update.citation.text[:50]}...")

print(f"Total citations found: {len(citations)}")
```

### Using JSONL Files

```python
# Load from JSONL file
request = CitationRequest(
    jsonl_url="https://example.com/chunks.jsonl",
    question="What is machine learning?",
    start=0,
    limit=100
)

response = client.get_citations(request)
```

## Text Summarization

### Basic Summarization

```python
from bookwyrm.models import SummarizeRequest

# Load JSONL content
with open("book_phrases.jsonl", "r") as f:
    content = f.read()

request = SummarizeRequest(
    content=content,
    max_tokens=5000,
    debug=True  # Include intermediate summaries
)

response = client.summarize(request)

print("Summary:")
print(response.summary)
print(f"\nUsed {response.levels_used} levels")
print(f"Created {response.subsummary_count} subsummaries")
```



## PDF Extraction

### Extract Text from PDF

```python
from bookwyrm.models import PDFExtractRequest
import base64

# Load PDF file
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()
    pdf_content = base64.b64encode(pdf_bytes).decode('ascii')

request = PDFExtractRequest(
    pdf_content=pdf_content,
    filename="document.pdf"
)

response = client.extract_pdf(request)

print(f"Extracted {response.total_pages} pages")
for page in response.pages:
    print(f"Page {page.page_number}: {len(page.text_blocks)} text elements")
    for element in page.text_blocks[:3]:  # Show first 3 elements
        print(f"  - {element.text[:50]}...")
```

### Stream PDF Extraction with Progress

```python
from rich.progress import Progress, BarColumn, TaskProgressColumn

request = PDFExtractRequest(
    pdf_url="https://example.com/document.pdf",
    start_page=1,
    num_pages=10
)

pages = []
with Progress(BarColumn(), TaskProgressColumn()) as progress:
    task = progress.add_task("Extracting PDF...", total=100)
    
    for response in client.stream_extract_pdf(request):
        if hasattr(response, 'total_pages'):  # Metadata
            progress.update(task, total=response.total_pages)
        elif hasattr(response, 'page_data'):  # Page response
            pages.append(response.page_data)
            progress.update(task, completed=response.current_page)

print(f"Extracted {len(pages)} pages")
```

## File Classification

### Classify File Content

```python
from bookwyrm.models import ClassifyRequest
import base64

# Read file as binary
with open("unknown_file.dat", "rb") as f:
    file_bytes = f.read()
    content = base64.b64encode(file_bytes).decode('ascii')

request = ClassifyRequest(
    content=content,
    filename="unknown_file.dat"
)

response = client.classify(request)

print(f"Format: {response.classification.format_type}")
print(f"Content Type: {response.classification.content_type}")
print(f"MIME Type: {response.classification.mime_type}")
print(f"Confidence: {response.classification.confidence:.2%}")

if response.classification.details:
    print("Details:")
    for key, value in response.classification.details.items():
        print(f"  {key}: {value}")
```

## Async Usage

### Using AsyncBookWyrmClient

```python
import asyncio
from bookwyrm import AsyncBookWyrmClient

async def main():
    async with AsyncBookWyrmClient() as client:
        # Async citation finding
        response = await client.get_citations(request)
        print(f"Found {response.total_citations} citations")
        
        # Async streaming
        async for update in client.stream_citations(request):
            if hasattr(update, 'citation'):
                print(f"Citation: {update.citation.text}")

# Run async code
asyncio.run(main())
```

## Error Handling

### Comprehensive Error Handling

```python
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

try:
    response = client.get_citations(request)
except BookWyrmAPIError as e:
    if e.status_code == 401:
        print("Authentication failed - check your API key")
    elif e.status_code == 429:
        print("Rate limit exceeded - please wait")
    elif e.status_code == 500:
        print("Server error - please try again later")
    else:
        print(f"API error: {e}")
except BookWyrmClientError as e:
    print(f"Client error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
