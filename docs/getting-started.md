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

Extract structured content from PDF documents with modern processing options:

```python
# Extract content from the same PDF file
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamPageError

def basic_pdf_extraction():
    """Basic PDF extraction - fast, uses native text when possible."""
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
    return pages, extracted_text

def advanced_pdf_extraction():
    """Advanced PDF extraction with table detection and simple table format."""
    extracted_text = ""
    pages = []
    tables = []
    
    for response in client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="research_paper.pdf",
        enable_layout_detection=True,  # Enable table detection
        force_ocr=False  # Use native text when possible (faster)
    ):
        if isinstance(response, PDFStreamMetadata):
            print(f"Processing {response.total_pages} pages with table detection")
        elif isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
            
            # Process text blocks
            for text_block in response.page_data.text_blocks:
                extracted_text += text_block.text + "\n"
            
            # NEW: Process tables with simple format
            for region in response.page_data.layout_regions:
                if region.content.content_type == "table" and region.content.simple:
                    table_data = {
                        "page": response.document_page,
                        "headers": region.content.simple.rows[0] if region.content.simple.rows else [],
                        "data": region.content.simple.rows[1:] if len(region.content.simple.rows) > 1 else []
                    }
                    tables.append(table_data)
                    print(f"Found table on page {response.document_page}: {len(table_data['data'])} rows")
            
            print(f"Page {response.document_page}: {len(response.page_data.text_blocks)} text elements")
        elif isinstance(response, PDFStreamPageError):
            print(f"Error on page {response.document_page}: {response.error}")
    
    print(f"Extracted {len(extracted_text)} characters and {len(tables)} tables from PDF")
    return pages, extracted_text, tables

# Choose your extraction approach:
pages, extracted_text = basic_pdf_extraction()  # Fast basic extraction
# pages, extracted_text, tables = advanced_pdf_extraction()  # With table detection
```

#### Working with Simple Table Data

If you used the advanced extraction, you can easily work with the extracted tables:

```python
def process_extracted_tables(tables):
    """Process tables extracted with the simple format."""
    for i, table in enumerate(tables):
        print(f"\nTable {i+1} on page {table['page']}:")
        print(f"Headers: {table['headers']}")
        print(f"Data rows: {len(table['data'])}")
        
        # Convert to dictionary records for easy processing
        records = []
        for row in table['data']:
            if len(row) == len(table['headers']):
                record = dict(zip(table['headers'], row))
                records.append(record)
        
        print(f"Sample record: {records[0] if records else 'No complete records'}")
        
        # Example: Save to CSV format
        import csv
        import io
        
        csv_output = io.StringIO()
        writer = csv.writer(csv_output)
        writer.writerow(table['headers'])  # Write headers
        writer.writerows(table['data'])    # Write data rows
        
        csv_content = csv_output.getvalue()
        print(f"CSV format:\n{csv_content[:200]}..." if len(csv_content) > 200 else csv_content)
        
        return records

# If you have tables from advanced extraction:
# table_records = process_extracted_tables(tables)
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
