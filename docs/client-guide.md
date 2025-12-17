# BookWyrm Client Library Guide

This guide demonstrates how to use the BookWyrm Python client library to perform the same operations shown in the CLI guide. We'll focus on the synchronous `BookWyrmClient` for simplicity.

All examples include proper type annotations following the project conventions.

## Installation and Setup

<!--pytest.mark.skip-->

```bash
pip install bookwyrm
```

Set your API key as an environment variable:

<!--pytest.mark.skip-->

```python
import os
os.environ["BOOKWYRM_API_KEY"] = "your-api-key-here"
```

Or pass it directly to the client:

```python
from bookwyrm import BookWyrmClient

client = BookWyrmClient()
```

## 1. File Classification

Classify documents to understand their content type and structure:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import ClassifyProgressUpdate, ClassifyStreamResponse, ClassifyErrorResponse, ClassifyResponse
from pathlib import Path
from typing import Optional

def classify_pdf() -> ClassifyResponse:
    """Classify a PDF file to understand its content."""
    client = BookWyrmClient()

    # Read PDF file as bytes
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()

    # Non-streaming classification
    response = client.classify(
        content_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf"
    )

    print(f"Format Type: {response.classification.format_type}")
    print(f"Content Type: {response.classification.content_type}")
    print(f"MIME Type: {response.classification.mime_type}")
    print(f"Confidence: {response.classification.confidence:.2%}")
    print(f"File Size: {response.file_size} bytes")
    print(f"Details: {response.classification.details}")

    return response

def classify_pdf_with_progress() -> Optional[ClassifyStreamResponse]:
    """Classify a PDF with real-time progress updates."""
    from bookwyrm.utils import collect_classification_from_stream

    client = BookWyrmClient()

    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()

    stream = client.stream_classify(
        content_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf"
    )

    classification_result = collect_classification_from_stream(stream, verbose=True)

    if classification_result:
        print(f"Format: {classification_result.classification.format_type}")

    return classification_result

# Run classification
result = classify_pdf()
```

## 2. PDF Structure Extraction

Extract structured data from specific pages of a PDF with advanced features:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamComplete, PDFStreamPageError, PDFPage
from pathlib import Path
from typing import List
import json

def extract_pdf_structure_basic() -> List[PDFPage]:
    """Extract structured data from PDF pages 1-4 (basic extraction)."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    # Basic extraction - fast, uses native text when possible
    stream = client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    
    # Save extracted data to JSON file
    output_data = {
        "metadata": metadata.model_dump() if metadata else None,
        "pages": [page.model_dump() for page in pages]
    }
    
    output_path = Path("data/SOA_2025_Final_pages_1-4.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved structured data to {output_path}")
    return pages

def extract_pdf_with_tables() -> List[PDFPage]:
    """Extract PDF with table detection and simple table format."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    # Advanced extraction with table detection
    stream = client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4,
        enable_layout_detection=True  # Enables table detection and simple table format
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    
    # Process extracted tables
    for page in pages:
        for region in page.layout_regions:
            if region.content.content_type == "table":
                table = region.content
                print(f"Found table on page {page.page}")
                
                # NEW: Easy table access with simple field
                if table.simple:
                    headers = table.simple.rows[0] if table.simple.rows else []
                    data_rows = table.simple.rows[1:] if len(table.simple.rows) > 1 else []
                    print(f"  Headers: {headers}")
                    print(f"  Data rows: {len(data_rows)}")
                    
                    # Example: Convert to dictionary format
                    if headers and data_rows:
                        table_records = []
                        for row in data_rows:
                            record = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                            table_records.append(record)
                        print(f"  Sample record: {table_records[0] if table_records else 'None'}")
    
    return pages

def extract_pdf_force_ocr() -> List[PDFPage]:
    """Force OCR processing for better text quality."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    # Force OCR even for native text PDFs (useful for poor quality native text)
    stream = client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4,
        force_ocr=True  # NEW: Force OCR processing
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    return pages

def extract_pdf_from_url() -> List[PDFPage]:
    """Extract PDF structure from a URL with modern parameters."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    stream = client.stream_extract_pdf(
        pdf_url="https://example.com/document.pdf",
        start_page=1,
        num_pages=5,
        enable_layout_detection=True,  # Enable table detection
        force_ocr=False  # Use native text when available
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    return pages

# Extract PDF structure - choose the approach that fits your needs
pages = extract_pdf_structure_basic()  # Fast, basic extraction
# pages = extract_pdf_with_tables()    # Advanced with table detection  
# pages = extract_pdf_force_ocr()      # Force OCR for quality
```

### Working with Simple Table Data

The new Simple Table format makes it easy to work with extracted table data:

```python
from bookwyrm.models import PDFPage, SimpleTable
from typing import List, Dict, Any
import pandas as pd
import csv
import io

def process_simple_tables(pages: List[PDFPage]) -> List[Dict[str, Any]]:
    """Process all tables found in PDF pages using the simple format."""
    all_tables = []
    
    for page in pages:
        for region_idx, region in enumerate(page.layout_regions):
            if region.content.content_type == "table" and region.content.simple:
                table = region.content
                simple_data = table.simple
                
                if not simple_data.rows:
                    continue
                
                # Extract table information
                table_info = {
                    "page": page.page,
                    "region": region_idx,
                    "headers": simple_data.rows[0] if simple_data.rows else [],
                    "data_rows": simple_data.rows[1:] if len(simple_data.rows) > 1 else [],
                    "total_rows": len(simple_data.rows),
                    "total_cols": len(simple_data.rows[0]) if simple_data.rows else 0,
                }
                
                print(f"Table on page {page.page}, region {region_idx}:")
                print(f"  Size: {table_info['total_rows']} rows × {table_info['total_cols']} cols")
                print(f"  Headers: {table_info['headers']}")
                
                all_tables.append(table_info)
    
    return all_tables

def convert_table_to_pandas(simple_table: SimpleTable) -> pd.DataFrame:
    """Convert a SimpleTable to pandas DataFrame."""
    if not simple_table.rows or len(simple_table.rows) < 2:
        return pd.DataFrame()  # Empty DataFrame for tables without data
    
    headers = simple_table.rows[0]
    data_rows = simple_table.rows[1:]
    
    # Create DataFrame with headers as column names
    df = pd.DataFrame(data_rows, columns=headers)
    return df

def convert_table_to_csv(simple_table: SimpleTable) -> str:
    """Convert a SimpleTable to CSV format."""
    if not simple_table.rows:
        return ""
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(simple_table.rows)
    return output.getvalue()

def convert_table_to_records(simple_table: SimpleTable) -> List[Dict[str, str]]:
    """Convert a SimpleTable to list of dictionary records."""
    if not simple_table.rows or len(simple_table.rows) < 2:
        return []
    
    headers = simple_table.rows[0]
    data_rows = simple_table.rows[1:]
    
    records = []
    for row in data_rows:
        # Create record dictionary, handling mismatched row lengths
        record = {}
        for i, header in enumerate(headers):
            record[header] = row[i] if i < len(row) else ""
        records.append(record)
    
    return records

def comprehensive_table_processing() -> None:
    """Complete example of table processing workflow."""
    client = BookWyrmClient()
    
    # Extract PDF with table detection
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    stream = client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4,
        enable_layout_detection=True  # Required for table detection
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    
    # Process all tables found
    table_data = process_simple_tables(pages)
    
    # Work with each table
    for table_info in table_data:
        print(f"\nProcessing table on page {table_info['page']}:")
        
        # Reconstruct SimpleTable object for conversion functions
        simple_table = SimpleTable(rows=[table_info['headers']] + table_info['data_rows'])
        
        # Convert to different formats
        print("1. As pandas DataFrame:")
        df = convert_table_to_pandas(simple_table)
        print(df.head() if not df.empty else "Empty table")
        
        print("2. As CSV:")
        csv_data = convert_table_to_csv(simple_table)
        print(csv_data[:200] + "..." if len(csv_data) > 200 else csv_data)
        
        print("3. As dictionary records:")
        records = convert_table_to_records(simple_table)
        print(f"First record: {records[0] if records else 'No records'}")
        
        # Save table data
        if records:
            table_file = Path(f"data/table_page_{table_info['page']}_region_{table_info['region']}.json")
            with open(table_file, "w") as f:
                json.dump(records, f, indent=2)
            print(f"Saved table to {table_file}")

# Example usage:
# comprehensive_table_processing()
```

## 3. PDF to Text Conversion with Character Mapping

Convert the extracted PDF data to raw text with character position mapping using the built-in utility functions. The preferred approach is to work directly with the page objects in memory:

```python
from bookwyrm.utils import create_pdf_text_mapping_from_pages
from bookwyrm.models import PDFPage, PDFTextMapping
from typing import List

def convert_pdf_to_text(pages: List[PDFPage]) -> PDFTextMapping:
    """Convert PDF page objects to raw text with character mapping (in-memory)."""
    
    # Convert PDF pages to text mapping directly in memory (preferred)
    mapping = create_pdf_text_mapping_from_pages(pages)
    
    print(f"Created raw text with {len(mapping.raw_text)} characters")
    print(f"Created {len(mapping.character_mappings)} character mappings")
    print(f"Processed {mapping.total_pages} pages")
    
    return mapping

# Convert PDF data to text with mapping (using pages from previous example)
# mapping = convert_pdf_to_text(pages)
```

If you need to save the mapping to files:

<!--pytest.mark.skip-->

```python
from pathlib import Path

# Save text and mapping files if needed (mapping from previous example)
# Path("data/SOA_2025_Final_raw.txt").write_text(mapping.raw_text, encoding="utf-8")
# Path("data/SOA_2025_Final_mapping.json").write_text(mapping.model_dump_json(indent=2), encoding="utf-8")
```

For working with JSON data directly (if you have extraction data as a dictionary):

<!--pytest.mark.skip-->

```python
from bookwyrm.utils import pdf_to_text_with_mapping_from_json

# If you have the extraction data as JSON (pages from previous example)
# extraction_data = {"pages": [page.model_dump() for page in pages]}
# mapping = pdf_to_text_with_mapping_from_json(extraction_data)
# print(f"Converted {len(mapping.raw_text)} characters")
```

For loading from saved JSON files (less preferred):

<!--pytest.mark.skip-->

```python
from bookwyrm.utils import pdf_to_text_with_mapping
from pathlib import Path

# Only use this if you need to load from a saved JSON file
mapping = pdf_to_text_with_mapping(
    Path("data/SOA_2025_Final_pages_1-4.json")
)
```

## 4. Querying Character Positions

Query specific character ranges to get their bounding box coordinates using the built-in utilities:

<!--pytest.mark.skip-->

```python
from bookwyrm.utils import query_mapping_range_in_memory
from bookwyrm import BookWyrmClient
from pathlib import Path

def query_character_positions(mapping):
    """Query character positions from mapping (requires mapping from previous examples)."""
    client = BookWyrmClient()

    # Query character positions from in-memory mapping (preferred approach)
    result = query_mapping_range_in_memory(mapping, 974, 1089)

    print(f"Character range 974-1089:")
    print(f"Pages: {result['pages']}")
    print(f"Sample text: {result['sample_text'][:100]}...")
    print(f"Bounding boxes found on {len(result['bounding_boxes'])} pages")

    # Or use the client method directly with in-memory mapping
    result = client.query_character_range(
        mapping=mapping,
        start_char=974,
        end_char=1089
    )

    return result

# Example usage (requires mapping from previous examples):
# result = query_character_positions(mapping)

# Alternative: Query from saved mapping file
def query_from_saved_mapping():
    """Query from a saved mapping file."""
    from bookwyrm.utils import query_character_range

    result = query_character_range(
        Path("data/SOA_2025_Final_mapping.json"),
        start_char=974,
        end_char=1089
    )
    print(f"Found bounding boxes on {len(result['pages'])} pages")
    return result

# Query from saved file
# result = query_from_saved_mapping()
```

## 5. Phrasal Text Processing

Process text files to extract meaningful phrases and text spans:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import TextResult, TextSpanResult, PhraseProgressUpdate, ResponseFormat
from pathlib import Path
from typing import List, Union
import json

def process_text_to_phrases() -> List[Union[TextResult, TextSpanResult]]:
    """Create phrasal analysis of a text file."""
    from bookwyrm.utils import collect_phrases_from_stream

    client = BookWyrmClient()

    # Read text file (available in the repository)
    text_file = Path("data/country-of-the-blind.txt")
    text_content = text_file.read_text(encoding='utf-8')

    # Stream phrasal processing with utility function
    stream = client.stream_process_text(
        text=text_content,
        response_format=ResponseFormat.WITH_OFFSETS
    )

    output_file = Path("data/country-of-the-blind-phrases.jsonl")
    phrases = collect_phrases_from_stream(stream, verbose=True, output_file=output_file)

    print(f"Saved {len(phrases)} phrases to {output_file}")
    return phrases

def process_text_from_url() -> List[Union[TextResult, TextSpanResult]]:
    """Process text from a URL."""
    from bookwyrm.utils import collect_phrases_from_stream

    client = BookWyrmClient()

    stream = client.stream_process_text(
        text_url="https://www.gutenberg.org/files/11/11-0.txt",
        chunk_size=2000,
        response_format=ResponseFormat.WITH_OFFSETS
    )

    # Save to JSONL using utility function
    output_file = Path("data/alice-phrases.jsonl")
    phrases = collect_phrases_from_stream(stream, verbose=False, output_file=output_file)

    return phrases

# Process text to phrases
phrases = process_text_to_phrases()
```

## 6. Text Summarization

Create summaries from phrasal data using both basic and structured approaches:

### Basic Summarization

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import TextSpan, SummaryResponse, SummarizeProgressUpdate
from pathlib import Path
from typing import List, Optional
import json

# Use the enhanced utility function from bookwyrm.utils
from bookwyrm.utils import load_phrases_from_jsonl

def basic_summarization() -> Optional[SummaryResponse]:
    """Generate a basic summary from phrases."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_summary_from_stream, save_model_to_json

    client = BookWyrmClient()

    # Load phrases from JSONL file
    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    # Stream summarization with progress using utility function
    stream = client.stream_summarize(
        phrases=phrases,
        model_strength="smart",
        debug=True
    )

    final_summary = collect_summary_from_stream(stream, verbose=True)

    if final_summary:
        # Save summary to JSON file using utility function
        output_file = Path("data/country-of-the-blind-summary.json")
        save_model_to_json(final_summary, output_file)

        print(f"Summary: {final_summary.summary}")
        print(f"Levels used: {final_summary.levels_used}")
        print(f"Total tokens: {final_summary.total_tokens}")
        print(f"Saved to {output_file}")

    return final_summary

# Generate basic summary
summary = basic_summarization()
```

### Structured Literary Analysis with Pydantic Models

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import SummaryResponse, SummarizeProgressUpdate
from pathlib import Path
from typing import Optional
import json
import sys

# Add the data directory to Python path to import the Summary model
sys.path.append('data')
from summary import Summary

def structured_literary_analysis() -> Optional[SummaryResponse]:
    """Generate structured literary analysis using the Summary model."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_summary_from_stream, save_model_to_json

    client = BookWyrmClient()

    # Load phrases
    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    # Create structured summary using the Summary Pydantic model
    stream = client.stream_summarize(
        phrases=phrases,
        summary_class=Summary,
        model_strength="smart",
        debug=True
    )

    final_result = collect_summary_from_stream(stream, verbose=True)

    if final_result:
        # Save structured summary using utility function
        output_file = Path("data/country-structured-summary.json")
        save_model_to_json(final_result, output_file)

        # Display structured results
        if isinstance(final_result.summary, dict):
            summary_data = final_result.summary
            print(f"Title: {summary_data.get('title')}")
            print(f"Author: {summary_data.get('author')}")
            print(f"Publication Date: {summary_data.get('date_of_publication')}")
            print(f"Plot: {summary_data.get('plot', '')[:200]}...")
            print(f"Important Characters: {summary_data.get('important_characters')}")

        print(f"Saved structured analysis to {output_file}")

    return final_result

def high_quality_analysis() -> Optional[SummaryResponse]:
    """Generate high-quality analysis using the 'wise' model."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_summary_from_stream, save_model_to_json

    client = BookWyrmClient()

    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    stream = client.stream_summarize(
        phrases=phrases,
        summary_class=Summary,
        model_strength="wise",
        debug=True
    )

    final_result = collect_summary_from_stream(stream, verbose=True)

    if final_result:
        output_file = Path("data/country-detailed-analysis.json")
        save_model_to_json(final_result, output_file)
        print(f"High-quality analysis saved to {output_file}")

    return final_result

# Generate structured analysis
structured_result = structured_literary_analysis()
detailed_result = high_quality_analysis()
```

### Custom Prompts for Specialized Analysis

```python
from typing import Optional
from bookwyrm import BookWyrmClient
from bookwyrm.models import SummaryResponse
from pathlib import Path

def custom_prompt_analysis() -> Optional[SummaryResponse]:
    """Use custom prompts for specialized literary analysis."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_summary_from_stream, save_model_to_json

    client = BookWyrmClient()

    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    stream = client.stream_summarize(
        phrases=phrases,
        chunk_prompt="Extract key themes, symbols, and literary devices from this text",
        summary_of_summaries_prompt="Create a comprehensive literary analysis focusing on themes, symbolism, and narrative techniques",
        model_strength="clever",
    )

    final_result = collect_summary_from_stream(stream, verbose=False)

    if final_result:
        output_file = Path("data/country-literary-analysis.json")
        save_model_to_json(final_result, output_file)
        print(f"Literary analysis saved to {output_file}")

    return final_result

# Generate custom analysis
custom_result = custom_prompt_analysis()
```

## 7. Citation Finding

Find specific citations related to questions in the text:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import CitationProgressUpdate, CitationStreamResponse, CitationSummaryResponse, CitationErrorResponse, Citation
from pathlib import Path
from typing import List
import json

def find_citations() -> List[Citation]:
    """Find citations about life-threatening situations."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_citations_from_stream, save_models_list_to_json

    client = BookWyrmClient()

    # Load phrases
    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    # Stream citations with utility function
    stream = client.stream_citations(
        chunks=phrases,
        question="Where does the protagonist experience life threatening situations?"
    )

    citations, usage = collect_citations_from_stream(stream, verbose=True)

    # Save citations to JSON using utility function
    output_file = Path("data/protagonist-dangers.json")
    save_models_list_to_json(citations, output_file)

    print(f"Saved {len(citations)} citations to {output_file}")
    return citations

def find_multiple_citations() -> List[Citation]:
    """Ask multiple questions about the story."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_citations_from_stream

    client = BookWyrmClient()

    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    questions = [
        "What are the main conflicts in the story?",
        "How does the protagonist adapt to his environment?",
        "What role does blindness play in the narrative?"
    ]

    all_citations: List[Citation] = []
    for question in questions:
        print(f"\nSearching for: {question}")

        stream = client.stream_citations(
            chunks=phrases,
            question=question
        )

        question_citations, usage = collect_citations_from_stream(stream, verbose=True)
        all_citations.extend(question_citations)

    return all_citations

def find_citations_with_limits() -> List[Citation]:
    """Find citations with start and limit parameters."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_citations_from_stream

    client = BookWyrmClient()

    phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

    stream = client.stream_citations(
        chunks=phrases,
        question="What are the key themes in the story?",
        start=10,  # Start from chunk 10
        limit=50,  # Process only 50 chunks
    )

    citations, usage = collect_citations_from_stream(stream, verbose=False)
    return citations

# Find citations
citations = find_citations()
multiple_citations = find_multiple_citations()
```

## Complete Workflow Example

Here's a complete workflow that processes a PDF from extraction to citation finding, including the new Simple Table features:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.utils import create_pdf_text_mapping_from_pages, query_mapping_range_in_memory, save_mapping_query_in_memory
from bookwyrm.models import PDFStreamPageResponse, PDFPage, PDFTextMapping, SimpleTable
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

def complete_pdf_workflow_with_tables() -> Tuple[List[PDFPage], PDFTextMapping, List[Dict], List]:
    """Complete workflow from PDF to citations with table extraction."""
    client = BookWyrmClient()

    print("Step 1: Extract PDF structure with table detection")
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()

    pages = []
    for response in client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4,
        enable_layout_detection=True,  # Enable table detection
        force_ocr=False  # Use native text when possible (faster)
    ):
        if isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)

    print("Step 2: Process extracted tables")
    all_tables = []
    for page in pages:
        for region_idx, region in enumerate(page.layout_regions):
            if region.content.content_type == "table" and region.content.simple:
                table_data = {
                    "page": page.page,
                    "region": region_idx,
                    "headers": region.content.simple.rows[0] if region.content.simple.rows else [],
                    "data": region.content.simple.rows[1:] if len(region.content.simple.rows) > 1 else [],
                    "bbox": region.coordinates.model_dump()
                }
                all_tables.append(table_data)
                print(f"Found table on page {page.page}: {len(table_data['data'])} rows")
    
    # Save table data separately for easy access
    if all_tables:
        table_file = Path("data/extracted_tables.json")
        with open(table_file, "w") as f:
            json.dump(all_tables, f, indent=2)
        print(f"Saved {len(all_tables)} tables to {table_file}")

    print("Step 3: Create text mapping directly from pages (in-memory)")
    # Use in-memory utility function - no file I/O needed
    mapping = create_pdf_text_mapping_from_pages(pages)

    print("Step 4: Process text for phrases")
    # Process the extracted text to create phrases
    from bookwyrm.utils import collect_phrases_from_stream
    from bookwyrm.models import ResponseFormat

    stream = client.stream_process_text(
        text=mapping.raw_text,  # Use the raw text from the PDF mapping
        response_format=ResponseFormat.WITH_OFFSETS
    )

    phrases = collect_phrases_from_stream(stream, verbose=True)
    print(f"Created {len(phrases)} phrases from PDF text")

    print("Workflow complete!")
    return pages, mapping, all_tables, phrases

def advanced_table_workflow() -> List[Dict[str, Any]]:
    """Advanced workflow focused on table extraction and processing."""
    client = BookWyrmClient()
    
    print("Advanced Table Processing Workflow")
    print("=" * 40)
    
    # Extract with forced OCR for better table text quality
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    pages = []
    for response in client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4,
        enable_layout_detection=True,  # Required for tables
        force_ocr=True  # Force OCR for better table text quality
    ):
        if isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
    
    # Advanced table processing
    processed_tables = []
    
    for page in pages:
        for region_idx, region in enumerate(page.layout_regions):
            if region.content.content_type == "table":
                table_content = region.content
                
                # Process both simple and legacy formats
                table_result = {
                    "page": page.page,
                    "region": region_idx,
                    "bbox": region.coordinates.model_dump(),
                    "legacy_data": {
                        "rows": table_content.rows,
                        "cols": table_content.cols,
                        "has_header": table_content.has_header,
                        "cell_count": len(table_content.cells)
                    }
                }
                
                # NEW: Simple table processing
                if table_content.simple and table_content.simple.rows:
                    headers = table_content.simple.rows[0]
                    data_rows = table_content.simple.rows[1:]
                    
                    # Convert to structured records
                    records = []
                    for row in data_rows:
                        record = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                        records.append(record)
                    
                    table_result["simple_data"] = {
                        "headers": headers,
                        "records": records,
                        "row_count": len(data_rows),
                        "col_count": len(headers)
                    }
                    
                    print(f"Table page {page.page}, region {region_idx}:")
                    print(f"  Headers: {headers}")
                    print(f"  Records: {len(records)}")
                    print(f"  Sample: {records[0] if records else 'No data'}")
                
                processed_tables.append(table_result)
    
    # Save comprehensive table analysis
    if processed_tables:
        analysis_file = Path("data/comprehensive_table_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(processed_tables, f, indent=2)
        print(f"\nSaved comprehensive analysis to {analysis_file}")
    
    return processed_tables

# Choose your workflow based on needs:
# Basic workflow with table support
results = complete_pdf_workflow_with_tables()

# Advanced table-focused workflow  
# table_results = advanced_table_workflow()
```

## Error Handling and Best Practices

<!--pytest.mark.skip-->

```python
from bookwyrm import BookWyrmClient
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError
from bookwyrm.models import Citation, CitationStreamResponse, CitationErrorResponse
from typing import List

def robust_citation_search() -> List[Citation]:
    """Example with proper error handling."""
    from bookwyrm.utils import load_phrases_from_jsonl, collect_citations_from_stream

    client = BookWyrmClient()

    try:
        phrases = load_phrases_from_jsonl(Path("data/country-of-the-blind-phrases.jsonl"))

        stream = client.stream_citations(
            chunks=phrases,
            question="What are the main themes?"
        )

        citations, usage = collect_citations_from_stream(stream, verbose=True)
        return citations

    except BookWyrmAPIError as e:
        print(f"API Error: {e}")
        if e.status_code:
            print(f"Status Code: {e.status_code}")
    except BookWyrmClientError as e:
        print(f"Client Error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return []

# Use context manager for automatic cleanup
def safe_client_usage() -> ClassifyResponse:
    """Use client with context manager for automatic cleanup."""
    with BookWyrmClient() as client:
        # Client will be automatically closed when exiting the context
        response = client.classify(
            content_bytes=Path("data/document.pdf").read_bytes(),
            filename="document.pdf"
        )
        return response

# Safe usage
result = safe_client_usage()
```

## Expected Output Files

After running these examples, you should have the same files as the CLI guide:

- `data/SOA_2025_Final_pages_1-4.json` - Structured PDF data with bounding boxes
- `data/SOA_2025_Final_pages_1-4_raw.txt` - Raw text extracted from PDF
- `data/SOA_2025_Final_pages_1-4_mapping.json` - Character position to bounding box mapping
- `data/character_positions.json` - Query results for specific character ranges
- `data/country-of-the-blind-phrases.jsonl` - Phrasal analysis
- `data/country-of-the-blind-summary.json` - Basic text summary
- `data/country-structured-summary.json` - Structured literary analysis using Summary model
- `data/country-detailed-analysis.json` - High-quality structured analysis
- `data/protagonist-dangers.json` - Citation results

## Sample Data Files

The examples in this guide use sample data files from the repository:

- [`data/SOA_2025_Final.pdf`](https://github.com/scidonia/bookwyrm-client/blob/main/data/SOA_2025_Final.pdf) - State-of-the-Art spacecraft technology PDF for extraction examples
- [`data/country-of-the-blind.txt`](https://github.com/scidonia/bookwyrm-client/blob/main/data/country-of-the-blind.txt) - H.G. Wells' "The Country of the Blind" text for phrasal analysis and summarization examples
- [`data/summary.py`](https://github.com/scidonia/bookwyrm-client/blob/main/data/summary.py) - Example Pydantic model for structured literary analysis

**Note**: Download these files from the GitHub repository to your local `data/` directory to run the examples. If you don't have these files, you can substitute your own PDF and text files, adjusting the filenames and page numbers as needed.

## Key Differences from CLI

1. **Streaming by default**: Most client methods return streaming responses, giving you real-time progress updates
1. **Type safety**: All responses are typed Pydantic models, providing better IDE support and validation
1. **Programmatic control**: You can process responses as they arrive and implement custom logic
1. **Error handling**: Structured exception handling with specific error types
1. **Context managers**: Automatic resource cleanup with `with` statements
1. **Memory efficiency**: Streaming responses don't load all results into memory at once

The client library provides the same functionality as the CLI but with more programmatic control and better integration into Python applications.
