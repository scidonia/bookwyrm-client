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

Extract structured data from specific pages of a PDF:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamComplete, PDFStreamPageError, PDFPage
from pathlib import Path
from typing import List
import json

def extract_pdf_structure() -> List[PDFPage]:
    """Extract structured data from PDF pages 1-4."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
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

def extract_pdf_from_url() -> List[PDFPage]:
    """Extract PDF structure from a URL."""
    from bookwyrm.utils import collect_pdf_pages_from_stream
    
    client = BookWyrmClient()
    
    stream = client.stream_extract_pdf(
        pdf_url="https://example.com/document.pdf",
        start_page=1,
        num_pages=5
    )
    
    pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)
    return pages

# Extract PDF structure
pages = extract_pdf_structure()
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

Here's a complete workflow that processes a PDF from extraction to citation finding:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.utils import create_pdf_text_mapping_from_pages, query_mapping_range_in_memory, save_mapping_query_in_memory
from bookwyrm.models import PDFStreamPageResponse, PDFPage, PDFTextMapping
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

def complete_pdf_workflow() -> Tuple[List[PDFPage], PDFTextMapping, Dict[str, Any], Dict[str, Any], List]:
    """Complete workflow from PDF to citations."""
    client = BookWyrmClient()

    print("Step 1: Extract PDF structure")
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()

    pages = []
    for response in client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4
    ):
        if isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)

    print("Step 2: Create text mapping directly from pages (in-memory)")
    # Use in-memory utility function - no file I/O needed
    mapping = create_pdf_text_mapping_from_pages(pages)

    print("Step 3: Query character ranges (in-memory)")
    # Use in-memory mapping (preferred)
    result1 = query_mapping_range_in_memory(mapping, 0, 100)
    result2 = save_mapping_query_in_memory(mapping, 1000, 2000, Path("data/positions_1000-2000.json"))

    # Optional: Save extracted data and mapping if needed for later use
    save_files = False  # Set to True if you want to save files
    if save_files:
        output_data = {"pages": [page.model_dump() for page in pages]}
        with open("data/SOA_2025_Final_extracted.json", "w") as f:
            json.dump(output_data, f, indent=2)

        # Save text and mapping files
        Path("data/SOA_2025_Final_raw.txt").write_text(mapping.raw_text, encoding="utf-8")
        Path("data/SOA_2025_Final_mapping.json").write_text(mapping.model_dump_json(indent=2), encoding="utf-8")

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
    return pages, mapping, result1, result2, phrases

# Run complete workflow
results = complete_pdf_workflow()
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

The examples in this guide use sample data files included in the repository:

- [`data/SOA_2025_Final.pdf`](https://github.com/scidonia/bookwyrm-client/blob/main/data/SOA_2025_Final.pdf) - State-of-the-Art spacecraft technology PDF for extraction examples
- [`data/country-of-the-blind.txt`](https://github.com/scidonia/bookwyrm-client/blob/main/data/country-of-the-blind.txt) - H.G. Wells' "The Country of the Blind" text for phrasal analysis and summarization examples

**Note**: The SOA PDF file (`data/SOA_2025_Final.pdf`) should be present in your local repository's `data/` directory. If you don't have this file, you can substitute any PDF file for the extraction examples, adjusting the filename and page numbers as needed.

## Key Differences from CLI

1. **Streaming by default**: Most client methods return streaming responses, giving you real-time progress updates
1. **Type safety**: All responses are typed Pydantic models, providing better IDE support and validation
1. **Programmatic control**: You can process responses as they arrive and implement custom logic
1. **Error handling**: Structured exception handling with specific error types
1. **Context managers**: Automatic resource cleanup with `with` statements
1. **Memory efficiency**: Streaming responses don't load all results into memory at once

The client library provides the same functionality as the CLI but with more programmatic control and better integration into Python applications.
