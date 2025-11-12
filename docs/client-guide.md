# BookWyrm Client Library Guide

This guide demonstrates how to use the BookWyrm Python client library to perform the same operations shown in the CLI guide. We'll focus on the synchronous `BookWyrmClient` for simplicity.

## Installation and Setup

```bash
pip install bookwyrm
```

Set your API key as an environment variable:

```python
import os
os.environ["BOOKWYRM_API_KEY"] = "your-api-key-here"
```

Or pass it directly to the client:

```python
from bookwyrm import BookWyrmClient

client = BookWyrmClient(api_key="your-api-key")
```

## 1. File Classification

Classify documents to understand their content type and structure:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import ClassifyProgressUpdate, ClassifyStreamResponse, ClassifyErrorResponse
from pathlib import Path

def classify_pdf():
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
    print(f"File Size: {response.classification.file_size} bytes")
    
    return response

def classify_pdf_with_progress():
    """Classify a PDF with real-time progress updates."""
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    classification_result = None
    for response in client.stream_classify(
        content_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf"
    ):
        if isinstance(response, ClassifyProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, ClassifyStreamResponse):
            classification_result = response
            print(f"Classification complete!")
            print(f"Format: {response.classification.format_type}")
        elif isinstance(response, ClassifyErrorResponse):
            print(f"Error: {response.message}")
    
    return classification_result

# Run classification
result = classify_pdf()
```

## 2. PDF Structure Extraction

Extract structured data from specific pages of a PDF:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import PDFStreamMetadata, PDFStreamPageResponse, PDFStreamComplete, PDFStreamPageError
from pathlib import Path
import json

def extract_pdf_structure():
    """Extract structured data from PDF pages 1-4."""
    client = BookWyrmClient()
    
    pdf_path = Path("data/SOA_2025_Final.pdf")
    pdf_bytes = pdf_path.read_bytes()
    
    pages = []
    metadata = None
    
    for response in client.stream_extract_pdf(
        pdf_bytes=pdf_bytes,
        filename="SOA_2025_Final.pdf",
        start_page=1,
        num_pages=4
    ):
        if isinstance(response, PDFStreamMetadata):
            metadata = response
            print(f"Processing {response.total_pages} pages")
        elif isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
            print(f"Extracted page {response.document_page}: {len(response.page_data.text_blocks)} text elements")
        elif isinstance(response, PDFStreamPageError):
            print(f"Error on page {response.document_page}: {response.error}")
        elif isinstance(response, PDFStreamComplete):
            print(f"Extraction complete: {response.pages_processed} pages processed")
    
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

def extract_pdf_from_url():
    """Extract PDF structure from a URL."""
    client = BookWyrmClient()
    
    pages = []
    for response in client.stream_extract_pdf(
        pdf_url="https://example.com/document.pdf",
        start_page=1,
        num_pages=5
    ):
        if isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
            print(f"Extracted page {response.document_page}")
    
    return pages

# Extract PDF structure
pages = extract_pdf_structure()
```

## 3. PDF to Text Conversion with Character Mapping

The client library doesn't have a direct equivalent to the CLI's `pdf-to-text` command, but you can process the extracted PDF data to create text and character mappings:

```python
from bookwyrm.models import PDFTextMapping, CharacterMapping
from pathlib import Path
import json

def pdf_to_text_with_mapping(pdf_data_file):
    """Convert PDF extraction data to raw text with character mapping."""
    
    # Load the extracted PDF data
    with open(pdf_data_file, 'r') as f:
        pdf_data = json.load(f)
    
    # Extract text and create character mappings
    raw_text = ""
    character_mappings = []
    
    for page_data in pdf_data.get("pages", []):
        page_num = page_data.get("page_number", 1)
        
        for text_block in page_data.get("text_blocks", []):
            text = text_block.get("text", "")
            bbox = text_block.get("bbox", {})
            
            # Create character mapping for this text block
            start_char = len(raw_text)
            end_char = start_char + len(text)
            
            mapping = CharacterMapping(
                start_char=start_char,
                end_char=end_char,
                page_number=page_num,
                bounding_box=bbox,
                confidence=text_block.get("confidence", 1.0),
                text_sample=text[:50]  # First 50 characters as sample
            )
            character_mappings.append(mapping)
            
            raw_text += text + "\n"
    
    # Create the complete mapping
    text_mapping = PDFTextMapping(
        raw_text=raw_text,
        character_mappings=character_mappings,
        total_pages=len(pdf_data.get("pages", [])),
        source_file=str(pdf_data_file)
    )
    
    # Save raw text
    base_name = Path(pdf_data_file).stem
    text_file = Path(f"data/{base_name}_raw.txt")
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(raw_text)
    
    # Save character mapping
    mapping_file = Path(f"data/{base_name}_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(text_mapping.model_dump(), f, indent=2)
    
    print(f"Created raw text file: {text_file}")
    print(f"Created character mapping: {mapping_file}")
    
    return text_mapping

# Convert PDF data to text with mapping
mapping = pdf_to_text_with_mapping("data/SOA_2025_Final_pages_1-4.json")
```

## 4. Querying Character Positions

Query specific character ranges to get their bounding box coordinates:

```python
from pathlib import Path
import json

def query_character_range(mapping_file, start_char, end_char):
    """Query character positions to get bounding boxes."""
    
    # Load the character mapping
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    mapping = PDFTextMapping.model_validate(mapping_data)
    
    # Get bounding boxes for the character range
    bounding_boxes = mapping.get_bounding_boxes_for_range(start_char, end_char)
    pages = mapping.get_pages_for_range(start_char, end_char)
    
    # Get sample text from the range
    raw_text = mapping.raw_text
    sample_text = raw_text[start_char:end_char]
    
    result = {
        "start_char": start_char,
        "end_char": end_char,
        "pages": pages,
        "bounding_boxes": bounding_boxes,
        "sample_text": sample_text[:200],  # First 200 characters
        "total_characters": end_char - start_char
    }
    
    print(f"Character range {start_char}-{end_char}:")
    print(f"Pages: {pages}")
    print(f"Sample text: {sample_text[:100]}...")
    print(f"Bounding boxes found on {len(bounding_boxes)} pages")
    
    return result

def save_character_query(mapping_file, start_char, end_char, output_file):
    """Query character range and save results to file."""
    result = query_character_range(mapping_file, start_char, end_char)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved query results to {output_file}")
    return result

# Query character positions
result = query_character_range("data/SOA_2025_Final_pages_1-4_mapping.json", 974, 1089)
save_character_query("data/SOA_2025_Final_pages_1-4_mapping.json", 974, 1089, "data/character_positions.json")
```

## 5. Phrasal Text Processing

Process text files to extract meaningful phrases and text spans:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import TextResult, TextSpanResult, PhraseProgressUpdate, ResponseFormat
from pathlib import Path
import json

def process_text_to_phrases():
    """Create phrasal analysis of a text file."""
    client = BookWyrmClient()
    
    # Read text file (available in the repository)
    text_file = Path("data/country-of-the-blind.txt")
    text_content = text_file.read_text(encoding='utf-8')
    
    phrases = []
    for response in client.stream_process_text(
        text=text_content,
        response_format=ResponseFormat.WITH_OFFSETS
    ):
        if isinstance(response, PhraseProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)
    
    # Save phrases to JSONL file
    output_file = Path("data/country-of-the-blind-phrases.jsonl")
    with open(output_file, 'w') as f:
        for phrase in phrases:
            f.write(phrase.model_dump_json() + '\n')
    
    print(f"Saved {len(phrases)} phrases to {output_file}")
    return phrases

def process_text_from_url():
    """Process text from a URL."""
    client = BookWyrmClient()
    
    phrases = []
    for response in client.stream_process_text(
        text_url="https://www.gutenberg.org/files/11/11-0.txt",
        chunk_size=2000,
        response_format=ResponseFormat.WITH_OFFSETS
    ):
        if isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)
    
    # Save to JSONL
    output_file = Path("data/alice-phrases.jsonl")
    with open(output_file, 'w') as f:
        for phrase in phrases:
            f.write(phrase.model_dump_json() + '\n')
    
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
import json

def load_phrases_from_jsonl(file_path):
    """Load phrases from a JSONL file."""
    phrases = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('type') == 'text_span':
                    phrase = TextSpan(
                        text=data['text'],
                        start_char=data['start_char'],
                        end_char=data['end_char']
                    )
                    phrases.append(phrase)
    return phrases

def basic_summarization():
    """Generate a basic summary from phrases."""
    client = BookWyrmClient()
    
    # Load phrases from JSONL file
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    # Stream summarization with progress
    final_summary = None
    for response in client.stream_summarize(
        phrases=phrases,
        max_tokens=500,
        debug=True
    ):
        if isinstance(response, SummarizeProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, SummaryResponse):
            final_summary = response
            print("Summary complete!")
    
    if final_summary:
        # Save summary to JSON file
        output_file = Path("data/country-of-the-blind-summary.json")
        with open(output_file, 'w') as f:
            json.dump(final_summary.model_dump(), f, indent=2)
        
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
import json
import sys

# Add the data directory to Python path to import the Summary model
sys.path.append('data')
from summary import Summary

def structured_literary_analysis():
    """Generate structured literary analysis using the Summary model."""
    client = BookWyrmClient()
    
    # Load phrases
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    # Create structured summary using the Summary Pydantic model
    final_result = None
    for response in client.stream_summarize(
        phrases=phrases,
        summary_class=Summary,
        model_strength="smart",
        max_tokens=2000,
        debug=True
    ):
        if isinstance(response, SummarizeProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, SummaryResponse):
            final_result = response
            print("Structured analysis complete!")
    
    if final_result:
        # Save structured summary
        output_file = Path("data/country-structured-summary.json")
        with open(output_file, 'w') as f:
            json.dump(final_result.model_dump(), f, indent=2)
        
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

def high_quality_analysis():
    """Generate high-quality analysis using the 'wise' model."""
    client = BookWyrmClient()
    
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    final_result = None
    for response in client.stream_summarize(
        phrases=phrases,
        summary_class=Summary,
        model_strength="wise",
        max_tokens=4000,
        debug=True
    ):
        if isinstance(response, SummarizeProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, SummaryResponse):
            final_result = response
    
    if final_result:
        output_file = Path("data/country-detailed-analysis.json")
        with open(output_file, 'w') as f:
            json.dump(final_result.model_dump(), f, indent=2)
        print(f"High-quality analysis saved to {output_file}")
    
    return final_result

# Generate structured analysis
structured_result = structured_literary_analysis()
detailed_result = high_quality_analysis()
```

### Custom Prompts for Specialized Analysis

```python
def custom_prompt_analysis():
    """Use custom prompts for specialized literary analysis."""
    client = BookWyrmClient()
    
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    final_result = None
    for response in client.stream_summarize(
        phrases=phrases,
        chunk_prompt="Extract key themes, symbols, and literary devices from this text",
        summary_of_summaries_prompt="Create a comprehensive literary analysis focusing on themes, symbolism, and narrative techniques",
        model_strength="clever",
        max_tokens=3000
    ):
        if isinstance(response, SummaryResponse):
            final_result = response
    
    if final_result:
        output_file = Path("data/country-literary-analysis.json")
        with open(output_file, 'w') as f:
            json.dump(final_result.model_dump(), f, indent=2)
        print(f"Literary analysis saved to {output_file}")
    
    return final_result

# Generate custom analysis
custom_result = custom_prompt_analysis()
```

## 7. Citation Finding

Find specific citations related to questions in the text:

```python
from bookwyrm import BookWyrmClient
from bookwyrm.models import CitationProgressUpdate, CitationStreamResponse, CitationSummaryResponse, CitationErrorResponse
from pathlib import Path
import json

def find_citations():
    """Find citations about life-threatening situations."""
    client = BookWyrmClient()
    
    # Load phrases
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    citations = []
    for response in client.stream_citations(
        chunks=phrases,
        question="Where does the protagonist experience life threatening situations?"
    ):
        if isinstance(response, CitationProgressUpdate):
            print(f"Progress: {response.message}")
        elif isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
            print(f"Found citation: {response.citation.text[:100]}...")
        elif isinstance(response, CitationSummaryResponse):
            print(f"Search complete: {response.total_citations} citations found")
        elif isinstance(response, CitationErrorResponse):
            print(f"Error: {response.message}")
    
    # Save citations to JSON
    output_file = Path("data/protagonist-dangers.json")
    citations_data = [citation.model_dump() for citation in citations]
    with open(output_file, 'w') as f:
        json.dump(citations_data, f, indent=2)
    
    print(f"Saved {len(citations)} citations to {output_file}")
    return citations

def find_multiple_citations():
    """Ask multiple questions about the story."""
    client = BookWyrmClient()
    
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    questions = [
        "What are the main conflicts in the story?",
        "How does the protagonist adapt to his environment?",
        "What role does blindness play in the narrative?"
    ]
    
    all_citations = []
    for question in questions:
        print(f"\nSearching for: {question}")
        question_citations = []
        
        for response in client.stream_citations(
            chunks=phrases,
            question=question
        ):
            if isinstance(response, CitationStreamResponse):
                question_citations.append(response.citation)
            elif isinstance(response, CitationSummaryResponse):
                print(f"Found {response.total_citations} citations for this question")
        
        all_citations.extend(question_citations)
    
    return all_citations

def find_citations_with_limits():
    """Find citations with start and limit parameters."""
    client = BookWyrmClient()
    
    phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
    
    citations = []
    for response in client.stream_citations(
        chunks=phrases,
        question="What are the key themes in the story?",
        start=10,  # Start from chunk 10
        limit=50,  # Process only 50 chunks
        max_tokens_per_chunk=1500
    ):
        if isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
    
    return citations

# Find citations
citations = find_citations()
multiple_citations = find_multiple_citations()
```

## Complete Workflow Example

Here's a complete workflow that processes a PDF from extraction to citation finding:

```python
from bookwyrm import BookWyrmClient
from pathlib import Path
import json

def complete_pdf_workflow():
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
    
    # Save extracted data
    output_data = {"pages": [page.model_dump() for page in pages]}
    with open("data/SOA_2025_Final_extracted.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("Step 2: Convert to text and create character mapping")
    mapping = pdf_to_text_with_mapping("data/SOA_2025_Final_extracted.json")
    
    print("Step 3: Query character ranges")
    result1 = query_character_range("data/SOA_2025_Final_extracted_mapping.json", 0, 100)
    result2 = save_character_query("data/SOA_2025_Final_extracted_mapping.json", 1000, 2000, 
                                   "data/positions_1000-2000.json")
    
    print("Step 4: Process text for phrases (if we have a text file)")
    # This would require having the text content available
    
    print("Workflow complete!")
    return pages, mapping, result1, result2

# Run complete workflow
results = complete_pdf_workflow()
```

## Error Handling and Best Practices

```python
from bookwyrm import BookWyrmClient
from bookwyrm.client import BookWyrmAPIError, BookWyrmClientError

def robust_citation_search():
    """Example with proper error handling."""
    client = BookWyrmClient()
    
    try:
        phrases = load_phrases_from_jsonl("data/country-of-the-blind-phrases.jsonl")
        
        citations = []
        for response in client.stream_citations(
            chunks=phrases,
            question="What are the main themes?"
        ):
            if isinstance(response, CitationStreamResponse):
                citations.append(response.citation)
            elif isinstance(response, CitationErrorResponse):
                print(f"Citation error: {response.message}")
                break
        
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
def safe_client_usage():
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

## Key Differences from CLI

1. **Streaming by default**: Most client methods return streaming responses, giving you real-time progress updates
2. **Type safety**: All responses are typed Pydantic models, providing better IDE support and validation
3. **Programmatic control**: You can process responses as they arrive and implement custom logic
4. **Error handling**: Structured exception handling with specific error types
5. **Context managers**: Automatic resource cleanup with `with` statements
6. **Memory efficiency**: Streaming responses don't load all results into memory at once

The client library provides the same functionality as the CLI but with more programmatic control and better integration into Python applications.
