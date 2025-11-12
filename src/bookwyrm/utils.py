"""Utility functions for BookWyrm client operations."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator

from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TaskID,
)

from .models import (
    TextSpan,
    Citation,
    PDFTextMapping,
    CharacterMapping,
    PDFPage,
    # Streaming response types
    ClassifyProgressUpdate,
    ClassifyStreamResponse,
    ClassifyErrorResponse,
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    PhraseProgressUpdate,
    TextResult,
    TextSpanResult,
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
    UsageInfo,
)

console = Console()


def load_chunks_from_jsonl(file_path: Path) -> List[TextSpan]:
    """Load text chunks from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing text chunks

    Returns:
        List of TextSpan objects loaded from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid JSON or missing required fields

    Examples:
        ```python
        from bookwyrm.utils import load_chunks_from_jsonl
        from pathlib import Path

        chunks = load_chunks_from_jsonl(Path("data/chunks.jsonl"))
        print(f"Loaded {len(chunks)} chunks")
        ```
    """
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunk = TextSpan(
                        text=data["text"],
                        start_char=data.get("start_char", 0),
                        end_char=data.get("end_char", len(data["text"])),
                    )
                    chunks.append(chunk)
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"Error parsing line {line_num}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    return chunks


def load_phrases_from_jsonl(file_path: Path) -> List[TextSpan]:
    """Load phrases from a JSONL file.

    Args:
        file_path: Path to the JSONL file containing phrases

    Returns:
        List of TextSpan objects loaded from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid JSON or missing required fields

    Examples:
        ```python
        from bookwyrm.utils import load_phrases_from_jsonl
        from pathlib import Path

        phrases = load_phrases_from_jsonl(Path("data/phrases.jsonl"))
        print(f"Loaded {len(phrases)} phrases")
        ```
    """
    phrases = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    phrase = TextSpan(
                        text=data["text"],
                        start_char=data.get("start_char", 0),
                        end_char=data.get("end_char", len(data["text"])),
                    )
                    phrases.append(phrase)
                except (json.JSONDecodeError, KeyError) as e:
                    raise ValueError(f"Error parsing line {line_num}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")

    return phrases


def load_jsonl_content(file_path: Path) -> str:
    """Load raw content from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Raw file content as string

    Raises:
        FileNotFoundError: If the file doesn't exist

    Examples:
        ```python
        from bookwyrm.utils import load_jsonl_content
        from pathlib import Path

        content = load_jsonl_content(Path("data/phrases.jsonl"))
        # Use content directly with client methods that accept raw JSONL
        ```
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def save_citations_to_json(citations: List[Citation], output_path: Path) -> None:
    """Save citations to a JSON file.

    Args:
        citations: List of Citation objects to save
        output_path: Path where to save the JSON file

    Raises:
        ValueError: If there's an error saving the file

    Examples:
        ```python
        from bookwyrm.utils import save_citations_to_json
        from pathlib import Path

        # After getting citations from client
        save_citations_to_json(citations, Path("output/citations.json"))
        ```
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([citation.model_dump() for citation in citations], f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving citations: {e}")


def append_citation_to_jsonl(citation: Citation, output_path: Path) -> None:
    """Append a single citation to a JSONL file.

    Args:
        citation: Citation object to append
        output_path: Path to the JSONL file

    Raises:
        ValueError: If there's an error writing to the file

    Examples:
        ```python
        from bookwyrm.utils import append_citation_to_jsonl
        from pathlib import Path

        # Append citations as they're found during streaming
        for response in client.stream_citations(...):
            if hasattr(response, 'citation'):
                append_citation_to_jsonl(response.citation, Path("output/citations.jsonl"))
        ```
    """
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(citation.model_dump()) + "\n")
            f.flush()  # Ensure immediate write to disk
    except Exception as e:
        raise ValueError(f"Error appending citation: {e}")


def create_pdf_text_mapping_from_pages(pages: List[PDFPage]) -> PDFTextMapping:
    """Create PDF text mapping directly from page objects (in-memory).

    This function takes a list of PDFPage objects and converts them to:
    1. Raw text with all text elements joined by newlines
    2. Character mapping that maps each character position to its bounding box coordinates

    Args:
        pages: List of PDFPage objects from PDF extraction

    Returns:
        PDFTextMapping object containing the text and character mappings

    Examples:
        ```python
        from bookwyrm.utils import create_pdf_text_mapping_from_pages

        # After extracting PDF pages
        pages = []
        for response in client.stream_extract_pdf(pdf_bytes=pdf_bytes, filename="doc.pdf"):
            if hasattr(response, 'page_data'):
                pages.append(response.page_data)

        # Convert to text mapping in memory
        mapping = create_pdf_text_mapping_from_pages(pages)
        print(f"Converted {len(mapping.raw_text)} characters")
        ```
    """
    # Process pages to create raw text and mappings
    raw_text_parts = []
    character_mappings = []
    current_char_index = 0

    for page in pages:
        page_number = page.page_number
        text_blocks = page.text_blocks

        for element_index, text_block in enumerate(text_blocks):
            text = text_block.text
            coordinates = text_block.coordinates
            confidence = text_block.confidence

            # Add character mappings for each character in the text
            for char_offset, char in enumerate(text):
                mapping = CharacterMapping(
                    char_index=current_char_index,
                    page_number=page_number,
                    x1=coordinates.x1,
                    y1=coordinates.y1,
                    x2=coordinates.x2,
                    y2=coordinates.y2,
                    confidence=confidence,
                    original_text_element_index=element_index,
                )
                character_mappings.append(mapping)
                current_char_index += 1

            # Add the text to raw text parts
            raw_text_parts.append(text)

            # Add newline mapping (using the bounding box of the last character)
            if text:  # Only add newline if there was text
                newline_mapping = CharacterMapping(
                    char_index=current_char_index,
                    page_number=page_number,
                    x1=coordinates.x1,
                    y1=coordinates.y1,
                    x2=coordinates.x2,
                    y2=coordinates.y2,
                    confidence=confidence,
                    original_text_element_index=element_index,
                )
                character_mappings.append(newline_mapping)
                current_char_index += 1

    # Join all text with newlines
    raw_text = "\n".join(raw_text_parts)

    # Create the mapping object
    pdf_mapping = PDFTextMapping(
        raw_text=raw_text,
        character_mappings=character_mappings,
        total_pages=len(pages),
        total_characters=len(raw_text),
        source_file="in-memory",
    )

    return pdf_mapping


def query_mapping_range_in_memory(
    mapping: PDFTextMapping, start_char: int, end_char: int
) -> Dict[str, Any]:
    """Query character positions from in-memory mapping object to get bounding boxes.

    Args:
        mapping: PDFTextMapping object containing character mappings
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)

    Returns:
        Dictionary containing query results with bounding boxes, pages, and sample text

    Raises:
        ValueError: If the character range is invalid

    Examples:
        ```python
        from bookwyrm.utils import query_mapping_range_in_memory

        # After creating mapping from pages
        mapping = create_pdf_text_mapping_from_pages(pages)

        # Query character range in memory
        result = query_mapping_range_in_memory(mapping, 100, 200)

        print(f"Found bounding boxes on {len(result['pages'])} pages")
        print(f"Sample text: {result['sample_text']}")
        ```
    """
    # Validate character range
    if start_char < 0:
        raise ValueError(f"start_char must be >= 0 (got {start_char})")

    if end_char <= start_char:
        raise ValueError(
            f"end_char must be > start_char (got {end_char} <= {start_char})"
        )

    if start_char >= mapping.total_characters:
        raise ValueError(
            f"start_char {start_char} is beyond text length {mapping.total_characters}"
        )

    # Adjust end_char if it exceeds text length
    if end_char > mapping.total_characters:
        end_char = mapping.total_characters

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
        "total_characters": end_char - start_char,
    }

    return result


def collect_classification_from_stream(
    stream: Iterator,
    verbose: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Optional[ClassifyStreamResponse]:
    """Collect classification results from a streaming response.

    Args:
        stream: Iterator of streaming classification responses
        verbose: Whether to show detailed progress information

    Returns:
        ClassifyStreamResponse with classification results, or None if error

    Raises:
        ValueError: If classification fails or returns an error

    Examples:
        ```python
        from bookwyrm.utils import collect_classification_from_stream

        stream = client.stream_classify(content_bytes=pdf_bytes, filename="doc.pdf")
        result = collect_classification_from_stream(stream, verbose=True)

        if result:
            print(f"Format: {result.classification.format_type}")
        ```
    """
    classification_result = None

    for response in stream:
        if isinstance(response, ClassifyProgressUpdate):
            if progress and task_id:
                progress.update(task_id, description=response.message)
            elif verbose:
                console.print(f"[dim]Progress: {response.message}[/dim]")
        elif isinstance(response, ClassifyStreamResponse):
            classification_result = response
            if progress and task_id:
                progress.update(task_id, description="Classification complete!")
            elif verbose:
                console.print("[green]✓ Classification complete![/green]")
        elif isinstance(response, ClassifyErrorResponse):
            raise ValueError(f"Classification error: {response.message}")

    return classification_result


def collect_pdf_pages_from_stream(
    stream: Iterator,
    verbose: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Tuple[List[PDFPage], Optional[PDFStreamMetadata]]:
    """Collect PDF pages from a streaming extraction response.

    Args:
        stream: Iterator of streaming PDF extraction responses
        verbose: Whether to show detailed progress information

    Returns:
        Tuple of (list of PDFPage objects, metadata)

    Raises:
        ValueError: If PDF extraction fails or returns an error

    Examples:
        ```python
        from bookwyrm.utils import collect_pdf_pages_from_stream

        stream = client.stream_extract_pdf(pdf_bytes=pdf_bytes, filename="doc.pdf")
        pages, metadata = collect_pdf_pages_from_stream(stream, verbose=True)

        print(f"Extracted {len(pages)} pages")
        ```
    """
    pages = []
    metadata = None
    total_elements = 0

    for response in stream:
        if isinstance(response, PDFStreamMetadata):
            metadata = response
            if progress and task_id:
                progress.update(
                    task_id,
                    total=response.total_pages,
                    completed=0,
                    description=f"Processing {response.total_pages} pages (doc pages {response.start_page}-{response.start_page + response.total_pages - 1})",
                )
            elif verbose:
                console.print(
                    f"[dim]Document has {response.total_pages_in_document} total pages, "
                    f"processing {response.total_pages} pages starting from page {response.start_page}[/dim]"
                )
        elif isinstance(response, PDFStreamPageResponse):
            pages.append(response.page_data)
            total_elements += len(response.page_data.text_blocks)
            if progress and task_id:
                progress.update(
                    task_id,
                    completed=response.current_page,
                    description=f"Page {response.document_page} - {len(response.page_data.text_blocks)} elements found",
                )
            elif verbose:
                console.print(
                    f"[green]Page {response.document_page}: {len(response.page_data.text_blocks)} text elements[/green]"
                )
        elif isinstance(response, PDFStreamPageError):
            if progress and task_id:
                progress.update(
                    task_id,
                    completed=response.current_page,
                    description=f"Error on page {response.document_page}",
                )
            elif verbose:
                console.print(
                    f"[red]Error on page {response.document_page}: {response.error}[/red]"
                )
        elif isinstance(response, PDFStreamComplete):
            if progress and task_id:
                progress.update(
                    task_id, completed=response.current_page, description="Complete!"
                )
            elif verbose:
                console.print("[green]✓ PDF extraction complete![/green]")
        elif isinstance(response, PDFStreamError):
            raise ValueError(f"PDF extraction error: {response.error}")

    return pages, metadata


def collect_phrases_from_stream(
    stream: Iterator,
    verbose: bool = False,
    output_file: Optional[Path] = None,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> List[Union[TextResult, TextSpanResult]]:
    """Collect phrases from a streaming phrasal processing response.

    Args:
        stream: Iterator of streaming phrasal processing responses
        verbose: Whether to show detailed progress information
        output_file: Optional path to save phrases as JSONL

    Returns:
        List of TextResult or TextSpanResult objects

    Examples:
        ```python
        from bookwyrm.utils import collect_phrases_from_stream

        stream = client.stream_process_text(text=content, response_format=ResponseFormat.WITH_OFFSETS)
        phrases = collect_phrases_from_stream(stream, verbose=True, output_file=Path("phrases.jsonl"))

        print(f"Found {len(phrases)} phrases")
        ```
    """
    phrases = []

    for response in stream:
        if isinstance(response, PhraseProgressUpdate):
            if progress and task_id:
                progress.update(task_id, description=response.message)
            elif verbose:
                console.print(
                    f"[dim]Processed {response.phrases_processed} phrases, "
                    f"created {response.chunks_created} chunks[/dim]"
                )
        elif isinstance(response, (TextResult, TextSpanResult)):
            phrases.append(response)

            if verbose:
                if isinstance(response, TextSpanResult):
                    console.print(
                        f"[green]Phrase ({response.start_char}-{response.end_char}):[/green] {response.text[:100]}{'...' if len(response.text) > 100 else ''}"
                    )
                else:
                    console.print(
                        f"[green]Phrase:[/green] {response.text[:100]}{'...' if len(response.text) > 100 else ''}"
                    )

            # Save to output file if specified
            if output_file:
                try:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(response.model_dump_json(exclude_none=True) + "\n")
                        f.flush()
                except Exception as e:
                    raise ValueError(f"Error writing to output file: {e}")

    return phrases


def collect_summary_from_stream(
    stream: Iterator,
    verbose: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Optional[SummaryResponse]:
    """Collect summary results from a streaming summarization response.

    Args:
        stream: Iterator of streaming summarization responses
        verbose: Whether to show detailed progress information

    Returns:
        SummaryResponse with final summary, or None if error

    Raises:
        ValueError: If summarization fails or returns an error

    Examples:
        ```python
        from bookwyrm.utils import collect_summary_from_stream

        stream = client.stream_summarize(phrases=phrases, max_tokens=1000)
        result = collect_summary_from_stream(stream, verbose=True)

        if result:
            print(f"Summary: {result.summary}")
        ```
    """
    final_result = None

    for response in stream:
        if isinstance(response, SummarizeProgressUpdate):
            if progress and task_id:
                # Create or update task for this level if we have multiple levels
                progress.update(
                    task_id,
                    completed=response.chunks_processed,
                    total=response.total_chunks,
                    description=f"Level {response.current_level}/{response.total_levels}: {response.message}",
                )
            elif verbose:
                console.print(
                    f"[dim]Level {response.current_level}/{response.total_levels}: {response.message}[/dim]"
                )
        elif isinstance(response, RateLimitMessage):
            if verbose:
                console.print(
                    f"[orange1]⚠ Rate limit retry {response.attempt}/{response.max_attempts}[/orange1]"
                )
        elif isinstance(response, StructuralErrorMessage):
            if verbose:
                if response.error_type == "fallback":
                    console.print(f"[orange1]⚠ {response.message}[/orange1]")
                else:
                    console.print(
                        f"[orange1]⚠ Structured output retry {response.attempt}/{response.max_attempts}[/orange1]"
                    )
        elif isinstance(response, SummaryResponse):
            final_result = response
            if progress and task_id:
                progress.update(task_id, description="Complete!")
            elif verbose:
                console.print("[green]✓ Summarization complete![/green]")
        elif isinstance(response, SummarizeErrorResponse):
            raise ValueError(f"Summarization error: {response.error}")

    return final_result


def collect_citations_from_stream(
    stream: Iterator,
    verbose: bool = False,
    progress: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
) -> Tuple[List[Citation], Optional[UsageInfo]]:
    """Collect citations from a streaming citation response.

    Args:
        stream: Iterator of streaming citation responses
        verbose: Whether to show detailed progress information

    Returns:
        Tuple of (list of Citation objects, usage information)

    Examples:
        ```python
        from bookwyrm.utils import collect_citations_from_stream

        stream = client.stream_citations(chunks=chunks, question="What is mentioned?")
        citations, usage = collect_citations_from_stream(stream, verbose=True)

        print(f"Found {len(citations)} citations")
        ```
    """
    citations = []
    usage_info = None

    for response in stream:
        if isinstance(response, CitationProgressUpdate):
            if progress and task_id:
                # For URL sources, set total when we first get it
                if progress.tasks[task_id].total is None:
                    progress.update(task_id, total=response.total_chunks)
                progress.update(
                    task_id,
                    completed=response.chunks_processed,
                    description=response.message,
                )
            elif verbose:
                console.print(f"[dim]{response.message}[/dim]")
        elif isinstance(response, CitationStreamResponse):
            citations.append(response.citation)
            if verbose:
                quality_text = f"quality {response.citation.quality}/4"
                console.print(f"[green]Found citation ({quality_text})[/green]")
        elif isinstance(response, CitationSummaryResponse):
            usage_info = response.usage
            if progress and task_id:
                progress.update(
                    task_id,
                    completed=response.chunks_processed,
                    description="Complete!",
                )
            elif verbose:
                console.print(
                    f"[blue]Processing complete: {response.total_citations} citations found[/blue]"
                )
                if response.usage:
                    cost_str = (
                        f"${response.usage.estimated_cost:.4f}"
                        if response.usage.estimated_cost is not None
                        else "N/A"
                    )
                    console.print(
                        f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: {cost_str}[/dim]"
                    )
        elif isinstance(response, CitationErrorResponse):
            raise ValueError(f"Citation error: {response.error}")

    return citations, usage_info


def save_model_to_json(model: BaseModel, output_path: Path) -> None:
    """Save a Pydantic model to a JSON file.

    Args:
        model: Pydantic model instance to save
        output_path: Path where to save the JSON file

    Raises:
        ValueError: If there's an error saving the file

    Examples:
        ```python
        from bookwyrm.utils import save_model_to_json
        from pathlib import Path

        # After getting a summary result
        save_model_to_json(summary_result, Path("output/summary.json"))
        ```
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))
    except Exception as e:
        raise ValueError(f"Error saving model to {output_path}: {e}")


def save_models_list_to_json(models: List[BaseModel], output_path: Path) -> None:
    """Save a list of Pydantic models to a JSON file.

    Args:
        models: List of Pydantic model instances to save
        output_path: Path where to save the JSON file

    Raises:
        ValueError: If there's an error saving the file

    Examples:
        ```python
        from bookwyrm.utils import save_models_list_to_json
        from pathlib import Path

        # After collecting citations
        save_models_list_to_json(citations, Path("output/citations.json"))
        ```
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([model.model_dump() for model in models], f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving models list to {output_path}: {e}")


def pdf_to_text_with_mapping_from_json(
    extraction_data: Dict[str, Any],
) -> PDFTextMapping:
    """Convert PDF extraction JSON data to raw text with character position mapping (in-memory).

    This function takes the JSON data from PDF extraction and converts it to:
    1. Raw text with all text elements joined by newlines
    2. Character mapping that maps each character position to its bounding box coordinates

    Args:
        extraction_data: Dictionary containing PDF extraction data with 'pages' key

    Returns:
        PDFTextMapping object containing the text and character mappings

    Raises:
        ValueError: If the extraction data is invalid or malformed

    Examples:
        ```python
        from bookwyrm.utils import pdf_to_text_with_mapping_from_json
        import json

        # Load extraction data
        with open("data/extracted.json", "r") as f:
            extraction_data = json.load(f)

        # Convert to text mapping in memory
        mapping = pdf_to_text_with_mapping_from_json(extraction_data)
        print(f"Converted {len(mapping.raw_text)} characters")
        ```
    """
    # Validate the structure
    if "pages" not in extraction_data:
        raise ValueError("Invalid JSON format - missing 'pages' key")

    pages_data = extraction_data["pages"]
    if not pages_data:
        raise ValueError("No pages found in extraction data")

    # Convert JSON data to PDFPage objects
    from .models import PDFPage, PDFTextElement, PDFBoundingBox

    pages = []
    for page_data in pages_data:
        text_blocks = []
        for block_data in page_data.get("text_blocks", []):
            coords_data = block_data.get("coordinates", {})
            coordinates = PDFBoundingBox(
                x1=coords_data.get("x1", 0.0),
                y1=coords_data.get("y1", 0.0),
                x2=coords_data.get("x2", 0.0),
                y2=coords_data.get("y2", 0.0),
            )
            text_element = PDFTextElement(
                text=block_data.get("text", ""),
                coordinates=coordinates,
                confidence=block_data.get("confidence", 0.0),
            )
            text_blocks.append(text_element)

        page = PDFPage(
            page_number=page_data.get("page_number", 1), text_blocks=text_blocks
        )
        pages.append(page)

    # Use the in-memory function to create the mapping
    pdf_mapping = create_pdf_text_mapping_from_pages(pages)

    # Set source as in-memory by default
    pdf_mapping.source_file = "in-memory"

    return pdf_mapping


def pdf_to_text_with_mapping(
    pdf_data_file: Path, output_path: Path = None, mapping_output: Path = None
) -> PDFTextMapping:
    """Convert PDF extraction JSON file to raw text with character position mapping.

    This function takes the JSON output from PDF extraction and converts it to:
    1. Raw text file with all text elements joined by newlines
    2. Character mapping JSON that maps each character position to its bounding box coordinates

    Args:
        pdf_data_file: Path to the JSON file from PDF extraction
        output_path: Optional path for raw text output (default: pdf_data_file_raw.txt)
        mapping_output: Optional path for mapping JSON (default: pdf_data_file_mapping.json)

    Returns:
        PDFTextMapping object containing the text and character mappings

    Raises:
        FileNotFoundError: If the PDF data file doesn't exist
        ValueError: If the PDF data file is invalid or malformed

    Examples:
        ```python
        from bookwyrm.utils import pdf_to_text_with_mapping
        from pathlib import Path

        # Convert PDF extraction to text with mapping
        mapping = pdf_to_text_with_mapping(
            Path("data/extracted.json"),
            output_path=Path("data/text.txt"),
            mapping_output=Path("data/mapping.json")
        )

        print(f"Converted {len(mapping.raw_text)} characters")
        ```
    """
    try:
        # Load the PDF extraction JSON
        with open(pdf_data_file, "r", encoding="utf-8") as f:
            extraction_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {pdf_data_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Use the in-memory function to create the mapping
    pdf_mapping = pdf_to_text_with_mapping_from_json(extraction_data)

    # Update source file reference
    pdf_mapping.source_file = str(pdf_data_file)

    # Generate output filenames if not provided
    if not output_path:
        base_name = pdf_data_file.stem  # Gets filename without .json
        output_path = pdf_data_file.parent / f"{base_name}_raw.txt"
    if not mapping_output:
        base_name = pdf_data_file.stem  # Gets filename without .json
        mapping_output = pdf_data_file.parent / f"{base_name}_mapping.json"

    # Save raw text
    try:
        output_path.write_text(pdf_mapping.raw_text, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Error saving raw text: {e}")

    # Save mapping JSON
    try:
        mapping_json = pdf_mapping.model_dump_json(indent=2)
        mapping_output.write_text(mapping_json, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Error saving mapping: {e}")

    return pdf_mapping


def query_character_range_from_mapping(
    mapping_data: Dict[str, Any], start_char: int, end_char: int
) -> Dict[str, Any]:
    """Query character positions to get bounding boxes from mapping data (in-memory).

    Args:
        mapping_data: Dictionary containing PDFTextMapping data
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)

    Returns:
        Dictionary containing query results with bounding boxes, pages, and sample text

    Raises:
        ValueError: If the mapping data is invalid or character range is invalid

    Examples:
        ```python
        from bookwyrm.utils import query_character_range_from_mapping
        import json

        # Load mapping data
        with open("data/mapping.json", "r") as f:
            mapping_data = json.load(f)

        # Query character range in memory
        result = query_character_range_from_mapping(mapping_data, 100, 200)
        print(f"Found bounding boxes on {len(result['pages'])} pages")
        ```
    """
    try:
        mapping = PDFTextMapping.model_validate(mapping_data)
    except Exception as e:
        raise ValueError(f"Invalid mapping data format: {e}")

    # Use the in-memory function
    return query_mapping_range_in_memory(mapping, start_char, end_char)


def query_character_range(
    mapping_file: Path, start_char: int, end_char: int
) -> Dict[str, Any]:
    """Query character positions to get bounding boxes from a mapping file.

    Args:
        mapping_file: Path to the character mapping JSON file
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)

    Returns:
        Dictionary containing query results with bounding boxes, pages, and sample text

    Raises:
        FileNotFoundError: If the mapping file doesn't exist
        ValueError: If the mapping file is invalid or character range is invalid

    Examples:
        ```python
        from bookwyrm.utils import query_character_range
        from pathlib import Path

        result = query_character_range(
            Path("data/mapping.json"),
            start_char=100,
            end_char=200
        )

        print(f"Found bounding boxes on {len(result['pages'])} pages")
        print(f"Sample text: {result['sample_text']}")
        ```
    """
    try:
        # Load the character mapping
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {mapping_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Use the in-memory function
    return query_character_range_from_mapping(mapping_data, start_char, end_char)


def save_character_query_from_mapping(
    mapping_data: Dict[str, Any], start_char: int, end_char: int, output_file: Path
) -> Dict[str, Any]:
    """Query character range from mapping data and save results to file (in-memory).

    Args:
        mapping_data: Dictionary containing PDFTextMapping data
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)
        output_file: Path to save the query results

    Returns:
        Dictionary containing query results

    Raises:
        ValueError: If the mapping data is invalid, character range is invalid, or saving fails

    Examples:
        ```python
        from bookwyrm.utils import save_character_query_from_mapping
        from pathlib import Path
        import json

        # Load mapping data
        with open("data/mapping.json", "r") as f:
            mapping_data = json.load(f)

        result = save_character_query_from_mapping(
            mapping_data,
            start_char=100,
            end_char=200,
            output_file=Path("output/query_results.json")
        )
        ```
    """
    # Use the in-memory query function
    result = query_character_range_from_mapping(mapping_data, start_char, end_char)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving query results: {e}")

    return result


def save_character_query(
    mapping_file: Path, start_char: int, end_char: int, output_file: Path
) -> Dict[str, Any]:
    """Query character range and save results to file.

    Args:
        mapping_file: Path to the character mapping JSON file
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)
        output_file: Path to save the query results

    Returns:
        Dictionary containing query results

    Raises:
        FileNotFoundError: If the mapping file doesn't exist
        ValueError: If the mapping file is invalid, character range is invalid, or saving fails

    Examples:
        ```python
        from bookwyrm.utils import save_character_query
        from pathlib import Path

        result = save_character_query(
            Path("data/mapping.json"),
            start_char=100,
            end_char=200,
            output_file=Path("output/query_results.json")
        )
        ```
    """
    # Use the file-based query function which now calls the in-memory version
    result = query_character_range(mapping_file, start_char, end_char)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving query results: {e}")

    return result


def save_mapping_query_in_memory(
    mapping: PDFTextMapping, start_char: int, end_char: int, output_file: Path
) -> Dict[str, Any]:
    """Query character range from in-memory mapping and save results to file.

    Args:
        mapping: PDFTextMapping object containing character mappings
        start_char: Starting character index (inclusive)
        end_char: Ending character index (exclusive)
        output_file: Path to save the query results

    Returns:
        Dictionary containing query results

    Raises:
        ValueError: If the character range is invalid or saving fails

    Examples:
        ```python
        from bookwyrm.utils import save_mapping_query_in_memory
        from pathlib import Path

        # After creating mapping from pages
        mapping = create_pdf_text_mapping_from_pages(pages)

        result = save_mapping_query_in_memory(
            mapping,
            start_char=100,
            end_char=200,
            output_file=Path("output/query_results.json")
        )
        ```
    """
    # Use the in-memory query function
    result = query_mapping_range_in_memory(mapping, start_char, end_char)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving query results: {e}")

    return result
