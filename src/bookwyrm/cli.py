"""Command-line interface for BookWyrm client."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Annotated

import typer
from rich.console import Console
import sys
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

try:
    from importlib.metadata import version

    __version__ = version("bookwyrm")
except ImportError:
    __version__ = "unknown"

from bookwyrm.client import BookWyrmClient, BookWyrmAPIError
from bookwyrm.models import (
    CitationRequest,
    TextSpan,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse as BW_CitationSummaryResponse,
    CitationErrorResponse,
    SummarizeRequest,
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
    ProcessTextRequest,
    ResponseFormat,
    PhraseProgressUpdate,
    TextResult,
    TextSpanResult,
    ClassifyRequest,
    ClassifyResponse,
    PDFExtractRequest,
    PDFExtractResponse,
    StreamingPDFResponse,
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
)

console = Console()
error_console = Console(stderr=True)


def load_chunks_from_jsonl(file_path: Path) -> List[TextSpan]:
    """Load text chunks from a JSONL file."""
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
                    error_console.print(
                        f"[red]Error parsing line {line_num}: {e}[/red]"
                    )
                    sys.exit(1)
    except FileNotFoundError:
        error_console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)

    return chunks


def load_phrases_from_jsonl(file_path: Path) -> List[TextSpan]:
    """Load phrases from a JSONL file."""
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
                    error_console.print(
                        f"[red]Error parsing line {line_num}: {e}[/red]"
                    )
                    sys.exit(1)
    except FileNotFoundError:
        error_console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)

    return phrases


def load_jsonl_content(file_path: Path) -> str:
    """Load raw content from a JSONL file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        error_console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        error_console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)


def save_citations_to_json(citations, output_path: Path):
    """Save citations to a JSON file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([citation.model_dump() for citation in citations], f, indent=2)
        console.print(f"[green]Citations saved to {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving citations: {e}[/red]")


def append_citation_to_jsonl(citation, output_path: Path):
    """Append a single citation to a JSONL file."""
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(citation.model_dump()) + "\n")
            f.flush()  # Ensure immediate write to disk
    except Exception as e:
        console.print(f"[red]Error appending citation: {e}[/red]")


def display_citations_table(citations, questions=None, long=False):
    """Display citations in a rich table."""
    if not citations:
        console.print("[yellow]No citations found.[/yellow]")
        return

    table = Table(title="Found Citations")
    table.add_column("Quality", justify="center", style="cyan", no_wrap=True)
    table.add_column("Chunks", justify="center", style="magenta")
    
    # Add question column if we have multiple questions
    has_multiple_questions = questions and len(questions) > 1
    if has_multiple_questions:
        table.add_column("Question", style="yellow")
    
    table.add_column("Text", style="green")
    table.add_column("Reasoning", style="blue")

    for citation in citations:
        quality_color = (
            "red"
            if citation.quality <= 1
            else "yellow" if citation.quality <= 2 else "green"
        )
        quality_text = f"[{quality_color}]{citation.quality}/4[/{quality_color}]"

        chunk_range = (
            f"{citation.start_chunk}-{citation.end_chunk}"
            if citation.start_chunk != citation.end_chunk
            else str(citation.start_chunk)
        )

        # Apply truncation only if not in long mode
        if long:
            text_display = citation.text
            reasoning_display = citation.reasoning
        else:
            text_display = citation.text[:100] + "..." if len(citation.text) > 100 else citation.text
            reasoning_display = (
                citation.reasoning[:150] + "..."
                if len(citation.reasoning) > 150
                else citation.reasoning
            )

        # Prepare row data
        row_data = [quality_text, chunk_range]
        
        # Add question info if multiple questions
        if has_multiple_questions:
            if citation.question_index and citation.question_index <= len(questions):
                question_text = questions[citation.question_index - 1]
                question_display = f"{citation.question_index}. {question_text[:50]}{'...' if len(question_text) > 50 else ''}"
            else:
                question_display = "N/A"
            row_data.append(question_display)
        
        row_data.extend([text_display, reasoning_display])
        table.add_row(*row_data)

    console.print(table)


def display_verbose_citation(citation, questions=None, long=False):
    """Display a single citation with full details."""
    quality_color = (
        "red"
        if citation.quality <= 1
        else "yellow" if citation.quality <= 2 else "green"
    )

    chunk_range = (
        f"{citation.start_chunk}-{citation.end_chunk}"
        if citation.start_chunk != citation.end_chunk
        else str(citation.start_chunk)
    )

    panel_content = f"""[bold]Quality:[/bold] [{quality_color}]{citation.quality}/4[/{quality_color}]
[bold]Chunks:[/bold] {chunk_range}"""

    # Add question info if we have multiple questions and a question index
    if questions and len(questions) > 1 and citation.question_index:
        if citation.question_index <= len(questions):
            question_text = questions[citation.question_index - 1]
            panel_content += f"""
[bold]Question {citation.question_index}:[/bold] {question_text}"""

    panel_content += f"""
[bold]Text:[/bold] {citation.text}
[bold]Reasoning:[/bold] {citation.reasoning}"""

    console.print(
        Panel(panel_content, title="Citation Found", border_style=quality_color)
    )


app = typer.Typer(
    help="""BookWyrm Client CLI - Accelerate RAG and AI agent development.

The BookWyrm client provides powerful text processing capabilities through a simple CLI,
making it easy to build sophisticated document analysis and citation systems.

## Key Capabilities

- **Citation Finding** - Find relevant citations for questions in text chunks
- **Text Summarization** - Generate summaries with custom Pydantic models  
- **Phrasal Analysis** - Extract phrases and chunks from text using NLP
- **PDF Extraction** - Extract structured text data from PDFs with OCR
- **File Classification** - Intelligently classify files by format and type
- **Streaming Support** - Real-time progress updates for all operations

## Environment Variables

- `BOOKWYRM_API_KEY` - Your BookWyrm API key (required)
- `BOOKWYRM_API_URL` - Base URL (default: https://api.bookwyrm.ai:443)
- `BOOKWYRM_PDF_API_URL` - PDF API URL (falls back to BOOKWYRM_API_URL)

Get your API key at: https://api.bookwyrm.ai
""",
    epilog="""## Examples

```bash
bookwyrm cite "What is machine learning?" data/chunks.jsonl
bookwyrm summarize book.jsonl --model-class-file models/summary.py --model-class-name BookSummary -o summary.json
bookwyrm phrasal -f document.txt --chunk-size 1000 -o chunks.jsonl
bookwyrm extract-pdf document.pdf --start-page 5 --num-pages 10 -o extracted.json
bookwyrm classify document.pdf -o classification.json
```

For detailed help on any command, use: `bookwyrm COMMAND --help`
""",
    rich_markup_mode="markdown",
)


def version_callback(value: bool):
    if value:
        typer.echo(f"bookwyrm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", callback=version_callback, help="Show version and exit"
        ),
    ] = None,
):
    """BookWyrm Client CLI - Accelerate RAG and AI agent development.

    The BookWyrm client provides powerful text processing capabilities for building
    sophisticated document analysis and citation systems.
    """
    pass


# Global state for CLI options
class GlobalState:
    def __init__(self):
        self.base_url: str = ""
        self.api_key: Optional[str] = None
        self.verbose: bool = False


state = GlobalState()


def get_base_url(base_url: Optional[str] = None) -> str:
    """Get base URL from CLI option, environment variable, or default."""
    if base_url is not None:
        return base_url
    return os.getenv("BOOKWYRM_API_URL", "https://api.bookwyrm.ai:443")


def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Get API key from CLI option or environment variable."""
    if api_key is not None:
        return api_key
    return os.getenv("BOOKWYRM_API_KEY")


def validate_api_key(api_key: Optional[str]) -> None:
    """Validate that an API key is provided and inform user if missing."""
    if not api_key:
        error_console.print("[red]Error: No API key provided![/red]")
        error_console.print(
            "[yellow]Please provide an API key using one of these methods:[/yellow]"
        )
        error_console.print(
            "  1. Set environment variable: [cyan]export BOOKWYRM_API_KEY='your-api-key'[/cyan]"
        )
        error_console.print("  2. Use CLI option: [cyan]--api-key your-api-key[/cyan]")
        error_console.print(
            "\n[dim]You can get an API key from https://api.bookwyrm.ai[/dim]"
        )
        raise typer.Exit(1)


@app.command()
def cite(
    jsonl_input: Annotated[
        Optional[str],
        typer.Argument(
            help="Path to JSONL file containing text chunks (optional if using --file or --url)"
        ),
    ] = None,
    question: Annotated[
        Optional[List[str]],
        typer.Option("--question", "-q", help="Question to find citations for (can be used multiple times)"),
    ] = None,
    questions_file: Annotated[
        Optional[Path],
        typer.Option("--questions-file", help="File containing questions, one per line", exists=True),
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(help="URL to JSONL file (alternative to file path)"),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("--file", help="JSONL file to read chunks from", exists=True),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output file for citations (JSON for non-streaming, JSONL for streaming)",
        ),
    ] = None,
    start: Annotated[int, typer.Option(help="Start chunk index (default: 0)")] = 0,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit number of chunks to process")
    ] = None,
    max_tokens: Annotated[
        int, typer.Option(help="Maximum tokens per chunk (default: 1000)")
    ] = 1000,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the BookWyrm API (overrides BOOKWYRM_API_URL env var)"
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Show detailed citation information")
    ] = False,
    long: Annotated[
        bool, typer.Option("--long", help="Show full citation text without truncation")
    ] = False,
):
    """Find citations for questions in text chunks.

    This command searches through text chunks to find relevant citations that answer
    questions. It supports both local JSONL files and remote URLs, and can handle
    single or multiple questions.

    ## Input Format

    The JSONL file should contain text chunks in this format:
    ```json
    {"text": "chunk text", "start_char": 0, "end_char": 10}
    ```

    ## Question Input Methods

    1. **Single question**: Use `--question "Your question here"`
    2. **Multiple questions**: Use `--question` multiple times: `--question "Q1" --question "Q2"`
    3. **Questions file**: Use `--questions-file questions.txt` with one question per line

    ## Examples

    ```bash
    # Single question
    bookwyrm cite --question "What is machine learning?" ml_chunks.jsonl

    # Multiple questions
    bookwyrm cite --question "What is AI?" --question "How does ML work?" data.jsonl

    # Questions from file
    bookwyrm cite --questions-file questions.txt data.jsonl -o citations.json

    # From URL with multiple questions
    bookwyrm cite --question "Q1" --question "Q2" --url https://example.com/chunks.jsonl

    # Limit processing
    bookwyrm cite --question "Question" data.jsonl --start 10 --limit 50
    ```

    ## Output Formats

    - **JSON (non-streaming)**: Array of citation objects
    - **JSONL (streaming)**: One citation per line as they're found
    - For multiple questions, citations include question index and text
    """

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate and collect questions
    questions = []
    question_sources = [question, questions_file]
    provided_question_sources = [s for s in question_sources if s is not None]

    if len(provided_question_sources) != 1:
        error_console.print(
            "[red]Error: Exactly one of --question or --questions-file must be provided[/red]"
        )
        raise typer.Exit(1)

    if question:
        questions = question
    elif questions_file:
        try:
            with open(questions_file, "r", encoding="utf-8") as f:
                questions = [line.strip() for line in f if line.strip()]
            if not questions:
                error_console.print(
                    "[red]Error: Questions file is empty or contains no valid questions[/red]"
                )
                raise typer.Exit(1)
            console.print(f"[blue]Loaded {len(questions)} questions from {questions_file}[/blue]")
        except Exception as e:
            error_console.print(f"[red]Error reading questions file: {e}[/red]")
            raise typer.Exit(1)

    # Validate input sources
    input_sources = [jsonl_input, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        error_console.print(
            "[red]Error: Exactly one of file argument, --url, or --file must be provided[/red]"
        )
        raise typer.Exit(1)

    # Handle different input sources
    if file or jsonl_input:
        # Use local file
        file_path = file if file else Path(jsonl_input)
        console.print(f"[blue]Loading chunks from {file_path}...[/blue]")
        chunks = load_chunks_from_jsonl(file_path)
        console.print(f"[green]Loaded {len(chunks)} chunks[/green]")

        # Determine final question format
        final_question = questions[0] if len(questions) == 1 else questions
        
        request = CitationRequest(
            chunks=chunks,
            question=final_question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens,
        )
    else:
        # Use URL
        console.print(f"[blue]Using JSONL from URL: {url}[/blue]")
        
        # Determine final question format
        final_question = questions[0] if len(questions) == 1 else questions
        
        request = CitationRequest(
            jsonl_url=url,
            question=final_question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens,
        )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        # Display questions being processed
        if len(questions) == 1:
            console.print(f"[blue]Streaming citations for: {questions[0]}[/blue]")
        else:
            console.print(f"[blue]Streaming citations for {len(questions)} questions:[/blue]")
            for i, q in enumerate(questions, 1):
                console.print(f"[dim]  {i}. {q}[/dim]")
        
        if url:
            console.print(f"[dim]Source: {url}[/dim]")
        if output:
            console.print(f"[dim]Streaming citations to {output} (JSONL format)[/dim]")

        citations = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            # For URL sources, we don't know the total chunks initially
            total_chunks = len(chunks) if not url else None
            task = progress.add_task("Processing chunks...", total=total_chunks)

            for response in client.stream_citations(
                **request.model_dump(exclude_none=True)
            ):
                if isinstance(response, CitationProgressUpdate):
                    # For URL sources, set total when we first get it
                    if url and progress.tasks[task].total is None:
                        progress.update(task, total=response.total_chunks)
                    progress.update(
                        task,
                        completed=response.chunks_processed,
                        description=response.message,
                    )
                elif isinstance(response, CitationStreamResponse):
                    citations.append(response.citation)
                    if state.verbose:
                        display_verbose_citation(response.citation, questions=questions, long=long)
                    else:
                        quality_text = f"quality {response.citation.quality}/4"
                        if len(questions) > 1 and response.citation.question_index:
                            question_info = f" for Q{response.citation.question_index}"
                        else:
                            question_info = ""
                        console.print(
                            f"[green]Found citation ({quality_text}){question_info}[/green]"
                        )
                    # Immediately append to output file if specified
                    if output:
                        append_citation_to_jsonl(response.citation, output)
                elif isinstance(response, BW_CitationSummaryResponse):
                    progress.update(
                        task,
                        completed=response.chunks_processed,
                        description="Complete!",
                    )
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
                    console.print(f"[red]Error: {response.error}[/red]")

        display_citations_table(citations, questions=questions, long=long)

        if output:
            console.print(f"[green]Citations streamed to {output}[/green]")

    except BookWyrmAPIError as e:
        error_console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            error_console.print(f"[red]Status Code: {e.status_code}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        error_console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def summarize(
    jsonl_file: Annotated[
        Path, typer.Argument(help="JSONL file containing phrases", exists=True)
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output file for summary (JSON format)"),
    ] = None,
    max_tokens: Annotated[
        int,
        typer.Option(help="Maximum tokens per chunk (max: 131,072, default: 10000)"),
    ] = 10000,
    include_debug: Annotated[
        bool, typer.Option("--include-debug", help="Include intermediate summaries")
    ] = False,
    # Structured output options (commented out)
    # model_class_file: Annotated[
    #     Optional[Path],
    #     typer.Option(
    #         "--model-class-file",
    #         help="Python file containing Pydantic model class for structured output",
    #         exists=True,
    #     ),
    # ] = None,
    # model_class_name: Annotated[
    #     Optional[str],
    #     typer.Option(
    #         "--model-class-name",
    #         help="Name of the Pydantic model class to use (required with --model-class-file)",
    #     ),
    # ] = None,
    # chunk_prompt: Annotated[
    #     Optional[str],
    #     typer.Option(
    #         "--chunk-prompt",
    #         help="Custom prompt for chunk summarization (requires --summary-prompt)",
    #     ),
    # ] = None,
    # summary_prompt: Annotated[
    #     Optional[str],
    #     typer.Option(
    #         "--summary-prompt",
    #         help="Custom prompt for summary of summaries (requires --chunk-prompt)",
    #     ),
    # ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the BookWyrm API (overrides BOOKWYRM_API_URL env var)"
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Show detailed information")
    ] = False,
):
    """Summarize text content from JSONL files.
    
    This command performs hierarchical summarization of text phrases, with support for
    structured output using Pydantic models and custom prompts.
    
    ## Input Format
    
    The JSONL file should contain phrases in this format:
    ```json
    {"text": "phrase text", "start_char": 0, "end_char": 15}
    ```
    
    ## Features (Some Temporarily Disabled)
    
    - **Structured Output** (temporarily disabled): Use `--model-class-file` and `--model-class-name` to generate structured summaries that conform to your Pydantic model schema. The output file is required when using structured output.
    
    - **Custom Prompts** (temporarily disabled): Use `--chunk-prompt` and `--summary-prompt` together to customize the summarization process. Both prompts are required when using custom prompts.
    
    ## Examples
    
    ```bash
    # Basic summarization
    bookwyrm summarize book_phrases.jsonl --output summary.json
    
    # With debug information
    bookwyrm summarize data.jsonl --debug --output detailed_summary.json
    
    # Larger chunks
    bookwyrm summarize large_text.jsonl --max-tokens 20000 --output summary.json
    ```
    
    ### Temporarily Disabled Examples
    
    ```bash
    # Structured output with Pydantic model (temporarily disabled)
    # bookwyrm summarize book.jsonl \
    #   --model-class-file models/book_summary.py \
    #   --model-class-name BookSummary \
    #   --output structured_summary.json
    
    # Custom prompts (temporarily disabled)
    # bookwyrm summarize scientific_text.jsonl \
    #   --chunk-prompt "Extract key scientific concepts and findings" \
    #   --summary-prompt "Create a comprehensive scientific overview" \
    #   --output science_summary.json
    ```
    
    ## Output Format
    
    JSON file containing summary, metadata, and optionally intermediate summaries
    """

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate max_tokens
    if max_tokens > 131072:
        console.print(
            f"[red]Error: max_tokens cannot exceed 131,072 (got {max_tokens})[/red]"
        )
        raise typer.Exit(1)
    if max_tokens < 1:
        console.print(
            f"[red]Error: max_tokens must be at least 1 (got {max_tokens})[/red]"
        )
        raise typer.Exit(1)

    # Structured output validation (commented out)
    # Validate model class options
    # if model_class_file and not model_class_name:
    #     console.print(
    #         "[red]Error: --model-class-name is required when --model-class-file is provided[/red]"
    #     )
    #     raise typer.Exit(1)
    # if model_class_name and not model_class_file:
    #     console.print(
    #         "[red]Error: --model-class-file is required when --model-class-name is provided[/red]"
    #     )
    #     raise typer.Exit(1)

    # Validate custom prompt options
    # if chunk_prompt and not summary_prompt:
    #     console.print(
    #         "[red]Error: --summary-prompt is required when --chunk-prompt is provided[/red]"
    #     )
    #     raise typer.Exit(1)
    # if summary_prompt and not chunk_prompt:
    #     console.print(
    #         "[red]Error: --chunk-prompt is required when --summary-prompt is provided[/red]"
    #     )
    #     raise typer.Exit(1)

    # Validate mutually exclusive options
    # if (model_class_file or model_class_name) and (chunk_prompt or summary_prompt):
    #     console.print(
    #         "[red]Error: Cannot specify both model class options and custom prompt options. These are mutually exclusive.[/red]"
    #     )
    #     raise typer.Exit(1)

    # Require output file when using structured output with Pydantic model
    # if (model_class_file or model_class_name) and not output:
    #     console.print(
    #         "[red]Error: --output is required when using structured output with --model-class-file and --model-class-name[/red]"
    #     )
    #     console.print(
    #         "[dim]Structured output generates JSON that should be saved to a file for proper access.[/dim]"
    #     )
    #     raise typer.Exit(1)

    console.print(f"[blue]Loading JSONL file: {jsonl_file}[/blue]")
    content = load_jsonl_content(jsonl_file)

    # Structured output handling (commented out)
    # Handle model class loading if specified
    model_name = None
    model_schema_json = None
    # if model_class_file and model_class_name:
    #     try:
    #         console.print(f"[blue]Loading model class '{model_class_name}' from {model_class_file}[/blue]")
    #
    #         # Load the Python file as a module
    #         import importlib.util
    #         import sys
    #
    #         spec = importlib.util.spec_from_file_location("user_model", model_class_file)
    #         if spec is None or spec.loader is None:
    #             raise ImportError(f"Could not load module from {model_class_file}")
    #
    #         user_module = importlib.util.module_from_spec(spec)
    #         sys.modules["user_model"] = user_module
    #         spec.loader.exec_module(user_module)
    #
    #         # Get the model class
    #         if not hasattr(user_module, model_class_name):
    #             raise AttributeError(f"Class '{model_class_name}' not found in {model_class_file}")
    #
    #         model_class = getattr(user_module, model_class_name)
    #
    #         # Validate it's a Pydantic model
    #         from pydantic import BaseModel
    #         if not issubclass(model_class, BaseModel):
    #             raise TypeError(f"Class '{model_class_name}' must be a Pydantic BaseModel")
    #
    #         # Get the schema
    #         model_name = model_class_name
    #         model_schema_json = json.dumps(model_class.model_json_schema())
    #
    #         console.print(f"[green]Successfully loaded model class '{model_class_name}'[/green]")
    #
    #     except Exception as e:
    #         console.print(f"[red]Error loading model class: {e}[/red]")
    #         raise typer.Exit(1)

    request = SummarizeRequest(
        content=content,
        max_tokens=max_tokens,
        debug=include_debug,
        # Structured output options (set to None)
        # model_name=model_name,
        # model_schema_json=model_schema_json,
        # chunk_prompt=chunk_prompt,
        # summary_of_summaries_prompt=summary_prompt,
    )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        console.print("[blue]Starting summarization...[/blue]")

        final_result = None
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            level_tasks = {}  # Track tasks for each level

            for response in client.stream_summarize(
                **request.model_dump(exclude_none=True)
            ):
                if isinstance(response, SummarizeProgressUpdate):
                    # Create or update task for this level
                    task_id = f"level_{response.current_level}"

                    if task_id not in level_tasks:
                        # Create new task for this level
                        level_tasks[task_id] = progress.add_task(
                            f"Level {response.current_level}/{response.total_levels}",
                            total=response.total_chunks,
                        )

                    # Update the task
                    progress.update(
                        level_tasks[task_id],
                        completed=response.chunks_processed,
                        description=f"Level {response.current_level}/{response.total_levels}: {response.message}",
                    )

                elif isinstance(response, RateLimitMessage):
                    console.print(
                        f"[orange1]⚠ Rate limit retry {response.attempt}/{response.max_attempts}[/orange1]",
                        end="\r",
                    )

                elif isinstance(response, StructuralErrorMessage):
                    if response.error_type == "fallback":
                        console.print(f"[orange1]⚠ {response.message}[/orange1]")
                    else:
                        console.print(
                            f"[orange1]⚠ Structured output retry {response.attempt}/{response.max_attempts}[/orange1]",
                            end="\r",
                        )

                elif isinstance(response, SummaryResponse):
                    final_result = response

                    # Complete all remaining tasks
                    for task_id in level_tasks.values():
                        # Get the task from the progress object
                        for task in progress.tasks:
                            if task.id == task_id:
                                progress.update(task_id, completed=task.total)
                                break

                    console.print("[green]✓ Summarization complete![/green]")

                elif isinstance(response, SummarizeErrorResponse):
                    console.print(f"[red]Error: {response.error}[/red]")
                    sys.exit(1)

        if final_result is None:
            console.print("[red]No summary received from server[/red]")
            sys.exit(1)

        # Display results
        if state.verbose or include_debug:
            console.print(
                f"[dim]Total tokens processed: {final_result.total_tokens}[/dim]"
            )
            console.print(
                f"[dim]Subsummaries created: {final_result.subsummary_count}[/dim]"
            )
            console.print(f"[dim]Levels used: {final_result.levels_used}[/dim]")

        # Show intermediate summaries if debug mode
        if include_debug and final_result.intermediate_summaries:
            console.print("\n[bold]Intermediate Summaries by Level:[/bold]")
            for level, summaries in enumerate(final_result.intermediate_summaries, 1):
                console.print(
                    f"\n[blue]Level {level} ({len(summaries)} summaries):[/blue]"
                )
                for i, summary in enumerate(summaries, 1):
                    console.print(f"[dim]{i}.[/dim] {summary}")

        console.print("\n[bold]Final Summary:[/bold]")

        # Structured output display (commented out)
        # If we used a structured model, try to parse and display the JSON nicely
        # if model_name and model_schema_json:
        #     try:
        #         structured_data = json.loads(final_result.summary)
        #         console.print("[dim]Structured output:[/dim]")
        #         for key, value in structured_data.items():
        #             if value is not None:
        #                 console.print(f"[cyan]{key}:[/cyan] {value}")
        #     except json.JSONDecodeError:
        #         # Fallback to raw text if JSON parsing fails
        #         console.print(final_result.summary)
        # else:
        console.print(final_result.summary)

        # Save to output file if specified
        if output:
            try:
                # Structured output parsing (commented out)
                # If we used a structured model, parse the JSON summary for better storage
                summary_data = final_result.summary
                # if model_name and model_schema_json:
                #     try:
                #         summary_data = json.loads(final_result.summary)
                #     except json.JSONDecodeError:
                #         # Keep as string if parsing fails
                #         pass

                output_data = {
                    "summary": summary_data,
                    "subsummary_count": final_result.subsummary_count,
                    "levels_used": final_result.levels_used,
                    "total_tokens": final_result.total_tokens,
                    "source_file": str(jsonl_file),
                    "max_tokens": max_tokens,
                    # "model_used": model_name if model_name else None,
                    "intermediate_summaries": (
                        final_result.intermediate_summaries if include_debug else None
                    ),
                }

                output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
                console.print(f"\n[green]Summary saved to: {output}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving to {output}: {e}[/red]")

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def phrasal(
    input_text: Annotated[
        Optional[str],
        typer.Argument(help="Text to process (optional if using --url or --file)"),
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(help="URL to fetch text from"),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("-f", "--file", help="File to read text from", exists=True),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output file for phrases (JSONL format)"),
    ] = None,
    chunk_size: Annotated[
        Optional[int],
        typer.Option(
            help="Target size for each chunk (if not specified, returns phrases individually)"
        ),
    ] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help="Return text only without position data")
    ] = False,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the BookWyrm API (overrides BOOKWYRM_API_URL env var)"
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Show detailed information")
    ] = False,
):
    """Stream text processing using phrasal analysis to extract phrases or chunks.

    This command breaks down text into meaningful phrases or chunks using NLP with
    real-time streaming results. It supports processing from direct text input, files, or URLs.

    ## Response Formats

    - **with_offsets**: Include character position information (start_char, end_char)
    - **text_only**: Return only the text content without position data

    ## Response Format Control

    - **Default**: Include character position information (with_offsets)
    - **--text-only**: Return only text content without position data

    ## Chunking

    Use `--chunk-size` to create chunks of approximately the specified character count.
    Without `--chunk-size`, returns individual phrases.

    ## Examples

    ```bash
    # Process text directly
    bookwyrm phrasal "Natural language processing is fascinating." -o phrases.jsonl

    # Process file with position offsets (default behavior)
    bookwyrm phrasal -f document.txt --output phrases.jsonl

    # Create chunks of specific size (with position offsets by default)
    bookwyrm phrasal -f large_text.txt --chunk-size 1000 --output chunks.jsonl

    # Process from URL
    bookwyrm phrasal --url https://example.com/text.txt --output phrases.jsonl

    # Text only format using boolean flag
    bookwyrm phrasal -f text.txt --text-only --output simple_phrases.jsonl

    # Text-only format (no position data)
    bookwyrm phrasal -f text.txt --text-only --output simple_phrases.jsonl

    ```

    ## Output Format

    JSONL file with one phrase/chunk per line:
    ```json
    {"type": "text_span", "text": "phrase text", "start_char": 0, "end_char": 12}
    ```

    Or for text-only format:
    ```json
    {"type": "text", "text": "phrase text"}
    ```
    """

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Set format based on text_only flag
    if text_only:
        format = "text_only"
    else:
        format = "with_offsets"  # Default

    # Validate input sources
    input_sources = [input_text, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
            "[red]Error: Exactly one of text argument, --url, or --file must be provided[/red]"
        )
        raise typer.Exit(1)

    # Get text from the appropriate source
    if file:
        try:
            text = file.read_text(encoding="utf-8")
            console.print(
                f"[blue]Loaded text from {file} ({len(text)} characters)[/blue]"
            )
        except Exception as e:
            error_console.print(f"[red]Error reading file {file}: {e}[/red]")
            raise typer.Exit(1)
    elif url:
        text = None  # Will be handled by the API
        console.print(f"[blue]Processing text from URL: {url}[/blue]")
    else:
        text = input_text
        console.print(f"[blue]Processing provided text ({len(text)} characters)[/blue]")

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        console.print("[blue]Starting phrasal processing...[/blue]")
        if output:
            console.print(f"[dim]Streaming results to {output} (JSONL format)[/dim]")

        phrases = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task = progress.add_task("Processing text...", total=None)

            # Use function-based interface with boolean flags
            # Convert format string to ResponseFormat enum
            if format == "with_offsets":
                response_format = ResponseFormat.WITH_OFFSETS
            elif format == "text_only":
                response_format = ResponseFormat.TEXT_ONLY
            else:
                response_format = ResponseFormat.WITH_OFFSETS  # default

            for response in client.stream_process_text(
                text=text,
                text_url=url,
                chunk_size=chunk_size,
                response_format=response_format,
            ):
                # Show ALL responses in debug mode FIRST
                debug_enabled = os.getenv("BOOKWYRM_DEBUG") == "1"
                if debug_enabled:
                    console.print(f"[blue]DEBUG - Raw response received:[/blue]")
                    console.print(f"[dim]Type: {type(response)}[/dim]")
                    console.print(f"[dim]String representation: {response}[/dim]")
                    if hasattr(response, "model_dump_json"):
                        console.print(
                            f"[dim]JSON: {response.model_dump_json(exclude_none=True)}[/dim]"
                        )
                    elif hasattr(response, "__dict__"):
                        console.print(f"[dim]Dict: {response.__dict__}[/dim]")
                    console.print(
                        f"[dim]Attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}[/dim]"
                    )
                    console.print("[dim]" + "=" * 50 + "[/dim]")

                # Handle raw line debug responses
                if hasattr(response, "type") and response.type == "raw_line_debug":
                    if debug_enabled:
                        console.print(
                            f"[cyan]RAW LINE DEBUG:[/cyan] length={response.line_length}"
                        )
                        console.print(
                            f"[cyan]Raw line:[/cyan] {repr(response.raw_line)}"
                        )
                        console.print(
                            f"[cyan]Stripped:[/cyan] {repr(response.line_stripped)}"
                        )
                    continue  # Don't process raw debug lines further

                if isinstance(response, PhraseProgressUpdate):
                    progress.update(
                        task,
                        description=response.message,
                    )
                    if state.verbose:
                        console.print(
                            f"[dim]Processed {response.phrases_processed} phrases, "
                            f"created {response.chunks_created} chunks[/dim]"
                        )
                elif isinstance(response, (TextResult, TextSpanResult)):
                    # Handle both TextResult and TextSpanResult properly
                    phrases.append(response)

                    if state.verbose:
                        if isinstance(response, TextSpanResult):
                            console.print(
                                f"[green]Phrase ({response.start_char}-{response.end_char}):[/green] {response.text[:100]}{'...' if len(response.text) > 100 else ''}"
                            )
                        else:
                            console.print(
                                f"[green]Phrase:[/green] {response.text[:100]}{'...' if len(response.text) > 100 else ''}"
                            )

                    # Immediately append to output file if specified
                    if output:
                        try:
                            with open(output, "a", encoding="utf-8") as f:
                                f.write(
                                    response.model_dump_json(exclude_none=True) + "\n"
                                )
                                f.flush()
                        except Exception as e:
                            console.print(
                                f"[red]Error writing to output file: {e}[/red]"
                            )
                else:
                    # Unknown response types - always show these in debug mode
                    debug_enabled = os.getenv("BOOKWYRM_DEBUG") == "1"
                    if debug_enabled or state.verbose:
                        console.print(
                            f"[yellow]Unknown response type: {type(response)} - {getattr(response, 'type', 'no type field')}[/yellow]"
                        )
                        # Also show the raw data for unknown types
                        if hasattr(response, "model_dump"):
                            console.print(
                                f"[yellow]Raw data: {response.model_dump()}[/yellow]"
                            )

            progress.update(task, description="Complete!")

        console.print(
            f"[green]Processing complete: {len(phrases)} phrases/chunks found[/green]"
        )

        # Display summary table
        if phrases and not state.verbose:
            table = Table(title="Phrasal Processing Results")
            table.add_column("Index", justify="right", style="cyan", no_wrap=True)
            if format == "with_offsets":
                table.add_column("Position", justify="center", style="magenta")
            table.add_column("Text", style="green")

            for i, phrase in enumerate(phrases[:10]):  # Show first 10
                row = [str(i + 1)]
                if format == "with_offsets":
                    if isinstance(phrase, TextSpanResult):
                        row.append(f"{phrase.start_char}-{phrase.end_char}")
                    else:
                        row.append("N/A")

                text_preview = (
                    phrase.text[:80] + "..." if len(phrase.text) > 80 else phrase.text
                )
                row.append(text_preview)
                table.add_row(*row)

            if len(phrases) > 10:
                table.add_row("...", "..." if format == "with_offsets" else "", "...")

            console.print(table)

        if output:
            console.print(f"[green]Results saved to {output}[/green]")

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def classify(
    url: Annotated[
        Optional[str],
        typer.Option(help="URL to classify"),
    ] = None,
    file: Annotated[
        Optional[Path], typer.Option("--file", help="File to classify", exists=True)
    ] = None,
    filename: Annotated[
        Optional[str], typer.Option(help="Optional filename hint for classification")
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output file for classification results (JSON format)",
        ),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the BookWyrm API (overrides BOOKWYRM_API_URL env var)"
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Show detailed information")
    ] = False,
):
    """Classify files to determine their type and format.

    This command analyzes files, URLs, or stdin content to determine their format type, 
    content type, MIME type, and other classification details.

    ## Classification Includes

    - **Format type** (text, image, binary, archive, etc.)
    - **Content type** (python_code, json_data, jpeg_image, etc.)
    - **MIME type** detection
    - **Confidence score** (0.0-1.0)
    - **Additional details** (encoding, language, etc.)

    ## Examples

    ```bash
    # Classify local file
    bookwyrm classify --file document.pdf

    # Classify from URL
    bookwyrm classify --url https://example.com/file.dat

    # Classify from stdin
    echo "import pandas as pd" | bookwyrm classify --filename script.py

    # With output file
    bookwyrm classify --file unknown_file.bin --output classification.json

    # With filename hint
    bookwyrm classify --file data.txt --filename "research_data.csv" --output results.json
    ```

    ## Output Format

    JSON file containing classification results, file size, and sample preview
    """

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate input sources - allow --file, --url, or stdin (no args)
    input_sources = [url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) > 1:
        error_console.print(
            "[red]Error: Only one of --url or --file can be provided[/red]"
        )
        raise typer.Exit(1)

    # Get content from the appropriate source
    if file:
        try:
            # Always read as binary and base64 encode for multipart upload
            import base64

            binary_content = file.read_bytes()
            content = base64.b64encode(binary_content).decode("ascii")
            console.print(
                f"[blue]Classifying file: {file} ({len(binary_content)} bytes)[/blue]"
            )

            # Use the actual filename if no hint provided
            effective_filename = filename or file.name
        except Exception as e:
            console.print(f"[red]Error reading file {file}: {e}[/red]")
            raise typer.Exit(1)
    elif url:
        # Handle URL - we'll fetch it and send as multipart
        console.print(f"[blue]Classifying URL resource: {url}[/blue]")
        try:
            import httpx
            import base64

            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()
                content = base64.b64encode(response.content).decode("ascii")

            console.print(
                f"[blue]Downloaded {len(response.content)} bytes from URL[/blue]"
            )

            # Extract filename hint from URL if not provided
            effective_filename = filename
            if not effective_filename:
                from urllib.parse import urlparse

                parsed_url = urlparse(url)
                if parsed_url.path:
                    potential_filename = parsed_url.path.split("/")[-1]
                    if potential_filename and "." in potential_filename:
                        effective_filename = potential_filename
                        console.print(
                            f"[dim]Using filename hint from URL: {effective_filename}[/dim]"
                        )
        except Exception as e:
            error_console.print(f"[red]Error fetching URL {url}: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Read from stdin
        console.print("[blue]Reading content from stdin...[/blue]")
        try:
            import base64
            
            stdin_content = sys.stdin.read()
            if not stdin_content.strip():
                error_console.print("[red]Error: No content provided via stdin[/red]")
                raise typer.Exit(1)
                
            # Encode stdin content as base64
            content = base64.b64encode(stdin_content.encode('utf-8')).decode("ascii")
            console.print(f"[blue]Read {len(stdin_content)} characters from stdin[/blue]")
            
            # Use filename hint or default
            effective_filename = filename or "stdin_content"
        except Exception as e:
            error_console.print(f"[red]Error reading from stdin: {e}[/red]")
            raise typer.Exit(1)

    # Create request
    request = ClassifyRequest(
        content=content,
        filename=effective_filename,
        content_encoding="base64",
    )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        console.print("[blue]Starting classification...[/blue]")

        with console.status("[bold green]Analyzing..."):
            response = client.classify(**request.model_dump(exclude_none=True))

        console.print("[green]✓ Classification complete![/green]")

        # Display results in a nice table
        table = Table(title="Classification Results")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Format Type", response.classification.format_type)
        table.add_row("Content Type", response.classification.content_type)
        table.add_row("MIME Type", response.classification.mime_type)
        table.add_row("Confidence", f"{response.classification.confidence:.2%}")
        table.add_row("File Size", f"{response.file_size:,} bytes")

        if response.sample_preview:
            preview = (
                response.sample_preview[:100] + "..."
                if len(response.sample_preview) > 100
                else response.sample_preview
            )
            table.add_row("Sample Preview", preview)

        console.print(table)

        # Display classification methods if available
        if response.classification.classification_methods:
            console.print(
                f"\n[dim]Classification methods used: {', '.join(response.classification.classification_methods)}[/dim]"
            )

        # Display additional details if available
        if response.classification.details:
            console.print("\n[bold]Additional Details:[/bold]")
            details_table = Table()
            details_table.add_column("Key", style="cyan")
            details_table.add_column("Value", style="yellow")

            for key, value in response.classification.details.items():
                details_table.add_row(key, str(value))

            console.print(details_table)

        # Save to output file if specified
        if output:
            try:
                output_data = {
                    "classification": response.classification.model_dump(),
                    "file_size": response.file_size,
                    "sample_preview": response.sample_preview,
                    "source": {
                        "file": str(file) if file else None,
                        "url": url,
                        "filename_hint": filename,
                    },
                }

                output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
                console.print(
                    f"\n[green]Classification results saved to: {output}[/green]"
                )
            except Exception as e:
                console.print(f"[red]Error saving to {output}: {e}[/red]")

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def extract_pdf(
    pdf_file: Annotated[
        Optional[Path],
        typer.Argument(
            help="PDF file to extract from (optional if using --file or --url)"
        ),
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(help="PDF URL to extract from"),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("--file", help="PDF file to extract from", exists=True),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Output file for extracted data (JSON format)",
        ),
    ] = None,
    start_page: Annotated[
        Optional[int],
        typer.Option(help="1-based page number to start from"),
    ] = None,
    num_pages: Annotated[
        Optional[int],
        typer.Option(help="Number of pages to process from start_page"),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the PDF extraction API (overrides BOOKWYRM_API_URL env var)"
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
        ),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Show detailed information")
    ] = False,
):
    """Extract structured data from PDF files using OCR.

    This command extracts text elements from PDF files with position coordinates,
    confidence scores, and bounding box information. It supports both local files
    and remote URLs, with optional page range selection.

    ## Features

    - **OCR-based text extraction** with confidence scores
    - **Bounding box coordinates** for each text element
    - **Page range selection** (start_page + num_pages)
    - **Streaming progress updates**
    - **Support for both local files and URLs**

    ## Page Selection

    - `start_page`: 1-based page number to begin extraction
    - `num_pages`: Number of pages to process from start_page
    - Omit both to process entire document

    ## Examples

    ```bash
    # Extract entire PDF
    bookwyrm extract-pdf document.pdf --output extracted.json

    # Extract specific pages
    bookwyrm extract-pdf large_doc.pdf --start-page 5 --num-pages 10 --output pages_5_14.json

    # Extract from URL
    bookwyrm extract-pdf --url https://example.com/document.pdf --output extracted.json

    # Non-streaming mode
    bookwyrm extract-pdf doc.pdf --no-stream --output extracted.json

    # Verbose output
    bookwyrm extract-pdf document.pdf -v --output extracted.json

    # Auto-save with generated filename (no --output needed)
    bookwyrm extract-pdf my_document.pdf --start-page 5 --num-pages 3
    # Saves to: my_document_pages_5-7_extracted.json
    ```

    ## Output Format

    JSON file containing pages array with text elements, coordinates, and metadata
    """

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate input sources
    input_sources = [pdf_file, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        error_console.print(
            "[red]Error: Exactly one of file argument, --url, or --file must be provided[/red]"
        )
        raise typer.Exit(1)

    # Handle different input sources
    actual_file = pdf_file or file
    if actual_file:
        # Use local file
        if not actual_file.exists():
            error_console.print(f"[red]Error: File not found: {actual_file}[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Reading PDF file: {actual_file}[/blue]")

        try:
            # Read file as binary and base64 encode
            import base64

            pdf_bytes = actual_file.read_bytes()
            pdf_content = base64.b64encode(pdf_bytes).decode("ascii")
            console.print(f"[green]Loaded PDF file ({len(pdf_bytes)} bytes)[/green]")

            request = PDFExtractRequest(
                pdf_content=pdf_content,
                filename=actual_file.name,
                start_page=start_page,
                num_pages=num_pages,
            )
        except Exception as e:
            error_console.print(f"[red]Error reading PDF file: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Use URL
        console.print(f"[blue]Using PDF from URL: {url}[/blue]")
        request = PDFExtractRequest(
            pdf_url=url, start_page=start_page, num_pages=num_pages
        )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        console.print("[blue]Starting PDF extraction with streaming...[/blue]")
        if start_page or num_pages:
            page_info = f" (pages {start_page or 1}"
            if num_pages:
                page_info += f"-{(start_page or 1) + num_pages - 1}"
            page_info += ")"
            console.print(f"[dim]Processing{page_info}[/dim]")

        pages = []
        total_elements = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(
                "Processing PDF...", total=100
            )  # Start with 100 as placeholder

            for response in client.stream_extract_pdf(
                **request.model_dump(exclude_none=True)
            ):
                if isinstance(response, PDFStreamMetadata):
                    # Set up progress bar with known total
                    progress.update(
                        task,
                        total=response.total_pages,
                        completed=0,
                        description=f"Processing {response.total_pages} pages (doc pages {response.start_page}-{response.start_page + response.total_pages - 1})",
                    )
                    if state.verbose:
                        console.print(
                            f"[dim]Document has {response.total_pages_in_document} total pages, "
                            f"processing {response.total_pages} pages starting from page {response.start_page}[/dim]"
                        )
                elif isinstance(response, PDFStreamPageResponse):
                    pages.append(response.page_data)
                    total_elements += len(response.page_data.text_blocks)

                    progress.update(
                        task,
                        completed=response.current_page,
                        description=f"Page {response.document_page} - {len(response.page_data.text_blocks)} elements found",
                    )

                    if state.verbose:
                        console.print(
                            f"[green]Page {response.document_page}: {len(response.page_data.text_blocks)} text elements[/green]"
                        )
                elif isinstance(response, PDFStreamPageError):
                    progress.update(
                        task,
                        completed=response.current_page,
                        description=f"Error on page {response.document_page}",
                    )
                    console.print(
                        f"[red]Error on page {response.document_page}: {response.error}[/red]"
                    )
                elif isinstance(response, PDFStreamComplete):
                    progress.update(
                        task,
                        completed=response.current_page,
                        description="Complete!",
                    )
                    console.print("[green]✓ PDF extraction complete![/green]")
                elif isinstance(response, PDFStreamError):
                    console.print(f"[red]Extraction error: {response.error}[/red]")
                    raise typer.Exit(1)

        # Display summary
        console.print(f"[green]Extracted {len(pages)} pages[/green]")
        console.print(f"[green]Found {total_elements} text elements[/green]")

        # Display detailed results if verbose
        if state.verbose and pages:
            table = Table(title="Extracted Text Elements")
            table.add_column("Page", justify="right", style="cyan", no_wrap=True)
            table.add_column("Position", justify="center", style="magenta")
            table.add_column("Confidence", justify="center", style="yellow")
            table.add_column("Text", style="green")

            # Show first 20 elements across all pages
            element_count = 0
            for page in pages:
                for element in page.text_blocks:
                    if element_count >= 20:
                        break

                    position = f"({element.coordinates.x1:.0f},{element.coordinates.y1:.0f})-({element.coordinates.x2:.0f},{element.coordinates.y2:.0f})"
                    confidence = f"{element.confidence:.2f}"
                    text_preview = (
                        element.text[:60] + "..."
                        if len(element.text) > 60
                        else element.text
                    )

                    table.add_row(
                        str(page.page_number), position, confidence, text_preview
                    )
                    element_count += 1

                if element_count >= 20:
                    break

            if total_elements > 20:
                table.add_row("...", "...", "...", "...")

            console.print(table)

        # Save to output file (specified or default)
        if output or pages:  # Save if output specified OR if we have pages to save
            if not output:
                # Generate default filename
                if actual_file:
                    base_name = actual_file.stem
                elif url:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    base_name = (
                        parsed.path.split("/")[-1].replace(".pdf", "")
                        if parsed.path
                        else "pdf_extract"
                    )
                    if not base_name or base_name == "":
                        base_name = "pdf_extract"
                else:
                    base_name = "pdf_extract"

                # Add page range to filename if specified
                if start_page or num_pages:
                    page_suffix = f"_pages_{start_page or 1}"
                    if num_pages:
                        page_suffix += f"-{(start_page or 1) + num_pages - 1}"
                    base_name += page_suffix

                output = Path(f"{base_name}_extracted.json")
                console.print(
                    f"[dim]No output file specified, saving to: {output}[/dim]"
                )

            try:
                output_data = {
                    "pages": [page.model_dump() for page in pages],
                    "total_pages": len(pages),
                    "extraction_method": "paddleocr",
                    "source": {
                        "file": str(actual_file) if actual_file else None,
                        "url": url,
                        "start_page": start_page,
                        "num_pages": num_pages,
                    },
                }

                output.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
                console.print(f"\n[green]Extraction results saved to: {output}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving to {output}: {e}[/red]")

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
