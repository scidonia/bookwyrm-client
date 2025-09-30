"""Command-line interface for BookWyrm client."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Annotated

import typer
from rich.console import Console
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

from .client import BookWyrmClient, BookWyrmAPIError
from .models import (
    CitationRequest,
    TextChunk,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
    SummarizeRequest,
    Phrase,
    SummarizeProgressUpdate,
    SummaryResponse,
    SummarizeErrorResponse,
    ProcessTextRequest,
    ResponseFormat,
    PhraseProgressUpdate,
    PhraseResult,
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


def load_chunks_from_jsonl(file_path: Path) -> List[TextChunk]:
    """Load text chunks from a JSONL file."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunk = TextChunk(
                        text=data["text"],
                        start_char=data["start_char"],
                        end_char=data["end_char"],
                    )
                    chunks.append(chunk)
                except (json.JSONDecodeError, KeyError) as e:
                    console.print(f"[red]Error parsing line {line_num}: {e}[/red]")
                    sys.exit(1)
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)

    return chunks


def load_phrases_from_jsonl(file_path: Path) -> List[Phrase]:
    """Load phrases from a JSONL file."""
    phrases = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    phrase = Phrase(
                        text=data["text"],
                        start_char=data.get("start_char"),
                        end_char=data.get("end_char"),
                    )
                    phrases.append(phrase)
                except (json.JSONDecodeError, KeyError) as e:
                    console.print(f"[red]Error parsing line {line_num}: {e}[/red]")
                    sys.exit(1)
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        sys.exit(1)

    return phrases


def load_jsonl_content(file_path: Path) -> str:
    """Load raw content from a JSONL file."""
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
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


def display_citations_table(citations):
    """Display citations in a rich table."""
    if not citations:
        console.print("[yellow]No citations found.[/yellow]")
        return

    table = Table(title="Found Citations")
    table.add_column("Quality", justify="center", style="cyan", no_wrap=True)
    table.add_column("Chunks", justify="center", style="magenta")
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

        table.add_row(
            quality_text,
            chunk_range,
            citation.text[:100] + "..." if len(citation.text) > 100 else citation.text,
            (
                citation.reasoning[:150] + "..."
                if len(citation.reasoning) > 150
                else citation.reasoning
            ),
        )

    console.print(table)


def display_verbose_citation(citation):
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
[bold]Chunks:[/bold] {chunk_range}
[bold]Text:[/bold] {citation.text}
[bold]Reasoning:[/bold] {citation.reasoning}"""

    console.print(
        Panel(panel_content, title="Citation Found", border_style=quality_color)
    )


app = typer.Typer(help="BookWyrm Client CLI - Find citations in text using AI.")


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
        console.print("[red]Error: No API key provided![/red]")
        console.print(
            "[yellow]Please provide an API key using one of these methods:[/yellow]"
        )
        console.print(
            "  1. Set environment variable: [cyan]export BOOKWYRM_API_KEY='your-api-key'[/cyan]"
        )
        console.print("  2. Use CLI option: [cyan]--api-key your-api-key[/cyan]")
        console.print(
            "\n[dim]You can get an API key from https://api.bookwyrm.ai[/dim]"
        )
        raise typer.Exit(1)


@app.command()
def cite(
    question: Annotated[str, typer.Argument(help="Question to find citations for")],
    jsonl_input: Annotated[
        Optional[str], typer.Argument(help="JSONL file path")
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(help="URL to JSONL file (alternative to providing file path)"),
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
    start: Annotated[int, typer.Option(help="Start chunk index")] = 0,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit number of chunks to process")
    ] = None,
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk")] = 1000,
    stream: Annotated[
        bool, typer.Option("--stream/--no-stream", help="Use streaming API")
    ] = True,
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
):
    """Find citations for a question in text chunks from file or URL."""

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate input sources
    input_sources = [jsonl_input, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
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

        request = CitationRequest(
            chunks=chunks,
            question=question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens,
        )
    else:
        # Use URL
        console.print(f"[blue]Using JSONL from URL: {url}[/blue]")
        request = CitationRequest(
            jsonl_url=url,
            question=question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens,
        )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        if stream:
            console.print(f"[blue]Streaming citations for: {question}[/blue]")
            if url:
                console.print(f"[dim]Source: {url}[/dim]")
            if output:
                console.print(
                    f"[dim]Streaming citations to {output} (JSONL format)[/dim]"
                )

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

                for response in client.stream_citations(request):
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
                            display_verbose_citation(response.citation)
                        else:
                            console.print(
                                f"[green]Found citation (quality {response.citation.quality}/4)[/green]"
                            )
                        # Immediately append to output file if specified
                        if output:
                            append_citation_to_jsonl(response.citation, output)
                    elif isinstance(response, CitationSummaryResponse):
                        progress.update(
                            task,
                            completed=response.chunks_processed,
                            description="Complete!",
                        )
                        console.print(
                            f"[blue]Processing complete: {response.total_citations} citations found[/blue]"
                        )
                        if response.usage:
                            cost_str = f"${response.usage.estimated_cost:.4f}" if response.usage.estimated_cost is not None else "N/A"
                            console.print(
                                f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: {cost_str}[/dim]"
                            )
                    elif isinstance(response, CitationErrorResponse):
                        console.print(f"[red]Error: {response.error}[/red]")

            display_citations_table(citations)

            if output:
                console.print(f"[green]Citations streamed to {output}[/green]")

        else:
            console.print(f"[blue]Getting citations for: {question}[/blue]")
            if url:
                console.print(f"[dim]Source: {url}[/dim]")

            with console.status("[bold green]Processing..."):
                response = client.get_citations(request)

            console.print(f"[green]Found {response.total_citations} citations[/green]")
            if response.usage:
                cost_str = f"${response.usage.estimated_cost:.4f}" if response.usage.estimated_cost is not None else "N/A"
                console.print(
                    f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: {cost_str}[/dim]"
                )

            display_citations_table(response.citations)

            if output:
                save_citations_to_json(response.citations, output)

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
def summarize(
    jsonl_file: Annotated[
        Path, typer.Argument(help="JSONL file containing phrases", exists=True)
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output file for summary (JSON format)"),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option(help="Maximum tokens per chunk (max: 131,072)")
    ] = 10000,
    debug: Annotated[bool, typer.Option(help="Include intermediate summaries")] = False,
    stream: Annotated[
        bool, typer.Option("--stream/--no-stream", help="Use streaming API")
    ] = True,
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
    """Summarize a JSONL file containing phrases."""

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

    console.print(f"[blue]Loading JSONL file: {jsonl_file}[/blue]")
    content = load_jsonl_content(jsonl_file)

    request = SummarizeRequest(
        content=content,
        max_tokens=max_tokens,
        debug=debug,
    )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        if stream:
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

                for response in client.stream_summarize(request):
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
            if state.verbose or debug:
                console.print(
                    f"[dim]Total tokens processed: {final_result.total_tokens}[/dim]"
                )
                console.print(
                    f"[dim]Subsummaries created: {final_result.subsummary_count}[/dim]"
                )
                console.print(f"[dim]Levels used: {final_result.levels_used}[/dim]")

            # Show intermediate summaries if debug mode
            if debug and final_result.intermediate_summaries:
                console.print("\n[bold]Intermediate Summaries by Level:[/bold]")
                for level, summaries in enumerate(
                    final_result.intermediate_summaries, 1
                ):
                    console.print(
                        f"\n[blue]Level {level} ({len(summaries)} summaries):[/blue]"
                    )
                    for i, summary in enumerate(summaries, 1):
                        console.print(f"[dim]{i}.[/dim] {summary}")

            console.print("\n[bold]Final Summary:[/bold]")
            console.print(final_result.summary)

            # Save to output file if specified
            if output:
                try:
                    output_data = {
                        "summary": final_result.summary,
                        "subsummary_count": final_result.subsummary_count,
                        "levels_used": final_result.levels_used,
                        "total_tokens": final_result.total_tokens,
                        "source_file": str(jsonl_file),
                        "max_tokens": max_tokens,
                        "intermediate_summaries": (
                            final_result.intermediate_summaries if debug else None
                        ),
                    }

                    output.write_text(
                        json.dumps(output_data, indent=2), encoding="utf-8"
                    )
                    console.print(f"\n[green]Summary saved to: {output}[/green]")
                except Exception as e:
                    console.print(f"[red]Error saving to {output}: {e}[/red]")

        else:
            console.print("[blue]Starting summarization...[/blue]")

            with console.status("[bold green]Processing..."):
                response = client.summarize(request)

            console.print("[green]✓ Summarization complete![/green]")

            if state.verbose or debug:
                console.print(
                    f"[dim]Total tokens processed: {response.total_tokens}[/dim]"
                )
                console.print(
                    f"[dim]Subsummaries created: {response.subsummary_count}[/dim]"
                )
                console.print(f"[dim]Levels used: {response.levels_used}[/dim]")

            # Show intermediate summaries if debug mode
            if debug and response.intermediate_summaries:
                console.print("\n[bold]Intermediate Summaries by Level:[/bold]")
                for level, summaries in enumerate(response.intermediate_summaries, 1):
                    console.print(
                        f"\n[blue]Level {level} ({len(summaries)} summaries):[/blue]"
                    )
                    for i, summary in enumerate(summaries, 1):
                        console.print(f"[dim]{i}.[/dim] {summary}")

            console.print("\n[bold]Final Summary:[/bold]")
            console.print(response.summary)

            # Save to output file if specified
            if output:
                try:
                    output_data = {
                        "summary": response.summary,
                        "subsummary_count": response.subsummary_count,
                        "levels_used": response.levels_used,
                        "total_tokens": response.total_tokens,
                        "source_file": str(jsonl_file),
                        "max_tokens": max_tokens,
                        "intermediate_summaries": (
                            response.intermediate_summaries if debug else None
                        ),
                    }

                    output.write_text(
                        json.dumps(output_data, indent=2), encoding="utf-8"
                    )
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
    input_text: Annotated[Optional[str], typer.Argument(help="Text to process")] = None,
    url: Annotated[
        Optional[str],
        typer.Option(
            help="URL to fetch text from (alternative to providing text directly)"
        ),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option("--file", help="File to read text from", exists=True),
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
    format: Annotated[str, typer.Option(help="Response format")] = "with_offsets",
    spacy_model: Annotated[
        str, typer.Option(help="SpaCy model to use for processing")
    ] = "en_core_web_sm",
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
    """Process text using phrasal analysis to extract phrases or chunks."""

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate format choice
    if format not in ["text_only", "with_offsets"]:
        console.print(
            f"[red]Error: format must be 'text_only' or 'with_offsets', got '{format}'[/red]"
        )
        raise typer.Exit(1)

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
            console.print(f"[red]Error reading file {file}: {e}[/red]")
            raise typer.Exit(1)
    elif url:
        text = None  # Will be handled by the API
        console.print(f"[blue]Processing text from URL: {url}[/blue]")
    else:
        text = input_text
        console.print(f"[blue]Processing provided text ({len(text)} characters)[/blue]")

    # Create request
    request = ProcessTextRequest(
        text=text,
        text_url=url,
        chunk_size=chunk_size,
        response_format=ResponseFormat(format),
        spacy_model=spacy_model,
    )

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

            for response in client.process_text(request):
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
                elif isinstance(response, PhraseResult):
                    phrases.append(response)

                    if state.verbose:
                        if response.start_char is not None:
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
                if format == "with_offsets" and phrase.start_char is not None:
                    row.append(f"{phrase.start_char}-{phrase.end_char}")
                elif format == "with_offsets":
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
    file_path: Annotated[
        Optional[Path], typer.Argument(help="File to classify")
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(
            help="URL to classify"
        ),
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
    """Classify file or URL to determine its type and format."""

    # Set global state
    state.base_url = get_base_url(base_url)
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate input sources - allow positional file argument or --file or --url
    input_sources = [file_path, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
            "[red]Error: Exactly one of file argument, --url, or --file must be provided[/red]"
        )
        raise typer.Exit(1)

    # Check if positional file exists
    if file_path and not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    # Get content from the appropriate source
    actual_file = file_path or file
    if actual_file:
        try:
            # Always read as binary and base64 encode for multipart upload
            import base64
            binary_content = actual_file.read_bytes()
            content = base64.b64encode(binary_content).decode("ascii")
            console.print(
                f"[blue]Classifying file: {actual_file} ({len(binary_content)} bytes)[/blue]"
            )

            # Use the actual filename if no hint provided
            effective_filename = filename or actual_file.name
        except Exception as e:
            console.print(f"[red]Error reading file {actual_file}: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Handle URL - we'll fetch it and send as multipart
        console.print(f"[blue]Classifying URL resource: {url}[/blue]")
        try:
            import httpx
            import base64
            
            with httpx.Client() as client:
                response = client.get(url)
                response.raise_for_status()
                content = base64.b64encode(response.content).decode("ascii")
                
            console.print(f"[blue]Downloaded {len(response.content)} bytes from URL[/blue]")
            
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
            console.print(f"[red]Error fetching URL {url}: {e}[/red]")
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
            response = client.classify(request)

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
                        "file": str(actual_file) if actual_file else None,
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
        Optional[Path], typer.Argument(help="PDF file to extract from")
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option(
            help="PDF URL to extract from (alternative to providing file path)"
        ),
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
    stream: Annotated[
        bool, typer.Option("--stream/--no-stream", help="Use streaming API with progress")
    ] = True,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Base URL of the PDF extraction API (overrides BOOKWYRM_PDF_API_URL env var)"
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
    """Extract structured data from a PDF file using OCR."""

    # Set global state
    # Use PDF-specific base URL if available, otherwise fall back to main API
    pdf_base_url = base_url or os.getenv("BOOKWYRM_PDF_API_URL") or get_base_url(None)
    state.base_url = pdf_base_url
    state.api_key = get_api_key(api_key)
    state.verbose = verbose

    # Validate API key before proceeding
    validate_api_key(state.api_key)

    # Validate input sources
    input_sources = [pdf_file, url, file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
            "[red]Error: Exactly one of file argument, --url, or --file must be provided[/red]"
        )
        raise typer.Exit(1)

    # Handle different input sources
    actual_file = pdf_file or file
    if actual_file:
        # Use local file
        if not actual_file.exists():
            console.print(f"[red]Error: File not found: {actual_file}[/red]")
            raise typer.Exit(1)
            
        console.print(f"[blue]Reading PDF file: {actual_file}[/blue]")
        
        try:
            # Read file as binary and base64 encode
            import base64
            pdf_bytes = actual_file.read_bytes()
            pdf_content = base64.b64encode(pdf_bytes).decode('ascii')
            console.print(f"[green]Loaded PDF file ({len(pdf_bytes)} bytes)[/green]")
            
            request = PDFExtractRequest(
                pdf_content=pdf_content,
                filename=actual_file.name,
                start_page=start_page,
                num_pages=num_pages
            )
        except Exception as e:
            console.print(f"[red]Error reading PDF file: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Use URL
        console.print(f"[blue]Using PDF from URL: {url}[/blue]")
        request = PDFExtractRequest(
            pdf_url=url,
            start_page=start_page,
            num_pages=num_pages
        )

    client = BookWyrmClient(base_url=state.base_url, api_key=state.api_key)

    try:
        if stream:
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

                task = progress.add_task("Processing PDF...", total=100)  # Start with 100 as placeholder

                for response in client.stream_extract_pdf(request):
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
                        console.print(f"[red]Error on page {response.document_page}: {response.error}[/red]")
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
                            element.text[:60] + "..." if len(element.text) > 60 else element.text
                        )
                        
                        table.add_row(
                            str(page.page_number),
                            position,
                            confidence,
                            text_preview
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
                        base_name = parsed.path.split("/")[-1].replace(".pdf", "") if parsed.path else "pdf_extract"
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
                    console.print(f"[dim]No output file specified, saving to: {output}[/dim]")

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

        else:
            console.print("[blue]Starting PDF extraction...[/blue]")
            if start_page or num_pages:
                page_info = f" (pages {start_page or 1}"
                if num_pages:
                    page_info += f"-{(start_page or 1) + num_pages - 1}"
                page_info += ")"
                console.print(f"[dim]Processing{page_info}[/dim]")

            with console.status("[bold green]Processing PDF..."):
                response = client.extract_pdf(request)

            console.print("[green]✓ PDF extraction complete![/green]")

            # Display summary
            console.print(f"[green]Extracted {response.total_pages} pages[/green]")
            
            total_elements = sum(len(page.text_blocks) for page in response.pages)
            console.print(f"[green]Found {total_elements} text elements[/green]")
            
            if response.processing_time:
                console.print(f"[dim]Processing time: {response.processing_time:.2f}s[/dim]")

            # Display detailed results if verbose
            if state.verbose and response.pages:
                table = Table(title="Extracted Text Elements")
                table.add_column("Page", justify="right", style="cyan", no_wrap=True)
                table.add_column("Position", justify="center", style="magenta")
                table.add_column("Confidence", justify="center", style="yellow")
                table.add_column("Text", style="green")

                # Show first 20 elements across all pages
                element_count = 0
                for page in response.pages:
                    for element in page.text_blocks:
                        if element_count >= 20:
                            break
                        
                        position = f"({element.coordinates.x1:.0f},{element.coordinates.y1:.0f})-({element.coordinates.x2:.0f},{element.coordinates.y2:.0f})"
                        confidence = f"{element.confidence:.2f}"
                        text_preview = (
                            element.text[:60] + "..." if len(element.text) > 60 else element.text
                        )
                        
                        table.add_row(
                            str(page.page_number),
                            position,
                            confidence,
                            text_preview
                        )
                        element_count += 1
                    
                    if element_count >= 20:
                        break
                
                if total_elements > 20:
                    table.add_row("...", "...", "...", "...")
                
                console.print(table)

            # Save to output file (specified or default)
            if output or response.pages:  # Save if output specified OR if we have pages to save
                if not output:
                    # Generate default filename
                    if actual_file:
                        base_name = actual_file.stem
                    elif url:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        base_name = parsed.path.split("/")[-1].replace(".pdf", "") if parsed.path else "pdf_extract"
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
                    console.print(f"[dim]No output file specified, saving to: {output}[/dim]")

                try:
                    output_data = {
                        "pages": [page.model_dump() for page in response.pages],
                        "total_pages": response.total_pages,
                        "extraction_method": response.extraction_method,
                        "processing_time": response.processing_time,
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
