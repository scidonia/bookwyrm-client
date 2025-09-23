"""Command-line interface for BookWyrm client."""

import json
import os
import sys
from pathlib import Path
from typing import Optional, List

import click
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


@click.group()
@click.option(
    "--base-url", default="http://localhost:8000", help="Base URL of the BookWyrm API"
)
@click.option(
    "--api-key", help="API key for authentication (overrides BOOKWYRM_API_KEY env var)"
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed citation information"
)
@click.pass_context
def cli(ctx, base_url: str, api_key: Optional[str], verbose: bool):
    """BookWyrm Client CLI - Find citations in text using AI."""
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    # Use CLI flag if provided, otherwise fall back to environment variable
    ctx.obj["api_key"] = (
        api_key if api_key is not None else os.getenv("BOOKWYRM_API_KEY")
    )
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("jsonl_file", type=click.Path(exists=True, path_type=Path))
@click.argument("question")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for citations (JSON for non-streaming, JSONL for streaming)",
)
@click.option("--start", type=int, default=0, help="Start chunk index")
@click.option("--limit", type=int, help="Limit number of chunks to process")
@click.option("--max-tokens", type=int, default=1000, help="Maximum tokens per chunk")
@click.option("--stream/--no-stream", default=True, help="Use streaming API")
@click.pass_context
def cite(
    ctx,
    jsonl_file: Path,
    question: str,
    output: Optional[Path],
    start: int,
    limit: Optional[int],
    max_tokens: int,
    stream: bool,
):
    """Find citations for a question in a JSONL file of text chunks."""

    console.print(f"[blue]Loading chunks from {jsonl_file}...[/blue]")
    chunks = load_chunks_from_jsonl(jsonl_file)
    console.print(f"[green]Loaded {len(chunks)} chunks[/green]")

    request = CitationRequest(
        chunks=chunks,
        question=question,
        start=start,
        limit=limit,
        max_tokens_per_chunk=max_tokens,
        api_key=ctx.obj["api_key"],
    )

    client = BookWyrmClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])

    try:
        if stream:
            console.print(f"[blue]Streaming citations for: {question}[/blue]")
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

                task = progress.add_task("Processing chunks...", total=len(chunks))

                for response in client.stream_citations(request):
                    if isinstance(response, CitationProgressUpdate):
                        progress.update(
                            task,
                            completed=response.chunks_processed,
                            description=response.message,
                        )
                    elif isinstance(response, CitationStreamResponse):
                        citations.append(response.citation)
                        if ctx.obj["verbose"]:
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
                            console.print(
                                f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: ${response.usage.estimated_cost:.4f}[/dim]"
                            )
                    elif isinstance(response, CitationErrorResponse):
                        console.print(f"[red]Error: {response.error}[/red]")

            display_citations_table(citations)

            if output:
                console.print(f"[green]Citations streamed to {output}[/green]")

        else:
            console.print(f"[blue]Getting citations for: {question}[/blue]")

            with console.status("[bold green]Processing..."):
                response = client.get_citations(request)

            console.print(f"[green]Found {response.total_citations} citations[/green]")
            if response.usage:
                console.print(
                    f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: ${response.usage.estimated_cost:.4f}[/dim]"
                )

            display_citations_table(response.citations)

            if output:
                save_citations_to_json(response.citations, output)

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        client.close()


@cli.command()
@click.argument("url")
@click.argument("question")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for citations (JSON for non-streaming, JSONL for streaming)",
)
@click.option("--start", type=int, default=0, help="Start chunk index")
@click.option("--limit", type=int, help="Limit number of chunks to process")
@click.option("--max-tokens", type=int, default=1000, help="Maximum tokens per chunk")
@click.option("--stream/--no-stream", default=True, help="Use streaming API")
@click.pass_context
def cite_url(
    ctx,
    url: str,
    question: str,
    output: Optional[Path],
    start: int,
    limit: Optional[int],
    max_tokens: int,
    stream: bool,
):
    """Find citations for a question using a JSONL URL."""

    request = CitationRequest(
        jsonl_url=url,
        question=question,
        start=start,
        limit=limit,
        max_tokens_per_chunk=max_tokens,
        api_key=ctx.obj["api_key"],
    )

    client = BookWyrmClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])

    try:
        if stream:
            console.print(f"[blue]Streaming citations for: {question}[/blue]")
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

                task = progress.add_task("Processing chunks...", total=None)

                for response in client.stream_citations(request):
                    if isinstance(response, CitationProgressUpdate):
                        if progress.tasks[task].total is None:
                            progress.update(task, total=response.total_chunks)
                        progress.update(
                            task,
                            completed=response.chunks_processed,
                            description=response.message,
                        )
                    elif isinstance(response, CitationStreamResponse):
                        citations.append(response.citation)
                        if ctx.obj["verbose"]:
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
                            console.print(
                                f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: ${response.usage.estimated_cost:.4f}[/dim]"
                            )
                    elif isinstance(response, CitationErrorResponse):
                        console.print(f"[red]Error: {response.error}[/red]")

            display_citations_table(citations)

            if output:
                console.print(f"[green]Citations streamed to {output}[/green]")

        else:
            console.print(f"[blue]Getting citations for: {question}[/blue]")
            console.print(f"[dim]Source: {url}[/dim]")

            with console.status("[bold green]Processing..."):
                response = client.get_citations(request)

            console.print(f"[green]Found {response.total_citations} citations[/green]")
            if response.usage:
                console.print(
                    f"[dim]Tokens processed: {response.usage.tokens_processed}, Cost: ${response.usage.estimated_cost:.4f}[/dim]"
                )

            display_citations_table(response.citations)

            if output:
                save_citations_to_json(response.citations, output)

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        client.close()


@cli.command()
@click.argument("jsonl_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for summary (JSON format)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=10000,
    help="Maximum tokens per chunk (max: 131,072)",
)
@click.option("--debug", is_flag=True, help="Include intermediate summaries")
@click.option("--stream/--no-stream", default=True, help="Use streaming API")
@click.pass_context
def summarize(
    ctx,
    jsonl_file: Path,
    output: Optional[Path],
    max_tokens: int,
    debug: bool,
    stream: bool,
):
    """Summarize a JSONL file containing phrases."""

    # Validate max_tokens
    if max_tokens > 131072:
        console.print(
            f"[red]Error: max_tokens cannot exceed 131,072 (got {max_tokens})[/red]"
        )
        sys.exit(1)
    if max_tokens < 1:
        console.print(
            f"[red]Error: max_tokens must be at least 1 (got {max_tokens})[/red]"
        )
        sys.exit(1)

    console.print(f"[blue]Loading JSONL file: {jsonl_file}[/blue]")
    content = load_jsonl_content(jsonl_file)

    request = SummarizeRequest(
        content=content,
        max_tokens=max_tokens,
        debug=debug,
        api_key=ctx.obj["api_key"],
    )

    client = BookWyrmClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])

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
            if ctx.obj["verbose"] or debug:
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

            if ctx.obj["verbose"] or debug:
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
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        client.close()


@cli.command()
@click.argument("input_text", required=False)
@click.option(
    "--url", help="URL to fetch text from (alternative to providing text directly)"
)
@click.option(
    "--file",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    help="File to read text from (alternative to providing text directly)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for phrases (JSONL format)",
)
@click.option(
    "--chunk-size",
    type=int,
    help="Target size for each chunk (if not specified, returns phrases individually)",
)
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["text_only", "with_offsets"]),
    default="with_offsets",
    help="Response format",
)
@click.option(
    "--spacy-model",
    default="en_core_web_sm",
    help="SpaCy model to use for processing",
)
@click.pass_context
def phrasal(
    ctx,
    input_text: Optional[str],
    url: Optional[str],
    input_file: Optional[Path],
    output: Optional[Path],
    chunk_size: Optional[int],
    response_format: str,
    spacy_model: str,
):
    """Process text using phrasal analysis to extract phrases or chunks."""

    # Validate input sources
    input_sources = [input_text, url, input_file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
            "[red]Error: Exactly one of text argument, --url, or --file must be provided[/red]"
        )
        sys.exit(1)

    # Get text from the appropriate source
    if input_file:
        try:
            text = input_file.read_text(encoding="utf-8")
            console.print(
                f"[blue]Loaded text from {input_file} ({len(text)} characters)[/blue]"
            )
        except Exception as e:
            console.print(f"[red]Error reading file {input_file}: {e}[/red]")
            sys.exit(1)
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
        response_format=ResponseFormat(response_format),
        spacy_model=spacy_model,
    )

    client = BookWyrmClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])

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
                    if ctx.obj["verbose"]:
                        console.print(
                            f"[dim]Processed {response.phrases_processed} phrases, "
                            f"created {response.chunks_created} chunks[/dim]"
                        )
                elif isinstance(response, PhraseResult):
                    phrases.append(response)

                    if ctx.obj["verbose"]:
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
        if phrases and not ctx.obj["verbose"]:
            table = Table(title="Phrasal Processing Results")
            table.add_column("Index", justify="right", style="cyan", no_wrap=True)
            if response_format == "with_offsets":
                table.add_column("Position", justify="center", style="magenta")
            table.add_column("Text", style="green")

            for i, phrase in enumerate(phrases[:10]):  # Show first 10
                row = [str(i + 1)]
                if response_format == "with_offsets" and phrase.start_char is not None:
                    row.append(f"{phrase.start_char}-{phrase.end_char}")
                elif response_format == "with_offsets":
                    row.append("N/A")

                text_preview = (
                    phrase.text[:80] + "..." if len(phrase.text) > 80 else phrase.text
                )
                row.append(text_preview)
                table.add_row(*row)

            if len(phrases) > 10:
                table.add_row(
                    "...", "..." if response_format == "with_offsets" else "", "..."
                )

            console.print(table)

        if output:
            console.print(f"[green]Results saved to {output}[/green]")

    except BookWyrmAPIError as e:
        console.print(f"[red]API Error: {e}[/red]")
        if e.status_code:
            console.print(f"[red]Status Code: {e.status_code}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        client.close()


@cli.command()
@click.argument("input_content", required=False)
@click.option(
    "--url", help="URL to classify (alternative to providing content directly)"
)
@click.option(
    "--file",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    help="File to classify (alternative to providing content directly)",
)
@click.option("--filename", help="Optional filename hint for classification")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file for classification results (JSON format)",
)
@click.pass_context
def classify(
    ctx,
    input_content: Optional[str],
    url: Optional[str],
    input_file: Optional[Path],
    filename: Optional[str],
    output: Optional[Path],
):
    """Classify content, URL, or file to determine its type and format."""

    # Validate input sources
    input_sources = [input_content, url, input_file]
    provided_sources = [s for s in input_sources if s is not None]

    if len(provided_sources) != 1:
        console.print(
            "[red]Error: Exactly one of content argument, --url, or --file must be provided[/red]"
        )
        sys.exit(1)

    # Get content from the appropriate source
    content_encoding = None
    if input_file:
        try:
            # Try to read as text first, fall back to binary
            try:
                content = input_file.read_text(encoding="utf-8")
                console.print(
                    f"[blue]Classifying text file: {input_file} ({len(content)} characters)[/blue]"
                )
            except UnicodeDecodeError:
                # Read as binary and base64 encode for transmission
                import base64

                binary_content = input_file.read_bytes()
                content = base64.b64encode(binary_content).decode("ascii")
                content_encoding = "base64"
                console.print(
                    f"[blue]Classifying binary file: {input_file} ({len(binary_content)} bytes, base64 encoded)[/blue]"
                )

            # Use the actual filename if no hint provided
            if not filename:
                filename = input_file.name
        except Exception as e:
            console.print(f"[red]Error reading file {input_file}: {e}[/red]")
            sys.exit(1)
    elif url:
        content = None  # Will be handled by the API
        console.print(f"[blue]Classifying URL resource: {url}[/blue]")
        # Extract filename hint from URL if not provided
        if not filename:
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
            if parsed_url.path:
                filename = parsed_url.path.split("/")[-1]
                if filename:
                    console.print(
                        f"[dim]Using filename hint from URL: {filename}[/dim]"
                    )
    else:
        content = input_content
        console.print(
            f"[blue]Classifying provided content ({len(content)} characters)[/blue]"
        )

    # Create request
    request = ClassifyRequest(
        content=content,
        url=url,
        filename=filename,
        content_encoding=content_encoding,
    )

    client = BookWyrmClient(base_url=ctx.obj["base_url"], api_key=ctx.obj["api_key"])

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
                        "file": str(input_file) if input_file else None,
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
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    cli()
