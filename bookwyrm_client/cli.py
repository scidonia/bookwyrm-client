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
    
    console.print(Panel(panel_content, title="Citation Found", border_style=quality_color))


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
    "--output",
    "-o",
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
    "--output",
    "-o",
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


if __name__ == "__main__":
    cli()
