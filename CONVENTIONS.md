# Python Development Conventions

This document outlines the coding standards and best practices for Python development in this project.

## Type Annotations

**Use type annotations everywhere.** All functions, methods, variables, and class attributes should have proper type hints.

```python
# Good
def process_text(content: str, max_length: int = 100) -> List[str]:
    chunks: List[str] = []
    # ... implementation
    return chunks

# Bad
def process_text(content, max_length=100):
    chunks = []
    # ... implementation
    return chunks
```

### Key Rules:

- All function parameters and return types must be annotated
- Class attributes should be annotated using `typing` or built-in generics
- Use `Optional[T]` for nullable values
- Import types from `typing` when needed: `List`, `Dict`, `Optional`, `Union`, etc.

## Pydantic Models

**Use Pydantic models whenever possible** for data validation, serialization, and API contracts.

### Model Documentation

Annotate all Pydantic fields liberally with `Field()` descriptions that explain:

- The purpose of the field
- Expected format or constraints
- Relationship to other fields
- Business logic context

```python
# Good
class CitationRequest(BaseModel):
    """Request model for citation processing.

    Use this model to request citations for a question from text chunks.
    Provide exactly one of: chunks, jsonl_content, or jsonl_url.
    """

    question: str = Field(
        ...,
        description="The question to find citations for. Must be non-empty and specific enough for meaningful results."
    )
    max_tokens_per_chunk: int = Field(
        1000,
        description="Maximum tokens per chunk for processing. Higher values may improve context but increase cost."
    )
    start: Optional[int] = Field(
        0,
        description="Starting chunk index (0-based). Use for pagination or resuming interrupted processing."
    )

# Bad
class CitationRequest(BaseModel):
    question: str
    max_tokens_per_chunk: int = 1000
    start: Optional[int] = 0
```

### Model Validation

Use `@model_validator` for complex validation logic that involves multiple fields:

```python
@model_validator(mode="after")
def validate_input_source(self):
    """Validate that exactly one input source is provided."""
    sources = [self.chunks, self.jsonl_content, self.jsonl_url]
    provided_sources = [s for s in sources if s is not None]

    if len(provided_sources) != 1:
        raise ValueError(
            "Exactly one of 'chunks', 'jsonl_content', or 'jsonl_url' must be provided"
        )
    return self
```

## Error Handling

**Do not simply try/catch and throw away errors** unless you have specific instructions to suppress them.

### Good Error Handling:

```python
# Good - Specific exception handling with meaningful messages
try:
    result = api_call()
except requests.HTTPError as e:
    if e.response.status_code == 401:
        raise BookWyrmAPIError("Invalid API key provided") from e
    elif e.response.status_code == 429:
        raise BookWyrmAPIError("Rate limit exceeded, please try again later") from e
    else:
        raise BookWyrmAPIError(f"HTTP error {e.response.status_code}: {e}") from e
except requests.ConnectionError as e:
    raise BookWyrmAPIError("Failed to connect to BookWyrm API") from e

# Good - Re-raising with context
try:
    data = json.loads(response_text)
except json.JSONDecodeError as e:
    error_console.print(f"[red]Error parsing JSON response: {e}[/red]")
    raise typer.Exit(1)
```

### Bad Error Handling:

```python
# Bad - Swallowing exceptions
try:
    result = api_call()
except:
    pass

# Bad - Generic exception handling without context
try:
    result = api_call()
except Exception as e:
    print(f"Error: {e}")
```

### Error Handling Guidelines:

- Catch specific exceptions when possible
- Provide meaningful error messages to users
- Use `raise ... from e` to preserve the original exception chain
- Log errors appropriately (use `error_console` for CLI)
- Exit gracefully with appropriate exit codes

## Function Length and Organization

**If code starts getting longer than a few hundred lines, break it into functions** that implement specific tasks.

### Function Design Principles:

- Each function should have a single, clear responsibility
- Functions should be easily testable in isolation
- Use descriptive function names that explain what they do
- Keep functions under ~50-100 lines when possible
- Extract complex logic into helper functions

```python
# Good - Broken into focused functions
def load_chunks_from_jsonl(file_path: Path) -> List[TextSpan]:
    """Load text chunks from a JSONL file."""
    chunks = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                chunk = _parse_jsonl_line(line, line_num)
                chunks.append(chunk)
    except FileNotFoundError:
        error_console.print(f"[red]File not found: {file_path}[/red]")
        sys.exit(1)
    return chunks

def _parse_jsonl_line(line: str, line_num: int) -> TextSpan:
    """Parse a single JSONL line into a TextSpan."""
    try:
        data = json.loads(line)
        return TextSpan(
            text=data["text"],
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", len(data["text"])),
        )
    except (json.JSONDecodeError, KeyError) as e:
        error_console.print(f"[red]Error parsing line {line_num}: {e}[/red]")
        sys.exit(1)
```

## CLI Commands and pyproject.toml

**Assume CLI commands will be callable using pyproject.toml** configuration.

### CLI Command Structure:

- Use Typer for CLI framework
- Define commands as functions with proper type annotations
- Use `typer.Argument` and `typer.Option` with descriptive help text
- Commands should be registered in `pyproject.toml` under `[project.scripts]`

```toml
[project.scripts]
bookwyrm = "bookwyrm.main:main"
```

### CLI Best Practices:

- Always use typer
  - use subcommands where appropriate
- Provide comprehensive help text and examples
- Use Rich for formatted output and progress bars
- Handle errors gracefully with user-friendly messages
- Support both verbose and quiet modes
- Validate inputs early and provide clear error messages

## Additional Standards

### Imports

- Use absolute imports
- Group imports: standard library, third-party, local
- Use `from typing import` for type hints

### Documentation

- All public functions and classes should have docstrings
- Use Google-style docstrings
- Include examples in docstrings for complex functions
- Document expected exceptions

### Testing

- Write tests for all public functions
- Use descriptive test names that explain what is being tested
- Test both success and failure cases
- Use fixtures for common test data

### Code Style

- Follow PEP 8
- Use Black for code formatting
- Use meaningful variable and function names
- Prefer explicit over implicit
- Use constants for magic numbers and strings
