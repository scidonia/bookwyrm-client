# CLI Reference

The BookWyrm client includes a comprehensive command-line interface for all text processing operations. The documentation below is automatically generated from the CLI code.

::: mkdocs-typer
    :module: bookwyrm.cli
    :command: app
    :prog_name: bookwyrm
    :depth: 2

## Additional Information

### Environment Variables

- `BOOKWYRM_API_KEY` - Your BookWyrm API key (required)
- `BOOKWYRM_API_URL` - Base URL for the BookWyrm API (default: https://api.bookwyrm.ai:443)
- `BOOKWYRM_PDF_API_URL` - Base URL for PDF extraction API (falls back to BOOKWYRM_API_URL)

### Global Options

All commands support these global options:

- `--base-url TEXT` - Override the default API base URL
- `--api-key TEXT` - Provide API key (overrides BOOKWYRM_API_KEY env var)
- `--version` - Show version and exit
- `--help` - Show help message

### Error Handling

The CLI provides helpful error messages and exit codes:

- **Exit code 0**: Success
- **Exit code 1**: Error (API error, file not found, invalid arguments, etc.)

### Output Formats

#### Citations Output

**JSON format (non-streaming):**

```json
[
  {
    "start_chunk": 0,
    "end_chunk": 0,
    "text": "Citation text here",
    "reasoning": "Why this citation is relevant",
    "quality": 3
  }
]
```

**JSONL format (streaming):**

```jsonl
{"start_chunk": 0, "end_chunk": 0, "text": "Citation 1", "reasoning": "...", "quality": 3}
{"start_chunk": 1, "end_chunk": 1, "text": "Citation 2", "reasoning": "...", "quality": 4}
```

#### Summary Output

```json
{
  "summary": "The generated summary text or structured JSON",
  "subsummary_count": 5,
  "levels_used": 2,
  "total_tokens": 15000,
  "source_file": "input.jsonl",
  "max_tokens": 10000,
  "model_used": "BookSummary",
  "intermediate_summaries": [["level 1 summaries"], ["level 2 summaries"]]
}
```

#### Phrases Output (JSONL)

```jsonl
{"type": "text_span", "text": "First phrase", "start_char": 0, "end_char": 12}
{"type": "text_span", "text": "Second phrase", "start_char": 13, "end_char": 26}
```
