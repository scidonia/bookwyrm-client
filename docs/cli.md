# CLI Reference

The BookWyrm client includes a comprehensive command-line interface for all text processing operations. Most documentation is embedded in the CLI itself and can be accessed using the `--help` option.

## Installation

The CLI is included when you install the bookwyrm package:

```bash
pip install bookwyrm
```

## Quick Help

For comprehensive help on any command, use:

```bash
# General help
bookwyrm --help

# Command-specific help  
bookwyrm cite --help
bookwyrm summarize --help
bookwyrm phrasal --help
bookwyrm classify --help
bookwyrm extract-pdf --help
```

## Authentication

Set your API key as an environment variable:

```bash
export BOOKWYRM_API_KEY="your-api-key"
```

Or pass it with the `--api-key` option to any command.

## Environment Variables

- `BOOKWYRM_API_KEY` - Your BookWyrm API key (required)
- `BOOKWYRM_API_URL` - Base URL for the BookWyrm API (default: https://api.bookwyrm.ai:443)
- `BOOKWYRM_PDF_API_URL` - Base URL for PDF extraction API (falls back to BOOKWYRM_API_URL)

## Commands Overview

## Auto-Generated Command Documentation

The following documentation is automatically generated from the CLI code:

::: mkdocs-click
    :module: bookwyrm.cli
    :command: app
    :prog_name: bookwyrm
    :depth: 2

## Global Options

All commands support these global options:

- `--base-url TEXT` - Override the default API base URL
- `--api-key TEXT` - Provide API key (overrides BOOKWYRM_API_KEY env var)
- `--version` - Show version and exit
- `--help` - Show help message

## Output Formats

### Citations Output

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

### Summary Output

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

### Phrases Output (JSONL)

```jsonl
{"type": "phrase", "text": "First phrase", "start_char": 0, "end_char": 12}
{"type": "phrase", "text": "Second phrase", "start_char": 13, "end_char": 26}
```

## Error Handling

The CLI provides helpful error messages and exit codes:

- **Exit code 0**: Success
- **Exit code 1**: Error (API error, file not found, invalid arguments, etc.)

Common error scenarios:

```bash
# Missing API key
bookwyrm cite "question" data.jsonl
# Error: No API key provided!

# File not found
bookwyrm summarize nonexistent.jsonl
# Error: File not found: nonexistent.jsonl

# Invalid arguments
bookwyrm summarize data.jsonl --model-class-file model.py
# Error: --model-class-name is required when --model-class-file is provided
```
