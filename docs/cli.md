# CLI Reference

::: mkdocs-typer
    :module: bookwyrm.cli
    :command: app
    :prog_name: bookwyrm
    :depth: 2
# CLI Reference

The BookWyrm client includes a command-line interface for common operations.

## Installation

The CLI is included when you install the bookwyrm package:

```bash
pip install bookwyrm
```

## Authentication

Set your API key as an environment variable:

```bash
export BOOKWYRM_API_KEY="your-api-key"
```

Or pass it with the `--api-key` option to any command.

## Commands

### `bookwyrm cite`

Find citations for a question in text chunks.

```bash
bookwyrm cite "What causes climate change?" data/chunks.jsonl
```

**Arguments:**
- `question` - The question to find citations for
- `jsonl_input` - Path to JSONL file containing text chunks (optional if using --file or --url)

**Options:**
- `--url TEXT` - URL to JSONL file (alternative to file path)
- `--file PATH` - JSONL file to read chunks from
- `-o, --output PATH` - Output file for citations (JSON for non-streaming, JSONL for streaming)
- `--start INTEGER` - Start chunk index (default: 0)
- `--limit INTEGER` - Limit number of chunks to process
- `--max-tokens INTEGER` - Maximum tokens per chunk (default: 1000)
- `--stream/--no-stream` - Use streaming API (default: --stream)
- `--base-url TEXT` - Base URL of the BookWyrm API
- `--api-key TEXT` - API key for authentication
- `-v, --verbose` - Show detailed citation information

**Examples:**

```bash
# Basic citation finding
bookwyrm cite "What is machine learning?" ml_chunks.jsonl

# With output file
bookwyrm cite "Climate change causes" data.jsonl -o citations.json

# Streaming with verbose output
bookwyrm cite "AI applications" chunks.jsonl --stream -v

# From URL
bookwyrm cite "Question here" --url https://example.com/chunks.jsonl

# Limit processing
bookwyrm cite "Question" data.jsonl --start 10 --limit 50
```

### `bookwyrm summarize`

Summarize text content from JSONL files.

```bash
bookwyrm summarize data/phrases.jsonl --output summary.json
```

**Arguments:**
- `jsonl_file` - JSONL file containing phrases

**Options:**
- `-o, --output PATH` - Output file for summary (JSON format)
- `--max-tokens INTEGER` - Maximum tokens per chunk (max: 131,072, default: 10000)
- `--debug` - Include intermediate summaries
- `--stream/--no-stream` - Use streaming API (default: --stream)
- `--model-class-file PATH` - Python file containing Pydantic model class for structured output
- `--model-class-name TEXT` - Name of the Pydantic model class to use
- `--chunk-prompt TEXT` - Custom prompt for chunk summarization
- `--summary-prompt TEXT` - Custom prompt for summary of summaries
- `--base-url TEXT` - Base URL of the BookWyrm API
- `--api-key TEXT` - API key for authentication
- `-v, --verbose` - Show detailed information

**Examples:**

```bash
# Basic summarization
bookwyrm summarize book_phrases.jsonl --output summary.json

# With debug information
bookwyrm summarize data.jsonl --debug --output detailed_summary.json

# Structured output with Pydantic model
bookwyrm summarize book.jsonl \
  --model-class-file models/book_summary.py \
  --model-class-name BookSummary \
  --output structured_summary.json

# Custom prompts
bookwyrm summarize scientific_text.jsonl \
  --chunk-prompt "Extract key scientific concepts and findings" \
  --summary-prompt "Create a comprehensive scientific overview" \
  --output science_summary.json

# Larger chunks
bookwyrm summarize large_text.jsonl --max-tokens 20000 --output summary.json
```

### `bookwyrm phrasal`

Process text using phrasal analysis to extract phrases or chunks.

```bash
bookwyrm phrasal "Your text here" --output phrases.jsonl
```

**Arguments:**
- `input_text` - Text to process (optional if using --url or --file)

**Options:**
- `--url TEXT` - URL to fetch text from
- `-f, --file PATH` - File to read text from
- `-o, --output PATH` - Output file for phrases (JSONL format)
- `--chunk-size INTEGER` - Target size for each chunk (if not specified, returns phrases individually)
- `--format TEXT` - Response format: "text_only" or "with_offsets" (default: "with_offsets")
- `--spacy-model TEXT` - SpaCy model to use (default: "en_core_web_sm")
- `--base-url TEXT` - Base URL of the BookWyrm API
- `--api-key TEXT` - API key for authentication
- `-v, --verbose` - Show detailed information

**Examples:**

```bash
# Process text directly
bookwyrm phrasal "Natural language processing is fascinating." -o phrases.jsonl

# Process file
bookwyrm phrasal -f document.txt --output phrases.jsonl

# Create chunks of specific size
bookwyrm phrasal -f large_text.txt --chunk-size 1000 --output chunks.jsonl

# Process from URL
bookwyrm phrasal --url https://example.com/text.txt --output phrases.jsonl

# Text only format (no position offsets)
bookwyrm phrasal -f text.txt --format text_only --output simple_phrases.jsonl

# Different SpaCy model
bookwyrm phrasal -f text.txt --spacy-model en_core_web_lg --output phrases.jsonl
```

### `bookwyrm classify`

Classify files to determine their type and format.

```bash
bookwyrm classify document.pdf
```

**Arguments:**
- `file_path` - File to classify (optional if using --file or --url)

**Options:**
- `--url TEXT` - URL to classify
- `--file PATH` - File to classify
- `--filename TEXT` - Optional filename hint for classification
- `-o, --output PATH` - Output file for classification results (JSON format)
- `--base-url TEXT` - Base URL of the BookWyrm API
- `--api-key TEXT` - API key for authentication
- `-v, --verbose` - Show detailed information

**Examples:**

```bash
# Classify local file
bookwyrm classify document.pdf

# Classify from URL
bookwyrm classify --url https://example.com/file.dat

# With output file
bookwyrm classify unknown_file.bin --output classification.json

# With filename hint
bookwyrm classify data.txt --filename "research_data.csv" --output results.json
```

### `bookwyrm extract-pdf`

Extract structured data from PDF files using OCR.

```bash
bookwyrm extract-pdf document.pdf --output extracted.json
```

**Arguments:**
- `pdf_file` - PDF file to extract from (optional if using --file or --url)

**Options:**
- `--url TEXT` - PDF URL to extract from
- `--file PATH` - PDF file to extract from
- `-o, --output PATH` - Output file for extracted data (JSON format)
- `--start-page INTEGER` - 1-based page number to start from
- `--num-pages INTEGER` - Number of pages to process from start_page
- `--stream/--no-stream` - Use streaming API with progress (default: --stream)
- `--base-url TEXT` - Base URL of the PDF extraction API
- `--api-key TEXT` - API key for authentication
- `-v, --verbose` - Show detailed information

**Examples:**

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
```

## Global Options

All commands support these global options:

- `--base-url TEXT` - Override the default API base URL
- `--api-key TEXT` - Provide API key (overrides BOOKWYRM_API_KEY env var)
- `--version` - Show version and exit
- `--help` - Show help message

## Environment Variables

- `BOOKWYRM_API_KEY` - Your BookWyrm API key
- `BOOKWYRM_API_URL` - Base URL for the BookWyrm API (default: https://api.bookwyrm.ai:443)
- `BOOKWYRM_PDF_API_URL` - Base URL for PDF extraction API (falls back to BOOKWYRM_API_URL)

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
