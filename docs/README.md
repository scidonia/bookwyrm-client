# Documentation Testing

This directory contains live tests for documentation code examples.

## Setup

1. Install dependencies:
```bash
pip install -e ".[dev]"
```

2. Set your API key:
```bash
export BOOKWYRM_API_KEY="your-api-key-here"
```

3. Ensure required data files exist:
- `../data/SOA_2025_Final.pdf`
- `../data/country-of-the-blind.txt`

## Running Tests

```bash
# Test all documentation examples (from project root)
pytest docs/client-guide.md --codeblocks

# Test from docs directory
cd docs && pytest client-guide.md

# Test with verbose output
cd docs && pytest client-guide.md -v -s

# Test specific sections by filtering output
cd docs && pytest client-guide.md -k "classification"
```

## What Gets Tested

The tests extract and execute all Python code blocks from the markdown files, running them against the live BookWyrm API. This ensures that:

- All code examples actually work
- API responses match expectations
- File operations succeed
- Import statements are correct
- Type annotations are valid

## Generated Files

Tests will create temporary files in the `../data/` directory during execution. These are automatically cleaned up after each test run.

## Skipped Tests

Tests will be automatically skipped if:
- `BOOKWYRM_API_KEY` environment variable is not set
- Required data files are missing
- API is unavailable

## CI Integration

To run in CI, ensure the environment has:
- API key set as environment variable
- Required data files present
- Network access to BookWyrm API
