# BookWyrm Integration Tests

This directory contains integration tests for the BookWyrm client library and CLI.

## Test Organization

- `test_cli/` - CLI command integration tests
- `test_library/` - Python library integration tests
- `conftest.py` - Shared fixtures and configuration

## Running Tests

### Using tox (Recommended)

```bash
# Run all tests with current Python version
tox -e py311-dev

# Run only CLI tests
tox -e py311-cli-tests

# Run only library tests  
tox -e py311-library-tests

# Run with different Python versions
tox -e py39-dev,py310-dev,py311-dev,py312-dev

# Run against different client versions
tox -e py311-v0.1.0,py311-v0.2.0,py311-latest

# Run slow/comprehensive tests
tox -e py311-slow

# Test against live API (requires API key)
BOOKWYRM_API_KEY=your-key tox -e live-api

# Test with mocked API only
tox -e mock-api
```

### Using pytest directly

```bash
# Run all integration tests
pytest integration/

# Run specific test categories
pytest integration/ -m "cli"
pytest integration/ -m "library"
pytest integration/ -m "not slow"

# Run with live API (requires API key)
BOOKWYRM_API_KEY=your-key pytest integration/ -m "live_api"

# Run without live API tests
pytest integration/ -m "not live_api"

# Verbose output
pytest integration/ -v --tb=long
```

## Test Categories

Tests are organized using pytest markers:

- `@pytest.mark.cli` - CLI command tests
- `@pytest.mark.library` - Python library tests
- `@pytest.mark.slow` - Slow/comprehensive tests
- `@pytest.mark.live_api` - Tests requiring live API access
- `@pytest.mark.integration` - Full integration tests

## Environment Variables

- `BOOKWYRM_API_KEY` - Required for live API tests
- `BOOKWYRM_API_URL` - API base URL (defaults to https://api.bookwyrm.ai:443)

## Test Data

Tests use temporary files and mock data created by fixtures in `conftest.py`:

- `sample_chunks_jsonl` - Sample text chunks for citation tests
- `sample_phrases_jsonl` - Sample phrases for summarization tests
- `sample_pdf_file` - Minimal PDF file for extraction tests
- `mock_bookwyrm_api` - Mocked API responses

## Adding New Tests

1. Choose the appropriate directory (`test_cli/` or `test_library/`)
1. Add appropriate pytest markers
1. Use existing fixtures from `conftest.py`
1. Follow the naming convention `test_*.py`
1. Add live API tests sparingly and mark them with `@pytest.mark.live_api`

## Tox Environments

The `tox.ini` file defines several test environments:

- `py{39,310,311,312}-dev` - Test current development version
- `py{39,310,311,312}-v{0.1.0,0.2.0,latest}` - Test specific released versions
- `py{39,310,311,312}-{cli,library}-tests` - Test specific components
- `live-api` - Test against live API
- `mock-api` - Test with mocked API only

This allows comprehensive testing across Python versions and client versions.
