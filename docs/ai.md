# AI Integration Guide

This page provides the essential models and function signatures needed for AI agents and automated systems to integrate with BookWyrm.

## Core Models

### Input/Output Models

::: bookwyrm.models.TextSpan
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.Citation
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.CitationResponse
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.PDFExtractResponse
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.ClassifyResponse
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.SummaryResponse
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

### Streaming Response Models

::: bookwyrm.models.TextResult
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

::: bookwyrm.models.TextSpanResult
    options:
      show_root_heading: true
      show_source: false
      members_order: source
      show_bases: false

## Synchronous Client Methods

::: bookwyrm.BookWyrmClient.classify
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.BookWyrmClient.extract_pdf
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.BookWyrmClient.stream_process_text
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.BookWyrmClient.get_citations
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.BookWyrmClient.summarize
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

## Asynchronous Client Methods

::: bookwyrm.AsyncBookWyrmClient.classify
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.AsyncBookWyrmClient.extract_pdf
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.AsyncBookWyrmClient.stream_process_text
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.AsyncBookWyrmClient.get_citations
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

::: bookwyrm.AsyncBookWyrmClient.summarize
    options:
      show_root_heading: true
      show_source: false
      show_signature: true
      separate_signature: true

## Quick Reference

### Typical Workflow

1. **Classify** → `classify()` → `ClassifyResponse`
2. **Extract** → `extract_pdf()` → `PDFExtractResponse`
3. **Process** → `stream_process_text()` → `Iterator[TextResult|TextSpanResult]`
4. **Cite** → `get_citations()` → `CitationResponse`
5. **Summarize** → `summarize()` → `SummaryResponse`

### Key Type Signatures

```python
# Classification
def classify(*, content_bytes: bytes, filename: str) -> ClassifyResponse

# PDF Extraction  
def extract_pdf(*, pdf_bytes: bytes, filename: str) -> PDFExtractResponse

# Text Processing (always streaming)
def stream_process_text(*, text: str, chunk_size: int, offsets: bool) -> Iterator[TextResult|TextSpanResult]

# Citation Finding
def get_citations(*, chunks: List[TextSpan], question: str) -> CitationResponse

# Summarization
def summarize(*, content: str, max_tokens: int) -> SummaryResponse
```
