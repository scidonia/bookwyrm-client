# Tutorials

Choose your preferred interface to get started with BookWyrm:

## [CLI Guide](cli-guide.md) - Command Line Interface

**Best for:** Quick tasks, scripting, and getting started

Perfect for one-off document processing, shell scripting, and learning BookWyrm capabilities without writing code.

```bash
# Complete workflow example
bookwyrm extract-pdf document.pdf --output extracted.json
bookwyrm pdf-to-text extracted.json
bookwyrm cite extracted_phrases.jsonl --question "What are the key findings?"
```

## [Client Library Guide](client-guide.md) - Python API

**Best for:** Applications and complex workflows

Full programmatic access with streaming support, type safety, and error handling for production applications.

```python
# Streaming workflow example
with BookWyrmClient() as client:
    for response in client.stream_extract_pdf(pdf_bytes=data):
        if isinstance(response, PDFStreamPageResponse):
            process_page(response.page_data)
```

## Core Capabilities

Both tutorials cover:

- **📄 Document Processing** - PDF classification, structure extraction, character mapping
- **📝 Text Analysis** - Phrasal processing, smart chunking, position tracking  
- **🔍 Citation Finding** - Question answering with quality scoring
- **📊 Text Summarization** - Hierarchical summaries with structured output
- **🎯 Advanced Features** - Streaming operations, model selection, batch processing

## Sample Data

Tutorials use included sample files:
- `data/SOA_2025_Final.pdf` - Spacecraft technology document
- `data/country-of-the-blind.txt` - H.G. Wells text for analysis
- `data/summary.py` - Example Pydantic model

## Quick Start

1. **New to BookWyrm?** → [CLI Guide](cli-guide.md)
2. **Building an app?** → [Client Library Guide](client-guide.md)  
3. **Need examples?** → [Examples](examples.md)

Both tutorials include step-by-step instructions, expected outputs, and complete workflows from raw documents to structured insights.
