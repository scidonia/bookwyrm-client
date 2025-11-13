# Tutorials

Choose your interface:

## [CLI Guide](cli-guide.md) - Command Line

**For:** Quick tasks, scripting, getting started

```bash
bookwyrm extract-pdf doc.pdf --output extracted.json
bookwyrm cite phrases.jsonl --question "What are the findings?"
```

## [Client Library Guide](client-guide.md) - Python API

**For:** Applications, complex workflows

```python
with BookWyrmClient() as client:
    for response in client.stream_extract_pdf(pdf_bytes=data):
        process_page(response.page_data)
```

## What You'll Learn

- **📄 Document Processing** - PDF extraction, classification, character mapping
- **📝 Text Analysis** - Phrasal processing, smart chunking
- **🔍 Citation Finding** - Question answering with quality scores
- **📊 Summarization** - Hierarchical summaries, structured output

## Sample Files

- [`data/SOA_2025_Final.pdf`](https://github.com/scidonia/bookwyrm-client/blob/main/data/SOA_2025_Final.pdf) - Spacecraft technology document
- [`data/country-of-the-blind.txt`](https://github.com/scidonia/bookwyrm-client/blob/main/data/country-of-the-blind.txt) - H.G. Wells text
- [`data/summary.py`](https://github.com/scidonia/bookwyrm-client/blob/main/data/summary.py) - Pydantic model example

**Start here:** [CLI Guide](cli-guide.md) → [Client Library Guide](client-guide.md) → [Examples](examples.md)
