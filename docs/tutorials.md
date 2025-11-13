# Tutorials

Welcome to the BookWyrm tutorials! These comprehensive guides will walk you through using BookWyrm for document processing, text analysis, and citation finding.

## Choose Your Interface

BookWyrm provides two powerful ways to work with documents and text:

### [CLI Guide](cli-guide.md) - Command Line Interface

**Best for:** Quick tasks, scripting, automation, and getting started

The CLI provides immediate access to all BookWyrm features through simple commands. Perfect for:

- **One-off document processing** - Classify PDFs, extract text, find citations
- **Shell scripting and automation** - Integrate BookWyrm into existing workflows  
- **Learning BookWyrm capabilities** - Explore features without writing code
- **Batch processing** - Process multiple files with shell loops

**Key features:**
- PDF extraction and text mapping with character positions
- Phrasal text processing for better chunking
- Citation finding with quality scoring
- Text summarization with structured output using Pydantic models
- File classification and content analysis

**Example workflow:**
```bash
# Extract PDF → Convert to text → Find citations
bookwyrm extract-pdf document.pdf --output extracted.json
bookwyrm pdf-to-text extracted.json
bookwyrm cite extracted_phrases.jsonl --question "What are the key findings?"
```

### [Client Library Guide](client-guide.md) - Python API

**Best for:** Applications, complex workflows, and programmatic control

The Python client library provides full programmatic access with streaming support and type safety. Perfect for:

- **Building applications** - Integrate BookWyrm into Python applications
- **Complex processing pipelines** - Chain operations with custom logic
- **Real-time processing** - Stream results as they're generated
- **Error handling and retries** - Robust production workflows

**Key features:**
- Streaming operations with real-time progress updates
- Full type annotations and Pydantic models
- Async/await support for concurrent processing
- Context managers for automatic resource cleanup
- Comprehensive error handling

**Example workflow:**
```python
# Process PDF with streaming and error handling
with BookWyrmClient() as client:
    for response in client.stream_extract_pdf(pdf_bytes=data):
        if isinstance(response, PDFStreamPageResponse):
            process_page(response.page_data)
```

## What You'll Learn

Both tutorials cover the same core BookWyrm capabilities:

### 📄 **Document Processing**
- **PDF Classification** - Understand document types and content
- **Structure Extraction** - Get text, tables, and layout information
- **Character Mapping** - Link text positions to PDF coordinates

### 📝 **Text Analysis** 
- **Phrasal Processing** - Break text into meaningful chunks
- **Smart Chunking** - Create bounded chunks that respect sentence boundaries
- **Position Tracking** - Maintain character offsets for highlighting

### 🔍 **Citation Finding**
- **Question Answering** - Find relevant text passages for specific questions
- **Quality Scoring** - Get relevance scores (0-4) for each citation
- **Multi-question Support** - Ask multiple questions simultaneously

### 📊 **Text Summarization**
- **Hierarchical Summarization** - Multi-level text condensation
- **Structured Output** - Use Pydantic models for consistent JSON results
- **Custom Prompts** - Tailor summarization for specific domains

### 🎯 **Advanced Features**
- **Streaming Operations** - Real-time progress for long-running tasks
- **Model Strength Selection** - Choose speed vs. quality trade-offs
- **Batch Processing** - Handle multiple documents efficiently

## Sample Data

Both tutorials use the same sample files included in the repository:

- **`data/SOA_2025_Final.pdf`** - State-of-the-Art spacecraft technology document for PDF processing examples
- **`data/country-of-the-blind.txt`** - H.G. Wells' "The Country of the Blind" for text analysis and summarization
- **`data/summary.py`** - Example Pydantic model for structured literary analysis

## Getting Started

1. **New to BookWyrm?** Start with the [CLI Guide](cli-guide.md) to quickly explore capabilities
2. **Building an application?** Jump to the [Client Library Guide](client-guide.md) for programmatic access
3. **Want to see code examples?** Check out [Examples](examples.md) for specific use cases

Both tutorials are comprehensive and include:
- Step-by-step instructions with real examples
- Expected output and file formats
- Error handling and best practices
- Complete workflows from raw documents to insights

Choose the tutorial that matches your preferred way of working, or read both to get the full picture of BookWyrm's capabilities!
