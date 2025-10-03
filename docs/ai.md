# AI Integration Guide

**Point your AI here** - This page provides the essential models and function signatures needed for AI agents and automated systems to integrate with BookWyrm.

## Quick Reference for AI Systems

| Task | Method | Input | Output | Use Case |
|------|--------|-------|--------|----------|
| Text → Chunks | `stream_process_text()` | Raw text/URL | `TextSpanResult[]` | Document preprocessing |
| Question → Citations | `get_citations()` | Chunks + question | `Citation[]` | RAG, Q&A systems |
| PDF → Structure | `extract_pdf()` | PDF bytes/URL | `PDFPage[]` | Document parsing |
| File → Type | `classify()` | File bytes | `FileClassification` | Content routing |
| Content → Summary | `summarize()` | Text/phrases | Summary text | Document analysis |

## Complete Type Definitions

```python
# Essential imports for AI code generation
from bookwyrm import AsyncBookWyrmClient, BookWyrmClient, BookWyrmAPIError
from bookwyrm.models import (
    TextSpan, Citation, CitationResponse,
    PDFExtractResponse, ClassifyResponse, 
    SummaryResponse, TextResult, TextSpanResult,
    ResponseFormat, PhraseProgressUpdate
)
from typing import List, Optional, AsyncIterator, Union
import asyncio
```

## Common AI Integration Patterns

### RAG (Retrieval-Augmented Generation) Pipeline

```python
async def rag_pipeline(document_text: str, user_question: str) -> List[Citation]:
    """Complete RAG pipeline for AI agents."""
    # Step 1: Process document into chunks
    chunks = []
    async with AsyncBookWyrmClient(api_key="your-key") as client:
        async for response in client.stream_process_text(
            text=document_text,
            chunk_size=1000,
            offsets=True
        ):
            if isinstance(response, TextSpanResult):
                chunks.append(TextSpan(
                    text=response.text,
                    start_char=response.start_char,
                    end_char=response.end_char
                ))
    
    # Step 2: Find relevant citations
    async with AsyncBookWyrmClient(api_key="your-key") as client:
        response = await client.get_citations(
            chunks=chunks,
            question=user_question
        )
        return response.citations
```

### Function Calling for AI Agents

```python
async def find_citations_tool(question: str, document_chunks: List[dict]) -> List[dict]:
    """Tool function for AI agents to find citations in documents."""
    chunks = [TextSpan(**chunk) for chunk in document_chunks]
    async with AsyncBookWyrmClient() as client:
        response = await client.get_citations(chunks=chunks, question=question)
        return [citation.model_dump() for citation in response.citations]

async def process_document_tool(text: str, chunk_size: int = 1000) -> List[dict]:
    """Tool function to process documents into chunks."""
    chunks = []
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_process_text(
            text=text, chunk_size=chunk_size, offsets=True
        ):
            if isinstance(response, TextSpanResult):
                chunks.append(response.model_dump())
    return chunks
```

### Concurrent Processing Pattern

```python
async def process_multiple_documents(documents: List[str]) -> List[List[TextSpan]]:
    """Process multiple documents concurrently."""
    async def process_single(text: str) -> List[TextSpan]:
        chunks = []
        async with AsyncBookWyrmClient() as client:
            async for response in client.stream_process_text(text=text, offsets=True):
                if isinstance(response, TextSpanResult):
                    chunks.append(TextSpan(
                        text=response.text,
                        start_char=response.start_char,
                        end_char=response.end_char
                    ))
        return chunks
    
    return await asyncio.gather(*[process_single(doc) for doc in documents])
```

## Error Handling for AI Systems

```python
from bookwyrm import BookWyrmAPIError
import asyncio
import logging

async def robust_citation_search(chunks: List[TextSpan], question: str, max_retries: int = 3):
    """Citation search with proper error handling and retries."""
    for attempt in range(max_retries):
        try:
            async with AsyncBookWyrmClient() as client:
                response = await client.get_citations(chunks=chunks, question=question)
                return response.citations
        
        except BookWyrmAPIError as e:
            if e.status_code == 429:  # Rate limit
                wait_time = 2 ** attempt
                logging.warning(f"Rate limited, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            elif e.status_code == 413:  # Payload too large
                # Split chunks in half and retry
                mid = len(chunks) // 2
                chunks = chunks[:mid]
                logging.warning(f"Payload too large, reducing to {len(chunks)} chunks")
            else:
                logging.error(f"API error: {e}")
                if attempt == max_retries - 1:
                    raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
    
    return []  # Return empty list if all retries failed
```

## Minimal Working Examples

### Citation Finding (3 lines)

```python
async with AsyncBookWyrmClient(api_key="key") as client:
    response = await client.get_citations(chunks=text_chunks, question="What is X?")
    best_citation = max(response.citations, key=lambda c: c.quality)
```

### Document Processing (4 lines)

```python
chunks = []
async with AsyncBookWyrmClient() as client:
    async for response in client.stream_process_text(text=document, offsets=True):
        if isinstance(response, TextSpanResult): chunks.append(response)
```

### PDF Analysis (3 lines)

```python
async with AsyncBookWyrmClient() as client:
    response = await client.extract_pdf(pdf_bytes=pdf_data)
    text_elements = [elem for page in response.pages for elem in page.text_blocks]
```

## Performance Optimization for AI Systems

### Batch Processing

```python
async def batch_process_citations(chunk_groups: List[List[TextSpan]], questions: List[str]):
    """Process multiple citation requests concurrently."""
    async def single_request(chunks, question):
        async with AsyncBookWyrmClient() as client:
            return await client.get_citations(chunks=chunks, question=question)
    
    tasks = [single_request(chunks, q) for chunks, q in zip(chunk_groups, questions)]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### Memory-Efficient Streaming

```python
async def stream_large_document(text: str, chunk_size: int = 2000):
    """Process large documents without loading everything into memory."""
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_process_text(
            text=text, 
            chunk_size=chunk_size,
            offsets=True
        ):
            if isinstance(response, TextSpanResult):
                # Process chunk immediately, don't store all chunks
                yield response
            elif isinstance(response, PhraseProgressUpdate):
                print(f"Progress: {response.message}")
```

## AI Implementation Notes

- **Always use context managers**: `async with AsyncBookWyrmClient() as client:`
- **Handle streaming responses**: Check `isinstance(response, TextSpanResult)` before processing
- **Implement pagination**: Use `start` and `limit` parameters for large datasets
- **Rate limiting**: Implement exponential backoff for production systems
- **Memory management**: Process large documents in streaming chunks
- **Error recovery**: Handle network issues and API errors gracefully
- **Concurrent limits**: Don't exceed reasonable concurrent request limits
- **Chunk size optimization**: 500-2000 characters for balanced performance

## Core Models

::: bookwyrm.models.TextSpan
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.Citation
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.CitationResponse
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.PDFExtractResponse
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.ClassifyResponse
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.SummaryResponse
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.TextResult
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

::: bookwyrm.models.TextSpanResult
options:
show_root_heading: true
members_order: source
show_bases: true
inherited_members: true

## Synchronous Client Methods

::: bookwyrm.BookWyrmClient.classify
options:
show_root_heading: true


::: bookwyrm.BookWyrmClient.stream_process_text
options:
show_root_heading: true


::: bookwyrm.BookWyrmClient.summarize
options:
show_root_heading: true

## Asynchronous Client Methods

::: bookwyrm.AsyncBookWyrmClient.classify
options:
show_root_heading: true


::: bookwyrm.AsyncBookWyrmClient.stream_process_text
options:
show_root_heading: true


::: bookwyrm.AsyncBookWyrmClient.summarize
options:
show_root_heading: true
