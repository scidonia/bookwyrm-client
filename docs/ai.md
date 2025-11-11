# AI Integration Guide

**Point your AI here** - This page provides the essential models and function signatures needed for AI agents and automated systems to integrate with BookWyrm.

## Quick Reference for AI Systems

| Task | Method | Input | Output | Use Case |
|------|--------|-------|--------|----------|
| Text → Chunks | `stream_process_text()` | Raw text/URL | `TextSpanResult[]` | Document preprocessing |
| Question → Citations | `stream_citations()` | Chunks + question | `Citation[]` | RAG, Q&A systems |
| PDF → Structure | `stream_extract_pdf()` | PDF bytes/URL | `PDFPage[]` | Document parsing |
| File → Type | `classify()` | File bytes | `FileClassification` | Content routing |
| Content → Summary | `stream_summarize()` | Text/phrases | Summary text | Document analysis |
| **Structured Summary** | `stream_summarize()` | Text + Pydantic model | **Structured JSON** | **Data extraction** |

## Complete Type Definitions

```python
# Essential imports for AI code generation
from bookwyrm import AsyncBookWyrmClient, BookWyrmClient, BookWyrmAPIError
from bookwyrm.models import (
    TextSpan, Citation, CitationResponse,
    ClassifyResponse, SummaryResponse, TextResult, TextSpanResult,
    ResponseFormat, PhraseProgressUpdate, PDFPage
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
    citations = []
    async with AsyncBookWyrmClient(api_key="your-key") as client:
        async for stream_response in client.stream_citations(
            chunks=chunks,
            question=user_question
        ):
            if hasattr(stream_response, 'citation'):
                citations.append(stream_response.citation)
        return citations
```

### Function Calling for AI Agents

```python
async def find_citations_tool(question: str, document_chunks: List[dict]) -> List[dict]:
    """Tool function for AI agents to find citations in documents."""
    chunks = [TextSpan(**chunk) for chunk in document_chunks]
    citations = []
    async with AsyncBookWyrmClient() as client:
        async for stream_response in client.stream_citations(chunks=chunks, question=question):
            if hasattr(stream_response, 'citation'):
                citations.append(stream_response.citation)
        return [citation.model_dump() for citation in citations]

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

async def structured_summary_tool(text: str, model_schema: dict, model_name: str) -> dict:
    """Tool function for structured data extraction using Pydantic models."""
    import json
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_summarize(
            content=text,
            model_name=model_name,
            model_schema_json=json.dumps(model_schema),
            model_strength="smart"
        ):
            if hasattr(response, 'summary'):
                return json.loads(response.summary)
    return {}
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
            citations = []
            async with AsyncBookWyrmClient() as client:
                async for stream_response in client.stream_citations(chunks=chunks, question=question):
                    if hasattr(stream_response, 'citation'):
                        citations.append(stream_response.citation)
                return citations
        
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

### Citation Finding (4 lines)

```python
citations = []
async with AsyncBookWyrmClient(api_key="key") as client:
    async for r in client.stream_citations(chunks=text_chunks, question="What is X?"):
        if hasattr(r, 'citation'): citations.append(r.citation)
best_citation = max(citations, key=lambda c: c.quality) if citations else None
```

### Document Processing (4 lines)

```python
chunks = []
async with AsyncBookWyrmClient() as client:
    async for response in client.stream_process_text(text=document, offsets=True):
        if isinstance(response, TextSpanResult): chunks.append(response)
```

### PDF Analysis (4 lines)

```python
pages = []
async with AsyncBookWyrmClient() as client:
    async for response in client.stream_extract_pdf(pdf_bytes=pdf_data):
        if hasattr(response, 'page_data'): pages.append(response.page_data)
text_elements = [elem for page in pages for elem in page.text_blocks]
```

### Structured Data Extraction (4 lines)

```python
import json
async with AsyncBookWyrmClient() as client:
    async for r in client.stream_summarize(content=text, model_name="MyModel", model_schema_json=schema):
        if hasattr(r, 'summary'): return json.loads(r.summary)
```

## Performance Optimization for AI Systems

### Batch Processing

```python
async def batch_process_citations(chunk_groups: List[List[TextSpan]], questions: List[str]):
    """Process multiple citation requests concurrently."""
    async def single_request(chunks, question):
        citations = []
        async with AsyncBookWyrmClient() as client:
            async for stream_response in client.stream_citations(chunks=chunks, question=question):
                if hasattr(stream_response, 'citation'):
                    citations.append(stream_response.citation)
            return citations
    
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
- **Structured output**: Use detailed field descriptions in Pydantic models for better extraction
- **Model strength selection**: Use `smart`/`clever`/`wise` for structured output, `swift` for testing
- **JSON validation**: Always validate structured output with `json.loads()` and Pydantic models

## Structured Data Extraction Patterns

### Define Extraction Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
import json

class PersonInfo(BaseModel):
    """Extract person information from text."""
    name: Optional[str] = Field(None, description="Full name of the person")
    age: Optional[int] = Field(None, description="Age in years")
    occupation: Optional[str] = Field(None, description="Job title or profession")
    location: Optional[str] = Field(None, description="City, state, or country of residence")
    skills: Optional[List[str]] = Field(None, description="List of skills or expertise areas")

class CompanyInfo(BaseModel):
    """Extract company information from text."""
    name: Optional[str] = Field(None, description="Official company name")
    industry: Optional[str] = Field(None, description="Primary industry or sector")
    founded: Optional[int] = Field(None, description="Year the company was founded")
    employees: Optional[int] = Field(None, description="Number of employees")
    revenue: Optional[str] = Field(None, description="Annual revenue or financial information")
    products: Optional[List[str]] = Field(None, description="Main products or services offered")

class EventInfo(BaseModel):
    """Extract event information from text."""
    name: Optional[str] = Field(None, description="Name or title of the event")
    date: Optional[str] = Field(None, description="Date of the event in YYYY-MM-DD format")
    location: Optional[str] = Field(None, description="Venue or location where event takes place")
    attendees: Optional[int] = Field(None, description="Number of people attending")
    organizer: Optional[str] = Field(None, description="Person or organization organizing the event")
    topics: Optional[List[str]] = Field(None, description="Main topics or themes covered")
```

### AI Agent Integration

```python
async def extract_structured_data(text: str, extraction_type: str) -> dict:
    """AI agent function for structured data extraction."""
    
    models = {
        "person": PersonInfo,
        "company": CompanyInfo, 
        "event": EventInfo
    }
    
    if extraction_type not in models:
        raise ValueError(f"Unknown extraction type: {extraction_type}")
    
    model_class = models[extraction_type]
    schema = json.dumps(model_class.model_json_schema())
    
    async with AsyncBookWyrmClient() as client:
        async for response in client.stream_summarize(
            content=text,
            model_name=model_class.__name__,
            model_schema_json=schema,
            model_strength="smart"
        ):
            if hasattr(response, 'summary'):
                try:
                    structured_data = json.loads(response.summary)
                    validated_data = model_class.model_validate(structured_data)
                    return validated_data.model_dump()
                except (json.JSONDecodeError, ValueError) as e:
                    return {"error": f"Failed to parse structured output: {e}"}
    
    return {"error": "No response received"}

# Usage in AI agents
person_data = await extract_structured_data(resume_text, "person")
company_data = await extract_structured_data(company_description, "company")
event_data = await extract_structured_data(event_announcement, "event")
```

### Batch Structured Processing

```python
async def batch_extract_structured_data(texts: List[str], model_class: BaseModel) -> List[dict]:
    """Process multiple texts with the same structured model."""
    
    schema = json.dumps(model_class.model_json_schema())
    results = []
    
    async def process_single(text: str) -> dict:
        async with AsyncBookWyrmClient() as client:
            async for response in client.stream_summarize(
                content=text,
                model_name=model_class.__name__,
                model_schema_json=schema,
                model_strength="smart"
            ):
                if hasattr(response, 'summary'):
                    try:
                        return json.loads(response.summary)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response"}
        return {"error": "No response"}
    
    # Process all texts concurrently
    import asyncio
    results = await asyncio.gather(*[process_single(text) for text in texts])
    return results

# Usage
resume_texts = ["Resume 1 content...", "Resume 2 content...", "Resume 3 content..."]
person_data_list = await batch_extract_structured_data(resume_texts, PersonInfo)
```

### Error Handling for Structured Output

```python
async def robust_structured_extraction(text: str, model_class: BaseModel, max_retries: int = 3) -> dict:
    """Structured extraction with error handling and retries."""
    
    schema = json.dumps(model_class.model_json_schema())
    
    for attempt in range(max_retries):
        try:
            async with AsyncBookWyrmClient() as client:
                async for response in client.stream_summarize(
                    content=text,
                    model_name=model_class.__name__,
                    model_schema_json=schema,
                    model_strength="smart" if attempt == 0 else "clever"  # Upgrade model on retry
                ):
                    if hasattr(response, 'summary'):
                        try:
                            structured_data = json.loads(response.summary)
                            validated_data = model_class.model_validate(structured_data)
                            return {"success": True, "data": validated_data.model_dump()}
                        except json.JSONDecodeError as e:
                            if attempt == max_retries - 1:
                                return {"success": False, "error": f"JSON parsing failed: {e}", "raw": response.summary}
                            continue  # Retry with better model
                        except ValueError as e:
                            if attempt == max_retries - 1:
                                return {"success": False, "error": f"Validation failed: {e}", "raw": response.summary}
                            continue  # Retry with better model
        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": f"API error: {e}"}
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return {"success": False, "error": "Max retries exceeded"}
```

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

::: bookwyrm.models.PDFPage
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

::: bookwyrm.BookWyrmClient.stream_citations
options:
show_root_heading: true

::: bookwyrm.BookWyrmClient.stream_summarize
options:
show_root_heading: true

## Asynchronous Client Methods

::: bookwyrm.AsyncBookWyrmClient.classify
options:
show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.stream_process_text
options:
show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.stream_citations
options:
show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.stream_summarize
options:
show_root_heading: true


