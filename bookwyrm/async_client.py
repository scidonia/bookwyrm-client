"""Asynchronous client for BookWyrm API."""

import json
import os
import platform
from typing import AsyncIterator, Optional, Union, Dict, Any
import httpx

try:
    from importlib.metadata import version
    __version__ = version("bookwyrm")
except ImportError:
    __version__ = "unknown"
from .models import (
    CitationRequest,
    CitationResponse,
    StreamingCitationResponse,
    CitationProgressUpdate,
    CitationStreamResponse,
    CitationSummaryResponse,
    CitationErrorResponse,
    SummarizeRequest,
    SummaryResponse,
    StreamingSummarizeResponse,
    SummarizeProgressUpdate,
    SummarizeErrorResponse,
    RateLimitMessage,
    StructuralErrorMessage,
    ProcessTextRequest,
    StreamingPhrasalResponse,
    PhraseProgressUpdate,
    TextResult,
    TextSpanResult,
    ClassifyRequest,
    ClassifyResponse,
    PDFExtractRequest,
    PDFExtractResponse,
    StreamingPDFResponse,
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
)
from .client import BookWyrmClientError, BookWyrmAPIError


# User-Agent and client version headers
UA = f"bookwyrm-client/{__version__} (python/{platform.python_version()}; {platform.system()})"

DEFAULT_HEADERS = {
    "User-Agent": UA,
}


class AsyncBookWyrmClient:
    """Asynchronous client for BookWyrm API.
    
    The asynchronous client provides full async/await support for all BookWyrm API endpoints
    using the `httpx` library. It supports concurrent operations, streaming responses,
    and automatic session management with proper cleanup.
    
    Examples:
        Basic async client usage:
        
        ```python
        import asyncio
        from bookwyrm import AsyncBookWyrmClient
        
        async def main():
            # Using environment variable for API key
            client = AsyncBookWyrmClient()
            
            # Explicit API key
            client = AsyncBookWyrmClient(api_key="your-api-key")
            
            # Custom base URL
            client = AsyncBookWyrmClient(
                base_url="https://custom-api.example.com",
                api_key="your-api-key"
            )
        
        asyncio.run(main())
        ```
        
        Async context manager for automatic cleanup:
        
        ```python
        async def example():
            async with AsyncBookWyrmClient() as client:
                response = await client.get_citations(request)
                # Client is automatically closed when exiting the context
        ```
        
        Concurrent operations:
        
        ```python
        async def concurrent_operations():
            async with AsyncBookWyrmClient() as client:
                # Run multiple operations concurrently
                tasks = [
                    client.get_citations(request1),
                    client.get_citations(request2),
                    client.summarize(summarize_request)
                ]
                
                results = await asyncio.gather(*tasks)
                return results
        ```
    """

    def __init__(
        self,
        base_url: str = "https://api.bookwyrm.ai:443",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the async BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API. Defaults to "https://api.bookwyrm.ai:443"
            api_key: API key for authentication. If not provided, will attempt to read
                from BOOKWYRM_API_KEY environment variable
                
        Examples:
            ```python
            # Basic initialization
            client = AsyncBookWyrmClient()
            
            # With explicit API key
            client = AsyncBookWyrmClient(api_key="your-api-key")
            
            # With custom endpoint
            client = AsyncBookWyrmClient(
                base_url="https://localhost:8000",
                api_key="dev-key"
            )
            ```
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key or os.getenv("BOOKWYRM_API_KEY")
        self.client: httpx.AsyncClient = httpx.AsyncClient()

    async def get_citations(self, request: CitationRequest) -> CitationResponse:
        """Get citations for a question from text chunks asynchronously.
        
        This async method finds relevant citations that answer a specific question by analyzing
        the provided text chunks. Each citation includes a quality score (0-4) and reasoning
        for why it's relevant to the question.

        Args:
            request: Citation request containing chunks/URL and question to answer

        Returns:
            Citation response with found citations, total count, and usage information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async citation finding:
            
            ```python
            from bookwyrm.models import CitationRequest, TextChunk
            
            async def find_citations():
                chunks = [
                    TextChunk(text="The sky is blue.", start_char=0, end_char=16),
                    TextChunk(text="Water is wet.", start_char=17, end_char=30)
                ]
                
                request = CitationRequest(
                    chunks=chunks,
                    question="Why is the sky blue?"
                )
                
                async with AsyncBookWyrmClient() as client:
                    response = await client.get_citations(request)
                    print(f"Found {response.total_citations} citations")
                    
                    for citation in response.citations:
                        print(f"Quality: {citation.quality}/4")
                        print(f"Text: {citation.text}")
            
            asyncio.run(find_citations())
            ```
            
            Concurrent citation requests:
            
            ```python
            async def concurrent_citations():
                requests = [
                    CitationRequest(chunks=chunks1, question="Question 1"),
                    CitationRequest(chunks=chunks2, question="Question 2"),
                    CitationRequest(chunks=chunks3, question="Question 3")
                ]
                
                async with AsyncBookWyrmClient() as client:
                    # Process all requests concurrently
                    responses = await asyncio.gather(*[
                        client.get_citations(req) for req in requests
                    ])
                    
                    for i, response in enumerate(responses):
                        print(f"Request {i+1}: {response.total_citations} citations")
            ```
        """
        headers: Dict[str, str] = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/cite",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return CitationResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            status_code: int = e.response.status_code
            raise BookWyrmAPIError(f"API request failed: {e}", status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """Classify file content to determine file type and format asynchronously.
        
        This async method analyzes file content to determine format type, content type, MIME type,
        and other classification details. It supports both binary and text files, providing
        confidence scores and additional metadata about the detected format.

        Args:
            request: Classification request with base64-encoded content and optional filename hint

        Returns:
            Classification response with detected file type, confidence score, and additional details

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async file classification:
            
            ```python
            import base64
            from bookwyrm.models import ClassifyRequest
            
            async def classify_file():
                # Read file as binary and encode
                with open("document.pdf", "rb") as f:
                    content = base64.b64encode(f.read()).decode('ascii')
                
                request = ClassifyRequest(
                    content=content,
                    filename="document.pdf"
                )
                
                async with AsyncBookWyrmClient() as client:
                    response = await client.classify(request)
                    print(f"File type: {response.classification.format_type}")
                    print(f"Confidence: {response.classification.confidence:.2%}")
            
            asyncio.run(classify_file())
            ```
            
            Classify multiple files concurrently:
            
            ```python
            async def classify_multiple_files():
                files = ["doc1.pdf", "script.py", "data.json", "image.jpg"]
                
                async def classify_single(filename):
                    with open(filename, "rb") as f:
                        content = base64.b64encode(f.read()).decode('ascii')
                    
                    request = ClassifyRequest(content=content, filename=filename)
                    
                    async with AsyncBookWyrmClient() as client:
                        response = await client.classify(request)
                        return filename, response.classification
                
                results = await asyncio.gather(*[
                    classify_single(f) for f in files
                ])
                
                for filename, classification in results:
                    print(f"{filename}: {classification.content_type} ({classification.confidence:.1%})")
            ```
        """
        headers: Dict[str, str] = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if not request.content or request.content_encoding != "base64":
                raise BookWyrmAPIError("Content must be provided as base64-encoded data")
                
            # Decode base64 content and send as multipart form data
            import base64
            file_bytes: bytes = base64.b64decode(request.content)
            
            files: Dict[str, tuple] = {"file": (request.filename or "document", file_bytes)}
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/classify",
                files=files,
                headers=headers,
            )

            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return ClassifyResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def process_text(
        self, request: ProcessTextRequest
    ) -> AsyncIterator[StreamingPhrasalResponse]:
        """Process text using phrasal analysis with async streaming results.
        
        This async method breaks down text into meaningful phrases or chunks using NLP,
        supporting both direct text input and URLs. It can create fixed-size chunks
        or extract individual phrases with optional position information.

        Args:
            request: Text processing request with text/URL, chunking options, and format preferences

        Yields:
            StreamingPhrasalResponse: Union of progress updates and phrase/chunk results

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async phrasal processing:
            
            ```python
            from bookwyrm.models import ProcessTextRequest, ResponseFormat
            
            async def process_text_example():
                request = ProcessTextRequest(
                    text="Your text here",
                    response_format=ResponseFormat.WITH_OFFSETS
                )
                
                phrases = []
                async with AsyncBookWyrmClient() as client:
                    async for response in client.process_text(request):
                        if isinstance(response, (TextResult, TextSpanResult)):  # Phrase result
                            phrases.append(response)
                        elif isinstance(response, PhraseProgressUpdate):  # Progress
                            print(f"Progress: {response.message}")
                
                print(f"Extracted {len(phrases)} phrases")
            
            asyncio.run(process_text_example())
            ```
            
            Process multiple texts concurrently:
            
            ```python
            async def process_multiple_texts():
                requests = [
                    ProcessTextRequest(text=text1, chunk_size=500),
                    ProcessTextRequest(text=text2, chunk_size=500),
                    ProcessTextRequest(text_url="https://example.com/text.txt")
                ]
                
                async def process_single(req, name):
                    phrases = []
                    async with AsyncBookWyrmClient() as client:
                        async for response in client.process_text(req):
                            if isinstance(response, (TextResult, TextSpanResult)):
                                phrases.append(response)
                    return name, phrases
                
                results = await asyncio.gather(*[
                    process_single(req, f"Text{i+1}") for i, req in enumerate(requests)
                ])
                
                for name, phrases in results:
                    print(f"{name}: {len(phrases)} phrases")
            ```
        """
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/phrasal",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            if response_type == "progress":
                                yield PhraseProgressUpdate.model_validate(data)
                            elif response_type == "phrase":
                                # Determine if this is a TextResult or TextSpanResult based on presence of position data
                                if data.get("start_char") is not None and data.get("end_char") is not None:
                                    yield TextSpanResult.model_validate(data)
                                else:
                                    yield TextResult.model_validate(data)
                            else:
                                # Unknown response type, skip
                                continue
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_citations(
        self, request: CitationRequest
    ) -> AsyncIterator[StreamingCitationResponse]:
        """Stream citations as they are found with real-time progress updates.
        
        This async method provides real-time streaming of citation results, allowing you to
        process citations as they're found rather than waiting for all results. Useful
        for large datasets or when you want to show progress to users.

        Args:
            request: Citation request containing chunks/URL and question to answer

        Yields:
            StreamingCitationResponse: Union of progress updates, individual citations,
            final summary, or error messages

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async streaming:
            
            ```python
            async def stream_citations_example():
                async with AsyncBookWyrmClient() as client:
                    citations = []
                    async for response in client.stream_citations(request):
                        if isinstance(response, CitationProgressUpdate):  # Progress update
                            print(f"Progress: {response.message}")
                        elif isinstance(response, CitationStreamResponse):  # Citation found
                            citations.append(response.citation)
                            print(f"Found: {response.citation.text[:50]}...")
                        elif isinstance(response, CitationSummaryResponse):  # Summary
                            print(f"Complete: {response.total_citations} citations found")
            
            asyncio.run(stream_citations_example())
            ```
            
            Multiple concurrent streams:
            
            ```python
            async def handle_multiple_streams():
                async with AsyncBookWyrmClient() as client:
                    async def handle_stream_1():
                        async for response in client.stream_citations(request1):
                            if isinstance(response, CitationStreamResponse):
                                print(f"Stream 1: Found citation")
                    
                    async def handle_stream_2():
                        async for response in client.stream_citations(request2):
                            if isinstance(response, CitationStreamResponse):
                                print(f"Stream 2: Found citation")
                    
                    # Run both streams concurrently
                    await asyncio.gather(handle_stream_1(), handle_stream_2())
            ```
            
            With async context and error handling:
            
            ```python
            async def safe_streaming():
                try:
                    async with AsyncBookWyrmClient() as client:
                        async for response in client.stream_citations(request):
                            if isinstance(response, CitationErrorResponse):
                                print(f"Error: {response.error}")
                                break
                            # Process other response types...
                except BookWyrmAPIError as e:
                    print(f"API Error: {e}")
            ```
        """
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/cite/stream",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            if response_type == "progress":
                                yield CitationProgressUpdate.model_validate(data)
                            elif response_type == "citation":
                                yield CitationStreamResponse.model_validate(data)
                            elif response_type == "summary":
                                yield CitationSummaryResponse.model_validate(data)
                            elif response_type == "error":
                                yield CitationErrorResponse.model_validate(data)
                            else:
                                # Unknown response type, skip
                                continue
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def extract_pdf(self, request: PDFExtractRequest) -> PDFExtractResponse:
        """Extract structured data from a PDF file using OCR asynchronously.
        
        This async method extracts text elements from PDF files with position coordinates,
        confidence scores, and bounding box information. It supports both local files
        (base64-encoded) and remote URLs, with optional page range selection.

        Args:
            request: PDF extraction request with URL/content, optional page range, and filename

        Returns:
            PDF extraction response with structured page data, text elements, and metadata

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async PDF extraction:
            
            ```python
            from bookwyrm.models import PDFExtractRequest
            
            async def extract_pdf_example():
                request = PDFExtractRequest(
                    pdf_url="https://example.com/document.pdf",
                    start_page=1,
                    num_pages=5
                )
                
                async with AsyncBookWyrmClient() as client:
                    response = await client.extract_pdf(request)
                    print(f"Extracted {response.total_pages} pages")
                    
                    for page in response.pages:
                        print(f"Page {page.page_number}: {len(page.text_blocks)} elements")
            
            asyncio.run(extract_pdf_example())
            ```
            
            Extract multiple PDFs concurrently:
            
            ```python
            async def extract_multiple_pdfs():
                requests = [
                    PDFExtractRequest(pdf_url="https://example.com/doc1.pdf"),
                    PDFExtractRequest(pdf_url="https://example.com/doc2.pdf"),
                    PDFExtractRequest(pdf_url="https://example.com/doc3.pdf")
                ]
                
                async with AsyncBookWyrmClient() as client:
                    responses = await asyncio.gather(*[
                        client.extract_pdf(req) for req in requests
                    ])
                    
                    for i, response in enumerate(responses):
                        total_elements = sum(len(page.text_blocks) for page in response.pages)
                        print(f"PDF {i+1}: {response.total_pages} pages, {total_elements} elements")
            ```
        """
        headers = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if request.pdf_content:
                # Handle base64-encoded file content using form data
                import base64
                pdf_bytes: bytes = base64.b64decode(request.pdf_content)
                
                files: Dict[str, tuple] = {"file": (request.filename or "document.pdf", pdf_bytes, "application/pdf")}
                data: Dict[str, Union[int, str]] = {}
                if request.start_page is not None:
                    data["start_page"] = request.start_page
                if request.num_pages is not None:
                    data["num_pages"] = request.num_pages
                    
                response: httpx.Response = await self.client.post(
                    f"{self.base_url}/extract-structure",
                    files=files,
                    data=data,
                    headers=headers,
                )
            elif request.pdf_url:
                # Handle URL using JSON endpoint
                headers["Content-Type"] = "application/json"
                json_data: Dict[str, Union[str, int]] = {"pdf_url": request.pdf_url}
                if request.start_page is not None:
                    json_data["start_page"] = request.start_page
                if request.num_pages is not None:
                    json_data["num_pages"] = request.num_pages
                    
                response = await self.client.post(
                    f"{self.base_url}/extract-structure-json",
                    json=json_data,
                    headers=headers,
                )
            else:
                raise BookWyrmAPIError("Either pdf_url or pdf_content must be provided")

            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return PDFExtractResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_extract_pdf(
        self, request: PDFExtractRequest
    ) -> AsyncIterator[StreamingPDFResponse]:
        """Stream PDF extraction with real-time progress updates asynchronously.
        
        This async method provides real-time streaming of PDF extraction progress, yielding
        metadata, individual page results, and completion status. Useful for large PDFs
        where you want to show progress or process pages as they become available.

        Args:
            request: PDF extraction request with URL/content, optional page range, and filename

        Yields:
            StreamingPDFResponse: Union of metadata, page responses, page errors, completion, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async streaming PDF extraction:
            
            ```python
            async def stream_extract_pdf_example():
                pages = []
                async with AsyncBookWyrmClient() as client:
                    async for response in client.stream_extract_pdf(request):
                        if isinstance(response, PDFStreamPageResponse):  # Page extracted
                            pages.append(response.page_data)
                            print(f"Page {response.document_page}: {len(response.page_data.text_blocks)} elements")
                        elif isinstance(response, PDFStreamMetadata):  # Metadata
                            print(f"Processing {response.total_pages} pages")
                        elif isinstance(response, PDFStreamPageError):
                            print(f"Error on page {response.document_page}: {response.error}")
                
                print(f"Extracted {len(pages)} pages total")
            
            asyncio.run(stream_extract_pdf_example())
            ```
            
            Process multiple PDFs concurrently with streaming:
            
            ```python
            async def stream_multiple_pdfs():
                requests = [
                    PDFExtractRequest(pdf_url="https://example.com/doc1.pdf"),
                    PDFExtractRequest(pdf_url="https://example.com/doc2.pdf")
                ]
                
                async def stream_single_pdf(req, name):
                    pages = []
                    async with AsyncBookWyrmClient() as client:
                        async for response in client.stream_extract_pdf(req):
                            if isinstance(response, PDFStreamPageResponse):
                                pages.append(response.page_data)
                                print(f"{name} - Page {response.document_page} extracted")
                    return name, pages
                
                results = await asyncio.gather(*[
                    stream_single_pdf(req, f"PDF{i+1}") for i, req in enumerate(requests)
                ])
                
                for name, pages in results:
                    print(f"{name}: {len(pages)} pages extracted")
            ```
            
            Real-time page processing:
            
            ```python
            async def process_pages_realtime():
                async def process_page(page_data):
                    # Process page immediately when received
                    text = " ".join(element.text for element in page_data.text_blocks)
                    # Could save to database, send to another service, etc.
                    return len(text)
                
                total_chars = 0
                async with AsyncBookWyrmClient() as client:
                    async for response in client.stream_extract_pdf(request):
                        if isinstance(response, PDFStreamPageResponse):
                            char_count = await process_page(response.page_data)
                            total_chars += char_count
                            print(f"Processed page {response.document_page}: {char_count} chars")
                
                print(f"Total characters processed: {total_chars}")
            ```
        """
        headers = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if request.pdf_content:
                # Handle base64-encoded file content using form data
                import base64
                pdf_bytes = base64.b64decode(request.pdf_content)
                
                files = {"file": (request.filename or "document.pdf", pdf_bytes, "application/pdf")}
                data = {}
                if request.start_page is not None:
                    data["start_page"] = request.start_page
                if request.num_pages is not None:
                    data["num_pages"] = request.num_pages
                    
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/extract-structure-stream",
                    files=files,
                    data=data,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line and line.strip():
                            try:
                                data: Dict[str, Any] = json.loads(line)
                                response_type: Optional[str] = data.get("type")

                                if response_type == "metadata":
                                    yield PDFStreamMetadata.model_validate(data)
                                elif response_type == "page":
                                    yield PDFStreamPageResponse.model_validate(data)
                                elif response_type == "page_error":
                                    yield PDFStreamPageError.model_validate(data)
                                elif response_type == "complete":
                                    yield PDFStreamComplete.model_validate(data)
                                elif response_type == "error":
                                    yield PDFStreamError.model_validate(data)
                                elif response_type == "keepalive":
                                    # Ignore keepalive messages
                                    continue
                                else:
                                    # Unknown response type, skip
                                    continue
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
            elif request.pdf_url:
                # Handle URL using JSON endpoint
                headers["Content-Type"] = "application/json"
                json_data = {"pdf_url": request.pdf_url}
                if request.start_page is not None:
                    json_data["start_page"] = request.start_page
                if request.num_pages is not None:
                    json_data["num_pages"] = request.num_pages
                    
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/extract-structure-stream-json",
                    json=json_data,
                    headers=headers,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                response_type = data.get("type")

                                if response_type == "metadata":
                                    yield PDFStreamMetadata.model_validate(data)
                                elif response_type == "page":
                                    yield PDFStreamPageResponse.model_validate(data)
                                elif response_type == "page_error":
                                    yield PDFStreamPageError.model_validate(data)
                                elif response_type == "complete":
                                    yield PDFStreamComplete.model_validate(data)
                                elif response_type == "error":
                                    yield PDFStreamError.model_validate(data)
                                elif response_type == "keepalive":
                                    # Ignore keepalive messages
                                    continue
                                else:
                                    # Unknown response type, skip
                                    continue
                            except json.JSONDecodeError:
                                # Skip malformed JSON lines
                                continue
            else:
                raise BookWyrmAPIError("Either pdf_url or pdf_content must be provided")

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def close(self) -> None:
        """Close the client."""
        await self.client.aclose()

    async def __aenter__(self) -> "AsyncBookWyrmClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Async context manager exit."""
        await self.close()

    async def summarize(self, request: SummarizeRequest) -> SummaryResponse:
        """Get a summary of the provided content using hierarchical summarization.
        
        This async method performs intelligent summarization of text content, supporting both
        plain text summaries and structured output using Pydantic models. It can handle
        large documents through hierarchical chunking and summarization.

        Args:
            request: Summarization request with content, options, and optional structured output settings

        Returns:
            Summary response containing the generated summary, metadata, and optional debug information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async summarization:
            
            ```python
            from bookwyrm.models import SummarizeRequest
            
            async def summarize_text():
                with open("book_phrases.jsonl", "r") as f:
                    content = f.read()
                
                request = SummarizeRequest(
                    content=content,
                    max_tokens=10000,
                    debug=True
                )
                
                async with AsyncBookWyrmClient() as client:
                    response = await client.summarize(request)
                    print(response.summary)
                    print(f"Used {response.levels_used} levels")
            
            asyncio.run(summarize_text())
            ```
            
            Concurrent summarization of multiple documents:
            
            ```python
            async def summarize_multiple():
                requests = [
                    SummarizeRequest(content=content1, max_tokens=5000),
                    SummarizeRequest(content=content2, max_tokens=5000),
                    SummarizeRequest(content=content3, max_tokens=5000)
                ]
                
                async with AsyncBookWyrmClient() as client:
                    summaries = await asyncio.gather(*[
                        client.summarize(req) for req in requests
                    ])
                    
                    for i, summary in enumerate(summaries):
                        print(f"Document {i+1} summary: {summary.summary[:100]}...")
            ```
        """
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return SummaryResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            status_code: int = e.response.status_code
            raise BookWyrmAPIError(f"API request failed: {e}", status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_summarize(
        self, request: SummarizeRequest
    ) -> AsyncIterator[StreamingSummarizeResponse]:
        """Stream summarization progress and results with real-time updates.
        
        This async method provides real-time streaming of summarization progress, including
        hierarchical processing updates, retry attempts, and final results. Useful for
        long-running summarization tasks where you want to show progress to users.

        Args:
            request: Summarization request with content, options, and optional structured output settings

        Yields:
            StreamingSummarizeResponse: Union of progress updates, final summary, rate limit messages,
            structural error messages, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)
            
        Examples:
            Basic async streaming summarization:
            
            ```python
            async def stream_summarize_example():
                async with AsyncBookWyrmClient() as client:
                    final_result = None
                    async for response in client.stream_summarize(request):
                        if isinstance(response, SummarizeProgressUpdate):  # Progress update
                            print(f"Progress: {response.message}")
                        elif isinstance(response, SummaryResponse):  # Final summary
                            final_result = response
                            print("Summary complete!")
                    
                    if final_result:
                        print(final_result.summary)
            
            asyncio.run(stream_summarize_example())
            ```
            
            Concurrent streaming of multiple summarizations:
            
            ```python
            async def concurrent_streaming():
                async with AsyncBookWyrmClient() as client:
                    async def handle_summarization(req, name):
                        async for response in client.stream_summarize(req):
                            if isinstance(response, SummarizeProgressUpdate):
                                print(f"{name}: {response.message}")
                            elif isinstance(response, SummaryResponse):
                                print(f"{name}: Complete!")
                    
                    # Run multiple summarizations concurrently
                    await asyncio.gather(
                        handle_summarization(request1, "Doc1"),
                        handle_summarization(request2, "Doc2"),
                        handle_summarization(request3, "Doc3")
                    )
            ```
            
            With progress tracking and error handling:
            
            ```python
            async def advanced_streaming():
                async with AsyncBookWyrmClient() as client:
                    try:
                        async for response in client.stream_summarize(request):
                            if isinstance(response, (RateLimitMessage, StructuralErrorMessage)):  # Retry message
                                print(f"Retry {response.attempt}/{response.max_attempts}")
                            elif isinstance(response, SummarizeProgressUpdate):  # Progress
                                progress = response.chunks_processed / response.total_chunks
                                print(f"Level {response.current_level}: {progress:.1%}")
                            elif isinstance(response, SummarizeErrorResponse):  # Error
                                print(f"Error: {response.error}")
                                break
                    except BookWyrmAPIError as e:
                        print(f"API Error: {e}")
            ```
        """
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            if response_type == "progress":
                                yield SummarizeProgressUpdate.model_validate(data)
                            elif response_type == "summary":
                                yield SummaryResponse.model_validate(data)
                            elif response_type == "error":
                                yield SummarizeErrorResponse.model_validate(data)
                            elif response_type == "rate_limit":
                                yield RateLimitMessage.model_validate(data)
                            elif response_type == "structural_error":
                                yield StructuralErrorMessage.model_validate(data)
                            else:
                                # Unknown response type, skip
                                continue
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")
