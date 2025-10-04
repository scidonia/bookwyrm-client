"""Asynchronous client for BookWyrm API."""

import json
import os
import platform
from typing import AsyncIterator, Optional, Union, Dict, Any, List, Literal
import httpx
from pathlib import Path

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
    ResponseFormat,
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
    TextSpan,
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
        timeout: Optional[float] = 30.0,
    ) -> None:
        """Initialize the async BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API. Defaults to "https://api.bookwyrm.ai:443"
            api_key: API key for authentication. If not provided, will attempt to read
                from BOOKWYRM_API_KEY environment variable
            timeout: Request timeout in seconds. Defaults to 30.0 seconds. Set to None for no timeout.

        Examples:
            ```python
            # Basic initialization
            client = AsyncBookWyrmClient()

            # With explicit API key
            client = AsyncBookWyrmClient(api_key="your-api-key")

            # With custom endpoint and timeout
            client = AsyncBookWyrmClient(
                base_url="https://localhost:8000",
                api_key="dev-key",
                timeout=60.0
            )

            # With no timeout
            client = AsyncBookWyrmClient(timeout=None)
            ```
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key or os.getenv("BOOKWYRM_API_KEY")
        self.timeout: Optional[float] = timeout
        self.client: httpx.AsyncClient = httpx.AsyncClient(timeout=timeout)

    async def get_citations(
        self,
        *,
        chunks: Optional[List[TextSpan]] = None,
        jsonl_content: Optional[str] = None,
        jsonl_url: Optional[str] = None,
        question: str,
        start: Optional[int] = 0,
        limit: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = 1000,
    ) -> CitationResponse:
        """Get citations for a question from text chunks asynchronously.

        This async method finds relevant citations that answer a specific question by analyzing
        the provided text chunks. Each citation includes a quality score (0-4) and reasoning
        for why it's relevant to the question.

        Args:
            chunks: List of text chunks to search
            jsonl_content: Raw JSONL content as string
            jsonl_url: URL to fetch JSONL content from
            question: The question to find citations for
            start: Starting chunk index (0-based)
            limit: Maximum number of chunks to process
            max_tokens_per_chunk: Maximum tokens per chunk

        Returns:
            Citation response with found citations, total count, and usage information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async citation finding:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import TextSpan

            async def find_citations():
                # Create some example chunks
                chunks = [
                    TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
                    TextSpan(text="Water molecules are polar.", start_char=43, end_char=69),
                    TextSpan(text="Plants appear green due to chlorophyll.", start_char=70, end_char=109)
                ]

                async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                    response = await client.get_citations(
                        chunks=chunks,
                        question="Why is the sky blue?"
                    )
                    print(f"Found {response.total_citations} citations")

                    for citation in response.citations:
                        print(f"Quality: {citation.quality}/4")
                        print(f"Text: {citation.text}")

            asyncio.run(find_citations())
            ```

            Concurrent citation requests:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import TextSpan

            async def concurrent_citations():
                # Create example chunks for different topics
                chunks1 = [TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42)]
                chunks2 = [TextSpan(text="Water boils at 100°C at sea level.", start_char=0, end_char=34)]
                chunks3 = [TextSpan(text="Photosynthesis converts CO2 to oxygen.", start_char=0, end_char=38)]

                async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                    # Process all requests concurrently
                    responses = await asyncio.gather(
                        client.get_citations(chunks=chunks1, question="Why is the sky blue?"),
                        client.get_citations(chunks=chunks2, question="At what temperature does water boil?"),
                        client.get_citations(chunks=chunks3, question="What does photosynthesis produce?")
                    )

                    for i, response in enumerate(responses):
                        print(f"Request {i+1}: {response.total_citations} citations")

            asyncio.run(concurrent_citations())
            ```
        """
        request = CitationRequest(
            chunks=chunks,
            jsonl_content=jsonl_content,
            jsonl_url=jsonl_url,
            question=question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )
        headers: Dict[str, str] = {
            **DEFAULT_HEADERS,
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/cite",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return CitationResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            status_code: int = e.response.status_code
            raise BookWyrmAPIError(f"API request failed: {e}", status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def classify(
        self,
        *,
        content: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_encoding: str = "raw",
    ) -> ClassifyResponse:
        """Classify file content to determine file type and format asynchronously.

        This async method analyzes file content to determine format type, content type, MIME type,
        and other classification details. It supports both binary and text files, providing
        confidence scores and additional metadata about the detected format.

        Args:
            content: File content as string (raw text or base64-encoded)
            content_bytes: Raw file bytes
            filename: Optional filename hint for classification
            content_encoding: Content encoding format ("raw" for plain text, "base64" for encoded)

        Returns:
            Classification response with detected file type, confidence score, and additional details

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async file classification:

            ```python
            async def classify_file():
                # Read file as binary
                with open("document.pdf", "rb") as f:
                    file_bytes = f.read()

                async with AsyncBookWyrmClient() as client:
                    response = await client.classify(
                        content_bytes=file_bytes,
                        filename="document.pdf"
                    )
                    print(f"File type: {response.classification.format_type}")
                    print(f"Confidence: {response.classification.confidence:.2%}")

            asyncio.run(classify_file())
            ```

            Classify multiple files concurrently:

            ```python
            async def classify_multiple_files():
                files = ["doc1.pdf", "script.py", "data.json", "image.jpg"]

                async def classify_single(filename):
                    file_path = Path(filename)
                    return filename, await client.classify(
                        content_bytes=file_path.read_bytes(),
                        filename=file_path.name
                    )

                async with AsyncBookWyrmClient() as client:
                    results = await asyncio.gather(*[
                        classify_single(f) for f in files
                    ])

                    for filename, response in results:
                        print(f"{filename}: {response.classification.content_type} ({response.classification.confidence:.1%})")
            ```
        """
        if content is None and content_bytes is None:
            raise ValueError("Either content or content_bytes is required")

        request = ClassifyRequest(
            content=content,
            content_bytes=content_bytes,
            filename=filename,
            content_encoding=content_encoding,
        )
        headers: Dict[str, str] = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Handle marshalling at API level
            if request.content_bytes is not None:
                file_bytes: bytes = request.content_bytes
            elif request.content is not None:
                if request.content_encoding == "base64":
                    # Decode base64 content and send as multipart form data
                    import base64

                    file_bytes = base64.b64decode(request.content)
                else:
                    # Handle raw text content
                    file_bytes = request.content.encode("utf-8")
            else:
                raise BookWyrmAPIError(
                    "Either content or content_bytes must be provided"
                )

            files: Dict[str, tuple] = {
                "file": (request.filename or "document", file_bytes)
            }
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/classify",
                files=files,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return ClassifyResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_process_text(
        self,
        *,
        text: Optional[str] = None,
        text_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        response_format: Union[
            ResponseFormat, Literal["with_offsets", "offsets", "text_only", "text"]
        ] = ResponseFormat.WITH_OFFSETS,
        # Boolean flags for response format
        offsets: Optional[bool] = None,
        text_only: Optional[bool] = None,
    ) -> AsyncIterator[StreamingPhrasalResponse]:
        """Stream text processing using phrasal analysis with async real-time results.

        This async method breaks down text into meaningful phrases or chunks using NLP,
        supporting both direct text input and URLs. It can create fixed-size chunks
        or extract individual phrases with optional position information.

        Args:
            text: Text content to process
            text_url: URL to fetch text from
            chunk_size: Optional chunk size for fixed-size chunking
            response_format: Response format - use ResponseFormat enum, "with_offsets"/"offsets", or "text_only"/"text"
            offsets: Set to True for WITH_OFFSETS format (boolean flag)
            text_only: Set to True for TEXT_ONLY format (boolean flag)

        Yields:
            StreamingPhrasalResponse: Union of progress updates and phrase/chunk results

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async phrasal processing:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import ResponseFormat, TextResult, TextSpanResult, PhraseProgressUpdate

            async def process_text_example():
                text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."
                phrases = []
                async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                    async for response in client.stream_process_text(
                        text=text,
                        offsets=True  # or response_format="with_offsets" or ResponseFormat.WITH_OFFSETS
                    ):
                        if isinstance(response, (TextResult, TextSpanResult)):  # Phrase result
                            phrases.append(response)
                        elif isinstance(response, PhraseProgressUpdate):  # Progress
                            print(f"Progress: {response.message}")

                print(f"Extracted {len(phrases)} phrases")

            asyncio.run(process_text_example())
            ```

            Process multiple texts concurrently:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import TextResult, TextSpanResult

            async def process_multiple_texts():
                async def process_single(text, name):
                    phrases = []
                    async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                        async for response in client.stream_process_text(
                            text=text,
                            chunk_size=500
                        ):
                            if isinstance(response, (TextResult, TextSpanResult)):
                                phrases.append(response)
                    return name, phrases

                results = await asyncio.gather(
                    process_single(text1, "Text1"),
                    process_single(text2, "Text2"),
                )

                for name, phrases in results:
                    print(f"{name}: {len(phrases)} phrases")

            asyncio.run(process_multiple_texts())
            ```

            Process text from URL:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import TextResult, TextSpanResult

            async def process_from_url():
                async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                    phrases = []
                    async for response in client.stream_process_text(
                        text_url="https://www.gutenberg.org/files/11/11-0.txt",
                        chunk_size=2000,
                        text_only=True
                    ):
                        if isinstance(response, (TextResult, TextSpanResult)):
                            phrases.append(response)

                    print(f"Processed {len(phrases)} phrases from URL")

            asyncio.run(process_from_url())
            ```
        """
        if text is None and text_url is None:
            raise ValueError("Either text or text_url is required")

        # Handle boolean flags for response format
        boolean_flags = [offsets, text_only]
        true_flags = [flag for flag in boolean_flags if flag is True]

        if len(true_flags) > 1:
            raise ValueError("Only one response format flag can be True")

        if len(true_flags) == 1:
            if offsets:
                response_format = ResponseFormat.WITH_OFFSETS
            elif text_only:
                response_format = ResponseFormat.TEXT_ONLY

        # Convert string to enum if needed
        if isinstance(response_format, str):
            if response_format.lower() in ("with_offsets", "offsets"):
                response_format = ResponseFormat.WITH_OFFSETS
            elif response_format.lower() in ("text_only", "text"):
                response_format = ResponseFormat.TEXT_ONLY
            else:
                raise ValueError(
                    f"Invalid response_format: {response_format}. Use 'with_offsets'/'offsets' or 'text_only'/'text'"
                )

        request = ProcessTextRequest(
            text=text,
            text_url=text_url,
            chunk_size=chunk_size,
            response_format=response_format,
        )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/phrasal",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            match response_type:
                                case "progress":
                                    yield PhraseProgressUpdate.model_validate(data)
                                case "text":
                                    yield TextResult.model_validate(data)
                                case "text_span":
                                    yield TextSpanResult.model_validate(data)
                                case _:
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
        self,
        *,
        chunks: Optional[List[TextSpan]] = None,
        jsonl_content: Optional[str] = None,
        jsonl_url: Optional[str] = None,
        question: str,
        start: Optional[int] = 0,
        limit: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = 1000,
    ) -> AsyncIterator[StreamingCitationResponse]:
        """Stream citations as they are found with real-time progress updates.

        This async method provides real-time streaming of citation results, allowing you to
        process citations as they're found rather than waiting for all results. Useful
        for large datasets or when you want to show progress to users.

        Args:
            chunks: List of text chunks to search
            jsonl_content: Raw JSONL content as string
            jsonl_url: URL to fetch JSONL content from
            question: The question to find citations for
            start: Starting chunk index (0-based)
            limit: Maximum number of chunks to process
            max_tokens_per_chunk: Maximum tokens per chunk

        Yields:
            StreamingCitationResponse: Union of progress updates, individual citations,
            final summary, or error messages

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async streaming with function arguments:

            ```python
            import asyncio
            from bookwyrm import AsyncBookWyrmClient
            from bookwyrm.models import TextSpan, CitationProgressUpdate, CitationStreamResponse, CitationSummaryResponse

            async def stream_citations_example():
                # Create some example chunks
                chunks = [
                    TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
                    TextSpan(text="Water molecules are polar.", start_char=43, end_char=69),
                    TextSpan(text="Plants appear green due to chlorophyll.", start_char=70, end_char=109)
                ]

                async with AsyncBookWyrmClient(api_key="your-api-key") as client:
                    citations = []
                    async for response in client.stream_citations(
                        chunks=chunks,
                        question="Why is the sky blue?"
                    ):
                        if isinstance(response, CitationProgressUpdate):  # Progress update
                            print(f"Progress: {response.message}")
                        elif isinstance(response, CitationStreamResponse):  # Citation found
                            citations.append(response.citation)
                            print(f"Found: {response.citation.text[:50]}...")
                        elif isinstance(response, CitationSummaryResponse):  # Summary
                            print(f"Complete: {response.total_citations} citations found")

            asyncio.run(stream_citations_example())
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import CitationRequest

            request = CitationRequest(
                chunks=chunks,
                question="Why is the sky blue?"
            )

            async for response in client.stream_citations(request):
                # Process responses...
            ```
        """
        # Handle empty chunks list - return empty response immediately
        if chunks is not None and len(chunks) == 0:
            from .models import UsageInfo
            yield CitationSummaryResponse(
                total_citations=0,
                chunks_processed=0,
                token_chunks_processed=0,
                start_offset=0,
                usage=UsageInfo(
                    tokens_processed=0,
                    chunks_processed=0,
                    estimated_cost=None,
                    remaining_credits=0.0,
                ),
            )
            return

        request = CitationRequest(
            chunks=chunks,
            jsonl_content=jsonl_content,
            jsonl_url=jsonl_url,
            question=question,
            start=start,
            limit=limit,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/cite/stream",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            match response_type:
                                case "progress":
                                    yield CitationProgressUpdate.model_validate(data)
                                case "citation":
                                    yield CitationStreamResponse.model_validate(data)
                                case "summary":
                                    yield CitationSummaryResponse.model_validate(data)
                                case "error":
                                    yield CitationErrorResponse.model_validate(data)
                                case _:
                                    # Unknown response type, skip
                                    continue
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def extract_pdf(
        self,
        *,
        pdf_url: Optional[str] = None,
        pdf_content: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        start_page: Optional[int] = None,
        num_pages: Optional[int] = None,
    ) -> PDFExtractResponse:
        """Extract structured data from a PDF file using OCR asynchronously.

        This async method extracts text elements from PDF files with position coordinates,
        confidence scores, and bounding box information. It supports both local files
        (base64-encoded) and remote URLs, with optional page range selection.

        Args:
            pdf_url: URL to PDF file
            pdf_content: Base64 encoded PDF content
            pdf_bytes: Raw PDF bytes (will be encoded to base64)
            filename: Optional filename hint
            start_page: 1-based page number to start from
            num_pages: Number of pages to process from start_page

        Returns:
            PDF extraction response with structured page data, text elements, and metadata

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async PDF extraction with function arguments:

            ```python
            async def extract_pdf_example():
                with open("document.pdf", "rb") as f:
                    pdf_bytes = f.read()

                async with AsyncBookWyrmClient() as client:
                    response = await client.extract_pdf(
                        pdf_bytes=pdf_bytes,
                        filename="document.pdf",
                        start_page=1,
                        num_pages=5
                    )
                    print(f"Extracted {response.total_pages} pages")

                    for page in response.pages:
                        print(f"Page {page.page_number}: {len(page.text_blocks)} elements")

            asyncio.run(extract_pdf_example())
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import PDFExtractRequest

            request = PDFExtractRequest(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            )

            response = await client.extract_pdf(request)
            ```
        """
        sources = [pdf_url, pdf_content, pdf_bytes]
        provided_sources = [s for s in sources if s is not None]
        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"
            )

        request = PDFExtractRequest(
            pdf_url=pdf_url,
            pdf_content=pdf_content,
            pdf_bytes=pdf_bytes,
            filename=filename,
            start_page=start_page,
            num_pages=num_pages,
        )
        headers = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if request.pdf_content or request.pdf_bytes:
                # Handle marshalling at API level
                if request.pdf_bytes is not None:
                    pdf_bytes: bytes = request.pdf_bytes
                else:
                    # Handle base64-encoded file content using form data
                    import base64

                    pdf_bytes = base64.b64decode(request.pdf_content)

                files: Dict[str, tuple] = {
                    "file": (
                        request.filename or "document.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                }
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
                    timeout=self.timeout,
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
                    timeout=self.timeout,
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
        self,
        *,
        pdf_url: Optional[str] = None,
        pdf_content: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        start_page: Optional[int] = None,
        num_pages: Optional[int] = None,
    ) -> AsyncIterator[StreamingPDFResponse]:
        """Stream PDF extraction with real-time progress updates asynchronously.

        This async method provides real-time streaming of PDF extraction progress, yielding
        metadata, individual page results, and completion status. Useful for large PDFs
        where you want to show progress or process pages as they become available.

        Args:
            pdf_url: URL to PDF file
            pdf_content: Base64 encoded PDF content
            pdf_bytes: Raw PDF bytes (will be encoded to base64)
            filename: Optional filename hint
            start_page: 1-based page number to start from
            num_pages: Number of pages to process from start_page

        Yields:
            StreamingPDFResponse: Union of metadata, page responses, page errors, completion, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async streaming PDF extraction with function arguments:

            ```python
            async def stream_extract_pdf_example():
                pages = []
                async with AsyncBookWyrmClient() as client:
                    async for response in client.stream_extract_pdf(
                        pdf_bytes=pdf_bytes,
                        filename="document.pdf"
                    ):
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

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import PDFExtractRequest

            request = PDFExtractRequest(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            )

            async for response in client.stream_extract_pdf(request):
                # Process responses...
            ```
        """
        sources = [pdf_url, pdf_content, pdf_bytes]
        provided_sources = [s for s in sources if s is not None]
        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"
            )

        request = PDFExtractRequest(
            pdf_url=pdf_url,
            pdf_content=pdf_content,
            pdf_bytes=pdf_bytes,
            filename=filename,
            start_page=start_page,
            num_pages=num_pages,
        )
        headers = {**DEFAULT_HEADERS}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if request.pdf_content or request.pdf_bytes:
                # Handle marshalling at API level
                if request.pdf_bytes is not None:
                    pdf_bytes = request.pdf_bytes
                else:
                    # Handle base64-encoded file content using form data
                    import base64

                    pdf_bytes = base64.b64decode(request.pdf_content)

                files = {
                    "file": (
                        request.filename or "document.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                }
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
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line and line.strip():
                            try:
                                data: Dict[str, Any] = json.loads(line)
                                response_type: Optional[str] = data.get("type")

                                match response_type:
                                    case "metadata":
                                        yield PDFStreamMetadata.model_validate(data)
                                    case "page":
                                        yield PDFStreamPageResponse.model_validate(data)
                                    case "page_error":
                                        yield PDFStreamPageError.model_validate(data)
                                    case "complete":
                                        yield PDFStreamComplete.model_validate(data)
                                    case "error":
                                        yield PDFStreamError.model_validate(data)
                                    case "keepalive":
                                        # Ignore keepalive messages
                                        continue
                                    case _:
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
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                response_type = data.get("type")

                                match response_type:
                                    case "metadata":
                                        yield PDFStreamMetadata.model_validate(data)
                                    case "page":
                                        yield PDFStreamPageResponse.model_validate(data)
                                    case "page_error":
                                        yield PDFStreamPageError.model_validate(data)
                                    case "complete":
                                        yield PDFStreamComplete.model_validate(data)
                                    case "error":
                                        yield PDFStreamError.model_validate(data)
                                    case "keepalive":
                                        # Ignore keepalive messages
                                        continue
                                    case _:
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

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def summarize(
        self,
        *,
        content: Optional[str] = None,
        url: Optional[str] = None,
        phrases: Optional[List[TextSpan]] = None,
        max_tokens: int = 10000,
        debug: bool = False,
    ) -> SummaryResponse:
        """Get a summary of the provided content using hierarchical summarization.

        This async method performs intelligent summarization of text content, supporting both
        plain text summaries and structured output using Pydantic models. It can handle
        large documents through hierarchical chunking and summarization.

        Args:
            content: Text content to summarize
            url: URL to fetch content from
            phrases: List of text phrases to summarize
            max_tokens: Maximum tokens for chunking (default: 10000)
            debug: Include intermediate summaries in response

        Returns:
            Summary response containing the generated summary, metadata, and optional debug information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async summarization with function arguments:

            ```python
            async def summarize_text():
                with open("book_phrases.jsonl", "r") as f:
                    content = f.read()

                async with AsyncBookWyrmClient() as client:
                    response = await client.summarize(
                        content=content,
                        max_tokens=10000,
                        debug=True
                    )
                    print(response.summary)
                    print(f"Used {response.levels_used} levels")

            asyncio.run(summarize_text())
            ```

            Concurrent summarization of multiple documents:

            ```python
            async def summarize_multiple():
                async with AsyncBookWyrmClient() as client:
                    summaries = await asyncio.gather(
                        client.summarize(content=content1, max_tokens=5000),
                        client.summarize(content=content2, max_tokens=5000),
                        client.summarize(content=content3, max_tokens=5000)
                    )

                    for i, summary in enumerate(summaries):
                        print(f"Document {i+1} summary: {summary.summary[:100]}...")
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import SummarizeRequest

            request = SummarizeRequest(
                content=content,
                max_tokens=10000,
                debug=True
            )

            response = await client.summarize(request)
            ```
        """
        sources = [content, url, phrases]
        provided_sources = [s for s in sources if s is not None]
        if len(provided_sources) != 1:
            raise ValueError("Exactly one of content, url, or phrases must be provided")

        request = SummarizeRequest(
            content=content,
            url=url,
            phrases=phrases,
            max_tokens=max_tokens,
            debug=debug,
        )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: httpx.Response = await self.client.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                timeout=self.timeout,
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
        self,
        *,
        content: Optional[str] = None,
        url: Optional[str] = None,
        phrases: Optional[List[TextSpan]] = None,
        max_tokens: int = 10000,
        debug: bool = False,
    ) -> AsyncIterator[StreamingSummarizeResponse]:
        """Stream summarization progress and results with real-time updates.

        This async method provides real-time streaming of summarization progress, including
        hierarchical processing updates, retry attempts, and final results. Useful for
        long-running summarization tasks where you want to show progress to users.

        Args:
            content: Text content to summarize
            url: URL to fetch content from
            phrases: List of text phrases to summarize
            max_tokens: Maximum tokens for chunking (default: 10000)
            debug: Include intermediate summaries in response

        Yields:
            StreamingSummarizeResponse: Union of progress updates, final summary, rate limit messages,
            structural error messages, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic async streaming summarization with function arguments:

            ```python
            async def stream_summarize_example():
                async with AsyncBookWyrmClient() as client:
                    final_result = None
                    async for response in client.stream_summarize(
                        content=content,
                        max_tokens=5000,
                        debug=True
                    ):
                        if isinstance(response, SummarizeProgressUpdate):  # Progress update
                            print(f"Progress: {response.message}")
                        elif isinstance(response, SummaryResponse):  # Final summary
                            final_result = response
                            print("Summary complete!")

                    if final_result:
                        print(final_result.summary)

            asyncio.run(stream_summarize_example())
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import SummarizeRequest

            request = SummarizeRequest(
                content=content,
                max_tokens=5000,
                debug=True
            )

            async for response in client.stream_summarize(request):
                # Process responses...
            ```
        """
        sources = [content, url, phrases]
        provided_sources = [s for s in sources if s is not None]
        if len(provided_sources) != 1:
            raise ValueError("Exactly one of content, url, or phrases must be provided")

        request = SummarizeRequest(
            content=content,
            url=url,
            phrases=phrases,
            max_tokens=max_tokens,
            debug=debug,
        )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line and line.strip():
                        try:
                            data: Dict[str, Any] = json.loads(line)
                            response_type: Optional[str] = data.get("type")

                            match response_type:
                                case "progress":
                                    yield SummarizeProgressUpdate.model_validate(data)
                                case "summary":
                                    yield SummaryResponse.model_validate(data)
                                case "error":
                                    yield SummarizeErrorResponse.model_validate(data)
                                case "rate_limit":
                                    yield RateLimitMessage.model_validate(data)
                                case "structural_error":
                                    yield StructuralErrorMessage.model_validate(data)
                                case _:
                                    # Unknown response type, skip
                                    continue
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")
