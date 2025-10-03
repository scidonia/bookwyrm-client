"""Synchronous client for BookWyrm API."""

import json
import os
import platform
from typing import List, Iterator, Optional, Union, Dict, Any
import requests
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


# User-Agent and client version headers
UA = f"bookwyrm-client/{__version__} (python/{platform.python_version()}; {platform.system()})"

DEFAULT_HEADERS = {
    "User-Agent": UA,
}


class BookWyrmClientError(Exception):
    """Base exception for BookWyrm client errors."""

    pass


class BookWyrmAPIError(BookWyrmClientError):
    """Exception for API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class BookWyrmClient:
    """Synchronous client for BookWyrm API.

    The synchronous client provides access to all BookWyrm API endpoints using the
    `requests` library. It supports both streaming and non-streaming operations,
    automatic session management, and comprehensive error handling.

    Examples:
        Basic client initialization:

        ```python
        from bookwyrm import BookWyrmClient

        # Using environment variable for API key
        client = BookWyrmClient()

        # Explicit API key
        client = BookWyrmClient(api_key="your-api-key")

        # Custom base URL
        client = BookWyrmClient(
            base_url="https://custom-api.example.com",
            api_key="your-api-key"
        )
        ```

        Context manager usage for automatic cleanup:

        ```python
        with BookWyrmClient() as client:
            response = client.get_citations(request)
            # Client is automatically closed when exiting the context
        ```
    """

    def __init__(
        self,
        base_url: str = "https://api.bookwyrm.ai:443",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API. Defaults to "https://api.bookwyrm.ai:443"
            api_key: API key for authentication. If not provided, will attempt to read
                from BOOKWYRM_API_KEY environment variable

        Examples:
            ```python
            # Basic initialization
            client = BookWyrmClient()

            # With explicit API key
            client = BookWyrmClient(api_key="your-api-key")

            # With custom endpoint
            client = BookWyrmClient(
                base_url="https://localhost:8000",
                api_key="dev-key"
            )
            ```
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key or os.getenv("BOOKWYRM_API_KEY")
        self.session: requests.Session = requests.Session()

    def get_citations(
        self, 
        request: Optional[CitationRequest] = None,
        *,
        chunks: Optional[List[TextSpan]] = None,
        jsonl_content: Optional[str] = None,
        jsonl_url: Optional[str] = None,
        question: Optional[str] = None,
        start: Optional[int] = 0,
        limit: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = 1000,
    ) -> CitationResponse:
        """Get citations for a question from text chunks.

        This method finds relevant citations that answer a specific question by analyzing
        the provided text chunks. Each citation includes a quality score (0-4) and reasoning
        for why it's relevant to the question.

        Args:
            request: Citation request containing chunks/URL and question to answer (legacy)
            chunks: List of text chunks to search (alternative to request)
            jsonl_content: Raw JSONL content as string (alternative to request)
            jsonl_url: URL to fetch JSONL content from (alternative to request)
            question: The question to find citations for (required if not using request)
            start: Starting chunk index (0-based)
            limit: Maximum number of chunks to process
            max_tokens_per_chunk: Maximum tokens per chunk

        Returns:
            Citation response with found citations, total count, and usage information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Using function arguments directly:

            ```python
            from bookwyrm.models import TextSpan

            chunks = [
                TextSpan(text="The sky is blue.", start_char=0, end_char=16),
                TextSpan(text="Water is wet.", start_char=17, end_char=30)
            ]

            response = client.get_citations(
                chunks=chunks,
                question="Why is the sky blue?"
            )
            print(f"Found {response.total_citations} citations")

            for citation in response.citations:
                print(f"Quality: {citation.quality}/4")
                print(f"Text: {citation.text}")
                print(f"Reasoning: {citation.reasoning}")
            ```

            Using a JSONL URL:

            ```python
            response = client.get_citations(
                jsonl_url="https://example.com/chunks.jsonl",
                question="What is machine learning?",
                start=0,
                limit=100
            )
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import CitationRequest

            request = CitationRequest(
                chunks=chunks,
                question="Why is the sky blue?"
            )

            response = client.get_citations(request)
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            if question is None:
                raise ValueError("question is required when not using request object")
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
            response: requests.Response = self.session.post(
                f"{self.base_url}/cite",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return CitationResponse.model_validate(response_data)
        except requests.HTTPError as e:
            status_code: Optional[int] = (
                getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None
            )
            raise BookWyrmAPIError(f"API request failed: {e}", status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def classify(
        self,
        request: Optional[ClassifyRequest] = None,
        *,
        content: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_encoding: str = "base64",
    ) -> ClassifyResponse:
        """Classify file content to determine file type and format.

        This method analyzes file content to determine format type, content type, MIME type,
        and other classification details. It supports both binary and text files, providing
        confidence scores and additional metadata about the detected format.

        Args:
            request: Classification request with base64-encoded content and optional filename hint (legacy)
            content: Base64-encoded file content (alternative to request)
            content_bytes: Raw file bytes (alternative to request, will be encoded to base64)
            filename: Optional filename hint for classification
            content_encoding: Content encoding format (always "base64")

        Returns:
            Classification response with detected file type, confidence score, and additional details

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Classify using raw bytes directly:

            ```python
            # Read file as binary
            with open("document.pdf", "rb") as f:
                file_bytes = f.read()

            response = client.classify(
                content_bytes=file_bytes,
                filename="document.pdf"
            )
            print(f"Format: {response.classification.format_type}")
            print(f"Content Type: {response.classification.content_type}")
            print(f"MIME Type: {response.classification.mime_type}")
            print(f"Confidence: {response.classification.confidence:.2%}")
            ```

            Classify from file path:

            ```python
            file_path = Path("script.py")
            response = client.classify(
                content_bytes=file_path.read_bytes(),
                filename=file_path.name
            )
            print(f"Detected as: {response.classification.content_type}")
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import ClassifyRequest

            request = ClassifyRequest(
                content_bytes=file_bytes,
                filename="document.pdf"
            )

            response = client.classify(request)
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            if content is None and content_bytes is None:
                raise ValueError("Either content or content_bytes is required when not using request object")
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
                # Decode base64 content
                import base64
                file_bytes = base64.b64decode(request.content)
            else:
                raise BookWyrmAPIError("Either content or content_bytes must be provided")

            files: Dict[str, tuple] = {
                "file": (request.filename or "document", file_bytes)
            }
            response: requests.Response = self.session.post(
                f"{self.base_url}/classify",
                files=files,
                headers=headers,
            )

            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return ClassifyResponse.model_validate(response_data)
        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def process_text(
        self, 
        request: Optional[ProcessTextRequest] = None,
        *,
        text: Optional[str] = None,
        text_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        response_format: Optional[ResponseFormat] = ResponseFormat.WITH_OFFSETS,
        spacy_model: str = "en_core_web_sm",
    ) -> Iterator[StreamingPhrasalResponse]:
        """Process text using phrasal analysis with streaming results.

        This method breaks down text into meaningful phrases or chunks using NLP,
        supporting both direct text input and URLs. It can create fixed-size chunks
        or extract individual phrases with optional position information.

        Args:
            request: Text processing request with text/URL, chunking options, and format preferences (legacy)
            text: Text content to process (alternative to request)
            text_url: URL to fetch text from (alternative to request)
            chunk_size: Optional chunk size for fixed-size chunking
            response_format: Response format (WITH_OFFSETS or TEXT_ONLY)
            spacy_model: SpaCy model to use for processing

        Yields:
            StreamingPhrasalResponse: Union of progress updates and phrase/chunk results

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Extract phrases from text with position offsets:

            ```python
            from bookwyrm.models import ResponseFormat

            text = '''
            Natural language processing (NLP) is a subfield of linguistics, computer science,
            and artificial intelligence concerned with the interactions between computers and human language.
            '''

            phrases = []
            for response in client.process_text(
                text=text,
                response_format=ResponseFormat.WITH_OFFSETS,
                spacy_model="en_core_web_sm"
            ):
                if isinstance(response, (TextResult, TextSpanResult)):  # Phrase result
                    phrases.append(response)
                    print(f"Phrase: {response.text}")
                    if isinstance(response, TextSpanResult):
                        print(f"Position: {response.start_char}-{response.end_char}")
            ```

            Create fixed-size chunks:

            ```python
            chunks = []
            for response in client.process_text(
                text=long_text,
                chunk_size=1000,  # ~1000 characters per chunk
                response_format=ResponseFormat.WITH_OFFSETS
            ):
                if isinstance(response, (TextResult, TextSpanResult)):
                    chunks.append(response)

            print(f"Created {len(chunks)} chunks")
            ```

            Process text from URL:

            ```python
            # Save to JSONL file
            with open("alice_phrases.jsonl", "w") as f:
                for response in client.process_text(
                    text_url="https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
                    chunk_size=2000,
                    response_format=ResponseFormat.WITH_OFFSETS
                ):
                    if isinstance(response, (TextResult, TextSpanResult)):
                        f.write(response.model_dump_json() + "\n")
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import ProcessTextRequest, ResponseFormat

            request = ProcessTextRequest(
                text=text,
                response_format=ResponseFormat.WITH_OFFSETS
            )

            for response in client.process_text(request):
                # Process responses...
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            if text is None and text_url is None:
                raise ValueError("Either text or text_url is required when not using request object")
            request = ProcessTextRequest(
                text=text,
                text_url=text_url,
                chunk_size=chunk_size,
                response_format=response_format,
                spacy_model=spacy_model,
            )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: requests.Response = self.session.post(
                f"{self.base_url}/phrasal",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_citations(
        self, 
        request: Optional[CitationRequest] = None,
        *,
        chunks: Optional[List[TextSpan]] = None,
        jsonl_content: Optional[str] = None,
        jsonl_url: Optional[str] = None,
        question: Optional[str] = None,
        start: Optional[int] = 0,
        limit: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = 1000,
    ) -> Iterator[StreamingCitationResponse]:
        """Stream citations as they are found with real-time progress updates.

        This method provides real-time streaming of citation results, allowing you to
        process citations as they're found rather than waiting for all results. Useful
        for large datasets or when you want to show progress to users.

        Args:
            request: Citation request containing chunks/URL and question to answer (legacy)
            chunks: List of text chunks to search (alternative to request)
            jsonl_content: Raw JSONL content as string (alternative to request)
            jsonl_url: URL to fetch JSONL content from (alternative to request)
            question: The question to find citations for (required if not using request)
            start: Starting chunk index (0-based)
            limit: Maximum number of chunks to process
            max_tokens_per_chunk: Maximum tokens per chunk

        Yields:
            StreamingCitationResponse: Union of progress updates, individual citations,
            final summary, or error messages

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming with function arguments:

            ```python
            citations = []
            for response in client.stream_citations(
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
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import CitationRequest

            request = CitationRequest(
                chunks=chunks,
                question="Why is the sky blue?"
            )

            for response in client.stream_citations(request):
                # Process responses...
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            if question is None:
                raise ValueError("question is required when not using request object")
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
            response: requests.Response = self.session.post(
                f"{self.base_url}/cite/stream",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def extract_pdf(
        self,
        request: Optional[PDFExtractRequest] = None,
        *,
        pdf_url: Optional[str] = None,
        pdf_content: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        start_page: Optional[int] = None,
        num_pages: Optional[int] = None,
    ) -> PDFExtractResponse:
        """Extract structured data from a PDF file using OCR.

        This method extracts text elements from PDF files with position coordinates,
        confidence scores, and bounding box information. It supports both local files
        (base64-encoded) and remote URLs, with optional page range selection.

        Args:
            request: PDF extraction request with URL/content, optional page range, and filename (legacy)
            pdf_url: URL to PDF file (alternative to request)
            pdf_content: Base64 encoded PDF content (alternative to request)
            pdf_bytes: Raw PDF bytes (alternative to request, will be encoded to base64)
            filename: Optional filename hint
            start_page: 1-based page number to start from
            num_pages: Number of pages to process from start_page

        Returns:
            PDF extraction response with structured page data, text elements, and metadata

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Extract from local PDF file using raw bytes:

            ```python
            # Load PDF file
            with open("document.pdf", "rb") as f:
                pdf_bytes = f.read()

            response = client.extract_pdf(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            )
            print(f"Extracted {response.total_pages} pages")

            for page in response.pages:
                print(f"Page {page.page_number}: {len(page.text_blocks)} text elements")
                for element in page.text_blocks[:3]:  # Show first 3 elements
                    print(f"  - {element.text[:50]}...")
                    print(f"    Confidence: {element.confidence:.2f}")
                    print(f"    Position: ({element.coordinates.x1}, {element.coordinates.y1})")
            ```

            Extract from URL with page range:

            ```python
            response = client.extract_pdf(
                pdf_url="https://example.com/document.pdf",
                start_page=5,
                num_pages=10
            )
            print(f"Extracted pages 5-14: {response.total_pages} pages")

            if response.processing_time:
                print(f"Processing time: {response.processing_time:.2f}s")
            ```

            Extract from file path:

            ```python
            pdf_path = Path("document.pdf")
            response = client.extract_pdf(
                pdf_bytes=pdf_path.read_bytes(),
                filename=pdf_path.name,
                start_page=1,
                num_pages=5
            )
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import PDFExtractRequest

            request = PDFExtractRequest(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            )

            response = client.extract_pdf(request)
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            sources = [pdf_url, pdf_content, pdf_bytes]
            provided_sources = [s for s in sources if s is not None]
            if len(provided_sources) != 1:
                raise ValueError("Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided")
            
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
                    # Handle base64-encoded file content
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

                response: requests.Response = self.session.post(
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

                response = self.session.post(
                    f"{self.base_url}/extract-structure-json",
                    json=json_data,
                    headers=headers,
                )
            else:
                raise BookWyrmAPIError("Either pdf_url or pdf_content must be provided")

            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return PDFExtractResponse.model_validate(response_data)
        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_extract_pdf(
        self,
        request: Optional[PDFExtractRequest] = None,
        *,
        pdf_url: Optional[str] = None,
        pdf_content: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        start_page: Optional[int] = None,
        num_pages: Optional[int] = None,
    ) -> Iterator[StreamingPDFResponse]:
        """Stream PDF extraction with real-time progress updates.

        This method provides real-time streaming of PDF extraction progress, yielding
        metadata, individual page results, and completion status. Useful for large PDFs
        where you want to show progress or process pages as they become available.

        Args:
            request: PDF extraction request with URL/content, optional page range, and filename (legacy)
            pdf_url: URL to PDF file (alternative to request)
            pdf_content: Base64 encoded PDF content (alternative to request)
            pdf_bytes: Raw PDF bytes (alternative to request, will be encoded to base64)
            filename: Optional filename hint
            start_page: 1-based page number to start from
            num_pages: Number of pages to process from start_page

        Yields:
            StreamingPDFResponse: Union of metadata, page responses, page errors, completion, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming with function arguments:

            ```python
            pages = []
            for response in client.stream_extract_pdf(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            ):
                if hasattr(response, 'total_pages'):  # Metadata
                    print(f"Processing {response.total_pages} pages")
                elif hasattr(response, 'page_data'):  # Page extracted
                    pages.append(response.page_data)
                    print(f"Page {response.document_page}: {len(response.page_data.text_blocks)} elements")
                elif hasattr(response, 'error') and hasattr(response, 'document_page'):  # Page error
                    print(f"Error on page {response.document_page}: {response.error}")

            print(f"Extracted {len(pages)} pages total")
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import PDFExtractRequest

            request = PDFExtractRequest(
                pdf_bytes=pdf_bytes,
                filename="document.pdf"
            )

            for response in client.stream_extract_pdf(request):
                # Process responses...
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
            sources = [pdf_url, pdf_content, pdf_bytes]
            provided_sources = [s for s in sources if s is not None]
            if len(provided_sources) != 1:
                raise ValueError("Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided")
            
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

                response = self.session.post(
                    f"{self.base_url}/extract-structure-stream",
                    files=files,
                    data=data,
                    headers=headers,
                    stream=True,
                )
            elif request.pdf_url:
                # Handle URL using JSON endpoint
                headers["Content-Type"] = "application/json"
                json_data = {"pdf_url": request.pdf_url}
                if request.start_page is not None:
                    json_data["start_page"] = request.start_page
                if request.num_pages is not None:
                    json_data["num_pages"] = request.num_pages

                response = self.session.post(
                    f"{self.base_url}/extract-structure-stream-json",
                    json=json_data,
                    headers=headers,
                    stream=True,
                )
            else:
                raise BookWyrmAPIError("Either pdf_url or pdf_content must be provided")

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def close(self) -> None:
        """Close the client session."""
        self.session.close()

    def __enter__(self) -> "BookWyrmClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()

    def summarize(
        self,
        request: Optional[SummarizeRequest] = None,
        *,
        content: Optional[str] = None,
        url: Optional[str] = None,
        phrases: Optional[List[TextSpan]] = None,
        max_tokens: int = 10000,
        debug: bool = False,
    ) -> SummaryResponse:
        """Get a summary of the provided content using hierarchical summarization.

        This method performs intelligent summarization of text content, supporting both
        plain text summaries and structured output using Pydantic models. It can handle
        large documents through hierarchical chunking and summarization.

        Args:
            request: Summarization request with content, options, and optional structured output settings (legacy)
            content: Text content to summarize (alternative to request)
            url: URL to fetch content from (alternative to request)
            phrases: List of text phrases to summarize (alternative to request)
            max_tokens: Maximum tokens for chunking (default: 10000)
            debug: Include intermediate summaries in response

        Returns:
            Summary response containing the generated summary, metadata, and optional debug information

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic text summarization with function arguments:

            ```python
            # Load JSONL content
            with open("book_phrases.jsonl", "r") as f:
                content = f.read()

            response = client.summarize(
                content=content,
                max_tokens=5000,
                debug=True  # Include intermediate summaries
            )
            print("Summary:")
            print(response.summary)
            print(f"Used {response.levels_used} levels")
            print(f"Created {response.subsummary_count} subsummaries")
            ```

            Summarize from URL:

            ```python
            response = client.summarize(
                url="https://www.gutenberg.org/files/11/11-0.txt",
                max_tokens=10000
            )
            print(response.summary)
            ```

            Summarize from phrases:

            ```python
            response = client.summarize(
                phrases=phrase_list,
                max_tokens=2000
            )
            print(response.summary)
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import SummarizeRequest

            request = SummarizeRequest(
                content=content,
                max_tokens=5000,
                debug=True
            )

            response = client.summarize(request)
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
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
            response: requests.Response = self.session.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            response_data: Dict[str, Any] = response.json()
            return SummaryResponse.model_validate(response_data)
        except requests.HTTPError as e:
            status_code: Optional[int] = (
                getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None
            )
            raise BookWyrmAPIError(f"API request failed: {e}", status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_summarize(
        self,
        request: Optional[SummarizeRequest] = None,
        *,
        content: Optional[str] = None,
        url: Optional[str] = None,
        phrases: Optional[List[TextSpan]] = None,
        max_tokens: int = 10000,
        debug: bool = False,
    ) -> Iterator[StreamingSummarizeResponse]:
        """Stream summarization progress and results with real-time updates.

        This method provides real-time streaming of summarization progress, including
        hierarchical processing updates, retry attempts, and final results. Useful for
        long-running summarization tasks where you want to show progress to users.

        Args:
            request: Summarization request with content, options, and optional structured output settings (legacy)
            content: Text content to summarize (alternative to request)
            url: URL to fetch content from (alternative to request)
            phrases: List of text phrases to summarize (alternative to request)
            max_tokens: Maximum tokens for chunking (default: 10000)
            debug: Include intermediate summaries in response

        Yields:
            StreamingSummarizeResponse: Union of progress updates, final summary, rate limit messages,
            structural error messages, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming with function arguments:

            ```python
            final_result = None
            for response in client.stream_summarize(
                content=content,
                max_tokens=5000,
                debug=True
            ):
                if isinstance(response, SummarizeProgressUpdate):  # Progress update
                    print(f"Progress: {response.message}")
                elif isinstance(response, SummaryResponse):  # Final summary
                    final_result = response
                    print(f"Summary complete!")

            if final_result:
                print(final_result.summary)
            ```

            Legacy request object usage (still supported):

            ```python
            from bookwyrm.models import SummarizeRequest

            request = SummarizeRequest(
                content=content,
                max_tokens=5000,
                debug=True
            )

            for response in client.stream_summarize(request):
                # Process responses...
            ```
        """
        # Handle both new function interface and legacy request object
        if request is None:
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
            response: requests.Response = self.session.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line and line.strip() and line.startswith("data: "):
                    try:
                        data: Dict[str, Any] = json.loads(
                            line[6:]
                        )  # Remove "data: " prefix
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")
