"""Synchronous client for BookWyrm API."""

import json
import os
import platform
import sys
from typing import List, Iterator, Optional, Union, Dict, Any, Literal, Type
import requests
from pathlib import Path
from sseclient import SSEClient
from pydantic import BaseModel

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
    ContentEncoding,
    ClassifyRequest,
    ClassifyResponse,
    StreamingClassifyResponse,
    ClassifyProgressUpdate,
    ClassifyStreamResponse,
    ClassifyErrorResponse,
    PDFExtractRequest,
    PDFExtractResponse,
    StreamingPDFResponse,
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    TextSpan,
    UsageInfo,
)


# User-Agent and client version headers
UA = f"bookwyrm-client/{__version__} (python/{platform.python_version()}; {platform.system()})"
BOOKWYRM_CLIENT_DATE = "2025-10-16"

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Bookwyrm-Client-Date": BOOKWYRM_CLIENT_DATE,
}


class BookWyrmClientError(Exception):
    """Base exception for BookWyrm client errors."""

    pass


class BookWyrmAPIError(BookWyrmClientError):
    """Exception for API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _marshal_http_error(e: requests.HTTPError) -> BookWyrmAPIError:
    """Convert requests.HTTPError to BookWyrmAPIError with response text."""
    error_message = f"API request failed: {e}"
    status_code = None

    if e.response is not None:
        status_code = e.response.status_code
        try:
            error_text = e.response.text
            if error_text and error_text.strip():
                error_message += f" - {error_text.strip()}"
        except Exception:
            # If we can't read the response text, just use the original message
            pass

    return BookWyrmAPIError(error_message, status_code)


def _check_deprecation_headers(response: requests.Response) -> None:
    """Check for deprecation headers and write warnings to stderr."""
    deprecation = response.headers.get("Deprecation")
    warning = response.headers.get("Warning")
    
    if deprecation and deprecation != "false":
        if warning and warning.startswith("299 bookwyrm"):
            # Extract the warning message from the 299 warning format
            # Format: '299 bookwyrm "message"'
            try:
                warning_msg = warning.split('"', 1)[1].rsplit('"', 1)[0]
                print(f"WARNING: {warning_msg}", file=sys.stderr)
            except (IndexError, ValueError):
                # Fallback if parsing fails
                print(f"WARNING: Client version deprecation detected", file=sys.stderr)


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
        timeout: Optional[float] = None,
    ) -> None:
        """Initialize the BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API. Defaults to "https://api.bookwyrm.ai:443"
            api_key: API key for authentication. If not provided, will attempt to read
                from BOOKWYRM_API_KEY environment variable
            timeout: Request timeout in seconds. Defaults to None (no timeout). Set to a float for specific timeout.

        Examples:
            ```python
            # Basic initialization
            client = BookWyrmClient()

            # With explicit API key
            client = BookWyrmClient(api_key="your-api-key")

            # With custom endpoint and timeout
            client = BookWyrmClient(
                base_url="https://localhost:8000",
                api_key="dev-key",
                timeout=60.0
            )

            # With infinite timeout (default)
            client = BookWyrmClient()
            ```
        """
        self.base_url: str = base_url.rstrip("/")
        self.api_key: Optional[str] = api_key or os.getenv("BOOKWYRM_API_KEY")
        self.timeout: Optional[float] = timeout
        self.session: requests.Session = requests.Session()

    def classify(
        self,
        *,
        content: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_encoding: ContentEncoding = ContentEncoding.RAW,
    ) -> ClassifyResponse:
        """Classify file content to determine file type and format.

        This method analyzes file content to determine format type, content type, MIME type,
        and other classification details. It supports both binary and text files, providing
        confidence scores and additional metadata about the detected format.

        Args:
            content: Text or encoded file content
            content_bytes: Raw file bytes
            filename: Optional filename hint for classification
            content_encoding: Content encoding format (ContentEncoding enum)

        Returns:
            Classification response with detected file type, confidence score, and additional details

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Classify using raw bytes directly (recommended):

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

            Classify text content with UTF-8 encoding:

            ```python
            with open("script.py", "r") as f:
                text_content = f.read()

            response = client.classify(
                content=text_content,
                filename="script.py",
                content_encoding=ContentEncoding.UTF8
            )
            print(f"Detected as: {response.classification.content_type}")
            ```

            Classify base64-encoded content:

            ```python
            import base64

            with open("image.png", "rb") as f:
                raw_bytes = f.read()

            base64_content = base64.b64encode(raw_bytes).decode('ascii')

            response = client.classify(
                content=base64_content,
                filename="image.png",
                content_encoding=ContentEncoding.BASE64
            )
            print(f"Detected as: {response.classification.content_type}")
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
                # Handle content based on encoding
                if request.content_encoding == ContentEncoding.BASE64:
                    # Decode base64 content
                    import base64

                    file_bytes = base64.b64decode(request.content)
                elif request.content_encoding == ContentEncoding.UTF8:
                    # Encode UTF-8 text to bytes
                    file_bytes = request.content.encode("utf-8")
                elif request.content_encoding == ContentEncoding.RAW:
                    # Treat as raw bytes (assume content is already bytes-like)
                    # This case should typically use content_bytes instead
                    file_bytes = request.content.encode("latin-1")  # Preserve raw bytes
                else:
                    raise BookWyrmAPIError(
                        f"Unsupported content encoding: {request.content_encoding}"
                    )
            else:
                raise BookWyrmAPIError(
                    "Either content or content_bytes must be provided"
                )

            files: Dict[str, tuple] = {
                "file": (request.filename or "document", file_bytes)
            }
            response: requests.Response = self.session.post(
                f"{self.base_url}/classify",
                files=files,
                headers=headers,
                timeout=self.timeout,
            )

            response.raise_for_status()
            _check_deprecation_headers(response)
            response_data: Dict[str, Any] = response.json()
            return ClassifyResponse.model_validate(response_data)
        except requests.HTTPError as e:
            raise _marshal_http_error(e)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_classify(
        self,
        *,
        content: Optional[str] = None,
        content_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        content_encoding: ContentEncoding = ContentEncoding.RAW,
    ) -> Iterator[StreamingClassifyResponse]:
        """Stream file classification with real-time progress updates.

        This method provides real-time streaming of file classification progress, allowing you to
        process classification results as they become available. Useful for large files or when
        you want to show progress to users during classification.

        Args:
            content: Text or encoded file content
            content_bytes: Raw file bytes
            filename: Optional filename hint for classification
            content_encoding: Content encoding format (ContentEncoding enum)

        Yields:
            StreamingClassifyResponse: Union of progress updates, classification results, or error messages

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming classification:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import ClassifyProgressUpdate, ClassifyStreamResponse, ClassifyErrorResponse

            # Read file as binary
            with open("document.pdf", "rb") as f:
                file_bytes = f.read()

            client = BookWyrmClient(api_key="your-api-key")
            classification_result = None
            for response in client.stream_classify(
                content_bytes=file_bytes,
                filename="document.pdf"
            ):
                if isinstance(response, ClassifyProgressUpdate):  # Progress update
                    print(f"Progress: {response.message}")
                elif isinstance(response, ClassifyStreamResponse):  # Classification result
                    classification_result = response
                    print(f"Format: {response.classification.format_type}")
                    print(f"Content Type: {response.classification.content_type}")
                    print(f"Confidence: {response.classification.confidence:.2%}")
                elif isinstance(response, ClassifyErrorResponse):  # Error
                    print(f"Error: {response.message}")
            ```

            Stream classify with base64 content:

            ```python
            import base64

            with open("image.png", "rb") as f:
                raw_bytes = f.read()

            base64_content = base64.b64encode(raw_bytes).decode('ascii')

            for response in client.stream_classify(
                content=base64_content,
                filename="image.png",
                content_encoding=ContentEncoding.BASE64
            ):
                if isinstance(response, ClassifyStreamResponse):
                    print(f"Detected as: {response.classification.content_type}")
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
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Handle marshalling at API level - convert to base64 for SSE endpoint
            if request.content_bytes is not None:
                import base64
                content_base64 = base64.b64encode(request.content_bytes).decode('ascii')
            elif request.content is not None:
                if request.content_encoding == ContentEncoding.BASE64:
                    # Content is already base64 encoded
                    content_base64 = request.content
                elif request.content_encoding == ContentEncoding.UTF8:
                    # Encode UTF-8 text to base64
                    import base64
                    content_base64 = base64.b64encode(request.content.encode("utf-8")).decode('ascii')
                elif request.content_encoding == ContentEncoding.RAW:
                    # Treat as raw bytes and encode to base64
                    import base64
                    content_base64 = base64.b64encode(request.content.encode("latin-1")).decode('ascii')
                else:
                    raise BookWyrmAPIError(
                        f"Unsupported content encoding: {request.content_encoding}"
                    )
            else:
                raise BookWyrmAPIError(
                    "Either content or content_bytes must be provided"
                )

            # Prepare JSON request for SSE endpoint
            json_data = {
                "content_base64": content_base64,
                "filename": request.filename,
            }

            response: requests.Response = self.session.post(
                f"{self.base_url}/classify/sse",
                json=json_data,
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )

            response.raise_for_status()
            _check_deprecation_headers(response)

            # Use SSEClient for proper SSE parsing
            client = SSEClient(response)
            for event in client.events():
                if event.data and event.data.strip():
                    try:
                        data: Dict[str, Any] = json.loads(event.data)
                        
                        # Use the event type, or fall back to data.type
                        event_type = event.event or data.get("type")
                        
                        match event_type:
                            case "progress":
                                yield ClassifyProgressUpdate.model_validate(data)
                            case "classification":
                                yield ClassifyStreamResponse.model_validate(data)
                            case "error":
                                yield ClassifyErrorResponse.model_validate(data)
                            case _:
                                # Unknown response type, skip
                                continue
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

        except requests.HTTPError as e:
            raise _marshal_http_error(e)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_process_text(
        self,
        *,
        text: Optional[str] = None,
        text_url: Optional[str] = None,
        chunk_size: Optional[int] = None,
        response_format: Union[
            ResponseFormat, Literal["offsets", "text_only"]
        ] = ResponseFormat.WITH_OFFSETS,
        # Boolean flags for response format
        offsets: Optional[bool] = None,
        text_only: Optional[bool] = None,
    ) -> Iterator[StreamingPhrasalResponse]:
        """Stream text processing using phrasal analysis with real-time results.

        This method breaks down text into meaningful phrases or chunks using NLP,
        supporting both direct text input and URLs. It can create fixed-size chunks
        or extract individual phrases with optional position information.

        Args:
            text: Text content to process
            text_url: URL to fetch text from
            chunk_size: Optional chunk size for fixed-size chunking
            response_format: Response format - use ResponseFormat enum, "offsets", or "text_only"
            offsets: Set to True for WITH_OFFSETS format (boolean flag)
            text_only: Set to True for TEXT_ONLY format (boolean flag)

        Yields:
            StreamingPhrasalResponse: Union of progress updates and phrase/chunk results

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Extract phrases from text with position offsets:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import ResponseFormat, TextResult, TextSpanResult

            text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

            client = BookWyrmClient(api_key="your-api-key")
            phrases = []
            for response in client.stream_process_text(
                text=text,
                offsets=True,  # or response_format="with_offsets" or ResponseFormat.WITH_OFFSETS
            ):
                if isinstance(response, (TextResult, TextSpanResult)):  # Phrase result
                    phrases.append(response)
                    print(f"Phrase: {response.text}")
                    if isinstance(response, TextSpanResult):
                        print(f"Position: {response.start_char}-{response.end_char}")
            ```

            Create bounded phrasal chunks:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import TextResult, TextSpanResult

            client = BookWyrmClient(api_key="your-api-key")
            chunks = []
            for response in client.stream_process_text(
                text=long_text,
                chunk_size=1000,  # chunks composed of phrases, not exceeding ~1000 characters
                offsets=True  # boolean flag for WITH_OFFSETS
            ):
                if isinstance(response, (TextResult, TextSpanResult)):
                    chunks.append(response)

            print(f"Created {len(chunks)} chunks")
            ```

            Process text from URL:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import TextResult, TextSpanResult

            client = BookWyrmClient(api_key="your-api-key")
            phrases = []
            for response in client.stream_process_text(
                text_url="https://www.gutenberg.org/files/11/11-0.txt",
                chunk_size=2000,
                text_only=True
            ):
                if isinstance(response, (TextResult, TextSpanResult)):
                    phrases.append(response)

            print(f"Processed {len(phrases)} phrases from URL")
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
            request_data = request.model_dump(exclude_none=True)

            # Debug: Print the HTTP request details if BOOKWYRM_DEBUG is set
            if os.getenv("BOOKWYRM_DEBUG") == "1":
                print(f"DEBUG: Making POST request to: {self.base_url}/phrasal")
                print(f"DEBUG: Request headers: {headers}")
                print(f"DEBUG: Request JSON data: {json.dumps(request_data, indent=2)}")

            response: requests.Response = self.session.post(
                f"{self.base_url}/phrasal/sse",
                json=request_data,
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )

            # Debug: Print response details if BOOKWYRM_DEBUG is set
            if os.getenv("BOOKWYRM_DEBUG") == "1":
                print(f"DEBUG: Response status code: {response.status_code}")
                print(f"DEBUG: Response headers: {dict(response.headers)}")

            response.raise_for_status()
            _check_deprecation_headers(response)

            # Use SSEClient for proper SSE parsing
            event_count = 0
            client = SSEClient(response)
            for event in client.events():
                event_count += 1

                # Debug: Print every event received if BOOKWYRM_DEBUG is set
                if os.getenv("BOOKWYRM_DEBUG") == "1":
                    print(f"DEBUG: Event {event_count} - type: {event.event}, data: {repr(event.data)}")

                # Always yield raw event info for debugging if BOOKWYRM_DEBUG is set
                if os.getenv("BOOKWYRM_DEBUG") == "1":
                    from types import SimpleNamespace

                    raw_event_response = SimpleNamespace()
                    raw_event_response.type = "raw_event_debug"
                    raw_event_response.event_type = event.event
                    raw_event_response.event_data = event.data
                    raw_event_response.event_id = event.id
                    yield raw_event_response

                if event.data and event.data.strip():
                    try:
                        data: Dict[str, Any] = json.loads(event.data)
                        
                        # Use the event type, or fall back to data.type
                        event_type = event.event or data.get("type")
                        
                        match event_type:
                            case "progress":
                                yield PhraseProgressUpdate.model_validate(data)
                            case "text":
                                yield TextResult.model_validate(data)
                            case "text_span":
                                yield TextSpanResult.model_validate(data)
                            case _:
                                # Unknown response type - create a generic response object for debugging
                                from types import SimpleNamespace

                                unknown_response = SimpleNamespace(**data)
                                unknown_response.type = event_type
                                yield unknown_response
                    except json.JSONDecodeError as e:
                        # Create a debug object for malformed JSON
                        from types import SimpleNamespace

                        error_response = SimpleNamespace()
                        error_response.type = "json_decode_error"
                        error_response.raw_data = event.data
                        error_response.error = str(e)
                        yield error_response

        except requests.HTTPError as e:
            raise _marshal_http_error(e)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_citations(
        self,
        *,
        chunks: Optional[List[TextSpan]] = None,
        jsonl_content: Optional[str] = None,
        jsonl_url: Optional[str] = None,
        question: Union[str, List[str]],
        start: Optional[int] = 0,
        limit: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = 1000,
    ) -> Iterator[StreamingCitationResponse]:
        """Stream citations as they are found with real-time progress updates.

        This method provides real-time streaming of citation results, allowing you to
        process citations as they're found rather than waiting for all results. Useful
        for large datasets or when you want to show progress to users.

        Args:
            chunks: List of text chunks to search
            jsonl_content: Raw JSONL content as string
            jsonl_url: URL to fetch JSONL content from
            question: The question(s) to find citations for - can be a single string or list of strings
            start: Starting chunk index (0-based)
            limit: Maximum number of chunks to process
            max_tokens_per_chunk: Maximum tokens per chunk

        Yields:
            StreamingCitationResponse: Union of progress updates, individual citations,
            final summary, or error messages

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming with single question:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import TextSpan, CitationProgressUpdate, CitationStreamResponse, CitationSummaryResponse

            # Create some example chunks
            chunks = [
                TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
                TextSpan(text="Water molecules are polar.", start_char=43, end_char=69),
                TextSpan(text="Plants appear green due to chlorophyll.", start_char=70, end_char=109)
            ]

            client = BookWyrmClient(api_key="your-api-key")
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

            Multiple questions:

            ```python
            questions = [
                "Why is the sky blue?",
                "What causes plants to be green?",
                "How do water molecules behave?"
            ]

            for response in client.stream_citations(
                chunks=chunks,
                question=questions
            ):
                if isinstance(response, CitationStreamResponse):
                    citation = response.citation
                    if citation.question_index:
                        print(f"Question {citation.question_index}: {citation.text[:50]}...")
                    else:
                        print(f"Citation: {citation.text[:50]}...")
            ```
        """
        # Validate question(s)
        if isinstance(question, str):
            if not question or not question.strip():
                raise ValueError("question cannot be empty")
        elif isinstance(question, list):
            if not question:
                raise ValueError("question list cannot be empty")
            if len(question) > 20:
                raise ValueError("question list cannot contain more than 20 questions")
            for i, q in enumerate(question):
                if not q or not q.strip():
                    raise ValueError(f"question at index {i} cannot be empty")
        else:
            raise ValueError("question must be a string or list of strings")

        # Handle empty chunks list - return empty response immediately
        if chunks is not None and len(chunks) == 0:
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
        """Stream citations as they are found with real-time progress updates.

        This method provides real-time streaming of citation results, allowing you to
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
            Basic streaming:

            ```python
            from bookwyrm import BookWyrmClient
            from bookwyrm.models import TextSpan, CitationProgressUpdate, CitationStreamResponse, CitationSummaryResponse

            # Create some example chunks
            chunks = [
                TextSpan(text="The sky is blue due to Rayleigh scattering.", start_char=0, end_char=42),
                TextSpan(text="Water molecules are polar.", start_char=43, end_char=69),
                TextSpan(text="Plants appear green due to chlorophyll.", start_char=70, end_char=109)
            ]

            client = BookWyrmClient(api_key="your-api-key")
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
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: requests.Response = self.session.post(
                f"{self.base_url}/cite/sse",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            _check_deprecation_headers(response)

            # Use SSEClient for proper SSE parsing
            client = SSEClient(response)
            for event in client.events():
                if event.data and event.data.strip():
                    try:
                        data: Dict[str, Any] = json.loads(event.data)
                        
                        # Use the event type, or fall back to data.type
                        event_type = event.event or data.get("type")
                        
                        match event_type:
                            case "progress":
                                # SSE endpoint sends ProgressUpdate, convert to CitationProgressUpdate
                                progress_data = {
                                    "type": "progress",
                                    "chunks_processed": data.get("chunks_processed", 0),
                                    "total_chunks": data.get("total_chunks", 0),
                                    "citations_found": data.get("citations_found", 0),
                                    "current_chunk_range": data.get("message", "Processing..."),  # Use message as range
                                    "message": data.get("message", "Processing..."),
                                }
                                yield CitationProgressUpdate.model_validate(progress_data)
                            case "citation":
                                yield CitationStreamResponse.model_validate(data)
                            case "citation_span":
                                # Handle citation_span events as regular citations
                                citation_data = {
                                    "type": "citation",
                                    "citation": data.get("citation")
                                }
                                yield CitationStreamResponse.model_validate(citation_data)
                            case "summary":
                                # SSE endpoint sends SummaryResult, convert to CitationSummaryResponse
                                summary_data = {
                                    "type": "summary",
                                    "total_citations": data.get("total_citations", 0),
                                    "chunks_processed": data.get("chunks_processed", 0),
                                    "token_chunks_processed": data.get("token_chunks_processed", 0),
                                    "start_offset": 0,  # SSE endpoint doesn't provide this, default to 0
                                    "usage": data.get("usage", {
                                        "tokens_processed": 0,
                                        "chunks_processed": 0,
                                        "remaining_credits": 0.0
                                    }),
                                }
                                yield CitationSummaryResponse.model_validate(summary_data)
                            case "error":
                                yield CitationErrorResponse.model_validate(data)
                            case _:
                                # Unknown response type, skip
                                continue
                    except json.JSONDecodeError:
                        # Skip malformed JSON lines
                        continue

        except requests.HTTPError as e:
            raise _marshal_http_error(e)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_extract_pdf(
        self,
        *,
        pdf_url: Optional[str] = None,
        pdf_content: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        start_page: Optional[int] = None,
        num_pages: Optional[int] = None,
        lang: str = "en",
    ) -> Iterator[StreamingPDFResponse]:
        """Stream PDF extraction with real-time progress updates.

        This method provides real-time streaming of PDF extraction progress, yielding
        metadata, individual page results, and completion status. Useful for large PDFs
        where you want to show progress or process pages as they become available.

        Args:
            pdf_url: URL to PDF file
            pdf_content: Base64 encoded PDF content
            pdf_bytes: Raw PDF bytes
            filename: Optional filename hint
            start_page: 1-based page number to start from
            num_pages: Number of pages to process from start_page
            lang: Language code for OCR processing (default: "en")

        Yields:
            StreamingPDFResponse: Union of metadata, page responses, page errors, completion, or general errors

        Raises:
            BookWyrmAPIError: If the API request fails (network, authentication, server errors)

        Examples:
            Basic streaming:

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
            lang=lang,
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
                if request.lang:
                    data["lang"] = request.lang

                response = self.session.post(
                    f"{self.base_url}/extract-structure/sse",
                    files=files,
                    data=data,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                )
            elif request.pdf_url:
                # Handle URL using JSON endpoint
                headers["Content-Type"] = "application/json"
                json_data = {"pdf_url": request.pdf_url}
                if request.start_page is not None:
                    json_data["start_page"] = request.start_page
                if request.num_pages is not None:
                    json_data["num_pages"] = request.num_pages
                if request.lang:
                    json_data["lang"] = request.lang

                response = self.session.post(
                    f"{self.base_url}/extract-structure-json/sse",
                    json=json_data,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout,
                )
            else:
                raise BookWyrmAPIError("Either pdf_url or pdf_content must be provided")

            response.raise_for_status()
            _check_deprecation_headers(response)

            # Use SSEClient for proper SSE parsing
            client = SSEClient(response)
            for event in client.events():
                if event.data and event.data.strip():
                    try:
                        data: Dict[str, Any] = json.loads(event.data)
                        
                        # Use the event type, or fall back to data.type
                        event_type = event.event or data.get("type")
                        
                        match event_type:
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
            raise _marshal_http_error(e)
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

    def stream_summarize(
        self,
        *,
        content: Optional[str] = None,
        url: Optional[str] = None,
        phrases: Optional[List[TextSpan]] = None,
        max_tokens: int = 10000,
        model_strength: str = "swift",
        debug: bool = False,
        model_name: Optional[str] = None,
        model_schema_json: Optional[str] = None,
        summary_class: Optional[Type[BaseModel]] = None,
        chunk_prompt: Optional[str] = None,
        summary_of_summaries_prompt: Optional[str] = None,
    ) -> Iterator[StreamingSummarizeResponse]:
        """Stream summarization progress and results with real-time updates.

        This method provides real-time streaming of summarization progress, including
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
            Basic streaming:

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
            model_strength=model_strength,
            debug=debug,
            model_name=model_name,
            model_schema_json=model_schema_json,
            summary_class=summary_class,
            chunk_prompt=chunk_prompt,
            summary_of_summaries_prompt=summary_of_summaries_prompt,
        )
        headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response: requests.Response = self.session.post(
                f"{self.base_url}/summarize/sse",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            _check_deprecation_headers(response)

            # Use SSEClient for proper SSE parsing
            client = SSEClient(response)
            for event in client.events():
                if event.data and event.data.strip():
                    try:
                        data: Dict[str, Any] = json.loads(event.data)
                        
                        # Use the event type, or fall back to data.type
                        event_type = event.event or data.get("type")
                        
                        match event_type:
                            case "progress":
                                yield SummarizeProgressUpdate.model_validate(data)
                            case "data":
                                # SSE endpoint sends DataResult, but we need to convert to SummaryResponse
                                # Create a SummaryResponse from the data fields
                                summary_data = {
                                    "type": "summary",  # Set the expected type
                                    "summary": data.get("summary", ""),
                                    "subsummary_count": data.get("subsummary_count", 0),
                                    "levels_used": data.get("levels_used", 0),
                                    "total_tokens": data.get("total_tokens", 0),
                                    "intermediate_summaries": data.get("intermediate_summaries"),
                                }
                                yield SummaryResponse.model_validate(summary_data)
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
            raise _marshal_http_error(e)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")
