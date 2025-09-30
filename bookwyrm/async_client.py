"""Asynchronous client for BookWyrm API."""

import json
from typing import AsyncIterator, Optional
import httpx
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
    ProcessTextRequest,
    StreamingPhrasalResponse,
    PhraseProgressUpdate,
    PhraseResult,
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


class AsyncBookWyrmClient:
    """Asynchronous client for BookWyrm API."""

    def __init__(
        self,
        base_url: str = "https://api.bookwyrm.ai:443",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the async BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    async def get_citations(self, request: CitationRequest) -> CitationResponse:
        """
        Get citations for a question from text chunks.

        Args:
            request: Citation request with chunks and question

        Returns:
            Citation response with found citations

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self.client.post(
                f"{self.base_url}/cite",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return CitationResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """
        Classify file content to determine file type and format.

        Args:
            request: Classification request with base64-encoded content

        Returns:
            Classification response with detected file type and details

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            if not request.content or request.content_encoding != "base64":
                raise BookWyrmAPIError("Content must be provided as base64-encoded data")
                
            # Decode base64 content and send as multipart form data
            import base64
            file_bytes = base64.b64decode(request.content)
            
            files = {"file": (request.filename or "document", file_bytes)}
            response = await self.client.post(
                f"{self.base_url}/classify",
                files=files,
                headers=headers,
            )

            response.raise_for_status()
            return ClassifyResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def process_text(
        self, request: ProcessTextRequest
    ) -> AsyncIterator[StreamingPhrasalResponse]:
        """
        Process text using phrasal analysis and return streaming response.

        Args:
            request: ProcessTextRequest containing text/URL and processing parameters

        Yields:
            Streaming phrasal responses (progress updates and phrase results)

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
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
                    if line.strip():
                        try:
                            data = json.loads(line)
                            response_type = data.get("type")

                            if response_type == "progress":
                                yield PhraseProgressUpdate.model_validate(data)
                            elif response_type == "phrase":
                                yield PhraseResult.model_validate(data)
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
        """
        Stream citations as they are found.

        Args:
            request: Citation request with chunks and question

        Yields:
            Streaming citation responses (progress, citations, summary, or errors)

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
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
                    if line.strip():
                        try:
                            data = json.loads(line)
                            response_type = data.get("type")

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
        """
        Extract structured data from a PDF file.

        Args:
            request: PDF extraction request with URL or content

        Returns:
            PDF extraction response with structured data

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {}
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
                    
                response = await self.client.post(
                    f"{self.base_url}/extract-structure",
                    files=files,
                    data=data,
                    headers=headers,
                )
            elif request.pdf_url:
                # Handle URL using JSON endpoint
                headers["Content-Type"] = "application/json"
                json_data = {"pdf_url": request.pdf_url}
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
            response_data = response.json()
            return PDFExtractResponse.model_validate(response_data)
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_extract_pdf(
        self, request: PDFExtractRequest
    ) -> AsyncIterator[StreamingPDFResponse]:
        """
        Stream PDF extraction with progress updates.

        Args:
            request: PDF extraction request with URL or content

        Yields:
            Streaming PDF responses (metadata, pages, completion, or errors)

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {}
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

    async def close(self):
        """Close the client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def summarize(self, request: SummarizeRequest) -> SummaryResponse:
        """
        Get a summary of the provided content.

        Args:
            request: Summarization request with content and options

        Returns:
            Summary response with the generated summary

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = await self.client.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return SummaryResponse.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except httpx.RequestError as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    async def stream_summarize(
        self, request: SummarizeRequest
    ) -> AsyncIterator[StreamingSummarizeResponse]:
        """
        Stream summarization progress and results.

        Args:
            request: Summarization request with content and options

        Yields:
            Streaming summarization responses (progress, summary, or errors)

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
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
                    if line.strip():
                        try:
                            data = json.loads(line)
                            response_type = data.get("type")

                            if response_type == "progress":
                                yield SummarizeProgressUpdate.model_validate(data)
                            elif response_type == "summary":
                                yield SummaryResponse.model_validate(data)
                            elif response_type == "error":
                                yield SummarizeErrorResponse.model_validate(data)
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
