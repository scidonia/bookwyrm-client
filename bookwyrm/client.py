"""Synchronous client for BookWyrm API."""

import json
from typing import List, Iterator, Optional
import requests
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
)


class BookWyrmClientError(Exception):
    """Base exception for BookWyrm client errors."""

    pass


class BookWyrmAPIError(BookWyrmClientError):
    """Exception for API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class BookWyrmClient:
    """Synchronous client for BookWyrm API."""

    def __init__(
        self,
        base_url: str = "https://api.bookwyrm.ai:443",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the BookWyrm client.

        Args:
            base_url: Base URL of the BookWyrm API
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()

    def get_citations(self, request: CitationRequest) -> CitationResponse:
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
            response = self.session.post(
                f"{self.base_url}/cite",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return CitationResponse.model_validate(response.json())
        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def classify(self, request: ClassifyRequest) -> ClassifyResponse:
        """
        Classify content or URL to determine file type and format.

        Args:
            request: Classification request with content or URL

        Returns:
            Classification response with detected file type and details

        Raises:
            BookWyrmAPIError: If the API request fails
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return ClassifyResponse.model_validate(response.json())
        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def process_text(
        self, request: ProcessTextRequest
    ) -> Iterator[StreamingPhrasalResponse]:
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
            response = self.session.post(
                f"{self.base_url}/phrasal",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_citations(
        self, request: CitationRequest
    ) -> Iterator[StreamingCitationResponse]:
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
            response = self.session.post(
                f"{self.base_url}/cite/stream",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def close(self):
        """Close the client session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def summarize(self, request: SummarizeRequest) -> SummaryResponse:
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
            response = self.session.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
            )
            response.raise_for_status()
            return SummaryResponse.model_validate(response.json())
        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")

    def stream_summarize(
        self, request: SummarizeRequest
    ) -> Iterator[StreamingSummarizeResponse]:
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
            response = self.session.post(
                f"{self.base_url}/summarize",
                json=request.model_dump(exclude_none=True),
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if line.strip() and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
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

        except requests.HTTPError as e:
            raise BookWyrmAPIError(f"API request failed: {e}", e.response.status_code)
        except requests.RequestException as e:
            raise BookWyrmAPIError(f"Request failed: {e}")
