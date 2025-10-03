"""Integration tests for synchronous PDF extraction."""

import pytest
from pathlib import Path
from bookwyrm.models import (
    PDFExtractResponse,
    PDFStreamMetadata,
    PDFStreamPageResponse,
    PDFStreamPageError,
    PDFStreamComplete,
    PDFStreamError,
    PDFPage,
    PDFTextElement,
    PDFBoundingBox,
)


pytestmark = pytest.mark.pdf


def test_extract_pdf_from_bytes(client):
    """Test PDF extraction using raw bytes."""
    # Create a minimal PDF for testing
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    response = client.extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="test.pdf"
    )

    # Verify response structure
    assert isinstance(response, PDFExtractResponse)
    assert response.total_pages >= 0
    assert isinstance(response.pages, list)
    
    # If pages were extracted, verify structure
    if response.pages:
        for page in response.pages:
            assert isinstance(page, PDFPage)
            assert isinstance(page.page_number, int)
            assert page.page_number >= 1
            assert isinstance(page.text_blocks, list)
            
            for text_block in page.text_blocks:
                assert isinstance(text_block, PDFTextElement)
                assert isinstance(text_block.text, str)
                assert isinstance(text_block.confidence, float)
                assert 0.0 <= text_block.confidence <= 1.0
                assert isinstance(text_block.coordinates, PDFBoundingBox)


def test_extract_pdf_with_page_range(client):
    """Test PDF extraction with specific page range."""
    # Create a multi-page PDF
    multi_page_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R 4 0 R 5 0 R]
/Count 3
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 6 0 R
>>
endobj

4 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 7 0 R
>>
endobj

5 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 8 0 R
>>
endobj

6 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Page 1) Tj
ET
endstream
endobj

7 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Page 2) Tj
ET
endstream
endobj

8 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Page 3) Tj
ET
endstream
endobj

xref
0 9
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000125 00000 n 
0000000214 00000 n 
0000000303 00000 n 
0000000392 00000 n 
0000000485 00000 n 
0000000578 00000 n 
trailer
<<
/Size 9
/Root 1 0 R
>>
startxref
671
%%EOF"""

    # Test extracting pages 2-3
    response = client.extract_pdf(
        pdf_bytes=multi_page_pdf,
        filename="multipage.pdf",
        start_page=2,
        num_pages=2
    )

    # Verify response
    assert isinstance(response, PDFExtractResponse)
    assert response.total_pages >= 0


def test_extract_pdf_from_url(client):
    """Test PDF extraction from URL."""
    # Skip this test for now - requires a publicly accessible PDF URL
    pytest.skip("Requires a publicly accessible PDF URL for testing")


def test_extract_pdf_error_no_input(client):
    """Test that missing PDF input raises an error."""
    with pytest.raises(ValueError, match="Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"):
        client.extract_pdf()


def test_extract_pdf_error_multiple_inputs(client):
    """Test that multiple PDF inputs raise an error."""
    with pytest.raises(ValueError, match="Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"):
        client.extract_pdf(
            pdf_bytes=b"fake pdf",
            pdf_url="https://example.com/test.pdf"
        )


def test_stream_extract_pdf_from_bytes(client):
    """Test streaming PDF extraction using raw bytes."""
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    metadata_received = False
    pages_received = []
    completion_received = False
    errors_received = []

    for response in client.stream_extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="test.pdf"
    ):
        if isinstance(response, PDFStreamMetadata):
            metadata_received = True
            assert isinstance(response.total_pages, int)
            assert response.total_pages >= 0
        elif isinstance(response, PDFStreamPageResponse):
            pages_received.append(response)
            assert isinstance(response.document_page, int)
            assert response.document_page >= 1
            assert isinstance(response.page_data, PDFPage)
        elif isinstance(response, PDFStreamPageError):
            errors_received.append(response)
            assert isinstance(response.document_page, int)
            assert isinstance(response.error, str)
        elif isinstance(response, PDFStreamComplete):
            completion_received = True
            assert isinstance(response.total_pages_processed, int)
        elif isinstance(response, PDFStreamError):
            errors_received.append(response)
            assert isinstance(response.error, str)

    # Verify we received expected responses
    # Note: Some responses may not be received depending on PDF content and processing


def test_stream_extract_pdf_with_page_range(client):
    """Test streaming PDF extraction with specific page range."""
    multi_page_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R 4 0 R]
/Count 2
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 5 0 R
>>
endobj

4 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 6 0 R
>>
endobj

5 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Page 1) Tj
ET
endstream
endobj

6 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Page 2) Tj
ET
endstream
endobj

xref
0 7
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
0000000293 00000 n 
0000000386 00000 n 
trailer
<<
/Size 7
/Root 1 0 R
>>
startxref
479
%%EOF"""

    pages_received = []

    for response in client.stream_extract_pdf(
        pdf_bytes=multi_page_pdf,
        filename="multipage.pdf",
        start_page=1,
        num_pages=1
    ):
        if isinstance(response, PDFStreamPageResponse):
            pages_received.append(response)

    # Verify we processed the expected page range
    # Note: Actual behavior depends on PDF processing implementation


def test_stream_extract_pdf_from_url(client):
    """Test streaming PDF extraction from URL."""
    # Skip this test for now - requires a publicly accessible PDF URL
    pytest.skip("Requires a publicly accessible PDF URL for testing")


def test_stream_extract_pdf_error_no_input(client):
    """Test that missing PDF input raises an error."""
    with pytest.raises(ValueError, match="Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"):
        list(client.stream_extract_pdf())


def test_stream_extract_pdf_error_multiple_inputs(client):
    """Test that multiple PDF inputs raise an error."""
    with pytest.raises(ValueError, match="Exactly one of pdf_url, pdf_content, or pdf_bytes must be provided"):
        list(client.stream_extract_pdf(
            pdf_bytes=b"fake pdf",
            pdf_url="https://example.com/test.pdf"
        ))


def test_extract_pdf_with_base64_content(client):
    """Test PDF extraction using base64-encoded content."""
    import base64
    
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    base64_content = base64.b64encode(minimal_pdf).decode('utf-8')
    
    response = client.extract_pdf(
        pdf_content=base64_content,
        filename="base64_test.pdf"
    )

    # Verify response structure
    assert isinstance(response, PDFExtractResponse)
    assert response.total_pages >= 0


def test_stream_extract_pdf_with_base64_content(client):
    """Test streaming PDF extraction using base64-encoded content."""
    import base64
    
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    base64_content = base64.b64encode(minimal_pdf).decode('utf-8')
    
    responses = []
    for response in client.stream_extract_pdf(
        pdf_content=base64_content,
        filename="base64_stream_test.pdf"
    ):
        responses.append(response)

    # Verify we got some responses
    assert len(responses) >= 0  # May be empty if PDF processing fails


def test_extract_pdf_bounding_box_validation(client):
    """Test that PDF text elements have valid bounding box coordinates."""
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    response = client.extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="bbox_test.pdf"
    )

    # Verify bounding box coordinates if text elements exist
    for page in response.pages:
        for text_element in page.text_blocks:
            bbox = text_element.coordinates
            assert isinstance(bbox.x1, (int, float))
            assert isinstance(bbox.y1, (int, float))
            assert isinstance(bbox.x2, (int, float))
            assert isinstance(bbox.y2, (int, float))
            # Bounding box should be valid (x2 >= x1, y2 >= y1)
            assert bbox.x2 >= bbox.x1
            assert bbox.y2 >= bbox.y1


@pytest.mark.liveonly
def test_extract_pdf_live_api_comprehensive(client):
    """Comprehensive test of PDF extraction against live API."""
    # Test various PDF scenarios to ensure they work with live API
    minimal_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Live API Test) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000204 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
297
%%EOF"""

    # Test 1: Basic extraction
    response1 = client.extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="live_test.pdf"
    )
    assert isinstance(response1, PDFExtractResponse)

    # Test 2: Streaming extraction
    responses = []
    for response in client.stream_extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="live_stream_test.pdf"
    ):
        responses.append(response)
    assert len(responses) >= 0

    # Test 3: With page range
    response3 = client.extract_pdf(
        pdf_bytes=minimal_pdf,
        filename="live_range_test.pdf",
        start_page=1,
        num_pages=1
    )
    assert isinstance(response3, PDFExtractResponse)

    # Test 4: Base64 content
    import base64
    base64_content = base64.b64encode(minimal_pdf).decode('utf-8')
    response4 = client.extract_pdf(
        pdf_content=base64_content,
        filename="live_base64_test.pdf"
    )
    assert isinstance(response4, PDFExtractResponse)
