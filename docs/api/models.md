# Data Models

## Core Models

::: bookwyrm.TextChunk
    options:
      show_source: true

::: bookwyrm.Phrase
    options:
      show_source: true

::: bookwyrm.Citation
    options:
      show_source: true

::: bookwyrm.UsageInfo
    options:
      show_source: true

::: bookwyrm.FileClassification
    options:
      show_source: true

## Request Models

::: bookwyrm.CitationRequest
    options:
      show_source: true

::: bookwyrm.SummarizeRequest
    options:
      show_source: true

::: bookwyrm.ProcessTextRequest
    options:
      show_source: true

::: bookwyrm.ClassifyRequest
    options:
      show_source: true

::: bookwyrm.PDFExtractRequest
    options:
      show_source: true

## Response Models

::: bookwyrm.CitationResponse
    options:
      show_source: true

::: bookwyrm.SummaryResponse
    options:
      show_source: true

::: bookwyrm.ClassifyResponse
    options:
      show_source: true

::: bookwyrm.PDFExtractResponse
    options:
      show_source: true

## PDF Models

::: bookwyrm.PDFTextElement
    options:
      show_source: true

::: bookwyrm.PDFPage
    options:
      show_source: true

::: bookwyrm.PDFStructuredData
    options:
      show_source: true

## Streaming Response Models

::: bookwyrm.CitationProgressUpdate
    options:
      show_source: true

::: bookwyrm.CitationStreamResponse
    options:
      show_source: true

::: bookwyrm.CitationSummaryResponse
    options:
      show_source: true

::: bookwyrm.CitationErrorResponse
    options:
      show_source: true

::: bookwyrm.SummarizeProgressUpdate
    options:
      show_source: true

::: bookwyrm.SummarizeErrorResponse
    options:
      show_source: true

::: bookwyrm.PhraseProgressUpdate
    options:
      show_source: true

::: bookwyrm.PhraseResult
    options:
      show_source: true

## Union Types

::: bookwyrm.StreamingCitationResponse
    options:
      show_source: true

::: bookwyrm.StreamingSummarizeResponse
    options:
      show_source: true

::: bookwyrm.StreamingPhrasalResponse
    options:
      show_source: true

## Enums

::: bookwyrm.ResponseFormat
    options:
      show_source: true
