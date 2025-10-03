# AI Integration Guide

**Point your AI here** - This page provides the essential models and function signatures needed for AI agents and automated systems to integrate with BookWyrm.

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

::: bookwyrm.BookWyrmClient.extract_pdf
    options:
      show_root_heading: true

::: bookwyrm.BookWyrmClient.stream_process_text
    options:
      show_root_heading: true

::: bookwyrm.BookWyrmClient.get_citations
    options:
      show_root_heading: true

::: bookwyrm.BookWyrmClient.summarize
    options:
      show_root_heading: true

## Asynchronous Client Methods

::: bookwyrm.AsyncBookWyrmClient.classify
    options:
      show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.extract_pdf
    options:
      show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.stream_process_text
    options:
      show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.get_citations
    options:
      show_root_heading: true

::: bookwyrm.AsyncBookWyrmClient.summarize
    options:
      show_root_heading: true
