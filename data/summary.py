from pydantic import BaseModel
from typing import Optional, List
from datetime import date


class Summary(BaseModel):
    """Structured summary model for literary works."""

    title: Optional[str] = None
    author: Optional[str] = None
    date_of_publication: Optional[date] = None
    plot: Optional[str] = None
    timeline: Optional[str] = None
    important_characters: Optional[List[str]] = None
