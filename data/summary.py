from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class Summary(BaseModel):
    """Structured summary model for literary works."""

    title: Optional[str] = Field(
        None,
        description="The title of the literary work. Extract the exact title as it appears in the text, or infer it if clearly referenced.",
    )

    author: Optional[str] = Field(
        None,
        description="The author or authors of the literary work. Include full names when available, or partial names if that's all that's provided.",
    )

    date_of_publication: Optional[date] = Field(
        None,
        description="The publication date of the work in YYYY-MM-DD format. Use the earliest known publication date. If only a year is known, use January 1st of that year.",
    )

    plot: Optional[str] = Field(
        None,
        description="A comprehensive summary of the main plot, storyline, or narrative arc. Include key events, conflicts, and resolutions. For non-fiction, describe the main arguments or themes presented.",
    )

    timeline: Optional[str] = Field(
        None,
        description="The temporal setting or chronological framework of the work. This could include historical periods, fictional timelines, or the sequence of events. Describe when the story takes place or unfolds.",
    )

    important_characters: Optional[List[str]] = Field(
        None,
        description="A list of the most significant characters, people, or entities mentioned in the work. Include protagonists, antagonists, and other key figures. For non-fiction, include important historical figures or people discussed.",
    )
