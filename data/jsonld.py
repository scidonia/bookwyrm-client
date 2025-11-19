from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, AnyUrl, Field


class Person(BaseModel):
    # JSON-LD keys with aliases
    context: str = Field("https://schema.org", alias="@context")
    type: str = Field("Person", alias="@type")
    name: str
    same_as: Optional[List[AnyUrl]] = Field(default=None, alias="sameAs")


class LiteraryWork(BaseModel):
    # JSON-LD keys
    context: str = Field("https://schema.org", alias="@context")
    type: str = Field(
        "Book", alias="@type"  # or "CreativeWork" if you want to be more general
    )

    # Core creative work fields
    name: str  # title of the work
    author: Union[Person, List[Person]]  # schema: author
    description: Optional[str] = None
    in_language: Optional[str] = Field(  # e.g. "en", "fr"
        default=None, alias="inLanguage"
    )
    date_published: Optional[str] = Field(  # ISO date string "YYYY-MM-DD" or "YYYY"
        default=None, alias="datePublished"
    )
    same_as: Optional[List[AnyUrl]] = Field(
        default=None, alias="sameAs"
    )  # links to Wikidata, Wikipedia, VIAF, etc.

    # Book-specific / bibliographic fields
    isbn: Optional[str] = None
    number_of_pages: Optional[int] = Field(default=None, alias="numberOfPages")
    book_edition: Optional[str] = Field(default=None, alias="bookEdition")
    # you could make bookFormat a literal/enum of schema.org formats
    book_format: Optional[str] = Field(default=None, alias="bookFormat")

    # Optional identifiers, publishers, etc.
    publisher: Optional[Union[Person, str]] = None  # could also model as Organization
    url: Optional[AnyUrl] = None
    identifier: Optional[str] = (
        None  # for things like OCLC/LCCN if you don’t want full PropertyValue
    )

    class Config:
        allow_population_by_field_name = True
        # Ensure that when you .json(by_alias=True) you get @context, @type, etc.
