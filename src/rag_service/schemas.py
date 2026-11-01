from __future__ import annotations

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    query: str = Field(description="Standalone search query with all needed context")


class RelevanceGrades(BaseModel):
    relevant_ids: list[int] = Field(
        description="Ids of fragments that contain information useful to answer the question"
    )


class GeneratedAnswer(BaseModel):
    answer: str = Field(description="Answer in Markdown with [n] citations after each claim")
    cited_ids: list[int] = Field(description="Ids of fragments actually used in the answer")
    insufficient_context: bool = Field(
        description="True if the fragments do not contain enough information to answer"
    )


class Citation(BaseModel):
    n: int
    location: str
    source: str
    snippet: str


class Answer(BaseModel):
    """Final API response."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    insufficient_context: bool = False
    search_query: str
    rewrites: int = 0
