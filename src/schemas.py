from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal["pdf", "excel", "web"]


class Document(BaseModel):
    source: SourceType
    doc_id: str                 # e.g., Program_ID or URL hash
    title: str                  # pdf filename, program card title, web page title
    text: str                   # extracted plaintext/markdown
    meta: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    source: SourceType
    doc_id: str
    chunk_id: str               # stable: f"{source}::{doc_id}::{i}"
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: SourceType
    reference: str              # pdf filename, Program_ID, or URL
    chunk_id: Optional[str] = None


class AnswerResult(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
