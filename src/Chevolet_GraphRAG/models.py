from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvidenceSource(BaseModel):
    source_file: str
    page_no: int
    chunk_id: str
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
    score_kind: str | None = None
    relevance_label: str | None = None
    path_summary: str
    source_type: str = "manual"
    page_id: str | None = None
    display_page_label: str | None = None
    manual_type: str | None = None
    model: str | None = None


class RetrievalItem(BaseModel):
    chunk_id: str
    text: str
    source_file: str
    page_no: int
    page_id: str | None = None
    display_page_label: str | None = None
    score: float
    source_type: str = "manual"
    relations: list[str] = Field(default_factory=list)
    image_path: str | None = None
    manual_type: str | None = None
    model: str | None = None


class PageRetrievalResult(BaseModel):
    page_id: str
    source_file: str
    page_no: int
    display_page_label: str | None = None
    score: float
    retrieval_score: float | None = None
    rerank_score: float | None = None
    path_summary: str
    image_path: str | None = None
    manual_type: str | None = None
    model: str | None = None
    supporting_items: list[RetrievalItem] = Field(default_factory=list)


class GuardrailDecision(BaseModel):
    allow: bool
    reason: str
    normalized_brand: str = "쉐보레"
    normalized_model: str | None = None
    model_candidates: list[str] = Field(default_factory=list)
    fallback_category: str | None = None
    preferred_manual_types: list[str] = Field(default_factory=list)
    requested_action: str = "answer"
    prefer_faq: bool = False


class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: str
    user_query: str
    model_hint: str | None = None
    feedback: str | None = None
    resolved: bool | None = None
    top_k: int = 5


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    confidence: float
    resolved: bool | None = None
    top_image_path: str | None = None
    top_manual_sources: list[EvidenceSource] = Field(default_factory=list)
    top_faq_sources: list[EvidenceSource] = Field(default_factory=list)
    # Legacy field kept for compatibility (manual-first sources).
    top_sources: list[EvidenceSource] = Field(default_factory=list)
    graph_paths: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


class IngestStats(BaseModel):
    manuals: int = 0
    pages: int = 0
    chunks: int = 0
    images: int = 0
    tables: int = 0


class ManualFile(BaseModel):
    brand: str = "쉐보레"
    model: str
    manual_type: str
    file_path: Path


def build_manual_key(manual: ManualFile) -> str:
    digest = hashlib.sha1(manual.file_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{manual.model}::{manual.file_path.stem}::{digest}"


def build_manual_artifact_slug(manual: ManualFile) -> str:
    digest = hashlib.sha1(manual.file_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{manual.file_path.stem}_{digest}"


class PageArtifact(BaseModel):
    page_no: int
    display_page_label: str | None = None
    text: str
    image_path: Path
    chunks: list["ChunkArtifact"]
    has_three_column_layout: bool = False


class ChunkArtifact(BaseModel):
    chunk_order: int
    text: str


class ParsedManual(BaseModel):
    manual: ManualFile
    pages: list[PageArtifact]
