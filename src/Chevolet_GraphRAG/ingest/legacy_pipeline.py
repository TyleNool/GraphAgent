from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Chevolet_GraphRAG.config import Settings
from Chevolet_GraphRAG.ingest.catalog import discover_manual_files
from Chevolet_GraphRAG.ingest.parser import PdfManualParser
from Chevolet_GraphRAG.legacy_neo4j_store import LegacyChunkInput, LegacyNeo4jStore
from Chevolet_GraphRAG.models import IngestStats, build_manual_key
from Chevolet_GraphRAG.providers import build_embeddings


_WHITESPACE_RE = re.compile(r"\s+")


def _split_fixed_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = _WHITESPACE_RE.sub(" ", text or "").strip()
    if not clean:
        return []

    size = max(64, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size - 1))
    step = max(1, size - overlap)

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        chunk = clean[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(clean):
            break
        start += step
    return chunks


@dataclass(slots=True)
class LegacyIngestionPipeline:
    settings: Settings

    def run(
        self,
        data_root: Path,
        init_schema: bool = True,
        max_manuals: int | None = None,
        include_models: Iterable[str] | None = None,
        filename_keywords: Iterable[str] | None = None,
        skip_existing: bool = False,
    ) -> dict[str, object]:
        catalog = discover_manual_files(
            data_root,
            include_models=include_models,
            filename_keywords=filename_keywords,
        )
        parser = PdfManualParser(
            artifact_root=self.settings.artifact_root,
            chunk_size=self.settings.chunk_size_chars,
            chunk_overlap=self.settings.chunk_overlap_chars,
        )
        embeddings = build_embeddings(self.settings)
        store = LegacyNeo4jStore(self.settings)

        stats = IngestStats()
        type_counter: Counter[str] = Counter()
        skipped_existing = 0

        try:
            if init_schema:
                dim = len(embeddings.embed_query("테스트"))
                store.apply_schema(embedding_dim=dim)

            manuals = catalog.manuals
            if max_manuals is not None:
                manuals = manuals[:max_manuals]

            for manual in manuals:
                if skip_existing and store.manual_exists_by_source(manual.file_path.as_posix()):
                    skipped_existing += 1
                    continue

                parsed = parser.parse(manual)
                manual_id = build_manual_key(parsed.manual)
                legacy_chunks: list[LegacyChunkInput] = []

                for page in parsed.pages:
                    fixed_chunks = _split_fixed_chunks(
                        page.text,
                        self.settings.chunk_size_chars,
                        self.settings.chunk_overlap_chars,
                    )
                    page_id = f"{manual_id}::p{page.page_no:04d}"
                    for idx, chunk_text in enumerate(fixed_chunks, start=1):
                        legacy_chunks.append(
                            LegacyChunkInput(
                                chunk_id=f"{page_id}::legacy::{idx:03d}",
                                page_id=page_id,
                                page_no=page.page_no,
                                image_path=page.image_path.as_posix() if page.image_path else None,
                                has_three_column_layout=page.has_three_column_layout,
                                chunk_order=idx,
                                text=chunk_text,
                            )
                        )

                chunk_embeddings = (
                    embeddings.embed_documents([chunk.text for chunk in legacy_chunks])
                    if legacy_chunks
                    else []
                )
                local_stats = store.upsert_manual(
                    parsed=parsed,
                    chunks=legacy_chunks,
                    chunk_embeddings=chunk_embeddings,
                )

                stats.manuals += local_stats.manuals
                stats.pages += local_stats.pages
                stats.chunks += local_stats.chunks
                stats.images += local_stats.images
                stats.tables += local_stats.tables
                type_counter[manual.manual_type] += 1
        finally:
            store.close()

        return {
            "catalog": catalog.summary(),
            "selected_manual_count": len(manuals),
            "skipped_existing": skipped_existing,
            "ingested": stats.model_dump(),
            "manual_type_counter": dict(type_counter),
            "variant": "legacy",
        }
