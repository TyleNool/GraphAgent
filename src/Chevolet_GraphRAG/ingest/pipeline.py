from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chevy_troubleshooter.config import Settings
from chevy_troubleshooter.ingest.catalog import discover_manual_files
from chevy_troubleshooter.ingest.parser import PdfManualParser
from chevy_troubleshooter.models import IngestStats, build_manual_key
from chevy_troubleshooter.neo4j_store import Neo4jStore
from chevy_troubleshooter.providers import build_embeddings


DTC_PATTERN = re.compile(r"\b[PCBU][0-9]{4}\b", re.IGNORECASE)

ENTITY_KEYWORDS = [
    "엔진",
    "배터리",
    "브레이크",
    "변속기",
    "냉각수",
    "오일",
    "타이어",
    "연료",
    "에어백",
    "히터",
    "에어컨",
    "점화",
]

SYMPTOM_KEYWORDS = [
    "경고등",
    "시동",
    "소음",
    "진동",
    "과열",
    "누유",
    "출력저하",
    "냄새",
    "제동불량",
]

ACTION_KEYWORDS = [
    "점검",
    "교체",
    "확인",
    "정비",
    "청소",
    "재시동",
    "리셋",
]


@dataclass(slots=True)
class IngestionPipeline:
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
        store = Neo4jStore(self.settings)

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
                text_chunks = [chunk.text for page in parsed.pages for chunk in page.chunks]

                chunk_embeddings = embeddings.embed_documents(text_chunks) if text_chunks else []

                local_stats = store.upsert_manual(
                    parsed=parsed,
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
        }

    def _upsert_semantic_entities(self, store: Neo4jStore, parsed_manual) -> None:
        manual_id = build_manual_key(parsed_manual.manual)

        for page in parsed_manual.pages:
            page_id = f"{manual_id}::p{page.page_no:04d}"
            for chunk in page.chunks:
                chunk_id = f"{page_id}::c{chunk.chunk_order:03d}"
                chunk_text = chunk.text

                entities = {word for word in ENTITY_KEYWORDS if word in chunk_text}
                symptoms = {word for word in SYMPTOM_KEYWORDS if word in chunk_text}
                actions = {word for word in ACTION_KEYWORDS if word in chunk_text}
                dtc_codes = {c.upper() for c in DTC_PATTERN.findall(chunk_text)}

                for entity in entities:
                    store.run_query(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (e:Entity {key: $key})
                        SET e.name = $name
                        MERGE (c)-[:MENTIONS_ENTITY]->(e)
                        """,
                        {
                            "chunk_id": chunk_id,
                            "key": f"entity::{entity}",
                            "name": entity,
                        },
                    )

                for symptom in symptoms:
                    store.run_query(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (s:Symptom {key: $key})
                        SET s.name = $name
                        MERGE (c)-[:HAS_SYMPTOM]->(s)
                        """,
                        {
                            "chunk_id": chunk_id,
                            "key": f"symptom::{symptom}",
                            "name": symptom,
                        },
                    )

                for action in actions:
                    store.run_query(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (a:Action {key: $key})
                        SET a.name = $name
                        MERGE (c)-[:RESOLVED_BY]->(a)
                        """,
                        {
                            "chunk_id": chunk_id,
                            "key": f"action::{action}",
                            "name": action,
                        },
                    )

                for code in dtc_codes:
                    store.run_query(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (d:DTC {code: $code})
                        MERGE (c)-[:REFERS_TO]->(d)
                        """,
                        {"chunk_id": chunk_id, "code": code},
                    )
