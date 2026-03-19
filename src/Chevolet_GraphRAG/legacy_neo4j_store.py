from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from neo4j import Driver, GraphDatabase

from Chevolet_GraphRAG.config import Settings
from Chevolet_GraphRAG.ingest.legacy_schema import load_legacy_schema_cypher
from Chevolet_GraphRAG.models import IngestStats, ParsedManual, build_manual_key
from Chevolet_GraphRAG.neo4j_store import RetrievedRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LegacyChunkInput:
    chunk_id: str
    page_id: str
    page_no: int
    image_path: str | None
    has_three_column_layout: bool
    chunk_order: int
    text: str


class LegacyNeo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.vector_index_name: str | None = None
        self.fulltext_index_name: str | None = None
        self._refresh_index_capabilities()

    def close(self) -> None:
        self.driver.close()

    def apply_schema(self, embedding_dim: int = 1024) -> None:
        cypher = load_legacy_schema_cypher(embedding_dim=embedding_dim)
        statements = [stmt.strip() for stmt in cypher.split(";") if stmt.strip()]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for stmt in statements:
                session.run(stmt)
        self._refresh_index_capabilities()

    def _refresh_index_capabilities(self) -> None:
        try:
            rows = self.run_query("SHOW INDEXES YIELD name RETURN name")
        except Exception as exc:
            logger.warning("Failed to inspect Neo4j indexes for legacy mode: %s", exc)
            return

        names = {str(row.get("name", "")) for row in rows}
        self.vector_index_name = next(
            (name for name in ("legacy_chunk_embedding_idx", "chunk_embedding_idx") if name in names),
            None,
        )
        self.fulltext_index_name = next(
            (name for name in ("legacy_chunk_text_ft", "chunk_text_ft") if name in names),
            None,
        )
        logger.info(
            "Legacy store index selection: vector=%s fulltext=%s",
            self.vector_index_name or "missing",
            self.fulltext_index_name or "disabled",
        )

    def manual_exists_by_source(self, source_file: str) -> bool:
        rows = self.run_query(
            """
            MATCH (m:Manual {source_file: $source_file})
            RETURN count(m) > 0 AS exists
            """,
            {"source_file": source_file},
        )
        if not rows:
            return False
        return bool(rows[0].get("exists", False))

    def upsert_manual(
        self,
        parsed: ParsedManual,
        chunks: list[LegacyChunkInput],
        chunk_embeddings: list[list[float]],
    ) -> IngestStats:
        stats = IngestStats(manuals=1)

        manual = parsed.manual
        manual_id = build_manual_key(manual)
        section_id = f"{manual_id}::{manual.manual_type}"
        model_key = manual.model.lower().replace(" ", "_")

        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(
                """
                MERGE (b:Brand {name: $brand})
                MERGE (m:Model {key: $model_key})
                SET m.name = $model_name
                MERGE (b)-[:HAS_MODEL]->(m)
                MERGE (manual:Manual {id: $manual_id})
                SET manual.name = $manual_name,
                    manual.manual_type = $manual_type,
                    manual.source_file = $source_file,
                    manual.updated_at = datetime($updated_at)
                MERGE (m)-[:HAS_MANUAL]->(manual)
                MERGE (sec:Section {id: $section_id})
                SET sec.name = $manual_type
                MERGE (manual)-[:HAS_SECTION]->(sec)
                """,
                brand=manual.brand,
                model_key=model_key,
                model_name=manual.model,
                manual_id=manual_id,
                manual_name=manual.file_path.stem,
                manual_type=manual.manual_type,
                source_file=manual.file_path.as_posix(),
                updated_at=datetime.utcnow().isoformat(),
                section_id=section_id,
            )

            seen_pages: set[str] = set()
            for idx, chunk in enumerate(chunks):
                embedding = chunk_embeddings[idx] if idx < len(chunk_embeddings) else []
                if chunk.page_id not in seen_pages:
                    session.run(
                        """
                        MATCH (manual:Manual {id: $manual_id})
                        MATCH (sec:Section {id: $section_id})
                        MERGE (p:Page {id: $page_id})
                        SET p.page_no = $page_no,
                            p.source_file = $source_file,
                            p.image_path = $image_path,
                            p.has_three_column_layout = $has_three_column_layout,
                            p.updated_at = datetime($updated_at)
                        MERGE (manual)-[:HAS_PAGE]->(p)
                        MERGE (sec)-[:HAS_PAGE]->(p)
                        """,
                        manual_id=manual_id,
                        section_id=section_id,
                        page_id=chunk.page_id,
                        page_no=chunk.page_no,
                        source_file=manual.file_path.as_posix(),
                        image_path=chunk.image_path,
                        has_three_column_layout=chunk.has_three_column_layout,
                        updated_at=datetime.utcnow().isoformat(),
                    )
                    seen_pages.add(chunk.page_id)
                    stats.pages += 1
                    if chunk.image_path:
                        stats.images += 1

                session.run(
                    """
                    MATCH (p:Page {id: $page_id})
                    MATCH (manual:Manual {id: $manual_id})
                    MATCH (model:Model {key: $model_key})
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $text,
                        c.chunk_order = $chunk_order,
                        c.source_file = $source_file,
                        c.page_no = $page_no,
                        c.embedding = $embedding,
                        c.updated_at = datetime($updated_at)
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    MERGE (manual)-[:HAS_CHUNK]->(c)
                    MERGE (model)-[:HAS_CHUNK]->(c)
                    """,
                    page_id=chunk.page_id,
                    manual_id=manual_id,
                    model_key=model_key,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    chunk_order=chunk.chunk_order,
                    source_file=manual.file_path.as_posix(),
                    page_no=chunk.page_no,
                    embedding=embedding,
                    updated_at=datetime.utcnow().isoformat(),
                )
                stats.chunks += 1

        return stats

    def run_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver.session(database=self.settings.neo4j_database) as session:
            result = session.run(cypher, **(params or {}))
            return [record.data() for record in result]

    def search_chunks_by_vector(
        self,
        embedding: list[float],
        top_k: int,
        model_candidates: list[str] | None = None,
    ) -> list[RetrievedRecord]:
        if not self.vector_index_name:
            raise RuntimeError(
                "No compatible vector index found. Expected one of: legacy_chunk_embedding_idx, chunk_embedding_idx"
            )

        cypher = f"""
        CALL db.index.vector.queryNodes('{self.vector_index_name}', $top_k, $embedding)
        YIELD node, score
        MATCH (model:Model)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (manual:Manual)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(node)
        WITH node, score, model, manual, p
        WHERE size($model_candidates) = 0
           OR ANY(candidate IN $model_candidates WHERE toLower(model.name) CONTAINS toLower(candidate))
        RETURN node.id AS chunk_id,
               node.text AS text,
               node.source_file AS source_file,
               coalesce(p.page_no, node.page_no, 0) AS page_no,
               coalesce(p.id, node.id) AS page_id,
               ('PDF ' + toString(coalesce(p.page_no, node.page_no, 0))) AS display_page_label,
               score,
               coalesce(model.name, '') AS model,
               coalesce(manual.manual_type, '') AS manual_type,
               coalesce(node.chunk_order, 0) AS chunk_order,
               p.image_path AS page_image_path,
               ('LegacyVector:' + coalesce(model.name, '') + ' -> Manual:' + coalesce(manual.manual_type, '') + ' -> Page:PDF ' + toString(coalesce(p.page_no, node.page_no, 0))) AS path_summary
        ORDER BY score DESC
        LIMIT $top_k
        """
        rows = self.run_query(
            cypher,
            {
                "embedding": embedding,
                "top_k": top_k,
                "model_candidates": model_candidates or [],
            },
        )
        return [RetrievedRecord(**row) for row in rows]

    def search_chunks_by_fulltext(
        self,
        query_text: str,
        top_k: int,
        model_candidates: list[str] | None = None,
    ) -> list[RetrievedRecord]:
        if not query_text.strip() or not self.fulltext_index_name:
            return []

        cypher = f"""
        CALL db.index.fulltext.queryNodes('{self.fulltext_index_name}', $query_text)
        YIELD node, score
        MATCH (model:Model)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (manual:Manual)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(node)
        WITH node, score, model, manual, p
        WHERE size($model_candidates) = 0
           OR ANY(candidate IN $model_candidates WHERE toLower(model.name) CONTAINS toLower(candidate))
        RETURN DISTINCT node.id AS chunk_id,
               node.text AS text,
               node.source_file AS source_file,
               coalesce(p.page_no, node.page_no, 0) AS page_no,
               coalesce(p.id, node.id) AS page_id,
               ('PDF ' + toString(coalesce(p.page_no, node.page_no, 0))) AS display_page_label,
               score,
               coalesce(model.name, '') AS model,
               coalesce(manual.manual_type, '') AS manual_type,
               coalesce(node.chunk_order, 0) AS chunk_order,
               p.image_path AS page_image_path,
               ('LegacyFullText:' + coalesce(model.name, '') + ' -> Manual:' + coalesce(manual.manual_type, '') + ' -> Page:PDF ' + toString(coalesce(p.page_no, node.page_no, 0))) AS path_summary
        ORDER BY score DESC
        LIMIT $top_k
        """
        rows = self.run_query(
            cypher,
            {
                "query_text": query_text,
                "top_k": top_k,
                "model_candidates": model_candidates or [],
            },
        )
        return [RetrievedRecord(**row) for row in rows]
