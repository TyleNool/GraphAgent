from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from neo4j import Driver, GraphDatabase

from Chevolet_GraphRAG.config import Settings
from Chevolet_GraphRAG.ingest.schema import load_schema_cypher
from Chevolet_GraphRAG.models import IngestStats, ParsedManual, build_manual_key


@dataclass(slots=True)
class RetrievedRecord:
    chunk_id: str
    text: str
    source_file: str
    page_no: int
    page_id: str
    display_page_label: str | None
    score: float
    model: str
    manual_type: str
    chunk_order: int
    page_image_path: str | None
    path_summary: str


class Neo4jStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def apply_schema(self, embedding_dim: int = 1024) -> None:
        cypher = load_schema_cypher(embedding_dim=embedding_dim)
        statements = [stmt.strip() for stmt in cypher.split(";") if stmt.strip()]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for stmt in statements:
                session.run(stmt)

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
        chunk_embeddings: list[list[float]],
    ) -> IngestStats:
        stats = IngestStats(manuals=1)

        manual = parsed.manual
        manual_id = build_manual_key(manual)
        section_id = f"{manual_id}::{manual.manual_type}"
        model_key = manual.model.lower().replace(" ", "_")

        chunk_vector_idx = 0

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

            previous_chunk_id: str | None = None
            for page in parsed.pages:
                page_id = f"{manual_id}::p{page.page_no:04d}"
                session.run(
                    """
                    MATCH (manual:Manual {id: $manual_id})
                    MATCH (sec:Section {id: $section_id})
                    MERGE (p:Page {id: $page_id})
                    SET p.page_no = $page_no,
                        p.display_page_label = $display_page_label,
                        p.image_path = $image_path,
                        p.source_file = $source_file,
                        p.model_name = $model_name,
                        p.manual_type = $manual_type,
                        p.has_three_column_layout = $has_three_column_layout,
                        p.updated_at = datetime($updated_at)
                    MERGE (manual)-[:HAS_PAGE]->(p)
                    MERGE (sec)-[:HAS_PAGE]->(p)
                    """,
                    manual_id=manual_id,
                    section_id=section_id,
                    page_id=page_id,
                    page_no=page.page_no,
                    display_page_label=page.display_page_label,
                    image_path=page.image_path.as_posix(),
                    source_file=manual.file_path.as_posix(),
                    model_name=manual.model,
                    manual_type=manual.manual_type,
                    has_three_column_layout=page.has_three_column_layout,
                    updated_at=datetime.utcnow().isoformat(),
                )
                stats.pages += 1

                for chunk in page.chunks:
                    chunk_id = f"{page_id}::c{chunk.chunk_order:03d}"
                    embedding = (
                        chunk_embeddings[chunk_vector_idx]
                        if chunk_vector_idx < len(chunk_embeddings)
                        else []
                    )
                    chunk_vector_idx += 1

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
                            c.page_id = $page_id,
                            c.display_page_label = $display_page_label,
                            c.manual_type = $manual_type,
                            c.model_name = $model_name,
                            c.embedding = $embedding,
                            c.updated_at = datetime($updated_at)
                        MERGE (p)-[:HAS_CHUNK]->(c)
                        MERGE (manual)-[:HAS_CHUNK]->(c)
                        MERGE (model)-[:HAS_CHUNK]->(c)
                        """,
                        page_id=page_id,
                        manual_id=manual_id,
                        model_key=model_key,
                        chunk_id=chunk_id,
                        text=chunk.text,
                        chunk_order=chunk.chunk_order,
                        source_file=manual.file_path.as_posix(),
                        page_no=page.page_no,
                        display_page_label=page.display_page_label,
                        manual_type=manual.manual_type,
                        model_name=manual.model,
                        embedding=embedding,
                        updated_at=datetime.utcnow().isoformat(),
                    )
                    stats.chunks += 1

                    if previous_chunk_id is not None:
                        session.run(
                            """
                            MATCH (prev:Chunk {id: $prev_id})
                            MATCH (cur:Chunk {id: $cur_id})
                            MERGE (prev)-[:NEXT_STEP]->(cur)
                            """,
                            prev_id=previous_chunk_id,
                            cur_id=chunk_id,
                        )
                    previous_chunk_id = chunk_id

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
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_idx', $top_k, $embedding)
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
               node.page_no AS page_no,
               coalesce(node.page_id, p.id, '') AS page_id,
               p.display_page_label AS display_page_label,
               score,
               coalesce(model.name, '') AS model,
               coalesce(manual.manual_type, '') AS manual_type,
               coalesce(node.chunk_order, 0) AS chunk_order,
               p.image_path AS page_image_path,
               ('Model:' + coalesce(model.name, '') + ' -> Manual:' + coalesce(manual.manual_type, '') + ' -> Page:' + coalesce(p.display_page_label, 'PDF ' + toString(coalesce(node.page_no, 0)))) AS path_summary
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
        if not query_text.strip():
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes('chunk_text_ft', $query_text)
        YIELD node, score
        MATCH (model:Model)-[:HAS_CHUNK]->(node)
        WHERE size($model_candidates) = 0
           OR ANY(candidate IN $model_candidates WHERE toLower(model.name) CONTAINS toLower(candidate))
        OPTIONAL MATCH (manual:Manual)-[:HAS_CHUNK]->(node)
        OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(node)
        RETURN DISTINCT node.id AS chunk_id,
               node.text AS text,
               node.source_file AS source_file,
               node.page_no AS page_no,
               coalesce(node.page_id, p.id, '') AS page_id,
               p.display_page_label AS display_page_label,
               score,
               coalesce(model.name, '') AS model,
               coalesce(manual.manual_type, '') AS manual_type,
               coalesce(node.chunk_order, 0) AS chunk_order,
               p.image_path AS page_image_path,
               ('FullText:' + coalesce(model.name, '') + ' -> Manual:' + coalesce(manual.manual_type, '') + ' -> Page:' + coalesce(p.display_page_label, 'PDF ' + toString(coalesce(node.page_no, 0)))) AS path_summary
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
