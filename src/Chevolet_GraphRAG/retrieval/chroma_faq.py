from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from Chevolet_GraphRAG.models import RetrievalItem

logger = logging.getLogger(__name__)

COLLECTION_NAME = "chevrolet_faq"


class ChromaFAQStore:
    """ChromaDB-backed vector store for Chevrolet FAQ data."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_faq(
        self,
        json_path: str | Path,
        embeddings,
        *,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Load FAQ JSON and upsert into ChromaDB with embeddings."""
        json_path = Path(json_path)
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array, got {type(raw).__name__}")

        if reset:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        ids: list[str] = []

        for idx, item in enumerate(raw):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            category = item.get("category", "일반").strip()

            if not question:
                continue

            doc_text = f"Q: {question}\nA: {answer}"
            doc_id = f"faq::{category}::{idx:04d}"

            documents.append(doc_text)
            metadatas.append({
                "category": category,
                "question": question,
                "source": "faq",
                "source_file": json_path.name,
            })
            ids.append(doc_id)

        if not documents:
            return {"ingested": 0}

        # Embed all documents
        logger.info("Embedding %d FAQ documents...", len(documents))
        vectors = embeddings.embed_documents(documents)

        # Upsert in batches (ChromaDB limit: 5461 per batch)
        batch_size = 500
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Ingested %d FAQ items into ChromaDB", len(documents))
        return {
            "ingested": len(documents),
            "collection": COLLECTION_NAME,
            "persist_dir": str(self.persist_dir),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_faq(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievalItem]:
        """Search FAQ by vector similarity, returning RetrievalItem list."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items: list[RetrievalItem] = []
        if not results or not results.get("ids"):
            return items

        for doc_id, doc, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score: 1 - (distance / 2)
            score = max(0.0, 1.0 - distance / 2.0)

            items.append(
                RetrievalItem(
                    chunk_id=doc_id,
                    text=doc,
                    source_file=meta.get("source_file", "faq"),
                    page_no=0,
                    score=score,
                    source_type="faq",
                    relations=[f"FAQ 카테고리: {meta.get('category', '')}"],
                    image_path=None,
                )
            )

        return items
