from __future__ import annotations

import logging

from Chevolet_GraphRAG.legacy_neo4j_store import LegacyNeo4jStore
from Chevolet_GraphRAG.models import PageRetrievalResult, RetrievalItem
from Chevolet_GraphRAG.providers import build_chat_model, build_embeddings
from Chevolet_GraphRAG.retrieval.chroma_faq import ChromaFAQStore
from Chevolet_GraphRAG.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class LegacyHybridRetriever(HybridRetriever):
    def __init__(self, settings, store: LegacyNeo4jStore) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = build_embeddings(settings)
        self.chat_model = build_chat_model(settings)
        self.reranker = None
        self.faq_store = ChromaFAQStore(persist_dir=settings.chroma_persist_dir)
        logger.info("Legacy retriever enabled (flat chunks, no reranker)")
        logger.info("ChromaDB FAQ store loaded (%d docs)", self.faq_store.count)

    def retrieve(
        self,
        query: str,
        top_k: int,
        model_candidates: list[str] | None = None,
        prefer_faq: bool = False,
        excluded_chunk_ids: list[str] | None = None,
        preferred_manual_types: list[str] | None = None,
    ) -> tuple[list[RetrievalItem], list[PageRetrievalResult], list[str], dict[str, object]]:
        compact_query = self._compact_query_for_embedding(query)
        query_vector = self.embeddings.embed_query(compact_query)
        preferred_manual_types = preferred_manual_types or []
        model_candidates = model_candidates or []
        relevance_keywords = self._extract_relevance_keywords(query, model_candidates)

        candidate_k = max(top_k * 4, 20)
        vector_hits = self.store.search_chunks_by_vector(
            embedding=query_vector,
            top_k=candidate_k,
            model_candidates=model_candidates,
        )
        lexical_hits = self.store.search_chunks_by_fulltext(
            query_text=self._build_fulltext_query(query),
            top_k=candidate_k,
            model_candidates=model_candidates,
        )
        faq_hits = self.faq_store.search_faq(
            query_embedding=query_vector,
            top_k=max(top_k, 5),
        )

        manual_items = self._fuse_manual(
            vector_hits=vector_hits,
            lexical_hits=lexical_hits,
            top_k=candidate_k,
            excluded_chunk_ids=excluded_chunk_ids or [],
            preferred_manual_types=preferred_manual_types,
        )

        page_candidates = [
            PageRetrievalResult(
                page_id=item.page_id or item.chunk_id,
                source_file=item.source_file,
                page_no=item.page_no,
                display_page_label=item.display_page_label,
                score=item.score,
                retrieval_score=item.score,
                rerank_score=None,
                path_summary=(item.relations or [""])[0],
                image_path=item.image_path,
                manual_type=item.manual_type,
                model=item.model,
                supporting_items=[item],
            )
            for item in manual_items
        ]
        page_candidates = self._filter_pages_by_query_relevance(
            page_candidates,
            relevance_keywords,
        )
        faq_hits = self._filter_faq_hits_by_query_relevance(
            faq_hits,
            relevance_keywords,
        )
        selected_pages = page_candidates[:top_k]
        selected_items = self._collect_supporting_items(
            selected_pages=selected_pages,
            faq_hits=faq_hits,
            top_k=max(top_k, 5),
            prefer_faq=prefer_faq,
        )

        debug = {
            "variant": "legacy",
            "vector_hit_count": len(vector_hits),
            "lexical_hit_count": len(lexical_hits),
            "faq_hit_count": len(faq_hits),
            "manual_candidate_count": len(manual_items),
            "page_candidate_count": len(page_candidates),
            "reranked_page_count": 0,
            "returned_count": len(selected_items),
            "manual_returned_count": len([item for item in selected_items if item.source_type == "manual"]),
            "relevance_keywords": relevance_keywords,
            "preferred_manual_types": preferred_manual_types,
            "model_candidates": model_candidates,
            "prefer_faq": prefer_faq,
            "excluded_chunk_ids": excluded_chunk_ids or [],
            "compact_query_chars": len(compact_query),
        }
        return selected_items, selected_pages, [], debug
