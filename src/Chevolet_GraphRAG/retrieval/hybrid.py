from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from Chevolet_GraphRAG.config import Settings
from Chevolet_GraphRAG.models import PageRetrievalResult, RetrievalItem
from Chevolet_GraphRAG.neo4j_store import Neo4jStore, RetrievedRecord
from Chevolet_GraphRAG.providers import build_chat_model, build_embeddings, build_reranker
from Chevolet_GraphRAG.retrieval.chroma_faq import ChromaFAQStore

logger = logging.getLogger(__name__)


STOPWORDS = {
    "그리고",
    "그런데",
    "문의",
    "질문",
    "차량",
    "쉐보레",
    "에서",
    "으로",
    "합니다",
    "해주세요",
}

RELEVANCE_STOPWORDS = STOPWORDS | {
    "모델",
    "차종",
    "관련",
    "대한",
    "무엇",
    "뭔가요",
    "어떻게",
    "되나요",
    "있나요",
    "알려줘",
    "알려주세요",
    "확인",
    "사용",
    "가능",
}


class HybridRetriever:
    def __init__(self, settings: Settings, store: Neo4jStore) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = build_embeddings(settings)
        self.chat_model = build_chat_model(settings)
        self.reranker = build_reranker(settings)
        self.faq_store = ChromaFAQStore(persist_dir=settings.chroma_persist_dir)
        logger.info("ChromaDB FAQ store loaded (%d docs)", self.faq_store.count)
        if self.reranker:
            logger.info("CrossEncoder reranker initialized.")

    def retrieve(
        self,
        query: str,
        top_k: int,
        model_candidates: list[str] | None = None,
        prefer_faq: bool = False,
        excluded_chunk_ids: list[str] | None = None,
        preferred_manual_types: list[str] | None = None,
    ) -> tuple[list[RetrievalItem], list[PageRetrievalResult], list[str], dict[str, Any]]:
        compact_query = self._compact_query_for_embedding(query)
        query_vector = self.embeddings.embed_query(compact_query)
        preferred_manual_types = preferred_manual_types or []
        model_candidates = model_candidates or []
        relevance_keywords = self._extract_relevance_keywords(query, model_candidates)

        child_candidate_k = max(top_k * 10, 40)

        vector_hits = self.store.search_chunks_by_vector(
            embedding=query_vector,
            top_k=child_candidate_k,
            model_candidates=model_candidates,
        )
        lexical_hits = self.store.search_chunks_by_fulltext(
            query_text=self._build_fulltext_query(query),
            top_k=child_candidate_k,
            model_candidates=model_candidates,
        )

        # FAQ retrieval with increased candidates for reranking
        faq_candidate_k = max(top_k * 3, 10)  # Get more candidates for reranking
        faq_hits = self.faq_store.search_faq(
            query_embedding=query_vector,
            top_k=faq_candidate_k,
        )

        # Rerank FAQ hits using Cohere reranker
        faq_hits = self._rerank_faq_hits(
            query=query,
            faq_hits=faq_hits,
            top_k=max(top_k, 3),
        )

        manual_candidates = self._fuse_manual(
            vector_hits=vector_hits,
            lexical_hits=lexical_hits,
            top_k=child_candidate_k,
            excluded_chunk_ids=excluded_chunk_ids or [],
            preferred_manual_types=preferred_manual_types,
        )
        page_candidates = self._aggregate_pages(
            manual_candidates,
            preferred_manual_types=preferred_manual_types,
        )
        selected_pages = self._rerank_pages(
            query=query,
            page_candidates=page_candidates,
            top_k=top_k,
        )
        selected_pages = self._filter_pages_by_query_relevance(
            selected_pages,
            relevance_keywords,
        )
        faq_hits = self._filter_faq_hits_by_query_relevance(
            faq_hits,
            relevance_keywords,
        )
        selected_items = self._collect_supporting_items(
            selected_pages=selected_pages,
            faq_hits=faq_hits,
            top_k=max(top_k, 5),
            prefer_faq=prefer_faq,
        )

        graph_paths = self._collect_graph_paths(
            [item.chunk_id for item in selected_items if not item.chunk_id.startswith("faq::")]
        )

        debug = {
            "vector_hit_count": len(vector_hits),
            "lexical_hit_count": len(lexical_hits),
            "faq_hit_count": len(faq_hits),
            "manual_candidate_count": len(manual_candidates),
            "page_candidate_count": len(page_candidates),
            "reranked_page_count": len(selected_pages),
            "returned_count": len(selected_items),
            "manual_returned_count": len([item for item in selected_items if item.source_type == "manual"]),
            "relevance_keywords": relevance_keywords,
            "preferred_manual_types": preferred_manual_types,
            "model_candidates": model_candidates,
            "prefer_faq": prefer_faq,
            "excluded_chunk_ids": excluded_chunk_ids or [],
            "compact_query_chars": len(compact_query),
        }
        return selected_items, selected_pages, graph_paths, debug

    def graph_cypher_probe(self, query: str) -> dict[str, Any]:
        """Optional GraphCypherQAChain probe for additional structured evidence."""
        try:
            from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
            from langchain_neo4j import Neo4jGraph

            if self.chat_model is None:
                return {}

            graph = Neo4jGraph(
                url=self.settings.neo4j_uri,
                username=self.settings.neo4j_username,
                password=self.settings.neo4j_password,
                database=self.settings.neo4j_database,
            )
            chain = GraphCypherQAChain.from_llm(
                llm=self.chat_model,
                graph=graph,
                verbose=False,
                return_intermediate_steps=True,
            )
            result = chain.invoke({"query": query})
            return {
                "result": result.get("result", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
            }
        except Exception:
            return {}

    def _extract_keywords(self, query: str) -> list[str]:
        terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query)
        filtered = [t for t in terms if t not in STOPWORDS]
        # High-value tokens first.
        return sorted(set(filtered), key=lambda x: len(x), reverse=True)[:8]

    def _normalize_match_text(self, text: str) -> str:
        return re.sub(r"[^0-9a-z가-힣]+", "", (text or "").lower())

    def _extract_relevance_keywords(
        self,
        query: str,
        model_candidates: list[str],
    ) -> list[str]:
        model_tokens = {self._normalize_match_text(candidate) for candidate in model_candidates}
        keywords: list[str] = []
        for term in self._extract_keywords(query):
            normalized = self._normalize_match_text(term)
            if len(normalized) < 2:
                continue
            if normalized in RELEVANCE_STOPWORDS:
                continue
            if any(
                normalized == token or normalized in token or token in normalized
                for token in model_tokens
                if token
            ):
                continue
            keywords.append(normalized)
        return list(dict.fromkeys(keywords))

    def _text_matches_query_keywords(self, text: str, keywords: list[str]) -> bool:
        if not keywords:
            return True
        normalized_text = self._normalize_match_text(text)
        return any(keyword in normalized_text for keyword in keywords)

    def _compact_query_for_embedding(self, query: str) -> str:
        clean = re.sub(r"\s+", " ", query).strip()
        max_chars = max(120, int(self.settings.query_text_max_chars))
        if len(clean) <= max_chars:
            return clean

        keywords = self._extract_keywords(clean)
        keyword_payload = " ".join(keywords)
        remaining = max_chars - len(keyword_payload) - 5
        if remaining < 40:
            return clean[:max_chars]
        return f"{clean[:remaining]} | {keyword_payload}"[:max_chars]

    def _build_fulltext_query(self, query: str) -> str:
        keywords = self._extract_keywords(query)
        if not keywords:
            return re.sub(r"[^\w가-힣\s]", " ", query).strip()

        terms: list[str] = []
        for keyword in keywords[:8]:
            safe = re.sub(r"[^\w가-힣]", "", keyword)
            if len(safe) < 2:
                continue
            terms.append(f'"{safe}"^2')
            terms.append(safe)
        return " OR ".join(dict.fromkeys(terms))

    def _fuse_manual(
        self,
        vector_hits: list[RetrievedRecord],
        lexical_hits: list[RetrievedRecord],
        top_k: int,
        excluded_chunk_ids: list[str],
        preferred_manual_types: list[str],
    ) -> list[RetrievalItem]:
        excluded = set(excluded_chunk_ids)
        scores: dict[str, float] = defaultdict(float)
        payload: dict[str, dict[str, Any]] = {}
        lexical_max = max((float(hit.score) for hit in lexical_hits), default=1.0) or 1.0
        preferred_types = set(preferred_manual_types)

        for rank, hit in enumerate(vector_hits, start=1):
            if hit.chunk_id in excluded:
                continue
            rrf = 1.0 / (60 + rank)
            score = 0.62 * rrf + 0.38 * float(hit.score)
            if hit.manual_type in preferred_types:
                score += 0.05
            scores[hit.chunk_id] += score
            payload[hit.chunk_id] = {
                "chunk_id": hit.chunk_id,
                "text": hit.text,
                "source_file": hit.source_file,
                "page_no": hit.page_no,
                "page_id": hit.page_id,
                "display_page_label": hit.display_page_label,
                "score": hit.score,
                "source_type": "manual",
                "relations": [hit.path_summary],
                "image_path": hit.page_image_path,
                "manual_type": hit.manual_type,
                "model": hit.model,
            }

        for rank, hit in enumerate(lexical_hits, start=1):
            if hit.chunk_id in excluded:
                continue
            rrf = 1.0 / (60 + rank)
            lexical_score = float(hit.score) / lexical_max
            score = 0.46 * rrf + 0.24 * lexical_score
            if hit.manual_type in preferred_types:
                score += 0.07
            scores[hit.chunk_id] += score
            payload.setdefault(
                hit.chunk_id,
                {
                    "chunk_id": hit.chunk_id,
                    "text": hit.text,
                    "source_file": hit.source_file,
                    "page_no": hit.page_no,
                    "page_id": hit.page_id,
                    "display_page_label": hit.display_page_label,
                    "score": lexical_score,
                    "source_type": "manual",
                    "relations": [],
                    "image_path": hit.page_image_path,
                    "manual_type": hit.manual_type,
                    "model": hit.model,
                },
            )
            payload[hit.chunk_id]["relations"].append(hit.path_summary)

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]
        items: list[RetrievalItem] = []
        for cid in ranked_ids:
            item = payload[cid]
            item["score"] = float(scores[cid])
            items.append(RetrievalItem(**item))
        return items

    def _aggregate_pages(
        self,
        manual_items: list[RetrievalItem],
        preferred_manual_types: list[str],
    ) -> list[PageRetrievalResult]:
        grouped: dict[str, list[RetrievalItem]] = defaultdict(list)
        for item in manual_items:
            if item.source_type != "manual" or not item.page_id:
                continue
            grouped[item.page_id].append(item)

        preferred_types = set(preferred_manual_types)
        pages: list[PageRetrievalResult] = []
        for page_id, items in grouped.items():
            ranked = sorted(items, key=lambda item: item.score, reverse=True)
            top_score = ranked[0].score
            support_bonus = 0.06 * max(0, min(len(ranked) - 1, 3))
            tail_bonus = 0.03 * sum(item.score for item in ranked[1:3])
            manual_type_bonus = 0.05 if ranked[0].manual_type in preferred_types else 0.0
            page_score = top_score + support_bonus + tail_bonus + manual_type_bonus

            first = ranked[0]
            pages.append(
                PageRetrievalResult(
                    page_id=page_id,
                    source_file=first.source_file,
                    page_no=first.page_no,
                    display_page_label=first.display_page_label,
                    score=page_score,
                    retrieval_score=page_score,
                    rerank_score=None,
                    path_summary=(first.relations or [""])[0],
                    image_path=first.image_path,
                    manual_type=first.manual_type,
                    model=first.model,
                    supporting_items=ranked[:4],
                )
            )

        return sorted(pages, key=lambda page: page.score, reverse=True)

    def _rerank_pages(
        self,
        query: str,
        page_candidates: list[PageRetrievalResult],
        top_k: int,
    ) -> list[PageRetrievalResult]:
        if not page_candidates:
            return []

        rerank_pool = page_candidates[: max(top_k * 3, 8)]
        if self.reranker is None:
            return rerank_pool[:top_k]

        try:
            from langchain_core.documents import Document

            docs = [
                Document(
                    page_content=self._page_to_rerank_text(page),
                    metadata={"page_id": page.page_id},
                )
                for page in rerank_pool
            ]
            reranked_docs = self.reranker.compress_documents(docs, query)
            scored_pages: list[PageRetrievalResult] = []
            page_map = {page.page_id: page for page in rerank_pool}
            seen: set[str] = set()
            for doc in reranked_docs:
                page_id = str(doc.metadata.get("page_id", ""))
                page = page_map.get(page_id)
                if page is None:
                    continue
                page.rerank_score = float(doc.metadata.get("relevance_score", page.score))
                page.score = page.rerank_score
                scored_pages.append(page)
                seen.add(page_id)

            for page in rerank_pool:
                if page.page_id not in seen:
                    scored_pages.append(page)
            return scored_pages[:top_k]
        except Exception:
            return rerank_pool[:top_k]

    def _rerank_faq_hits(
        self,
        query: str,
        faq_hits: list[RetrievalItem],
        top_k: int,
    ) -> list[RetrievalItem]:
        """Rerank FAQ hits using Cohere reranker (same as Manual reranking)"""
        if not faq_hits:
            return []

        rerank_pool = faq_hits[: max(top_k * 2, 8)]
        if self.reranker is None:
            logger.warning("Reranker not available, returning FAQ hits without reranking")
            return rerank_pool[:top_k]

        try:
            from langchain_core.documents import Document

            # Convert FAQ hits to documents for reranking
            docs = [
                Document(
                    page_content=item.text,  # FAQ text already contains Q&A
                    metadata={"chunk_id": item.chunk_id},
                )
                for item in rerank_pool
            ]

            # Rerank using Cohere
            reranked_docs = self.reranker.compress_documents(docs, query)

            # Apply rerank scores
            scored_items: list[RetrievalItem] = []
            item_map = {item.chunk_id: item for item in rerank_pool}
            seen: set[str] = set()

            for doc in reranked_docs:
                chunk_id = str(doc.metadata.get("chunk_id", ""))
                item = item_map.get(chunk_id)
                if item is None:
                    continue

                # Update score with rerank score
                rerank_score = float(doc.metadata.get("relevance_score", item.score))
                item.score = rerank_score
                scored_items.append(item)
                seen.add(chunk_id)

            # Add unseen items at the end (with original scores)
            for item in rerank_pool:
                if item.chunk_id not in seen:
                    scored_items.append(item)

            return scored_items[:top_k]

        except Exception as e:
            logger.warning(f"FAQ reranking failed: {e}, returning original order")
            return rerank_pool[:top_k]

    def _page_to_rerank_text(self, page: PageRetrievalResult) -> str:
        snippets = "\n".join(
            f"[근거{idx}] {item.text[:320]}"
            for idx, item in enumerate(page.supporting_items[:3], start=1)
        )
        page_label = page.display_page_label or f"PDF {page.page_no}"
        return (
            f"모델:{page.model or ''}\n"
            f"매뉴얼:{page.manual_type or ''}\n"
            f"페이지:{page_label}\n"
            f"{snippets}"
        ).strip()

    def _page_to_match_text(self, page: PageRetrievalResult) -> str:
        supporting = "\n".join(item.text[:500] for item in page.supporting_items[:3])
        return "\n".join(
            [
                page.source_file or "",
                page.model or "",
                page.manual_type or "",
                page.display_page_label or "",
                supporting,
            ]
        )

    def _filter_pages_by_query_relevance(
        self,
        pages: list[PageRetrievalResult],
        relevance_keywords: list[str],
    ) -> list[PageRetrievalResult]:
        if not pages or not relevance_keywords:
            return pages
        return [
            page
            for page in pages
            if self._text_matches_query_keywords(
                self._page_to_match_text(page),
                relevance_keywords,
            )
        ]

    def _filter_faq_hits_by_query_relevance(
        self,
        faq_hits: list[RetrievalItem],
        relevance_keywords: list[str],
    ) -> list[RetrievalItem]:
        if not faq_hits or not relevance_keywords:
            return faq_hits
        return [
            item
            for item in faq_hits
            if self._text_matches_query_keywords(
                "\n".join([item.text, *(item.relations or [])]),
                relevance_keywords,
            )
        ]

    def _collect_supporting_items(
        self,
        selected_pages: list[PageRetrievalResult],
        faq_hits: list[RetrievalItem],
        top_k: int,
        prefer_faq: bool,
    ) -> list[RetrievalItem]:
        items: list[RetrievalItem] = []
        seen_chunk_ids: set[str] = set()

        def append_item(item: RetrievalItem) -> None:
            if item.chunk_id in seen_chunk_ids:
                return
            items.append(item)
            seen_chunk_ids.add(item.chunk_id)

        faq_hint = prefer_faq
        faq_top = faq_hits[0] if faq_hits else None
        manual_top_score = float(selected_pages[0].score) if selected_pages else 0.0

        # FAQ hint is a soft signal. Keep at least the strongest FAQ candidate in
        # the evidence pool so the workflow can arbitrate between FAQ and manual.
        include_top_faq = bool(
            faq_top
            and (
                faq_hint
                or not selected_pages
                or float(faq_top.score) >= 0.82
                or (
                    float(faq_top.score) >= 0.72
                    and manual_top_score <= 0.45
                )
            )
        )
        if include_top_faq and faq_top is not None:
            append_item(faq_top)

        for page in selected_pages:
            for item in page.supporting_items[:2]:
                append_item(item)
            if len(items) >= top_k:
                break

        if len(items) < top_k:
            for item in faq_hits:
                append_item(item)
                if len(items) >= max(2, top_k):
                    break

        return items[:top_k]

    def _collect_graph_paths(self, chunk_ids: list[str]) -> list[str]:
        # Semantic entity enrichment is disabled in the current ingestion path,
        # so graph path expansion would only generate Neo4j warnings for missing
        # relationship types. Keep the surface stable by returning no paths.
        return []
