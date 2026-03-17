from __future__ import annotations

import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from chevy_troubleshooter.config import Settings
from chevy_troubleshooter.ingest.catalog import DataCatalog
from chevy_troubleshooter.models import EvidenceSource
from chevy_troubleshooter.neo4j_store import Neo4jStore
from chevy_troubleshooter.observability import LangSmithTracer
from chevy_troubleshooter.providers import build_chat_model
from chevy_troubleshooter.retrieval import GuardrailEngine, HybridRetriever


NEGATIVE_FEEDBACK_PATTERNS = [
    re.compile(r"해결.*안|안됨|여전|동일|아니|실패", re.IGNORECASE),
    re.compile(r"still|not fixed|doesn't work", re.IGNORECASE),
]

REWRITE_SYNONYMS = {
    "시동": ["시동불량", "점화", "크랭크"],
    "경고등": ["체크엔진", "MIL", "계기판 경고"],
    "소음": ["잡음", "이상음", "떨림"],
    "브레이크": ["제동", "브레이크패드", "브레이크오일"],
}


class WorkflowState(TypedDict, total=False):
    session_id: str
    user_query: str
    active_query: str
    model_hint: str | None
    feedback: str | None
    resolved: bool | None
    top_k: int
    max_requery: int
    retry_count: int

    history_text: str
    compact_summary: str

    normalized_model: str | None
    model_candidates: list[str]
    fallback_category: str | None
    preferred_manual_types: list[str]
    prefer_faq: bool
    guardrail_allow: bool
    guardrail_reason: str

    retrieval_items: list[dict[str, Any]]
    retrieval_pages: list[dict[str, Any]]
    graph_paths: list[str]
    top_sources: list[dict[str, Any]]
    top_manual_sources: list[dict[str, Any]]
    top_faq_sources: list[dict[str, Any]]
    top_image_path: str | None

    answer: str
    confidence: float
    should_requery: bool
    excluded_chunk_ids: list[str]
    debug: dict[str, Any]


class TroubleshootingWorkflow:
    def __init__(self, settings: Settings, catalog: DataCatalog) -> None:
        self.settings = settings
        self.catalog = catalog

        self.store = Neo4jStore(settings)
        self.retriever = HybridRetriever(settings, self.store)
        self.guardrails = GuardrailEngine(settings, catalog)
        self.chat_model = build_chat_model(settings)
        self.tracer = LangfuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )

        self.graph = self._build_graph()

    def close(self) -> None:
        self.store.close()

    def run(self, payload: dict[str, Any]) -> WorkflowState:
        state: WorkflowState = {
            "session_id": payload["session_id"],
            "user_query": payload["user_query"],
            "active_query": payload["user_query"],
            "model_hint": payload.get("model_hint"),
            "feedback": payload.get("feedback"),
            "resolved": payload.get("resolved"),
            "top_k": int(payload.get("top_k", self.settings.default_top_k)),
            "max_requery": self.settings.max_requery,
            "retry_count": int(payload.get("retry_count", 0)),
            "history_text": payload.get("history_text", ""),
            "compact_summary": payload.get("compact_summary", ""),
            "excluded_chunk_ids": payload.get("excluded_chunk_ids", []),
            "debug": {},
        }

        with self.tracer.trace(
            "chevy-troubleshooting-workflow",
            {
                "session_id": state["session_id"],
                "query": state["user_query"],
            },
        ) as trace:
            result = self.graph.invoke(state)
            self.tracer.event(
                trace,
                "final",
                {
                    "confidence": result.get("confidence", 0.0),
                    "retry_count": result.get("retry_count", 0),
                    "allow": result.get("guardrail_allow", False),
                },
            )
            return result

    def _build_graph(self):
        graph = StateGraph(WorkflowState)

        graph.add_node("compact_context", self._compact_context)
        graph.add_node("guardrail_check", self._guardrail_check)
        graph.add_node("retrieve_hybrid", self._retrieve_hybrid)
        graph.add_node("compose_answer", self._compose_answer)
        graph.add_node("supervisor_review", self._supervisor_review)
        graph.add_node("evaluate_feedback", self._evaluate_feedback)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "compact_context")
        graph.add_edge("compact_context", "guardrail_check")

        graph.add_conditional_edges(
            "guardrail_check",
            self._route_after_guardrail,
            {
                "retrieve_hybrid": "retrieve_hybrid",
                "finalize": "finalize",
            },
        )

        graph.add_edge("retrieve_hybrid", "compose_answer")
        graph.add_edge("compose_answer", "supervisor_review")
        graph.add_edge("supervisor_review", "evaluate_feedback")

        graph.add_conditional_edges(
            "evaluate_feedback",
            self._route_after_feedback,
            {
                "rewrite_query": "rewrite_query",
                "finalize": "finalize",
            },
        )

        graph.add_edge("rewrite_query", "retrieve_hybrid")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _compact_context(self, state: WorkflowState) -> WorkflowState:
        history_text = state.get("history_text", "")
        if len(history_text) <= self.settings.context_compaction_chars:
            return state

        summary = history_text[-self.settings.context_compaction_chars :]
        if self.chat_model is not None:
            prompt = f"""
            Summarize the following conversation history in Korean within 8 lines.
            Keep only the diagnostic-related symptoms, actions taken, and causes of failure.

            {history_text}
            """.strip()
            try:
                resp = self.chat_model.invoke([HumanMessage(content=prompt)])
                content = resp.content if isinstance(resp.content, str) else str(resp.content)
                if content.strip():
                    summary = content.strip()
            except Exception:
                pass

        state["compact_summary"] = summary
        state.setdefault("debug", {})["context_compacted"] = True
        return state

    def _guardrail_check(self, state: WorkflowState) -> WorkflowState:
        decision = self.guardrails.evaluate(
            query=state["active_query"],
            model_hint=state.get("model_hint"),
        )
        state["guardrail_allow"] = decision.allow
        state["guardrail_reason"] = decision.reason
        state["normalized_model"] = decision.normalized_model
        state["model_candidates"] = decision.model_candidates
        state["fallback_category"] = decision.fallback_category
        state["preferred_manual_types"] = decision.preferred_manual_types
        state["prefer_faq"] = decision.prefer_faq
        state.setdefault("debug", {})["guardrail"] = decision.model_dump()

        if not decision.allow:
            state["answer"] = (
                "요청을 처리할 수 없습니다. 쉐보레 차량 진단 및 브랜드 FAQ 질문만 지원합니다. "
                f"사유: {decision.reason}"
            )
            state["confidence"] = 1.0
            state["top_sources"] = []
            state["top_manual_sources"] = []
            state["top_faq_sources"] = []
            state["graph_paths"] = []
            state["top_image_path"] = None
            state["retrieval_pages"] = []

        return state

    def _route_after_guardrail(self, state: WorkflowState) -> str:
        if state.get("guardrail_allow"):
            return "retrieve_hybrid"
        return "finalize"

    def _retrieve_hybrid(self, state: WorkflowState) -> WorkflowState:
        model_candidates = list(state.get("model_candidates", []))
        if not model_candidates and state.get("fallback_category"):
            fallback_model = self._pick_fallback_model(state["fallback_category"])
            model_candidates = self.guardrails.expand_model_candidates(fallback_model)

        items, pages, paths, debug = self.retriever.retrieve(
            query=state["active_query"],
            top_k=state["top_k"],
            model_candidates=model_candidates,
            prefer_faq=bool(state.get("prefer_faq")),
            excluded_chunk_ids=state.get("excluded_chunk_ids", []),
            preferred_manual_types=state.get("preferred_manual_types", []),
        )

        state["retrieval_items"] = [item.model_dump() for item in items]
        state["retrieval_pages"] = [page.model_dump() for page in pages]
        state["graph_paths"] = paths
        state.setdefault("debug", {})["retrieval"] = debug
        state.setdefault("debug", {})["retrieval"]["effective_model_candidates"] = model_candidates

        manual_count = len(state["retrieval_pages"])
        should_broaden = bool(model_candidates) and (
            not state["retrieval_pages"] or manual_count == 0
        )
        if should_broaden:
            had_no_items = not state["retrieval_pages"]
            items, pages, paths, debug2 = self.retriever.retrieve(
                query=state["active_query"],
                top_k=state["top_k"],
                model_candidates=None,
                prefer_faq=bool(state.get("prefer_faq")),
                excluded_chunk_ids=state.get("excluded_chunk_ids", []),
                preferred_manual_types=state.get("preferred_manual_types", []),
            )
            broadened_items = [item.model_dump() for item in items]
            broadened_pages = [page.model_dump() for page in pages]
            broadened_manual_count = len(broadened_pages)

            if broadened_pages and (
                broadened_manual_count > manual_count or not state["retrieval_pages"]
            ):
                state["retrieval_items"] = broadened_items
                state["retrieval_pages"] = broadened_pages
                state["graph_paths"] = paths
            state["debug"]["retrieval_fallback_broadened"] = debug2
            state["debug"]["retrieval_fallback_reason"] = (
                "no_evidence_with_model_candidates"
                if had_no_items
                else "faq_only_with_model_candidates"
            )

        return state

    def _compose_answer(self, state: WorkflowState) -> WorkflowState:
        items = state.get("retrieval_items", [])
        pages = state.get("retrieval_pages", [])
        prefer_faq = bool(state.get("prefer_faq"))
        if not items and not pages:
            state["answer"] = (
                "관련 FAQ 근거를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."
                if prefer_faq
                else "관련 매뉴얼 근거를 찾지 못했습니다. 증상을 더 구체적으로 알려주세요."
            )
            state["confidence"] = 0.2
            state["top_sources"] = []
            state["top_manual_sources"] = []
            state["top_faq_sources"] = []
            state["top_image_path"] = None
            return state

        # --- Score threshold filtering: score가 0인 항목만 제외 ---
        qualified_items = [item for item in items if float(item["score"]) > 0]

        if not qualified_items:
            state["answer"] = (
                "관련 FAQ를 검색했으나 충분히 신뢰할 수 있는 근거를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."
                if prefer_faq
                else "관련 매뉴얼을 검색했으나 충분히 신뢰할 수 있는 근거를 찾지 못했습니다. 증상을 더 구체적으로 설명해주시면 더 정확한 답변을 드릴 수 있습니다."
            )
            state["confidence"] = 0.2
            state["top_sources"] = []
            state["top_manual_sources"] = []
            state["top_faq_sources"] = []
            state["top_image_path"] = None
            return state

        manual_pages = [page for page in pages if float(page.get("score", 0.0)) > 0]
        manual_items = [
            item for item in qualified_items if str(item.get("source_type", "manual")).lower() == "manual"
        ]
        faq_items = [
            item for item in qualified_items if str(item.get("source_type", "manual")).lower() == "faq"
        ]
        prioritized_items = faq_items + manual_items if prefer_faq else manual_items + faq_items

        if prefer_faq and faq_items:
            top_score_basis = faq_items[0]
        else:
            top_score_basis = manual_pages[0] if manual_pages else (manual_items[0] if manual_items else prioritized_items[0])
        top_score = float(top_score_basis["score"])

        top_manual_sources = []
        if not (prefer_faq and faq_items):
            top_manual_sources = [
                EvidenceSource(
                    source_file=page["source_file"],
                    page_no=page["page_no"],
                    chunk_id=(page.get("supporting_items") or [{}])[0].get("chunk_id", ""),
                    score=float(page["score"]),
                    path_summary=page.get("path_summary", ""),
                    source_type="manual",
                    page_id=page.get("page_id"),
                    display_page_label=page.get("display_page_label"),
                    manual_type=page.get("manual_type"),
                    model=page.get("model"),
                ).model_dump()
                for page in manual_pages[:7]
            ]
        top_faq_sources = [
            EvidenceSource(
                source_file=item["source_file"],
                page_no=item["page_no"],
                chunk_id=item["chunk_id"],
                score=float(item["score"]),
                path_summary=(item.get("relations") or [""])[0],
                source_type="faq",
                page_id=item.get("page_id"),
                display_page_label=item.get("display_page_label"),
                manual_type=item.get("manual_type"),
                model=item.get("model"),
            ).model_dump()
            for item in faq_items[:7]
        ]

        top_image = None if prefer_faq else next((page.get("image_path") for page in manual_pages if page.get("image_path")), None)
        if top_image is None:
            top_image = next((item.get("image_path") for item in prioritized_items if item.get("image_path")), None)
        confidence = max(0.25, min(0.95, top_score))

        # --- Confidence hint (답변은 항상 생성, 낮은 score에는 힌트만 추가) ---
        if top_score >= 0.50:
            confidence_hint = ""
        else:
            confidence_hint = "[참고: 검색된 근거의 관련도가 높지 않을 수 있습니다. 가능한 범위 내에서 답변합니다.]\n\n"

        evidence_text = confidence_hint + "\n\n".join(
            [
                f"[근거{i+1}] {item['text'][:500]}"
                for i, item in enumerate(prioritized_items[:5])
            ]
        )
        path_text = "\n".join(state.get("graph_paths", [])[:8])

        answer = self._generate_answer_text(
            query=state["active_query"],
            evidence_text=evidence_text,
            path_text=path_text,
            fallback_category=state.get("fallback_category"),
            prefer_faq=prefer_faq,
        )

        state["answer"] = answer
        state["confidence"] = confidence
        state["top_manual_sources"] = top_manual_sources
        state["top_faq_sources"] = top_faq_sources
        # Legacy output: top_sources is manual-first for backward compatibility.
        state["top_sources"] = top_faq_sources if (prefer_faq and top_faq_sources) else (top_manual_sources if top_manual_sources else top_faq_sources)
        state["top_image_path"] = top_image
        return state

    def _supervisor_review(self, state: WorkflowState) -> WorkflowState:
        """Supervisor agent: 생성된 답변을 사용자 질문 의도, 근거 충실도, 논리 구조 관점에서 검증하고 개선합니다."""
        answer = state.get("answer", "")
        query = state.get("active_query", "")

        # 답변이 없거나 confidence가 이미 매우 낮으면 스킵
        if not answer or float(state.get("confidence", 0)) < 0.25:
            state.setdefault("debug", {})["supervisor"] = {"action": "skipped", "reason": "no_answer_or_low_confidence"}
            return state

        if self.chat_model is None:
            state.setdefault("debug", {})["supervisor"] = {"action": "skipped", "reason": "no_chat_model"}
            return state

        # 근거 텍스트 재구성 (supervisor에게 원본 근거 제공)
        items = state.get("retrieval_items", [])
        evidence_summary = "\n".join(
            f"[근거{i+1}] {item['text'][:300]}" for i, item in enumerate(items[:5])
        )

        supervisor_prompt = f"""You are the Quality Supervisor for a Chevrolet vehicle maintenance diagnostic system.
        Review the [User Query], [Retrieved Evidence], and [Generated Answer] below, and evaluate the quality of the answer.

        [User Query]
        {query}

        [Retrieved Evidence]
        {evidence_summary}

        [Generated Answer]
        {answer}

        ---
        Evaluate based on the following criteria:
        1. Does the answer perfectly align with the intent of the user's query?
        2. Is the answer free of information not found in the evidence? (No hallucination)
        3. Are the inspection procedures presented in a logical sequence?
        4. Is the answer free of unnecessary verbosity or redundant content?
        5. Are all vehicle safety-related precautions appropriately included?

        Output Format:
        Line 1: Decision: PASS or REVISE
        Line 2: Provide a 1-sentence reason for your decision.
        If PASS: Do not write anything after the second line.
        If REVISE: Insert a "---" separator on the third line, and write the completely revised answer in **Korean** starting from the fourth line.
        """.strip()

        try:
            resp = self.chat_model.invoke([HumanMessage(content=supervisor_prompt)])
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            lines = content.strip().split("\n", 2)

            verdict = lines[0].strip().upper() if lines else "PASS"
            reason = lines[1].strip() if len(lines) > 1 else ""

            if "REVISE" in verdict and len(lines) > 2:
                # "---" 구분선 이후의 내용이 수정된 답변
                revised_body = lines[2].strip()
                if revised_body.startswith("---"):
                    revised_body = revised_body[3:].strip()
                if revised_body:
                    state["answer"] = revised_body
                    state.setdefault("debug", {})["supervisor"] = {
                        "action": "revised",
                        "reason": reason,
                        "original_answer_len": len(answer),
                        "revised_answer_len": len(revised_body),
                    }
                else:
                    state.setdefault("debug", {})["supervisor"] = {
                        "action": "pass_fallback",
                        "reason": "REVISE but no revised content",
                    }
            else:
                state.setdefault("debug", {})["supervisor"] = {
                    "action": "passed",
                    "reason": reason,
                }
        except Exception as exc:
            state.setdefault("debug", {})["supervisor"] = {
                "action": "error",
                "reason": str(exc)[:200],
            }

        return state

    def _evaluate_feedback(self, state: WorkflowState) -> WorkflowState:
        resolved = state.get("resolved")
        feedback = state.get("feedback") or ""
        confidence = float(state.get("confidence", 0.0))

        if resolved is True:
            state["should_requery"] = False
            return state

        negative_feedback = any(p.search(feedback) for p in NEGATIVE_FEEDBACK_PATTERNS)
        low_conf = confidence < 0.45
        no_evidence = not state.get("retrieval_items")

        should_requery = (resolved is False and negative_feedback) or no_evidence or low_conf
        reached_limit = int(state.get("retry_count", 0)) >= int(state.get("max_requery", 0))

        state["should_requery"] = bool(should_requery and not reached_limit)
        state.setdefault("debug", {})["feedback_eval"] = {
            "negative_feedback": negative_feedback,
            "low_confidence": low_conf,
            "no_evidence": no_evidence,
            "reached_limit": reached_limit,
        }
        return state

    def _route_after_feedback(self, state: WorkflowState) -> str:
        if state.get("should_requery"):
            return "rewrite_query"
        return "finalize"

    def _rewrite_query(self, state: WorkflowState) -> WorkflowState:
        original = state["active_query"]
        feedback = state.get("feedback") or ""
        terms = self._extract_terms(f"{original} {feedback}")

        expanded_terms: list[str] = []
        for term in terms:
            expanded_terms.append(term)
            for key, synonyms in REWRITE_SYNONYMS.items():
                if key in term:
                    expanded_terms.extend(synonyms)

        fallback_text = ""
        if state.get("fallback_category"):
            fallback_text = f" 동급 {state['fallback_category']} 차종 근거 우선 검색"

        rewritten = (
            f"{original} | 재질의 컨텍스트: {feedback} | "
            f"핵심키워드: {', '.join(sorted(set(expanded_terms))[:12])}.{fallback_text}"
        ).strip()

        previous_top_chunks = [
            str(item.get("chunk_id", ""))
            for item in state.get("retrieval_items", [])[:3]
            if item.get("chunk_id")
        ]
        excluded = list(set(state.get("excluded_chunk_ids", []) + previous_top_chunks))

        state["active_query"] = rewritten
        state["retry_count"] = int(state.get("retry_count", 0)) + 1
        state["top_k"] = min(10, int(state.get("top_k", 5)) + 2)
        state["excluded_chunk_ids"] = excluded
        state.setdefault("debug", {})["requery"] = {
            "rewritten_query": rewritten,
            "retry_count": state["retry_count"],
            "top_k": state["top_k"],
            "excluded_chunk_ids": excluded,
            "strategy": {
                "expand_keywords": True,
                "increase_top_k": True,
                "graph_depth": 2 + state["retry_count"],
                "penalize_previous_paths": True,
            },
        }
        return state

    def _finalize(self, state: WorkflowState) -> WorkflowState:
        state.setdefault("top_manual_sources", [])
        state["top_manual_sources"] = state["top_manual_sources"][:5]
        state.setdefault("top_faq_sources", [])
        state["top_faq_sources"] = state["top_faq_sources"][:5]
        state.setdefault("top_sources", [])
        state["top_sources"] = state["top_manual_sources"][:5]
        state.setdefault("graph_paths", [])
        state["graph_paths"] = state["graph_paths"][:10]
        state.setdefault("confidence", 0.0)
        return state

    def _pick_fallback_model(self, category: str | None) -> str | None:
        if not category:
            return None
        for model, cat in self.catalog.model_to_category.items():
            if cat == category:
                return model
        return None

    def _extract_terms(self, text: str) -> list[str]:
        terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
        return sorted(set(terms), key=len, reverse=True)

    def _generate_answer_text(
        self,
        query: str,
        evidence_text: str,
        path_text: str,
        fallback_category: str | None,
        prefer_faq: bool,
    ) -> str:
        if self.chat_model is None:
            fallback_note = (
                f"\n참고: 요청 모델 데이터가 없어서 {fallback_category} 카테고리로 대체 검색했습니다."
                if fallback_category
                else ""
            )
            if prefer_faq:
                return (
                    "다음은 검색된 FAQ 근거를 바탕으로 정리한 답변입니다.\n\n"
                    f"{evidence_text[:1200]}"
                    f"{fallback_note}"
                )
            return (
                "다음 순서로 점검하세요.\n"
                "1) 경고등/계기판 상태 확인\n"
                "2) 관련 부품 기본 점검\n"
                "3) 정비 항목 순차 수행\n\n"
                f"근거 요약:\n{evidence_text[:1200]}"
                f"\n\n그래프 경로:\n{path_text[:800]}"
                f"{fallback_note}"
            )

        if prefer_faq:
            prompt = f"""
            You are a Chevrolet Customer Support FAQ Assistant.
            You must answer exclusively in Korean and provide a concise explanation based solely on the provided FAQ evidence.

            User Query:
            {query}

            Evidence Text:
            {evidence_text}

            Writing Rules:
            1) Answer the user's query directly.
            2) Summarize procedures using short, numbered lists only when necessary.
            3) Do not omit any benefits, conditions, limits, or exceptions.
            4) Do not assume or guess information not found in the evidence.
            5) Use only Korean in your response.
            """.strip()
        else:
            prompt = f"""
            You are a Chevrolet Vehicle Maintenance Diagnostic Assistant.
            You must answer exclusively in Korean and base your response solely on the provided evidence.

            User Query:
            {query}

            Evidence Text:
            {evidence_text}

            Graph Multi-hop Path:
            {path_text}

            Writing Rules:
            1) Present the inspection procedures in order of priority.
            2) Provide a 1-sentence reason next to each procedure.
            3) At the end, add a "Summary of Evidence Path" (근거 경로 요약) in 3 lines or less.
            4) State clearly if you do not know the answer or if it is not in the evidence.
            5) Use only Korean.
            """.strip()

        try:
            resp = self.chat_model.invoke([HumanMessage(content=prompt)])
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            if fallback_category:
                content = (
                    f"{content}\n\n참고: 요청 모델 데이터가 명확하지 않아 "
                    f"쉐보레 {fallback_category} 카테고리 근거를 함께 사용했습니다."
                )
            return content
        except Exception:
            return (
                "근거를 바탕으로 점검 절차를 생성하지 못했습니다. 입력 증상을 더 자세히 제공해 주세요."
            )
