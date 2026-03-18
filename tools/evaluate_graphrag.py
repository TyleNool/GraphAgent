#!/usr/bin/env python3
"""
GraphRAG 5-Category Evaluation System
======================================
5개 평가 카테고리, 15개 세부 지표로 GraphRAG 파이프라인을 종합 평가합니다.

1. Routing & Guardrail
   - Guardrail Accuracy
   - Routing Accuracy (FAQ/Manual source selection)
   - Entity/Intent Resolution (model normalization)

2. Graph & Document Retrieval
   - Document Retrieval Recall (source_file hit@k)
   - Graph Completeness (entity hit, multi-hop completeness)
   - Hierarchy Alignment (page hit@k, manual_type match)

3. Generation & Grounding
   - Faithfulness / Groundedness
   - Answer Relevancy (LLM-judged)
   - Fact Coverage (expected_facts recall)

4. Multimodal & UX Alignment
   - Image-Source Alignment
   - Confidence Calibration

5. Operational Metrics
   - Latency (p50, p95)
   - Cost per Query (token estimate)
   - Stability & Requery Rate

Usage:
    # 1단계: 파이프라인 실행 + 원시 결과 수집
    python tools/evaluate_graphrag.py run \
        --dataset Comprehensive_GraphRAG_Evaluation_Dataset_300.json \
        --output eval_results_raw.json \
        --top-k 5

    # 2단계: 수집된 결과로 지표 계산 + 리포트 생성
    python tools/evaluate_graphrag.py report \
        --results eval_results_raw.json \
        --output eval_report.json

    # (선택) 1+2 한번에 실행
    python tools/evaluate_graphrag.py run-and-report \
        --dataset Comprehensive_GraphRAG_Evaluation_Dataset_300.json \
        --output eval_report.json \
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 데이터 모델
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvalItem:
    """평가 데이터셋의 개별 항목."""
    id: str
    category: str
    difficulty: str
    question: str
    ground_truth: dict[str, Any]
    eval_stage_focus: list[str]
    notes: str = ""


@dataclass
class RunResult:
    """파이프라인 실행 결과."""
    item_id: str
    category: str
    difficulty: str
    question: str
    ground_truth: dict[str, Any]

    # 워크플로우 출력
    guardrail_allow: bool = False
    guardrail_reason: str = ""
    normalized_model: str | None = None
    model_candidates: list[str] = field(default_factory=list)
    prefer_faq: bool = False
    faq_priority: bool = False

    answer: str = ""
    confidence: float = 0.0
    top_image_path: str | None = None
    top_manual_sources: list[dict] = field(default_factory=list)
    top_faq_sources: list[dict] = field(default_factory=list)
    retrieval_pages: list[dict] = field(default_factory=list)
    retrieval_items: list[dict] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)

    # 운영 지표
    latency_sec: float = 0.0
    retry_count: int = 0
    should_requery: bool = False

    # 디버그
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "difficulty": self.difficulty,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "guardrail_allow": self.guardrail_allow,
            "guardrail_reason": self.guardrail_reason,
            "normalized_model": self.normalized_model,
            "model_candidates": self.model_candidates,
            "prefer_faq": self.prefer_faq,
            "faq_priority": self.faq_priority,
            "answer": self.answer,
            "confidence": self.confidence,
            "top_image_path": self.top_image_path,
            "top_manual_sources": self.top_manual_sources,
            "top_faq_sources": self.top_faq_sources,
            "retrieval_pages": self.retrieval_pages,
            "retrieval_items": self.retrieval_items,
            "graph_paths": self.graph_paths,
            "latency_sec": self.latency_sec,
            "retry_count": self.retry_count,
            "should_requery": self.should_requery,
            "debug": self.debug,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════
# 1단계: 파이프라인 실행 (Run)
# ═══════════════════════════════════════════════════════════════

def load_dataset(path: Path) -> list[EvalItem]:
    """평가 데이터셋 로드."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = raw.get("dataset", raw if isinstance(raw, list) else [])
    items = []
    for d in dataset:
        items.append(EvalItem(
            id=d["id"],
            category=d["category"],
            difficulty=d.get("difficulty", "basic"),
            question=d["question"],
            ground_truth=d.get("ground_truth", {}),
            eval_stage_focus=d.get("eval_stage_focus", []),
            notes=d.get("notes", ""),
        ))
    return items


def run_pipeline(dataset_path: Path, output_path: Path, top_k: int = 5,
                 max_items: int | None = None, categories: list[str] | None = None) -> list[RunResult]:
    """데이터셋의 각 항목에 대해 워크플로우를 실행하고 결과를 수집합니다."""
    from Chevolet_GraphRAG.agent import TroubleshootingWorkflow
    from Chevolet_GraphRAG.config import get_settings
    from Chevolet_GraphRAG.ingest import discover_manual_files

    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)
    workflow = TroubleshootingWorkflow(settings=settings, catalog=catalog)

    items = load_dataset(dataset_path)
    if categories:
        cat_set = set(categories)
        items = [it for it in items if it.category in cat_set]
    if max_items:
        items = items[:max_items]

    results: list[RunResult] = []
    total = len(items)

    try:
        for idx, item in enumerate(items, 1):
            logger.info(f"[{idx}/{total}] {item.id} ({item.category}) — {item.question[:60]}...")

            # 모델 힌트 추출
            gt = item.ground_truth
            model_hint = None
            guardrail_gt = gt.get("guardrail", {})
            candidates = guardrail_gt.get("expected_model_candidates", [])
            if candidates:
                model_hint = candidates[0]

            t0 = time.perf_counter()
            try:
                state = workflow.run({
                    "session_id": f"eval-{item.id}-{uuid.uuid4().hex[:6]}",
                    "user_query": item.question,
                    "model_hint": model_hint,
                    "top_k": top_k,
                    "history_text": "",
                    "feedback": None,
                    "resolved": None,
                })
            except Exception as exc:
                logger.error(f"  ERROR: {exc}")
                state = {"answer": f"[ERROR] {exc}", "confidence": 0.0}
            elapsed = time.perf_counter() - t0

            rr = RunResult(
                item_id=item.id,
                category=item.category,
                difficulty=item.difficulty,
                question=item.question,
                ground_truth=gt,
                guardrail_allow=bool(state.get("guardrail_allow", False)),
                guardrail_reason=str(state.get("guardrail_reason", "")),
                normalized_model=state.get("normalized_model"),
                model_candidates=state.get("model_candidates", []),
                prefer_faq=bool(state.get("prefer_faq", False)),
                faq_priority=bool(state.get("faq_priority", False)),
                answer=str(state.get("answer", "")),
                confidence=float(state.get("confidence", 0.0)),
                top_image_path=state.get("top_image_path"),
                top_manual_sources=state.get("top_manual_sources", []),
                top_faq_sources=state.get("top_faq_sources", []),
                retrieval_pages=state.get("retrieval_pages", []),
                retrieval_items=state.get("retrieval_items", []),
                graph_paths=state.get("graph_paths", []),
                latency_sec=elapsed,
                retry_count=int(state.get("retry_count", 0)),
                should_requery=bool(state.get("should_requery", False)),
                debug=state.get("debug", {}),
            )
            results.append(rr)

            logger.info(f"  → allow={rr.guardrail_allow}, conf={rr.confidence:.3f}, "
                        f"pages={len(rr.retrieval_pages)}, lat={rr.latency_sec:.2f}s")

            # 중간 저장 (매 10건)
            if idx % 10 == 0:
                _save_results(results, output_path)

    finally:
        workflow.close()
        _save_results(results, output_path)

    return results


def _save_results(results: list[RunResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"  중간 저장: {path} ({len(results)}건)")


# ═══════════════════════════════════════════════════════════════
# 2단계: 지표 계산 (Report)
# ═══════════════════════════════════════════════════════════════

def load_results(path: Path) -> list[RunResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [RunResult.from_dict(r) for r in raw.get("results", [])]


# ───────────────────────────────────────
# Category 1: Routing & Guardrail
# ───────────────────────────────────────

def eval_guardrail_accuracy(results: list[RunResult]) -> dict:
    """가드레일 allow/reject 정확도."""
    tp = fp = tn = fn = 0
    details = []

    for r in results:
        gt = r.ground_truth.get("guardrail", {})
        expected_allow = gt.get("expected_allow")
        if expected_allow is None:
            continue

        actual_allow = r.guardrail_allow
        if expected_allow and actual_allow:
            tp += 1
        elif expected_allow and not actual_allow:
            fn += 1
            details.append({"id": r.item_id, "type": "false_negative",
                            "question": r.question[:80], "reason": r.guardrail_reason})
        elif not expected_allow and not actual_allow:
            tn += 1
        else:
            fp += 1
            details.append({"id": r.item_id, "type": "false_positive",
                            "question": r.question[:80], "reason": r.guardrail_reason})

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_evaluated": total,
        "errors": details[:20],
    }


def eval_routing_accuracy(results: list[RunResult]) -> dict:
    """FAQ/Manual 소스 선택 정확도."""
    correct = 0
    total = 0
    details = []

    for r in results:
        gt = r.ground_truth.get("guardrail", {})
        expected_family = gt.get("expected_source_family")
        if not expected_family or not r.guardrail_allow:
            continue

        # 실제 소스 판별
        has_manual = len(r.top_manual_sources) > 0
        has_faq = len(r.top_faq_sources) > 0
        actual_priority = "faq" if r.faq_priority else "manual"

        if expected_family == "manual":
            match = actual_priority == "manual" or (has_manual and not r.faq_priority)
        elif expected_family == "faq":
            match = actual_priority == "faq" or (has_faq and r.faq_priority)
        elif expected_family == "mixed":
            match = has_manual and has_faq
        else:
            continue

        total += 1
        if match:
            correct += 1
        else:
            details.append({
                "id": r.item_id,
                "expected": expected_family,
                "actual_priority": actual_priority,
                "has_manual": has_manual,
                "has_faq": has_faq,
                "question": r.question[:80],
            })

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "errors": details[:20],
    }


def eval_entity_resolution(results: list[RunResult]) -> dict:
    """모델명 정규화 + 후보 확장 정확도."""
    model_correct = 0
    model_total = 0
    candidate_hit = 0
    candidate_total = 0
    details = []

    for r in results:
        gt = r.ground_truth.get("guardrail", {})
        expected_family = gt.get("expected_model_family")
        expected_candidates = gt.get("expected_model_candidates", [])

        if not r.guardrail_allow:
            continue

        # 모델 패밀리 매칭
        if expected_family:
            model_total += 1
            actual_model = (r.normalized_model or "").lower().replace("_", "").replace(" ", "")
            expected_clean = expected_family.lower().replace("_", "").replace(" ", "")
            if expected_clean in actual_model or actual_model in expected_clean:
                model_correct += 1
            else:
                # 후보 목록에서도 확인
                actual_candidates_lower = [c.lower().replace("_", "").replace(" ", "") for c in r.model_candidates]
                if any(expected_clean in c or c in expected_clean for c in actual_candidates_lower):
                    model_correct += 1
                else:
                    details.append({
                        "id": r.item_id,
                        "expected_family": expected_family,
                        "actual_model": r.normalized_model,
                        "actual_candidates": r.model_candidates,
                        "question": r.question[:80],
                    })

        # 후보 목록 hit
        if expected_candidates:
            candidate_total += 1
            actual_set = set(c.lower() for c in r.model_candidates)
            if any(ec.lower() in actual_set for ec in expected_candidates):
                candidate_hit += 1

    return {
        "model_family_accuracy": round(model_correct / model_total, 4) if model_total else 0.0,
        "model_family_total": model_total,
        "candidate_hit_rate": round(candidate_hit / candidate_total, 4) if candidate_total else 0.0,
        "candidate_total": candidate_total,
        "errors": details[:20],
    }


# ───────────────────────────────────────
# Category 2: Graph & Document Retrieval
# ───────────────────────────────────────

def eval_document_retrieval_recall(results: list[RunResult], k: int = 5) -> dict:
    """source_file hit@k — 올바른 소스 파일을 검색했는지."""
    hit = 0
    total = 0
    mrr_scores = []
    details = []

    for r in results:
        gt = r.ground_truth.get("retrieval", {})
        expected_files = gt.get("expected_source_files", [])
        if not expected_files or not r.guardrail_allow:
            continue

        total += 1
        # 실제 검색된 파일 목록 (manual + faq)
        actual_files = []
        for src in r.top_manual_sources[:k]:
            actual_files.append(src.get("source_file", ""))
        for src in r.top_faq_sources[:k]:
            actual_files.append(src.get("source_file", ""))
        # retrieval_pages 에서도
        for page in r.retrieval_pages[:k]:
            actual_files.append(page.get("source_file", ""))

        # 파일 이름 정규화 후 비교
        actual_normalized = [_norm_filename(f) for f in actual_files if f]
        expected_normalized = [_norm_filename(f) for f in expected_files]

        found = False
        first_rank = None
        for ei, exp in enumerate(expected_normalized):
            for ai, act in enumerate(actual_normalized):
                if exp in act or act in exp:
                    found = True
                    if first_rank is None:
                        first_rank = ai + 1
                    break
            if found:
                break

        if found:
            hit += 1
            mrr_scores.append(1.0 / first_rank if first_rank else 0.0)
        else:
            mrr_scores.append(0.0)
            details.append({
                "id": r.item_id,
                "expected": expected_files,
                "actual_top5": actual_files[:5],
                "question": r.question[:80],
            })

    return {
        f"hit_at_{k}": round(hit / total, 4) if total else 0.0,
        "mrr": round(mean(mrr_scores), 4) if mrr_scores else 0.0,
        "total": total,
        "errors": details[:20],
    }


def eval_graph_completeness(results: list[RunResult]) -> dict:
    """GraphRAG 고유 지표 — entity hit, multi-hop completeness."""
    entity_hit = 0
    entity_total = 0
    multihop_complete = 0
    multihop_total = 0
    entity_recall_scores = []

    for r in results:
        gt = r.ground_truth.get("graphrag", {})
        expected_entities = gt.get("expected_entities", [])
        expected_path_completeness = gt.get("expected_path_completeness", False)

        if not r.guardrail_allow:
            continue

        # Entity hit: 답변 또는 검색 결과에 엔터티가 등장하는지
        if expected_entities:
            entity_total += 1
            # 답변 + 검색 텍스트에서 확인
            search_corpus = r.answer.lower()
            for item in r.retrieval_items[:10]:
                search_corpus += " " + item.get("text", "").lower()

            found_count = sum(1 for ent in expected_entities if ent.lower() in search_corpus)
            if found_count > 0:
                entity_hit += 1
            entity_recall_scores.append(found_count / len(expected_entities))

        # Multi-hop completeness
        if expected_path_completeness:
            multihop_total += 1
            # 그래프 경로가 존재하고 답변에 관련 정보가 포함되는지
            has_paths = len(r.graph_paths) > 0
            has_multi_source = len(set(
                item.get("source_file", "") for item in r.retrieval_items[:10]
            )) > 1
            if has_paths or has_multi_source:
                multihop_complete += 1

    return {
        "entity_hit_rate": round(entity_hit / entity_total, 4) if entity_total else 0.0,
        "entity_recall_avg": round(mean(entity_recall_scores), 4) if entity_recall_scores else 0.0,
        "entity_total": entity_total,
        "multihop_completeness": round(multihop_complete / multihop_total, 4) if multihop_total else 0.0,
        "multihop_total": multihop_total,
    }


def eval_hierarchy_alignment(results: list[RunResult], k: int = 5) -> dict:
    """page hit@k + manual_type 일치율."""
    page_hit = 0
    page_total = 0
    type_match = 0
    type_total = 0
    should_not_manual_violations = 0
    should_not_manual_total = 0

    for r in results:
        gt = r.ground_truth.get("retrieval", {})
        expected_pages = gt.get("expected_pages", [])
        expected_types = gt.get("expected_manual_types", [])
        should_not_manual = gt.get("should_not_return_manual", False)

        if not r.guardrail_allow:
            continue

        # Page hit@k
        if expected_pages:
            page_total += 1
            actual_pages = [p.get("page_no", -1) for p in r.retrieval_pages[:k]]
            if any(ep in actual_pages for ep in expected_pages):
                page_hit += 1

        # Manual type match
        if expected_types:
            type_total += 1
            actual_types = set()
            for page in r.retrieval_pages[:k]:
                mt = page.get("manual_type")
                if mt:
                    actual_types.add(mt.lower())
            for src in r.top_manual_sources[:k]:
                mt = src.get("manual_type")
                if mt:
                    actual_types.add(mt.lower())
            expected_lower = set(t.lower() for t in expected_types)
            if actual_types & expected_lower:
                type_match += 1

        # should_not_return_manual 위반 체크
        if should_not_manual:
            should_not_manual_total += 1
            if r.top_manual_sources and not r.faq_priority:
                should_not_manual_violations += 1

    return {
        f"page_hit_at_{k}": round(page_hit / page_total, 4) if page_total else 0.0,
        "page_total": page_total,
        "manual_type_match": round(type_match / type_total, 4) if type_total else 0.0,
        "type_total": type_total,
        "should_not_manual_violation_rate": round(
            should_not_manual_violations / should_not_manual_total, 4
        ) if should_not_manual_total else 0.0,
        "should_not_manual_total": should_not_manual_total,
    }


# ───────────────────────────────────────
# Category 3: Generation & Grounding
# ───────────────────────────────────────

def eval_faithfulness(results: list[RunResult]) -> dict:
    """근거 충실도 — prohibited_facts 미포함 여부 + groundedness pass율."""
    groundedness_pass = 0
    groundedness_total = 0
    hallucination_count = 0
    hallucination_total = 0
    violation_details = []

    for r in results:
        gt_answer = r.ground_truth.get("answer", {})
        gt_quality = r.ground_truth.get("quality", {})
        prohibited = gt_answer.get("prohibited_facts", [])
        expected_grounded = gt_quality.get("expected_groundedness_pass")

        if not r.guardrail_allow or not r.answer:
            continue

        # Groundedness pass 체크
        if expected_grounded is not None:
            groundedness_total += 1
            # prohibited facts가 답변에 없으면 pass
            has_violation = False
            found_violations = []
            for pf in prohibited:
                if pf and pf.lower() in r.answer.lower():
                    has_violation = True
                    found_violations.append(pf)

            if not has_violation and expected_grounded:
                groundedness_pass += 1
            elif has_violation and not expected_grounded:
                groundedness_pass += 1  # 예상대로 위반 발생
            elif has_violation:
                violation_details.append({
                    "id": r.item_id,
                    "violations": found_violations,
                    "question": r.question[:80],
                })

        # 할루시네이션 체크 (prohibited_facts 기반)
        if prohibited:
            hallucination_total += 1
            found_any = any(pf.lower() in r.answer.lower() for pf in prohibited if pf)
            if found_any:
                hallucination_count += 1

    return {
        "groundedness_pass_rate": round(groundedness_pass / groundedness_total, 4) if groundedness_total else 0.0,
        "groundedness_total": groundedness_total,
        "hallucination_rate": round(hallucination_count / hallucination_total, 4) if hallucination_total else 0.0,
        "hallucination_total": hallucination_total,
        "violations": violation_details[:20],
    }


def eval_answer_relevancy(results: list[RunResult], use_llm: bool = False) -> dict:
    """답변 관련성 — expected_facts 커버리지 + 선택적 LLM 판정.

    use_llm=True일 경우 LLM을 사용하여 답변의 질문 부합도를 0~1로 평가합니다.
    기본적으로는 규칙 기반(expected_facts 커버리지)만 사용합니다.
    """
    fact_coverage_scores = []
    total = 0
    llm_scores = []
    chat_model = None

    if use_llm:
        try:
            from Chevolet_GraphRAG.config import get_settings
            from Chevolet_GraphRAG.providers import build_chat_model
            settings = get_settings()
            chat_model = build_chat_model(settings)
        except Exception:
            logger.warning("LLM 로드 실패, 규칙 기반 평가만 수행합니다.")

    for r in results:
        gt_answer = r.ground_truth.get("answer", {})
        expected_facts = gt_answer.get("expected_facts", [])

        if not r.guardrail_allow or not r.answer:
            continue

        # Expected facts coverage
        if expected_facts:
            total += 1
            found = sum(1 for f in expected_facts if f and f.lower() in r.answer.lower())
            coverage = found / len(expected_facts)
            fact_coverage_scores.append(coverage)

        # LLM 기반 관련성 평가
        if chat_model and r.answer and r.question:
            try:
                score = _llm_judge_relevancy(chat_model, r.question, r.answer)
                llm_scores.append(score)
            except Exception:
                pass

    result = {
        "fact_coverage_avg": round(mean(fact_coverage_scores), 4) if fact_coverage_scores else 0.0,
        "fact_coverage_total": total,
    }
    if llm_scores:
        result["llm_relevancy_avg"] = round(mean(llm_scores), 4)
        result["llm_relevancy_total"] = len(llm_scores)

    return result


def _llm_judge_relevancy(chat_model, question: str, answer: str) -> float:
    """LLM으로 답변 관련성을 0~1 스코어로 평가."""
    from langchain_core.messages import HumanMessage

    prompt = f"""You are evaluating the relevancy of an answer to a question about Chevrolet vehicle maintenance.
Rate on a scale of 0.0 to 1.0 how well the answer addresses the question.

Question: {question}
Answer: {answer[:1000]}

Output ONLY a single float number between 0.0 and 1.0. Nothing else."""

    resp = chat_model.invoke([HumanMessage(content=prompt)])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    # 숫자 추출
    match = re.search(r"(\d+\.?\d*)", content.strip())
    if match:
        return min(1.0, max(0.0, float(match.group(1))))
    return 0.5


def eval_fact_coverage(results: list[RunResult]) -> dict:
    """개별 expected_fact별 커버리지 상세 분석."""
    per_category = defaultdict(lambda: {"covered": 0, "total": 0, "missing_facts": []})

    for r in results:
        gt_answer = r.ground_truth.get("answer", {})
        expected_facts = gt_answer.get("expected_facts", [])

        if not r.guardrail_allow or not expected_facts:
            continue

        cat = r.category
        for fact in expected_facts:
            if not fact:
                continue
            per_category[cat]["total"] += 1
            if fact.lower() in r.answer.lower():
                per_category[cat]["covered"] += 1
            else:
                per_category[cat]["missing_facts"].append({
                    "id": r.item_id, "fact": fact
                })

    summary = {}
    for cat, data in per_category.items():
        summary[cat] = {
            "coverage": round(data["covered"] / data["total"], 4) if data["total"] else 0.0,
            "covered": data["covered"],
            "total": data["total"],
            "sample_missing": data["missing_facts"][:5],
        }

    # 전체 통계
    all_covered = sum(d["covered"] for d in per_category.values())
    all_total = sum(d["total"] for d in per_category.values())

    return {
        "overall_coverage": round(all_covered / all_total, 4) if all_total else 0.0,
        "overall_covered": all_covered,
        "overall_total": all_total,
        "per_category": summary,
    }


# ───────────────────────────────────────
# Category 4: Multimodal & UX Alignment
# ───────────────────────────────────────

def eval_image_source_alignment(results: list[RunResult]) -> dict:
    """이미지-소스 정합성 — top_image가 top_manual_source의 페이지와 일치하는지."""
    aligned = 0
    total = 0
    image_page_match = 0
    image_page_total = 0
    details = []

    for r in results:
        gt = r.ground_truth.get("retrieval", {})
        expected_image_page = gt.get("expected_image_page")

        if not r.guardrail_allow:
            continue

        # 이미지가 있는 경우, top manual source와 일치하는지
        if r.top_image_path and r.top_manual_sources:
            total += 1
            top_src = r.top_manual_sources[0]
            # 이미지 경로에 페이지 번호가 포함되는지 확인
            image_path = r.top_image_path or ""
            src_page = top_src.get("page_no", -1)

            # page_XXXX.png 패턴 매칭
            img_match = re.search(r"page[_-]?0*(\d+)", image_path, re.IGNORECASE)
            if img_match:
                img_page = int(img_match.group(1))
                if img_page == src_page:
                    aligned += 1
                else:
                    details.append({
                        "id": r.item_id,
                        "image_page": img_page,
                        "source_page": src_page,
                    })
            else:
                aligned += 1  # 패턴 매칭 불가시 기본 pass

        # Expected image page 매칭
        if expected_image_page is not None and r.top_image_path:
            image_page_total += 1
            img_match = re.search(r"page[_-]?0*(\d+)", r.top_image_path or "", re.IGNORECASE)
            if img_match and int(img_match.group(1)) == expected_image_page:
                image_page_match += 1

    return {
        "image_source_alignment": round(aligned / total, 4) if total else 0.0,
        "alignment_total": total,
        "expected_image_page_hit": round(image_page_match / image_page_total, 4) if image_page_total else 0.0,
        "image_page_total": image_page_total,
        "errors": details[:10],
    }


def eval_confidence_calibration(results: list[RunResult]) -> dict:
    """신뢰도 보정 — expected bucket vs actual confidence."""
    bucket_results = {"high": [], "medium": [], "low": []}
    correct_bucket = 0
    total = 0

    for r in results:
        gt = r.ground_truth.get("quality", {})
        expected_bucket = gt.get("expected_confidence_bucket")
        if not expected_bucket or not r.guardrail_allow:
            continue

        total += 1
        conf = r.confidence

        # 실제 버킷 결정
        if conf >= 0.7:
            actual_bucket = "high"
        elif conf >= 0.4:
            actual_bucket = "medium"
        else:
            actual_bucket = "low"

        if actual_bucket == expected_bucket:
            correct_bucket += 1

        bucket_results[expected_bucket].append({
            "id": r.item_id,
            "confidence": conf,
            "actual_bucket": actual_bucket,
            "match": actual_bucket == expected_bucket,
        })

    # 버킷별 통계
    bucket_stats = {}
    for bucket, items in bucket_results.items():
        if not items:
            bucket_stats[bucket] = {"count": 0, "accuracy": 0.0, "avg_confidence": 0.0}
            continue
        match_count = sum(1 for it in items if it["match"])
        confs = [it["confidence"] for it in items]
        bucket_stats[bucket] = {
            "count": len(items),
            "accuracy": round(match_count / len(items), 4),
            "avg_confidence": round(mean(confs), 4),
            "min_confidence": round(min(confs), 4),
            "max_confidence": round(max(confs), 4),
        }

    return {
        "bucket_accuracy": round(correct_bucket / total, 4) if total else 0.0,
        "total": total,
        "per_bucket": bucket_stats,
    }


# ───────────────────────────────────────
# Category 5: Operational Metrics
# ───────────────────────────────────────

def eval_latency(results: list[RunResult]) -> dict:
    """응답 지연시간 분포."""
    latencies = [r.latency_sec for r in results if r.latency_sec > 0]
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "total": 0}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50_idx = int(n * 0.50)
    p95_idx = min(int(n * 0.95), n - 1)

    # 카테고리별 분석
    cat_latencies = defaultdict(list)
    for r in results:
        if r.latency_sec > 0:
            cat_latencies[r.category].append(r.latency_sec)

    per_category = {}
    for cat, lats in cat_latencies.items():
        per_category[cat] = {
            "mean": round(mean(lats), 3),
            "p50": round(sorted(lats)[len(lats) // 2], 3),
            "count": len(lats),
        }

    return {
        "p50": round(latencies_sorted[p50_idx], 3),
        "p95": round(latencies_sorted[p95_idx], 3),
        "mean": round(mean(latencies), 3),
        "min": round(min(latencies), 3),
        "max": round(max(latencies), 3),
        "total": n,
        "per_category": per_category,
    }


def eval_cost_per_query(results: list[RunResult]) -> dict:
    """쿼리당 비용 추정 (토큰 기반).

    GPT-4.1-mini 기준: input $0.40/1M, output $1.60/1M
    Cohere rerank-v3.5: $2.00/1000 searches
    bge-m3 임베딩: 로컬 (무료)
    """
    INPUT_COST_PER_1M = 0.40   # GPT-4.1-mini input
    OUTPUT_COST_PER_1M = 1.60  # GPT-4.1-mini output
    RERANK_COST_PER_SEARCH = 0.002  # Cohere

    total_est_cost = 0.0
    query_costs = []

    for r in results:
        if not r.guardrail_allow:
            # 가드레일 거부 시 LLM 호출 없음 (규칙 기반)
            query_costs.append(0.0)
            continue

        # 추정: 질의 + 근거 텍스트 입력 ~2000 토큰, 답변 ~500 토큰
        # Supervisor review: ~1500 입력, ~200 출력
        # 가드레일 LLM judge (fallback): ~200 입력, ~50 출력
        est_input_tokens = 2000 + 1500  # compose + supervisor
        est_output_tokens = 500 + 200

        if r.retry_count > 0:
            est_input_tokens += r.retry_count * 2000
            est_output_tokens += r.retry_count * 500

        llm_cost = (est_input_tokens * INPUT_COST_PER_1M / 1_000_000
                     + est_output_tokens * OUTPUT_COST_PER_1M / 1_000_000)
        rerank_cost = RERANK_COST_PER_SEARCH * (1 + r.retry_count)

        query_cost = llm_cost + rerank_cost
        query_costs.append(query_cost)
        total_est_cost += query_cost

    return {
        "avg_cost_per_query_usd": round(mean(query_costs), 6) if query_costs else 0.0,
        "total_estimated_cost_usd": round(total_est_cost, 4),
        "total_queries": len(results),
        "note": "Estimated based on GPT-4.1-mini ($0.40/1M in, $1.60/1M out) + Cohere rerank ($2/1K)",
    }


def eval_stability_requery(results: list[RunResult]) -> dict:
    """안정성 & 재질의율."""
    error_count = 0
    requery_triggered = 0
    total = len(results)
    retry_counts = []

    for r in results:
        if "[ERROR]" in r.answer:
            error_count += 1
        if r.retry_count > 0:
            requery_triggered += 1
        retry_counts.append(r.retry_count)

    return {
        "error_rate": round(error_count / total, 4) if total else 0.0,
        "requery_rate": round(requery_triggered / total, 4) if total else 0.0,
        "avg_retry_count": round(mean(retry_counts), 3) if retry_counts else 0.0,
        "max_retry_count": max(retry_counts) if retry_counts else 0,
        "error_count": error_count,
        "requery_count": requery_triggered,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════
# 리포트 생성
# ═══════════════════════════════════════════════════════════════

def _norm_filename(f: str) -> str:
    """파일명 정규화 (비교용)."""
    return f.lower().replace(" ", "_").replace("-", "_").split("/")[-1].split("\\")[-1]


def generate_report(results: list[RunResult], use_llm: bool = False) -> dict:
    """전체 평가 리포트 생성."""
    logger.info(f"평가 항목 수: {len(results)}")

    # 카테고리별 분포
    cat_dist = defaultdict(int)
    for r in results:
        cat_dist[r.category] += 1

    report = {
        "summary": {
            "total_items": len(results),
            "category_distribution": dict(cat_dist),
        },
        "categories": {},
    }

    # ── Category 1: Routing & Guardrail ──
    logger.info("Category 1: Routing & Guardrail 평가 중...")
    report["categories"]["1_routing_guardrail"] = {
        "guardrail_accuracy": eval_guardrail_accuracy(results),
        "routing_accuracy": eval_routing_accuracy(results),
        "entity_resolution": eval_entity_resolution(results),
    }

    # ── Category 2: Graph & Document Retrieval ──
    logger.info("Category 2: Graph & Document Retrieval 평가 중...")
    report["categories"]["2_graph_document_retrieval"] = {
        "document_retrieval_recall": eval_document_retrieval_recall(results),
        "graph_completeness": eval_graph_completeness(results),
        "hierarchy_alignment": eval_hierarchy_alignment(results),
    }

    # ── Category 3: Generation & Grounding ──
    logger.info("Category 3: Generation & Grounding 평가 중...")
    report["categories"]["3_generation_grounding"] = {
        "faithfulness": eval_faithfulness(results),
        "answer_relevancy": eval_answer_relevancy(results, use_llm=use_llm),
        "fact_coverage": eval_fact_coverage(results),
    }

    # ── Category 4: Multimodal & UX Alignment ──
    logger.info("Category 4: Multimodal & UX Alignment 평가 중...")
    report["categories"]["4_multimodal_ux"] = {
        "image_source_alignment": eval_image_source_alignment(results),
        "confidence_calibration": eval_confidence_calibration(results),
    }

    # ── Category 5: Operational Metrics ──
    logger.info("Category 5: Operational Metrics 평가 중...")
    report["categories"]["5_operational"] = {
        "latency": eval_latency(results),
        "cost_per_query": eval_cost_per_query(results),
        "stability_requery": eval_stability_requery(results),
    }

    # ── 종합 스코어카드 ──
    report["scorecard"] = _build_scorecard(report["categories"])

    return report


def _build_scorecard(categories: dict) -> dict:
    """각 카테고리의 핵심 지표를 한 눈에 볼 수 있는 스코어카드."""
    c1 = categories.get("1_routing_guardrail", {})
    c2 = categories.get("2_graph_document_retrieval", {})
    c3 = categories.get("3_generation_grounding", {})
    c4 = categories.get("4_multimodal_ux", {})
    c5 = categories.get("5_operational", {})

    def _safe(d, *keys, default=0.0):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    scorecard = {
        "1_routing_guardrail": {
            "guardrail_accuracy": _safe(c1, "guardrail_accuracy", "accuracy"),
            "guardrail_f1": _safe(c1, "guardrail_accuracy", "f1"),
            "routing_accuracy": _safe(c1, "routing_accuracy", "accuracy"),
            "model_family_accuracy": _safe(c1, "entity_resolution", "model_family_accuracy"),
        },
        "2_graph_document_retrieval": {
            "doc_hit_at_5": _safe(c2, "document_retrieval_recall", "hit_at_5"),
            "doc_mrr": _safe(c2, "document_retrieval_recall", "mrr"),
            "entity_hit_rate": _safe(c2, "graph_completeness", "entity_hit_rate"),
            "page_hit_at_5": _safe(c2, "hierarchy_alignment", "page_hit_at_5"),
            "manual_type_match": _safe(c2, "hierarchy_alignment", "manual_type_match"),
        },
        "3_generation_grounding": {
            "groundedness_pass_rate": _safe(c3, "faithfulness", "groundedness_pass_rate"),
            "hallucination_rate": _safe(c3, "faithfulness", "hallucination_rate"),
            "fact_coverage": _safe(c3, "fact_coverage", "overall_coverage"),
        },
        "4_multimodal_ux": {
            "image_source_alignment": _safe(c4, "image_source_alignment", "image_source_alignment"),
            "confidence_bucket_accuracy": _safe(c4, "confidence_calibration", "bucket_accuracy"),
        },
        "5_operational": {
            "latency_p50_sec": _safe(c5, "latency", "p50"),
            "latency_p95_sec": _safe(c5, "latency", "p95"),
            "avg_cost_usd": _safe(c5, "cost_per_query", "avg_cost_per_query_usd"),
            "error_rate": _safe(c5, "stability_requery", "error_rate"),
            "requery_rate": _safe(c5, "stability_requery", "requery_rate"),
        },
    }

    return scorecard


def print_scorecard(report: dict):
    """콘솔에 스코어카드를 출력."""
    sc = report.get("scorecard", {})
    total = report.get("summary", {}).get("total_items", 0)

    print("\n" + "=" * 70)
    print(f"  GraphRAG 5-Category Evaluation Scorecard  ({total} items)")
    print("=" * 70)

    section_names = {
        "1_routing_guardrail": "1. Routing & Guardrail",
        "2_graph_document_retrieval": "2. Graph & Document Retrieval",
        "3_generation_grounding": "3. Generation & Grounding",
        "4_multimodal_ux": "4. Multimodal & UX Alignment",
        "5_operational": "5. Operational Metrics",
    }

    for key, name in section_names.items():
        data = sc.get(key, {})
        print(f"\n  [{name}]")
        for metric, value in data.items():
            if isinstance(value, float):
                if "rate" in metric or "accuracy" in metric or "coverage" in metric or "f1" in metric or "hit" in metric or "match" in metric or "alignment" in metric:
                    print(f"    {metric:40s} {value * 100:6.1f}%")
                elif "cost" in metric or "usd" in metric:
                    print(f"    {metric:40s} ${value:.5f}")
                elif "sec" in metric or "latency" in metric:
                    print(f"    {metric:40s} {value:6.3f}s")
                else:
                    print(f"    {metric:40s} {value:.4f}")
            else:
                print(f"    {metric:40s} {value}")

    print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GraphRAG 5-Category Evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    # run: 파이프라인 실행
    run_p = sub.add_parser("run", help="Run pipeline on dataset and collect raw results")
    run_p.add_argument("--dataset", required=True, help="평가 데이터셋 JSON 경로")
    run_p.add_argument("--output", required=True, help="원시 결과 JSON 출력 경로")
    run_p.add_argument("--top-k", type=int, default=5)
    run_p.add_argument("--max-items", type=int, default=None, help="최대 평가 항목 수 (디버그용)")
    run_p.add_argument("--categories", default=None, help="쉼표 구분 카테고리 필터")

    # report: 결과로 리포트 생성
    rep_p = sub.add_parser("report", help="Generate evaluation report from raw results")
    rep_p.add_argument("--results", required=True, help="원시 결과 JSON 경로")
    rep_p.add_argument("--output", required=True, help="평가 리포트 JSON 출력 경로")
    rep_p.add_argument("--use-llm", action="store_true", help="LLM 기반 답변 관련성 평가 포함")

    # run-and-report: 한번에 실행
    rr_p = sub.add_parser("run-and-report", help="Run pipeline + generate report")
    rr_p.add_argument("--dataset", required=True)
    rr_p.add_argument("--output", required=True, help="최종 리포트 JSON")
    rr_p.add_argument("--raw-output", default=None, help="원시 결과도 별도 저장")
    rr_p.add_argument("--top-k", type=int, default=5)
    rr_p.add_argument("--max-items", type=int, default=None)
    rr_p.add_argument("--categories", default=None)
    rr_p.add_argument("--use-llm", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
        run_pipeline(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            top_k=args.top_k,
            max_items=args.max_items,
            categories=cats,
        )
        print(f"\n원시 결과 저장 완료: {args.output}")

    elif args.command == "report":
        results = load_results(Path(args.results))
        report = generate_report(results, use_llm=args.use_llm)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_scorecard(report)
        print(f"\n평가 리포트 저장: {args.output}")

    elif args.command == "run-and-report":
        cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
        raw_path = Path(args.raw_output) if args.raw_output else Path(args.output).with_suffix(".raw.json")

        results = run_pipeline(
            dataset_path=Path(args.dataset),
            output_path=raw_path,
            top_k=args.top_k,
            max_items=args.max_items,
            categories=cats,
        )
        report = generate_report(results, use_llm=args.use_llm)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_scorecard(report)
        print(f"\n원시 결과: {raw_path}")
        print(f"평가 리포트: {args.output}")


if __name__ == "__main__":
    main()
