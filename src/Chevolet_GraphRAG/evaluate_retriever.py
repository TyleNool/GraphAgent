import argparse
import os
import json
import time
from typing import Any
from dataclasses import dataclass

from chevy_troubleshooter.config import get_settings
from chevy_troubleshooter.ingest.catalog import discover_manual_files
from chevy_troubleshooter.neo4j_store import Neo4jStore
from chevy_troubleshooter.retrieval.hybrid import HybridRetriever
from chevy_troubleshooter.retrieval.guardrails import GuardrailEngine

# Simple Ground Truth Dataset (Query, Model, Expected Substring in Answer/Chunk)
EVAL_DATASET = [
    {
        "query": "트레일블레이저 버튼을 눌러도 시동이 안 걸려요",
        "model_hint": "트레일블레이저",
        "expected_keywords": ["방전", "스마트키", "브레이크", "점프"],
    },
    {
        "query": "이쿼녹스 타이어 공기압 경고등 떠서 리셋하고 싶음",
        "model_hint": "이쿼녹스",
        "expected_keywords": ["공기압", "초기화", "TPMS", "재설정"],
    },
    {
        "query": "볼트EV 배터리가 너무 빨리 다는 것 같아요",
        "model_hint": "BOLT_EV",
        "expected_keywords": ["주행가능거리", "회생제동", "히터", "에어컨"],
    },
    {
        "query": "말리부 엔진오일 수명 확인 방법",
        "model_hint": "말리부",
        "expected_keywords": ["엔진오일", "수명", "DIC", "정보 디스플레이"],
    },
    {
        "query": "콜로라도 4륜 구동 전환은 어떻게 하나요?",
        "model_hint": "콜로라도",
        "expected_keywords": ["4WD", "트랜스퍼 케이스", "사륜구동", "노브"],
    }
]


@dataclass
class EvalResult:
    query: str
    latency_sec: float
    retrieved_count: int
    hit: bool  # Expected keywords found in top_k?
    mrr: float  # Mean Reciprocal Rank
    top_chunk_text: str


def evaluate_retriever(
    retriever: HybridRetriever,
    guardrails: GuardrailEngine,
    top_k: int = 5,
) -> list[EvalResult]:
    results = []
    
    for item in EVAL_DATASET:
        query = item["query"]
        model = item["model_hint"]
        expected_kws = item["expected_keywords"]
        
        start_time = time.time()
        # retriever() returns components (merged_items, graph_paths, debug_info)
        model_candidates = guardrails.expand_model_candidates(model)
        merged, _, _, _ = retriever.retrieve(
            query=query,
            top_k=top_k,
            model_candidates=model_candidates,
        )
        latency = time.time() - start_time
        
        hit = False
        mrr = 0.0
        top_text = merged[0].text if merged else "No Results"
        
        for rank, chunk in enumerate(merged, start=1):
            text_lower = chunk.text.lower()
            if any(kw.lower() in text_lower for kw in expected_kws):
                hit = True
                mrr = 1.0 / rank
                break # Only count first hit for MRR
                
        results.append(EvalResult(
            query=query,
            latency_sec=latency,
            retrieved_count=len(merged),
            hit=hit,
            mrr=mrr,
            top_chunk_text=top_text
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate HybridRetriever with/without Reranker")
    parser.add_argument("--disable-reranker", action="store_true", help="Run evaluation without Cross-Encoder Reranker")
    args = parser.parse_args()

    # Load base settings
    settings = get_settings()
    
    if args.disable_reranker:
        print("--- RUNNING EVALUATION WITHOUT RERANKER (BASELINE) ---")
        # Overwrite reranker setting for this run
        settings.reranker_provider = "none" 
    else:
        print(f"--- RUNNING EVALUATION WITH RERANKER ({settings.reranker_model}) ---")
        
    store = Neo4jStore(settings=settings)
    catalog = discover_manual_files(settings.data_root)
    guardrails = GuardrailEngine(settings=settings, catalog=catalog)
    
    try:
        retriever = HybridRetriever(settings=settings, store=store)
        results = evaluate_retriever(retriever, guardrails)
        
        total_queries = len(results)
        hits = sum(1 for r in results if r.hit)
        recall = (hits / total_queries) * 100
        avg_mrr = sum(r.mrr for r in results) / total_queries
        avg_latency = sum(r.latency_sec for r in results) / total_queries
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Total Queries : {total_queries}")
        print(f"Avg Latency   : {avg_latency:.3f} seconds")
        print(f"Hit Rate      : {recall:.1f}% ({hits}/{total_queries})")
        print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")
        print("==========================\n")
        
        for i, r in enumerate(results, 1):
            print(f"Q{i}: {r.query}")
            print(f"  - Hit: {r.hit} | MRR: {r.mrr:.3f} | Latency: {r.latency_sec:.3f}s")
            print(f"  - Top Chunk Preview: {r.top_chunk_text[:100]}...\n")
            
    finally:
        store.close()


if __name__ == "__main__":
    main()
