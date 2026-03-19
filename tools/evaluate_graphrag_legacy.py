#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "Comprehensive_GraphRAG_Evaluation_Dataset_300.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_base_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_graphrag_base",
        Path(__file__).resolve().parent / "evaluate_graphrag.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/evaluate_graphrag.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
logger = logging.getLogger(__name__)

EvalItem = BASE.EvalItem
RunResult = BASE.RunResult
generate_report = BASE.generate_report
load_dataset = BASE.load_dataset
load_results = BASE.load_results
print_scorecard = BASE.print_scorecard
save_results = BASE._save_results


def run_pipeline(
    dataset_path: Path,
    output_path: Path,
    top_k: int = 5,
    max_items: int | None = None,
    categories: list[str] | None = None,
) -> list[RunResult]:
    from Chevolet_GraphRAG.agent.legacy_workflow import LegacyTroubleshootingWorkflow
    from Chevolet_GraphRAG.config import get_settings
    from Chevolet_GraphRAG.ingest.catalog import discover_manual_files

    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)
    workflow = LegacyTroubleshootingWorkflow(settings=settings, catalog=catalog)

    items = load_dataset(dataset_path)
    if categories:
        category_set = set(categories)
        items = [item for item in items if item.category in category_set]
    if max_items:
        items = items[:max_items]

    results: list[RunResult] = []
    total = len(items)

    try:
        for idx, item in enumerate(items, 1):
            logger.info("[%d/%d] %s (%s) — %s...", idx, total, item.id, item.category, item.question[:60])

            ground_truth = item.ground_truth
            model_hint = None
            guardrail_gt = ground_truth.get("guardrail", {})
            candidates = guardrail_gt.get("expected_model_candidates", [])
            if candidates:
                model_hint = candidates[0]

            t0 = time.perf_counter()
            try:
                state = workflow.run(
                    {
                        "session_id": f"legacy-eval-{item.id}-{uuid.uuid4().hex[:6]}",
                        "user_query": item.question,
                        "model_hint": model_hint,
                        "top_k": top_k,
                        "history_text": "",
                        "feedback": None,
                        "resolved": None,
                    }
                )
            except Exception as exc:
                logger.error("  ERROR: %s", exc)
                state = {"answer": f"[ERROR] {exc}", "confidence": 0.0}
            elapsed = time.perf_counter() - t0

            results.append(
                RunResult(
                    item_id=item.id,
                    category=item.category,
                    difficulty=item.difficulty,
                    question=item.question,
                    ground_truth=ground_truth,
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
            )

            current = results[-1]
            logger.info(
                "  → allow=%s, conf=%.3f, pages=%d, lat=%.2fs",
                current.guardrail_allow,
                current.confidence,
                len(current.retrieval_pages),
                current.latency_sec,
            )

            if idx % 10 == 0:
                save_results(results, output_path)
    finally:
        workflow.close()

    save_results(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy GraphRAG 5-Category Evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run legacy pipeline on dataset and collect raw results")
    run_p.add_argument(
        "--dataset",
        default=str(DEFAULT_EVAL_DATASET),
        help=f"평가 데이터셋 JSON 경로 (기본값: {DEFAULT_EVAL_DATASET.name})",
    )
    run_p.add_argument("--output", required=True, help="원시 결과 JSON 출력 경로")
    run_p.add_argument("--top-k", type=int, default=5)
    run_p.add_argument("--max-items", type=int, default=None)
    run_p.add_argument("--categories", default=None)

    rep_p = sub.add_parser("report", help="Generate evaluation report from raw legacy results")
    rep_p.add_argument("--results", required=True, help="원시 결과 JSON 경로")
    rep_p.add_argument("--output", required=True, help="평가 리포트 JSON 출력 경로")
    rep_p.add_argument("--use-llm", action="store_true", help="LLM 기반 답변 관련성 평가 포함")

    rr_p = sub.add_parser("run-and-report", help="Run legacy pipeline + generate report")
    rr_p.add_argument(
        "--dataset",
        default=str(DEFAULT_EVAL_DATASET),
        help=f"평가 데이터셋 JSON 경로 (기본값: {DEFAULT_EVAL_DATASET.name})",
    )
    rr_p.add_argument("--output", required=True, help="최종 리포트 JSON")
    rr_p.add_argument("--raw-output", default=None, help="원시 결과도 별도 저장")
    rr_p.add_argument("--top-k", type=int, default=5)
    rr_p.add_argument("--max-items", type=int, default=None)
    rr_p.add_argument("--categories", default=None)
    rr_p.add_argument("--use-llm", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
        run_pipeline(
            dataset_path=Path(args.dataset),
            output_path=Path(args.output),
            top_k=args.top_k,
            max_items=args.max_items,
            categories=categories,
        )
        print(f"\n원시 결과 저장 완료: {args.output}")

    elif args.command == "report":
        results = load_results(Path(args.results))
        report = generate_report(results, use_llm=args.use_llm)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_scorecard(report)
        print(f"\n평가 리포트 저장: {args.output}")

    elif args.command == "run-and-report":
        categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
        raw_path = Path(args.raw_output) if args.raw_output else Path(args.output).with_suffix(".raw.json")
        results = run_pipeline(
            dataset_path=Path(args.dataset),
            output_path=raw_path,
            top_k=args.top_k,
            max_items=args.max_items,
            categories=categories,
        )
        report = generate_report(results, use_llm=args.use_llm)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_scorecard(report)
        print(f"\n원시 결과: {raw_path}")
        print(f"평가 리포트: {args.output}")


if __name__ == "__main__":
    main()
