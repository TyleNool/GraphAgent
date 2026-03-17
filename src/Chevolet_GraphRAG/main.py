from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import uvicorn

from chevy_troubleshooter.agent import TroubleshootingWorkflow
from chevy_troubleshooter.config import get_settings
from chevy_troubleshooter.ingest import (
    IngestionPipeline,
    discover_manual_files,
    profile_dataset,
)
from chevy_troubleshooter.providers import build_embeddings
from chevy_troubleshooter.retrieval.chroma_faq import ChromaFAQStore


def cmd_ingest_data(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = IngestionPipeline(settings=settings)
    payload = pipeline.run(
        data_root=Path(args.data_root).resolve(),
        init_schema=args.init_schema,
        max_manuals=args.max_manuals,
        include_models=_split_csv(args.include_models),
        filename_keywords=_split_csv(args.filename_keywords),
        skip_existing=args.skip_existing,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_ingest_faq(args: argparse.Namespace) -> None:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    store = ChromaFAQStore(persist_dir=settings.chroma_persist_dir)
    result = store.ingest_faq(
        json_path=Path(args.faq_path).resolve(),
        embeddings=embeddings,
        reset=args.reset,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_profile_data(args: argparse.Namespace) -> None:
    payload = profile_dataset(
        data_root=Path(args.data_root).resolve(),
        include_page_counts=args.include_page_counts,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_run_graph_once(args: argparse.Namespace) -> None:
    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)
    workflow = TroubleshootingWorkflow(settings=settings, catalog=catalog)

    try:
        result = workflow.run(
            {
                "session_id": f"once-{uuid.uuid4().hex[:8]}",
                "user_query": args.query,
                "model_hint": args.model,
                "top_k": args.top_k,
                "feedback": args.feedback,
                "resolved": None,
                "history_text": "",
            }
        )
        print("\n=== ANSWER ===")
        print(result.get("answer", ""))
        print("\n=== CONFIDENCE ===")
        print(result.get("confidence", 0.0))
        print("\n=== TOP IMAGE ===")
        print(result.get("top_image_path"))
        print("\n=== TOP-5 MANUAL SOURCES ===")
        for i, source in enumerate(result.get("top_manual_sources", result.get("top_sources", []))[:5], start=1):
            score_details = []
            if source.get("relevance_label"):
                score_details.append(f"관련도={source['relevance_label']}")
            if source.get("retrieval_score") is not None:
                score_details.append(f"retrieval={source['retrieval_score']:.4f}")
            if source.get("rerank_score") is not None:
                score_details.append(f"rerank={source['rerank_score']:.4f}")
            print(
                f"{i}. {source['source_file']} p.{source['page_no']} "
                f"{' '.join(score_details)} path={source['path_summary']}"
            )
        faq_sources = result.get("top_faq_sources", [])[:5]
        if faq_sources:
            print("\n=== TOP-5 FAQ SOURCES ===")
            for i, source in enumerate(faq_sources, start=1):
                score_details = []
                if source.get("relevance_label"):
                    score_details.append(f"관련도={source['relevance_label']}")
                if source.get("retrieval_score") is not None:
                    score_details.append(f"retrieval={source['retrieval_score']:.4f}")
                print(
                    f"{i}. {source['source_file']} "
                    f"{' '.join(score_details)} path={source['path_summary']}"
                )
    finally:
        workflow.close()


def cmd_run_graph_session(args: argparse.Namespace) -> None:
    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)
    workflow = TroubleshootingWorkflow(settings=settings, catalog=catalog)
    session_id = f"session-{uuid.uuid4().hex[:8]}"

    history = []

    try:
        print("쉐보레 진단 세션을 시작합니다. 종료하려면 'exit' 입력")
        while True:
            query = input("\n사용자> ").strip()
            if query.lower() in {"exit", "quit"}:
                break

            history.append(f"[user] {query}")
            result = workflow.run(
                {
                    "session_id": session_id,
                    "user_query": query,
                    "model_hint": args.model,
                    "top_k": args.top_k,
                    "history_text": "\n".join(history),
                    "feedback": None,
                    "resolved": None,
                }
            )
            answer = str(result.get("answer", ""))
            history.append(f"[assistant] {answer}")

            print(f"\n에이전트> {answer}")
            print(f"신뢰도: {result.get('confidence', 0.0):.3f}")
            print(f"Top 이미지: {result.get('top_image_path')}")

            feedback = input("해결되지 않았다면 피드백 입력(없으면 Enter): ").strip()
            if feedback:
                feedback_result = workflow.run(
                    {
                        "session_id": session_id,
                        "user_query": query,
                        "model_hint": args.model,
                        "top_k": args.top_k,
                        "history_text": "\n".join(history),
                        "feedback": feedback,
                        "resolved": False,
                    }
                )
                answer2 = str(feedback_result.get("answer", ""))
                history.append(f"[assistant-requery] {answer2}")
                print(f"\n재질의 응답> {answer2}")
                print(f"신뢰도: {feedback_result.get('confidence', 0.0):.3f}")
    finally:
        workflow.close()


def cmd_evaluate_graph(args: argparse.Namespace) -> None:
    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)
    workflow = TroubleshootingWorkflow(settings=settings, catalog=catalog)

    queries = _load_queries(Path(args.queries_file))
    rows = []

    try:
        for i, item in enumerate(queries, start=1):
            query = item.get("query", "")
            model_hint = item.get("model")
            expected = item.get("expected", "")

            result = workflow.run(
                {
                    "session_id": f"eval-{i}",
                    "user_query": query,
                    "model_hint": model_hint,
                    "top_k": args.top_k,
                    "history_text": "",
                    "feedback": None,
                    "resolved": None,
                }
            )
            answer = str(result.get("answer", ""))
            f1 = _simple_f1(expected, answer)
            rows.append(
                {
                    "id": i,
                    "query": query,
                    "model": model_hint,
                    "expected": expected,
                    "answer": answer,
                    "f1": f1,
                    "confidence": result.get("confidence", 0.0),
                    "top_sources": result.get("top_manual_sources", result.get("top_sources", []))[:5],
                    "top_manual_sources": result.get("top_manual_sources", result.get("top_sources", []))[:5],
                    "top_faq_sources": result.get("top_faq_sources", [])[:5],
                }
            )

        report = {
            "count": len(rows),
            "avg_f1": sum(r["f1"] for r in rows) / max(1, len(rows)),
            "rows": rows,
        }

        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"평가 리포트 저장: {out_path}")
    finally:
        workflow.close()


def cmd_evaluate_graphrag(args: argparse.Namespace) -> None:
    """5-Category GraphRAG 종합 평가 실행."""
    # tools/evaluate_graphrag.py 를 내부적으로 호출
    tools_dir = Path(__file__).resolve().parent.parent.parent / "tools"
    import importlib.util
    spec = importlib.util.spec_from_file_location("evaluate_graphrag", tools_dir / "evaluate_graphrag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output_file).resolve()
    raw_path = output_path.with_suffix(".raw.json")

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None

    results = mod.run_pipeline(
        dataset_path=dataset_path,
        output_path=raw_path,
        top_k=args.top_k,
        max_items=args.max_items,
        categories=cats,
    )
    report = mod.generate_report(results, use_llm=args.use_llm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mod.print_scorecard(report)
    print(f"\n원시 결과: {raw_path}")
    print(f"평가 리포트: {output_path}")


def cmd_serve_api(args: argparse.Namespace) -> None:
    uvicorn.run(
        "chevy_troubleshooter.api.app:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=args.reload,
    )


def _load_queries(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "queries" in payload:
            return payload["queries"]
        return []

    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append({"query": line, "expected": ""})
    return rows


def _simple_f1(expected: str, predicted: str) -> float:
    exp_tokens = set(expected.split()) if expected else set()
    pred_tokens = set(predicted.split()) if predicted else set()
    if not exp_tokens or not pred_tokens:
        return 0.0

    tp = len(exp_tokens & pred_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chevrolet GraphRAG Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest-data", help="Parse PDFs and ingest into Neo4j")
    ingest.add_argument("--data-root", default="data")
    ingest.add_argument("--init-schema", action="store_true")
    ingest.add_argument("--max-manuals", type=int, default=None)
    ingest.add_argument(
        "--include-models",
        default=None,
        help="쉼표 구분 모델명 목록만 적재 (예: SPARK_EV,말리부,BOLT_EV,BOLT_EUV,VOLT)",
    )
    ingest.add_argument(
        "--filename-keywords",
        default=None,
        help="쉼표 구분 파일명 키워드 중 하나라도 포함 시 적재 (예: 긴급조치,긴급상황)",
    )
    ingest.add_argument(
        "--skip-existing",
        action="store_true",
        help="source_file 기준으로 기존 Manual 존재 시 스킵",
    )
    ingest.set_defaults(func=cmd_ingest_data)

    ingest_faq = sub.add_parser("ingest-faq", help="Ingest FAQ JSON into ChromaDB")
    ingest_faq.add_argument(
        "--faq-path",
        default="data/FAQ/chevrolet_faq_target_data.json",
        help="FAQ JSON 파일 경로",
    )
    ingest_faq.add_argument(
        "--reset",
        action="store_true",
        help="기존 FAQ 컬렉션 삭제 후 재적재",
    )
    ingest_faq.set_defaults(func=cmd_ingest_faq)

    profile = sub.add_parser("profile-data", help="Inspect data structure before ingest")
    profile.add_argument("--data-root", default="data")
    profile.add_argument("--include-page-counts", action="store_true")
    profile.set_defaults(func=cmd_profile_data)

    once = sub.add_parser("run-graph-once", help="Single query execution")
    once.add_argument("--query", required=True)
    once.add_argument("--model", default=None)
    once.add_argument("--top-k", type=int, default=5)
    once.add_argument("--feedback", default=None)
    once.set_defaults(func=cmd_run_graph_once)

    session = sub.add_parser("run-graph-session", help="Interactive diagnosis session")
    session.add_argument("--model", default=None)
    session.add_argument("--top-k", type=int, default=5)
    session.set_defaults(func=cmd_run_graph_session)

    evaluate = sub.add_parser("evaluate-graph", help="Run simple QA evaluation")
    evaluate.add_argument("--queries-file", required=True)
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--output-file", required=True)
    evaluate.set_defaults(func=cmd_evaluate_graph)

    eval5 = sub.add_parser("evaluate-graphrag", help="5-Category GraphRAG evaluation")
    eval5.add_argument("--dataset", required=True, help="평가 데이터셋 JSON")
    eval5.add_argument("--output-file", required=True, help="평가 리포트 JSON 출력")
    eval5.add_argument("--top-k", type=int, default=5)
    eval5.add_argument("--max-items", type=int, default=None, help="최대 평가 항목 수")
    eval5.add_argument("--categories", default=None, help="쉼표 구분 카테고리 필터")
    eval5.add_argument("--use-llm", action="store_true", help="LLM 기반 답변 관련성 평가")
    eval5.set_defaults(func=cmd_evaluate_graphrag)

    api = sub.add_parser("serve-api", help="Run FastAPI server")
    api.add_argument("--host", default=get_settings().api_host)
    api.add_argument("--port", type=int, default=get_settings().api_port)
    api.add_argument("--reload", action="store_true")
    api.set_defaults(func=cmd_serve_api)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
