from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from pathlib import Path

from Chevolet_GraphRAG.agent.legacy_workflow import LegacyTroubleshootingWorkflow
from Chevolet_GraphRAG.config import get_settings
from Chevolet_GraphRAG.ingest.catalog import discover_manual_files
from Chevolet_GraphRAG.ingest.legacy_pipeline import LegacyIngestionPipeline
from Chevolet_GraphRAG.ingest.profiler import profile_dataset
from Chevolet_GraphRAG.providers import build_embeddings
from Chevolet_GraphRAG.retrieval.chroma_faq import ChromaFAQStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPHRAG_EVAL_DATASET = PROJECT_ROOT / "Comprehensive_GraphRAG_Evaluation_Dataset_300.json"


def cmd_ingest_data(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = LegacyIngestionPipeline(settings=settings)
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
    workflow = LegacyTroubleshootingWorkflow(settings=settings, catalog=catalog)

    try:
        result = workflow.run(
            {
                "session_id": f"legacy-once-{uuid.uuid4().hex[:8]}",
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
            page_label = source.get("display_page_label") or f"PDF {source.get('page_no', 0)}"
            print(
                f"{i}. {source['source_file']} {page_label} "
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
    workflow = LegacyTroubleshootingWorkflow(settings=settings, catalog=catalog)
    session_id = f"legacy-session-{uuid.uuid4().hex[:8]}"
    history: list[str] = []

    try:
        print("쉐보레 레거시 진단 세션을 시작합니다. 종료하려면 'exit' 입력")
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


def cmd_evaluate_graphrag(args: argparse.Namespace) -> None:
    tools_dir = PROJECT_ROOT / "tools"
    spec = importlib.util.spec_from_file_location(
        "evaluate_graphrag_legacy",
        tools_dir / "evaluate_graphrag_legacy.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/evaluate_graphrag_legacy.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output_file).resolve()
    raw_path = output_path.with_suffix(".raw.json")
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    results = module.run_pipeline(
        dataset_path=dataset_path,
        output_path=raw_path,
        top_k=args.top_k,
        max_items=args.max_items,
        categories=categories,
    )
    report = module.generate_report(results, use_llm=args.use_llm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    module.print_scorecard(report)
    print(f"\n원시 결과: {raw_path}")
    print(f"평가 리포트: {output_path}")


def _split_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chevolet GraphRAG legacy CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest-data", help="Legacy manual ingestion")
    ingest_p.add_argument("--data-root", default="data")
    ingest_p.add_argument("--init-schema", action="store_true")
    ingest_p.add_argument("--max-manuals", type=int, default=None)
    ingest_p.add_argument("--include-models", default=None)
    ingest_p.add_argument("--filename-keywords", default=None)
    ingest_p.add_argument("--skip-existing", action="store_true")
    ingest_p.set_defaults(func=cmd_ingest_data)

    faq_p = sub.add_parser("ingest-faq", help="FAQ ingestion")
    faq_p.add_argument("--faq-path", required=True)
    faq_p.add_argument("--reset", action="store_true")
    faq_p.set_defaults(func=cmd_ingest_faq)

    profile_p = sub.add_parser("profile-data", help="Dataset profile")
    profile_p.add_argument("--data-root", default="data")
    profile_p.add_argument("--include-page-counts", action="store_true")
    profile_p.set_defaults(func=cmd_profile_data)

    once_p = sub.add_parser("run-graph-once", help="Legacy workflow single query")
    once_p.add_argument("--query", required=True)
    once_p.add_argument("--model", default=None)
    once_p.add_argument("--top-k", type=int, default=5)
    once_p.add_argument("--feedback", default=None)
    once_p.set_defaults(func=cmd_run_graph_once)

    session_p = sub.add_parser("run-graph-session", help="Legacy workflow session")
    session_p.add_argument("--model", default=None)
    session_p.add_argument("--top-k", type=int, default=5)
    session_p.set_defaults(func=cmd_run_graph_session)

    eval_p = sub.add_parser("evaluate-graphrag", help="Legacy GraphRAG evaluation")
    eval_p.add_argument(
        "--dataset",
        default=str(DEFAULT_GRAPHRAG_EVAL_DATASET),
        help=f"평가 데이터셋 JSON 경로 (기본값: {DEFAULT_GRAPHRAG_EVAL_DATASET.name})",
    )
    eval_p.add_argument("--output-file", required=True)
    eval_p.add_argument("--top-k", type=int, default=5)
    eval_p.add_argument("--max-items", type=int, default=None)
    eval_p.add_argument("--categories", default=None)
    eval_p.add_argument("--use-llm", action="store_true")
    eval_p.set_defaults(func=cmd_evaluate_graphrag)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
