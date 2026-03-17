from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chevy_troubleshooter.agent import SessionStore, TroubleshootingWorkflow
from chevy_troubleshooter.config import get_settings
from chevy_troubleshooter.ingest.catalog import discover_manual_files
from chevy_troubleshooter.models import (
    ChatRequest,
    ChatResponse,
)


class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str
    resolved: bool = False


class TopSourcesResponse(BaseModel):
    session_id: str
    top_sources: list[dict[str, Any]]
    top_manual_sources: list[dict[str, Any]]
    top_faq_sources: list[dict[str, Any]]
    top_image_path: str | None = None


def create_app() -> FastAPI:
    settings = get_settings()
    catalog = discover_manual_files(settings.data_root)

    workflow = TroubleshootingWorkflow(settings=settings, catalog=catalog)
    sessions = SessionStore()

    app = FastAPI(title="Chevrolet GraphRAG Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_dir = Path(__file__).resolve().parent.parent / "ui_static"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")
    if settings.artifact_root.exists():
        app.mount(
            "/artifacts",
            StaticFiles(directory=settings.artifact_root, html=False),
            name="artifacts",
        )

    @app.get("/")
    def root():
        return RedirectResponse(url="/ui/")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "database": settings.neo4j_database}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        session = sessions.get(request.session_id)
        history_text = _history_to_text(session.history, session.summary)

        sessions.append_turn(request.session_id, "user", request.user_query)

        result = workflow.run(
            {
                "session_id": request.session_id,
                "user_query": request.user_query,
                "model_hint": request.model_hint,
                "feedback": request.feedback,
                "resolved": request.resolved,
                "top_k": request.top_k,
                "history_text": history_text,
                "compact_summary": session.summary,
            }
        )

        answer = str(result.get("answer", ""))
        sessions.append_turn(request.session_id, "assistant", answer)

        if result.get("compact_summary"):
            sessions.update_summary(request.session_id, str(result["compact_summary"]))

        sessions.update_debug(
            request.session_id,
            {
                "top_sources": result.get("top_sources", []),
                "top_manual_sources": result.get("top_manual_sources", []),
                "top_faq_sources": result.get("top_faq_sources", []),
                "top_image_path": result.get("top_image_path"),
                "debug": result.get("debug", {}),
            },
        )

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            confidence=float(result.get("confidence", 0.0)),
            resolved=request.resolved,
            top_image_path=_to_public_path(result.get("top_image_path"), settings),
            top_manual_sources=result.get("top_manual_sources", result.get("top_sources", [])),
            top_faq_sources=result.get("top_faq_sources", []),
            top_sources=result.get("top_manual_sources", result.get("top_sources", [])),
            graph_paths=result.get("graph_paths", []),
            debug=result.get("debug", {}),
        )

    @app.post("/feedback", response_model=ChatResponse)
    def feedback(request: FeedbackRequest) -> ChatResponse:
        session = sessions.get(request.session_id)
        if not session.history:
            raise HTTPException(status_code=404, detail="session not found")

        prior_user = next((t.content for t in reversed(session.history) if t.role == "user"), "")
        chat_req = ChatRequest(
            session_id=request.session_id,
            user_query=prior_user,
            feedback=request.feedback,
            resolved=request.resolved,
        )
        return chat(chat_req)

    @app.get("/sources/top5/{session_id}", response_model=TopSourcesResponse)
    def top_sources(session_id: str) -> TopSourcesResponse:
        session = sessions.get(session_id)
        payload = session.last_debug or {}
        return TopSourcesResponse(
            session_id=session_id,
            top_sources=payload.get("top_manual_sources", payload.get("top_sources", []))[:5],
            top_manual_sources=payload.get("top_manual_sources", payload.get("top_sources", []))[:5],
            top_faq_sources=payload.get("top_faq_sources", [])[:5],
            top_image_path=_to_public_path(payload.get("top_image_path"), settings),
        )

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        workflow.close()

    return app


def _history_to_text(history, summary: str) -> str:
    turns = [f"[summary]\n{summary}"] if summary else []
    for turn in history[-20:]:
        turns.append(f"[{turn.role}] {turn.content}")
    return "\n".join(turns)


def _to_public_path(path: str | None, settings) -> str | None:
    if not path:
        return None
    raw = Path(path)
    if not raw.is_absolute():
        raw = (settings.project_root / raw).resolve()
    try:
        rel = raw.relative_to(settings.artifact_root)
        return f"/artifacts/{rel.as_posix()}"
    except ValueError:
        return path
