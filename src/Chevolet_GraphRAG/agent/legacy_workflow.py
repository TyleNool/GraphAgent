from __future__ import annotations

from Chevolet_GraphRAG.agent.workflow import TroubleshootingWorkflow
from Chevolet_GraphRAG.legacy_neo4j_store import LegacyNeo4jStore
from Chevolet_GraphRAG.observability import LangSmithTracer
from Chevolet_GraphRAG.providers import build_chat_model
from Chevolet_GraphRAG.retrieval.guardrails import GuardrailEngine
from Chevolet_GraphRAG.retrieval.legacy_hybrid import LegacyHybridRetriever


class LegacyTroubleshootingWorkflow(TroubleshootingWorkflow):
    def __init__(self, settings, catalog) -> None:
        self.settings = settings
        self.catalog = catalog

        self.store = LegacyNeo4jStore(settings)
        self.retriever = LegacyHybridRetriever(settings, self.store)
        self.guardrails = GuardrailEngine(settings, catalog)
        self.chat_model = build_chat_model(settings)
        self.tracer = LangSmithTracer(
            api_key=settings.langsmith_api_key,
            endpoint=settings.langsmith_endpoint,
            project_name=settings.langsmith_project,
            enabled=settings.langsmith_tracing,
            workspace_id=settings.langsmith_workspace_id,
        )

        self.graph = self._build_graph()
