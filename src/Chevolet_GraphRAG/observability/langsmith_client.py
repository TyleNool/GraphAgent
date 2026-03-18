from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from typing import Any


class LangSmithTracer:
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        project_name: str,
        enabled: bool = True,
        workspace_id: str = "",
    ) -> None:
        self.project_name = project_name or "Chevolet_GraphRAG"
        self.enabled = bool(enabled and api_key)
        self._client = None
        self._tracing_context = None
        self._run_tree_cls = None

        if not self.enabled:
            return

        try:
            from langsmith import Client, tracing_context
            from langsmith.run_trees import RunTree

            self._client = Client(
                api_key=api_key,
                api_url=endpoint or None,
                workspace_id=workspace_id or None,
            )
            self._tracing_context = tracing_context
            self._run_tree_cls = RunTree

            # Keep LangChain and LangGraph tracing aligned with the same project.
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_ENDPOINT"] = endpoint or "https://api.smith.langchain.com"
            os.environ["LANGSMITH_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
        except Exception:
            self.enabled = False
            self._client = None
            self._tracing_context = None
            self._run_tree_cls = None

    @contextmanager
    def trace(
        self,
        name: str,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ):
        if not self.enabled or self._client is None or self._run_tree_cls is None:
            yield None
            return

        trace = None
        tracing_cm = nullcontext()
        try:
            trace = self._run_tree_cls(
                name=name,
                run_type="chain",
                inputs=input_data or {},
                project_name=self.project_name,
                ls_client=self._client,
                tags=tags or ["workflow"],
                extra={"metadata": metadata or {}},
            )
            trace.post()
            tracing_cm = self._tracing_context(
                project_name=self.project_name,
                parent=trace,
                enabled=True,
                client=self._client,
                metadata=metadata or {},
                tags=tags or ["workflow"],
            )
        except Exception:
            trace = None
            tracing_cm = nullcontext()

        try:
            with tracing_cm:
                yield trace
            if trace is not None:
                try:
                    trace.end(outputs={"status": "ok"})
                    trace.patch()
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover
            if trace is not None:
                try:
                    trace.end(
                        outputs={"status": "error"},
                        error=str(exc),
                    )
                    trace.patch()
                except Exception:
                    pass
            raise

    def event(self, trace: Any, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled or trace is None:
            return
        try:
            trace.add_event(
                {
                    "name": name,
                    "time": datetime.now(timezone.utc).isoformat(),
                    "message": name,
                    "metadata": payload,
                }
            )
            trace.patch()
        except Exception:
            return
