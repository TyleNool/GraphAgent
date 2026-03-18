from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from Chevolet_GraphRAG.models import ChatTurn


@dataclass
class SessionData:
    history: list[ChatTurn] = field(default_factory=list)
    summary: str = ""
    last_debug: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionData] = {}
        self._lock = Lock()

    def _get_unlocked(self, session_id: str) -> SessionData:
        if session_id not in self._data:
            self._data[session_id] = SessionData()
        return self._data[session_id]

    def get(self, session_id: str) -> SessionData:
        with self._lock:
            return self._get_unlocked(session_id)

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._get_unlocked(session_id)
            session.history.append(ChatTurn(role=role, content=content))

    def update_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            session = self._get_unlocked(session_id)
            session.summary = summary

    def update_debug(self, session_id: str, debug: dict[str, Any]) -> None:
        with self._lock:
            session = self._get_unlocked(session_id)
            session.last_debug = debug
