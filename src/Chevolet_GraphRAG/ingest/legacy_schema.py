from __future__ import annotations

from pathlib import Path


def load_legacy_schema_cypher(embedding_dim: int = 1024) -> str:
    schema_path = Path(__file__).resolve().parent.parent / "cypher" / "legacy_schema.cypher"
    query = schema_path.read_text(encoding="utf-8")
    return query.replace("__EMBEDDING_DIM__", str(embedding_dim))
