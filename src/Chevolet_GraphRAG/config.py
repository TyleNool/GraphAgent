from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(override=True)


def _clean_env(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    return raw.strip().strip('"').strip("'")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _clean_env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    project_root: Path
    data_root: Path
    artifact_root: Path

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    llm_provider: str
    llm_model: str
    llm_temperature: float

    embedding_provider: str
    embedding_model: str

    language: str
    default_top_k: int
    max_requery: int
    context_compaction_chars: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    embedding_text_max_chars: int
    query_text_max_chars: int

    reranker_provider: str
    reranker_model: str
    cohere_api_key: str
    huggingface_local_files_only: bool

    langsmith_api_key: str
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_tracing: bool
    langsmith_workspace_id: str

    chroma_persist_dir: Path

    api_host: str
    api_port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(_clean_env("PROJECT_ROOT", Path.cwd().as_posix())).resolve()
    data_root = Path(_clean_env("DATA_ROOT", (project_root / "data").as_posix())).resolve()
    artifact_root = Path(
        _clean_env("ARTIFACT_ROOT", (project_root / "artifacts").as_posix())
    ).resolve()

    artifact_root.mkdir(parents=True, exist_ok=True)
    langsmith_api_key = _clean_env("LANGSMITH_API_KEY", _clean_env("LANGCHAIN_API_KEY", ""))

    return Settings(
        project_root=project_root,
        data_root=data_root,
        artifact_root=artifact_root,
        neo4j_uri=_clean_env("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        neo4j_username=_clean_env("NEO4J_USERNAME", "neo4j"),
        neo4j_password=_clean_env("NEO4J_PASSWORD", "neo4j"),
        neo4j_database=_clean_env("NEO4J_DATABASE", "chevrolet"),
        llm_provider=_clean_env("LLM_PROVIDER", "openai").lower(),
        llm_model=_clean_env("LLM_MODEL", "gpt-4.1-mini"),
        llm_temperature=float(_clean_env("LLM_TEMPERATURE", "0.1")),
        embedding_provider=_clean_env("EMBEDDING_PROVIDER", "huggingface").lower(),
        embedding_model=_clean_env(
            # "EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        ),
        language=_clean_env("RESPONSE_LANGUAGE", "ko"),
        default_top_k=int(_clean_env("DEFAULT_TOP_K", "5")),
        max_requery=int(_clean_env("MAX_REQUERY", "2")),
        context_compaction_chars=int(_clean_env("CONTEXT_COMPACTION_CHARS", "8000")),
        chunk_size_chars=int(_clean_env("CHUNK_SIZE_CHARS", "420")),
        chunk_overlap_chars=int(_clean_env("CHUNK_OVERLAP_CHARS", "80")),
        embedding_text_max_chars=int(_clean_env("EMBEDDING_TEXT_MAX_CHARS", "420")),
        query_text_max_chars=int(_clean_env("QUERY_TEXT_MAX_CHARS", "420")),
        reranker_provider=_clean_env("RERANKER_PROVIDER", "cohere").lower(),
        reranker_model=_clean_env("RERANKER_MODEL", "rerank-v3.5"),
        cohere_api_key=_clean_env("COHERE_API_KEY", _clean_env("CO_API_KEY", "")),
        huggingface_local_files_only=_env_bool("HF_LOCAL_FILES_ONLY", True),
        langsmith_api_key=langsmith_api_key,
        langsmith_endpoint=_clean_env(
            "LANGSMITH_ENDPOINT",
            _clean_env("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
        ),
        langsmith_project=_clean_env(
            "LANGSMITH_PROJECT",
            _clean_env("LANGCHAIN_PROJECT", "chevy-troubleshooter"),
        ),
        langsmith_tracing=_env_bool("LANGSMITH_TRACING", True),
        langsmith_workspace_id=_clean_env("LANGSMITH_WORKSPACE_ID", ""),
        chroma_persist_dir=Path(
            _clean_env("CHROMA_PERSIST_DIR", (artifact_root / "chroma_faq").as_posix())
        ).resolve(),
        api_host=_clean_env("API_HOST", "0.0.0.0"),
        api_port=int(_clean_env("API_PORT", "8000")),
    )
